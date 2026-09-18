from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME_ROOT = Path('/kaggle/tmp/hindibhajans_echomimic_runtime')
FINAL_ROOT = Path('/kaggle/working')
ROOT = RUNTIME_ROOT
INPUT_ROOT = Path('/kaggle/input')
SECONDS_FILE = 'duration.txt'
IMAGE_ENCODER_FILE = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
FUN_HF = 'alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP'
FLASH_HF = 'BadToBest/EchoMimicV3'


def run(*args: object) -> None:
    print('RUN:', *args, flush=True)
    subprocess.run([str(x) for x in args], check=True)


def find_input(name: str) -> Path:
    matches: list[Path] = []
    for root in (INPUT_ROOT, ROOT):
        if root.exists():
            matches.extend(p for p in root.rglob(name) if p.is_file())
    if not matches:
        raise RuntimeError(f'KAGGLE_INPUT_FILE_MISSING: {name}')
    return min(matches, key=lambda p: len(p.parts))


def find_mounted_file(name: str) -> Path:
    matches = [p for p in INPUT_ROOT.rglob(name) if p.is_file()] if INPUT_ROOT.exists() else []
    if not matches:
        raise RuntimeError(f'KAGGLE_MOUNTED_FILE_MISSING: {name}')
    return min(matches, key=lambda p: len(p.parts))


def newest_mp4(directory: Path) -> Path:
    files = sorted(directory.rglob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError(f'NO_MP4_GENERATED: {directory}')
    return files[0]


def disk_report(label: str) -> None:
    p = shutil.disk_usage(ROOT)
    free = p.free / 1024**3
    print('DISK', label, f'free_gb={free:.2f}', f'total_gb={p.total / 1024**3:.2f}', flush=True)
    if p.free < 1_500_000_000:
        raise RuntimeError(f'WORKING_DISK_LOW: {free:.2f}GB')


def link_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise RuntimeError(f'REQUIRED_FILE_MISSING: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source)


def ensure_fun_base(runtime_base: Path) -> None:
    required = [
        runtime_base / 'config.json',
        runtime_base / 'Wan2.1_VAE.pth',
        runtime_base / 'models_t5_umt5-xxl-enc-bf16.pth',
        runtime_base / 'google' / 'umt5-xxl' / 'tokenizer.json',
    ]
    missing = [str(p.relative_to(runtime_base)) for p in required if not p.exists()]
    if not missing:
        print('FUN_BASE_READY existing', flush=True)
        return
    print('FUN_BASE_DOWNLOAD', ','.join(missing), flush=True)
    from huggingface_hub import snapshot_download
    disk_report('before_fun_base_download')
    if shutil.disk_usage(ROOT).free < 14 * 1024**3:
        raise RuntimeError('INSUFFICIENT_DISK_FOR_FUN_BASE: need >=14GB free')
    snapshot_download(
        FUN_HF,
        local_dir=str(runtime_base),
        allow_patterns=[
            'config.json',
            'Wan2.1_VAE.pth',
            'models_t5_umt5-xxl-enc-bf16.pth',
            'google/umt5-xxl/*',
        ],
    )
    still = [str(p.relative_to(runtime_base)) for p in required if not p.exists()]
    if still:
        raise RuntimeError(f'FUN_BASE_INCOMPLETE: {still}')
    disk_report('after_fun_base_download')


def patch_echomimic_low_cpu_compat(repo: Path) -> None:
    # EchoMimic's custom loaders were written against an older diffusers import path.
    # Current diffusers moved load_model_dict_into_meta from modeling_utils to
    # model_loading_utils. If the old import fails, EchoMimic silently falls back
    # to full CPU model loading, which is fatal for the 11.4GB umT5-XXL checkpoint.
    replacements = 0
    old = "from diffusers.models.modeling_utils import \\\n                    load_model_dict_into_meta"
    new = "from diffusers.models.model_loading_utils import \\\n                    load_model_dict_into_meta"
    for path in (repo / "src").rglob("*.py"):
        source = path.read_text()
        if old in source:
            path.write_text(source.replace(old, new))
            replacements += 1
    if replacements < 2:
        raise RuntimeError(
            f"LOW_CPU_MEM_COMPAT_PATCH_FAILED: expected >=2 loader imports, found={replacements}"
        )
    print(
        f"ECHOMIMIC_LOW_CPU_MEM_COMPAT_PATCHED files={replacements}",
        flush=True,
    )

def patch_infer_for_cpu_offload(repo: Path) -> None:
    path = repo / 'infer_flash.py'
    text = path.read_text()

    if 'INFER_FLASH_PATCHED_BOUNDED_LONGVIDEO' in text:
        print('INFER_FLASH_ALREADY_PATCHED', flush=True)
        return

    text = text.replace('import sys\n', 'import sys\nimport gc\nimport subprocess\nimport shutil\n', 1)

    text = text.replace(
        '    pipeline.to(device=device)\n',
        '    pipeline.enable_model_cpu_offload(device=device)\n    print("CPU_OFFLOAD_READY", flush=True)\n',
    )

    # Remove the upstream duplicate Flash checkpoint load. The model is already
    # initialized from the verified checkpoint exposed at model_name.
    transformer_patch_start = text.index('    if transformer_path is not None:')
    transformer_patch_end = text.index('    # Get Vae', transformer_patch_start)
    transformer_patch = '''    if transformer_path is not None:
        transformer_root_ckpt = os.path.join(
            model_name, "diffusion_pytorch_model.safetensors"
        )
        if os.path.exists(transformer_root_ckpt):
            print(
                f"FLASH_CHECKPOINT_ALREADY_LOADED path={transformer_root_ckpt}; "
                "skipping duplicate state_dict load",
                flush=True,
            )
        else:
            print(f"From checkpoint: {transformer_path}")
            if transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(transformer_path)
            else:
                state_dict = torch.load(
                    os.path.join(transformer_path, f"checkpoint-{ckpt_idx}.pth"),
                    map_location="cpu",
                )
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
            m, u = transformer.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            print(f"missing keys: {len(m)}, unexpected keys: {len(u)}", flush=True)

'''
    text = text[:transformer_patch_start] + transformer_patch + text[transformer_patch_end:]

    audio_start_marker = '        # Get audio batch '
    audio_end_marker = '        validation_image_start = Image.fromarray(ref_start).convert("RGB")'
    audio_start = text.index(audio_start_marker)
    audio_end = text.index(audio_end_marker, audio_start)
    new_audio = '''        # Keep full audio embeddings on CPU; only the active window goes to GPU.
        audio_embeds = audio_feature_wav2vec.to(dtype=weight_dtype)

        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        center_indices = torch.arange(
            0,
            video_length_actual,
            1,
        ).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0] - 1)
        audio_embeds = audio_embeds[center_indices].contiguous()

        print(f"Audio embeds shape (CPU): {audio_embeds.shape}", flush=True)

'''
    text = text[:audio_start] + new_audio + text[audio_end:]
    start_marker = '        validation_image_start = Image.fromarray(ref_start).convert("RGB")'
    end_marker = '        print(f"Saved output to: {output_video_path}")'
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len(end_marker)

    bounded_block = '''        validation_image_start = Image.fromarray(ref_start).convert("RGB")
        validation_image_end = None
        sample_size_0, sample_size_1 = get_sample_size(validation_image_start, sample_size)

        # 113 frames ~= 4.5 seconds at 25 FPS. Small bounded windows keep
        # CPU/GPU tensors bounded while one model instance is reused.
        chunk_frames = 113
        overlap_frames = 8
        total_frames = video_length_actual
        chunk_dir = os.path.join(save_path, "_bounded_chunks")
        os.makedirs(chunk_dir, exist_ok=True)

        print(
            f"LONG_VIDEO_PLAN total_frames={total_frames} "
            f"chunk_frames={chunk_frames} overlap={overlap_frames} "
            f"bounded_inference=True codec=libx264",
            flush=True,
        )

        chunk_paths = []
        previous_ref = validation_image_start
        start_frame = 0
        chunk_index = 0

        while start_frame < total_frames:
            remaining = total_frames - start_frame
            current_frames = min(chunk_frames, remaining)
            if current_frames <= overlap_frames and start_frame > 0:
                break

            if current_frames > 1:
                current_frames = int(
                    (current_frames - 1) // vae.config.temporal_compression_ratio
                    * vae.config.temporal_compression_ratio
                ) + 1

            end_frame = min(total_frames, start_frame + current_frames)
            actual_frames = end_frame - start_frame
            if actual_frames <= 0:
                break

            input_video, input_video_mask, clip_image = get_image_to_video_latent2(
                previous_ref,
                validation_image_end,
                video_length=actual_frames,
                sample_size=[sample_size_0, sample_size_1],
            )

            partial_audio_embeds = audio_embeds[start_frame:end_frame].unsqueeze(0).to(
                device=device, dtype=weight_dtype
            )

            print(
                f"LONG_VIDEO_CHUNK index={chunk_index} "
                f"start={start_frame} frames={actual_frames}",
                flush=True,
            )

            with torch.inference_mode():
                sample = pipeline(
                    prompt,
                    num_frames=actual_frames,
                    negative_prompt=negative_prompt,
                    audio_embeds=partial_audio_embeds,
                    audio_scale=audio_scale,
                    ip_mask=None,
                    use_un_ip_mask=use_un_ip_mask,
                    height=sample_size_0,
                    width=sample_size_1,
                    generator=generator,
                    neg_scale=neg_scale,
                    neg_steps=neg_steps,
                    use_dynamic_cfg=use_dynamic_cfg,
                    use_dynamic_acfg=use_dynamic_acfg,
                    guidance_scale=guidance_scale,
                    audio_guidance_scale=audio_guidance_scale,
                    num_inference_steps=num_inference_steps,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    cfg_skip_ratio=cfg_skip_ratio,
                    shift=shift,
                ).videos

            write_sample = sample if start_frame == 0 else sample[:, :, overlap_frames:]
            raw_chunk = os.path.join(chunk_dir, f"chunk_{chunk_index:04d}_raw.mp4")
            encoded_chunk = os.path.join(chunk_dir, f"chunk_{chunk_index:04d}.mp4")

            # Encode immediately; never accumulate raw frame/video artifacts.
            save_videos_grid(write_sample, raw_chunk, fps=fps)
            subprocess.run([
                "ffmpeg", "-y", "-v", "error",
                "-i", raw_chunk,
                "-an",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                encoded_chunk,
            ], check=True)
            os.remove(raw_chunk)
            chunk_paths.append(encoded_chunk)

            tail = sample[0, :, -overlap_frames:].detach().float().cpu()
            previous_ref = [
                Image.fromarray(
                    (tail[:, j].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                )
                for j in range(tail.shape[1])
            ]

            del input_video, input_video_mask, partial_audio_embeds
            del sample, write_sample, tail, clip_image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

            start_frame += actual_frames - overlap_frames
            chunk_index += 1

        if not chunk_paths:
            raise RuntimeError("LONG_VIDEO_NO_CHUNKS")

        concat_file = os.path.join(chunk_dir, "concat.txt")
        with open(concat_file, "w") as fh:
            for p in chunk_paths:
                fh.write("file '" + os.path.abspath(p).replace("'", "'\\\\''") + "'\\n")

        silent_path = os.path.join(save_path, f"{image_name}_silent.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            silent_path,
        ], check=True)

        # Final persistent video: H.264 CRF 20 + AAC. This is the only video
        # artifact kept after inference; all intermediates are deleted.
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-i", silent_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(video_length_actual / fps),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "160k",
            "-ar", "48000",
            "-movflags", "+faststart",
            output_video_path,
        ], check=True)

        shutil.rmtree(chunk_dir, ignore_errors=True)
        if os.path.exists(silent_path):
            os.remove(silent_path)

        output_size_mb = os.path.getsize(output_video_path) / 1024**2
        print(
            f"OUTPUT_VIDEO_READY path={output_video_path} "
            f"size_mb={output_size_mb:.1f} duration={video_length_actual / fps:.2f}s",
            flush=True,
        )
        print(f"Saved output to: {output_video_path}")'''

    text = text[:start] + bounded_block + text[end:]

    path.write_text(text)
    print('INFER_FLASH_PATCHED_CPU_OFFLOAD', flush=True)
    print('INFER_FLASH_PATCHED_LOW_CPU_MEM_TRUE', flush=True)
    print('INFER_FLASH_PATCHED_TRANSFORMER_PATH_GUARD', flush=True)
    print('INFER_FLASH_PATCHED_BOUNDED_LONGVIDEO', flush=True)

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    image = find_input('singer.png')
    audio = find_input('bhajan.mp3')
    duration_file = find_input(SECONDS_FILE)
    seconds = int(duration_file.read_text().strip())
    if not 180 <= seconds <= 300 or seconds % 15:
        raise RuntimeError(f'INVALID_DURATION: {seconds}')

    print('INPUT_READY', image, audio, image.stat().st_size, audio.stat().st_size, flush=True)
    run('nvidia-smi')

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('NO_GPU_ALLOCATED')
    vram = torch.cuda.get_device_properties(0).total_memory
    print('GPU_READY', torch.cuda.get_device_name(0), vram, flush=True)
    if vram < 13_000_000_000:
        raise RuntimeError('GPU_VRAM_TOO_SMALL')

    repo = ROOT / 'echomimic_v3'
    models = ROOT / 'models'
    segments = ROOT / 'segments'
    outputs = ROOT / 'outputs'
    run('git', 'clone', '--depth', '1', 'https://github.com/antgroup/echomimic_v3.git', str(repo))
    patch_echomimic_low_cpu_compat(repo)
    patch_infer_for_cpu_offload(repo)

    runtime_requirements = ROOT / 'echomimic_v3_flash_requirements.txt'
    lines = (repo / 'requirements.txt').read_text().splitlines()
    excluded = {'tensorflow', 'tensorflow-gpu', 'tensorflow-cpu', 'retina-face', 'retina_face'}
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            kept.append(line)
            continue
        package = stripped.split('=', 1)[0].split('<', 1)[0].split('>', 1)[0].split('!', 1)[0].strip().lower()
        if package not in excluded:
            kept.append(line)
    runtime_requirements.write_text('\n'.join(kept) + '\n')
    run(sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', '-r', str(runtime_requirements))
    run(sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', 'huggingface_hub', 'pyloudnorm')
    run(sys.executable, '-m', 'pip', 'cache', 'purge')
    disk_report('after_runtime_install')

    runtime_base = models / 'Wan2.1-Fun-V1.1-1.3B-InP'
    runtime_base.mkdir(parents=True, exist_ok=True)
    ensure_fun_base(runtime_base)

    image_encoder_source = find_mounted_file(IMAGE_ENCODER_FILE)
    print('WAN_IMAGE_ENCODER_INPUT', image_encoder_source, flush=True)
    link_file(image_encoder_source, runtime_base / IMAGE_ENCODER_FILE)

    # The EchoMimicV3 Flash transformer is mounted as a Kaggle dataset.
    # This avoids downloading the 3.73GB Xet/LFS object into the T4 working disk.
    flash_root = models / 'echomimicv3-flash-pro'

    # Locate the Flash transformer by its verified checkpoint size/hash, not by
    # the Kaggle dataset folder name (Kaggle may mount the dataset under a
    # generated/renamed path).
    flash_candidates = list(INPUT_ROOT.rglob('diffusion_pytorch_model.safetensors'))
    if not flash_candidates:
        raise RuntimeError('FLASH_DATASET_NOT_MOUNTED: no diffusion_pytorch_model.safetensors found under /kaggle/input')
    expected_size = 3_727_671_120
    flash_candidates = [p for p in flash_candidates if p.stat().st_size == expected_size]
    if not flash_candidates:
        found = [(str(p), p.stat().st_size) for p in INPUT_ROOT.rglob('diffusion_pytorch_model.safetensors')]
        raise RuntimeError(f'FLASH_TRANSFORMER_NOT_FOUND: expected_size={expected_size} candidates={found}')
    flash_source = min(flash_candidates, key=lambda p: len(p.parts))
    print('FLASH_MOUNTED_INPUT', flash_source, flash_source.stat().st_size, flush=True)

    expected_sha256 = '5ebdbb2fc709108bf2a1728fd92eb2874804e4bc0324e92a2cd55425968c85a4'
    actual_size = flash_source.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f'FLASH_TRANSFORMER_SIZE_MISMATCH: expected={expected_size} actual={actual_size}')
    import hashlib
    digest = hashlib.sha256()
    with flash_source.open('rb') as fh:
        for block in iter(lambda: fh.read(16 * 1024 * 1024), b''):
            digest.update(block)
    actual_sha256 = digest.hexdigest()
    print('FLASH_TRANSFORMER_SHA256', actual_sha256, flush=True)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f'FLASH_TRANSFORMER_SHA256_MISMATCH: expected={expected_sha256} actual={actual_sha256}')
    flash_config = flash_source.parent / 'config.json'
    if not flash_config.exists():
        raise RuntimeError(f'FLASH_CONFIG_MISSING: {flash_config}')
    # EchoMimic's config uses transformer_subpath='./', so WanTransformer.from_pretrained
    # looks for the transformer checkpoint directly in runtime_base. The Flash checkpoint
    # must therefore also be exposed at the runtime model root; passing --transformer_path
    # alone is too late because infer_flash.py constructs the transformer first.
    transformer_link = flash_source
    link_file(flash_source, runtime_base / 'diffusion_pytorch_model.safetensors')
    print('FLASH_TRANSFORMER_VERIFIED', transformer_link, actual_size, flush=True)
    print('FLASH_TRANSFORMER_RUNTIME_LINK', runtime_base / 'diffusion_pytorch_model.safetensors', flush=True)
    wav = models / 'chinese-wav2vec2-base'
    from huggingface_hub import snapshot_download
    if not wav.exists():
        snapshot_download(
            'TencentGameMate/chinese-wav2vec2-base',
            local_dir=str(wav),
            allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
        )
    disk_report('after_wav2vec')

    disk_report('models_ready')

    segments.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    norm = ROOT / 'audio16k.wav'
    run('ffmpeg', '-y', '-v', 'error', '-i', str(audio), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(norm))

    fps = 25
    total_frames = int(seconds * fps)
    print(
        'INFERENCE_PLAN',
        'model_loads=1',
        f'target_frames={total_frames}',
        f'target_seconds={seconds}',
        'chunk_seconds=4.52',
        'chunk_frames=113',
        'bounded_inference=True',
        'raw_frame_accumulation=False',
        'publish=False',
        flush=True,
    )

    infer_dir = ROOT / 'infer_output'
    shutil.rmtree(infer_dir, ignore_errors=True)
    infer_dir.mkdir(parents=True)
    print('BOUNDED_INFER_RUNTIME_START', flush=True)
    run(
        sys.executable, str(repo / 'infer_flash.py'),
        '--image_path', str(image),
        '--audio_path', str(audio),
        '--prompt', 'A single Indian devotional singer performing a Hindi bhajan in traditional Indian clothing before the specified Hindu deity in a serene temple setting; only the same singer is visible; natural singing mouth movement, subtle expressive head and upper-body motion, stable identity.',
        '--num_inference_steps', '8',
        '--config_path', str(repo / 'config/config.yaml'),
        '--model_name', str(runtime_base),
        '--ckpt_idx', '50000',
        '--transformer_path', str(transformer_link),
        '--save_path', str(infer_dir),
        '--wav2vec_model_dir', str(wav),
        '--sampler_name', 'Flow_Unipc',
        '--video_length', str(total_frames),
        '--guidance_scale', '5.0',
        '--audio_guidance_scale', '2.5',
        '--audio_scale', '1.0',
        '--neg_scale', '1.0',
        '--neg_steps', '0',
        '--seed', '4300',
        '--enable_teacache',
        '--teacache_threshold', '0.1',
        '--num_skip_start_steps', '5',
        '--weight_dtype', 'float16',
        '--sample_size', '768', '768',
        '--fps', str(fps),
        '--negative_prompt', 'blurry, distorted face, identity drift, extra person, duplicate person, malformed hands, fused fingers, deformed mouth, jitter, flicker, camera cut, text, watermark',
    )

    generated = list(infer_dir.rglob('*.mp4'))
    if not generated:
        raise RuntimeError('ECHOMIMIC_OUTPUT_MISSING')
    generated.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    master = generated[0]
    probe = subprocess.run(
        [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(master),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_duration = float(probe.stdout.strip())
    if abs(actual_duration - seconds) > 0.75:
        raise RuntimeError(
            f'OUTPUT_DURATION_MISMATCH: expected={seconds:.2f}s actual={actual_duration:.2f}s'
        )
    final_size = master.stat().st_size
    if final_size > 1_500_000_000:
        raise RuntimeError(f'FINAL_VIDEO_TOO_LARGE: {final_size / 1024**3:.2f}GB')
    shutil.copy2(master, FINAL_ROOT / 'master.mp4')
    # Keep only the final master video in /kaggle/working.
    # The model/checkpoint/repository files are runtime inputs, not deliverables.
    for cleanup_path in [
        repo,
        models,
        segments,
        outputs,
        infer_dir,
        runtime_requirements,
        norm,
    ]:
        try:
            if isinstance(cleanup_path, Path) and cleanup_path.is_dir():
                shutil.rmtree(cleanup_path, ignore_errors=True)
            elif isinstance(cleanup_path, Path) and cleanup_path.exists():
                cleanup_path.unlink()
        except Exception as exc:
            print('CLEANUP_WARNING', cleanup_path, repr(exc), flush=True)

    final_size = (FINAL_ROOT / 'master.mp4').stat().st_size
    if final_size > 1_500_000_000:
        raise RuntimeError(f'FINAL_VIDEO_TOO_LARGE: {final_size / 1024**3:.2f}GB')

    print(
        'BHAJAN_KAGGLE_WORKER_OK',
        (FINAL_ROOT / 'master.mp4').stat().st_size,
        f'duration={actual_duration:.2f}s',
        flush=True,
    )
    shutil.rmtree(infer_dir, ignore_errors=True)

    return

    videos = sorted(segments.glob('video_*.mp4'))
    if not videos:
        raise RuntimeError('NO_SEGMENTS_GENERATED')
    concat = segments / 'concat.txt'
    concat.write_text(''.join(f"file '{p.resolve()}'\n" for p in videos))
    visual = outputs / 'visual.mp4'
    run('ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat), '-c', 'copy', str(visual))

    final = outputs / 'master.mp4'
    run('ffmpeg', '-y', '-v', 'error', '-i', str(visual), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-t', str(seconds), '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', str(final))
    if not final.exists() or final.stat().st_size < 500_000:
        raise RuntimeError('MASTER_NOT_CREATED')
    shutil.copy2(final, ROOT / 'master.mp4')
    print('BHAJAN_KAGGLE_WORKER_OK', final.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
