from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working')
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


def patch_infer_for_cpu_offload(repo: Path) -> None:
    path = repo / 'infer_flash.py'
    text = path.read_text()
    first = '    pipeline.to(device=device)\n\n    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None'
    second = '    # Create output directory'
    if first not in text or second not in text:
        raise RuntimeError('INFER_FLASH_PATCH_TARGET_NOT_FOUND')
    text = text.replace('import sys\n', 'import sys\nimport subprocess\nimport shutil\n', 1)
    text = text.replace(first, '    # Keep components on CPU; model CPU offload is enabled before inference.\n\n    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None', 1)
    text = text.replace(second, '    # T4-safe: offload pipeline components between GPU operations.\n    # This must happen before any pipeline.to(cuda) call.\n    pipeline.enable_model_cpu_offload(device=device)\n    print("CPU_OFFLOAD_READY", flush=True)\n\n    # Create output directory', 1)
    # Keep checkpoint loading memory-efficient. The previous forced False setting caused
    # large CPU-RAM peaks while materializing the multi-GB transformer/T5 weights.
    text = text.replace('low_cpu_mem_usage=True if not fsdp_dit else False,', 'low_cpu_mem_usage=True,', 1)
    text = text.replace('low_cpu_mem_usage=True,\n        torch_dtype=weight_dtype,', 'low_cpu_mem_usage=True,\n        torch_dtype=weight_dtype,', 1)
    # Wav2Vec is also loaded with the memory-efficient HF loader; otherwise its .bin
    # checkpoint can temporarily require an extra full copy in system RAM.
    text = text.replace(
        'Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True)',
        'Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True, low_cpu_mem_usage=True)',
        1,
    )
    text = text.replace(
        'if transformer_path is not None:',
        'if transformer_path and not os.path.exists(os.path.join(model_name, "diffusion_pytorch_model.safetensors")):',
        1,
    )
    text = text.replace(
        '    pipeline.to(device=device)\\n',
        '',
    )
    start = text.index('        validation_image_start = Image.fromarray(ref_start).convert("RGB")')
    end = text.index('        print(f"Saved output to: {output_video_path}")', start) + len('        print(f"Saved output to: {output_video_path}")')
    long_block = r'''        validation_image_start = Image.fromarray(ref_start).convert("RGB")
        validation_image_end = None
        sample_size_0, sample_size_1 = get_sample_size(validation_image_start, sample_size)

        total_frames = video_length_actual
        chunk_frames = 81
        overlap_frames = 8
        chunk_dir = os.path.join(save_path, "_chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        concat_file = os.path.join(chunk_dir, "concat.txt")
        chunk_paths = []
        previous_ref = validation_image_start
        start_frame = 0
        chunk_index = 0
        clip_image = None

        print(f"LONG_VIDEO_PLAN total_frames={total_frames} chunk_frames={chunk_frames} overlap={overlap_frames}", flush=True)

        while start_frame < total_frames:
            current_frames = min(chunk_frames, total_frames - start_frame)
            if current_frames < 2:
                break
            if current_frames != total_frames - start_frame:
                current_frames = int((current_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            if current_frames <= overlap_frames and start_frame > 0:
                break

            input_video, input_video_mask, clip_image = get_image_to_video_latent2(
                previous_ref, validation_image_end,
                video_length=current_frames,
                sample_size=[sample_size_0, sample_size_1],
            )
            end_frame = min(total_frames, start_frame + current_frames)
            partial_audio_embeds = audio_embeds[:, start_frame:end_frame]

            print(f"LONG_VIDEO_CHUNK index={chunk_index} start={start_frame} frames={current_frames}", flush=True)
            with torch.no_grad():
                sample = pipeline(
                    prompt,
                    num_frames=current_frames,
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

            is_final_chunk = (start_frame + current_frames) >= total_frames
            write_sample = sample if is_final_chunk else (sample[:, :, overlap_frames:] if start_frame > 0 else sample)
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:04d}.mp4")
            save_videos_grid(write_sample, chunk_path, fps=fps)
            chunk_paths.append(chunk_path)

            tail = sample[0, :, -overlap_frames:].detach().float().cpu()
            previous_ref = [
                Image.fromarray(
                    (tail[:, j].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                )
                for j in range(tail.shape[1])
            ]

            del input_video, input_video_mask, partial_audio_embeds, sample, write_sample, tail
            torch.cuda.empty_cache()

            start_frame += current_frames - overlap_frames
            chunk_index += 1

        if not chunk_paths:
            raise RuntimeError("LONG_VIDEO_NO_CHUNKS")

        with open(concat_file, "w") as fh:
            for p in chunk_paths:
                fh.write(f"file '{os.path.abspath(p)}'\\n")

        silent_path = os.path.join(save_path, f"{image_name}_silent.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c", "copy", silent_path
        ], check=True)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-i", silent_path, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-t", str(video_length_actual / fps),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-movflags", "+faststart",
            output_video_path
        ], check=True)

        shutil.rmtree(chunk_dir, ignore_errors=True)
        if os.path.exists(silent_path):
            os.remove(silent_path)
        print(f"Saved output to: {output_video_path}")'''
    text = text[:start] + long_block + text[end:]
    path.write_text(text)
    print('INFER_FLASH_PATCHED_CPU_OFFLOAD', flush=True)
    print('INFER_FLASH_PATCHED_LOW_CPU_MEM_TRUE', flush=True)
    print('INFER_FLASH_PATCHED_TRANSFORMER_PATH_GUARD', flush=True)
    print('INFER_FLASH_PATCHED_LONG_VIDEO_SINGLE_MODEL_LOAD', flush=True)
def main() -> None:
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

    mounted_flash = [
        p for p in INPUT_ROOT.rglob('diffusion_pytorch_model.safetensors')
        if 'hindibhajans-echomimic-flash' in str(p)
    ]
    if not mounted_flash:
        raise RuntimeError('FLASH_DATASET_NOT_MOUNTED')
    flash_source = min(mounted_flash, key=lambda p: len(p.parts))
    print('FLASH_MOUNTED_INPUT', flash_source, flash_source.stat().st_size, flush=True)

    expected_size = 3_727_671_120
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
    print('INFERENCE_PLAN', f'model_loads=1', f'target_frames={total_frames}', f'target_seconds={seconds}', flush=True)

    infer_dir = ROOT / 'infer_output'
    shutil.rmtree(infer_dir, ignore_errors=True)
    infer_dir.mkdir(parents=True)
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
    shutil.copy2(generated[0], ROOT / 'master.mp4')
    print('BHAJAN_KAGGLE_WORKER_OK', (ROOT / 'master.mp4').stat().st_size, flush=True)
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
