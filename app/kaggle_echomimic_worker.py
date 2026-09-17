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
    text = text.replace(first, '    # Keep components on CPU while TeaCache is configured.\n\n    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None', 1)
    text = text.replace(second, '    # T4-safe: offload pipeline components between GPU operations.\n    pipeline.enable_model_cpu_offload(device=device)\n    print("CPU_OFFLOAD_READY", flush=True)\n\n    # Create output directory', 1)
    text = text.replace('low_cpu_mem_usage=True if not fsdp_dit else False,', 'low_cpu_mem_usage=False,', 1)
    text = text.replace('low_cpu_mem_usage=True,\n        torch_dtype=weight_dtype,', 'low_cpu_mem_usage=False,\n        torch_dtype=weight_dtype,', 1)
    # EchoMimic's parser defaults transformer_path to an empty string. Its
    # upstream loader checks only `is not None`, which then tries to open
    # checkpoint-50000.pth even though the transformer is already loaded from
    # the mounted safetensors model. Only load an explicit transformer path.
    text = text.replace('if transformer_path is not None:', 'if transformer_path:', 1)
    path.write_text(text)
    print('INFER_FLASH_PATCHED_CPU_OFFLOAD', flush=True)
    print('INFER_FLASH_PATCHED_LOW_CPU_MEM_FALSE', flush=True)
    print('INFER_FLASH_PATCHED_EMPTY_TRANSFORMER_PATH', flush=True)


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

    transformer_source = find_mounted_file('diffusion_pytorch_model.safetensors')
    if transformer_source.parent.name != 'Wan2.1-Fun-V1.1-1.3B-InP':
        raise RuntimeError(f'FLASH_TRANSFORMER_WRONG_SOURCE: {transformer_source}')
    print('FLASH_TRANSFORMER_INPUT', transformer_source, transformer_source.stat().st_size, flush=True)
    transformer_link = runtime_base / 'diffusion_pytorch_model.safetensors'
    link_file(transformer_source, transformer_link)

    wav = models / 'chinese-wav2vec2-base'
    from huggingface_hub import snapshot_download
    if not wav.exists():
        snapshot_download(
            'TencentGameMate/chinese-wav2vec2-base',
            local_dir=str(wav),
            allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
        )
    disk_report('after_wav2vec')

    flash = models / 'echomimicv3-flash-pro'
    flash_root = flash / 'echomimicv3-flash-pro'
    if not (flash_root / 'config.json').exists():
        snapshot_download(
            FLASH_HF,
            local_dir=str(flash),
            allow_patterns=['echomimicv3-flash-pro/config.json'],
        )
    if not (flash_root / 'config.json').exists():
        raise RuntimeError('FLASH_CONFIG_MISSING')

    disk_report('models_ready')

    segments.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    norm = ROOT / 'audio16k.wav'
    run('ffmpeg', '-y', '-v', 'error', '-i', str(audio), '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(norm))

    fps = 25
    frames = 81
    segment_seconds = frames / fps
    count = int((seconds + segment_seconds - 1) // segment_seconds)
    print('INFERENCE_PLAN', f'segments={count}', f'segment_seconds={segment_seconds:.2f}', flush=True)

    for i in range(count):
        disk_report(f'before_segment_{i:04d}')
        start = i * segment_seconds
        remain = max(0.1, min(segment_seconds, seconds - start))
        if remain < 0.5:
            break
        wav_file = segments / f'audio_{i:04d}.wav'
        raw_dir = segments / f'raw_{i:04d}'
        try:
            run('ffmpeg', '-y', '-v', 'error', '-ss', f'{start:.3f}', '-i', str(norm), '-t', f'{remain:.3f}', '-ar', '16000', '-ac', '1', str(wav_file))
            raw_dir.mkdir(exist_ok=True)
            run(
                sys.executable, str(repo / 'infer_flash.py'),
                '--image_path', str(image),
                '--audio_path', str(wav_file),
                '--prompt', 'A single Indian devotional singer performing a Hindi bhajan in traditional Indian clothing before the specified Hindu deity in a serene temple setting; only the same singer is visible; natural singing mouth movement, subtle expressive head and upper-body motion, stable identity.',
                '--num_inference_steps', '8',
                '--config_path', str(repo / 'config/config.yaml'),
                '--model_name', str(runtime_base),
                '--ckpt_idx', '50000',
                '--save_path', str(raw_dir),
                '--wav2vec_model_dir', str(wav),
                '--sampler_name', 'Flow_Unipc',
                '--video_length', str(frames),
                '--guidance_scale', '5.0',
                '--audio_guidance_scale', '2.5',
                '--audio_scale', '1.0',
                '--neg_scale', '1.0',
                '--neg_steps', '0',
                '--seed', str(4300 + i),
                '--enable_teacache',
                '--teacache_threshold', '0.1',
                '--num_skip_start_steps', '5',
                '--weight_dtype', 'float16',
                '--sample_size', '768', '768',
                '--fps', str(fps),
                '--negative_prompt', 'blurry, distorted face, identity drift, extra person, duplicate person, malformed hands, fused fingers, deformed mouth, jitter, flicker, camera cut, text, watermark',
            )
            raw = newest_mp4(raw_dir)
            silent = segments / f'video_{i:04d}.mp4'
            run('ffmpeg', '-y', '-v', 'error', '-i', str(raw), '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', str(silent))
            if not silent.exists() or silent.stat().st_size < 100_000:
                raise RuntimeError(f'SEGMENT_NOT_CREATED: {silent}')
        finally:
            wav_file.unlink(missing_ok=True)
            if raw_dir.exists():
                shutil.rmtree(raw_dir, ignore_errors=True)
        disk_report(f'after_segment_{i:04d}')

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
