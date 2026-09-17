from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working')
INPUT_ROOT = Path('/kaggle/input')
SECONDS_FILE = 'duration.txt'
IMAGE_ENCODER_FILE = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
BASE_HF = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
FLASH_HF = 'BadToBest/EchoMimicV3'


def run(*args: object) -> None:
    print('RUN:', *args, flush=True)
    subprocess.run([str(x) for x in args], check=True)


def find_input(name: str) -> Path:
    roots = [INPUT_ROOT, ROOT]
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(p for p in root.rglob(name) if p.is_file())
    if not matches:
        raise RuntimeError(f'KAGGLE_INPUT_FILE_MISSING: {name}')
    return min(matches, key=lambda p: len(p.parts))


def find_dir(name: str) -> Path:
    matches: list[Path] = []
    for root in (INPUT_ROOT, ROOT):
        if root.exists():
            matches.extend(p for p in root.rglob(name) if p.is_dir())
    if not matches:
        raise RuntimeError(f'KAGGLE_INPUT_DIR_MISSING: {name}')
    return min(matches, key=lambda p: len(p.parts))


def newest_mp4(directory: Path) -> Path:
    files = sorted(directory.rglob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError(f'NO_MP4_GENERATED: {directory}')
    return files[0]


def disk_report(label: str) -> None:
    p = shutil.disk_usage(ROOT)
    print('DISK', label, f'free_gb={p.free / 1024**3:.2f}', f'total_gb={p.total / 1024**3:.2f}', flush=True)
    if p.free < 1_500_000_000:
        raise RuntimeError(f'WORKING_DISK_LOW: {p.free / 1024**3:.2f}GB')


def link_tree(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = src / name
        target = dst / name
        if not source.exists():
            raise RuntimeError(f'WAN_BASE_COMPONENT_MISSING: {source}')
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(source, target_is_directory=source.is_dir())


def link_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise RuntimeError(f'WAN_IMAGE_ENCODER_MISSING: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source)


def ensure_hf_components(runtime_base: Path, names: list[str]) -> None:
    missing = [name for name in names if not (runtime_base / name).exists()]
    if not missing:
        return
    print('BASE_COMPONENTS_MISSING', ','.join(missing), flush=True)
    from huggingface_hub import snapshot_download
    patterns = [f'{name}/**' for name in missing]
    snapshot_download(BASE_HF, local_dir=str(runtime_base), allow_patterns=patterns)
    still_missing = [name for name in names if not (runtime_base / name).exists()]
    if still_missing:
        raise RuntimeError(f'WAN_BASE_COMPONENT_MISSING_AFTER_HF: {still_missing}')


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
        if package in excluded:
            continue
        kept.append(line)
    runtime_requirements.write_text('\n'.join(kept) + '\n')
    run(sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', '-r', str(runtime_requirements))
    run(sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-q', 'huggingface_hub', 'pyloudnorm')
    run(sys.executable, '-m', 'pip', 'cache', 'purge')
    disk_report('after_runtime_install')

    source_base = find_dir('Wan2.1-T2V-1.3B')
    print('WAN_BASE_INPUT', source_base, flush=True)
    runtime_base = models / 'Wan2.1-Fun-V1.1-1.3B-InP'
    runtime_base.mkdir(parents=True, exist_ok=True)

    # The mounted Kaggle model contains the large T2V text encoder and the
    # small tokenizer/other files when present. EchoMimic also needs a VAE;
    # if the uploaded model omitted it, fetch only the missing VAE/tokenizer
    # from the public T2V Diffusers repo instead of downloading the full model.
    for name in ('text_encoder',):
        source = source_base / name
        if not source.exists():
            raise RuntimeError(f'WAN_BASE_COMPONENT_MISSING: {source}')
        target = runtime_base / name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        target.symlink_to(source, target_is_directory=True)

    for name in ('vae', 'tokenizer'):
        source = source_base / name
        target = runtime_base / name
        if source.exists():
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            target.symlink_to(source, target_is_directory=True)

    ensure_hf_components(runtime_base, ['vae', 'tokenizer'])

    image_encoder_source = find_input(IMAGE_ENCODER_FILE)
    print('WAN_IMAGE_ENCODER_INPUT', image_encoder_source, flush=True)
    link_file(image_encoder_source, runtime_base / 'image_encoder')

    fun_base = find_dir('Wan2.1-Fun-V1.1-1.3B-InP')
    fun_transformer = fun_base / 'diffusion_pytorch_model.safetensors'
    if not fun_transformer.exists():
        raise RuntimeError(f'FLASH_TRANSFORMER_MISSING: {fun_transformer}')
    print('FLASH_TRANSFORMER_INPUT', fun_transformer, fun_transformer.stat().st_size, flush=True)

    from huggingface_hub import snapshot_download
    wav = models / 'chinese-wav2vec2-base'
    flash = models / 'echomimicv3-flash-pro'

    if not wav.exists():
        snapshot_download(
            'TencentGameMate/chinese-wav2vec2-base',
            local_dir=str(wav),
            allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
        )
    disk_report('after_wav2vec')

    # Only the 577-byte Flash config is downloaded. The 3.7 GB Flash weights
    # are mounted from the Kaggle PAI model above, so writable disk stays low.
    if not (flash / 'config.json').exists():
        snapshot_download(
            FLASH_HF,
            local_dir=str(flash),
            allow_patterns=['echomimicv3-flash-pro/config.json'],
        )
    flash_root = flash / 'echomimicv3-flash-pro'
    if not (flash_root / 'config.json').exists():
        raise RuntimeError('FLASH_CONFIG_MISSING')

    transformer_link = runtime_base / 'transformer'
    if transformer_link.exists() or transformer_link.is_symlink():
        if transformer_link.is_dir() and not transformer_link.is_symlink():
            shutil.rmtree(transformer_link)
        else:
            transformer_link.unlink()
    transformer_link.symlink_to(flash_root, target_is_directory=True)
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
                '--transformer_path', str(fun_transformer),
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
