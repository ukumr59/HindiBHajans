from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/kaggle/working')
INPUT_ROOT = Path('/kaggle/input')
SECONDS_FILE = 'duration.txt'


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


def link_tree(src: Path, dst: Path, names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = src / name
        target = dst / name
        if not source.exists():
            raise RuntimeError(f'WAN_BASE_COMPONENT_MISSING: {source}')
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source, target_is_directory=source.is_dir())


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

    # The full Wan2.1-Fun-V1.1-1.3B-InP Hub repository is ~19.8 GB. Kaggle's
    # writable disk is only about 20 GB, so downloading it into /kaggle/working
    # leaves no room for the Flash transformer or wav2vec model. The kernel
    # metadata attaches a public Kaggle Model containing the Wan base components
    # under /kaggle/input; input/model mounts are read-only and do not consume
    # the writable working volume. Build a tiny writable symlink tree pointing
    # at those mounted components.
    source_base = find_dir('Wan2.1-T2V-1.3B')
    print('WAN_BASE_INPUT', source_base, flush=True)
    runtime_base = models / 'Wan2.1-Fun-V1.1-1.3B-InP'
    link_tree(source_base, runtime_base, ['vae', 'text_encoder', 'tokenizer', 'image_encoder'])

    from huggingface_hub import snapshot_download
    wav = models / 'chinese-wav2vec2-base'
    flash = models / 'echomimicv3-flash-pro'

    # Use the compact 380 MB PyTorch wav2vec file; the repository's optional
    # 1.14 GB fairseq checkpoint is not used by infer_flash.py.
    if not wav.exists():
        snapshot_download(
            'TencentGameMate/chinese-wav2vec2-base',
            local_dir=str(wav),
            allow_patterns=['config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
        )
    disk_report('after_wav2vec')

    if not flash.exists():
        snapshot_download(
            'BadToBest/EchoMimicV3',
            local_dir=str(flash),
            allow_patterns=['echomimicv3-flash-pro/config.json', 'echomimicv3-flash-pro/diffusion_pytorch_model.safetensors'],
        )
    flash_root = flash / 'echomimicv3-flash-pro'
    if not (flash_root / 'config.json').exists() or not (flash_root / 'diffusion_pytorch_model.safetensors').exists():
        raise RuntimeError('FLASH_MODEL_INCOMPLETE')

    # infer_flash.py first constructs its WanTransformer from model_name/transformer
    # and then replaces the weights with --transformer_path. The Flash checkpoint
    # supplies the compatible transformer config, so expose it as a symlink too.
    transformer_link = runtime_base / 'transformer'
    if transformer_link.exists() or transformer_link.is_symlink():
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

    for i in range(count):
        start = i * segment_seconds
        remain = max(0.1, min(segment_seconds, seconds - start))
        if remain < 0.5:
            break
        wav_file = segments / f'audio_{i:04d}.wav'
        run('ffmpeg', '-y', '-v', 'error', '-ss', f'{start:.3f}', '-i', str(norm), '-t', f'{remain:.3f}', '-ar', '16000', '-ac', '1', str(wav_file))
        raw_dir = segments / f'raw_{i:04d}'
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
            '--transformer_path', str(flash_root / 'diffusion_pytorch_model.safetensors'),
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
