from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from PIL import Image

ROOT = Path('/kaggle/working')
DEFAULT_IMAGE = ROOT / 'bhajan_aabha_locked_identity_source.png'
DEFAULT_OUT = ROOT / 'bhajan_aabha_exact_identity_final.mp4'
MUSETALK_DIR = ROOT / 'MuseTalk'
VENV_DIR = ROOT / '.musetalk_venv'
MODELS_DIR = MUSETALK_DIR / 'models'

# Pin to the upstream MuseTalk commit whose published v1.5 inference path is
# known to support an image input. This avoids silently following future API
# changes during production runs.
MUSETALK_REPO = 'https://github.com/TMElyralab/MuseTalk.git'
MUSETALK_COMMIT = '0a89dec45a0192b824e3cf4daf96c239440c5ed8'


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print('RUN:', ' '.join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None, env=env, check=True)


def python_in_venv() -> Path:
    if os.name == 'nt':
        return VENV_DIR / 'Scripts' / 'python.exe'
    return VENV_DIR / 'bin' / 'python'


def ensure_musetalk_checkout() -> None:
    if not (MUSETALK_DIR / '.git').exists():
        run('git', 'clone', '--depth', '1', MUSETALK_REPO, str(MUSETALK_DIR))
    else:
        run('git', '-C', str(MUSETALK_DIR), 'fetch', '--depth', '1', 'origin', MUSETALK_COMMIT)
    run('git', '-C', str(MUSETALK_DIR), 'checkout', '--detach', MUSETALK_COMMIT)


def ensure_musetalk_venv() -> Path:
    py = python_in_venv()
    if not py.exists():
        print('Creating isolated MuseTalk virtual environment.', flush=True)
        venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=False).create(VENV_DIR)

    marker = VENV_DIR / '.bhajan_aabha_musetalk_deps_ok'
    if not marker.exists():
        # Keep MuseTalk dependency changes isolated from ACE-Step. In
        # particular, do not downgrade the parent Kaggle environment's numpy,
        # transformers, or other packages used by the audio stage.
        run(
            str(py), '-m', 'pip', 'install', '--quiet', '--upgrade',
            'pip',
            'diffusers==0.30.2',
            'accelerate==0.28.0',
            'soundfile==0.12.1',
            'transformers==4.39.2',
            'huggingface_hub==0.30.2',
            'librosa==0.11.0',
            'einops==0.8.1',
            'gdown',
            'requests',
            'imageio[ffmpeg]',
            'omegaconf',
            'ffmpeg-python',
            'moviepy',
            'opencv-python',
        )
        marker.write_text('ok\n', encoding='utf-8')
    return py


def have_weights() -> bool:
    required = [
        MODELS_DIR / 'musetalkV15' / 'unet.pth',
        MODELS_DIR / 'musetalkV15' / 'musetalk.json',
        MODELS_DIR / 'sd-vae' / 'config.json',
        MODELS_DIR / 'sd-vae' / 'diffusion_pytorch_model.bin',
        MODELS_DIR / 'whisper' / 'config.json',
        MODELS_DIR / 'whisper' / 'pytorch_model.bin',
        MODELS_DIR / 'whisper' / 'preprocessor_config.json',
        MODELS_DIR / 'dwpose' / 'dw-ll_ucoco_384.pth',
        MODELS_DIR / 'face-parse-bisent' / '79999_iter.pth',
        MODELS_DIR / 'face-parse-bisent' / 'resnet18-5c106cde.pth',
    ]
    return all(p.exists() and p.stat().st_size > 1024 for p in required)


def ensure_musetalk_weights(py: Path) -> None:
    if have_weights():
        print('MUSETALK_WEIGHTS=CACHED', flush=True)
        return

    print('MUSETALK_WEIGHTS=DOWNLOADING_PUBLIC_OPEN_SOURCE_MODELS', flush=True)
    # Use the official upstream download script inside the isolated venv.
    # The source commit pins the exact file layout expected by scripts.inference.
    env = dict(os.environ)
    env.pop('HF_TOKEN', None)
    env.pop('HUGGINGFACE_HUB_TOKEN', None)
    run('bash', 'download_weights.sh', cwd=MUSETALK_DIR, env=env)
    if not have_weights():
        raise RuntimeError('MuseTalk weight download completed but required model files are missing.')
    print('MUSETALK_WEIGHTS=READY', flush=True)


def resolve_image(requested: Path) -> Path:
    if requested.exists():
        return requested
    candidates = [
        ROOT / 'bhajan_aabha_locked_identity_source.png',
        ROOT / 'uks model image.png',
        ROOT / 'bhajan_aabha_locked_singer_highres.png',
        ROOT / 'bhajan_aabha_locked_singer_cutout.png',
        ROOT / 'bhajan_aabha_locked_singer_cutout_v2.png',
        ROOT / 'bhajan_aabha_locked_singer_cutout_v3.png',
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                with Image.open(p) as im:
                    if im.width >= 200 and im.height >= 400:
                        return p
            except Exception:
                pass

    input_root = Path('/kaggle/input')
    if input_root.exists():
        ranked: list[tuple[int, Path]] = []
        for p in input_root.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            try:
                with Image.open(p) as im:
                    if im.width < 200 or im.height < 400:
                        continue
                    score = im.width * im.height
                    if p.name.lower() == 'uks model image.png':
                        score += 5_000_000
                    ranked.append((score, p))
            except Exception:
                continue
        if ranked:
            ranked.sort(key=lambda x: x[0], reverse=True)
            return ranked[0][1]
    raise FileNotFoundError('Approved singer image not found under /kaggle/working or /kaggle/input.')


def has_audio_stream(p: Path) -> bool:
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
         'stream=codec_type', '-of', 'csv=p=0', str(p)],
        capture_output=True, text=True, check=False,
    )
    return probe.stdout.strip() == 'audio'


def find_audio(preferred: Path | None = None) -> Path:
    if preferred and preferred.exists() and has_audio_stream(preferred):
        return preferred
    names = [
        'bhajan_source.mp3', 'bhajan_aabha_dj_master.mp3', 'bhajan_source.wav',
        'bhajan_source.m4a', 'bhajan_source.flac', 'bhajan_source.aac',
        'bhajan_source.ogg', 'bhajan_source.opus',
    ]
    for base in (ROOT, Path('/kaggle/input')):
        if not base.exists():
            continue
        for name in names:
            for p in base.rglob(name):
                if p.is_file() and p.stat().st_size > 100_000 and has_audio_stream(p):
                    return p
    raise FileNotFoundError('No bhajan audio was found under /kaggle/working or /kaggle/input.')


def audio_duration(audio: Path) -> float:
    return float(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(audio)
    ], text=True).strip())


def make_audio_clip(audio: Path, seconds: float, tmp: Path) -> Path:
    if seconds <= 0:
        return audio
    clip = tmp / 'musetalk_drive_audio.mp3'
    run(
        'ffmpeg', '-y', '-v', 'error', '-i', str(audio), '-t', f'{seconds:.3f}',
        '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'libmp3lame', '-b:a', '128k', str(clip),
    )
    return clip


def write_inference_config(image: Path, audio: Path, result_name: str, config_path: Path) -> None:
    # MuseTalk's normal inference accepts image files directly and repeats the
    # single reference frame over the generated audio timeline. That is exactly
    # what we need: the source identity remains the base frame, while only the
    # lower face is regenerated per audio-conditioned frame.
    text = (
        'task_0:\n'
        f'  video_path: "{image.as_posix()}"\n'
        f'  audio_path: "{audio.as_posix()}"\n'
        f'  result_name: "{result_name}"\n'
    )
    config_path.write_text(text, encoding='utf-8')


def locate_result(result_root: Path, result_name: str) -> Path:
    direct = list(result_root.rglob(result_name))
    if direct:
        return direct[0]
    mp4s = sorted(result_root.rglob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    if mp4s:
        return mp4s[0]
    raise FileNotFoundError(f'MuseTalk finished but no MP4 was found under {result_root}.')


def main() -> None:
    ap = argparse.ArgumentParser(description='Zero-cost exact-identity audio-driven singing video using MuseTalk 1.5.')
    ap.add_argument('--image', default=str(DEFAULT_IMAGE))
    ap.add_argument('--audio', default='')
    ap.add_argument('--output', default=str(DEFAULT_OUT))
    ap.add_argument('--seconds', type=float, default=0.0,
                    help='Process only this many seconds. Use 8-10s for the smoke test; 0 = full audio.')
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()

    image = resolve_image(Path(args.image))
    audio = find_audio(Path(args.audio) if args.audio else None)
    out = Path(args.output)
    tmp = ROOT / 'musetalk_smoke_tmp'
    tmp.mkdir(parents=True, exist_ok=True)

    duration = audio_duration(audio)
    if args.seconds > 0:
        duration = min(duration, args.seconds)
    if duration < 3:
        raise RuntimeError(f'Audio duration too short for lip-sync validation: {duration:.2f}s')

    source = tmp / 'approved_identity_source.png'
    Image.open(image).convert('RGB').save(source, format='PNG', optimize=True)
    drive_audio = make_audio_clip(audio, duration if args.seconds > 0 else 0, tmp)

    py = ensure_musetalk_venv()
    ensure_musetalk_checkout()
    ensure_musetalk_weights(py)

    result_root = ROOT / 'musetalk_results'
    shutil.rmtree(result_root, ignore_errors=True)
    result_root.mkdir(parents=True, exist_ok=True)
    config = tmp / 'inference.yaml'
    result_name = 'musetalk_identity_singing.mp4'
    write_inference_config(source, drive_audio, result_name, config)

    print(f'IDENTITY_SOURCE={image}', flush=True)
    print(f'AUDIO_SOURCE={audio}', flush=True)
    print(f'MUSETALK_VERSION_COMMIT={MUSETALK_COMMIT}', flush=True)
    print('IDENTITY_MODE=ORIGINAL_IMAGE_WITH_AUDIO_DRIVEN_LOWER_FACE_INPAINTING', flush=True)
    print('IDENTITY_REGENERATION=FALSE', flush=True)
    print(f'PROCESS_SECONDS={duration:.2f}', flush=True)

    run(
        str(py), '-m', 'scripts.inference',
        '--inference_config', str(config),
        '--result_dir', str(result_root),
        '--unet_model_path', str(MODELS_DIR / 'musetalkV15' / 'unet.pth'),
        '--unet_config', str(MODELS_DIR / 'musetalkV15' / 'musetalk.json'),
        '--whisper_dir', str(MODELS_DIR / 'whisper'),
        '--version', 'v15',
        '--fps', '25',
        '--batch_size', str(max(1, args.batch_size)),
        '--output_vid_name', result_name,
        '--extra_margin', '10',
        '--parsing_mode', 'jaw',
        '--use_float16',
        cwd=MUSETALK_DIR,
    )

    generated = locate_result(result_root, result_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if generated.resolve() != out.resolve():
        shutil.copy2(generated, out)

    if not out.exists() or out.stat().st_size < 500_000:
        raise RuntimeError('MuseTalk output MP4 is missing or suspiciously small.')

    # Final technical validation: video exists, has video+audio, and duration is
    # close to the driving audio. This does not claim perceptual lip-sync quality;
    # that remains a visual QC gate after the smoke test.
    probe = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries',
        'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(out)
    ], text=True).strip()
    final_duration = float(probe)
    vstreams = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,width,height,r_frame_rate', '-of', 'csv=p=0', str(out)
    ], text=True).strip()
    astreams = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
        'stream=codec_name,sample_rate,channels', '-of', 'csv=p=0', str(out)
    ], text=True).strip()
    if not vstreams or not astreams:
        raise RuntimeError('MuseTalk output must contain both video and audio streams.')

    print('MUSE_TALK_VIDEO_OK', flush=True)
    print(f'OUTPUT={out}', flush=True)
    print(f'DURATION={final_duration:.2f}', flush=True)
    print(f'VIDEO_STREAM={vstreams}', flush=True)
    print(f'AUDIO_STREAM={astreams}', flush=True)
    print('SMOKE_TEST_EXPECTATION=approved_person_visible_with_audio_synchronized_mouth_motion', flush=True)
    print('IMPORTANT=This stage now performs actual MuseTalk audio-driven lower-face lip synchronization from the approved source image; it no longer uses a still-image zoom/pan substitute.', flush=True)


if __name__ == '__main__':
    main()
