from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output'
KDIR = ROOT / '.kaggle_worker'
KERNEL = 'bhajanaabha/hindibhajans-echomimic-v3'
WORKER_SOURCE = ROOT / 'app' / 'kaggle_echomimic_worker.py'


def main(seconds: int) -> None:
    token = os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    if not token:
        raise RuntimeError('KAGGLE_API_TOKEN missing')
    image = ROOT / 'assets' / 'singer_image.png'
    audio = OUT / 'bhajan_source.mp3'
    if not image.exists() or not audio.exists():
        raise RuntimeError('canonical inputs missing')
    if not 180 <= seconds <= 300 or seconds % 15:
        raise RuntimeError(f'invalid duration: {seconds}')
    if not WORKER_SOURCE.exists():
        raise RuntimeError(f'worker source missing: {WORKER_SOURCE}')

    # Build a small, static Kaggle package. Media is uploaded as data files,
    # never embedded into Python source and never interpolated into the worker.
    shutil.rmtree(KDIR, ignore_errors=True)
    payload = KDIR / 'payload'
    payload.mkdir(parents=True)
    shutil.copy2(WORKER_SOURCE, KDIR / 'worker.py')
    shutil.copy2(image, payload / 'singer.png')
    shutil.copy2(audio, payload / 'bhajan.mp3')
    (payload / 'duration.txt').write_text(str(seconds))
    metadata = {
        'id': KERNEL,
        'title': 'hindibhajans-echomimic-v3',
        'code_file': 'worker.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [],
        'competition_sources': [],
        'kernel_sources': [],
        'model_sources': [],
    }
    (KDIR / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))

    # Local gate: the exact source that will be sent to Kaggle must compile.
    subprocess.run(['python', '-m', 'py_compile', str(KDIR / 'worker.py')], cwd=ROOT, check=True)
    worker_bytes = (KDIR / 'worker.py').stat().st_size
    image_bytes = image.stat().st_size
    audio_bytes = audio.stat().st_size
    print('KAGGLE_WORKER_PACKAGE', f'worker_bytes={worker_bytes}', f'image_bytes={image_bytes}', f'audio_bytes={audio_bytes}', flush=True)
    if worker_bytes > 100_000:
        raise RuntimeError(f'WORKER_SOURCE_TOO_LARGE: {worker_bytes}')
    if image_bytes < 100_000 or audio_bytes < 100_000:
        raise RuntimeError('KAGGLE_INPUT_ARTIFACT_TOO_SMALL')

    env = dict(os.environ)
    env['KAGGLE_API_TOKEN'] = token
    subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', str(KDIR), '--timeout', '39600'],
        cwd=ROOT,
        env=env,
        check=True,
    )

    deadline = time.time() + 39600
    while time.time() < deadline:
        p = subprocess.run(
            ['kaggle', 'kernels', 'status', KERNEL],
            capture_output=True,
            text=True,
            env=env,
        )
        status = (p.stdout + p.stderr).strip()
        print(status, flush=True)
        t = status.lower()
        if 'complete' in t and not any(x in t for x in ('incomplete', 'not complete')):
            break
        if any(x in t for x in ('error', 'failed', 'cancelled', 'canceled')):
            log = subprocess.run(
                ['kaggle', 'kernels', 'logs', KERNEL],
                capture_output=True,
                text=True,
                env=env,
            )
            print(log.stdout or log.stderr or 'KAGGLE_KERNEL_LOG_EMPTY', flush=True)
            raise RuntimeError('KAGGLE_KERNEL_FAILED')
        time.sleep(30)
    else:
        raise TimeoutError('KAGGLE_KERNEL_TIMEOUT')

    dest = OUT / 'kaggle_output'
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    subprocess.run(
        ['kaggle', 'kernels', 'output', KERNEL, '-p', str(dest), '--force', '--file-pattern', r'.*master\.mp4$'],
        cwd=ROOT,
        env=env,
        check=True,
    )
    files = list(dest.rglob('master.mp4'))
    if not files:
        raise RuntimeError('KAGGLE_COMPLETED_BUT_MASTER_MP4_MISSING')
    shutil.copy2(files[0], OUT / 'master.mp4')
    print('KAGGLE_ECHOMIMIC_MASTER_READY', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=180)
    main(parser.parse_args().seconds)
