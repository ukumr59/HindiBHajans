from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import app.zero_cost_pipeline_v5_4 as longform
import app.zero_cost_pipeline_v5_2 as music

base = longform.base
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME', '').strip()
KAGGLE_KEY = os.getenv('KAGGLE_KEY', '').strip()
BASE_SLUG = 'bhajan-aabha-ace-step-gpu-worker'


def _run(*args, cwd=None, check=True, capture=False):
    print('KAGGLE_LAUNCH:', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=check, text=True, capture_output=capture)


def _require_kaggle():
    if not KAGGLE_USERNAME:
        raise RuntimeError('SETUP_REQUIRED: KAGGLE_USERNAME repository secret is missing')
    # Prefer the legacy username/key pair when both credential styles exist.
    # The current Kaggle CLI prioritizes KAGGLE_API_TOKEN over legacy credentials,
    # and the token-backed path is the one returning Permission kernels.get denied
    # for this public worker. The legacy key already has the permissions needed
    # for push/status/output in this workflow.
    if not KAGGLE_KEY and not os.getenv('KAGGLE_API_TOKEN'):
        raise RuntimeError('SETUP_REQUIRED: KAGGLE_KEY or KAGGLE_API_TOKEN repository secret is missing')


def _select_kernel() -> tuple[str, str]:
    stamp = str(time.time_ns())[-12:]
    slug = f'{BASE_SLUG}-{stamp}'
    title = f'Bhajan Aabha ACE-Step GPU Worker {stamp}'
    print(f'KAGGLE_DISCOVERY: creating unique kernel slug={slug}', flush=True)
    print(f'KAGGLE_DISCOVERY: matching kernel title={title}', flush=True)
    return f'{KAGGLE_USERNAME}/{slug}', title


def _prepare_kernel(kernel_id: str, title: str):
    root = Path('kaggle_job')
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    shutil.copy2(Path(__file__).with_name('kaggle_ace_step_worker.py'), root / 'worker.py')
    request = {
        'caption': base.PACK['music_prompt'],
        'lyrics': base.PACK['lyrics'],
        'duration': int(base.VIDEO_SECONDS),
        'bpm': 128,
        'keyscale': 'C Major',
        'timesignature': '4/4',
        'vocal_language': 'hi',
    }
    (root / 'bhajan_request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')

    metadata = {
        'id': kernel_id,
        'title': title,
        'code_file': 'worker.py',
        'language': 'python',
        'kernel_type': 'script',
        # Keep the worker public. Output retrieval is performed with the account's
        # legacy Kaggle credentials rather than the newer token-precedence path.
        'is_private': False,
        'enable_gpu': True,
        'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [],
        'competition_sources': [],
        'kernel_sources': [],
        'model_sources': [],
    }
    metadata_path = root / 'kernel-metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'KAGGLE_METADATA_READY: id={kernel_id} title={title} gpu=NvidiaTeslaT4 public=true', flush=True)
    return root


def _configure_kaggle_cli():
    _run('python', '-m', 'pip', 'install', '-q', '-U', 'kaggle')
    if KAGGLE_KEY:
        # Kaggle CLI gives KAGGLE_API_TOKEN precedence over legacy credentials.
        # Remove the token from this process so every CLI call below uses the
        # known-good KAGGLE_USERNAME + KAGGLE_KEY pair.
        os.environ.pop('KAGGLE_API_TOKEN', None)
        os.environ.pop('KAGGLE_ACCESS_TOKEN', None)
        print('KAGGLE_AUTH: using legacy KAGGLE_USERNAME + KAGGLE_KEY credentials', flush=True)
    else:
        print('KAGGLE_AUTH: using KAGGLE_API_TOKEN fallback', flush=True)
    print(f'KAGGLE_LAUNCH: authenticated as {KAGGLE_USERNAME}', flush=True)


def _push_kernel(root: Path, kernel_id: str):
    for attempt in range(1, 3):
        print(f'KAGGLE_LAUNCH: push attempt {attempt}/2 kernel={kernel_id}', flush=True)
        p = _run('kaggle', 'kernels', 'push', '-p', str(root), check=False, capture=True)
        output = (p.stdout + '\n' + p.stderr).strip()
        if output:
            print('KAGGLE_PUSH_OUTPUT:', output[-2200:], flush=True)
        if p.returncode == 0:
            return
        if '409' not in output and 'Conflict' not in output:
            raise subprocess.CalledProcessError(p.returncode, p.args, p.stdout, p.stderr)
        if attempt == 1:
            print('KAGGLE_PUSH: transient 409; waiting 15s before one retry', flush=True)
            time.sleep(15)
    raise RuntimeError('KAGGLE_ACE_FATAL: Kaggle kernels push returned HTTP 409 for a fresh slug. Check the preceding slug/title lines.')


def _download_output_once(kernel_id: str, out_dir: Path) -> tuple[bool, str]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    p = _run('kaggle', 'kernels', 'output', kernel_id, '-p', str(out_dir), '-o', '-q', check=False, capture=True)
    text = (p.stdout + '\n' + p.stderr).strip()
    if text:
        print('KAGGLE_OUTPUT:', text[-1200:], flush=True)
    candidates = list(out_dir.rglob('bhajan_source.mp3'))
    if candidates:
        return True, 'COMPLETE'
    if p.returncode != 0:
        low = text.lower()
        # A permission/authentication error is fatal. In particular, match the
        # exact Kaggle message "Permission 'kernels.get' was denied" so this
        # cannot silently enter another endless polling loop.
        fatal_markers = (
            'permission',
            '403',
            '404',
            'authentication',
            'unauthorized',
            'forbidden',
            'kernels.get',
        )
        running_markers = ('has not finished', 'not finished', 'still running', 'no output')
        if any(marker in low for marker in fatal_markers) and not any(marker in low for marker in running_markers):
            raise RuntimeError(f'KAGGLE_ACE_FATAL: output retrieval failed: {text[-1200:]}')
    return False, 'RUNNING'


def generate_music_kaggle() -> Path:
    _require_kaggle()
    _configure_kaggle_cli()
    kernel_id, title = _select_kernel()
    root = _prepare_kernel(kernel_id, title)

    print(f'KAGGLE_LAUNCH: pushing kernel={kernel_id} on free NvidiaTeslaT4 GPU', flush=True)
    _push_kernel(root, kernel_id)

    out_dir = Path('kaggle_output')
    deadline = time.time() + 65 * 60
    poll = 0
    while time.time() < deadline:
        poll += 1
        print(f'KAGGLE_LAUNCH: waiting for GPU output poll={poll} kernel={kernel_id}', flush=True)
        complete, state = _download_output_once(kernel_id, out_dir)
        print(f'KAGGLE_LAUNCH: output_state={state}', flush=True)
        if complete:
            break
        time.sleep(30)
    else:
        raise RuntimeError('KAGGLE_ACE_FATAL: kernel output timed out after 65 minutes')

    candidates = list(out_dir.rglob('bhajan_source.mp3')) or list(out_dir.rglob('*.mp3'))
    if not candidates:
        raise RuntimeError(f'KAGGLE_ACE_FATAL: no MP3 returned by kernel. Files={list(out_dir.rglob("*"))}')
    source = candidates[0]
    target = base.AUDIO / 'bhajan_source.mp3'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if target.stat().st_size < 100_000:
        raise RuntimeError('KAGGLE_ACE_FATAL: returned MP3 is suspiciously small')
    print(f'KAGGLE_ACE_OK: {target} {target.stat().st_size} bytes', flush=True)
    return target


music.generate_music_gradio = generate_music_kaggle
base.ACESTEP_ROOT = 'kaggle://ACE-Step-1.5'


if __name__ == '__main__':
    longform.main()
    state_path = base.OUT / 'run_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    state.update({
        'architecture': 'v5.5-kaggle-gpu-longform',
        'music_backend': 'ACE-Step 1.5 on Kaggle free GPU worker',
        'music_api_mode': 'kaggle_ace_step_gpu_worker',
        'music_model': 'acestep-v15-turbo',
        'music_lm_model': 'acestep-5Hz-lm-0.6B',
        'kaggle': True,
        'huggingface_zero_gpu': False,
        'paid_services': False,
        'paid_gpu': False,
        'zero_cost': True,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    (base.OUT / 'manifest.json').write_text(json.dumps({'videos': [state]}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('STATE_OK backend=kaggle_ace_step_gpu_worker kaggle=true')
