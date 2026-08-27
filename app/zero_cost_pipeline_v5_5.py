from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import app.zero_cost_pipeline_v5_4 as longform
import app.zero_cost_pipeline_v5_2 as music

base = longform.base
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME', '').strip()
REQUESTED_SLUG = os.getenv('KAGGLE_KERNEL_SLUG', '').strip()
BASE_SLUG = 'bhajan-aabha-ace-step-gpu-worker'
KAGGLE_KERNEL_TITLE = 'Bhajan Aabha ACE-Step GPU Worker'


def _run(*args, cwd=None, check=True, capture=False):
    print('KAGGLE_LAUNCH:', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=check, text=True, capture_output=capture)


def _require_kaggle():
    if not KAGGLE_USERNAME:
        raise RuntimeError('SETUP_REQUIRED: KAGGLE_USERNAME repository secret is missing')
    if not os.getenv('KAGGLE_KEY') and not os.getenv('KAGGLE_API_TOKEN'):
        raise RuntimeError('SETUP_REQUIRED: KAGGLE_KEY or KAGGLE_API_TOKEN repository secret is missing')


def _try_pull_existing(kernel_id: str) -> bool:
    probe = Path('.kaggle_kernel_probe')
    if probe.exists():
        shutil.rmtree(probe)
    try:
        p = _run('kaggle', 'kernels', 'pull', kernel_id, '-p', str(probe), '-m', check=False, capture=True)
        text = (p.stdout + '\n' + p.stderr).strip()
        if p.returncode == 0 and (probe / 'kernel-metadata.json').exists():
            print(f'KAGGLE_DISCOVERY: existing kernel confirmed by pull: {kernel_id}', flush=True)
            return True
        print(f'KAGGLE_DISCOVERY: pull unavailable for {kernel_id}: {text[-700:]}', flush=True)
        return False
    finally:
        if probe.exists():
            shutil.rmtree(probe, ignore_errors=True)


def _select_kernel() -> tuple[str, bool]:
    # IMPORTANT: every normal run gets a fresh slug. Reusing the previous
    # slug is what caused the repeated SaveKernel HTTP 409 conflict.
    # KAGGLE_KERNEL_SLUG is intentionally ignored for normal production runs
    # so an old secret cannot force us back onto the conflicting kernel.
    run_slug = f'{BASE_SLUG}-{int(time.time())}'
    print(f'KAGGLE_DISCOVERY: creating unique isolated kernel slug={run_slug}', flush=True)
    return f'{KAGGLE_USERNAME}/{run_slug}', False


def _prepare_kernel(kernel_id: str, existing: bool):
    root = Path('kaggle_job')
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    if existing:
        _run('kaggle', 'kernels', 'pull', kernel_id, '-p', str(root), '-m')
        print(f'KAGGLE_METADATA_OK: pulled existing kernel {kernel_id}', flush=True)

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

    metadata_path = root / 'kernel-metadata.json'
    if existing and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata['code_file'] = 'worker.py'
        metadata['language'] = 'python'
        metadata['kernel_type'] = metadata.get('kernel_type', 'script')
        metadata['is_private'] = True
        metadata['enable_gpu'] = True
        metadata['enable_internet'] = True
        metadata['machine_shape'] = 'NvidiaTeslaT4'
    else:
        metadata = {
            'id': kernel_id,
            'title': KAGGLE_KERNEL_TITLE,
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
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'KAGGLE_METADATA_READY: id={kernel_id} existing={existing} gpu=NvidiaTeslaT4 title={metadata.get("title")}', flush=True)
    return root


def _configure_kaggle_cli():
    _run('python', '-m', 'pip', 'install', '-q', '-U', 'kaggle')
    print(f'KAGGLE_LAUNCH: authenticated as {KAGGLE_USERNAME}', flush=True)


def _push_kernel(root: Path, kernel_id: str):
    for attempt in range(1, 4):
        print(f'KAGGLE_LAUNCH: push attempt {attempt}/3 kernel={kernel_id}', flush=True)
        p = _run('kaggle', 'kernels', 'push', '-p', str(root), check=False, capture=True)
        output = (p.stdout + '\n' + p.stderr).strip()
        if output:
            print('KAGGLE_PUSH_OUTPUT:', output[-1800:], flush=True)
        if p.returncode == 0:
            return
        if '409' not in output and 'Conflict' not in output:
            raise subprocess.CalledProcessError(p.returncode, p.args, p.stdout, p.stderr)
        if attempt < 3:
            print('KAGGLE_PUSH: transient 409; waiting 15s before retry', flush=True)
            time.sleep(15)
    raise RuntimeError('KAGGLE_ACE_FATAL: Kaggle kernels push remained HTTP 409 after 3 attempts')


def _download_output_once(kernel_id: str, out_dir: Path) -> tuple[bool, str]:
    """Use the output endpoint as the completion signal; kernels/status is forbidden for this account."""
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
        fatal_markers = ('permission denied', '403', '404', 'authentication', 'unauthorized', 'not found')
        if any(marker in low for marker in fatal_markers) and 'has not finished' not in low and 'not finished' not in low:
            raise RuntimeError(f'KAGGLE_ACE_FATAL: output retrieval failed: {text[-1200:]}')
    return False, 'RUNNING'


def generate_music_kaggle() -> Path:
    _require_kaggle()
    _configure_kaggle_cli()
    kernel_id, existing = _select_kernel()
    root = _prepare_kernel(kernel_id, existing)

    print(f'KAGGLE_LAUNCH: pushing existing={existing} kernel={kernel_id} on free NvidiaTeslaT4 GPU', flush=True)
    _push_kernel(root, kernel_id)

    # Do not call `kaggle kernels status`: the account can push successfully but
    # is denied kernels.get. Polling the kernel output is sufficient and avoids
    # the forbidden status endpoint entirely.
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
