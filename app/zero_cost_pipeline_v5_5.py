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


def _csv_kernel_refs(search_term: str) -> list[str]:
    p = _run('kaggle', 'kernels', 'list', '--mine', '--search', search_term, '--page-size', '100', '-v', check=False, capture=True)
    if p.returncode != 0:
        text = (p.stdout + '\n' + p.stderr).strip()
        print(f'KAGGLE_DISCOVERY: kernels list failed for {search_term!r}: {text[-900:]}', flush=True)
        return []
    refs: list[str] = []
    try:
        rows = list(csv.DictReader(io.StringIO(p.stdout)))
        for row in rows:
            ref = (row.get('ref') or row.get('Ref') or '').strip()
            if ref and '/' in ref:
                refs.append(ref)
    except Exception as exc:
        print(f'KAGGLE_DISCOVERY: CSV parse failed: {exc}', flush=True)
    return refs


def _discover_owned_kernel() -> str | None:
    """Find the real kernel ref from Kaggle's own-kernel listing."""
    for term in (KAGGLE_KERNEL_TITLE, BASE_SLUG, 'Bhajan Aabha'):
        for ref in _csv_kernel_refs(term):
            owner, _, slug = ref.partition('/')
            if owner == KAGGLE_USERNAME and ('bhajan' in slug.lower() or 'ace-step' in slug.lower()):
                print(f'KAGGLE_DISCOVERY: owned kernel found via list: {ref}', flush=True)
                return ref
    print('KAGGLE_DISCOVERY: no owned Bhajan/ACE-Step kernel found in --mine listing', flush=True)
    return None


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
    discovered = _discover_owned_kernel()
    if discovered and _try_pull_existing(discovered):
        return discovered, True

    # Never recreate the known-conflicting slug. Use a distinct slug/title pair.
    slug = REQUESTED_SLUG or f'{BASE_SLUG}-v2'
    print(f'KAGGLE_DISCOVERY: creating isolated fallback kernel slug={slug}', flush=True)
    return f'{KAGGLE_USERNAME}/{slug}', False


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
        slug = kernel_id.split('/', 1)[1]
        title = 'Bhajan Aabha ACE-Step GPU Worker V2' if slug.endswith('-v2') else KAGGLE_KERNEL_TITLE
        metadata = {
            'id': kernel_id,
            'title': title,
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


def _kernel_status(kernel_id: str) -> str:
    p = _run('kaggle', 'kernels', 'status', kernel_id, check=False, capture=True)
    text = (p.stdout + '\n' + p.stderr).strip()
    print('KAGGLE_STATUS:', text[-1600:], flush=True)
    match = re.search(r'(?im)^\s*(?:status|state)\s*[:=]\s*([A-Za-z_]+)', text)
    if match:
        return match.group(1).upper()
    for state in ('COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED', 'FAILED', 'RUNNING', 'QUEUED', 'INITIALIZING'):
        if re.search(rf'\b{state}\b', text.upper()):
            return state
    return 'UNKNOWN'


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
            print('KAGGLE_PUSH: 409 conflict; waiting 60s before retry', flush=True)
            time.sleep(60)
    raise RuntimeError('KAGGLE_ACE_FATAL: Kaggle kernels push remained HTTP 409 after 3 attempts')


def generate_music_kaggle() -> Path:
    _require_kaggle()
    _configure_kaggle_cli()
    kernel_id, existing = _select_kernel()
    root = _prepare_kernel(kernel_id, existing)

    print(f'KAGGLE_LAUNCH: pushing existing={existing} kernel={kernel_id} on free NvidiaTeslaT4 GPU', flush=True)
    _push_kernel(root, kernel_id)

    deadline = time.time() + 65 * 60
    last_state = None
    while time.time() < deadline:
        state = _kernel_status(kernel_id)
        if state != last_state:
            print(f'KAGGLE_LAUNCH: state={state}', flush=True)
            last_state = state
        if state == 'COMPLETE':
            break
        if state in {'ERROR', 'FAILED', 'CANCELLED', 'CANCELED'}:
            raise RuntimeError(f'KAGGLE_ACE_FATAL: kernel ended with state={state}')
        time.sleep(30)
    else:
        raise RuntimeError('KAGGLE_ACE_FATAL: kernel timed out after 65 minutes')

    out_dir = Path('kaggle_output')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    _run('kaggle', 'kernels', 'output', kernel_id, '-p', str(out_dir), '-o', '-q')
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
        'kaggle_kernel': f'{KAGGLE_USERNAME}/{REQUESTED_SLUG or BASE_SLUG}',
        'huggingface_zero_gpu': False,
        'paid_services': False,
        'paid_gpu': False,
        'zero_cost': True,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    (base.OUT / 'manifest.json').write_text(json.dumps({'videos': [state]}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('STATE_OK backend=kaggle_ace_step_gpu_worker kaggle=true')
