from __future__ import annotations

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
KAGGLE_KERNEL_TITLE = 'Bhajan Aabha ACE-Step GPU Worker'
REQUESTED_SLUG = os.getenv('KAGGLE_KERNEL_SLUG', '').strip()

# Prefer the already-created worker. We never create a stream of disposable
# Kaggle kernels. The 409 seen in the previous run occurred because the code
# tried to create/update a kernel whose slug/title state already existed.
CANDIDATE_SLUGS = []
for _slug in (REQUESTED_SLUG, 'bhajan-aabha-ace-step-gpu-worker', 'bhajan-aabha-ace-step'):
    if _slug and _slug not in CANDIDATE_SLUGS:
        CANDIDATE_SLUGS.append(_slug)


def _run(*args, cwd=None, check=True, capture=False):
    print('KAGGLE_LAUNCH:', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=check, text=True, capture_output=capture)


def _require_kaggle():
    if not KAGGLE_USERNAME:
        raise RuntimeError('SETUP_REQUIRED: KAGGLE_USERNAME repository secret is missing')
    if not os.getenv('KAGGLE_KEY') and not os.getenv('KAGGLE_API_TOKEN'):
        raise RuntimeError('SETUP_REQUIRED: KAGGLE_KEY or KAGGLE_API_TOKEN repository secret is missing')


def _kernel_exists(kernel_id: str) -> bool:
    p = _run('kaggle', 'kernels', 'status', kernel_id, check=False, capture=True)
    text = (p.stdout + '\n' + p.stderr).strip()
    if p.returncode == 0:
        print(f'KAGGLE_DISCOVERY: existing kernel {kernel_id}', flush=True)
        return True
    print(f'KAGGLE_DISCOVERY: {kernel_id} not available ({text[-500:]})', flush=True)
    return False


def _select_kernel() -> tuple[str, bool]:
    for slug in CANDIDATE_SLUGS:
        kernel_id = f'{KAGGLE_USERNAME}/{slug}'
        if _kernel_exists(kernel_id):
            return kernel_id, True
    # No existing kernel: create exactly one, with a title whose slug matches.
    slug = 'bhajan-aabha-ace-step-gpu-worker'
    return f'{KAGGLE_USERNAME}/{slug}', False


def _prepare_kernel(kernel_id: str, existing: bool):
    root = Path('kaggle_job')
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    if existing:
        # Pull the existing metadata first. Kaggle documents this as the safe
        # update workflow for an existing kernel. Most importantly, we keep
        # its established title/slug relationship instead of sending a new
        # title that can trigger HTTP 409 Conflict.
        _run('kaggle', 'kernels', 'pull', '-p', str(root), '-k', kernel_id, '-m')
        print(f'KAGGLE_METADATA_OK: reusing existing kernel {kernel_id}', flush=True)
    else:
        print(f'KAGGLE_METADATA_OK: creating new kernel {kernel_id}', flush=True)

    # Replace only the executable worker and request payload. Do not replace
    # the existing kernel's title/identity metadata.
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
        # For an existing kernel title is optional; leaving the established
        # title untouched avoids the title/slug conflict that caused 409.
        metadata['id'] = kernel_id
        metadata['code_file'] = 'worker.py'
        metadata['language'] = 'python'
        metadata['kernel_type'] = metadata.get('kernel_type', 'script')
        metadata['is_private'] = True
        metadata['enable_gpu'] = True
        metadata['enable_internet'] = True
        metadata['machine_shape'] = 'NvidiaTeslaT4'
        metadata.pop('title', None)
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
    print(f'KAGGLE_METADATA_READY: id={kernel_id} existing={existing} gpu=NvidiaTeslaT4 title_in_push={"title" in metadata}', flush=True)
    return root


def _configure_kaggle_cli():
    _run('python', '-m', 'pip', 'install', '-q', '-U', 'kaggle')
    print(f'KAGGLE_LAUNCH: authenticated as {KAGGLE_USERNAME}', flush=True)


def _kernel_status(kernel_id: str) -> str:
    p = _run('kaggle', 'kernels', 'status', kernel_id, capture=True)
    text = (p.stdout + '\n' + p.stderr).strip()
    print('KAGGLE_STATUS:', text[-1600:], flush=True)
    match = re.search(r'(?im)^\s*(?:status|state)\s*[:=]\s*([A-Za-z_]+)', text)
    if match:
        return match.group(1).upper()
    for state in ('COMPLETE', 'ERROR', 'CANCELLED', 'CANCELED', 'FAILED', 'RUNNING', 'QUEUED', 'INITIALIZING'):
        if re.search(rf'\b{state}\b', text.upper()):
            return state
    return 'UNKNOWN'


def generate_music_kaggle() -> Path:
    _require_kaggle()
    _configure_kaggle_cli()
    kernel_id, existing = _select_kernel()
    root = _prepare_kernel(kernel_id, existing)

    print(f'KAGGLE_LAUNCH: pushing existing={existing} kernel={kernel_id} on free NvidiaTeslaT4 GPU', flush=True)
    _run('kaggle', 'kernels', 'push', '-p', str(root))

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
    candidates = list(out_dir.rglob('bhajan_source.mp3'))
    if not candidates:
        candidates = list(out_dir.rglob('*.mp3'))
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
        'kaggle_kernel': f'{KAGGLE_USERNAME}/{CANDIDATE_SLUGS[0] if CANDIDATE_SLUGS else "bhajan-aabha-ace-step-gpu-worker"}',
        'huggingface_zero_gpu': False,
        'paid_services': False,
        'paid_gpu': False,
        'zero_cost': True,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    (base.OUT / 'manifest.json').write_text(json.dumps({'videos': [state]}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('STATE_OK backend=kaggle_ace_step_gpu_worker kaggle=true')
