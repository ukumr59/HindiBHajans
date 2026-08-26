from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'worker' / 'kaggle_worker.ipynb'


def main() -> None:
    username = os.getenv('KAGGLE_USERNAME', '').strip()
    token = os.getenv('KAGGLE_API_TOKEN', '').strip()
    if not username or not token:
        print('SETUP_PENDING: KAGGLE_USERNAME/KAGGLE_API_TOKEN are not configured.')
        return

    slug = 'bhajan-aabha-worker'
    kernel_id = f'{username}/{slug}'
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = token

    # Kaggle returns HTTP 409 from SaveKernel when the previous version of
    # the same notebook is still queued/running/finalizing. A blind retry
    # is not enough: wait for the active kernel to become terminal first.
    # The official CLI exposes `kernels status` for this purpose.
    max_wait_seconds = 25 * 60
    poll_seconds = 30
    waited = 0
    while waited < max_wait_seconds:
        status = subprocess.run(
            ['kaggle', 'kernels', 'status', kernel_id],
            env=env,
            text=True,
            capture_output=True,
        )
        status_text = f'{status.stdout}\n{status.stderr}'.strip()
        if status.stdout:
            print(status.stdout)
        if status.stderr:
            print(status.stderr)

        # If the notebook cannot be found, proceed with the push. Otherwise
        # only wait for states that can cause SaveKernel conflicts.
        upper = status_text.upper()
        active = any(word in upper for word in ('RUNNING', 'QUEUED', 'ENQUEUED', 'STARTING'))
        if not active:
            break

        print(f'KAGGLE_ACTIVE: {kernel_id} is still active; waiting {poll_seconds}s ({waited}/{max_wait_seconds}s).')
        time.sleep(poll_seconds)
        waited += poll_seconds

    if waited >= max_wait_seconds:
        print('KAGGLE_DISPATCH_FAILED: existing Kaggle worker remained active for 25 minutes.')
        raise SystemExit(1)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy2(WORKER, d / 'bhajan-aabha-worker.ipynb')
        metadata = {
            'id': kernel_id,
            'title': 'Bhajan Aabha Worker',
            'code_file': 'bhajan-aabha-worker.ipynb',
            'language': 'python',
            'kernel_type': 'notebook',
            'is_private': True,
            'enable_gpu': True,
            'enable_internet': True,
            'dataset_sources': [],
            'competition_sources': [],
            'kernel_sources': [],
            'model_sources': [],
        }
        (d / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

        # A short retry remains useful for the small finalization window
        # between a terminal status and SaveKernel becoming writable.
        for attempt in range(1, 5):
            p = subprocess.run(
                ['kaggle', 'kernels', 'push', '-p', str(d)],
                env=env,
                text=True,
                capture_output=True,
            )
            if p.stdout:
                print(p.stdout)
            if p.returncode == 0:
                break

            combined = f'{p.stdout}\n{p.stderr}'
            if p.stderr:
                print(p.stderr)
            if '409' not in combined and 'Conflict' not in combined:
                raise SystemExit(p.returncode)
            if attempt == 4:
                print('KAGGLE_DISPATCH_FAILED: persistent 409 Conflict after status wait and 4 attempts.')
                raise SystemExit(p.returncode)

            wait_seconds = 30 * attempt
            print(f'KAGGLE_DISPATCH_RETRY: finalization 409; waiting {wait_seconds}s before attempt {attempt + 1}/4.')
            time.sleep(wait_seconds)

    print(f'Dispatched Kaggle worker: {kernel_id}')


if __name__ == '__main__':
    main()
