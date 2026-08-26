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
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy2(WORKER, d / 'bhajan-aabha-worker.ipynb')
        metadata = {
            'id': f'{username}/{slug}',
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
        env = os.environ.copy()
        env['KAGGLE_API_TOKEN'] = token

        # Kaggle may return HTTP 409 while the previous version of this
        # notebook is still being registered/finalized. Retry only that
        # transient condition; fail immediately for other errors.
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
                print('KAGGLE_DISPATCH_FAILED: persistent 409 Conflict after 4 attempts.')
                raise SystemExit(p.returncode)

            wait_seconds = 30 * attempt
            print(f'KAGGLE_DISPATCH_RETRY: transient 409; waiting {wait_seconds}s before attempt {attempt + 1}/4.')
            time.sleep(wait_seconds)

    print(f'Dispatched Kaggle worker: {username}/{slug}')


if __name__ == '__main__':
    main()
