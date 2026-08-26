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
    legacy_key = os.getenv('KAGGLE_KEY', '').strip()
    if not username or not (token or legacy_key):
        print('SETUP_PENDING: Kaggle username and credentials are not configured.')
        return

    run_id = ''.join(ch for ch in os.getenv('GITHUB_RUN_ID', '') if ch.isalnum()) or str(int(time.time()))
    slug = f'bhajan-aabha-worker-{run_id.lower()}'
    kernel_id = f'{username}/{slug}'
    print(f'KAGGLE_WORKER_ID: {kernel_id}')

    github_output = os.getenv('GITHUB_OUTPUT', '').strip()
    if github_output:
        with open(github_output, 'a', encoding='utf-8') as f:
            f.write(f'kernel_id={kernel_id}\n')

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy2(WORKER, d / 'bhajan-aabha-worker.ipynb')
        metadata = {
            'id': kernel_id,
            'title': f'Bhajan Aabha Worker {run_id}',
            'code_file': 'bhajan-aabha-worker.ipynb',
            'language': 'python',
            'kernel_type': 'notebook',
            'is_private': False,
            'enable_gpu': True,
            'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': [],
            'competition_sources': [],
            'kernel_sources': [],
            'model_sources': [],
        }
        (d / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        env = os.environ.copy()

        # Prefer the API token: it is the credential already proven to work
        # for kernel push in this repository. Use legacy credentials only as
        # a fallback when the token is absent.
        if token:
            env['KAGGLE_API_TOKEN'] = token
            env.pop('KAGGLE_KEY', None)
        else:
            config_dir = d / 'kaggle-config'
            config_dir.mkdir(mode=0o700)
            (config_dir / 'kaggle.json').write_text(
                json.dumps({'username': username, 'key': legacy_key}),
                encoding='utf-8',
            )
            (config_dir / 'kaggle.json').chmod(0o600)
            env.pop('KAGGLE_API_TOKEN', None)
            env['KAGGLE_CONFIG_DIR'] = str(config_dir)

        for attempt in range(1, 4):
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
            if attempt == 3:
                print('KAGGLE_DISPATCH_FAILED: persistent 409 Conflict for unique worker.')
                raise SystemExit(p.returncode)
            wait_seconds = 20 * attempt
            print(f'KAGGLE_DISPATCH_RETRY: 409; waiting {wait_seconds}s before attempt {attempt + 1}/3.')
            time.sleep(wait_seconds)

    print(f'Dispatched Kaggle worker: {kernel_id}')


if __name__ == '__main__':
    main()
