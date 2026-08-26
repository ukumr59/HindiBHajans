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


def run(cmd: list[str], cwd: Path | None = None) -> str:
    print('+', ' '.join(cmd))
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    print(p.stdout)
    if p.returncode:
        print(p.stderr)
        raise SystemExit(p.returncode)
    return p.stdout


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
        p = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(d)], env=env, text=True, capture_output=True)
        print(p.stdout)
        if p.returncode:
            print(p.stderr)
            raise SystemExit(p.returncode)

    print(f'Dispatched Kaggle worker: {username}/{slug}')


if __name__ == '__main__':
    main()
