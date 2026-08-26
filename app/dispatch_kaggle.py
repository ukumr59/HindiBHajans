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
DATASET_SLUG_PREFIX = 'bhajan-aabha-output'


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
    dataset_slug = f'{DATASET_SLUG_PREFIX}-{run_id.lower()}'
    dataset_id = f'{username}/{dataset_slug}'
    print(f'KAGGLE_WORKER_ID: {kernel_id}')
    print(f'KAGGLE_OUTPUT_DATASET: {dataset_id}')

    github_output = os.getenv('GITHUB_OUTPUT', '').strip()
    if github_output:
        with open(github_output, 'a', encoding='utf-8') as f:
            f.write(f'kernel_id={kernel_id}\n')
            f.write(f'dataset_id={dataset_id}\n')

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        notebook_path = d / 'bhajan-aabha-worker.ipynb'
        shutil.copy2(WORKER, notebook_path)

        # Kaggle's kernel-output API currently has a server-side permission
        # defect for some owner notebooks. Do not depend on it. Instead,
        # append a final cell that publishes the generated files as a private
        # Kaggle Dataset. The GitHub controller downloads that dataset, then
        # deletes it after the GitHub artifact is created.
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        notebook.setdefault('cells', []).append({
            'cell_type': 'code',
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': [
                'import json, shutil, subprocess, sys\n',
                'from pathlib import Path\n',
                "try:\n    import kagglehub\nexcept ImportError:\n    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'kagglehub'], check=True)\n    import kagglehub\n",
                f"dataset_id = {dataset_id!r}\n",
                "dataset_dir = WORK / 'dataset_upload'\n",
                "if dataset_dir.exists(): shutil.rmtree(dataset_dir)\n",
                "dataset_dir.mkdir(parents=True)\n",
                "shutil.copy2(final_path, dataset_dir / final_path.name)\n",
                "shutil.copy2(srt_path, dataset_dir / 'lyrics.srt')\n",
                "shutil.copy2(OUT / 'run_state.json', dataset_dir / 'run_state.json')\n",
                "(dataset_dir / 'manifest.json').write_text(json.dumps({'run_id': " + repr(run_id) + ", 'dataset_id': dataset_id, 'video': final_path.name}, ensure_ascii=False, indent=2), encoding='utf-8')\n",
                "kagglehub.dataset_upload(dataset_id, str(dataset_dir), version_notes=f'Bhajan Aabha output for GitHub run {" + repr(run_id) + "}')\n",
                "print('DATASET_UPLOAD_COMPLETE:', dataset_id)\n",
            ],
        })
        notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding='utf-8')

        metadata = {
            'id': kernel_id,
            'title': f'Bhajan Aabha Worker {run_id}',
            'code_file': 'bhajan-aabha-worker.ipynb',
            'language': 'python',
            'kernel_type': 'notebook',
            'is_private': True,
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
