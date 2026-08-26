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


def run_kaggle(args: list[str], env: dict[str, str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(['kaggle', *args], env=env, text=True, capture_output=True)
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print(p.stderr)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def main() -> None:
    username = os.getenv('KAGGLE_USERNAME', '').strip()
    token = os.getenv('KAGGLE_API_TOKEN', '').strip()
    legacy_key = os.getenv('KAGGLE_KEY', '').strip()
    github_upload_token = os.getenv('GITHUB_UPLOAD_TOKEN', '').strip()
    repo = os.getenv('GITHUB_REPOSITORY', 'ukumr59/HindiBHajans').strip()

    if not username or not (token or legacy_key):
        print('SETUP_PENDING: Kaggle username and credentials are not configured.')
        raise SystemExit(1)
    if not github_upload_token:
        print('SETUP_PENDING: GITHUB_UPLOAD_TOKEN repository secret is required.')
        raise SystemExit(1)

    run_id = ''.join(ch for ch in os.getenv('GITHUB_RUN_ID', '') if ch.isalnum()) or str(int(time.time()))
    slug = f'bhajan-aabha-worker-{run_id.lower()}'
    kernel_id = f'{username}/{slug}'
    secret_slug = f'bhajan-aabha-secrets-{run_id.lower()}'
    secret_dataset_id = f'{username}/{secret_slug}'
    release_tag = f'bhajan-run-{run_id.lower()}'

    print(f'KAGGLE_WORKER_ID: {kernel_id}')
    print(f'GITHUB_RELEASE_TAG: {release_tag}')

    github_output = os.getenv('GITHUB_OUTPUT', '').strip()
    if github_output:
        with open(github_output, 'a', encoding='utf-8') as f:
            f.write(f'kernel_id={kernel_id}\n')
            f.write(f'release_tag={release_tag}\n')
            f.write(f'secret_dataset_id={secret_dataset_id}\n')

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        env = os.environ.copy()
        if token:
            env['KAGGLE_API_TOKEN'] = token
            env.pop('KAGGLE_KEY', None)
        else:
            config_dir = d / 'kaggle-config'
            config_dir.mkdir(mode=0o700)
            (config_dir / 'kaggle.json').write_text(
                json.dumps({'username': username, 'key': legacy_key}), encoding='utf-8'
            )
            (config_dir / 'kaggle.json').chmod(0o600)
            env.pop('KAGGLE_API_TOKEN', None)
            env['KAGGLE_CONFIG_DIR'] = str(config_dir)

        # Create a unique PRIVATE Kaggle dataset containing only the GitHub upload
        # token. This is the supported headless workaround for secrets: API-pushed
        # kernels do not inherit interactive Kaggle Secrets, but can mount a private
        # dataset through kernel-metadata.json.
        secrets_dir = d / 'secrets-dataset'
        secrets_dir.mkdir()
        (secrets_dir / 'github_upload_token.txt').write_text(github_upload_token, encoding='utf-8')
        (secrets_dir / 'dataset-metadata.json').write_text(
            json.dumps({
                'title': f'Bhajan Aabha Secret {run_id}',
                'id': secret_dataset_id,
                'licenses': [{'name': 'other'}],
                'description': 'Private, temporary automation credential dataset. Delete after run.',
            }),
            encoding='utf-8',
        )
        print(f'Creating private Kaggle secret dataset: {secret_dataset_id}')
        run_kaggle(['datasets', 'create', '-p', str(secrets_dir), '--quiet', '--keep-tabular', '--dir-mode', 'skip'], env)

        notebook_path = d / 'bhajan-aabha-worker.ipynb'
        shutil.copy2(WORKER, notebook_path)
        notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
        notebook.setdefault('cells', []).append({
            'cell_type': 'code',
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': [
                'import json, os, shutil, subprocess, sys\n',
                'from pathlib import Path\n',
                f"RUN_ID = {run_id!r}\n",
                f"REPO = {repo!r}\n",
                f"RELEASE_TAG = {release_tag!r}\n",
                f"SECRET_SLUG = {secret_slug!r}\n",
                "candidates = [Path('/kaggle/input') / SECRET_SLUG, Path('/kaggle/input') / REPO.split('/')[0] / SECRET_SLUG]\n",
                "secret_file = next((p / 'github_upload_token.txt' for p in candidates if (p / 'github_upload_token.txt').exists()), None)\n",
                "if secret_file is None: raise FileNotFoundError('GITHUB_UPLOAD_TOKEN secret dataset was not mounted')\n",
                "github_token = secret_file.read_text(encoding='utf-8').strip()\n",
                "if not github_token: raise RuntimeError('GITHUB_UPLOAD_TOKEN is empty')\n",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'requests'], check=False)\n",
                "import requests\n",
                "headers = {'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {github_token}', 'X-GitHub-Api-Version': '2026-03-10'}\n",
                "release_url = f'https://api.github.com/repos/{REPO}/releases'\n",
                "r = requests.post(release_url, headers=headers, json={'tag_name': RELEASE_TAG, 'name': f'Bhajan Aabha {RUN_ID}', 'body': 'Automated Bhajan Aabha video output.', 'draft': False, 'prerelease': True}, timeout=60)\n",
                "if r.status_code not in (201, 422): raise RuntimeError(f'GITHUB_RELEASE_CREATE_FAILED {r.status_code}: {r.text[:500]}')\n",
                "if r.status_code == 422:\n",
                "    r = requests.get(f'{release_url}/tags/{RELEASE_TAG}', headers=headers, timeout=60)\n",
                "    r.raise_for_status()\n",
                "release = r.json()\n",
                "upload_url = release['upload_url'].split('{')[0]\n",
                "asset = Path(final_path)\n",
                "with asset.open('rb') as fh:\n",
                "    up = requests.post(upload_url, params={'name': asset.name}, headers={**headers, 'Content-Type': 'video/mp4'}, data=fh, timeout=900)\n",
                "if up.status_code != 201: raise RuntimeError(f'GITHUB_ASSET_UPLOAD_FAILED {up.status_code}: {up.text[:500]}')\n",
                "(OUT / 'github_release.json').write_text(json.dumps({'run_id': RUN_ID, 'release_tag': RELEASE_TAG, 'asset_name': asset.name, 'browser_download_url': up.json().get('browser_download_url', '')}, indent=2), encoding='utf-8')\n",
                "print('GITHUB_RELEASE_UPLOAD_COMPLETE:', RELEASE_TAG, asset.name)\n",
                "del github_token\n",
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
            'dataset_sources': [secret_dataset_id],
            'competition_sources': [],
            'kernel_sources': [],
            'model_sources': [],
        }
        (d / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

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
