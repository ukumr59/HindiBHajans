import json
import runpy
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/kaggle/working')
RUN_CONFIG = ROOT / 'run_config.json'
OUT = ROOT / 'transfer_out'
OUT.mkdir(exist_ok=True)

# Generate the established audio + exact-identity video on the Kaggle GPU.
runpy.run_path(str(ROOT / 'worker.py'), run_name='__main__')

expected = [
    ROOT / 'manifest.json',
    ROOT / 'bhajan_source.mp3',
    ROOT / 'bhajan_aabha_exact_identity.mp4',
]
for p in expected:
    if not p.exists():
        raise FileNotFoundError(f'Missing expected output: {p}')
    shutil.copy2(p, OUT / p.name)

cfg = json.loads(RUN_CONFIG.read_text(encoding='utf-8')) if RUN_CONFIG.exists() else {}
run_id = str(cfg.get('run_id', 'unknown'))
dataset_handle = 'bhjanaabha/bhajan-aabha-production-output'

manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
manifest['transfer'] = 'kaggle_dataset_version'
manifest['transfer_dataset'] = dataset_handle
(OUT / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding='utf-8',
)

# Use Kaggle notebook-native authentication to publish a NEW VERSION of the
# permanent public dataset. This removes the failing kernel-output API and
# removes per-run dataset creation/ownership problems.
subprocess.run([
    'python', '-m', 'pip', 'install', '--quiet', '--upgrade', 'kagglehub>=1.0.0'
], check=True)

import kagglehub
kagglehub.dataset_upload(
    dataset_handle,
    str(OUT),
    version_notes=f'Bhajan Aabha automated production run {run_id}',
)

print(f'TRANSFER_DATASET={dataset_handle}', flush=True)
print(f'TRANSFER_RUN_ID={run_id}', flush=True)
print('TRANSFER_MODE=EXISTING_PUBLIC_DATASET_NEW_VERSION', flush=True)
