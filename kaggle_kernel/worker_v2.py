import json
import runpy
import shutil
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
dataset_handle = f'bhjanaabha/bhajan-aabha-run-{run_id}'

manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
manifest['transfer'] = 'kaggle_dataset'
manifest['transfer_dataset'] = dataset_handle
(OUT / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding='utf-8',
)

# kagglehub is authenticated by default inside Kaggle notebooks. This avoids
# embedding a credential in the public kernel source and avoids the notebook
# output-download API that was returning kernels.get 403.
import subprocess
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
