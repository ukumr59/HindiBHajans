import json, os, shutil, subprocess, runpy
from pathlib import Path

RUN_CONFIG = Path('/kaggle/working/run_config.json')
ROOT = Path('/kaggle/working')
OUT = ROOT / 'transfer_out'
OUT.mkdir(exist_ok=True)

# Run the established GPU worker. It creates:
#   /kaggle/working/bhajan_source.mp3
#   /kaggle/working/bhajan_aabha_exact_identity.mp4
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

cfg = json.loads(RUN_CONFIG.read_text()) if RUN_CONFIG.exists() else {}
manifest = json.loads((OUT / 'manifest.json').read_text())
manifest['transfer'] = 'kaggle_dataset'
manifest['transfer_dataset'] = f"bhjanaabha/bhajan-aabha-run-{cfg.get('run_id','unknown')}"
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))

meta = {
    'title': f"Bhajan Aabha Run {cfg.get('run_id','unknown')}",
    'id': f"bhjanaabha/bhajan-aabha-run-{cfg.get('run_id','unknown')}",
    'description': 'Automated Bhajan Aabha production output. Generated on Kaggle GPU and transferred as a public dataset.',
    'licenses': [{'name': 'CC0-1.0'}]
}
(Path(OUT) / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2))

# Kaggle's current dataset API is used for the transfer instead of the broken
# kernel-output endpoint (which can return kernels.get 403 even for public notebooks).
subprocess.run([
    'python', '-m', 'pip', 'install', '--quiet', '--upgrade', 'kagglehub'
], check=True)

# Create a public dataset. The Kaggle CLI is authenticated inside Kaggle notebook
# environments, so no credential is embedded in the worker.
subprocess.run([
    'kaggle', 'datasets', 'create', '-p', str(OUT), '-u'
], check=True)
print(f"TRANSFER_DATASET={meta['id']}")
