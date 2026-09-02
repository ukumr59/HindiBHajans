import json
import runpy
import shutil
from pathlib import Path

ROOT = Path('/kaggle/working')
RUN_CONFIG = ROOT / 'run_config.json'
OUT = ROOT / 'transfer_out'
OUT.mkdir(exist_ok=True)

# Generate the established audio + exact-identity video on the Kaggle GPU.
# No dataset upload is performed here; GitHub retrieves these kernel output files.
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

manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
manifest['transfer'] = 'kaggle_kernel_output'
(OUT / 'manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding='utf-8',
)

# Also leave the expected files at the kernel root so `kaggle kernels output`
# can retrieve them directly.
print('TRANSFER_MODE=KAGGLE_KERNEL_OUTPUT', flush=True)
print(f'TRANSFER_RUN_ID={json.loads(RUN_CONFIG.read_text(encoding="utf-8")).get("run_id", "unknown") if RUN_CONFIG.exists() else "unknown"}', flush=True)
print('TRANSFER_READY=TRUE', flush=True)
