from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

ROOT = Path('/kaggle/working')
REPO = ROOT / 'HindiBHajans'
ROOT.mkdir(parents=True, exist_ok=True)

# The existing MuseTalk/ACE-Step implementation was developed under Kaggle paths.
# Saturn is only the GPU host; no Kaggle API or Kaggle credential is used here.
if not (REPO / '.git').exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/ukumr59/HindiBHajans.git',str(REPO)], check=True)
else:
    subprocess.run(['git','-C',str(REPO),'pull','--ff-only','origin','main'], check=True)

seconds = int(os.getenv('BH_RUN_SECONDS','10'))
if not (8 <= seconds <= 15 or (180 <= seconds <= 300 and seconds % 15 == 0)):
    raise SystemExit('BH_RUN_SECONDS must be 8-15 for smoke or 180-300 divisible by 15 for production')

config = REPO / 'app' / 'run_config.json'
config.write_text('{"run_id":"saturn","smoke_test":%s,"video_seconds":%d}\n' % ('true' if seconds < 180 else 'false', seconds), encoding='utf-8')

# Run the already-reviewed real pipeline: ACE-Step audio -> MuseTalk.
subprocess.run([sys.executable, str(REPO/'app'/'worker.py')], cwd=str(REPO), check=True)

src = ROOT / 'bhajan_aabha_exact_identity.mp4'
if not src.exists():
    raise SystemExit('SATURN_WORKER_FAILED: expected master MP4 not found')

# The GitHub release upload is deliberately kept outside the GPU code so the
# provider worker remains reusable. A later production bridge can consume the
# local file or publish it to the configured artifact endpoint.
print(f'SATURN_WORKER_OUTPUT={src}', flush=True)
