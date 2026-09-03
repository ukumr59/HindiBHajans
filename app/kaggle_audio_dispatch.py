"""Zero-cost Hindi bhajan audio dispatcher: Kaggle free GPU + ACE-Step 1.5.

Kaggle's kernels.get API is deliberately NOT used here. The account has
encountered the current 403 kernels.get failure, so the control plane is:
1) push a public Kaggle kernel (write/execute), then
2) download its public output bundle through Kaggle's public HTTP output URL.
"""
from __future__ import annotations
import io, json, os, re, shutil, subprocess, time, zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; WORK=ROOT/'.kaggle_audio_worker'
RAW_WORKER='https://raw.githubusercontent.com/ukumr59/HindiBHajans/main/app/kaggle_ace_step_worker.py'

def run(*args, env=None, check=True, capture=False):
    print('RUN:', ' '.join(map(str,args)), flush=True)
    return subprocess.run(list(map(str,args)), text=True, check=check, env=env, capture_output=capture)

def download_public_output(kernel_id: str, version: int | None, dest: Path) -> None:
    user, slug = kernel_id.split('/',1)
    url=f'https://www.kaggle.com/api/v1/kernels/output/download/{user}/{slug}'
    if version is not None: url += f'?version_number={version}'
    print('PUBLIC_OUTPUT_URL=',url.split('?')[0],flush=True)
    req=Request(url,headers={'User-Agent':'HindiBHajans/zero-cost-worker'})
    with urlopen(req,timeout=120) as r: data=r.read()
    if not data.startswith(b'PK'): raise RuntimeError(f'KAGGLE_PUBLIC_OUTPUT_NOT_ZIP: HTTP response was {len(data)} bytes')
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z: z.extractall(dest)

def main():
    token=os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    user=os.getenv('KAGGLE_USERNAME','').strip()
    if not token: raise RuntimeError('KAGGLE_API_TOKEN secret is required')
    if not user: raise RuntimeError('KAGGLE_USERNAME secret is required')
    seconds=int(os.getenv('VIDEO_SECONDS','180'))
    if not 180<=seconds<=300 or seconds%15: raise RuntimeError('VIDEO_SECONDS must be 180-300 and divisible by 15')
    WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    try:
        from app.generate_bhajan_audio import LYRICS, PROMPT
        lyrics, caption = LYRICS, PROMPT
    except Exception as e: raise RuntimeError(f'Unable to load proven Hindi lyrics/prompt: {e}')
    request={'duration':seconds,'caption':caption,'lyrics':lyrics,'bpm':128,'keyscale':'C Major','timesignature':'4/4','vocal_language':'hi'}
    (WORK/'request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
    bootstrap=f'''#!/usr/bin/env python3\nimport urllib.request,subprocess,sys,shutil\nurl={RAW_WORKER!r}\npath="/kaggle/working/worker.py"\nurllib.request.urlretrieve(url,path)\nshutil.copy2('/kaggle/working/request.json','/kaggle/working/bhajan_request.json')\nsubprocess.run([sys.executable,path],check=True)\n'''
    (WORK/'kernel.py').write_text(bootstrap,encoding='utf-8')
    slug=f'hindibhajans-ace-step-{int(time.time())}'
    meta={'id':f'{user}/{slug}','title':slug,'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':False,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (WORK/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    env=dict(os.environ); env['KAGGLE_API_TOKEN']=token
    push=run('kaggle','kernels','push','-p',str(WORK),'--accelerator','NvidiaTeslaT4','--timeout',str(11*60*60),env=env,capture=True)
    pushtext=(push.stdout or '')+(push.stderr or '')
    print(pushtext,flush=True)
    m=re.search(r'Kernel version (\d+) successfully pushed',pushtext,re.I)
    version=int(m.group(1)) if m else None
    wait_seconds=max(15*60, seconds+12*60)
    print(f'KAGGLE_AUDIO_LAUNCHED: {meta["id"]}; public output retrieval after {wait_seconds}s; version={version}',flush=True)
    time.sleep(wait_seconds)
    dl=OUT/'kaggle_audio_output'; shutil.rmtree(dl,ignore_errors=True)
    download_public_output(meta['id'],version,dl)
    candidates=list(dl.rglob('bhajan_source.mp3'))
    if not candidates: raise RuntimeError('KAGGLE_AUDIO_ARTIFACT_MISSING: public Kaggle output bundle was downloaded but bhajan_source.mp3 was not present')
    shutil.copy2(candidates[0],OUT/'bhajan_source.mp3')
    print('KAGGLE_AUDIO_READY',OUT/'bhajan_source.mp3',(OUT/'bhajan_source.mp3').stat().st_size,flush=True)

if __name__=='__main__': main()
