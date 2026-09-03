"""EchoMimicV3 Kaggle dispatcher with authenticated output retrieval.

The worker is submitted as a fresh public Kaggle kernel with a unique slug.
We deliberately do not use a hard-coded/leaderboard slug. Kaggle's output
endpoint may return 403 to anonymous callers, so the GitHub runner passes the
Kaggle API token as a Bearer credential. No kernels.get call is required.
"""
from __future__ import annotations
import io, json, os, re, shutil, subprocess, time, zipfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from app.kaggle_echomimic_dispatch import worker_code

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; KDIR=ROOT/'.kaggle_worker'

def download_output(user: str, slug: str, version: int | None, dest: Path, token: str) -> None:
    suffix=f'?version_number={version}' if version is not None else ''
    urls=[
        f'https://www.kaggle.com/api/v1/kernels/output/download/{user}/{slug}{suffix}',
        f'https://www.kaggle.com/kernels/output/download/{user}/{slug}{suffix}',
    ]
    last=None
    for url in urls:
        try:
            print('PUBLIC_OUTPUT_URL=',url.split('?')[0],flush=True)
            req=Request(url,headers={
                'User-Agent':'HindiBHajans/zero-cost-worker',
                'Authorization':f'Bearer {token}',
                'Accept':'application/zip, application/octet-stream, */*',
            })
            with urlopen(req,timeout=180) as r: data=r.read()
            if not data.startswith(b'PK'): raise RuntimeError(f'KAGGLE_PUBLIC_OUTPUT_NOT_ZIP: {len(data)} bytes')
            dest.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(data)) as z: z.extractall(dest)
            return
        except HTTPError as e:
            last=e; print(f'KAGGLE_OUTPUT_DOWNLOAD_HTTP_{e.code}',flush=True)
            if e.code not in (403,404): raise
    raise RuntimeError(f'KAGGLE_ECHOMIMIC_OUTPUT_DOWNLOAD_FAILED: {last}')

def main():
    token=os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    user=os.getenv('KAGGLE_USERNAME','').strip()
    seconds=int(os.getenv('VIDEO_SECONDS','180'))
    if not token: raise RuntimeError('KAGGLE_API_TOKEN secret is required')
    if not user: raise RuntimeError('KAGGLE_USERNAME secret is required')
    if not 180<=seconds<=300 or seconds%15: raise RuntimeError('seconds must be 180-300 and divisible by 15')
    image=ROOT/'assets'/'uks model image.png'; audio=OUT/'bhajan_source.mp3'
    if not image.exists(): raise RuntimeError(f'Missing singer image: {image}')
    if not audio.exists(): raise RuntimeError(f'Missing generated Hindi bhajan audio: {audio}')
    shutil.rmtree(KDIR,ignore_errors=True); KDIR.mkdir(parents=True)
    shutil.copy2(image,KDIR/'singer.png'); shutil.copy2(audio,KDIR/'bhajan.mp3')
    (KDIR/'duration.txt').write_text(str(seconds),encoding='utf-8')
    (KDIR/'worker.py').write_text(worker_code(),encoding='utf-8')
    slug=f'hindibhajans-echomimic-v3-{int(time.time())}'
    meta={'id':f'{user}/{slug}','title':slug,'code_file':'worker.py','language':'python','kernel_type':'script','is_private':False,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (KDIR/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    env=dict(os.environ); env['KAGGLE_API_TOKEN']=token
    p=subprocess.run(['kaggle','kernels','push','-p',str(KDIR),'--accelerator','NvidiaTeslaT4','--timeout',str(11*60*60)],text=True,capture_output=True,env=env)
    text=(p.stdout or '')+(p.stderr or ''); print(text,flush=True)
    if p.returncode: raise RuntimeError('KAGGLE_ECHOMIMIC_PUSH_FAILED: '+text)
    m=re.search(r'Kernel version (\d+) successfully pushed',text,re.I); version=int(m.group(1)) if m else None
    wait=max(15*60,seconds+12*60)
    print(f'KAGGLE_ECHOMIMIC_LAUNCHED={meta["id"]} VERSION={version} WAIT={wait}s',flush=True); time.sleep(wait)
    outdir=OUT/'kaggle_output'; shutil.rmtree(outdir,ignore_errors=True)
    download_output(user,slug,version,outdir,token)
    xs=list(outdir.rglob('master.mp4'))
    if not xs: raise RuntimeError('KAGGLE_ECHOMIMIC_MASTER_MISSING: output downloaded but master.mp4 was absent')
    shutil.copy2(xs[0],OUT/'master.mp4'); print('KAGGLE_ECHOMIMIC_MASTER_READY',flush=True)

if __name__=='__main__': main()
