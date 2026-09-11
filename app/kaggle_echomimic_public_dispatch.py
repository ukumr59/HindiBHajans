"""EchoMimicV3 Kaggle dispatcher with robust completion/output retrieval.

Uses Kaggle's supported CLI workflow: push -> poll kernel status -> retrieve
kernel output with `kaggle kernels output`. We do not depend on undocumented
public HTTP output URLs, which have repeatedly returned 403/404 in production.
"""
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path
from app.kaggle_echomimic_dispatch import worker_code

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; KDIR=ROOT/'.kaggle_worker'

def main():
    token=os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    user=os.getenv('KAGGLE_USERNAME','').strip()
    seconds=int(os.getenv('VIDEO_SECONDS','180'))
    if not token: raise RuntimeError('KAGGLE_API_TOKEN secret is required')
    if not user: raise RuntimeError('KAGGLE_USERNAME secret is required')
    if not 180<=seconds<=300 or seconds%15: raise RuntimeError('seconds must be 180-300 and divisible by 15')

    image=ROOT/'assets'/'singer_image.png'; audio=OUT/'bhajan_source.mp3'
    if not image.exists(): raise RuntimeError(f'Missing singer image: {image}')
    if not audio.exists(): raise RuntimeError(f'Missing generated Hindi bhajan audio: {audio}')

    shutil.rmtree(KDIR,ignore_errors=True); KDIR.mkdir(parents=True)
    shutil.copy2(image,KDIR/'singer.png'); shutil.copy2(audio,KDIR/'bhajan.mp3')
    (KDIR/'duration.txt').write_text(str(seconds),encoding='utf-8')
    (KDIR/'worker.py').write_text(worker_code(),encoding='utf-8')

    slug=f'hindibhajans-echomimic-v3-{int(time.time())}'
    kernel=f'{user}/{slug}'
    meta={'id':kernel,'title':slug,'code_file':'worker.py','language':'python','kernel_type':'script','is_private':False,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (KDIR/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

    env=dict(os.environ); env['KAGGLE_API_TOKEN']=token
    p=subprocess.run(['kaggle','kernels','push','-p',str(KDIR),'--accelerator','NvidiaTeslaT4','--timeout',str(11*60*60)],text=True,capture_output=True,env=env,cwd=ROOT)
    text=(p.stdout or '')+(p.stderr or ''); print(text,flush=True)
    if p.returncode: raise RuntimeError('KAGGLE_ECHOMIMIC_PUSH_FAILED: '+text)

    deadline=time.time()+11*60*60
    while time.time()<deadline:
        s=subprocess.run(['kaggle','kernels','status',kernel],text=True,capture_output=True,env=env,cwd=ROOT)
        status=(s.stdout or '')+(s.stderr or '')
        print(status,flush=True)
        low=status.lower()
        if 'complete' in low:
            print('KAGGLE_ECHOMIMIC_STATUS=COMPLETE',flush=True)
            break
        if any(x in low for x in ('error','failed','cancelled','canceled')):
            raise RuntimeError('KAGGLE_KERNEL_FAILED: '+status)
        time.sleep(30)
    else:
        raise TimeoutError('KAGGLE_KERNEL_TIMEOUT')

    outdir=OUT/'kaggle_output'; shutil.rmtree(outdir,ignore_errors=True); outdir.mkdir(parents=True)
    # Use Kaggle's supported kernel-output command instead of undocumented
    # /api/v1/kernels/output/download URLs that have returned 403/404.
    p=subprocess.run(['kaggle','kernels','output',kernel,'-p',str(outdir),'--force'],text=True,capture_output=True,env=env,cwd=ROOT)
    output_log=(p.stdout or '')+(p.stderr or '')
    print(output_log,flush=True)
    if p.returncode:
        raise RuntimeError('KAGGLE_ECHOMIMIC_OUTPUT_COMMAND_FAILED: '+output_log)

    xs=list(outdir.rglob('master.mp4'))
    if not xs: raise RuntimeError('KAGGLE_ECHOMIMIC_MASTER_MISSING: kernel completed but master.mp4 was not returned by Kaggle')
    shutil.copy2(xs[0],OUT/'master.mp4'); print('KAGGLE_ECHOMIMIC_MASTER_READY',flush=True)

if __name__=='__main__': main()
