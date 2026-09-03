"""Zero-cost Hindi bhajan audio dispatcher: Kaggle free GPU + ACE-Step 1.5."""
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; WORK=ROOT/'.kaggle_audio_worker'
RAW_WORKER='https://raw.githubusercontent.com/ukumr59/HindiBHajans/main/app/kaggle_ace_step_worker.py'

def run(*args, env=None, check=True):
    print('RUN:', ' '.join(map(str,args)), flush=True)
    return subprocess.run(list(map(str,args)), text=True, check=check, env=env)

def main():
    token=os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    user=os.getenv('KAGGLE_USERNAME','').strip()
    if not token: raise RuntimeError('KAGGLE_API_TOKEN secret is required')
    if not user: raise RuntimeError('KAGGLE_USERNAME secret is required')
    seconds=int(os.getenv('VIDEO_SECONDS','180'))
    if not 180<=seconds<=300 or seconds%15: raise RuntimeError('VIDEO_SECONDS must be 180-300 and divisible by 15')
    WORK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    request={'duration':seconds,'caption':os.getenv('BH_MUSIC_PROMPT',''),'lyrics':os.getenv('BH_LYRICS',''),'bpm':128,'keyscale':'C Major','timesignature':'4/4','vocal_language':'hi'}
    (WORK/'bhajan_request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
    bootstrap=f'''#!/usr/bin/env python3\nimport urllib.request,subprocess,sys\nurl={RAW_WORKER!r}\npath="/kaggle/working/worker.py"\nurllib.request.urlretrieve(url,path)\nsubprocess.run([sys.executable,path],check=True)\n'''
    (WORK/'kernel.py').write_text(bootstrap,encoding='utf-8')
    meta={'id':f'{user}/hindibhajans-ace-step-daily','title':'HindiBHajans ACE-Step Daily Audio','code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (WORK/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    shutil.copy2(WORK/'bhajan_request.json',WORK/'request.json')
    # The bootstrap downloads the proven worker; request is copied into Kaggle working dir.
    (WORK/'kernel.py').write_text(bootstrap.replace("subprocess.run([sys.executable,path],check=True)","import shutil; shutil.copy2('/kaggle/working/request.json','/kaggle/working/bhajan_request.json'); subprocess.run([sys.executable,path],check=True)"),encoding='utf-8')
    env=dict(os.environ); env['KAGGLE_API_TOKEN']=token
    run('kaggle','kernels','push','-p',str(WORK),'--accelerator','NvidiaTeslaT4','--timeout',str(11*60*60),env=env)
    kernel=meta['id']; deadline=time.time()+11*60*60
    while time.time()<deadline:
        p=subprocess.run(['kaggle','kernels','status',kernel],text=True,capture_output=True,env=env)
        print(p.stdout or p.stderr,flush=True); text=(p.stdout+p.stderr).lower()
        if 'complete' in text: break
        if any(x in text for x in ('error','failed','cancelled','canceled')): raise RuntimeError('KAGGLE_AUDIO_KERNEL_FAILED: '+(p.stdout or p.stderr))
        time.sleep(30)
    else: raise TimeoutError('KAGGLE_AUDIO_KERNEL_TIMEOUT')
    dl=OUT/'kaggle_audio_output'; shutil.rmtree(dl,ignore_errors=True)
    run('kaggle','kernels','output',kernel,'-p',str(dl),'--force',env=env)
    candidates=list(dl.rglob('bhajan_source.mp3'))
    if not candidates: raise RuntimeError('KAGGLE_AUDIO_COMPLETED_BUT_MP3_MISSING')
    shutil.copy2(candidates[0],OUT/'bhajan_source.mp3')
    print('KAGGLE_AUDIO_READY',OUT/'bhajan_source.mp3', (OUT/'bhajan_source.mp3').stat().st_size,flush=True)

if __name__=='__main__': main()
