"""Zero-cost Hindi bhajan audio dispatcher: Kaggle free GPU + ACE-Step 1.5."""
from __future__ import annotations
import json, os, shutil, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'; WORK=ROOT/'.kaggle_audio_worker'
RAW_WORKER='https://raw.githubusercontent.com/ukumr59/HindiBHajans/main/app/kaggle_ace_step_worker.py'

def run(*args, env=None, check=True, capture=False):
    print('RUN:', ' '.join(map(str,args)), flush=True)
    return subprocess.run(list(map(str,args)), text=True, check=check, env=env, capture_output=capture)

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
    except Exception as e:
        raise RuntimeError(f'Unable to load proven Hindi lyrics/prompt: {e}')
    request={'duration':seconds,'caption':caption,'lyrics':lyrics,'bpm':128,'keyscale':'C Major','timesignature':'4/4','vocal_language':'hi'}
    (WORK/'request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
    bootstrap=f'''#!/usr/bin/env python3\nimport urllib.request,subprocess,sys,shutil\nurl={RAW_WORKER!r}\npath="/kaggle/working/worker.py"\nurllib.request.urlretrieve(url,path)\nshutil.copy2('/kaggle/working/request.json','/kaggle/working/bhajan_request.json')\nsubprocess.run([sys.executable,path],check=True)\n'''
    (WORK/'kernel.py').write_text(bootstrap,encoding='utf-8')
    # IMPORTANT: Kaggle's API can return kernels.get=403 even for a kernel that
    # was just pushed successfully (notably with restricted/new API tokens).
    # Do not use kernels status/output as the control plane. Push is the only
    # operation needed to launch the public script; retrieve the artifact from
    # the kernel's public output URL after the known execution window.
    slug=f'hindibhajans-ace-step-{int(time.time())}'
    meta={'id':f'{user}/{slug}','title':slug,'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':False,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (WORK/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    env=dict(os.environ); env['KAGGLE_API_TOKEN']=token
    push=run('kaggle','kernels','push','-p',str(WORK),'--accelerator','NvidiaTeslaT4','--timeout',str(11*60*60),env=env,capture=True)
    print(push.stdout or push.stderr,flush=True)
    # The push response is authoritative for launch. We deliberately avoid
    # kernels/status and kernels/output because the account token currently
    # returns Permission 'kernels.get' denied for those read APIs.
    # The public kernel executes independently on Kaggle. Give it enough time
    # for startup, model download, generation and ffmpeg conversion.
    wait_seconds=max(15*60, seconds+12*60)
    print(f'KAGGLE_AUDIO_LAUNCHED: {meta["id"]}; waiting {wait_seconds}s before public artifact retrieval',flush=True)
    time.sleep(wait_seconds)
    dl=OUT/'kaggle_audio_output'; shutil.rmtree(dl,ignore_errors=True)
    # Try the CLI artifact path once. If the token still lacks kernels.get,
    # fail with a precise, actionable message rather than looping for hours.
    outp=run('kaggle','kernels','output',meta['id'],'-p',str(dl),'--force','--file-pattern','bhajan_source\\.mp3$',env=env,check=False,capture=True)
    outtext=(outp.stdout or '')+(outp.stderr or '')
    print(outtext,flush=True)
    candidates=list(dl.rglob('bhajan_source.mp3')) if dl.exists() else []
    if not candidates:
        raise RuntimeError('KAGGLE_AUDIO_ARTIFACT_ACCESS_FAILED: Kaggle accepted the kernel push, but this API token cannot read kernel output (kernels.get=403). Replace KAGGLE_API_TOKEN with a Kaggle API token that has kernel read access, then rerun. No Hugging Face path is involved.')
    shutil.copy2(candidates[0],OUT/'bhajan_source.mp3')
    print('KAGGLE_AUDIO_READY',OUT/'bhajan_source.mp3',(OUT/'bhajan_source.mp3').stat().st_size,flush=True)

if __name__=='__main__': main()
