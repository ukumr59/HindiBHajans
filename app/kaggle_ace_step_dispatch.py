"""Run ACE-Step 1.5 Hindi bhajan generation on Kaggle's free T4 GPU.

This is deliberately separate from the video kernel so the generated audio is
an explicit artifact that can be downloaded, verified, and then uploaded to
the video kernel. No paid provider or fallback is used.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
KDIR = ROOT / ".kaggle_audio_worker"
KAGGLE_KERNEL = "bhajanaabha/hindibhajans-ace-step"
ACE_STEP_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
ACE_STEP_REPO = "https://github.com/ACE-Step/ACE-Step-1.5.git"

LYRICS = """[Intro]
श्री राम... श्री राम... जय जय राम...

[Verse 1]
मन में बसो रघुनंदन, चरणों में मेरा ध्यान
राम नाम की ज्योति जले, रोशन हो हर प्राण

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम

[Verse 2]
दुख की घड़ी में साथ दो, हे दीनदयाल भगवान
तेरा नाम ही आसरा, तेरा नाम ही सम्मान

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम

[Verse 3]
अयोध्या के राजकुमार, करुणा के भंडार
तेरे चरणों में मिल जाए, जीवन का सच्चा सार

[Final Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम
श्री राम जय राम, जय जय राम
जय जय राम... जय जय राम...

[Outro]
श्री राम... जय राम... जय जय राम..."""

PROMPT = "Modern high-energy Hindi devotional bhajan, 128 BPM, 4/4, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, memorable chorus, energetic polished arrangement with dhol/dholak, tabla, temple bells, bansuri, harmonium texture and modern synth production. NOT meditation, NOT ambient, NOT spoken narration, NOT humming, NOT instrumental-only."


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(args), cwd=str(cwd) if cwd else None, env=env, check=True)


def worker_code() -> str:
    return r'''from pathlib import Path
import os, re, subprocess, sys, tempfile

ROOT = Path('/kaggle/working')
OUT = ROOT / 'outputs'
REPO = ROOT / 'ACE-Step-1.5'
SECONDS = int((ROOT / 'duration.txt').read_text().strip())
LYRICS = r''' + repr(LYRICS) + r'''
PROMPT = r''' + repr(PROMPT) + r'''


def run(*args):
    print('RUN:', ' '.join(map(str,args)), flush=True)
    subprocess.run([str(x) for x in args], check=True)


def patch_dtype():
    candidates = [REPO / 'acestep/core/generation/handler/init_service_orchestrator.py']
    candidates.extend(sorted(REPO.glob('acestep/core/generation/handler/init_service*.py')))
    seen = set()
    for target in candidates:
        target = target.resolve()
        if target in seen or not target.exists():
            continue
        seen.add(target)
        text = target.read_text()
        if 'gpu_config.cuda_supports_bfloat16()' not in text or 'self.dtype = torch.float16' not in text:
            continue
        count = text.count('self.dtype = torch.float16')
        if count != 1:
            raise RuntimeError(f'ACE_STEP_SOURCE_LAYOUT_CHANGED:{target}:float16_count={count}')
        patched = text.replace('self.dtype = torch.float16', 'self.dtype = torch.float32', 1)
        patched = patched.replace('using float16 instead of bfloat16', 'using float32 instead of bfloat16', 1)
        target.write_text(patched)
        compile(patched, str(target), 'exec')
        print('ACE_STEP_T4_SOURCE_DTYPE_PATCH=PASS', target, flush=True)
        return
    paths = [str(p.relative_to(REPO)) for p in sorted(REPO.glob('acestep/core/generation/handler/init_service*.py'))]
    raise RuntimeError('ACE_STEP_SOURCE_LAYOUT_CHANGED: no CUDA dtype block found; files=' + ','.join(paths))


def main():
    run('nvidia-smi')
    run(sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip')
    run('git', 'init', str(REPO))
    run('git', '-C', str(REPO), 'remote', 'add', 'origin', 'https://github.com/ACE-Step/ACE-Step-1.5.git')
    run('git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', os.environ['ACE_STEP_COMMIT'])
    run('git', '-C', str(REPO), 'checkout', '--detach', os.environ['ACE_STEP_COMMIT'])
    patch_dtype()
    req = REPO / 'runtime-requirements.txt'
    lines = (REPO / 'requirements.txt').read_text().splitlines()
    req.write_text('\n'.join(x for x in lines if not x.strip().startswith(('flash-attn','triton'))) + '\n')
    run(sys.executable, '-m', 'pip', 'install', '-q', '-r', str(req))
    run(sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', str(REPO))
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['ACESTEP_SAVE_MEMORY'] = '1'
    os.environ['ACESTEP_DTYPE'] = 'float32'
    sys.path.insert(0, str(REPO))
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('ACE_STEP_NO_CUDA')
    print('GPU=', torch.cuda.get_device_name(0), 'CAPABILITY=', torch.cuda.get_device_capability(0), flush=True)
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    work = Path(tempfile.mkdtemp(prefix='bhajan-audio-'))
    dit = AceStepHandler()
    dit.initialize_service(project_root=str(REPO), config_path='acestep-v15-turbo', device='cuda', offload_to_cpu=True)
    actual_dtype = next(dit.model.parameters()).dtype if getattr(dit, 'model', None) is not None else None
    print('ACE_STEP_DIT_DTYPE=', actual_dtype, flush=True)
    if torch.cuda.get_device_capability(0)[0] < 8 and actual_dtype != torch.float32:
        raise RuntimeError('ACE_STEP_T4_DTYPE_POLICY_FAILED')
    llm = LLMHandler()
    llm.initialize(checkpoint_dir=str(REPO/'checkpoints'), lm_model_path='acestep-5Hz-lm-0.6B', backend='pt', device='cuda')
    params = GenerationParams(task_type='text2music', caption=PROMPT, lyrics=LYRICS, bpm=128, keyscale='C Major', timesignature='4/4', vocal_language='hi', duration=float(SECONDS), thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False, use_constrained_decoding=True, inference_steps=8, guidance_scale=1.0, seed=-1, shift=3.0, infer_method='ode', sampler_mode='euler', dcw_enabled=True, dcw_mode='double', dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet='haar')
    result = generate_music(dit_handler=dit, llm_handler=llm, params=params, config=GenerationConfig(batch_size=1, use_random_seed=True, audio_format='wav'), save_dir=str(work/'out'))
    if not result.success or not result.audios:
        raise RuntimeError(f'ACE_STEP_GENERATION_FAILED:{result.error or result.status_message}')
    src = Path(result.audios[0]['path'])
    OUT.mkdir(exist_ok=True)
    out = OUT / 'bhajan_source.mp3'
    run('ffmpeg','-y','-v','error','-i',str(src),'-af','loudnorm=I=-9:TP=-1.0:LRA=7','-ar','48000','-ac','2','-c:a','libmp3lame','-b:a','320k',str(out))
    if out.stat().st_size < 100000: raise RuntimeError('ACE_STEP_AUDIO_TOO_SMALL')
    print('KAGGLE_ACE_STEP_AUDIO_READY=', out, flush=True)

if __name__ == '__main__': main()
'''


def dispatch(seconds: int) -> None:
    token = os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    if not token: raise RuntimeError('KAGGLE_API_TOKEN secret is required')
    if not 180 <= seconds <= 300 or seconds % 15: raise RuntimeError('seconds must be 180-300 and divisible by 15')
    shutil.rmtree(KDIR, ignore_errors=True); KDIR.mkdir(parents=True)
    (KDIR/'worker.py').write_text(worker_code(), encoding='utf-8')
    (KDIR/'duration.txt').write_text(str(seconds), encoding='utf-8')
    (KDIR/'kernel-metadata.json').write_text(json.dumps({'id':KAGGLE_KERNEL,'title':'hindibhajans-ace-step','code_file':'worker.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[],'model_sources':[]},indent=2),encoding='utf-8')
    env=dict(os.environ); env['KAGGLE_API_TOKEN']=token; env['ACE_STEP_COMMIT']=ACE_STEP_COMMIT
    run('kaggle','kernels','push','-p',str(KDIR),'--accelerator','NvidiaTeslaT4','--timeout',str(11*60*60),cwd=ROOT,env=env)
    deadline=time.time()+11*60*60
    while time.time()<deadline:
        p=subprocess.run(['kaggle','kernels','status',KAGGLE_KERNEL],capture_output=True,text=True,env=env)
        status_text=p.stdout or p.stderr
        print(status_text,flush=True)
        t=(p.stdout+p.stderr).lower()
        if 'complete' in t: break
        if any(x in t for x in ('error','failed','cancelled','canceled')):
            # The status command only reports ERROR and hides the actual Kaggle
            # exception. Pull the kernel logs before failing so GitHub Actions
            # contains the real root cause instead of only KAGGLE_*_FAILED.
            print('KAGGLE_ACE_STEP_FETCHING_ERROR_LOGS=START', flush=True)
            lp=subprocess.run(['kaggle','kernels','logs',KAGGLE_KERNEL],capture_output=True,text=True,env=env)
            print(lp.stdout or lp.stderr,flush=True)
            if lp.returncode != 0:
                outdir=OUT/'kaggle_audio_error_output'; shutil.rmtree(outdir,ignore_errors=True)
                op=subprocess.run(['kaggle','kernels','output',KAGGLE_KERNEL,'-p',str(outdir),'--force'],capture_output=True,text=True,env=env)
                print(op.stdout or op.stderr,flush=True)
                for f in sorted(outdir.rglob('*')):
                    if f.is_file() and f.stat().st_size < 2_000_000:
                        try: print(f'--- {f} ---\n{f.read_text(errors="replace")}',flush=True)
                        except Exception: pass
            print('KAGGLE_ACE_STEP_FETCHING_ERROR_LOGS=END', flush=True)
            raise RuntimeError('KAGGLE_ACE_STEP_KERNEL_FAILED')
        time.sleep(30)
    else: raise TimeoutError('KAGGLE_ACE_STEP_KERNEL_TIMEOUT')
    outdir=OUT/'kaggle_audio_output'; shutil.rmtree(outdir,ignore_errors=True)
    run('kaggle','kernels','output',KAGGLE_KERNEL,'-p',str(outdir),'--force',cwd=ROOT,env=env)
    candidates=list(outdir.rglob('bhajan_source.mp3'))
    if not candidates: raise RuntimeError('KAGGLE_ACE_STEP_OUTPUT_MISSING')
    shutil.copy2(candidates[0],OUT/'bhajan_source.mp3')
    print('KAGGLE_ACE_STEP_MASTER_AUDIO_READY',flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--seconds',type=int,default=180)
    dispatch(ap.parse_args().seconds)
