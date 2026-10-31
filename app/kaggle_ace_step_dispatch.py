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
from app.deity_catalog import get_daily_bhajan

ACE_STEP_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(args), cwd=str(cwd) if cwd else None, env=env, check=True)


def worker_code(seconds: int, prompt: str, lyrics: str, bpm: int = 128) -> str:
    return r'''from pathlib import Path
import os, re, shutil, subprocess, sys, tempfile

ROOT = Path('/kaggle/working')
OUT = ROOT / 'outputs'
REPO = ROOT / 'ACE-Step-1.5'
SECONDS = ''' + str(seconds) + r'''
ACE_STEP_COMMIT = ''' + repr(ACE_STEP_COMMIT) + r'''
LYRICS = r''' + repr(lyrics) + r'''
PROMPT = r''' + repr(prompt) + r'''
BPM = ''' + str(bpm) + r'''


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
    run('git', '-C', str(REPO), 'fetch', '--depth', '1', 'origin', ACE_STEP_COMMIT)
    run('git', '-C', str(REPO), 'checkout', '--detach', ACE_STEP_COMMIT)
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
    params = GenerationParams(task_type='text2music', caption=PROMPT, lyrics=LYRICS, bpm=BPM, keyscale='C Major', timesignature='4/4', vocal_language='hi', duration=float(SECONDS), thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False, use_constrained_decoding=True, inference_steps=8, guidance_scale=1.0, seed=-1, shift=3.0, infer_method='ode', sampler_mode='euler', dcw_enabled=True, dcw_mode='double', dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet='haar')
    result = generate_music(dit_handler=dit, llm_handler=llm, params=params, config=GenerationConfig(batch_size=1, use_random_seed=True, audio_format='wav'), save_dir=str(work/'out'))
    if not result.success or not result.audios:
        raise RuntimeError(f'ACE_STEP_GENERATION_FAILED:{result.error or result.status_message}')
    src = Path(result.audios[0]['path'])
    OUT.mkdir(exist_ok=True)
    out = OUT / 'bhajan_source.mp3'
    run('ffmpeg','-y','-v','error','-i',str(src),'-af','loudnorm=I=-9:TP=-1.0:LRA=7','-ar','48000','-ac','2','-c:a','libmp3lame','-b:a','320k',str(out))
    if out.stat().st_size < 100000: raise RuntimeError('ACE_STEP_AUDIO_TOO_SMALL')

    # IMPORTANT: Kaggle packages the entire /kaggle/working tree as kernel output.
    # Keep only the final MP3 so `kaggle kernels output` never tries to download
    # the multi-GB ACE-Step repo/checkpoints again.
    final = ROOT / 'bhajan_source.mp3'
    shutil.copy2(out, final)
    size = final.stat().st_size
    print('KAGGLE_ACE_STEP_AUDIO_READY=', final, 'SIZE=', size, flush=True)

    # Remove the model, checkpoints and temporary generation tree before the
    # kernel terminates. The final artifact above is deliberately left in place.
    shutil.rmtree(REPO, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    print('KAGGLE_ACE_STEP_OUTPUT_CLEANED=PASS', flush=True)

if __name__ == '__main__': main()
'''


def dispatch(seconds: int) -> None:
    token = os.getenv('KAGGLE_API_TOKEN') or os.getenv('KAGGLE_API_TOKEN3')
    if not token: raise RuntimeError('KAGGLE_API_TOKEN secret is required')
    if not 180 <= seconds <= 300 or seconds % 15: raise RuntimeError('seconds must be 180-300 and divisible by 15')
    
    daily_entry = get_daily_bhajan()
    print(f"TODAY_DEITY={daily_entry['deity']} TITLE={daily_entry['title_en']}", flush=True)
    
    OUT.mkdir(exist_ok=True, parents=True)
    seo_info = {
        "title": f"{daily_entry['title_hi']} | {daily_entry['title_en']} | Bhajan Aabha",
        "description": (
            f"{daily_entry['short_caption']}\n\n"
            f"Original Hindu devotional music and performance by Bhajan Aabha.\n"
            f"Deity: {daily_entry['deity']}\n\n"
            f"Lyrics:\n{daily_entry['lyrics']}\n\n"
            f"#Bhajan #Devotional #{daily_entry['deity'].replace(' ', '')} #Hinduism #BhajanAabha"
        ),
        "tags": [
            daily_entry['deity'],
            "Bhajan",
            "Hindi Bhajan",
            "Devotional Song",
            "DJ Bhajan",
            "Hindu Devotional",
            "Bhajan Aabha",
            "Sanatan Dharma"
        ]
    }
    (OUT / "youtube_seo.json").write_text(json.dumps(seo_info, ensure_ascii=False, indent=2), encoding="utf-8")

    env = dict(os.environ); env['KAGGLE_API_TOKEN'] = token
    target_mp3 = OUT / 'bhajan_source.mp3'
    if target_mp3.exists() and target_mp3.stat().st_size > 100_000:
        print('KAGGLE_ACE_STEP_MASTER_AUDIO_ALREADY_EXISTS', target_mp3, flush=True)
        return

    run_id = os.getenv('GITHUB_RUN_ID', 'manual')
    # Check if an earlier attempt of the current run already produced the audio
    check_kernels = [f"bhajanaabha/hindibhajans-ace-{run_id}-1".lower()]
    for prev_k in check_kernels:
        st = subprocess.run(['kaggle', 'kernels', 'status', prev_k], capture_output=True, text=True, env=env)
        st_low = (st.stdout or st.stderr or '').lower()
        if 'complete' in st_low and not any(x in st_low for x in ('incomplete', 'not complete')):
            print(f'Found completed kernel {prev_k}; checking for bhajan_source.mp3 output...', flush=True)
            outdir = OUT / 'kaggle_audio_output'
            shutil.rmtree(outdir, ignore_errors=True)
            subprocess.run(['kaggle', 'kernels', 'output', prev_k, '-p', str(outdir), '--force'], cwd=ROOT, env=env, capture_output=True)
            candidates = list(outdir.rglob('bhajan_source.mp3'))
            if candidates and candidates[0].stat().st_size > 100_000:
                shutil.copy2(candidates[0], target_mp3)
                print('KAGGLE_ACE_STEP_REUSED_COMPLETED_AUDIO', target_mp3, flush=True)
                return

    run_id = os.getenv('GITHUB_RUN_ID', 'manual')
    for attempt in (1, 2):
        kernel = f"bhajanaabha/hindibhajans-ace-{run_id}-{attempt}".lower()
        print(f'=== ATTEMPT {attempt}: LAUNCHING KAGGLE AUDIO KERNEL {kernel} ===', flush=True)
        ok, detail = run_audio_kernel(env, kernel, seconds, daily_entry)
        if ok:
            outdir = OUT / 'kaggle_audio_output'
            shutil.rmtree(outdir, ignore_errors=True)
            output_cmd = ['kaggle', 'kernels', 'output', kernel, '-p', str(outdir), '--force']
            print('RUN:', ' '.join(output_cmd), flush=True)
            op = subprocess.run(output_cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            print(op.stdout, flush=True)
            if op.stderr: print(op.stderr, flush=True)
            candidates = list(outdir.rglob('bhajan_source.mp3'))
            if not candidates:
                raise RuntimeError(f'KAGGLE_ACE_STEP_OUTPUT_MISSING:output_rc={op.returncode}')
            shutil.copy2(candidates[0], target_mp3)
            print('KAGGLE_ACE_STEP_MASTER_AUDIO_READY', flush=True)
            return
        print(f'ATTEMPT {attempt} FAILED: {detail}', flush=True)
        if 'exhausted' in detail.lower():
            raise RuntimeError(detail)
        if attempt < 2:
            print('Waiting 60s before retrying with fresh kernel slug...', flush=True)
            time.sleep(60)
    raise RuntimeError('KAGGLE_ACE_STEP_FAILED_AFTER_RETRIES')


def run_audio_kernel(env: dict[str, str], kernel: str, seconds: int, daily_entry: dict) -> tuple[bool, str]:
    shutil.rmtree(KDIR, ignore_errors=True); KDIR.mkdir(parents=True)
    (KDIR/'worker.py').write_text(
        worker_code(seconds, prompt=daily_entry['prompt'], lyrics=daily_entry['lyrics'], bpm=daily_entry.get('bpm', 128)),
        encoding='utf-8'
    )
    kernel_title = kernel.rsplit('/', 1)[-1]
    (KDIR/'kernel-metadata.json').write_text(
        json.dumps({
            'id': kernel,
            'title': kernel_title,
            'code_file': 'worker.py',
            'language': 'python',
            'kernel_type': 'script',
            'is_private': True,
            'enable_gpu': True,
            'enable_internet': True,
            'machine_shape': 'NvidiaTeslaT4',
            'dataset_sources': [],
            'competition_sources': [],
            'kernel_sources': [],
            'model_sources': []
        }, indent=2),
        encoding='utf-8'
    )
    push = subprocess.run(['kaggle', 'kernels', 'push', '-p', str(KDIR), '--accelerator', 'NvidiaTeslaT4', '--timeout', '1800'], cwd=ROOT, env=env, capture_output=True, text=True)
    push_text = (push.stdout or '') + '\n' + (push.stderr or '')
    print('KAGGLE_PUSH_OUTPUT:', push_text.strip(), flush=True)
    low = push_text.lower()
    if 'maximum weekly gpu quota' in low or ('gpu quota' in low and 'reached' in low):
        raise RuntimeError('KAGGLE_GPU_QUOTA_EXHAUSTED: Kaggle refused GPU allocation.')
    if push.returncode != 0:
        raise RuntimeError(f'KAGGLE_KERNEL_PUSH_FAILED: {push_text.strip()}')

    deadline = time.time() + 35 * 60
    queued_since = None
    MAX_QUEUE_WAIT = 15 * 60
    while time.time() < deadline:
        p = subprocess.run(['kaggle', 'kernels', 'status', kernel], capture_output=True, text=True, env=env)
        status_text = (p.stdout or p.stderr).strip()
        print(f'{kernel} STATUS: {status_text}', flush=True)
        t = status_text.lower()
        if 'complete' in t and not any(x in t for x in ('incomplete', 'not complete')):
            return True, status_text
        if 'queued' in t:
            if queued_since is None:
                queued_since = time.time()
            elif time.time() - queued_since > MAX_QUEUE_WAIT:
                return False, f'KAGGLE_GPU_QUEUE_TIMEOUT: {kernel} remained in QUEUED for >15 min.'
        else:
            queued_since = None
        if any(x in t for x in ('error', 'failed', 'cancelled', 'canceled')):
            lp = subprocess.run(['kaggle', 'kernels', 'logs', kernel], capture_output=True, text=True, env=env)
            log_text = lp.stdout or lp.stderr or ''
            return False, f'KAGGLE_KERNEL_ERROR: {status_text}\n{log_text}'
        time.sleep(30)
    return False, f'KAGGLE_KERNEL_TIMEOUT: {kernel}'


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=int, default=180)
    dispatch(ap.parse_args().seconds)
