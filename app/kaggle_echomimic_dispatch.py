"""Dispatch EchoMimicV3-Flash to a free Kaggle GPU kernel.

This intentionally avoids Kaggle datasets/models APIs: the kernel is pushed with
only the small input files and downloads the open-source model weights directly
from Hugging Face. That bypasses the GetDataset 403 path that broke the old
kagglehub implementation.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
KAGGLE_DIR = ROOT / ".kaggle_worker"


def run(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("RUN:", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(args), cwd=str(cwd) if cwd else None, text=True, check=check)


def kaggle_env() -> dict[str, str]:
    env = dict(os.environ)
    token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_API_TOKEN3")
    if token:
        env["KAGGLE_API_TOKEN"] = token
    return env


def worker_code() -> str:
    return r'''#!/usr/bin/env python3
import os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path('/kaggle/working')
REPO = ROOT / 'echomimic_v3'
MODELS = ROOT / 'models'
INPUT = ROOT / 'input'
SEG = ROOT / 'segments'
OUT = ROOT / 'outputs'
IMAGE = INPUT / 'singer.png'
AUDIO = INPUT / 'bhajan.mp3'
SECONDS = int(os.environ.get('BH_VIDEO_SECONDS', '180'))
FPS = 25
FRAMES = 81                 # ~3.24 s; safe on 16 GB T4
SEG_SECONDS = FRAMES / FPS


def run(*args, cwd=None):
    print('RUN:', ' '.join(map(str,args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None, check=True)


def first_mp4(folder):
    xs = sorted(Path(folder).rglob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not xs:
        raise RuntimeError(f'No MP4 generated under {folder}')
    return xs[0]


def main():
    print('BHAJAN_KAGGLE_WORKER_START', flush=True)
    run('nvidia-smi')
    import torch
    print('TORCH=', torch.__version__, 'CUDA=', torch.version.cuda, flush=True)
    print('GPU_COUNT=', torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('NO_GPU_ALLOCATED')
    if torch.cuda.get_device_properties(0).total_memory < 13_000_000_000:
        raise RuntimeError('GPU_VRAM_TOO_SMALL_FOR_ECHOMIMICV3_FLASH')

    run('git', 'clone', '--depth', '1', 'https://github.com/antgroup/echomimic_v3.git', str(REPO))
    run(sys.executable, '-m', 'pip', 'install', '-q', '-r', str(REPO/'requirements.txt'))
    run(sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub')

    from huggingface_hub import snapshot_download
    MODELS.mkdir(exist_ok=True)
    base = MODELS / 'Wan2.1-Fun-V1.1-1.3B-InP'
    wav = MODELS / 'chinese-wav2vec2-base'
    flash = MODELS / 'echomimicv3-flash-pro'
    if not base.exists():
        snapshot_download('alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP', local_dir=str(base))
    if not wav.exists():
        snapshot_download('TencentGameMate/chinese-wav2vec2-base', local_dir=str(wav))
    if not flash.exists():
        snapshot_download('BadToBest/EchoMimicV3', local_dir=str(flash), allow_patterns=['echomimicv3-flash-pro/*'])

    SEG.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
    # Normalize audio once; every generated segment is later muxed against the
    # original continuous track, avoiding cumulative audio drift.
    norm = INPUT / 'audio16k.wav'
    run('ffmpeg','-y','-v','error','-i',str(AUDIO),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(norm))

    n = int((SECONDS + SEG_SECONDS - 1) // SEG_SECONDS)
    print(f'SEGMENTS={n} SEG_SECONDS={SEG_SECONDS:.3f}', flush=True)
    for i in range(n):
        start = i * SEG_SECONDS
        remain = max(0.1, min(SEG_SECONDS, SECONDS - start))
        if remain < 0.5: break
        a = SEG / f'audio_{i:04d}.wav'
        run('ffmpeg','-y','-v','error','-ss',f'{start:.3f}','-i',str(norm),'-t',f'{remain:.3f}','-ar','16000','-ac','1',str(a))
        od = SEG / f'raw_{i:04d}'
        od.mkdir(exist_ok=True)
        # Start from the approved singer reference for every segment. This is
        # deliberate: identity is anchored to the same uploaded pixels and no
        # face regeneration is allowed. Long-video stitching is handled by the
        # controller after generation.
        run(sys.executable, str(REPO/'infer_flash.py'),
            '--image_path',str(IMAGE),'--audio_path',str(a),
            '--prompt','A single Indian devotional singer performing a Hindi bhajan in traditional Indian clothing before the specified Hindu deity in a serene temple setting; only the same singer is visible; natural singing mouth movement, subtle expressive head and upper-body motion, stable identity.',
            '--num_inference_steps','8','--config_path',str(REPO/'config/config.yaml'),
            '--model_name',str(base),'--ckpt_idx','50000',
            '--transformer_path',str(flash/'echomimicv3-flash-pro/transformer/diffusion_pytorch_model.safetensors'),
            '--save_path',str(od),'--wav2vec_model_dir',str(wav),
            '--sampler_name','Flow_Unipc','--video_length',str(FRAMES),
            '--guidance_scale','5.0','--audio_guidance_scale','2.5','--audio_scale','1.0',
            '--neg_scale','1.0','--neg_steps','0','--seed',str(4300+i),
            '--enable_teacache','--teacache_threshold','0.1','--num_skip_start_steps','5',
            '--weight_dtype','float16','--sample_size','768','768','--fps',str(FPS),
            '--negative_prompt','blurry, distorted face, identity drift, extra person, duplicate person, malformed hands, fused fingers, deformed mouth, jitter, flicker, camera cut, text, watermark')
        raw = first_mp4(od)
        silent = SEG / f'video_{i:04d}.mp4'
        run('ffmpeg','-y','-v','error','-i',str(raw),'-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(silent))

    concat = SEG / 'concat.txt'
    files = sorted(SEG.glob('video_*.mp4'))
    if not files: raise RuntimeError('NO_SEGMENTS_GENERATED')
    concat.write_text(''.join(f"file '{p.resolve()}'\n" for p in files))
    visual = OUT / 'visual.mp4'
    run('ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy',str(visual))
    final = OUT / 'master.mp4'
    run('ffmpeg','-y','-v','error','-i',str(visual),'-i',str(AUDIO),'-map','0:v:0','-map','1:a:0','-t',str(SECONDS),'-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',str(final))
    if not final.exists() or final.stat().st_size < 500_000:
        raise RuntimeError('MASTER_NOT_CREATED')
    print('BHAJAN_KAGGLE_WORKER_OK', flush=True)
    print('OUTPUT=', final, flush=True)

if __name__ == '__main__': main()
'''


def dispatch(seconds: int) -> None:
    token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_API_TOKEN3")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN secret is required. It must have Kaggle kernel write/run permission.")

    image = ROOT / "assets" / "uks model image.png"
    audio = OUT / "bhajan_source.mp3"
    if not image.exists(): raise RuntimeError(f"Missing singer image: {image}")
    if not audio.exists(): raise RuntimeError(f"Missing generated Hindi bhajan audio: {audio}")

    shutil.rmtree(KAGGLE_DIR, ignore_errors=True)
    KAGGLE_DIR.mkdir(parents=True)
    shutil.copy2(image, KAGGLE_DIR / "singer.png")
    shutil.copy2(audio, KAGGLE_DIR / "bhajan.mp3")
    (KAGGLE_DIR / "worker.py").write_text(worker_code(), encoding="utf-8")
    username = os.getenv("KAGGLE_USERNAME", "")
    if not username:
        # The current CLI can derive the account from the token; metadata still
        # needs an id, so allow the caller to supply it explicitly.
        raise RuntimeError("KAGGLE_USERNAME repository variable/secret is required for kernel dispatch")
    meta = {
        "id": f"{username}/hindibhajans-echomimic-v3",
        "title": "HindiBHajans EchoMimic V3 Daily Worker",
        "code_file": "worker.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (KAGGLE_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    env = kaggle_env()
    env["BH_VIDEO_SECONDS"] = str(seconds)
    run("python", "-m", "pip", "install", "-q", "--upgrade", "kaggle", env=env) if False else None
    run("kaggle", "kernels", "push", "-p", str(KAGGLE_DIR), "--accelerator", "NvidiaTeslaT4", "--timeout", str(11*60*60), cwd=ROOT)
    kernel = meta["id"]
    deadline = time.time() + 11*60*60
    while time.time() < deadline:
        p = subprocess.run(["kaggle","kernels","status",kernel], text=True, capture_output=True, env=env)
        print(p.stdout or p.stderr, flush=True)
        text=(p.stdout+p.stderr).lower()
        if any(x in text for x in ("complete", "error", "failed")):
            if "complete" in text: break
            raise RuntimeError("KAGGLE_KERNEL_FAILED: " + (p.stdout or p.stderr))
        time.sleep(30)
    else:
        raise TimeoutError("KAGGLE_KERNEL_TIMEOUT")

    shutil.rmtree(OUT / "kaggle_output", ignore_errors=True)
    run("kaggle", "kernels", "output", kernel, "-p", str(OUT / "kaggle_output"), "--force", cwd=ROOT)
    candidate = OUT / "kaggle_output" / "master.mp4"
    if not candidate.exists():
        xs = list((OUT / "kaggle_output").rglob("master.mp4"))
        if xs: candidate=xs[0]
    if not candidate.exists():
        raise RuntimeError("KAGGLE_COMPLETED_BUT_MASTER_MP4_MISSING")
    shutil.copy2(candidate, OUT / "master.mp4")
    print("KAGGLE_ECHOMIMIC_MASTER_READY", flush=True)


if __name__ == "__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--seconds',type=int,default=180)
    a=ap.parse_args(); dispatch(a.seconds)
