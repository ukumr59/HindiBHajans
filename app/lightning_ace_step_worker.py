from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Lightning Studio executes uploaded files from /teamspace/studios/this_studio.
STUDIO_ROOT = Path(__file__).resolve().parent
WORK = STUDIO_ROOT / "bhajan_aabha_worker"
INPUT = STUDIO_ROOT / "bhajan_request.json"
OUTPUT = WORK / "bhajan_source.mp3"
ACE = WORK / "ACE-Step-1.5"


def run(*args, cwd=None):
    print("LIGHTNING_RUN:", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=True, text=True)


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"LIGHTNING_STUDIO_ROOT: {STUDIO_ROOT}", flush=True)
    print(f"LIGHTNING_REQUEST: {INPUT} exists={INPUT.exists()}", flush=True)
    print("LIGHTNING_GPU:", flush=True)
    run("nvidia-smi")
    if not ACE.exists():
        run("git", "clone", "--depth", "1", "https://github.com/ACE-Step/ACE-Step-1.5.git", str(ACE))

    # ACE-Step 1.5 uses uv dependency sources. Plain pip ignores the
    # [tool.uv.sources] mappings, so use uv against the existing Studio Python.
    print("LIGHTNING_DEPS: installing uv", flush=True)
    run(sys.executable, "-m", "pip", "install", "-q", "-U", "uv")
    print("LIGHTNING_DEPS: resolving ACE-Step with uv tool.uv.sources", flush=True)
    run(sys.executable, "-m", "uv", "pip", "install", "--system", "-e", ".", cwd=ACE)

    # IMPORTANT: use shutil.which before invoking ffmpeg. subprocess.run()
    # raises FileNotFoundError when the executable is absent, so the previous
    # check never reached the installer on a fresh Lightning Studio.
    if shutil.which("ffmpeg") is None:
        print("LIGHTNING_DEPS: ffmpeg not found; installing system ffmpeg", flush=True)
        if shutil.which("sudo") is not None:
            run("sudo", "apt-get", "update", "-qq")
            run("sudo", "apt-get", "install", "-y", "-qq", "ffmpeg")
        else:
            run("apt-get", "update", "-qq")
            run("apt-get", "install", "-y", "-qq", "ffmpeg")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("LIGHTNING_ACE_FATAL: ffmpeg is unavailable after installation attempt")
    run("ffmpeg", "-version")

    run(sys.executable, "-c", "import torch; print('LIGHTNING_TORCH_OK:', torch.__version__, 'CUDA=', torch.version.cuda, 'AVAILABLE=', torch.cuda.is_available())")
    run(sys.executable, "-c", "import nano_vllm; print('LIGHTNING_NANOVLLM_OK')")


def generate(req: dict) -> Path:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACESTEP_SAVE_MEMORY", "1")
    sys.path.insert(0, str(ACE))
    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not torch.cuda.is_available():
        raise RuntimeError("LIGHTNING_ACE_FATAL: CUDA GPU is not available")
    print(f"LIGHTNING_ACE_GPU: {torch.cuda.get_device_name(0)} VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB", flush=True)

    dit = AceStepHandler()
    dit.initialize_service(project_root=str(ACE), config_path="acestep-v15-turbo", device="cuda", offload_to_cpu=True)
    llm = LLMHandler()
    llm.initialize(checkpoint_dir=str(ACE / "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B", backend="pt", device="cuda")

    duration = float(req.get("duration", 180))
    params = GenerationParams(
        task_type="text2music", caption=req["caption"], lyrics=req["lyrics"], bpm=int(req.get("bpm", 128)),
        keyscale=req.get("keyscale", "C Major"), timesignature=req.get("timesignature", "4/4"),
        vocal_language=req.get("vocal_language", "hi"), duration=duration, thinking=False,
        use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
        use_constrained_decoding=True, inference_steps=8, guidance_scale=1.0, seed=-1,
        shift=3.0, infer_method="ode", sampler_mode="euler", dcw_enabled=True,
        dcw_mode="double", dcw_scaler=0.05, dcw_high_scaler=0.02, dcw_wavelet="haar",
    )
    config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")
    save_dir = WORK / "ace_output"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"LIGHTNING_ACE_STEP: generating {duration:.0f}s full-length Hindi bhajan", flush=True)
    started = time.time()
    result = generate_music(dit_handler=dit, llm_handler=llm, params=params, config=config, save_dir=str(save_dir))
    if not result.success or not result.audios:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: {result.error or result.status_message}")
    source = Path(result.audios[0].get("path", ""))
    if not source.exists():
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: generated audio path missing: {source}")
    print(f"LIGHTNING_ACE_STEP: generation completed in {time.time() - started:.1f}s", flush=True)
    run("ffmpeg", "-y", "-i", str(source), "-af", "loudnorm=I=-9:TP=-1.0:LRA=7", "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", str(OUTPUT))
    if OUTPUT.stat().st_size < 100_000:
        raise RuntimeError("LIGHTNING_ACE_FATAL: output MP3 is suspiciously small")
    print(f"LIGHTNING_OUTPUT: {OUTPUT} {OUTPUT.stat().st_size} bytes", flush=True)
    return OUTPUT


def main():
    if not INPUT.exists():
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: bhajan_request.json not found at {INPUT}")
    req = json.loads(INPUT.read_text(encoding="utf-8"))
    prepare()
    out = generate(req)
    print("LIGHTNING_ACE_OK", out, out.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
