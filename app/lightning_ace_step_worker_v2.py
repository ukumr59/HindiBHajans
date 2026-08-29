from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "bhajan_aabha_worker"
INPUT = ROOT / "bhajan_request.json"
OUTPUT = WORK / "bhajan_source.mp3"
ACE = WORK / "ACE-Step-1.5"
ACE_COMMIT = "7202bc354d7fc31d1c0e5a90b0b49fb610e52362"
ACE_REPO = "https://github.com/ACE-Step/ACE-Step-1.5.git"


def run(*args, cwd=None):
    print("LIGHTNING_RUN:", " ".join(map(str, args)), flush=True)
    return subprocess.run([str(x) for x in args], cwd=cwd, check=True, text=True)


def prepare_ace():
    WORK.mkdir(parents=True, exist_ok=True)
    run("nvidia-smi")
    if not ACE.exists():
        run("git", "clone", "--no-tags", "--depth", "1", ACE_REPO, ACE)
    else:
        run("git", "fetch", "--depth", "1", "origin", ACE_COMMIT, cwd=ACE)
    run("git", "reset", "--hard", ACE_COMMIT, cwd=ACE)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ACE, text=True).strip()
    print("LIGHTNING_ACE_COMMIT:", actual, flush=True)
    if actual != ACE_COMMIT:
        raise RuntimeError("LIGHTNING_ACE_FATAL: ACE-Step checkout mismatch")

    target = ACE / "acestep/core/generation/handler/init_service_orchestrator.py"
    text = target.read_text(encoding="utf-8")
    start_marker = '            elif resolved_device == "cuda":\n'
    end_marker = '            else:\n                self.dtype = torch.bfloat16'
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise RuntimeError("LIGHTNING_ACE_FATAL: CUDA dtype block not found in pinned ACE-Step source")
    replacement = '''            elif resolved_device == "cuda":
                # T4/Turing: explicitly force FP32 for the DiT to avoid the
                # diffusion overflow seen with the upstream pre-Ampere path.
                self.dtype = torch.float32 if os.environ.get("ACESTEP_DTYPE", "").lower() == "float32" else torch.float16
                logger.info(f"[initialize_service] CUDA dtype override: {self.dtype}")
'''
    target.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    print("LIGHTNING_ACE_PATCH_OK: T4 CUDA dtype branch patched", flush=True)

    print("LIGHTNING_DEPS: installing pinned ACE-Step", flush=True)
    run(sys.executable, "-m", "pip", "install", "-q", "-U", "uv")
    run(sys.executable, "-m", "uv", "pip", "install", "--system", "-e", ".", cwd=ACE)

    if shutil.which("ffmpeg") is None:
        run("sudo", "apt-get", "update", "-qq")
        run("sudo", "apt-get", "install", "-y", "-qq", "ffmpeg")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("LIGHTNING_ACE_FATAL: ffmpeg unavailable")

    # The 0.6B LM is an OPTIONAL sub-model; the main ACE-Step download does
    # NOT contain it. Download it explicitly only when it is missing.
    sys.path.insert(0, str(ACE))
    from acestep.model_downloader import check_model_exists, download_submodel
    checkpoints = ACE / "checkpoints"
    if not check_model_exists("acestep-5Hz-lm-0.6B", checkpoints):
        print("LIGHTNING_LM_DOWNLOAD: downloading missing 0.6B LM", flush=True)
        ok, msg = download_submodel("acestep-5Hz-lm-0.6B", checkpoints_dir=checkpoints)
        print("LIGHTNING_LM_DOWNLOAD_RESULT:", ok, msg, flush=True)
        if not ok:
            raise RuntimeError("LIGHTNING_ACE_FATAL: could not download required 0.6B LM")
    else:
        print("LIGHTNING_LM_OK: 0.6B LM already present", flush=True)


def generate(req):
    os.environ["ACESTEP_DTYPE"] = "float32"
    os.environ["ACESTEP_LLM_BACKEND"] = "pt"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("ACESTEP_SAVE_MEMORY", "1")
    sys.path.insert(0, str(ACE))

    import torch
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    if not torch.cuda.is_available():
        raise RuntimeError("LIGHTNING_ACE_FATAL: CUDA GPU unavailable")
    capability = torch.cuda.get_device_capability(0)
    print(f"LIGHTNING_ACE_GPU: {torch.cuda.get_device_name(0)} CC={capability[0]}.{capability[1]}", flush=True)

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(ACE), config_path="acestep-v15-turbo", device="cuda", offload_to_cpu=True
    )
    print(f"LIGHTNING_ACE_INIT: ok={ok} status={status}", flush=True)
    if not ok:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: DiT initialization failed: {status}")

    model_dtype = None
    if getattr(dit, "model", None) is not None:
        for p in dit.model.parameters():
            if p.is_floating_point():
                model_dtype = p.dtype
                break
    print("LIGHTNING_ACE_ACTUAL_HANDLER_DTYPE:", getattr(dit, "dtype", None), flush=True)
    print("LIGHTNING_ACE_ACTUAL_MODEL_DTYPE:", model_dtype, flush=True)
    if capability[0] < 8 and model_dtype != torch.float32:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: T4 DiT dtype is {model_dtype}, expected torch.float32")

    llm = LLMHandler()
    llm_status, llm_ok = llm.initialize(
        checkpoint_dir=str(ACE / "checkpoints"),
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend="pt",
        device="cuda",
        offload_to_cpu=True,
        dtype=torch.float16,
    )
    print(f"LIGHTNING_ACE_LLM_INIT: ok={llm_ok} status={llm_status}", flush=True)
    if not llm_ok:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: 0.6B LM initialization failed: {llm_status}")

    duration = float(req.get("duration", 180))
    candidates = {
        "task_type": "text2music", "caption": req["caption"], "lyrics": req["lyrics"],
        "bpm": int(req.get("bpm", 128)), "keyscale": req.get("keyscale", "C Major"),
        "timesignature": req.get("timesignature", "4/4"), "vocal_language": req.get("vocal_language", "hi"),
        "duration": duration, "thinking": False, "use_cot_metas": False,
        "use_cot_caption": False, "use_cot_language": False, "use_constrained_decoding": True,
        "inference_steps": 8, "guidance_scale": 1.0, "seed": -1,
    }
    supported = set(inspect.signature(GenerationParams).parameters)
    kwargs = {k: v for k, v in candidates.items() if k in supported}
    print("LIGHTNING_ACE_GENERATION_PARAMS_SUPPORTED:", sorted(kwargs), flush=True)
    params = GenerationParams(**kwargs)
    config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")
    save_dir = WORK / "ace_output"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"LIGHTNING_ACE_STEP: generating {duration:.0f}s full-length Hindi bhajan", flush=True)
    started = time.time()
    result = generate_music(dit, llm, params=params, config=config, save_dir=str(save_dir))
    if not result.success or not result.audios:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: {result.error or result.status_message}")
    source = Path(result.audios[0].get("path", ""))
    if not source.exists():
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: generated audio missing: {source}")
    print(f"LIGHTNING_ACE_STEP: generation completed in {time.time()-started:.1f}s", flush=True)
    run("ffmpeg", "-y", "-i", source, "-af", "loudnorm=I=-9:TP=-1.0:LRA=7", "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", OUTPUT)
    if OUTPUT.stat().st_size < 100_000:
        raise RuntimeError("LIGHTNING_ACE_FATAL: output MP3 suspiciously small")
    print(f"LIGHTNING_OUTPUT: {OUTPUT} {OUTPUT.stat().st_size} bytes", flush=True)


def main():
    if not INPUT.exists():
        raise RuntimeError("LIGHTNING_ACE_FATAL: bhajan_request.json missing")
    req = json.loads(INPUT.read_text(encoding="utf-8"))
    prepare_ace()
    generate(req)
    print("LIGHTNING_ACE_OK", OUTPUT, OUTPUT.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
