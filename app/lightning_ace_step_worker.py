from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STUDIO_ROOT = Path(__file__).resolve().parent
WORK = STUDIO_ROOT / "bhajan_aabha_worker"
INPUT = STUDIO_ROOT / "bhajan_request.json"
OUTPUT = WORK / "bhajan_source.mp3"
ACE = WORK / "ACE-Step-1.5"

ACE_COMMIT = "7202bc354d7fc31d1c0e5a90b0b49fb610e52362"
ACE_REPO = "https://github.com/ACE-Step/ACE-Step-1.5.git"


def run(*args, cwd=None):
    print("LIGHTNING_RUN:", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=True, text=True)


def pin_ace_checkout() -> None:
    if not ACE.exists():
        run("git", "clone", "--no-tags", "--depth", "1", ACE_REPO, str(ACE))
    else:
        run("git", "fetch", "--depth", "1", "origin", ACE_COMMIT, cwd=ACE)
    run("git", "reset", "--hard", ACE_COMMIT, cwd=ACE)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ACE, text=True).strip()
    print(f"LIGHTNING_ACE_COMMIT: {actual}", flush=True)
    if actual != ACE_COMMIT:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: ACE-Step checkout mismatch: {actual} != {ACE_COMMIT}")


def patch_ace_t4_precision() -> None:
    """Patch the exact upstream CUDA dtype branch; T4 must use FP32."""
    target = ACE / "acestep/core/generation/handler/init_service_orchestrator.py"
    if not target.exists():
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: missing ACE-Step orchestrator: {target}")

    text = target.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(?ms)^(\s*)elif resolved_device == "cuda":\n.*?(?=^\s*else:\n\s*self\.dtype = torch\.bfloat16)'
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError("LIGHTNING_ACE_FATAL: could not locate ACE-Step CUDA dtype selection block")

    indent = match.group(1)
    replacement = (
        f'{indent}elif resolved_device == "cuda":\n'
        f'{indent}    # T4/Turing is pre-Ampere. Force FP32 when requested.\n'
        f'{indent}    env_dtype_str = os.environ.get("ACESTEP_DTYPE", "").strip().lower()\n'
        f'{indent}    if env_dtype_str == "float32":\n'
        f'{indent}        self.dtype = torch.float32\n'
        f'{indent}        logger.info("[initialize_service] ACESTEP_DTYPE=float32 override applied.")\n'
        f'{indent}    elif env_dtype_str == "float16":\n'
        f'{indent}        self.dtype = torch.float16\n'
        f'{indent}    elif env_dtype_str == "bfloat16":\n'
        f'{indent}        self.dtype = torch.bfloat16\n'
        f'{indent}    elif gpu_config.cuda_supports_bfloat16():\n'
        f'{indent}        self.dtype = torch.bfloat16\n'
        f'{indent}    else:\n'
        f'{indent}        self.dtype = torch.float16\n'
        f'{indent}        logger.info("[initialize_service] Pre-Ampere CUDA detected: using float16 instead of bfloat16.")\n'
    )
    target.write_text(text[:match.start()] + replacement + text[match.end():], encoding="utf-8")
    print("LIGHTNING_ACE_PATCH_OK: CUDA dtype branch patched for T4", flush=True)


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    print(f"LIGHTNING_STUDIO_ROOT: {STUDIO_ROOT}", flush=True)
    print(f"LIGHTNING_REQUEST: {INPUT} exists={INPUT.exists()}", flush=True)
    run("nvidia-smi")
    pin_ace_checkout()
    patch_ace_t4_precision()

    print("LIGHTNING_DEPS: installing uv", flush=True)
    run(sys.executable, "-m", "pip", "install", "-q", "-U", "uv")
    print("LIGHTNING_DEPS: installing pinned ACE-Step checkout", flush=True)
    run(sys.executable, "-m", "uv", "pip", "install", "--system", "-e", ".", cwd=ACE)

    if shutil.which("ffmpeg") is None:
        print("LIGHTNING_DEPS: ffmpeg not found; installing system ffmpeg", flush=True)
        run("sudo", "apt-get", "update", "-qq")
        run("sudo", "apt-get", "install", "-y", "-qq", "ffmpeg")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("LIGHTNING_ACE_FATAL: ffmpeg is unavailable after installation attempt")
    run("ffmpeg", "-version")
    run(sys.executable, "-c", "import torch; print('LIGHTNING_TORCH_OK:', torch.__version__, 'CUDA=', torch.version.cuda, 'AVAILABLE=', torch.cuda.is_available())")
    print("LIGHTNING_LLM_BACKEND: pt", flush=True)


def generate(req: dict) -> Path:
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
        raise RuntimeError("LIGHTNING_ACE_FATAL: CUDA GPU is not available")
    capability = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"LIGHTNING_ACE_GPU: {gpu_name} VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB CC={capability[0]}.{capability[1]}", flush=True)
    print("LIGHTNING_ACE_DTYPE_REQUESTED: float32", flush=True)

    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(ACE),
        config_path="acestep-v15-turbo",
        device="cuda",
        offload_to_cpu=True,
    )
    print(f"LIGHTNING_ACE_INIT: ok={ok} status={status}", flush=True)
    if not ok:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: DiT initialization failed: {status}")

    model_dtype = None
    model = getattr(dit, "model", None)
    if model is not None:
        for parameter in model.parameters():
            if parameter.is_floating_point():
                model_dtype = parameter.dtype
                break
    handler_dtype = getattr(dit, "dtype", None)
    print(f"LIGHTNING_ACE_ACTUAL_HANDLER_DTYPE: {handler_dtype}", flush=True)
    print(f"LIGHTNING_ACE_ACTUAL_MODEL_DTYPE: {model_dtype}", flush=True)
    if capability[0] < 8 and model_dtype != torch.float32:
        raise RuntimeError(
            f"LIGHTNING_ACE_FATAL: T4 dtype guard failed; loaded DiT dtype={model_dtype}, expected torch.float32"
        )

    llm = LLMHandler()
    llm_status, llm_ok = llm.initialize(
        checkpoint_dir=str(ACE / "checkpoints"),
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend="pt",
        device="cuda",
        dtype=torch.float32,
    )
    print(f"LIGHTNING_ACE_LLM_INIT: ok={llm_ok} status={llm_status}", flush=True)
    if not llm_ok:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: 5Hz LM initialization failed: {llm_status}")

    duration = float(req.get("duration", 180))
    candidate_params = {
        "task_type": "text2music",
        "caption": req["caption"],
        "lyrics": req["lyrics"],
        "bpm": int(req.get("bpm", 128)),
        "keyscale": req.get("keyscale", "C Major"),
        "timesignature": req.get("timesignature", "4/4"),
        "vocal_language": req.get("vocal_language", "hi"),
        "duration": duration,
        "thinking": False,
        "use_cot_metas": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "use_constrained_decoding": True,
        "inference_steps": 8,
        "guidance_scale": 1.0,
        "seed": -1,
    }
    supported = set(inspect.signature(GenerationParams).parameters)
    params_kwargs = {key: value for key, value in candidate_params.items() if key in supported}
    rejected = sorted(set(candidate_params) - supported)
    print(f"LIGHTNING_ACE_GENERATION_PARAMS_SUPPORTED: {sorted(params_kwargs)}", flush=True)
    if rejected:
        print(f"LIGHTNING_ACE_GENERATION_PARAMS_SKIPPED: {rejected}", flush=True)
    if "dtype" in supported:
        raise RuntimeError("LIGHTNING_ACE_FATAL: unexpected dtype field detected in GenerationParams; worker refuses API mismatch")

    params = GenerationParams(**params_kwargs)
    config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="wav")
    save_dir = WORK / "ace_output"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"LIGHTNING_ACE_STEP: generating {duration:.0f}s full-length Hindi bhajan", flush=True)
    started = time.time()
    result = generate_music(
        dit_handler=dit,
        llm_handler=llm,
        params=params,
        config=config,
        save_dir=str(save_dir),
    )
    if not result.success or not result.audios:
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: {result.error or result.status_message}")
    source = Path(result.audios[0].get("path", ""))
    if not source.exists():
        raise RuntimeError(f"LIGHTNING_ACE_FATAL: generated audio path missing: {source}")
    print(f"LIGHTNING_ACE_STEP: generation completed in {time.time() - started:.1f}s", flush=True)
    run(
        "ffmpeg", "-y", "-i", str(source),
        "-af", "loudnorm=I=-9:TP=-1.0:LRA=7",
        "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k",
        str(OUTPUT),
    )
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
