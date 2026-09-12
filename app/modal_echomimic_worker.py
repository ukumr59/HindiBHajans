"""Modal GPU worker for the real-singer EchoMimicV3 pipeline.

This module is intentionally isolated from GitHub Actions. The first production
integration should call the `run_job` entrypoint with local image/audio paths.
No static-image fallback is permitted.
"""
from __future__ import annotations

import modal

app = modal.App("hindibhajans-echomimic-v3")

MODEL_VOLUME = modal.Volume.from_name("hindibhajans-model-cache", create_if_missing=True)
MODEL_DIR = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch", "torchvision", "huggingface_hub", "opencv-python-headless", "numpy")
)

@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={MODEL_DIR: MODEL_VOLUME},
    env={"HF_HOME": f"{MODEL_DIR}/hf"},
)
def run_job(image_bytes: bytes, audio_bytes: bytes, seconds: int = 20) -> bytes:
    """Run a short controlled EchoMimicV3 generation and return MP4 bytes."""
    if seconds < 5 or seconds > 30:
        raise ValueError("controlled test duration must be 5-30 seconds")
    import os, subprocess, tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp(prefix="bhajan-modal-"))
    ref = root / "singer.png"
    audio = root / "bhajan.mp3"
    out = root / "out.mp4"
    ref.write_bytes(image_bytes)
    audio.write_bytes(audio_bytes)
    if ref.stat().st_size == 0 or audio.stat().st_size == 0:
        raise RuntimeError("EMPTY_INPUT")

    repo = root / "echomimic_v3"
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/antgroup/echomimic_v3.git", str(repo)], check=True)
    subprocess.run(["python", "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-t", str(seconds), "-ac", "1", "-ar", "16000", str(root / "audio.wav")], check=True)

    # Model downloads are cached on the persistent Modal Volume. This is the
    # production model family already selected by the architecture.
    from huggingface_hub import snapshot_download
    base = Path(MODEL_DIR) / "Wan2.1-Fun-V1.1-1.3B-InP"
    wav = Path(MODEL_DIR) / "chinese-wav2vec2-base"
    flash = Path(MODEL_DIR) / "echomimicv3-flash-pro"
    snapshot_download("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", local_dir=str(base))
    snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir=str(wav))
    snapshot_download("BadToBest/EchoMimicV3", local_dir=str(flash), allow_patterns=["echomimicv3-flash-pro/*"])

    # Keep this entrypoint deliberately small and explicit; upstream infer_flash
    # owns the actual model invocation and is passed the uploaded identity image.
    raw = root / "raw"
    raw.mkdir()
    cmd = ["python", str(repo / "infer_flash.py"),
           "--image_path", str(ref), "--audio_path", str(root / "audio.wav"),
           "--prompt", "A single Indian devotional singer, the same person as the reference image, singing a Hindi bhajan before a Hindu deity in a serene temple, wearing traditional Indian devotional clothing, natural singing mouth movement synchronized to the audio, subtle devotional head and upper-body movement, stable facial identity, only one person visible.",
           "--num_inference_steps", "8", "--config_path", str(repo / "config/config.yaml"),
           "--model_name", str(base), "--ckpt_idx", "50000",
           "--transformer_path", str(flash / "echomimicv3-flash-pro/transformer/diffusion_pytorch_model.safetensors"),
           "--save_path", str(raw), "--wav2vec_model_dir", str(wav),
           "--sampler_name", "Flow_Unipc", "--video_length", "81",
           "--guidance_scale", "5.0", "--audio_guidance_scale", "2.5", "--audio_scale", "1.0",
           "--neg_scale", "1.0", "--neg_steps", "0", "--seed", "4300",
           "--enable_teacache", "--teacache_threshold", "0.1", "--num_skip_start_steps", "5",
           "--weight_dtype", "float16", "--sample_size", "768", "768", "--fps", "25",
           "--negative_prompt", "blurry, distorted face, identity drift, extra person, duplicate person, malformed hands, fused fingers, deformed mouth, jitter, flicker, camera cut, text, watermark"]
    subprocess.run(cmd, check=True)
    candidates = sorted(raw.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("ECHOMIMIC_RETURNED_NO_MP4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(candidates[0]), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-t", str(seconds), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)], check=True)
    return out.read_bytes()
