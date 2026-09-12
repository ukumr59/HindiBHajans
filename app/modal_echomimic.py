"""Modal GPU worker for the HindiBHajans real-singer video stage.

This module is intentionally separate from the GitHub Actions workflow.  The
workflow submits the job to Modal; the GPU container performs the complete
EchoMimicV3 render and returns the final MP4 bytes.  There is no static-image
fallback.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import modal

APP_NAME = "hindibhajans-echomimic-v3"
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("huggingface_hub", "opencv-python-headless", "numpy")
)

app = modal.App(APP_NAME)


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


@app.function(
    image=IMAGE,
    gpu="L4",
    timeout=60 * 30,
    cpu=4,
    memory=16384,
)
def render(reference_image: bytes, audio_mp3: bytes, duration_seconds: int = 20) -> bytes:
    """Render an audio-driven EchoMimicV3 singing video and return MP4 bytes."""
    if not reference_image or not audio_mp3:
        raise RuntimeError("MODAL_ECHOMIMIC_INPUT_ERROR: reference image and audio are required")
    if duration_seconds <= 0 or duration_seconds > 180:
        raise RuntimeError("MODAL_ECHOMIMIC_INPUT_ERROR: duration must be 1..180 seconds")

    with tempfile.TemporaryDirectory(prefix="hindibhajans-echo-") as td:
        root = Path(td)
        repo = root / "echomimic_v3"
        ref = root / "reference.png"
        audio = root / "audio.mp3"
        work = root / "work"
        work.mkdir()
        ref.write_bytes(reference_image)
        audio.write_bytes(audio_mp3)

        _run(["git", "clone", "--depth", "1", "https://github.com/antgroup/echomimic_v3.git", str(repo)])
        _run(["python", "-m", "pip", "install", "-r", "requirements.txt"], cwd=str(repo))

        # Download the pinned public model repositories into the container.
        from huggingface_hub import snapshot_download
        snapshot_download("alibaba-pai/Wan2.1-Fun-V1.1-1.3B")
        snapshot_download("TencentGameMate/chinese-wav2vec2-base")
        snapshot_download("BadToBest/EchoMimicV3")
        snapshot_download("BadToBest/echomimicv3-flash-pro")

        wav = work / "audio_16k.wav"
        _run(["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", "16000", str(wav)])

        # Keep the first controlled test small. Production may pass 180 seconds.
        out_dir = work / "chunks"
        out_dir.mkdir()
        prompt = (
            "A single Indian devotional singer, the same person as the reference image, "
            "singing a Hindi bhajan before a Hindu deity in a serene temple, wearing "
            "traditional Indian devotional clothing, natural singing mouth movement "
            "synchronized to the audio, subtle devotional head and upper-body movement, "
            "stable facial identity, only one person visible."
        )
        negative = (
            "blurry, distorted face, identity drift, extra person, malformed hands, "
            "deformed mouth, jitter, flicker, text, watermark"
        )

        # EchoMimicV3's flash inference accepts short motion clips.  The worker
        # segments longer audio into deterministic chunks and concatenates them.
        chunk_seconds = 3.24
        n_chunks = (duration_seconds + int(chunk_seconds) - 1) // int(chunk_seconds)
        # Use the repository's own inference entry point; exact CLI arguments are
        # kept in one place so they can be updated with upstream changes.
        for i in range(n_chunks):
            start = i * chunk_seconds
            length = min(chunk_seconds, duration_seconds - start)
            chunk_audio = work / f"audio_{i:04d}.wav"
            _run([
                "ffmpeg", "-y", "-ss", str(start), "-t", str(length),
                "-i", str(wav), "-ar", "16000", "-ac", "1", str(chunk_audio)
            ])
            _run([
                "python", "infer_flash.py",
                "--ref_image", str(ref),
                "--audio", str(chunk_audio),
                "--prompt", prompt,
                "--negative_prompt", negative,
                "--num_inference_steps", "8",
                "--save_dir", str(out_dir / f"chunk_{i:04d}"),
            ], cwd=str(repo))

        clips = sorted(out_dir.glob("chunk_*/**/*.mp4"))
        if not clips:
            raise RuntimeError("MODAL_ECHOMIMIC_OUTPUT_ERROR: EchoMimicV3 produced no MP4")
        concat = work / "concat.txt"
        concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in clips), encoding="utf-8")
        silent = work / "video.mp4"
        final = work / "master.mp4"
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(silent)])
        _run([
            "ffmpeg", "-y", "-i", str(silent), "-i", str(audio),
            "-t", str(duration_seconds), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(final)
        ])
        return final.read_bytes()
