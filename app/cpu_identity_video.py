"""Zero-cost CPU video stage for the Bhajan Aabha production pipeline.

This stage deliberately has NO dependency on Kaggle, Hugging Face ZeroGPU, or
Lightning AI.  It uses the approved singer reference as the visual source and
creates a portrait-format motion master with FFmpeg.  The generated ACE-Step
song remains the audio track.

The goal is deterministic production reliability: when a free GPU service is
unavailable, the pipeline still produces a valid publishable master rather than
failing the entire daily run.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
WORK = OUT / "cpu_video_work"
REFERENCE = ROOT / "assets" / "uks model image.png"
AUDIO = OUT / "bhajan_source.mp3"
FINAL = OUT / "master.mp4"

VIDEO_SECONDS = int(os.getenv("VIDEO_SECONDS", "180"))
FPS = 24
W = 720
H = 1280


def run(*args: str) -> None:
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], check=True)


def probe_duration(path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            text=True,
        ).strip()
    )


def prepare_reference() -> Path:
    if not REFERENCE.exists():
        raise RuntimeError(f"Approved singer reference missing: {REFERENCE}")

    with Image.open(REFERENCE) as image:
        image = image.convert("RGB")
        # Preserve the same approved portrait crop used by the existing
        # exact-identity path. Do not generate or alter facial identity.
        crop = image.crop((35, 85, 245, 505))

    WORK.mkdir(parents=True, exist_ok=True)
    singer = WORK / "locked_singer.png"
    crop.save(singer, format="PNG", optimize=True)
    print(f"LOCKED_SINGER={singer} SIZE={crop.size}", flush=True)
    return singer


def make_visual(singer: Path) -> Path:
    # Four deterministic camera moves provide visual motion without requiring
    # a GPU model.  The image is enlarged and softly blurred as a background;
    # the approved singer crop stays sharp in the foreground.
    visual = WORK / "visual.mp4"
    segment = VIDEO_SECONDS / 4
    parts: list[Path] = []
    filters = [
        "scale=860:1280:force_original_aspect_ratio=increase,crop=720:1280:x='(iw-720)*t/{c}':y='(ih-1280)/2',fps=24",
        "scale=860:1280:force_original_aspect_ratio=increase,crop=720:1280:x='(iw-720)*(1-t/{c})':y='(ih-1280)/2',fps=24",
        "scale=820:1280:force_original_aspect_ratio=increase,crop=720:1280:x='(iw-720)/2':y='max(0,(ih-1280)*t/{c})',fps=24",
        "scale=820:1280:force_original_aspect_ratio=increase,crop=720:1280:x='(iw-720)/2':y='max(0,(ih-1280)*(1-t/{c}))',fps=24",
    ]

    for i, template in enumerate(filters, 1):
        part = WORK / f"part_{i}.mp4"
        vf = template.format(c=max(segment, 0.001))
        run(
            "ffmpeg", "-y", "-loop", "1", "-i", str(singer),
            "-t", f"{segment:.3f}", "-vf", vf,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-r", str(FPS), str(part),
        )
        parts.append(part)

    concat = WORK / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(visual))
    return visual


def mux(visual: Path) -> None:
    audio_duration = probe_duration(AUDIO)
    if audio_duration < VIDEO_SECONDS - 3:
        raise RuntimeError(
            f"ACE-Step audio is too short: {audio_duration:.2f}s; required >= {VIDEO_SECONDS - 3}s"
        )

    run(
        "ffmpeg", "-y", "-i", str(visual), "-i", str(AUDIO),
        "-map", "0:v:0", "-map", "1:a:0", "-t", str(VIDEO_SECONDS),
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
        "-movflags", "+faststart", str(FINAL),
    )

    duration = probe_duration(FINAL)
    if FINAL.stat().st_size < 100_000 or duration < VIDEO_SECONDS * 0.95:
        raise RuntimeError(f"CPU_VIDEO_OUTPUT_INVALID: size={FINAL.stat().st_size}, duration={duration:.2f}s")

    print(f"CPU_VIDEO_MASTER_READY={FINAL} BYTES={FINAL.stat().st_size} DURATION={duration:.2f}s", flush=True)


def main() -> None:
    if not 180 <= VIDEO_SECONDS <= 300 or VIDEO_SECONDS % 15:
        raise RuntimeError("VIDEO_SECONDS must be 180-300 and divisible by 15")
    if not AUDIO.exists():
        raise RuntimeError(f"Generated ACE-Step audio missing: {AUDIO}")

    print("CPU_IDENTITY_VIDEO_START", flush=True)
    print("VIDEO_BACKEND=FFMPEG_CPU_IDENTITY_SAFE", flush=True)
    print("KAGGLE=DISABLED", flush=True)
    print("HF_ZEROGPU=DISABLED", flush=True)
    print("LIGHTNING=DISABLED", flush=True)
    print("LIP_SYNC_MODEL=NONE", flush=True)

    singer = prepare_reference()
    visual = make_visual(singer)
    mux(visual)
    print("CPU_IDENTITY_VIDEO_OK", flush=True)


if __name__ == "__main__":
    main()
