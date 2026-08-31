from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "wan2_worker"
WAN = WORK / "Wan2.1"
MODEL = WORK / "Wan2.1-T2V-1.3B"
REQUEST = ROOT / "wan_request.json"
OUT = WORK / "scenes"
WAN_REPO = "https://github.com/Wan-Video/Wan2.1.git"


def run(*args, cwd=None):
    print("WAN_RUN:", " ".join(map(str, args)), flush=True)
    return subprocess.run([str(x) for x in args], cwd=cwd, check=True, text=True)


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    run("nvidia-smi")
    if not WAN.exists():
        run("git", "clone", "--no-tags", "--depth", "1", WAN_REPO, WAN)
    if not (WORK / ".deps_ok").exists():
        run(sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "ftfy", "imageio", "imageio-ffmpeg")
        run(sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", cwd=WAN)
        (WORK / ".deps_ok").write_text("ok", encoding="utf-8")

    marker = MODEL / ".download_complete"
    if not marker.exists():
        from huggingface_hub import snapshot_download
        print("WAN_MODEL_DOWNLOAD: downloading Wan-AI/Wan2.1-T2V-1.3B", flush=True)
        snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=str(MODEL))
        marker.write_text("ok", encoding="utf-8")
        print("WAN_MODEL_DOWNLOAD_OK", flush=True)
    else:
        print("WAN_MODEL_CACHE_OK", flush=True)


def generate_scene(index: int, prompt: str, seed: int):
    raw = OUT / f"scene_{index}_raw.mp4"
    final = OUT / f"scene_{index}.mp4"
    if final.exists() and final.stat().st_size > 50_000:
        print(f"WAN_SCENE_CACHED scene={index}", flush=True)
        return
    raw.unlink(missing_ok=True)
    final.unlink(missing_ok=True)

    cmd = [
        sys.executable, "generate.py",
        "--task", "t2v-1.3B",
        "--size", "480*832",
        "--ckpt_dir", str(MODEL),
        "--offload_model", "True",
        "--t5_cpu",
        "--sample_shift", "8",
        "--sample_guide_scale", "6",
        "--sample_steps", os.getenv("WAN_SAMPLE_STEPS", "30"),
        "--frame_num", "81",
        "--base_seed", str(seed),
        "--save_file", str(raw),
        "--prompt", prompt,
    ]
    print(f"WAN_SCENE_START scene={index} seed={seed}", flush=True)
    run(*cmd, cwd=WAN)
    if not raw.exists() or raw.stat().st_size < 50_000:
        raise RuntimeError(f"WAN_FATAL: scene {index} raw MP4 missing/small")

    run(
        "ffmpeg", "-y", "-i", raw,
        "-vf", "setpts=3.0*PTS,fps=16,scale=480:832:flags=lanczos",
        "-an", "-t", "15",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", final,
    )
    if not final.exists() or final.stat().st_size < 50_000:
        raise RuntimeError(f"WAN_FATAL: scene {index} final MP4 missing/small")
    raw.unlink(missing_ok=True)
    print(f"WAN_SCENE_OK scene={index} path={final} bytes={final.stat().st_size}", flush=True)


def main():
    prepare()
    req = json.loads(REQUEST.read_text(encoding="utf-8"))
    for item in req["scenes"]:
        generate_scene(int(item["index"]), item["prompt"], int(item["seed"]))
    manifest = {
        "backend": "Wan2.1-T2V-1.3B",
        "license": "Apache-2.0",
        "resolution": "480x832",
        "source_seconds": 5,
        "output_seconds": 15,
        "scenes": req["scenes"],
    }
    (WORK / "wan_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WAN_ALL_SCENES_OK", flush=True)


if __name__ == "__main__":
    main()
