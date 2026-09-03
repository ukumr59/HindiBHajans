"""HindiBHajans production orchestrator.

Control plane only: no paid compute, no local-machine dependency, provider-neutral
GPU dispatch, persistent job state, and hard identity/input gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from app.provider_router import run_with_failover

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "state"
OUTPUT = ROOT / "output"
REFERENCE = ROOT / "assets" / "uks model image.png"


def save_state(job_id: str, **updates: object) -> dict:
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "current.json"
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.update({"job_id": job_id, "updated_at": datetime.now(timezone.utc).isoformat(), **updates})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_reference() -> None:
    if not REFERENCE.exists() or REFERENCE.stat().st_size == 0:
        raise RuntimeError("IDENTITY_GATE_FAILED: approved singer image is missing")
    from PIL import Image
    with Image.open(REFERENCE) as im:
        if im.width < 500 or im.height < 500:
            raise RuntimeError(f"IDENTITY_GATE_FAILED: reference too small: {im.size}")
        print(f"IDENTITY_SOURCE={REFERENCE}")
        print(f"IDENTITY_SOURCE_SHA256={sha256(REFERENCE)}")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "HindiBHajans/1.0"})
    with urllib.request.urlopen(req, timeout=600) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def ffprobe_duration(path: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True).strip())


def validate_master(path: Path, requested_seconds: int) -> None:
    if not path.exists() or path.stat().st_size < 500_000:
        raise RuntimeError("MASTER_GATE_FAILED: generated MP4 is missing or too small")
    streams = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", str(path)
    ], text=True).splitlines()
    if "video" not in streams or "audio" not in streams:
        raise RuntimeError("MASTER_GATE_FAILED: MP4 must contain both video and audio")
    duration = ffprobe_duration(path)
    if duration < requested_seconds * 0.95:
        raise RuntimeError(f"MASTER_GATE_FAILED: duration {duration:.2f}s < requested {requested_seconds}s")
    print(f"MASTER_OK duration={duration:.2f}s size={path.stat().st_size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily", "health"], default="daily")
    ap.add_argument("--seconds", type=int, default=180)
    args = ap.parse_args()

    if args.mode == "health":
        print("HINDIBHAJANS_PROVIDER_HEALTH")
        for key in ("BH_PROVIDER_ORDER", "BH_MODAL_ENDPOINT", "BH_HF_SPACE_ENDPOINT", "BH_BEAM_ENDPOINT", "BH_KAGGLE_ENDPOINT"):
            print(f"{key}={'SET' if os.getenv(key) else 'NOT_SET'}")
        return

    if args.seconds < 180 or args.seconds > 300 or args.seconds % 15:
        raise SystemExit("VIDEO_SECONDS must be 180-300 and divisible by 15")

    job_id = datetime.now().strftime("HB-%Y-%m-%d")
    validate_reference()
    save_state(job_id, status="CREATED", identity_source=str(REFERENCE.relative_to(ROOT)))

    payload = {
        "job_id": job_id,
        "duration_seconds": args.seconds,
        "identity_source": "assets/uks model image.png",
        "identity_policy": "LOCKED_REFERENCE_NO_FACE_REGENERATION",
        "performance": "SINGER_ACTUALLY_SINGING",
        "wardrobe": "TRADITIONAL_INDIAN",
        "setting": "DEVOTIONAL_TEMPLE_WITH_SPECIFIED_DEITY",
        "audio": {"language": "hi", "type": "bhajan"},
        "output": {"format": "mp4", "aspect": "16:9", "video_audio": True},
    }
    result = run_with_failover(payload)
    save_state(job_id, status="GPU_SUBMITTED", provider=result.get("provider"), provider_job_id=result.get("job_id"))

    video_url = result.get("video_url")
    if not video_url:
        raise RuntimeError("GPU provider returned no video_url; async polling adapter is required before production")
    out = OUTPUT / "master.mp4"
    download(str(video_url), out)
    validate_master(out, args.seconds)
    save_state(job_id, status="MASTER_READY", master=str(out.relative_to(ROOT)))
    print("BHAJAN_MASTER_READY")


if __name__ == "__main__":
    main()
