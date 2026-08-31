from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
LEDGER = Path("data/pexels_used.json")
CREDITS = OUT / "pexels_credits.txt"
PEXELS_ROOT = "https://api.pexels.com/v1/videos/search"
API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
WIDTH, HEIGHT = 720, 1280
FPS = 24

QUERY_POOLS = [
    ["Indian temple", "temple diya", "Hindu temple bells", "temple flowers"],
    ["oil lamp diya", "incense smoke", "marigold flowers", "puja thali"],
    ["sunrise India", "golden sunrise mountains", "sun rays nature", "morning sky"],
    ["river sunrise India", "flowing river nature", "water reflection sunrise", "sacred river"],
    ["Indian heritage", "Indian architecture", "ancient temple architecture", "heritage India"],
    ["flowers close up", "orange marigold", "flower petals slow motion", "lotus flower"],
    ["forest sun rays", "green forest sunlight", "nature peaceful", "mountain landscape"],
    ["candle flame close up", "lamp flame bokeh", "warm light bokeh", "golden light"],
    ["temple courtyard", "Indian prayer", "devotional worship", "temple entrance"],
    ["clouds sunrise", "sky timelapse", "golden clouds", "peaceful sky"],
    ["Indian village morning", "rural India sunrise", "Indian countryside", "village temple"],
    ["waterfall nature", "mountain stream", "misty mountains", "peaceful waterfall"],
]

FILTERS = [
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=saturation=1.08:contrast=1.03",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=saturation=0.94:contrast=1.06",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=brightness=0.02:saturation=1.12",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=brightness=-0.01:contrast=1.08",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,hue=h=4:s=1.04",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=saturation=1.03:gamma=1.04",
]


def _load_ledger() -> dict:
    if not LEDGER.exists():
        return {"used_video_ids": [], "history": []}
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
        data.setdefault("used_video_ids", [])
        data.setdefault("history", [])
        return data
    except Exception:
        return {"used_video_ids": [], "history": []}


def _save_ledger(data: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _api(session: requests.Session, query: str, per_page: int = 80) -> dict:
    if not API_KEY:
        raise RuntimeError("SETUP_REQUIRED: PEXELS_API_KEY repository secret is missing")
    r = session.get(
        PEXELS_ROOT,
        params={"query": query, "size": "medium", "per_page": per_page},
        headers={"Authorization": API_KEY},
        timeout=45,
    )
    if r.status_code == 429:
        raise RuntimeError("PEXELS_RATE_LIMIT: Pexels API rate limit reached; stop rather than hammering the API")
    r.raise_for_status()
    return r.json()


def _candidate_file(video: dict) -> dict | None:
    files = [f for f in video.get("video_files", []) if str(f.get("file_type", "")).lower() == "video/mp4" and f.get("link")]
    if not files:
        return None
    portrait = [f for f in files if (f.get("height") or 0) >= (f.get("width") or 0) and (f.get("height") or 0) >= 900]
    pool = portrait or files
    suitable = [f for f in pool if (f.get("width") or 0) >= 540 and (f.get("height") or 0) >= 540]
    return min(suitable or pool, key=lambda f: (f.get("width") or 99999) * (f.get("height") or 99999))


def _pick(session: requests.Session, query: str, used: set[str], rng: random.Random) -> tuple[dict, dict]:
    data = _api(session, query, per_page=80)
    videos = list(data.get("videos", []))
    rng.shuffle(videos)
    for video in videos:
        vid = str(video.get("id", ""))
        vf = _candidate_file(video)
        if vid and vid not in used and vf:
            return video, vf
    raise RuntimeError(f"PEXELS_EXHAUSTED: no unused suitable stock video found for query '{query}'")


def _download(session: requests.Session, url: str, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".part")
    with session.get(url, stream=True, timeout=(20, 180)) as r:
        r.raise_for_status()
        with partial.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if partial.stat().st_size < 50_000:
        partial.unlink(missing_ok=True)
        raise RuntimeError("PEXELS_DOWNLOAD_FATAL: downloaded stock video is suspiciously small")
    partial.replace(path)


def _render_clip(source: Path, target: Path, index: int, seed: int) -> None:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source)],
        capture_output=True, text=True, check=True,
    )
    duration = max(1.0, float(probe.stdout.strip() or "1"))
    max_start = max(0.0, duration - 15.0)
    start = (seed % 1000) / 1000.0 * max_start if max_start else 0.0
    vf = FILTERS[(index - 1) % len(FILTERS)]
    zoom = 1.0 + (((seed >> 8) % 7) / 100.0)
    zoom_filter = f"scale=iw*{zoom:.3f}:ih*{zoom:.3f},crop=720:1280:(iw-720)/2:(ih-1280)/2"
    filter_chain = vf + "," + zoom_filter
    subprocess.run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-ss", f"{start:.3f}", "-i", str(source),
        "-t", "15", "-vf", filter_chain, "-r", str(FPS),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if not target.exists() or target.stat().st_size < 50_000:
        raise RuntimeError(f"PEXELS_RENDER_FATAL: {target} missing or too small")


def generate_stock_clip(session: requests.Session, prompt: str, index: int) -> Path:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger()
    used = set(map(str, ledger.get("used_video_ids", [])))
    run_used: set[str] = set()
    run_seed = int(hashlib.sha256(f"{datetime.now(timezone.utc).date()}-{index}-{time.time_ns()}".encode()).hexdigest()[:12], 16)
    rng = random.Random(run_seed)
    pool = QUERY_POOLS[(len(ledger.get("history", [])) + index - 1) % len(QUERY_POOLS)]
    queries = list(pool)
    rng.shuffle(queries)

    last_error = None
    for query in queries:
        try:
            video, vf = _pick(session, query, used | run_used, rng)
            vid = str(video["id"])
            raw = OUT / f"pexels_{vid}.mp4"
            _download(session, vf["link"], raw)
            target = VIDEOS / f"scene_{index}.mp4"
            _render_clip(raw, target, index, run_seed)
            raw.unlink(missing_ok=True)
            run_used.add(vid)
            used.add(vid)
            user = video.get("user") or {}
            credit = {
                "video_id": vid,
                "query": query,
                "page_url": video.get("url", ""),
                "creator": user.get("name", "Pexels creator"),
                "creator_url": user.get("url", ""),
            }
            ledger["used_video_ids"] = sorted(used)
            ledger.setdefault("history", []).append({**credit, "used_at": datetime.now(timezone.utc).isoformat()})
            _save_ledger(ledger)
            with CREDITS.open("a", encoding="utf-8") as f:
                f.write(f"Scene {index}: {query} — {credit['creator']} — {credit['page_url']}\n")
            print(f"PEXELS_OK: scene={index} video_id={vid} query={query} creator={credit['creator']}", flush=True)
            return target
        except Exception as exc:
            last_error = exc
            print(f"PEXELS_WARNING: query={query} failed: {exc}", flush=True)
    raise RuntimeError(f"PEXELS_FATAL: unable to obtain a unique stock clip for scene {index}: {last_error}")
