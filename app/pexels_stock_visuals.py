from __future__ import annotations

import hashlib
import json
import os
import random
import re
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

SCENE_PLANS = {
    1: {"name": "Shri Ram invocation", "queries": ["Lord Rama idol Hindu temple devotional", "Shri Ram murti temple diya", "Lord Rama statue Indian temple worship"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "ayodhya", "temple", "diya", "idol", "murti"]},
    2: {"name": "Raghunandan in the heart", "queries": ["Lord Rama close up idol Hindu devotional", "Shri Ram murti close up temple", "Lord Rama deity worship diya"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "deity", "idol", "murti", "temple", "diya"]},
    3: {"name": "Ram naam jyoti", "queries": ["Lord Rama idol diya aarti Hindu temple", "Shri Ram deity oil lamp devotional", "Rama temple aarti diya worship"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "diya", "aarti", "lamp", "temple", "idol"]},
    4: {"name": "Jai Shri Ram", "queries": ["Lord Rama deity Hindu devotional temple", "Shri Ram murti Indian temple bells", "Lord Rama statue devotional worship India"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "deity", "murti", "idol", "temple", "bells"]},
    5: {"name": "Ram as protector", "queries": ["Lord Rama protecting devotee Hindu devotional", "Shri Ram devotee prayer temple", "Lord Rama deity devotee worship"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "devotee", "prayer", "temple", "idol"]},
    6: {"name": "Ram naam as refuge", "queries": ["Lord Rama idol folded hands devotee Hindu temple", "Shri Ram deity prayer devotional India", "Rama murti devotee namaskar temple"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "prayer", "namaskar", "devotee", "temple", "murti"]},
    7: {"name": "Power of Ram naam", "queries": ["Lord Rama and Hanuman Hindu devotional temple", "Shri Ram Hanuman murti worship", "Rama Hanuman devotional statue India"], "required": ["rama", "ram", "hanuman"], "preferred": ["rama", "ram", "hanuman", "temple", "idol", "murti", "devotional"]},
    8: {"name": "Prince of Ayodhya", "queries": ["Lord Rama Ayodhya Ram Mandir devotional", "Shri Ram Ram Mandir Ayodhya temple", "Ayodhya Ram temple Lord Rama deity"], "required": ["rama", "ram", "ayodhya"], "preferred": ["rama", "ram", "ayodhya", "mandir", "temple", "deity", "idol"]},
    9: {"name": "Jai Shri Ram chorus", "queries": ["Lord Rama devotees Hindu temple worship", "Shri Ram temple devotees saffron devotional", "Ram bhakti devotees temple aarti"], "required": ["rama", "ram"], "preferred": ["rama", "ram", "devotees", "temple", "aarti", "saffron", "bhakti"]},
    10: {"name": "Ram Hanuman aarti", "queries": ["Lord Rama Hanuman aarti Hindu temple", "Shri Ram Hanuman devotional worship diya", "Rama Hanuman temple aarti devotees"], "required": ["rama", "ram", "hanuman"], "preferred": ["rama", "ram", "hanuman", "aarti", "diya", "temple", "worship"]},
    11: {"name": "Final Ram hero shot", "queries": ["Lord Rama deity hero shot Hindu temple", "Shri Ram murti beautiful devotional temple", "Lord Rama statue close up Indian temple"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "deity", "idol", "murti", "temple", "close up"]},
    12: {"name": "Jai Jai Ram closing", "queries": ["Lord Rama idol diya temple closing devotional", "Shri Ram deity aarti Hindu temple final", "Lord Rama murti glowing diya devotional"], "required": ["rama", "ram", "shri ram"], "preferred": ["rama", "ram", "shri ram", "diya", "aarti", "temple", "idol", "murti"]},
}

FORBIDDEN_TERMS = {
    "mosque", "masjid", "minaret", "minar", "islamic", "quran", "muslim",
    "church", "chapel", "cathedral", "cross", "christian", "bible",
    "synagogue", "torah", "gurudwara", "gurdwara", "sikh", "golden temple",
    "buddhist temple", "pagoda", "stupa", "jain temple", "jainism",
}

FILTERS = [
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=saturation=1.08:contrast=1.03",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=saturation=1.02:contrast=1.05",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=brightness=0.02:saturation=1.10",
    "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,eq=brightness=-0.01:contrast=1.08",
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
    r = session.get(PEXELS_ROOT, params={"query": query, "size": "medium", "per_page": per_page}, headers={"Authorization": API_KEY}, timeout=45)
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


def _text_blob(video: dict) -> str:
    user = video.get("user") or {}
    fields = [video.get("url", ""), video.get("image", ""), user.get("name", ""), user.get("url", "")]
    return " ".join(str(x).lower() for x in fields if x)


def _forbidden(video: dict) -> str | None:
    blob = _text_blob(video)
    for term in FORBIDDEN_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", blob):
            return term
    return None


def _matches_scene(video: dict, plan: dict) -> tuple[bool, int, str]:
    blob = _text_blob(video)
    forbidden = _forbidden(video)
    if forbidden:
        return False, -1000, f"forbidden={forbidden}"
    required_hits = [term for term in plan["required"] if re.search(rf"\b{re.escape(term)}\b", blob)]
    if not required_hits:
        return False, -900, "required devotional subject absent from metadata"
    score = 100 * len(required_hits)
    preferred_hits = [term for term in plan["preferred"] if re.search(rf"\b{re.escape(term)}\b", blob)]
    score += 10 * len(preferred_hits)
    return True, score, f"required={required_hits}; preferred={preferred_hits}"


def _pick(session: requests.Session, query: str, used: set[str], plan: dict, rng: random.Random) -> tuple[dict, dict, str]:
    data = _api(session, query, per_page=80)
    videos = list(data.get("videos", []))
    rng.shuffle(videos)
    candidates = []
    for video in videos:
        vid = str(video.get("id", ""))
        vf = _candidate_file(video)
        if not vid or vid in used or not vf:
            continue
        ok, score, reason = _matches_scene(video, plan)
        if ok:
            candidates.append((score, video, vf, reason))
        else:
            print(f"PEXELS_REJECT: scene_subject={plan['name']} video_id={vid} reason={reason}", flush=True)
    if not candidates:
        raise RuntimeError(f"PEXELS_NO_MATCH: query '{query}' returned no scene-safe devotional footage")
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, video, vf, reason = candidates[0]
    return video, vf, reason


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
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source)], capture_output=True, text=True, check=True)
    duration = max(1.0, float(probe.stdout.strip() or "1"))
    max_start = max(0.0, duration - 15.0)
    start = (seed % 1000) / 1000.0 * max_start if max_start else 0.0
    vf = FILTERS[(index - 1) % len(FILTERS)]
    zoom = 1.0 + (((seed >> 8) % 7) / 100.0)
    zoom_filter = f"scale=iw*{zoom:.3f}:ih*{zoom:.3f},crop=720:1280:(iw-720)/2:(ih-1280)/2"
    filter_chain = vf + "," + zoom_filter
    subprocess.run(["ffmpeg", "-y", "-stream_loop", "-1", "-ss", f"{start:.3f}", "-i", str(source), "-t", "15", "-vf", filter_chain, "-r", str(FPS), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if not target.exists() or target.stat().st_size < 50_000:
        raise RuntimeError(f"PEXELS_RENDER_FATAL: {target} missing or too small")


def generate_stock_clip(session: requests.Session, image_url: str, prompt: str, index: int) -> Path:
    del image_url
    VIDEOS.mkdir(parents=True, exist_ok=True)
    if index not in SCENE_PLANS:
        raise RuntimeError(f"PEXELS_FATAL: no devotional scene plan exists for scene {index}")
    plan = SCENE_PLANS[index]
    ledger = _load_ledger()
    used = set(map(str, ledger.get("used_video_ids", [])))
    run_used: set[str] = set()
    run_seed = int(hashlib.sha256(f"{datetime.now(timezone.utc).date()}-{index}-{time.time_ns()}".encode()).hexdigest()[:12], 16)
    rng = random.Random(run_seed)
    prompt_note = " ".join(str(prompt or "").split())[:500]
    last_error = None
    for query in plan["queries"]:
        try:
            video, vf, match_reason = _pick(session, query, used | run_used, plan, rng)
            vid = str(video["id"])
            raw = OUT / f"pexels_{vid}.mp4"
            _download(session, vf["link"], raw)
            target = VIDEOS / f"scene_{index}.mp4"
            _render_clip(raw, target, index, run_seed)
            raw.unlink(missing_ok=True)
            run_used.add(vid)
            used.add(vid)
            user = video.get("user") or {}
            credit = {"video_id": vid, "scene": index, "scene_name": plan["name"], "query": query, "match_reason": match_reason, "base_prompt": prompt_note, "page_url": video.get("url", ""), "creator": user.get("name", "Pexels creator"), "creator_url": user.get("url", "")}
            ledger["used_video_ids"] = sorted(used)
            ledger.setdefault("history", []).append({**credit, "used_at": datetime.now(timezone.utc).isoformat()})
            _save_ledger(ledger)
            with CREDITS.open("a", encoding="utf-8") as f:
                f.write(f"Scene {index} [{plan['name']}]: {query} — {credit['creator']} — match={match_reason} — {credit['page_url']}\n")
            print(f"PEXELS_OK: scene={index} subject={plan['name']} video_id={vid} query={query} match={match_reason} creator={credit['creator']}", flush=True)
            return target
        except Exception as exc:
            last_error = exc
            print(f"PEXELS_WARNING: scene={index} query={query} failed: {exc}", flush=True)
    raise RuntimeError(f"PEXELS_FATAL: scene {index} ({plan['name']}) could not obtain scene-safe Rama/Hanuman devotional footage. No generic scenery fallback is permitted. Last error: {last_error}")
