from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
AUDIO = OUT / "audio"
AGNES_ROOT = os.getenv("AGNES_ROOT", "https://apihub.agnes-ai.com").rstrip("/")
AGNES_API_KEY = os.getenv("AGNES_API_KEY", "").strip()
ACESTEP_ROOT = os.getenv("ACESTEP_ROOT", "https://ace-step-ace-step-v1-5.hf.space").rstrip("/")
VIDEO_SECONDS = 45
SCENE_SECONDS = 15
FPS = 24
WIDTH, HEIGHT = 720, 1280

PACK = {
    "slug": "ram",
    "deity": "श्री राम",
    "title": "श्री राम — भक्ति की मधुर धुन",
    "lyrics": """[Intro]\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Outro]\nश्री राम... जय राम... जय जय राम...""",
    "music_prompt": "Modern cinematic Hindi devotional bhajan for YouTube, expressive natural male lead vocal singing in clear Hindi, memorable devotional melody, harmonium, tabla, dholak, bansuri flute, tanpura drone, soft temple bells, subtle acoustic strings, tasteful bass, polished contemporary stereo production, emotional and peaceful, strong verse and chorus dynamics, natural breaths and phrasing, NOT spoken word, NOT narration, NOT humming, NOT a cappella.",
    "image_prompt": "Lord Rama as a revered Hindu deity, serene compassionate divine face, blue-tinted skin, golden crown, traditional yellow silk dhoti, ornate jewelry, bow beside him, subtle divine aura, magnificent Ayodhya-inspired temple courtyard at dawn, warm golden sunlight, marigold flowers, oil lamps, soft incense haze, cinematic devotional sacred art, highly detailed face and hands, dignified traditional Hindu iconography, vertical composition, no text, no watermark, no modern clothing.",
    "scene_prompts": [
        "Cinematic slow push toward Lord Rama in the temple courtyard, warm dawn rays, gently moving garments and flower petals, flickering diyas and incense haze, serene devotional atmosphere, realistic natural motion, preserve the deity's face, hands and identity exactly.",
        "Cinematic lateral camera movement around Lord Rama, temple lamps flickering, incense drifting, petals floating, subtle movement of cloth and jewelry, warm golden light, shallow depth of field, reverent devotional mood, preserve the same deity identity and facial details.",
        "Cinematic gradual rise from foreground diyas toward Lord Rama's peaceful face as sunrise brightens the temple, petals drifting, soft divine aura, emotional devotional climax, realistic camera movement, preserve the same deity identity and facial details.",
    ],
}


def http_json(session: requests.Session, method: str, url: str, *, headers=None, body=None, timeout=90, retries=4):
    last = None
    for attempt in range(retries):
        try:
            r = session.request(method, url, headers=headers, json=body, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}: {r.text[:500]}"
                time.sleep(min(60, 5 * (attempt + 1)))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = str(exc)
            if attempt < retries - 1:
                time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError(f"HTTP request failed after retries: {url} | {last}")


def download(session: requests.Session, url: str, path: Path, *, min_bytes=10000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if path.stat().st_size < min_bytes:
        raise RuntimeError(f"DOWNLOAD_FATAL: suspiciously small file: {path}")


def require_env():
    if not AGNES_API_KEY:
        raise RuntimeError("SETUP_REQUIRED: AGNES_API_KEY repository secret is missing")


def generate_image(session: requests.Session) -> str:
    print("IMAGE: generating canonical deity reference")
    data = http_json(session, "POST", f"{AGNES_ROOT}/v1/images/generations", headers={"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"}, body={"model": "agnes-image-2.1-flash", "prompt": PACK["image_prompt"], "n": 1, "size": "576x1024"}, timeout=180)
    try:
        url = data["data"][0]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"IMAGE_FATAL: unexpected Agnes response: {data}") from exc
    if not str(url).startswith("http"):
        raise RuntimeError(f"IMAGE_FATAL: invalid image URL: {url}")
    print("IMAGE_OK")
    return url


def generate_video_clip(session: requests.Session, image_url: str, prompt: str, index: int) -> Path:
    if index > 1:
        print("VIDEO: free Agnes RPM guard — waiting 65 seconds")
        time.sleep(65)
    payload = {"model": "agnes-video-v2.0", "prompt": prompt, "image": image_url, "width": WIDTH, "height": HEIGHT, "num_frames": SCENE_SECONDS * FPS + 1, "frame_rate": FPS, "negative_prompt": "deformed face, extra fingers, extra limbs, duplicate deity, distorted hands, text, watermark, logo, flicker, jitter, cartoon, low detail"}
    print(f"VIDEO: submitting scene {index}/3")
    data = http_json(session, "POST", f"{AGNES_ROOT}/v1/videos", headers={"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"}, body=payload, timeout=120)
    video_id = data.get("video_id") or data.get("id")
    if not video_id:
        raise RuntimeError(f"VIDEO_FATAL: no video_id in response: {data}")
    deadline = time.time() + 25 * 60
    while time.time() < deadline:
        r = session.get(f"{AGNES_ROOT}/agnesapi", params={"video_id": video_id, "model_name": "agnes-video-v2.0"}, headers={"Authorization": f"Bearer {AGNES_API_KEY}"}, timeout=45)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(20)
            continue
        r.raise_for_status()
        status = r.json()
        state = str(status.get("status", "")).lower()
        print(f"VIDEO: scene={index} status={state} progress={status.get('progress', 0)}%")
        if state == "completed":
            url = status.get("url")
            if not url:
                raise RuntimeError(f"VIDEO_FATAL: completed response has no URL: {status}")
            path = VIDEOS / f"scene_{index}.mp4"
            download(session, url, path, min_bytes=50000)
            return path
        if state == "failed":
            raise RuntimeError(f"VIDEO_FATAL: scene {index} failed: {status.get('error')}")
        time.sleep(20)
    raise RuntimeError(f"VIDEO_FATAL: scene {index} timed out")


def _extract_task_id(data):
    if isinstance(data, dict):
        if data.get("task_id"):
            return data["task_id"]
        inner = data.get("data")
        if isinstance(inner, dict) and inner.get("task_id"):
            return inner["task_id"]
    raise RuntimeError(f"MUSIC_FATAL: ACE-Step task id missing: {data}")


def generate_music(session: requests.Session) -> Path:
    print("MUSIC: submitting Hindi sung bhajan to public ACE-Step 1.5 ZeroGPU Space")
    payload = {"prompt": PACK["music_prompt"], "lyrics": PACK["lyrics"], "vocal_language": "hi", "audio_duration": VIDEO_SECONDS, "model": "acestep-v15-turbo", "thinking": False, "sample_mode": False, "use_format": False, "inference_steps": 8, "batch_size": 1, "use_random_seed": True}
    data = http_json(session, "POST", f"{ACESTEP_ROOT}/release_task", body=payload, timeout=120, retries=5)
    task_id = _extract_task_id(data)
    print("MUSIC_TASK", task_id)
    deadline = time.time() + 25 * 60
    while time.time() < deadline:
        try:
            result = http_json(session, "POST", f"{ACESTEP_ROOT}/query_result", body={"task_id_list": [task_id]}, timeout=60, retries=2)
        except RuntimeError as exc:
            print("MUSIC: transient query error:", exc)
            time.sleep(15)
            continue
        items = result.get("data") if isinstance(result, dict) else result
        if isinstance(items, dict):
            items = items.get("data") or items.get("results") or []
        if not isinstance(items, list) or not items:
            time.sleep(15)
            continue
        item = items[0]
        status = int(item.get("status", 0))
        if status == 2:
            raise RuntimeError(f"MUSIC_FATAL: ACE-Step failed: {item.get('result', item)}")
        if status != 1:
            print(f"MUSIC: still running status={status}")
            time.sleep(15)
            continue
        raw = item.get("result", "[]")
        try:
            results = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"MUSIC_FATAL: invalid result JSON: {raw}") from exc
        if not results:
            raise RuntimeError(f"MUSIC_FATAL: successful task returned no audio: {item}")
        audio_ref = results[0].get("file")
        if not audio_ref:
            raise RuntimeError(f"MUSIC_FATAL: result contains no audio file: {results[0]}")
        audio_url = audio_ref if str(audio_ref).startswith("http") else urljoin(ACESTEP_ROOT + "/", str(audio_ref).lstrip("/"))
        path = AUDIO / "bhajan_source.mp3"
        download(session, audio_url, path, min_bytes=20000)
        print("MUSIC_OK", path)
        return path
    raise RuntimeError("MUSIC_FATAL: ACE-Step task timed out")


def ffmpeg(*args: str):
    p = subprocess.run(["ffmpeg", "-y", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError("FFMPEG_FATAL: " + p.stderr[-4000:])


def make_srt(path: Path):
    entries = [(1, 6, "मन में बसो रघुनंदन, चरणों में मेरा ध्यान"), (6, 11, "राम नाम की ज्योति जले, रोशन हो हर प्राण"), (12, 17, "श्री राम जय राम, जय जय राम"), (17, 22, "मेरे मन के दीप में, बसते श्री राम"), (23, 28, "दुख की घड़ी में साथ दो, हे दीनदयाल भगवान"), (28, 33, "तेरा नाम ही आसरा, तेरा नाम ही सम्मान"), (34, 39, "श्री राम जय राम, जय जय राम"), (39, 44, "मेरे मन के दीप में, बसते श्री राम"), (44, 45, "श्री राम... जय राम... जय जय राम...")]
    def stamp(sec): return f"00:00:{sec:02},000"
    lines = []
    for i, (start, end, text) in enumerate(entries, 1):
        lines += [str(i), f"{stamp(start)} --> {stamp(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def assemble(scene_paths: list[Path], music: Path, final_path: Path):
    concat = OUT / "scenes.txt"
    concat.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_paths) + "\n", encoding="utf-8")
    visual = OUT / "visual.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(visual))
    srt = OUT / "lyrics.srt"
    make_srt(srt)
    subtitle_filter = "subtitles=" + str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
    ffmpeg("-i", str(visual), "-i", str(music), "-filter_complex", "[1:a]atrim=0:45,asetpts=N/SR/TB,volume=1.0[a]", "-map", "0:v:0", "-map", "[a]", "-vf", subtitle_filter, "-t", "45", "-r", str(FPS), "-s", f"{WIDTH}x{HEIGHT}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final_path))


def validate(path: Path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height:format=duration,size", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    data = json.loads(probe.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not audio: raise RuntimeError("OUTPUT_FATAL: final MP4 must contain video and audio")
    if (video.get("width"), video.get("height")) != (WIDTH, HEIGHT): raise RuntimeError(f"OUTPUT_FATAL: wrong dimensions: {video.get('width')}x{video.get('height')}")
    if audio.get("codec_name") != "aac": raise RuntimeError("OUTPUT_FATAL: audio codec is not AAC")
    duration = float(data.get("format", {}).get("duration", 0))
    if duration < 43: raise RuntimeError(f"OUTPUT_FATAL: final duration too short: {duration:.2f}s")
    return duration


def main():
    require_env()
    for d in (OUT, VIDEOS, AUDIO): d.mkdir(parents=True, exist_ok=True)
    for p in VIDEOS.glob("*.mp4"): p.unlink()
    session = requests.Session()
    session.headers.update({"User-Agent": "BhajanAabha/5.1"})
    print("ARCHITECTURE=v5.1 ZERO_COST=true KAGGLE=false PAID_SERVICES=false ACTIONS_ARTIFACTS=false")
    print("VISUAL_BACKEND=Agnes Image 2.1 Flash + Agnes Video v2.0")
    print("MUSIC_BACKEND=ACE-Step 1.5 public ZeroGPU Space via /release_task")
    image_url = generate_image(session)
    music = generate_music(session)
    scenes = [generate_video_clip(session, image_url, prompt, i) for i, prompt in enumerate(PACK["scene_prompts"], 1)]
    final_path = VIDEOS / f"{datetime.now(timezone.utc):%Y%m%d}_bhajan-aabha_{PACK['slug']}_v5.mp4"
    assemble(scenes, music, final_path)
    duration = validate(final_path)
    state = {"channel": "Bhajan Aabha", "architecture": "v5.1-agnes-visuals-ace-step-music", "status": "READY_FOR_RELEASE", "zero_cost": True, "kaggle": False, "paid_services": False, "paid_gpu": False, "actions_artifacts": False, "deity_reference": "Agnes Image 2.1 Flash", "video_backend": "Agnes Video v2.0", "music_backend": "ACE-Step 1.5 public ZeroGPU Space", "vocal_language": "hi", "duration_sec": duration, "output": str(final_path), "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VIDEO_OK", final_path)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
