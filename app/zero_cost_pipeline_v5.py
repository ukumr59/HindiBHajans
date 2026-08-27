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
VIDEO_SECONDS = int(os.getenv("VIDEO_SECONDS", "45"))
SCENE_SECONDS = 15
FPS = 24
WIDTH, HEIGHT = 720, 1280

PACK = {
    "slug": "ram",
    "deity": "श्री राम",
    "title": "श्री राम — भक्ति की मधुर धुन",
    "lyrics": """[Intro]\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Outro]\nश्री राम... जय राम... जय जय राम...""",
    "music_prompt": "High-energy modern Hindi devotional DJ bhajan / devotional EDM remix, 128 BPM, 4/4, loud punchy club-ready production with a strong four-on-the-floor kick, deep controlled sub-bass, tight electronic bassline, bright modern synth leads and pads, energetic electronic percussion, powerful dhol and dholak grooves, crisp claps, tasteful temple bells, short risers and impact transitions, catchy devotional hook, cinematic build-ups and satisfying chorus drops. Strong expressive male Hindi lead vocal clearly singing every lyric, upfront and intelligible, with tasteful vocal doubles and spacious reverb; harmonium and bansuri are supporting colors, not the dominant instruments. Modern Indian devotional dance sound, energetic festival/DJ feel, wide stereo image, punchy transients, full frequency spectrum, polished commercial master, loud but clean, no clipping, no muddy low end. Arrange as a complete DJ-friendly song with intro, verse, pre-chorus build, big chorus/drop, second verse, bigger final chorus/drop and outro. NOT spoken word, NOT narration, NOT humming, NOT a cappella, NOT an acoustic-only bhajan, NOT a soft meditation track.",
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
    # DJ-oriented final master: keep the generated arrangement intact while
    # applying clean loudness normalization and a -1 dBTP ceiling. This makes
    # the embedded song substantially louder and more consistent without hard
    # clipping or destructive brick-wall limiting.
    audio_filter = "loudnorm=I=-9:TP=-1.0:LRA=7"
    ffmpeg("-i", str(visual), "-i", str(music), "-filter_complex", f"[1:a]atrim=0:{VIDEO_SECONDS},asetpts=N/SR/TB,{audio_filter}[a]", "-map", "0:v:0", "-map", "[a]", "-vf", subtitle_filter, "-t", str(VIDEO_SECONDS), "-r", str(FPS), "-s", f"{WIDTH}x{HEIGHT}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(final_path))
    # Also export the mastered audio as a standalone MP3 so the generated
    # devotional track is directly usable in DJ software/players.
    dj_mp3 = AUDIO / "bhajan_aabha_dj_master.mp3"
    ffmpeg("-i", str(music), "-af", audio_filter, "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3", "-metadata", "title=Bhajan Aabha — DJ Master", "-metadata", "artist=Bhajan Aabha", "-metadata", "genre=Devotional EDM", str(dj_mp3))


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
    if duration < max(43, VIDEO_SECONDS - 2): raise RuntimeError(f"OUTPUT_FATAL: final duration too short: {duration:.2f}s")
    dj_mp3 = AUDIO / "bhajan_aabha_dj_master.mp3"
    if not dj_mp3.exists() or dj_mp3.stat().st_size < 100000: raise RuntimeError("OUTPUT_FATAL: standalone DJ master MP3 missing or suspiciously small")
    return duration


def main():
    require_env()
    for d in (OUT, VIDEOS, AUDIO): d.mkdir(parents=True, exist_ok=True)
    for p in VIDEOS.glob("*.mp4"): p.unlink()
    for p in AUDIO.glob("*.mp3"): p.unlink()
    session = requests.Session()
    session.headers.update({"User-Agent": "BhajanAabha/5.2"})
    print("ARCHITECTURE=v5.2 ZERO_COST=true KAGGLE=false PAID_SERVICES=false ACTIONS_ARTIFACTS=false")
    print("VISUAL_BACKEND=Agnes Image 2.1 Flash + Agnes Video v2.0")
    print("MUSIC_BACKEND=ACE-Step 1.5 HTTP API /v1/music/generate")
    print("MUSIC_STYLE=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128")
    image_url = generate_image(session)
    music = generate_music(session)
    scenes = [generate_video_clip(session, image_url, prompt, i) for i, prompt in enumerate(PACK["scene_prompts"], 1)]
    final_path = VIDEOS / f"{datetime.now(timezone.utc):%Y%m%d}_bhajan-aabha_{PACK['slug']}_v5.mp4"
    assemble(scenes, music, final_path)
    duration = validate(final_path)
    state = {"channel": "Bhajan Aabha", "architecture": "v5.2-agnes-visuals-ace-step-http", "status": "READY_FOR_RELEASE", "zero_cost": True, "kaggle": False, "paid_services": False, "paid_gpu": False, "actions_artifacts": False, "deity_reference": "Agnes Image 2.1 Flash", "video_backend": "Agnes Video v2.0", "music_backend": "ACE-Step 1.5 HTTP API", "music_style": "loud modern devotional EDM / DJ-ready", "bpm": 128, "time_signature": "4/4", "mastering": "-9 LUFS integrated, -1 dBTP ceiling", "dj_master": str(AUDIO / "bhajan_aabha_dj_master.mp3"), "vocal_language": "hi", "duration_sec": duration, "output": str(final_path), "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VIDEO_OK", final_path)
    print("DJ_MASTER_OK", AUDIO / "bhajan_aabha_dj_master.mp3")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
