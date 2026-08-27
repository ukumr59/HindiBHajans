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

VIDEO_SECONDS = max(30, min(60, int(os.getenv("VIDEO_SECONDS", "45"))))
SCENE_SECONDS = 15
FPS = 24
WIDTH, HEIGHT = 720, 1280

PACK = {
    "slug": "ram",
    "deity": "श्री राम",
    "title": "श्री राम — भक्ति की मधुर धुन",
    "lyrics": """[Intro]\n\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Outro]\nश्री राम... जय राम... जय जय राम...""",
    "music_prompt": (
        "A modern cinematic Hindi devotional bhajan for YouTube, warm expressive male lead singing, "
        "clear Hindi pronunciation, devotional melody with memorable chorus, harmonium, tabla, dholak, "
        "bansuri flute, tanpura drone, soft temple bells, subtle acoustic strings, tasteful bass, "
        "polished contemporary devotional production, emotional but peaceful, natural human-like singing, "
        "strong melodic phrasing, dynamic intro-verse-chorus structure, professional stereo mix, "
        "NOT spoken word, NOT narration, NOT humming, NOT a cappella."
    ),
    "image_prompt": (
        "Lord Rama as a revered Hindu deity, serene compassionate divine face, blue-tinted skin, "
        "golden crown, traditional yellow silk dhoti, ornate jewelry, bow resting beside him, "
        "subtle divine aura, standing in a magnificent Ayodhya-inspired temple courtyard at dawn, "
        "warm golden sunlight, marigold flowers, oil lamps, soft incense haze, cinematic devotional "
        "photorealistic sacred art, highly detailed face and hands, dignified traditional iconography, "
        "vertical composition, no text, no watermark, no modern clothing"
    ),
    "scene_prompts": [
        "Slow cinematic push-in toward the deity, gentle golden light rays, subtle flowing cloth and aura, floating flower petals, calm devotional atmosphere, natural camera motion, preserve deity identity and facial details.",
        "Slow lateral camera movement around the deity in the temple courtyard, lamps flickering softly, incense drifting, gentle breeze moving garments, subtle divine glow, cinematic depth of field, preserve deity identity and facial details.",
        "Gradual upward camera move from the lamps toward the deity's peaceful face, warm sunrise brightening the temple, petals drifting through the air, serene divine aura, cinematic devotional climax, preserve deity identity and facial details.",
    ],
}


def require_env() -> None:
    if not AGNES_API_KEY:
        raise RuntimeError(
            "SETUP_REQUIRED: repository secret AGNES_API_KEY is missing. "
            "Create a free Agnes AI API key and add it to GitHub repository secrets as AGNES_API_KEY. "
            "This pipeline does not use paid APIs, paid GPU, Kaggle, or Actions artifacts."
        )


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "BhajanAabha/5.0"})
    return s


def request_json(s: requests.Session, method: str, url: str, *, headers=None, json_body=None, timeout=60, retries=4):
    last = None
    for attempt in range(retries):
        try:
            r = s.request(method, url, headers=headers, json=json_body, timeout=timeout)
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


def generate_image(s: requests.Session) -> str:
    print("IMAGE: generating a single canonical deity reference...")
    body = {
        "model": "agnes-image-2.1-flash",
        "prompt": PACK["image_prompt"],
        "n": 1,
        "size": "576x1024",
        "extra_body": {"response_format": "url"},
    }
    data = request_json(
        s,
        "POST",
        f"{AGNES_ROOT}/v1/images/generations",
        headers={"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"},
        json_body=body,
        timeout=120,
    )
    try:
        url = data["data"][0]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"IMAGE_FATAL: unexpected Agnes image response: {data}") from exc
    print("IMAGE_OK", url[:100])
    return url


def download_binary(s: requests.Session, url: str, path: Path) -> None:
    with s.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if path.stat().st_size < 10_000:
        raise RuntimeError(f"DOWNLOAD_FATAL: suspiciously small file: {path}")


def generate_video_clip(s: requests.Session, image_url: str, prompt: str, index: int) -> Path:
    # Free/default Agnes video access is currently documented at 1 actual request/minute.
    # Generation itself usually takes longer, but this guard prevents rapid 429 retries.
    if index > 1:
        print("VIDEO: respecting Agnes free-tier RPM; waiting 65 seconds before next scene...")
        time.sleep(65)
    body = {
        "model": "agnes-video-v2.0",
        "prompt": prompt,
        "image": image_url,
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": SCENE_SECONDS * FPS + 1,
        "frame_rate": FPS,
        "negative_prompt": "deformed face, extra fingers, extra limbs, duplicate deity, distorted hands, text, watermark, logo, flicker, jitter, cartoon, low detail",
    }
    print(f"VIDEO: submitting scene {index}/3...")
    data = request_json(
        s,
        "POST",
        f"{AGNES_ROOT}/v1/videos",
        headers={"Authorization": f"Bearer {AGNES_API_KEY}", "Content-Type": "application/json"},
        json_body=body,
        timeout=90,
    )
    video_id = data.get("video_id") or data.get("task_id")
    if not video_id:
        raise RuntimeError(f"VIDEO_FATAL: Agnes returned no video_id: {data}")

    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        r = s.get(
            f"{AGNES_ROOT}/agnesapi",
            params={"video_id": video_id, "model_name": "agnes-video-v2.0"},
            headers={"Authorization": f"Bearer {AGNES_API_KEY}"},
            timeout=30,
        )
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(20)
            continue
        r.raise_for_status()
        status = r.json()
        state = str(status.get("status", "")).lower()
        print(f"VIDEO: scene {index} status={state} progress={status.get('progress', 0)}%")
        if state == "completed":
            url = status.get("url") or status.get("video_url")
            if not url or not str(url).startswith("http"):
                raise RuntimeError(f"VIDEO_FATAL: completed response has no downloadable URL: {status}")
            path = VIDEOS / f"scene_{index}.mp4"
            download_binary(s, str(url), path)
            print("VIDEO_OK", path)
            return path
        if state == "failed":
            raise RuntimeError(f"VIDEO_FATAL: scene {index} failed: {status.get('error')}")
        time.sleep(20)
    raise RuntimeError(f"VIDEO_FATAL: scene {index} timed out after 20 minutes")


def generate_music(s: requests.Session) -> Path:
    print("MUSIC: submitting Hindi singing + instrumental music to ACE-Step 1.5 ZeroGPU Space...")
    body = {
        "caption": PACK["music_prompt"],
        "lyrics": PACK["lyrics"],
        "vocal_language": "hi",
        "audio_format": "mp3",
        "audio_duration": VIDEO_SECONDS,
        "model": "acestep-v15-turbo",
        "thinking": True,
        "use_format": True,
        "inference_steps": 8,
        "batch_size": 1,
        "use_random_seed": True,
    }
    data = request_json(
        s,
        "POST",
        f"{ACESTEP_ROOT}/v1/music/generate",
        headers={"Content-Type": "application/json"},
        json_body=body,
        timeout=120,
        retries=5,
    )
    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"MUSIC_FATAL: ACE-Step returned no job_id: {data}")

    deadline = time.time() + 20 * 60
    while time.time() < deadline:
        r = s.get(f"{ACESTEP_ROOT}/v1/jobs/{job_id}", timeout=30)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10)
            continue
        r.raise_for_status()
        status = r.json()
        state = str(status.get("status", "")).lower()
        print(f"MUSIC: status={state} queue={status.get('queue_position')} eta={status.get('eta_seconds')}")
        if state == "succeeded":
            result = status.get("result") or {}
            audio_path = result.get("first_audio_path")
            if not audio_path:
                paths = result.get("audio_paths") or []
                audio_path = paths[0] if paths else None
            if not audio_path:
                raise RuntimeError(f"MUSIC_FATAL: no audio path in success response: {status}")
            url = audio_path if audio_path.startswith("http") else urljoin(ACESTEP_ROOT + "/", audio_path.lstrip("/"))
            path = AUDIO / "bhajan_master.mp3"
            download_binary(s, url, path)
            print("MUSIC_OK", path)
            return path
        if state == "failed":
            raise RuntimeError(f"MUSIC_FATAL: ACE-Step generation failed: {status.get('error')}")
        time.sleep(10)
    raise RuntimeError("MUSIC_FATAL: ACE-Step generation timed out after 20 minutes")


def make_lyrics_srt(path: Path) -> None:
    entries = [
        (1, 6, "मन में बसो रघुनंदन, चरणों में मेरा ध्यान"),
        (6, 11, "राम नाम की ज्योति जले, रोशन हो हर प्राण"),
        (12, 17, "श्री राम जय राम, जय जय राम"),
        (17, 22, "मेरे मन के दीप में, बसते श्री राम"),
        (23, 28, "दुख की घड़ी में साथ दो, हे दीनदयाल भगवान"),
        (28, 33, "तेरा नाम ही आसरा, तेरा नाम ही सम्मान"),
        (34, 39, "श्री राम जय राम, जय जय राम"),
        (39, 44, "मेरे मन के दीप में, बसते श्री राम"),
        (44, 45, "श्री राम... जय राम... जय जय राम..."),
    ]
    def ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int(round((sec - int(sec)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    lines = []
    for i, (a, b, text) in enumerate(entries, 1):
        lines += [str(i), f"{ts(a)} --> {ts(b)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def ffmpeg(*args: str) -> None:
    p = subprocess.run(["ffmpeg", "-y", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError("FFMPEG_FATAL: " + p.stderr[-3000:])


def assemble(scene_paths: list[Path], music: Path, final_path: Path) -> None:
    concat = OUT / "scenes.txt"
    concat.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_paths) + "\n", encoding="utf-8")
    silent = OUT / "visual.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(silent))
    srt = OUT / "lyrics.srt"
    make_lyrics_srt(srt)
    ffmpeg(
        "-i", str(silent), "-i", str(music), "-t", str(VIDEO_SECONDS),
        "-filter_complex", f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[v];[v]subtitles={srt.as_posix()}:force_style='FontName=Noto Sans Devanagari,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=90'[v2]",
        "-map", "[v2]", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(final_path),
    )
    if final_path.stat().st_size < 100_000:
        raise RuntimeError("OUTPUT_FATAL: final MP4 is unexpectedly small")


def validate(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height",
        "-of", "json", str(path)
    ], text=True)
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((x for x in streams if x.get("codec_type") == "video"), None)
    audio = next((x for x in streams if x.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise RuntimeError("OUTPUT_FATAL: final MP4 must contain both video and audio")
    if video.get("width") != WIDTH or video.get("height") != HEIGHT:
        raise RuntimeError(f"OUTPUT_FATAL: expected {WIDTH}x{HEIGHT}, got {video.get('width')}x{video.get('height')}")
    if audio.get("codec_name") != "aac":
        raise RuntimeError("OUTPUT_FATAL: audio must be AAC")
    duration = float(data.get("format", {}).get("duration", 0))
    if duration < VIDEO_SECONDS - 2:
        raise RuntimeError(f"OUTPUT_FATAL: final duration too short: {duration:.2f}s")
    return {"duration": duration, "size": int(data.get("format", {}).get("size", 0)), "video": video, "audio": audio}


def main() -> None:
    require_env()
    for d in (OUT, VIDEOS, AUDIO):
        d.mkdir(parents=True, exist_ok=True)
    for p in VIDEOS.glob("*.mp4"):
        p.unlink()
    s = session()
    print("ARCHITECTURE=v5 ZERO_COST=true KAGGLE=false PAID_SERVICES=false ACTIONS_ARTIFACTS=false")
    print("VIDEO_BACKEND=Agnes Video v2.0 free/default tier")
    print("MUSIC_BACKEND=ACE-Step 1.5 public ZeroGPU Space")

    image_url = generate_image(s)
    music = generate_music(s)
    scenes = [generate_video_clip(s, image_url, prompt, i) for i, prompt in enumerate(PACK["scene_prompts"], 1)]
    final_path = VIDEOS / f"{datetime.now(timezone.utc):%Y%m%d}_bhajan-aabha_{PACK['slug']}_v5.mp4"
    assemble(scenes, music, final_path)
    stats = validate(final_path)

    state = {
        "channel": "Bhajan Aabha",
        "architecture": "v5-agnes-visuals-ace-step-music",
        "status": "READY_FOR_RELEASE",
        "zero_cost": True,
        "kaggle": False,
        "paid_services": False,
        "paid_gpu": False,
        "actions_artifacts": False,
        "deity_reference": "Agnes Image 2.1 Flash",
        "video_backend": "Agnes Video v2.0",
        "music_backend": "ACE-Step 1.5 ZeroGPU Space",
        "vocal_language": "hi",
        "duration_sec": stats["duration"],
        "output": str(final_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VIDEO_OK", final_path)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
