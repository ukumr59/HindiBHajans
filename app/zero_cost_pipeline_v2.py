from __future__ import annotations

import html
import json
import math
import os
import shutil
import subprocess
import wave
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
MAX_VIDEOS = max(1, min(3, int(os.getenv("MAX_VIDEOS", "3"))))
SECONDS = max(30, min(60, int(os.getenv("VIDEO_SECONDS", "45"))))
FPS = 8
W, H = 720, 1280

PACKS = [
    {
        "slug": "ram", "deity": "श्री राम", "title": "राम नाम की भक्ति",
        "mantra": "श्री राम जय राम जय जय राम", "bg": (72, 24, 18), "accent": (222, 157, 72),
        "captions": ["राम नाम में मन को शांति मिले", "भक्ति की ज्योति हर हृदय में जले", "श्री राम का स्मरण जीवन को उजला करे", "हर सांस में राम, हर धड़कन में राम"],
        "narration": "श्री राम। राम नाम में मन को शांति मिले। भक्ति की ज्योति हर हृदय में जले। श्री राम का स्मरण जीवन को उजला करे। हर सांस में राम, हर धड़कन में राम। श्री राम जय राम जय जय राम।",
    },
    {
        "slug": "krishna", "deity": "श्री कृष्ण", "title": "कृष्ण भक्ति की मधुर धुन",
        "mantra": "राधे कृष्ण, राधे कृष्ण", "bg": (16, 39, 72), "accent": (86, 158, 218),
        "captions": ["मुरली की मधुर धुन मन को छू जाए", "श्याम नाम से हर चिंता दूर हो जाए", "राधे कृष्ण की भक्ति मन में बस जाए", "हर पल प्रेम, हर पल कृष्ण स्मरण"],
        "narration": "श्री कृष्ण। मुरली की मधुर धुन मन को छू जाए। श्याम नाम से हर चिंता दूर हो जाए। राधे कृष्ण की भक्ति मन में बस जाए। हर पल प्रेम, हर पल कृष्ण स्मरण। राधे कृष्ण, राधे कृष्ण।",
    },
    {
        "slug": "bhakti", "deity": "भक्ति संध्या", "title": "भक्ति की मधुर प्रार्थना",
        "mantra": "ॐ शांति शांति शांति", "bg": (43, 24, 54), "accent": (198, 116, 68),
        "captions": ["भक्ति में मन को ठहरने दो", "दीप की लौ में शांति को महसूस करो", "प्रार्थना के इन पलों को अपने नाम करो", "मन शांत हो, हृदय भक्ति से भर जाए"],
        "narration": "भक्ति संध्या। भक्ति में मन को ठहरने दो। दीप की लौ में शांति को महसूस करो। प्रार्थना के इन पलों को अपने नाम करो। मन शांत हो, हृदय भक्ति से भर जाए। ॐ शांति शांति शांति।",
    },
]


def trends() -> list[dict]:
    urls = ["https://trends.google.com/trending/rss?geo=IN", "https://trends.google.co.in/trends/trendingsearches/daily/rss?geo=IN"]
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "BhajanAabha/4.0"})
            root = ET.fromstring(urlopen(req, timeout=15).read())
            out = []
            for item in root.findall(".//item"):
                title = html.unescape(item.findtext("title", "").strip())
                if title:
                    out.append({"title": title, "traffic": item.findtext("{*}approx_traffic", "")})
            if out:
                return out
        except Exception as exc:
            print("TREND_SOURCE_FAILED", type(exc).__name__, str(exc)[:180])
    # Deterministic fallback: generation must never fail just because trends are unavailable.
    return [{"title": p["deity"], "traffic": "fallback"} for p in PACKS]


def choose_packs(items: list[dict]) -> list[dict]:
    # Keep deterministic and copyright-safe: all visual/audio material is generated locally.
    chosen = []
    seen = set()
    keys = {
        "ram": ("ram", "राम", "ayodhya", "अयोध्या", "sita", "सीता"),
        "krishna": ("krishna", "कृष्ण", "radha", "राधा", "vrindavan", "वृंदावन"),
        "bhakti": ("bhajan", "भजन", "aarti", "आरती", "mantra", "मंत्र", "bhakti", "भक्ति"),
    }
    for item in items:
        text = item["title"].lower()
        for p in PACKS:
            if p["slug"] in seen:
                continue
            if any(k.lower() in text for k in keys[p["slug"]]):
                chosen.append(p); seen.add(p["slug"]); break
        if len(chosen) >= MAX_VIDEOS:
            return chosen
    for p in PACKS:
        if p["slug"] not in seen:
            chosen.append(p); seen.add(p["slug"])
        if len(chosen) >= MAX_VIDEOS:
            break
    return chosen


def voice_path() -> str:
    for name in ("espeak-ng", "espeak"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("VOICE_FATAL: espeak-ng/espeak is not installed")


def wav_stats(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        rate = w.getframerate()
        duration = w.getnframes() / float(rate or 1)
        if not frames:
            return duration, 0.0
        import array
        data = array.array("h")
        data.frombytes(frames[:len(frames) - (len(frames) % 2)])
        if not data:
            return duration, 0.0
        rms = math.sqrt(sum(x * x for x in data) / len(data)) / 32768.0
        return duration, rms


def ensure_voice(text: str, path: Path) -> None:
    exe = voice_path()
    # The critical fix: generate actual spoken Hindi and reject silent/invalid output.
    attempts = [
        [exe, "-q", "-v", "hi", "-s", "138", "-p", "48", "-a", "145", "-w", str(path), text],
        [exe, "-q", "-v", "hi+f2", "-s", "138", "-p", "48", "-a", "145", "-w", str(path), text],
    ]
    errors = []
    for cmd in attempts:
        path.unlink(missing_ok=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0 or not path.exists():
                errors.append(f"rc={r.returncode} stderr={r.stderr[-250:]}")
                continue
            duration, rms = wav_stats(path)
            print(f"VOICE_TEST executable={exe} bytes={path.stat().st_size} duration={duration:.2f}s rms={rms:.5f}")
            if path.stat().st_size >= 1000 and duration >= 2.0 and rms >= 0.002:
                return
            errors.append(f"invalid audio duration={duration:.2f}s rms={rms:.5f}")
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError("VOICE_FATAL: Hindi spoken audio failed validation: " + " | ".join(errors))


def font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def frame(pack: dict, n: int, title_font, body_font, small_font) -> bytes:
    t = n / FPS
    base = Image.new("RGB", (W, H), pack["bg"])
    d = ImageDraw.Draw(base, "RGBA")
    cx, cy = W // 2, int(H * 0.47)
    # Procedural devotional mandala + diya; no downloaded artwork.
    for rr in range(300, 50, -25):
        alpha = max(12, 80 - rr // 5)
        d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=pack["accent"] + (alpha,), width=2)
    for i in range(24):
        a = t * 0.25 + i * math.pi / 12
        x = cx + int(math.cos(a) * 255); y = cy + int(math.sin(a) * 255)
        d.ellipse((x-4, y-4, x+4, y+4), fill=pack["accent"] + (150,))
    d.ellipse((cx-105, cy+40, cx+105, cy+105), fill=(105, 53, 24, 235))
    d.ellipse((cx-11, cy-5, cx+11, cy+62), fill=(255, 192, 52, 255))
    d.ellipse((cx-24, cy+10, cx+24, cy+55), fill=(255, 245, 185, 245))
    # Header
    d.rounded_rectangle((24, 24, W-24, 142), radius=26, fill=(5, 5, 9, 205), outline=pack["accent"] + (230,), width=2)
    d.text((W//2, 45), "BHAJAN AABHA", font=small_font, anchor="ma", fill=(255, 240, 210, 255))
    d.text((W//2, 76), pack["deity"], font=title_font, anchor="ma", fill=(255, 250, 235, 255))
    # Caption changes through the video.
    idx = min(len(pack["captions"])-1, int(t / SECONDS * len(pack["captions"])))
    y = H - 230
    d.rounded_rectangle((24, y-40, W-24, H-70), radius=25, fill=(5, 5, 9, 215))
    d.text((W//2, y), pack["captions"][idx], font=body_font, anchor="ma", fill=(255,255,255,255))
    d.text((W//2, H-45), pack["mantra"], font=small_font, anchor="ms", fill=pack["accent"] + (255,))
    return base.tobytes()


def render(video: Path, audio: Path, pack: dict) -> None:
    tf, bf, sf = font(42, True), font(30), font(20)
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", str(audio), "-t", str(SECONDS), "-map", "0:v:0", "-map", "1:a:0", "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-af", "apad", "-t", str(SECONDS), "-movflags", "+faststart", str(video)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for n in range(SECONDS * FPS):
            p.stdin.write(frame(pack, n, tf, bf, sf))
        p.stdin.close()
        err = p.stderr.read().decode("utf-8", errors="ignore")
        if p.wait() != 0:
            raise RuntimeError("FFMPEG_FATAL: " + err[-1800:])
    finally:
        if p.poll() is None:
            p.kill()


def validate(video: Path) -> None:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,duration", "-of", "json", str(video)]
    data = json.loads(subprocess.check_output(cmd, text=True))
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v or not a or v.get("width") != W or v.get("height") != H or a.get("codec_name") != "aac":
        raise RuntimeError("OUTPUT_FATAL: MP4 failed video/audio stream validation")
    if float(a.get("duration", 0)) < 1:
        raise RuntimeError("OUTPUT_FATAL: MP4 contains no usable audio duration")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); VIDEOS.mkdir(parents=True, exist_ok=True)
    for p in VIDEOS.glob("*.mp4"): p.unlink()
    for p in OUT.glob("*_narration.wav"): p.unlink()
    items = trends(); packs = choose_packs(items)
    print(f"ARCHITECTURE=v2 ZERO_COST=true KAGGLE=false PAID_SERVICES=false")
    print(f"TREND_ITEMS={len(items)} SELECTED={len(packs)}")
    results = []
    for i, pack in enumerate(packs, 1):
        print(f"PREPARING {i}/{len(packs)}: {pack['deity']}")
        audio = OUT / f"{pack['slug']}_narration.wav"
        ensure_voice(pack["narration"], audio)
        video = VIDEOS / f"{datetime.now(timezone.utc):%Y%m%d}_{i}_{pack['slug']}.mp4"
        render(video, audio, pack)
        validate(video)
        audio.unlink(missing_ok=True)
        results.append({"topic": pack["deity"], "title": pack["title"], "video": str(video), "duration_sec": SECONDS})
        print(f"VIDEO_OK {video}")
    state = {"channel": "Bhajan Aabha", "architecture": "github-runner-only-v2", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "videos": results, "paid_services": False, "paid_gpu": False, "kaggle": False, "external_media_downloads": False, "voice": "local eSpeak Hindi speech with RMS validation", "status": "READY_FOR_RELEASE"}
    (OUT / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"trends": items[:20], "videos": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
