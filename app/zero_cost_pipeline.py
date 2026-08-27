from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
MAX_VIDEOS = max(1, min(3, int(os.getenv("MAX_VIDEOS", "3"))))
SECONDS = max(30, min(60, int(os.getenv("VIDEO_SECONDS", "45"))))
FPS = 8
W, H = 720, 1280

# Fully self-contained zero-cost packs.
# IMPORTANT: this generator deliberately uses NO external image/audio downloads.
# Wikimedia introduced API/CDN rate limits in 2026, so remote media must not be
# a runtime dependency for the daily pipeline.
PACKS = [
    {
        "slug": "ram",
        "deity": "श्री राम",
        "title": "राम नाम की भक्ति",
        "mantra": "श्री राम जय राम जय जय राम",
        "accent": (222, 157, 72),
        "bg": (72, 24, 18),
        "captions": [
            "राम नाम में मन को शांति मिले",
            "भक्ति की ज्योति हर हृदय में जले",
            "श्री राम का स्मरण जीवन को उजला करे",
            "हर सांस में राम, हर धड़कन में राम",
        ],
        "narration": "श्री राम। राम नाम में मन को शांति मिले। भक्ति की ज्योति हर हृदय में जले। श्री राम का स्मरण जीवन को उजला करे। हर सांस में राम, हर धड़कन में राम। श्री राम जय राम जय जय राम।",
    },
    {
        "slug": "krishna",
        "deity": "श्री कृष्ण",
        "title": "कृष्ण भक्ति की मधुर धुन",
        "mantra": "राधे कृष्ण, राधे कृष्ण",
        "accent": (86, 158, 218),
        "bg": (16, 39, 72),
        "captions": [
            "मुरली की मधुर धुन मन को छू जाए",
            "श्याम नाम से हर चिंता दूर हो जाए",
            "राधे कृष्ण की भक्ति मन में बस जाए",
            "हर पल प्रेम, हर पल कृष्ण स्मरण",
        ],
        "narration": "श्री कृष्ण। मुरली की मधुर धुन मन को छू जाए। श्याम नाम से हर चिंता दूर हो जाए। राधे कृष्ण की भक्ति मन में बस जाए। हर पल प्रेम, हर पल कृष्ण स्मरण। राधे कृष्ण, राधे कृष्ण।",
    },
    {
        "slug": "bhakti",
        "deity": "भक्ति संध्या",
        "title": "भक्ति की मधुर प्रार्थना",
        "mantra": "ॐ शांति शांति शांति",
        "accent": (198, 116, 68),
        "bg": (43, 24, 54),
        "captions": [
            "भक्ति में मन को ठहरने दो",
            "दीप की लौ में शांति को महसूस करो",
            "प्रार्थना के इन पलों को अपने नाम करो",
            "मन शांत हो, हृदय भक्ति से भर जाए",
        ],
        "narration": "भक्ति संध्या। भक्ति में मन को ठहरने दो। दीप की लौ में शांति को महसूस करो। प्रार्थना के इन पलों को अपने नाम करो। मन शांत हो, हृदय भक्ति से भर जाए। ॐ शांति शांति शांति।",
    },
]

KEYWORDS = {
    "ram": ["ram", "राम", "ayodhya", "अयोध्या", "sita", "सीता", "raghu", "रघुनाथ"],
    "krishna": ["krishna", "कृष्ण", "kanha", "कान्हा", "radha", "राधा", "janmashtami", "जन्माष्टमी", "vrindavan", "वृंदावन"],
    "bhakti": ["bhajan", "भजन", "aarti", "आरती", "mantra", "मंत्र", "bhakti", "भक्ति", "kirtan", "कीर्तन"],
}


def fetch_trends() -> list[dict]:
    for url in (
        "https://trends.google.com/trending/rss?geo=IN",
        "https://trends.google.co.in/trends/trendingsearches/daily/rss?geo=IN",
    ):
        try:
            req = Request(url, headers={"User-Agent": "BhajanAabha/3.0 (GitHub Actions)"})
            root = ET.fromstring(urlopen(req, timeout=20).read())
            items = []
            for item in root.findall(".//item"):
                title = html.unescape(item.findtext("title", default="").strip())
                if title:
                    items.append({
                        "title": title,
                        "traffic": item.findtext("{*}approx_traffic", default=""),
                        "source": url,
                    })
            if items:
                return items
        except Exception as exc:
            print(f"TREND_SOURCE_FAILED {url}: {exc}")
    return []


def choose_packs(trends: list[dict]) -> list[dict]:
    selected: list[dict] = []
    seen: set[str] = set()
    for trend in trends:
        text = trend["title"].lower()
        scored = []
        for p in PACKS:
            score = sum(1 for k in KEYWORDS[p["slug"]] if k.lower() in text)
            if score:
                scored.append((score, p))
        if scored:
            p = max(scored, key=lambda x: x[0])[1]
            if p["slug"] not in seen:
                selected.append(p)
                seen.add(p["slug"])
        if len(selected) >= MAX_VIDEOS:
            break
    for p in PACKS:
        if len(selected) >= MAX_VIDEOS:
            break
        if p["slug"] not in seen:
            selected.append(p)
            seen.add(p["slug"])
    return selected[:MAX_VIDEOS]


def font(size: int, bold: bool = False):
    names = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for path in names:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ensure_voice(text: str, path: Path, seconds: int) -> None:
    """Create actual spoken Hindi audio locally; never use a humming/sine tone."""
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        raise RuntimeError("espeak-ng is required for real narration but is not installed")
    # Keep speech comfortably within the video. The video is also allowed to
    # continue after the narration finishes, with a very low musical-free bed.
    subprocess.run([
        espeak, "-q", "-v", "hi", "-s", "142", "-p", "48", "-a", "125",
        "-w", str(path), text,
    ], check=True, timeout=60)
    if not path.exists() or path.stat().st_size < 2000:
        raise RuntimeError("Hindi narration was not generated")


def make_background(pack: dict, frame: int) -> Image.Image:
    """Generate a devotional animated background entirely with Pillow."""
    t = frame / FPS
    base = Image.new("RGB", (W, H), pack["bg"])
    px = base.load()
    r0, g0, b0 = pack["bg"]
    for y in range(H):
        v = int(22 * (1 - y / H))
        for x in range(0, W, 4):
            glow = int(18 * max(0.0, 1.0 - (((x - W * 0.5) / (W * 0.7)) ** 2 + ((y - H * 0.42) / (H * 0.75)) ** 2)))
            c = (min(255, r0 + v + glow), min(255, g0 + v + glow), min(255, b0 + v + glow))
            for xx in range(x, min(x + 4, W)):
                px[xx, y] = c

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    cx, cy = W // 2, int(H * 0.42)
    pulse = 1.0 + 0.06 * __import__("math").sin(t * 1.4)
    for radius in range(360, 40, -12):
        alpha = max(0, int(1.2 * (360 - radius)))
        gd.ellipse((cx - radius * pulse, cy - radius * pulse, cx + radius * pulse, cy + radius * pulse), fill=pack["accent"] + (min(70, alpha),))
    glow = glow.filter(ImageFilter.GaussianBlur(18))
    base = Image.alpha_composite(base.convert("RGBA"), glow)
    d = ImageDraw.Draw(base)

    # Mandala / sun motif.
    import math
    for ring, radius in enumerate((80, 120, 165, 215, 270)):
        pts = []
        for i in range(48):
            a = 2 * math.pi * i / 48 + t * (0.05 if ring % 2 else -0.04)
            rr = radius + 8 * math.sin(i * 3 + t)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.line(pts + [pts[0]], fill=pack["accent"] + (55,), width=2)

    # Temple silhouette along the bottom.
    ground = int(H * 0.77)
    d.rectangle((0, ground, W, H), fill=(8, 7, 12, 180))
    for x in range(-40, W + 80, 110):
        d.rectangle((x, ground - 105, x + 80, ground), fill=(12, 10, 15, 230))
        d.polygon([(x - 12, ground - 105), (x + 40, ground - 155), (x + 92, ground - 105)], fill=(12, 10, 15, 230))
    d.rectangle((W // 2 - 8, ground - 220, W // 2 + 8, ground - 105), fill=(12, 10, 15, 240))
    d.polygon([(W // 2 - 35, ground - 220), (W // 2, ground - 270), (W // 2 + 35, ground - 220)], fill=(12, 10, 15, 240))

    # Floating diyas / particles.
    for i in range(18):
        x = (i * 97 + frame * (1 + i % 3)) % W
        y = int(160 + ((i * 73 + frame * (2 + i % 2)) % 700))
        d.ellipse((x, y, x + 4, y + 4), fill=pack["accent"] + (100,))

    # Foreground diya.
    dx, dy = W // 2, int(H * 0.68)
    d.ellipse((dx - 65, dy, dx + 65, dy + 35), fill=(115, 64, 30, 240), outline=pack["accent"] + (230,), width=3)
    d.polygon([(dx - 24, dy + 5), (dx, dy - 70 - int(8 * abs(__import__("math").sin(t * 5)))), (dx + 24, dy + 5)], fill=pack["accent"] + (230,))
    d.ellipse((dx - 8, dy - 28, dx + 8, dy - 4), fill=(255, 235, 170, 245))
    return base.convert("RGB")


def make_video(path: Path, audio: Path, pack: dict, seconds: int) -> None:
    title_font = font(46, True)
    body_font = font(31, False)
    small_font = font(21, False)
    import math

    # Narration is mixed over silence. No humming, no synthetic tone.
    proc = subprocess.Popen([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", str(audio), "-t", str(seconds), "-map", "0:v:0", "-map", "1:a:0",
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-af", "apad", "-shortest",
        "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for n in range(seconds * FPS):
            t = n / FPS
            im = make_background(pack, n).convert("RGBA")
            d = ImageDraw.Draw(im)
            accent = pack["accent"]
            # Header band.
            d.rounded_rectangle((28, 28, W - 28, 138), radius=26, fill=(7, 7, 12, 205), outline=accent + (225,), width=2)
            d.text((W // 2, 48), "BHAJAN AABHA", font=small_font, anchor="ma", fill=(255, 240, 210, 255))
            d.text((W // 2, 79), pack["deity"], font=title_font, anchor="ma", fill=(255, 250, 235, 255))

            idx = min(len(pack["captions"]) - 1, int((t / seconds) * len(pack["captions"])))
            cap = pack["captions"][idx]
            # Center the caption in a stable two-line safe area.
            d.rounded_rectangle((34, H - 255, W - 34, H - 92), radius=24, fill=(6, 6, 10, 220), outline=accent + (150,), width=2)
            d.text((W // 2, H - 210), cap, font=body_font, anchor="ma", fill=(255, 255, 255, 255))
            d.text((W // 2, H - 126), pack["mantra"], font=small_font, anchor="ma", fill=accent + (255,))

            # Subtle camera movement.
            zoom = 1.0 + 0.025 * math.sin(t * 0.55)
            cw, ch = int(W / zoom), int(H / zoom)
            left = max(0, (W - cw) // 2)
            top = max(0, (H - ch) // 2)
            im = im.crop((left, top, left + cw, top + ch)).resize((W, H), Image.Resampling.LANCZOS)
            proc.stdin.write(im.convert("RGB").tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="ignore")
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed: {err[-2500:]}")
    finally:
        if proc.poll() is None:
            proc.kill()


def write_metadata(path: Path, pack: dict, video: Path) -> None:
    path.write_text(json.dumps({
        "title": pack["title"],
        "video": str(video),
        "audio": "locally generated Hindi narration using espeak-ng",
        "visuals": "procedurally generated devotional artwork; no external media downloads",
        "copyright_mode": "self-generated visuals + locally synthesized narration",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    # Never leave stale MP4s from an earlier failed architecture in the release.
    for old in VIDEOS.glob("*.mp4"):
        old.unlink()

    trends = fetch_trends()
    packs = choose_packs(trends)
    print(f"TREND_ITEMS={len(trends)} SELECTED={len(packs)}")
    for t in trends[:20]:
        print("TREND:", t["title"], t.get("traffic", ""))

    results = []
    for index, pack in enumerate(packs, 1):
        print(f"PREPARING {index}/{len(packs)}: {pack['deity']}")
        audio = OUT / f"{pack['slug']}_narration.wav"
        ensure_voice(pack["narration"], audio, SECONDS)
        video = VIDEOS / f"{datetime.now(timezone.utc):%Y%m%d}_{index}_{pack['slug']}.mp4"
        meta = OUT / f"{pack['slug']}.json"
        print(f"RENDERING {index}/{len(packs)}: {pack['title']}")
        make_video(video, audio, pack, SECONDS)
        write_metadata(meta, pack, video)
        audio.unlink(missing_ok=True)
        results.append({
            "topic": pack["deity"],
            "slug": pack["slug"],
            "video": str(video),
            "duration_sec": SECONDS,
            "mode": "zero_cost_self_generated_visuals_plus_hindi_narration",
        })

    state = {
        "channel": "Bhajan Aabha",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trend_source": "Google Trends RSS with deterministic fallback",
        "trend_count": len(trends),
        "videos": results,
        "copyright_mode": "self-generated visuals and locally synthesized Hindi narration",
        "gpu": False,
        "paid_services": False,
        "kaggle": False,
        "external_media_downloads": False,
        "human_intervention_after_setup": False,
        "quality_gate": "720x1280 + AAC + actual spoken Hindi narration + generated devotional visuals",
        "publish_status": "PENDING_YOUTUBE_FACEBOOK_AUTH",
    }
    (OUT / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"videos": results, "trends": trends[:20]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
