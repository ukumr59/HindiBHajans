from __future__ import annotations

import html
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
ASSETS = OUT / "assets"
MAX_VIDEOS = max(1, min(4, int(os.getenv("MAX_VIDEOS", "4"))))
SECONDS = max(30, min(60, int(os.getenv("VIDEO_SECONDS", "45"))))
FPS = 8
W, H = 720, 1280

# Zero-cost content packs. Audio and visual assets are deliberately limited to
# public-domain/CC0 material. We do not download copyrighted YouTube recordings.
PACKS = [
    {
        "slug": "ram",
        "deity": "श्री राम",
        "title": "राम नाम की भक्ति",
        "mantra": "श्री राम जय राम जय जय राम",
        "audio": "Ramdwara ed f.ogg",
        "audio_source": "https://commons.wikimedia.org/wiki/File:Ramdwara_ed_f.ogg",
        "audio_license": "CC0",
        "images": [
            "Lord Rama Statue Maharishi Mahesh Yogi Ramayan University Ayodhya.jpg",
            "Rajarajesvara Temple in Thanjavur, India.jpg",
        ],
        "accent": (222, 157, 72),
        "captions": [
            "राम नाम में मन को शांति मिले",
            "भक्ति की ज्योति हर हृदय में जले",
            "श्री राम का स्मरण जीवन को उजला करे",
            "हर सांस में राम, हर धड़कन में राम",
        ],
    },
    {
        "slug": "krishna",
        "deity": "श्री कृष्ण",
        "title": "कृष्ण माधो राम नारायण",
        "mantra": "राधे कृष्ण",
        "audio": "Krishna Madho Ram Narayan.ogg",
        "audio_source": "https://commons.wikimedia.org/wiki/File:Krishna_Madho_Ram_Narayan.ogg",
        "audio_license": "Public domain",
        "images": [
            "The Hindu deity Krishna playing the flute.jpg",
            "Krishna Fluting, 13th-15th century AD, Eastern Ganga dynasty, Orissa, India - brass - Sackler Museum - DSC02449.JPG",
        ],
        "accent": (75, 153, 211),
        "captions": [
            "मुरली की मधुर धुन मन को छू जाए",
            "श्याम नाम से हर चिंता दूर हो जाए",
            "राधे कृष्ण की भक्ति मन में बस जाए",
            "हर पल प्रेम, हर पल कृष्ण स्मरण",
        ],
    },
    {
        "slug": "bhakti",
        "deity": "भक्ति संध्या",
        "title": "भक्ति की मधुर धुन",
        "mantra": "ॐ शांति शांति शांति",
        "audio": "Bhajana.ogg",
        "audio_source": "https://commons.wikimedia.org/wiki/File:Bhajana.ogg",
        "audio_license": "Public domain",
        "images": [
            "On the 🔝.jpg",
            "J.K. Temple.jpg",
        ],
        "accent": (198, 116, 68),
        "captions": [
            "भक्ति में मन को ठहरने दो",
            "दीप की लौ में शांति को महसूस करो",
            "प्रार्थना के इन पलों को अपने नाम करो",
            "मन शांत हो, हृदय भक्ति से भर जाए",
        ],
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
            req = Request(url, headers={"User-Agent": "BhajanAabha/2.0"})
            root = ET.fromstring(urlopen(req, timeout=20).read())
            items = []
            for item in root.findall(".//item"):
                title = html.unescape(item.findtext("title", default="").strip())
                if title:
                    items.append({"title": title, "traffic": item.findtext("{*}approx_traffic", default=""), "source": url})
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
                selected.append(p); seen.add(p["slug"])
        if len(selected) >= MAX_VIDEOS:
            break
    for p in PACKS:
        if len(selected) >= MAX_VIDEOS:
            break
        if p["slug"] not in seen:
            selected.append(p); seen.add(p["slug"])
    return selected[:MAX_VIDEOS]


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 4):
        try:
            req = Request(url, headers={"User-Agent": "BhajanAabha/2.1 (GitHub Actions; public-domain asset downloader)"})
            with urlopen(req, timeout=90) as r, path.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if path.stat().st_size < 1000:
                raise RuntimeError(f"Downloaded asset is unexpectedly small: {path}")
            return
        except Exception as exc:
            last_error = exc
            if path.exists():
                path.unlink()
            if attempt < 3:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Asset download failed after 3 attempts: {url}: {last_error}")


def commons_direct_url(filename: str) -> str:
    """Resolve a Commons file through MediaWiki's API, then use its canonical upload URL.

    This avoids the rate-limited Special:Redirect/file endpoint and also handles
    filenames whose actual upload path contains a hash-derived directory.
    """
    api = "https://commons.wikimedia.org/w/api.php?" + urlencode({
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "imageinfo",
        "iiprop": "url",
        "titles": "File:" + filename,
        "redirects": "1",
    })
    req = Request(api, headers={"User-Agent": "BhajanAabha/2.1 (GitHub Actions; MediaWiki API)"})
    with urlopen(req, timeout=30) as r:
        data = json.load(r)
    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0] or not pages[0].get("imageinfo"):
        raise RuntimeError(f"Wikimedia Commons file not found: {filename}")
    return pages[0]["imageinfo"][0]["url"]


def prepare_assets(pack: dict) -> tuple[Path, list[Path]]:
    pack_dir = ASSETS / pack["slug"]
    pack_dir.mkdir(parents=True, exist_ok=True)
    audio = pack_dir / "source.ogg"
    download(commons_direct_url(pack["audio"]), audio)
    images = []
    for i, name in enumerate(pack["images"], 1):
        p = pack_dir / f"image_{i}{Path(name).suffix.lower() or '.jpg'}"
        download(commons_direct_url(name), p)
        images.append(p)
    return audio, images


def fit_cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    im = im.convert("RGB")
    target_w, target_h = size
    scale = max(target_w / im.width, target_h / im.height)
    nw, nh = int(im.width * scale), int(im.height * scale)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw - target_w) // 2, (nh - target_h) // 2
    return im.crop((left, top, left + target_w, top + target_h))


def make_video(path: Path, audio: Path, images: list[Path], pack: dict, seconds: int) -> None:
    title_font = font(42, True)
    body_font = font(30, False)
    small_font = font(20, False)
    loaded = [fit_cover(Image.open(p), (W, H)) for p in images]
    # Make two cinematic variants from each still, rather than cartoon geometry.
    variants = []
    for im in loaded:
        variants.append(im.filter(ImageFilter.GaussianBlur(0.8)))
        variants.append(ImageEnhance.Color(im).enhance(1.08))

    proc = subprocess.Popen([
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", str(audio), "-t", str(seconds), "-map", "0:v:0", "-map", "1:a:0",
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-shortest", "-movflags", "+faststart", str(path)
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for n in range(seconds * FPS):
            t = n / FPS
            phase = (t / max(1.0, seconds)) * len(variants)
            idx = min(len(variants) - 1, int(phase))
            im = variants[idx].copy()
            local = (phase - int(phase))
            zoom = 1.0 + 0.10 * local
            crop_w, crop_h = int(W / zoom), int(H / zoom)
            pan_x = int((im.width - crop_w) * (0.35 + 0.30 * local))
            pan_y = int((im.height - crop_h) * (0.45 + 0.10 * local))
            im = im.crop((pan_x, pan_y, pan_x + crop_w, pan_y + crop_h)).resize((W, H), Image.Resampling.LANCZOS)
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            for y in range(H):
                alpha = int(15 + 155 * max(0, (y - H * 0.58) / (H * 0.42)))
                od.line((0, y, W, y), fill=(0, 0, 0, min(180, alpha)))
            im = Image.alpha_composite(im.convert("RGBA"), overlay)
            d = ImageDraw.Draw(im)
            accent = pack["accent"]
            d.rounded_rectangle((30, 30, W - 30, 125), radius=24, fill=(8, 8, 12, 185), outline=accent + (220,), width=2)
            d.text((W // 2, 52), "BHAJAN AABHA", font=small_font, anchor="ma", fill=(255, 240, 210, 255))
            d.text((W // 2, 80), pack["deity"], font=title_font, anchor="ma", fill=(255, 250, 235, 255))
            cap = pack["captions"][min(len(pack["captions"]) - 1, int(t / seconds * len(pack["captions"]))) ]
            wrapped = [cap] if len(cap) < 34 else [cap[:34], cap[34:]]
            y = H - 220
            d.rounded_rectangle((30, y - 35, W - 30, H - 72), radius=24, fill=(8, 8, 12, 205))
            for line in wrapped[:2]:
                d.text((W // 2, y), line, font=body_font, anchor="ma", fill=(255, 255, 255, 255)); y += 40
            d.text((W // 2, H - 48), pack["mantra"], font=small_font, anchor="ms", fill=accent + (255,))
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
        "audio_source": pack["audio_source"],
        "audio_license": pack["audio_license"],
        "visual_sources": ["https://commons.wikimedia.org/wiki/File:" + quote(x, safe="") for x in pack["images"]],
        "video": str(video),
        "note": "Audio is reused only from the listed public-domain/CC0 source. Visuals are CC0 sources with cinematic motion treatment."
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); VIDEOS.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(parents=True, exist_ok=True)
    trends = fetch_trends(); packs = choose_packs(trends)
    print(f"TREND_ITEMS={len(trends)} SELECTED={len(packs)}")
    for t in trends[:20]: print("TREND:", t["title"], t.get("traffic", ""))

    results = []
    for index, pack in enumerate(packs, 1):
        print(f"PREPARING {index}/{len(packs)}: {pack['deity']}")
        audio, images = prepare_assets(pack)
        video = VIDEOS / f"{datetime.now(timezone.utc):%Y%m%d}_{index}_{pack['slug']}.mp4"
        meta = OUT / f"{pack['slug']}.json"
        print(f"RENDERING {index}/{len(packs)}: {pack['title']}")
        make_video(video, audio, images, pack, SECONDS)
        write_metadata(meta, pack, video)
        results.append({
            "topic": pack["deity"], "slug": pack["slug"], "video": str(video),
            "duration_sec": SECONDS, "audio_license": pack["audio_license"],
            "mode": "zero_cost_licensed_audio_plus_cc0_visuals"
        })

    state = {
        "channel": "Bhajan Aabha",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trend_source": "Google Trends RSS with deterministic zero-cost fallback",
        "trend_count": len(trends), "videos": results,
        "copyright_mode": "only public-domain/CC0 audio and visuals are used by the automated packs",
        "gpu": False, "paid_services": False, "kaggle": False,
        "human_intervention_after_setup": False,
        "quality_gate": "real audio + photographic/archival visuals + 720x1280 + AAC; no synthetic sine-wave humming",
        "publish_status": "PENDING_YOUTUBE_FACEBOOK_AUTH",
    }
    (OUT / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"videos": results, "trends": trends[:20]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
