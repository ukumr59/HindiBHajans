from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(os.getenv("OUTPUT_DIR", "output"))
VIDEOS = OUT / "videos"
MAX_VIDEOS = max(1, min(3, int(os.getenv("MAX_VIDEOS", "3"))))
SECONDS = max(30, min(60, int(os.getenv("VIDEO_SECONDS", "45"))))
FPS, W, H = 8, 720, 1280

PACKS = [
    {"slug":"ram","deity":"श्री राम","title":"राम नाम की भक्ति","mantra":"श्री राम जय राम जय जय राम","bg":(72,24,18),"accent":(222,157,72),"captions":["राम नाम में मन को शांति मिले","भक्ति की ज्योति हर हृदय में जले","श्री राम का स्मरण जीवन को उजला करे","हर सांस में राम, हर धड़कन में राम"],"narration":"श्री राम। राम नाम में मन को शांति मिले। भक्ति की ज्योति हर हृदय में जले। श्री राम का स्मरण जीवन को उजला करे। हर सांस में राम, हर धड़कन में राम। श्री राम जय राम जय जय राम।"},
    {"slug":"krishna","deity":"श्री कृष्ण","title":"कृष्ण भक्ति की मधुर धुन","mantra":"राधे कृष्ण, राधे कृष्ण","bg":(16,39,72),"accent":(86,158,218),"captions":["मुरली की मधुर धुन मन को छू जाए","श्याम नाम से हर चिंता दूर हो जाए","राधे कृष्ण की भक्ति मन में बस जाए","हर पल प्रेम, हर पल कृष्ण स्मरण"],"narration":"श्री कृष्ण। मुरली की मधुर धुन मन को छू जाए। श्याम नाम से हर चिंता दूर हो जाए। राधे कृष्ण की भक्ति मन में बस जाए। हर पल प्रेम, हर पल कृष्ण स्मरण। राधे कृष्ण, राधे कृष्ण।"},
    {"slug":"bhakti","deity":"भक्ति संध्या","title":"भक्ति की मधुर प्रार्थना","mantra":"ॐ शांति शांति शांति","bg":(43,24,54),"accent":(198,116,68),"captions":["भक्ति में मन को ठहरने दो","दीप की लौ में शांति को महसूस करो","प्रार्थना के इन पलों को अपने नाम करो","मन शांत हो, हृदय भक्ति से भर जाए"],"narration":"भक्ति संध्या। भक्ति में मन को ठहरने दो। दीप की लौ में शांति को महसूस करो। प्रार्थना के इन पलों को अपने नाम करो। मन शांत हो, हृदय भक्ति से भर जाए। ॐ शांति शांति शांति।"},
]


def font(size, bold=False):
    names = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf" if bold else "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def find_espeak():
    for name in ("espeak-ng", "espeak"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("VOICE_FATAL: eSpeak executable missing")


def wav_stats(path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)
        channels = w.getnchannels()
        width = w.getsampwidth()
    return frames / max(rate, 1), len(raw), channels, width


def ensure_voice(text, path):
    exe = find_espeak()
    path.unlink(missing_ok=True)
    # IMPORTANT: use eSpeak's file output mode. The hosted runner in this
    # environment returned rc=0 but an empty stdout stream with --stdout.
    commands = [
        [exe, "-v", "hi", "-s", "138", "-p", "48", "-a", "150", "-w", str(path), text],
        [exe, "-v", "hi+f2", "-s", "138", "-p", "48", "-a", "150", "-w", str(path), text],
        [exe, "-v", "hi", "-s", "128", "-p", "45", "-a", "170", "-w", str(path), text],
    ]
    errors = []
    for cmd in commands:
        path.unlink(missing_ok=True)
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
            if path.exists() and path.stat().st_size > 1000:
                duration, size, channels, width = wav_stats(path)
                print(f"VOICE_TEST mode=file rc={r.returncode} bytes={size} duration={duration:.2f}s channels={channels} width={width}")
                if duration >= 2 and channels >= 1 and width >= 2:
                    print("VOICE_OK Hindi spoken WAV generated")
                    return
            errors.append(f"rc={r.returncode} file_bytes={path.stat().st_size if path.exists() else 0} stderr={r.stderr.decode(errors='ignore')[-300:]}")
        except Exception as exc:
            errors.append(repr(exc))
    raise RuntimeError("VOICE_FATAL: Hindi spoken WAV could not be generated: " + " | ".join(errors))


def make_frame(pack, n, deity_font, caption_font, small_font):
    t = n / FPS
    im = Image.new("RGB", (W, H), pack["bg"])
    d = ImageDraw.Draw(im, "RGBA")
    cx, cy = W // 2, int(H * 0.43)
    # Calm animated mandala / diya: fully procedural, no downloaded media.
    for rr in range(350, 55, -25):
        alpha = max(12, 78 - rr // 6)
        d.ellipse((cx-rr, cy-rr, cx+rr, cy+rr), outline=pack["accent"] + (alpha,), width=2)
    for i in range(32):
        a = t * 0.24 + i * math.pi / 16
        x = cx + int(math.cos(a) * 270)
        y = cy + int(math.sin(a) * 270)
        r = 3 + int(2 * (1 + math.sin(t * 1.7 + i)) / 2)
        d.ellipse((x-r, y-r, x+r, y+r), fill=pack["accent"] + (160,))
    d.ellipse((cx-118, cy+70, cx+118, cy+130), fill=(95, 48, 20, 245))
    d.polygon([(cx-18, cy+70), (cx, cy-5-int(8*math.sin(t*3))), (cx+18, cy+70)], fill=(255, 180, 35, 255))
    d.ellipse((cx-10, cy+28, cx+10, cy+70), fill=(255, 245, 185, 245))
    d.rounded_rectangle((24, 24, W-24, 150), radius=26, fill=(5,5,9,210), outline=pack["accent"]+(230,), width=2)
    d.text((W//2, 48), "BHAJAN AABHA", font=small_font, anchor="ma", fill=(255,240,210,255))
    d.text((W//2, 82), pack["deity"], font=deity_font, anchor="ma", fill=(255,250,235,255))
    idx = min(3, int(t / SECONDS * 4))
    y = H - 225
    d.rounded_rectangle((24, y-45, W-24, H-65), radius=25, fill=(5,5,9,225), outline=(255,255,255,45), width=1)
    d.text((W//2, y), pack["captions"][idx], font=caption_font, anchor="ma", fill=(255,255,255,255))
    d.text((W//2, H-42), pack["mantra"], font=small_font, anchor="ms", fill=pack["accent"]+(255,))
    return im.tobytes()


def render(video, audio, pack):
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-t", str(SECONDS),
        "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-af", "apad", "-t", str(SECONDS),
        "-movflags", "+faststart", str(video)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        df, cf, sf = font(42, True), font(30), font(20)
        for n in range(SECONDS * FPS):
            proc.stdin.write(make_frame(pack, n, df, cf, sf))
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        rc = proc.wait()
        if rc:
            raise RuntimeError("FFMPEG_FATAL: " + err[-1800:])
    finally:
        if proc.poll() is None:
            proc.kill()


def validate(video):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,width,height,duration", "-of", "json", str(video)
    ], text=True))
    vs = [s for s in data["streams"] if s.get("codec_type") == "video"]
    aa = [s for s in data["streams"] if s.get("codec_type") == "audio"]
    if not vs or not aa:
        raise RuntimeError("OUTPUT_FATAL: MP4 missing video or audio")
    if vs[0].get("width") != W or vs[0].get("height") != H:
        raise RuntimeError("OUTPUT_FATAL: wrong video dimensions")
    if aa[0].get("codec_name") != "aac":
        raise RuntimeError("OUTPUT_FATAL: wrong audio codec")
    if float(aa[0].get("duration", 0)) < 1:
        raise RuntimeError("OUTPUT_FATAL: audio duration missing")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    for old in VIDEOS.glob("*.mp4"):
        old.unlink()
    packs = PACKS[:MAX_VIDEOS]
    print("ARCHITECTURE=v4 ZERO_COST=true KAGGLE=false PAID_SERVICES=false")
    print(f"SELECTED={len(packs)}")
    results = []
    for i, pack in enumerate(packs, 1):
        print(f"PREPARING {i}/{len(packs)}: {pack['deity']}")
        audio = OUT / f"{pack['slug']}_narration.wav"
        ensure_voice(pack["narration"], audio)
        video = VIDEOS / f"{datetime.now(timezone.utc):%Y%m%d}_{i}_{pack['slug']}.mp4"
        render(video, audio, pack)
        validate(video)
        audio.unlink(missing_ok=True)
        print("VIDEO_OK", video)
        results.append({"topic": pack["deity"], "title": pack["title"], "video": str(video), "duration_sec": SECONDS})
    state = {
        "channel": "Bhajan Aabha",
        "architecture": "github-runner-only-v4",
        "videos": results,
        "paid_services": False,
        "paid_gpu": False,
        "kaggle": False,
        "external_media_downloads": False,
        "voice": "eSpeak Hindi file-mode WAV",
        "status": "READY_FOR_RELEASE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "run_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps({"videos": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
