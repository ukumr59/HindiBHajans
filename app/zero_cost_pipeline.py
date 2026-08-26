from __future__ import annotations

import html
import math
import os
import re
import subprocess
import wave
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

OUT = Path(os.getenv('OUTPUT_DIR', 'output'))
VIDEOS = OUT / 'videos'
MAX_VIDEOS = max(1, min(4, int(os.getenv('MAX_VIDEOS', '4'))))
SECONDS = max(15, min(45, int(os.getenv('VIDEO_SECONDS', '30'))))
FPS = 12
W, H = 540, 960

# Original, non-recording-based devotional concepts. Trend discovery only decides
# which themes to make; no copyrighted audio/video is downloaded or modified.
FALLBACKS = [
    ('हनुमान जी', 'hanuman', 'गदा और दीप की दिव्य ज्योति', 'ॐ हनुमते नमः', (232, 110, 55)),
    ('महादेव', 'mahadev', 'हिमालय में भोलेनाथ की शांति', 'ॐ नमः शिवाय', (100, 150, 205)),
    ('श्री कृष्ण', 'krishna', 'यमुना तट की मधुर भक्ति', 'राधे कृष्ण', (70, 150, 205)),
    ('श्री राम', 'ram', 'राम नाम की उजली ज्योति', 'श्री राम जय राम', (205, 155, 70)),
    ('माता रानी', 'mata', 'मां के चरणों में भक्ति', 'जय माता दी', (190, 80, 120)),
    ('गणेश जी', 'ganesh', 'विघ्नहर्ता की मंगल आरती', 'ॐ गं गणपतये नमः', (205, 120, 75)),
]
KEYWORDS = {
    'hanuman': ['hanuman', 'हनुमान', 'bajrang', 'बजरंग', 'sankat', 'संकटमोचन', 'sundarkand', 'सुंदरकांड', 'chalisa', 'चालीसा'],
    'mahadev': ['mahadev', 'महादेव', 'shiv', 'शिव', 'bholenath', 'भोलेनाथ', 'somnath', 'kedarnath', 'केदारनाथ', 'sawan', 'सावन'],
    'krishna': ['krishna', 'कृष्ण', 'kanha', 'कान्हा', 'radha', 'राधा', 'janmashtami', 'जन्माष्टमी', 'vrindavan', 'वृंदावन'],
    'ram': ['ram', 'राम', 'ayodhya', 'अयोध्या', 'sita', 'सीता', 'raghu', 'raghunath', 'रघुनाथ'],
    'mata': ['durga', 'दुर्गा', 'mata', 'माता', 'navratri', 'नवरात्रि', 'vaishno', 'वैष्णो', 'ambe', 'अंबे'],
    'ganesh': ['ganesh', 'गणेश', 'ganpati', 'गणपति', 'vinayak', 'विनायक'],
    'bhajan': ['bhajan', 'भजन', 'aarti', 'आरती', 'mantra', 'मंत्र', 'bhakti', 'भक्ति', 'kirtan', 'कीर्तन'],
}


def fetch_trends() -> list[dict]:
    urls = [
        'https://trends.google.com/trending/rss?geo=IN',
        'https://trends.google.co.in/trends/trendingsearches/daily/rss?geo=IN',
    ]
    for url in urls:
        try:
            req = Request(url, headers={'User-Agent': 'BhajanAabha/1.0'})
            data = urlopen(req, timeout=20).read()
            root = ET.fromstring(data)
            items = []
            for item in root.findall('.//item'):
                title = ''.join(item.findtext('title', default='').split())
                if title:
                    traffic = item.findtext('{*}approx_traffic', default='')
                    items.append({'title': html.unescape(title), 'traffic': traffic, 'source': url})
            if items:
                return items
        except Exception as exc:
            print(f'TREND_SOURCE_FAILED {url}: {exc}')
    return []


def choose_topics(trends: list[dict]) -> list[tuple]:
    ranked: list[tuple[int, tuple]] = []
    for trend in trends:
        text = trend['title'].lower()
        for topic in FALLBACKS:
            score = sum(1 for k in KEYWORDS[topic[1]] if k.lower() in text)
            if score:
                ranked.append((score, topic))
                break
        if any(k.lower() in text for k in KEYWORDS['bhajan']):
            # A generic bhajan trend is useful but doesn't identify a deity; use the
            # current devotional rotation so we still create an original concept.
            ranked.append((1, FALLBACKS[len(ranked) % len(FALLBACKS)]))
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = []
    seen = set()
    for _, topic in ranked:
        if topic[1] not in seen:
            selected.append(topic); seen.add(topic[1])
        if len(selected) >= MAX_VIDEOS:
            break
    # Always produce at least one video, then fill up to MAX_VIDEOS from the
    # deterministic rotation. This guarantees a zero-cost output even if trends fail.
    for topic in FALLBACKS:
        if len(selected) >= MAX_VIDEOS:
            break
        if topic[1] not in seen:
            selected.append(topic); seen.add(topic[1])
    return selected[:MAX_VIDEOS]


def font(size: int, bold: bool = False):
    candidates = [
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf' if bold else '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.ttf' if bold else '/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf',
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap(text: str, fnt, width: int) -> list[str]:
    words = text.split()
    lines, cur = [], ''
    for word in words:
        test = f'{cur} {word}'.strip()
        if fnt.getbbox(test)[2] <= width:
            cur = test
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    return lines


def lyrics(topic: tuple) -> list[str]:
    deity, _, _, mantra, _ = topic
    return [
        f'जय {deity}, मन में तेरी ज्योति जले',
        f'तेरा नाम जो ले, उसके मन में आशा पले',
        f'भक्ति की राह में, हर कदम तेरा साथ मिले',
        f'दुख की रात ढले, नई सुबह मुस्कान खिले',
        mantra,
        f'तेरी कृपा से {deity} जीवन पावन हो',
        f'हर सांस में भक्ति, हर धड़कन में तेरा नाम हो',
        f'जय {deity}, मन में तेरी ज्योति जले',
    ]


def make_audio(path: Path, topic: tuple, seconds: int) -> None:
    # Pure-Python original devotional instrumental: drone + melody + bell/percussion.
    # No sampled recording is used.
    sr = 16000
    total = sr * seconds
    deity, _, _, _, _ = topic
    roots = {'हनुमान जी': 220.0, 'महादेव': 196.0, 'श्री कृष्ण': 246.94, 'श्री राम': 261.63, 'माता रानी': 220.0, 'गणेश जी': 233.08}
    root = roots.get(deity, 220.0)
    scale = [0, 2, 4, 7, 9, 12, 14]
    bpm = 78
    beat = sr * 60 / bpm
    frames = bytearray()
    for i in range(total):
        t = i / sr
        beat_index = int(i / beat)
        note = scale[(beat_index // 2) % len(scale)]
        freq = root * (2 ** (note / 12))
        env = min(1.0, t * 4) * min(1.0, (seconds - t) * 4)
        drone = 0.10 * math.sin(2 * math.pi * root * t) + 0.06 * math.sin(2 * math.pi * root * 2 * t)
        melody = 0.16 * math.sin(2 * math.pi * freq * t) * (0.65 + 0.35 * math.sin(2 * math.pi * 2 * t) ** 2)
        bell = 0.0
        if i % int(sr * 2.0) < int(sr * 0.08):
            bt = (i % int(sr * 2.0)) / sr
            bell = 0.13 * math.sin(2 * math.pi * root * 4 * bt) * math.exp(-22 * bt)
        pulse = 0.0
        if i % int(beat) < int(sr * 0.035):
            pt = (i % int(beat)) / sr
            pulse = 0.10 * math.sin(2 * math.pi * 95 * pt) * math.exp(-45 * pt)
        sample = max(-0.8, min(0.8, (drone + melody + bell + pulse) * env))
        val = int(sample * 32767)
        frames += int(val).to_bytes(2, 'little', signed=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(frames)


def make_video(path: Path, audio: Path, topic: tuple, lines: list[str], seconds: int) -> None:
    deity, slug, visual, mantra, accent = topic
    f_title = font(34, True); f_body = font(25, False); f_small = font(19, False)
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-i', str(audio), '-t', str(seconds), '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(path)
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for n in range(seconds * FPS):
            t = n / FPS
            im = Image.new('RGB', (W, H), (12, 10, 28)); d = ImageDraw.Draw(im)
            # Moving radial devotional glow.
            cx = int(W * (0.50 + 0.08 * math.sin(t / 3.0)))
            cy = int(H * (0.38 + 0.04 * math.cos(t / 2.5)))
            for r in range(230, 10, -12):
                a = (230 - r) / 230
                col = tuple(int((12, 10, 28)[j] * (1-a) + accent[j] * a * 0.45) for j in range(3))
                d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=col)
            # Temple silhouette and animated lamps.
            base = int(H * 0.69)
            d.rectangle((70, base, W-70, H-90), fill=(28, 23, 38))
            d.polygon([(100, base), (W//2, base-120), (W-100, base)], fill=(39, 30, 48))
            d.rectangle((W//2-70, base-75, W//2+70, base), fill=(55, 42, 60))
            for x in (150, W//2, W-150):
                flick = 3 * math.sin(t * 7 + x)
                d.ellipse((x-10, base-35+flick, x+10, base-5+flick), fill=(255, 188, 70))
                d.ellipse((x-4, base-42+flick, x+4, base-22+flick), fill=(255, 240, 160))
            # Symbolic devotional emblem, not a copied recording/image.
            ey = int(H * 0.39)
            if slug == 'hanuman':
                d.ellipse((cx-48, ey-48, cx+48, ey+48), outline=(245, 210, 160), width=6); d.line((cx-35, ey+35, cx+40, ey-40), fill=(245,210,160), width=8)
            elif slug == 'mahadev':
                d.line((cx, ey-65, cx, ey+65), fill=(220,220,235), width=8); d.line((cx-38, ey-20, cx, ey-55, cx+38, ey-20), fill=(220,220,235), width=7)
            elif slug == 'krishna':
                d.arc((cx-55, ey-35, cx+55, ey+35), 200, 340, fill=(245,225,150), width=7); d.ellipse((cx+25, ey-70, cx+43, ey-52), fill=(70,180,100))
            elif slug == 'ram':
                d.arc((cx-55, ey-60, cx+55, ey+60), 200, 340, fill=(245,225,150), width=7); d.line((cx-35, ey, cx+45, ey-5), fill=(245,225,150), width=6)
            else:
                d.ellipse((cx-50, ey-50, cx+50, ey+50), outline=(245,210,160), width=6); d.line((cx-30,ey+10,cx+30,ey+10), fill=(245,210,160), width=6)
            # Text/caption band.
            d.rounded_rectangle((35, 45, W-35, 155), radius=24, fill=(0,0,0), outline=accent, width=2)
            d.text((W//2, 72), deity, font=f_title, anchor='ma', fill=(255,245,220))
            d.text((W//2, 118), mantra, font=f_small, anchor='ma', fill=(240,215,170))
            idx = min(len(lines)-1, int(t / seconds * len(lines)))
            caption = lines[idx]
            wrapped = wrap(caption, f_body, W-80)
            y = H-185
            d.rounded_rectangle((30, y-18, W-30, H-35), radius=22, fill=(0,0,0))
            for line in wrapped[:2]:
                d.text((W//2, y), line, font=f_body, anchor='ma', fill=(255,255,255)); y += 34
            d.text((W//2, H-25), 'Bhajan Aabha', font=f_small, anchor='ms', fill=(220,210,190))
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode('utf-8', errors='ignore')
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f'ffmpeg failed: {err[-1500:]}')
    finally:
        if proc.poll() is None:
            proc.kill()


def write_srt(path: Path, lines: list[str], seconds: int) -> None:
    step = seconds / len(lines)
    def ts(v: float) -> str:
        ms = int(round((v - int(v)) * 1000)); s = int(v) % 60; m = int(v) // 60
        return f'00:{m:02d}:{s:02d},{ms:03d}'
    with path.open('w', encoding='utf-8') as f:
        for i, line in enumerate(lines, 1):
            f.write(f'{i}\n{ts((i-1)*step)} --> {ts(min(seconds, i*step))}\n{line}\n\n')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); VIDEOS.mkdir(parents=True, exist_ok=True)
    trends = fetch_trends()
    topics = choose_topics(trends)
    print(f'TREND_ITEMS={len(trends)} SELECTED={len(topics)}')
    for t in trends[:20]: print('TREND:', t['title'], t.get('traffic',''))
    results = []
    for index, topic in enumerate(topics, 1):
        deity, slug, _, _, _ = topic
        lines = lyrics(topic)
        audio = OUT / f'{slug}.wav'
        video = VIDEOS / f'{datetime.now(timezone.utc):%Y%m%d}_{index}_{slug}.mp4'
        srt = OUT / f'{slug}.srt'
        print(f'GENERATING {index}/{len(topics)}: {deity}')
        make_audio(audio, topic, SECONDS)
        make_video(video, audio, topic, lines, SECONDS)
        write_srt(srt, lines, SECONDS)
        audio.unlink(missing_ok=True)
        results.append({'topic': deity, 'slug': slug, 'video': str(video), 'duration_sec': SECONDS, 'mode': 'zero_cost_cpu_original'})
    state = {
        'channel': 'Bhajan Aabha',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'trend_source': 'Google Trends RSS with deterministic devotional fallback',
        'trend_count': len(trends),
        'videos': results,
        'copyright_mode': 'original generation only; no copyrighted recordings or footage downloaded or modified',
        'gpu': False,
        'paid_services': False,
        'human_intervention_after_setup': False,
        'publish_status': 'PENDING_YOUTUBE_FACEBOOK_AUTH',
    }
    (OUT/'run_state.json').write_text(__import__('json').dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'manifest.json').write_text(__import__('json').dumps({'videos': results, 'trends': trends[:20]}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(__import__('json').dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
