from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# When this file is launched as `python app/exact_identity_production.py`,
# Python places /app on sys.path rather than the repository root. Add the
# repository root explicitly so sibling app modules can be imported reliably.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageEnhance

OUT = ROOT / "output"
WORK = OUT / "exact_identity_work"
AUDIO = OUT / "bhajan_source.mp3"
FINAL = OUT / "bhajan_aabha_exact_identity.mp4"
REFERENCE = ROOT / "assets" / "uks model image.png"

VIDEO_SECONDS = int(os.getenv("VIDEO_SECONDS", "180"))
SCENE_SECONDS = 15
SCENE_COUNT = VIDEO_SECONDS // SCENE_SECONDS

LYRICS = """[Intro]
श्री राम... श्री राम... जय जय राम...

[Verse 1]
मन में बसो रघुनंदन, चरणों में मेरा ध्यान
राम नाम की ज्योति जले, रोशन हो हर प्राण

[Pre-Chorus]
तेरे नाम की धुन बजे, हर धड़कन में आज
तेरी कृपा से खिल उठे, जीवन का हर राज

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम

[Instrumental Break]

[Verse 2]
दुख की घड़ी में साथ दो, हे दीनदयाल भगवान
तेरा नाम ही आसरा, तेरा नाम ही सम्मान

[Pre-Chorus]
तेरी राह में चल पड़ूँ, मन में लेकर विश्वास
राम नाम की शक्ति से, मिट जाए हर त्रास

[Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम

[Verse 3]
अयोध्या के राजकुमार, करुणा के भंडार
तेरे चरणों में मिल जाए, जीवन का सच्चा सार

[Build]
जय श्री राम की गूंज उठे, नभ से धरती तक
ढोल बजे और शंख बजे, प्रेम बहे हर पल

[Final Chorus]
श्री राम जय राम, जय जय राम
मेरे मन के दीप में, बसते श्री राम
श्री राम जय राम, जय जय राम
जय जय राम... जय जय राम...

[Outro]
श्री राम... जय राम... जय जय राम..."""

MUSIC_CAPTION = """Modern high-energy Hindi devotional bhajan made like a current YouTube DJ devotional song, 128 BPM, 4/4, polished commercial stereo production, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents, harmonium texture, short instrumental intro, strong verse build, massive chorus, rhythmic instrumental break, final chorus with layered backing vocals. NOT meditation music, NOT sleepy, NOT ambient, NOT acoustic-only, NOT spoken narration, NOT humming, NOT a cappella, NOT instrumental-only."""


def run(*args: str) -> None:
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], check=True)


def prepare_singer() -> Path:
    if not REFERENCE.exists():
        raise RuntimeError(f"Approved singer reference missing: {REFERENCE}")

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    image = Image.open(REFERENCE).convert("RGB")
    print(f"REFERENCE={REFERENCE}", flush=True)
    print(f"REFERENCE_SIZE={image.size}", flush=True)
    if image.size != (1536, 1024):
        raise RuntimeError(f"Expected original 1536x1024 identity sheet, got {image.size}")

    # Approved master portrait area from the original identity sheet.
    crop = image.crop((35, 85, 245, 505))
    crop = ImageEnhance.Brightness(crop).enhance(1.04)

    singer = OUT / "bhajan_aabha_locked_identity_source.png"
    crop.save(singer, format="PNG", optimize=True)
    print(f"LOCKED_SINGER={singer} SIZE={crop.size}", flush=True)
    return singer


def generate_audio_zerogpu() -> Path:
    """Generate the bhajan through the existing official ACE-Step HF ZeroGPU Space.

    No Lightning AI calls are made here.
    """
    import app.zero_cost_pipeline_v5 as base
    import app.zero_cost_pipeline_v5_2 as music

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN repository secret is missing")

    base.VIDEO_SECONDS = VIDEO_SECONDS
    base.PACK["lyrics"] = LYRICS
    base.PACK["music_prompt"] = MUSIC_CAPTION
    base.AUDIO.mkdir(parents=True, exist_ok=True)

    print("MUSIC_BACKEND=ACE-Step v1.5 official Hugging Face ZeroGPU Space", flush=True)
    print("LIGHTNING_AI=DISABLED", flush=True)
    print(f"MUSIC_DURATION={VIDEO_SECONDS}s", flush=True)

    generated = Path(music.generate_music_gradio())
    if not generated.exists() or generated.stat().st_size < 20_000:
        raise RuntimeError(f"Generated audio missing or suspiciously small: {generated}")

    shutil.copy2(generated, AUDIO)

    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(AUDIO)
    ], text=True).strip())
    if duration < VIDEO_SECONDS - 3:
        raise RuntimeError(f"MUSIC_FATAL: ACE-Step returned {duration:.2f}s; required >= {VIDEO_SECONDS - 3}s")

    print(f"AUDIO_OK={AUDIO} DURATION={duration:.2f}s", flush=True)
    return AUDIO


def assemble(singer: Path, audio: Path) -> Path:
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio)
    ], text=True).strip())
    if duration < VIDEO_SECONDS - 3:
        raise RuntimeError(f"Audio duration {duration:.2f}s is shorter than required {VIDEO_SECONDS}s")

    scenes: list[Path] = []
    chapter = VIDEO_SECONDS / SCENE_COUNT
    filters = [
        "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.00065,1.10)':d={d}:s=720x1280:fps=24",
        "scale=780:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.00055,1.07)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={d}:s=720x1280:fps=24",
        "scale=820:-1,crop=720:1280:x='(iw-720)/2':y='max(0,(ih-1280)*(t/{c}))',fps=24",
        "scale=820:-1,crop=720:1280:x='max(0,(iw-720)*(1-t/{c}))':y='(ih-1280)/2',fps=24",
    ]

    for i in range(SCENE_COUNT):
        scene = WORK / f"scene_{i + 1:02d}.mp4"
        vf = filters[i % len(filters)].format(d=max(1, round(chapter * 24)), c=max(chapter, 0.001))
        run(
            "ffmpeg", "-y", "-loop", "1", "-i", str(singer),
            "-t", f"{chapter:.3f}", "-vf", vf,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(scene),
        )
        scenes.append(scene)

    concat = WORK / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in scenes), encoding="utf-8")
    visual = WORK / "visual.mp4"
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(visual))
    run(
        "ffmpeg", "-y", "-i", str(visual), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-t", str(VIDEO_SECONDS),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(FINAL),
    )
    print(f"FINAL_VIDEO={FINAL} BYTES={FINAL.stat().st_size}", flush=True)
    return FINAL


def main() -> None:
    if not 180 <= VIDEO_SECONDS <= 300 or VIDEO_SECONDS % SCENE_SECONDS:
        raise RuntimeError("VIDEO_SECONDS must be 180-300 and divisible by 15")

    print("EXACT_IDENTITY_PRODUCTION_START", flush=True)
    print("LIGHTNING_AI=DISABLED", flush=True)

    singer = prepare_singer()
    audio = generate_audio_zerogpu()
    final = assemble(singer, audio)

    manifest = {
        "status": "OK",
        "video": str(final),
        "audio": str(audio),
        "identity_source": str(singer),
        "identity_reference": str(REFERENCE),
        "duration_seconds": VIDEO_SECONDS,
        "scene_count": SCENE_COUNT,
        "music_backend": "ACE-Step v1.5 official Hugging Face ZeroGPU Space",
        "identity_policy": "approved singer source pixels preserved; no generative face/body reconstruction",
        "lightning_ai": False,
        "lip_sync": False,
        "wardrobe_synthesis": False,
        "background_replacement": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EXACT_IDENTITY_PRODUCTION_OK", flush=True)


if __name__ == "__main__":
    main()
