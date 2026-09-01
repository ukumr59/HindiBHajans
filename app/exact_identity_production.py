from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "output"
WORK = OUT / "exact_identity_work"
AUDIO = OUT / "bhajan_source.mp3"
FINAL = OUT / "bhajan_aabha_exact_identity.mp4"

VIDEO_SECONDS = int(os.getenv("VIDEO_SECONDS", "180"))
SCENE_COUNT = VIDEO_SECONDS // 15
LIGHTNING_STUDIO = os.getenv("LIGHTNING_STUDIO", "bhajan-aabha-exact-identity").strip()
LIGHTNING_USER_ID = os.getenv("LIGHTNING_USER_ID", "").strip()
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY", "").strip()
LIGHTNING_USERNAME = os.getenv("LIGHTNING_USERNAME", "").strip()
LIGHTNING_ORG = os.getenv("LIGHTNING_ORG", "").strip()
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE", "").strip()
SINGER_REFERENCE_B64 = os.getenv("SINGER_REFERENCE_B64", "").strip()

LYRICS = """[Intro]\nश्री राम... श्री राम... जय जय राम...\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\n\n[Pre-Chorus]\nतेरे नाम की धुन बजे, हर धड़कन में आज\nतेरी कृपा से खिल उठे, जीवन का हर राज\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Instrumental Break]\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\n\n[Pre-Chorus]\nतेरी राह में चल पड़ूँ, मन में लेकर विश्वास\nराम नाम की शक्ति से, मिट जाए हर त्रास\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\n\n[Verse 3]\nअयोध्या के राजकुमार, करुणा के भंडार\nतेरे चरणों में मिल जाए, जीवन का सच्चा सार\n\n[Build]\nजय श्री राम की गूंज उठे, नभ से धरती तक\nढोल बजे और शंख बजे, प्रेम बहे हर पल\n\n[Final Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nजय जय राम... जय जय राम...\n\n[Outro]\nश्री राम... जय राम... जय जय राम..."""

MUSIC_CAPTION = """Modern high-energy Hindi devotional bhajan made like a current YouTube DJ devotional song, 128 BPM, 4/4, loud polished commercial stereo production, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, huge memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, modern synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents, harmonium texture, short instrumental intro, strong verse build, massive chorus/drop, rhythmic instrumental break, final chorus with layered backing vocals, professional YouTube/radio loudness and DJ playback energy. NOT meditation music, NOT sleepy, NOT ambient, NOT acoustic-only, NOT spoken narration, NOT humming, NOT a cappella, NOT instrumental-only."""


def run(*args: str, cwd: Path | None = None) -> None:
    print("RUN:", " ".join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def prepare_singer() -> Path:
    if not SINGER_REFERENCE_B64:
        raise RuntimeError("SINGER_REFERENCE_B64 secret is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(SINGER_REFERENCE_B64, validate=True)
    ref = WORK / "reference.png"
    WORK.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(raw)

    from PIL import Image, ImageEnhance
    image = Image.open(ref).convert("RGB")
    if image.size != (1024, 1536):
        raise RuntimeError(f"Expected approved 1024x1536 identity sheet, got {image.size}")
    w, h = image.size
    crop = image.crop((round(w * 35 / 1024), round(h * 85 / 1536), round(w * 245 / 1024), round(h * 505 / 1536)))
    crop = ImageEnhance.Brightness(crop).enhance(1.04)
    singer = OUT / "bhajan_aabha_locked_identity_source.png"
    crop.save(singer, format="PNG", optimize=True)
    print(f"LOCKED_SINGER={singer} SIZE={crop.size}", flush=True)
    return singer


def generate_audio_remote() -> Path:
    if not all([LIGHTNING_USER_ID, LIGHTNING_API_KEY, LIGHTNING_TEAMSPACE]):
        raise RuntimeError("Lightning secrets incomplete: LIGHTNING_USER_ID, LIGHTNING_API_KEY, LIGHTNING_TEAMSPACE required")
    if not LIGHTNING_USERNAME and not LIGHTNING_ORG:
        raise RuntimeError("Set LIGHTNING_USERNAME or LIGHTNING_ORG")
    if LIGHTNING_USERNAME and LIGHTNING_ORG:
        raise RuntimeError("Set only one of LIGHTNING_USERNAME or LIGHTNING_ORG")

    from lightning_sdk import Studio, Machine

    kwargs = {"name": LIGHTNING_STUDIO, "teamspace": LIGHTNING_TEAMSPACE, "create_ok": True}
    kwargs["org" if LIGHTNING_ORG else "user"] = LIGHTNING_ORG or LIGHTNING_USERNAME
    studio = Studio(**kwargs)
    started_here = False

    request = {
        "caption": MUSIC_CAPTION,
        "lyrics": LYRICS,
        "duration": VIDEO_SECONDS,
        "bpm": 128,
        "keyscale": "C Major",
        "timesignature": "4/4",
        "vocal_language": "hi",
    }
    request_path = WORK / "bhajan_request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    worker = ROOT / "app" / "lightning_ace_step_worker_v2.py"

    try:
        status = str(studio.status).lower()
        print(f"LIGHTNING status={status} studio={LIGHTNING_STUDIO}", flush=True)
        if "running" not in status:
            studio.start(Machine.T4)
            started_here = True
        elif "t4" not in str(studio.machine).lower():
            studio.switch_machine(Machine.T4)
            started_here = True

        studio.upload_file(str(worker), remote_path="lightning_ace_step_worker_v2.py")
        studio.upload_file(str(request_path), remote_path="bhajan_request.json")
        print("LIGHTNING_AUDIO_GENERATION_START", flush=True)
        output, code = studio.run_with_exit_code("python lightning_ace_step_worker_v2.py")
        print(output[-16000:], flush=True)
        if code != 0:
            raise RuntimeError(f"ACE-Step remote worker failed with code {code}")
        downloaded = studio.download_file("bhajan_aabha_worker/bhajan_source.mp3", str(AUDIO))
        path = Path(downloaded)
        if not path.exists() or path.stat().st_size < 100_000:
            raise RuntimeError("Generated bhajan MP3 is missing or too small")
        print(f"AUDIO_OK={path} BYTES={path.stat().st_size}", flush=True)
        return path
    finally:
        if started_here:
            def _stop() -> None:
                try:
                    studio.stop()
                except Exception as exc:
                    print(f"LIGHTNING_STOP_WARNING={exc}", flush=True)
            threading.Thread(target=_stop, daemon=True).start()


def assemble(singer: Path, audio: Path) -> Path:
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio)
    ], text=True).strip()
    duration = float(probe)
    if duration < VIDEO_SECONDS - 3:
        raise RuntimeError(f"Audio duration {duration:.2f}s is shorter than required {VIDEO_SECONDS}s")

    scenes = []
    chapter = VIDEO_SECONDS / SCENE_COUNT
    # Identity-safe stage: preserve the approved singer image exactly. Only
    # deterministic crop/zoom changes are applied; no face/body regeneration.
    filters = [
        "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.00065,1.10)':d={d}:s=720x1280:fps=24",
        "scale=780:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,zoompan=z='min(zoom+0.00055,1.07)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={d}:s=720x1280:fps=24",
        "scale=820:-1,crop=720:1280:x='(iw-720)/2':y='max(0,(ih-1280)*(t/{c}))',fps=24",
        "scale=820:-1,crop=720:1280:x='max(0,(iw-720)*(1-t/{c}))':y='(ih-1280)/2',fps=24",
    ]
    for i in range(SCENE_COUNT):
        scene = WORK / f"scene_{i+1:02d}.mp4"
        vf = filters[i % len(filters)].format(d=max(1, round(chapter * 24)), c=max(chapter, 0.001))
        run("ffmpeg", "-y", "-loop", "1", "-i", singer, "-t", f"{chapter:.3f}", "-vf", vf,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", scene)
        scenes.append(scene)

    concat = WORK / "concat.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in scenes), encoding="utf-8")
    visual = WORK / "visual.mp4"
    run("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat, "-c", "copy", visual)
    run("ffmpeg", "-y", "-i", visual, "-i", audio,
        "-map", "0:v:0", "-map", "1:a:0", "-t", str(VIDEO_SECONDS),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", FINAL)
    print(f"FINAL_VIDEO={FINAL} BYTES={FINAL.stat().st_size}", flush=True)
    return FINAL


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    singer = prepare_singer()
    audio = generate_audio_remote()
    final = assemble(singer, audio)
    manifest = {
        "status": "OK",
        "video": str(final),
        "audio": str(audio),
        "identity_source": str(singer),
        "duration_seconds": VIDEO_SECONDS,
        "scene_count": SCENE_COUNT,
        "identity_policy": "approved singer source pixels preserved; no generative face/body reconstruction",
        "lip_sync": False,
        "wardrobe_synthesis": False,
        "background_replacement": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EXACT_IDENTITY_PRODUCTION_OK", flush=True)


if __name__ == "__main__":
    main()
