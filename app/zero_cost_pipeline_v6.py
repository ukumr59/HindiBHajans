from __future__ import annotations

import json
import subprocess
from pathlib import Path

import app.zero_cost_pipeline_v5 as base
import app.zero_cost_pipeline_v5_2_fix as creative

# Production contract: every video is a complete 4-minute long-form song/video.
base.VIDEO_SECONDS = 240
base.SCENE_SECONDS = 30

base.PACK["lyrics"] = """[Intro]\nजय श्री राम... जय श्री राम...\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\n\n[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण\nअयोध्या के राज दुलारे, करुणा के भगवान\nतेरे चरणों में मिल जाए, जीवन को सम्मान\n\n[Pre-Chorus]\nजब भी मन घबराए, तेरा नाम पुकारूं\nअंधियारे में प्रभु मेरे, तेरा दीपक धारूं\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nतेरा नाम ही मेरा धाम\n\n[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान\nवन के पथ पर साथ रहे, भक्तों के रखवाल\nजो भी तेरी शरण में आए, कर दो उसका उद्धार\n\n[Pre-Chorus]\nहर धड़कन में राम तुम्हारा, हर सांस में तेरा नाम\nतेरी कृपा से जगमग हो, मेरा छोटा सा धाम\n\n[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nतेरा नाम ही मेरा धाम\n\n[Bridge / Drop]\nराम नाम की गूंज उठे, नभ से धरती तक\nभक्ति की यह ज्योति जले, मन से जीवन तक\nजय जय श्री राम... जय जय श्री राम...\n\n[Final Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nतेरा नाम ही मेरा धाम\nश्री राम जय राम, जय जय राम\nहर दिल बोले सुबह-शाम\n\n[Outro]\nश्री राम... जय राम... जय जय राम... जय श्री राम..."""

creative.DJ_MUSIC_PROMPT = """Complete 4-minute modern high-energy Hindi devotional DJ bhajan, 128 BPM, 4/4, loud polished commercial stereo production. Powerful expressive Hindi male lead vocal clearly singing the supplied lyrics with natural emotion and clean pronunciation. Catchy devotional melody, memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, modern synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents and harmonium texture. Structure as a genuine full song: DJ intro, verse, pre-chorus build, big chorus/drop, second verse, second build, chorus, instrumental bridge/drop, final extended chorus and clean DJ outro. Musical arrangement must evolve across the entire four minutes; do not make a 30-second loop or short clip stretched to four minutes. Strong bass and drums, clear upfront vocals, wide stereo image, professional loud clean master. NOT meditation, NOT sleepy, NOT ambient, NOT spoken narration, NOT humming, NOT a cappella, NOT acoustic-only, NOT instrumental-only."""

base.PACK["scene_prompts"] = [
    "30-second cinematic devotional opening: Lord Rama in a magnificent Ayodhya-inspired temple courtyard at dawn, slow push-in, warm rays, flickering diyas, drifting incense, gently moving garments and flower petals, serene sacred atmosphere, realistic natural motion, preserve deity identity, face and hands exactly.",
    "30-second cinematic devotional chorus sequence: camera moves laterally and rises around Lord Rama, brighter golden light, temple lamps, floating petals, subtle jewelry and cloth motion, stronger visual energy matching an EDM chorus/drop, realistic motion, preserve the same deity identity and facial details.",
    "30-second cinematic devotional climax sequence: camera rises from foreground diyas toward Lord Rama's peaceful face as sunrise intensifies, divine aura, drifting petals, richer golden light, emotional devotional climax, gentle push-in, realistic motion, preserve the same deity identity and facial details.",
]


def stamp(sec: int) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02},000"


def make_long_srt(path: Path):
    blocks = [
        (0, 12, "जय श्री राम... जय श्री राम..."),
        (12, 28, "मन में बसो रघुनंदन, चरणों में मेरा ध्यान"),
        (28, 44, "राम नाम की ज्योति जले, रोशन हो हर प्राण"),
        (44, 60, "अयोध्या के राज दुलारे, करुणा के भगवान"),
        (60, 78, "तेरे चरणों में मिल जाए, जीवन को सम्मान"),
        (78, 94, "जब भी मन घबराए, तेरा नाम पुकारूं"),
        (94, 110, "अंधियारे में प्रभु मेरे, तेरा दीपक धारूं"),
        (110, 128, "श्री राम जय राम, जय जय राम"),
        (128, 146, "मेरे मन के दीप में, बसते श्री राम"),
        (146, 164, "दुख की घड़ी में साथ दो, हे दीनदयाल भगवान"),
        (164, 182, "तेरा नाम ही आसरा, तेरा नाम ही सम्मान"),
        (182, 200, "वन के पथ पर साथ रहे, भक्तों के रखवाल"),
        (200, 218, "जो भी तेरी शरण में आए, कर दो उसका उद्धार"),
        (218, 236, "हर धड़कन में राम तुम्हारा, हर सांस में तेरा नाम"),
        (236, 254, "तेरी कृपा से जगमग हो, मेरा छोटा सा धाम"),
        (254, 272, "श्री राम जय राम, जय जय राम"),
        (272, 290, "मेरे मन के दीप में, बसते श्री राम"),
        (290, 308, "राम नाम की गूंज उठे, नभ से धरती तक"),
        (308, 326, "भक्ति की यह ज्योति जले, मन से जीवन तक"),
        (326, 344, "जय जय श्री राम... जय जय श्री राम..."),
        (344, 362, "श्री राम जय राम, जय जय राम"),
        (362, 378, "हर दिल बोले सुबह-शाम"),
        (378, 394, "श्री राम जय राम, जय जय राम"),
        (394, 410, "तेरा नाम ही मेरा धाम"),
        (410, 430, "श्री राम... जय राम... जय जय राम..."),
    ]
    lines = []
    for i, (start, end, text) in enumerate(blocks, 1):
        if start >= base.VIDEO_SECONDS:
            continue
        lines += [str(i), f"{stamp(start)} --> {stamp(min(end, base.VIDEO_SECONDS))}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def assemble_long(scene_paths: list[Path], music: Path, final_path: Path):
    if len(scene_paths) != 3:
        raise RuntimeError(f"VIDEO_FATAL: expected 3 source scenes, got {len(scene_paths)}")
    # 8 chapters x 30 seconds = exactly 240 seconds.
    sequence = [scene_paths[0], scene_paths[1], scene_paths[2], scene_paths[1], scene_paths[0], scene_paths[2], scene_paths[1], scene_paths[0]]
    concat = base.OUT / "scenes_long.txt"
    concat.write_text("\n".join(f"file '{p.resolve()}'" for p in sequence) + "\n", encoding="utf-8")
    visual = base.OUT / "visual_long.mp4"
    base.ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-t", str(base.VIDEO_SECONDS), str(visual))
    srt = base.OUT / "lyrics.srt"
    make_long_srt(srt)
    subtitle_filter = "subtitles=" + str(srt.resolve()).replace("\\", "/").replace(":", "\\:")
    audio_filter = "loudnorm=I=-9:TP=-1.0:LRA=7"
    base.ffmpeg("-i", str(visual), "-i", str(music), "-filter_complex", f"[1:a]atrim=0:{base.VIDEO_SECONDS},asetpts=N/SR/TB,{audio_filter}[a]", "-map", "0:v:0", "-map", "[a]", "-vf", subtitle_filter, "-t", str(base.VIDEO_SECONDS), "-r", str(base.FPS), "-s", f"{base.WIDTH}x{base.HEIGHT}", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(final_path))


def validate_long(path: Path):
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height:format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    data = json.loads(probe.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video or not audio:
        raise RuntimeError("OUTPUT_FATAL: final MP4 must contain video and audio")
    if (video.get("width"), video.get("height")) != (base.WIDTH, base.HEIGHT):
        raise RuntimeError("OUTPUT_FATAL: final video dimensions are incorrect")
    if audio.get("codec_name") != "aac":
        raise RuntimeError("OUTPUT_FATAL: final video audio is not AAC")
    duration = float(data.get("format", {}).get("duration", 0))
    if not 178 <= duration <= 302:
        raise RuntimeError(f"OUTPUT_FATAL: final video is not 3-5 minutes: {duration:.2f}s")
    return duration


base.assemble = assemble_long
base.validate = validate_long

if __name__ == "__main__":
    base.main()
    videos = sorted(base.VIDEOS.glob("*.mp4"))
    if not videos:
        raise RuntimeError("OUTPUT_FATAL: final long-form video was not produced")
    dj_master = creative.make_dj_master(videos[0])
    state_path = base.OUT / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({"architecture": "v6.0-longform-4min", "target_duration_sec": 240, "source_scene_seconds": 30, "source_scene_count": 3, "visual_chapters": 8, "music_duration_sec": 240, "dj_master": str(dj_master), "duration_contract": "180-300 seconds; default 240 seconds"})
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("LONGFORM_OK target=240s")
