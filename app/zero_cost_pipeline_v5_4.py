from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import app.zero_cost_pipeline_v5 as base
import app.zero_cost_pipeline_v5_2 as music

MIN_SECONDS = 180
MAX_SECONDS = 300
SCENE_SECONDS = 15
DJ_PROMPT = music.DJ_MUSIC_PROMPT

MOTION = [
    "slow cinematic push-in toward the deity, subtle garment motion, flickering diyas and drifting incense",
    "gentle lateral dolly around the deity, jewelry catching warm light, flower petals floating through foreground",
    "low-angle rise from rows of diyas toward the deity, sunrise rays expanding behind the crown",
    "slow orbit around the deity with temple pillars parallaxing naturally, bells moving slightly, incense haze drifting",
    "forward crane through marigold flowers toward the deity, shallow depth of field, warm practical lamp flicker",
    "wide establishing sweep of the temple courtyard resolving on the deity, flags and fabric moving gently",
    "hero close-up with restrained rack focus from the bow to the deity's compassionate face, natural detail",
    "overhead-to-eye-level devotional camera descent, petals and dust motes moving in sunbeams",
    "side tracking shot past glowing oil lamps revealing the same deity, rich golden highlights and realistic depth",
    "slow pull-back revealing the full shrine while keeping the deity dominant, subtle bells and incense motion",
    "cinematic push through a foreground arch toward the deity during a musical build, brighter sunrise",
    "majestic final reveal with a gentle circular camera move, petals rising and radiant temple lights",
]


def target_seconds() -> int:
    value = int(os.getenv("VIDEO_SECONDS", "180"))
    if not MIN_SECONDS <= value <= MAX_SECONDS or value % SCENE_SECONDS:
        raise RuntimeError(f"CONFIG_FATAL: VIDEO_SECONDS must be 180-300 and divisible by 15; got {value}")
    return value


def long_lyrics(seconds: int) -> str:
    blocks = [
        "[Intro]\nश्री राम... श्री राम... जय जय राम...",
        "[Verse 1]\nमन में बसो रघुनंदन, चरणों में मेरा ध्यान\nराम नाम की ज्योति जले, रोशन हो हर प्राण",
        "[Pre-Chorus]\nतेरे नाम की धुन बजे, हर धड़कन में आज\nतेरी कृपा से खिल उठे, जीवन का हर राज",
        "[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम",
        "[Instrumental Break]\n",
        "[Verse 2]\nदुख की घड़ी में साथ दो, हे दीनदयाल भगवान\nतेरा नाम ही आसरा, तेरा नाम ही सम्मान",
        "[Pre-Chorus]\nतेरी राह में चल पड़ूँ, मन में लेकर विश्वास\nराम नाम की शक्ति से, मिट जाए हर त्रास",
        "[Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम",
        "[Verse 3]\nअयोध्या के राजकुमार, करुणा के भंडार\nतेरे चरणों में मिल जाए, जीवन का सच्चा सार",
        "[Build]\nजय श्री राम की गूंज उठे, नभ से धरती तक\nढोल बजे और शंख बजे, प्रेम बहे हर पल",
        "[Final Chorus]\nश्री राम जय राम, जय जय राम\nमेरे मन के दीप में, बसते श्री राम\nश्री राम जय राम, जय जय राम\nजय जय राम... जय जय राम...",
        "[Outro]\nश्री राम... जय राम... जय जय राम...",
    ]
    text = ""
    while len(text) < max(1800, seconds * 9):
        text += "\n\n".join(blocks) + "\n\n"
    return text[:4090]


def scene_prompts(count: int) -> list[str]:
    return [
        f"{MOTION[i % len(MOTION)]}. Lord Rama is the exact same canonical Hindu deity from the supplied reference image, serene compassionate divine face, blue-tinted skin, golden crown, yellow silk dhoti, ornate jewelry, bow, traditional sacred iconography. Magnificent Ayodhya-inspired temple courtyard, warm cinematic golden light, marigold flowers, diyas and incense haze. Premium modern devotional YouTube music-video cinematography, realistic natural motion, stable identity, no morphing, no duplicate deity, no text, no watermark, no modern clothing."
        for i in range(count)
    ]


def make_srt(path: Path, seconds: int):
    lyrics = [
        "मन में बसो रघुनंदन, चरणों में मेरा ध्यान",
        "राम नाम की ज्योति जले, रोशन हो हर प्राण",
        "तेरे नाम की धुन बजे, हर धड़कन में आज",
        "श्री राम जय राम, जय जय राम",
        "मेरे मन के दीप में, बसते श्री राम",
        "दुख की घड़ी में साथ दो, हे दीनदयाल भगवान",
        "तेरा नाम ही आसरा, तेरा नाम ही सम्मान",
        "अयोध्या के राजकुमार, करुणा के भंडार",
        "जय श्री राम की गूंज उठे, नभ से धरती तक",
        "ढोल बजे और शंख बजे, प्रेम बहे हर पल",
        "श्री राम जय राम, जय जय राम",
        "जय जय राम... जय जय राम...",
    ]
    rows = []
    for n, start in enumerate(range(0, seconds, 6), 1):
        end = min(seconds, start + 6)
        h1, r1 = divmod(start, 3600); m1, s1 = divmod(r1, 60)
        h2, r2 = divmod(end, 3600); m2, s2 = divmod(r2, 60)
        rows += [str(n), f"{h1:02}:{m1:02}:{s1:02},000 --> {h2:02}:{m2:02}:{s2:02},000", lyrics[(n - 1) % len(lyrics)], ""]
    path.write_text("\n".join(rows), encoding="utf-8")


def configure() -> int:
    seconds = target_seconds()
    base.VIDEO_SECONDS = seconds
    base.SCENE_SECONDS = SCENE_SECONDS
    base.PACK["lyrics"] = long_lyrics(seconds)
    base.PACK["music_prompt"] = DJ_PROMPT
    base.PACK["scene_prompts"] = scene_prompts(seconds // SCENE_SECONDS)
    base.make_srt = lambda path: make_srt(path, seconds)
    print(f"ARCHITECTURE=v5.4 CLEAN_FULL_LENGTH ZERO_COST=true VIDEO_SECONDS={seconds} SCENES={seconds // SCENE_SECONDS}")
    print("VISUAL_BACKEND=Agnes Image 2.1 Flash + Agnes Video v2.0")
    print("MUSIC_BACKEND=ACE-Step v1.5 official public ZeroGPU Space via live Gradio API")
    print("MUSIC_DURATION=single full-length song; no looping")
    print("MUSIC_STYLE=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128")
    return seconds


def generate_music_verified(session=None) -> Path:
    # v5.2 owns the live ACE-Step positional contract. Current live contract:
    # 4 wrapper inputs + 45 generation inputs = 49 total. Do not duplicate,
    # delete, reorder, or reinterpret those positional arguments here.
    target = music.generate_music_gradio()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(target)],
        capture_output=True, text=True, check=True,
    )
    duration = float(probe.stdout.strip())
    required = target_seconds()
    if duration < required - 3:
        raise RuntimeError(f"MUSIC_FATAL: ACE-Step returned only {duration:.2f}s; required >= {required-3}s full-length audio")
    print(f"MUSIC_OK full_length_source={target} duration={duration:.2f}s")
    return target


def main():
    seconds = configure()
    base.generate_music = generate_music_verified
    base.ACESTEP_ROOT = "gradio://ACE-Step/Ace-Step-v1.5"
    base.main()

    videos = sorted(base.VIDEOS.glob("*.mp4"))
    if len(videos) != 1:
        raise RuntimeError(f"OUTPUT_FATAL: expected exactly one final MP4, found {len(videos)}")

    dj_master = music.make_dj_master(videos[0])
    state_path = base.OUT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({
        "architecture": "v5.4-clean-full-length",
        "target_duration_range_sec": "180-300",
        "target_duration_sec": seconds,
        "music_backend": "ACE-Step v1.5 official Hugging Face ZeroGPU Space via Gradio Client",
        "music_api_mode": "live_gradio_49_input_contract",
        "music_input_contract": "4 wrapper + 45 generation = 49 total",
        "music_style": "LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY",
        "bpm": 128,
        "time_signature": "4/4",
        "scene_seconds": SCENE_SECONDS,
        "scene_count": seconds // SCENE_SECONDS,
        "full_length_song": True,
        "music_duration_sec": seconds,
        "zero_cost": True,
        "kaggle": False,
        "paid_services": False,
        "paid_gpu": False,
        "actions_artifacts": False,
        "dj_master": str(dj_master),
        "dj_master_bitrate": "320k",
        "dj_master_sample_rate": 48000,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"LONGFORM_OK duration={seconds}s scenes={seconds // SCENE_SECONDS}")


if __name__ == "__main__":
    main()
