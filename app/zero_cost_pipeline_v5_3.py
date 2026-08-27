from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import app.zero_cost_pipeline_v5 as base
import app.zero_cost_pipeline_v5_2 as music

# Bhajan Aabha v5.3: full-length architecture.
# Target is 3-5 minutes per video, never a 30-45 second demo.
# 15-second visual scenes are generated from one canonical deity reference and
# stitched against one full-length ACE-Step song. Default = 180 seconds.
MIN_SECONDS = 180
MAX_SECONDS = 300
SCENE_SECONDS = 15


def target_seconds() -> int:
    raw = int(os.getenv("VIDEO_SECONDS", str(MIN_SECONDS)))
    if raw < MIN_SECONDS or raw > MAX_SECONDS or raw % SCENE_SECONDS:
        raise RuntimeError(
            f"CONFIG_FATAL: VIDEO_SECONDS must be 180-300 and divisible by 15; got {raw}"
        )
    return raw


def long_lyrics(seconds: int) -> str:
    core = [
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
    # Repeat the middle devotional structure as needed so ACE-Step has enough
    # lyrical material for a real song rather than stretching a 45-second demo.
    out = []
    while len("\n\n".join(out)) < max(900, int(seconds * 11)):
        out.extend(core[1:-1])
    out.append(core[-1])
    text = "\n\n".join(out)
    return text[:4000]


MOTION_VARIANTS = [
    "slow cinematic push-in toward the deity, subtle breathing-like garment motion, flickering diyas and drifting incense",
    "gentle lateral dolly around the deity, jewelry catching warm light, flower petals floating through the foreground",
    "low-angle rise from rows of diyas toward the deity, sunrise rays expanding behind the crown, soft divine aura",
    "slow orbit around the deity with temple pillars parallaxing naturally, bells moving slightly, incense haze drifting",
    "forward crane through marigold flowers toward the deity, shallow depth of field, warm practical lamp flicker",
    "wide establishing sweep of the temple courtyard resolving on the deity, flags and fabric moving gently in the breeze",
    "hero close-up with a restrained rack focus from the bow to the deity's compassionate face, natural eye and cloth detail",
    "overhead-to-eye-level devotional camera descent, petals and dust motes moving in sunbeams, dignified sacred atmosphere",
    "side tracking shot past glowing oil lamps revealing the same deity, rich golden highlights and realistic temple depth",
    "slow pull-back revealing the full shrine and architecture while keeping the deity dominant, subtle bells and incense motion",
    "cinematic push through a foreground arch toward the deity during a musical build, brighter sunrise and stronger aura",
    "majestic final reveal with a gentle circular camera move, petals rising in the breeze, radiant temple lights and devotional climax",
    "rhythmic but smooth DJ-style camera glide synchronized to the beat, alternating close and medium framing without changing identity",
    "dramatic temple doorway reveal, warm backlight around the deity, controlled lens flare, realistic fabric and jewelry movement",
    "slow dolly across sacred lamps and flowers, focus settles on the deity's face, peaceful but powerful devotional energy",
    "gentle upward tilt from feet and bow to crown, preserving traditional iconography, cinematic sunrise and floating petals",
    "wide-to-medium zoom through the temple courtyard, distant lamps shimmering, deity remains sharp and unchanged",
    "soft circular tracking move during the chorus, subtle cloth movement and glowing particles, premium devotional music-video look",
    "camera passes behind a temple pillar then reveals the deity again, natural parallax and consistent face, hands and clothing",
    "final celebratory devotional shot, slow orbit with brighter lamps, flower petals and golden rays, powerful YouTube music-video finish",
]


def scene_prompts(count: int) -> list[str]:
    prompts = []
    for i in range(count):
        motion = MOTION_VARIANTS[i % len(MOTION_VARIANTS)]
        prompts.append(
            f"{motion}. Lord Rama is the exact same canonical Hindu deity from the supplied reference image, serene compassionate divine face, blue-tinted skin, golden crown, yellow silk dhoti, ornate jewelry, bow, traditional sacred iconography. Magnificent Ayodhya-inspired temple courtyard, warm cinematic golden light, marigold flowers, diyas and incense haze. Premium modern devotional YouTube music-video cinematography, realistic natural motion, detailed hands and face, stable identity, no morphing, no duplicate deity, no text, no watermark, no modern clothing."
        )
    return prompts


def long_srt(path: Path, seconds: int):
    lines = [
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
    step = 6
    rows = []
    idx = 1
    for start in range(0, seconds, step):
        end = min(seconds, start + step)
        text = lines[(idx - 1) % len(lines)]
        def stamp(s):
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            return f"{h:02}:{m:02}:{sec:02},000"
        rows += [str(idx), f"{stamp(start)} --> {stamp(end)}", text, ""]
        idx += 1
    path.write_text("\n".join(rows), encoding="utf-8")


# --- Correct the live ACE-Step v1.5 Gradio contract ---
# Current public Space exposes 4 wrapper inputs + 46 generation inputs.
# The current extra generation input is instrumental_checkbox, after the
# regular generation parameters. v5.2 had 45 values, causing the subsequent
# positional values to drift / validation failures.
_ORIGINAL_VALUES = music._generation_values
music.GENERATION_ARGS = list(music.GENERATION_ARGS) + ["instrumental_checkbox"]
music._params = lambda info: list(info.get("parameters") or [])[:-1]


def fixed_generation_values():
    values = list(_ORIGINAL_VALUES())
    if len(values) != 45:
        raise RuntimeError(f"MUSIC_FATAL: unexpected legacy value count={len(values)}; expected 45 before instrumental checkbox")
    values.append(False)  # instrumental_checkbox: vocals are required for Bhajan Aabha
    return values


music._generation_values = fixed_generation_values


def configure():
    seconds = target_seconds()
    base.VIDEO_SECONDS = seconds
    base.SCENE_SECONDS = SCENE_SECONDS
    base.PACK["lyrics"] = long_lyrics(seconds)
    base.PACK["scene_prompts"] = scene_prompts(seconds // SCENE_SECONDS)
    # Keep the modern loud devotional arrangement requested by the channel.
    base.PACK["music_prompt"] = music.DJ_MUSIC_PROMPT
    base.make_srt = lambda path: long_srt(path, seconds)
    base.generate_music = lambda session: music.generate_music_gradio()
    base.ACESTEP_ROOT = "gradio://ACE-Step/Ace-Step-v1.5"
    print(f"ARCHITECTURE=v5.3 FULL_LENGTH ZERO_COST=true VIDEO_SECONDS={seconds} SCENES={seconds // SCENE_SECONDS}")
    print("MUSIC=ACE-Step-v1.5 official public ZeroGPU Space via live Gradio API")
    print("MUSIC_DURATION=full_length_single_song")
    print("VISUALS=canonical_deity_reference + 15s_motion_scenes")


if __name__ == "__main__":
    configure()
    base.main()
    videos = sorted(base.VIDEOS.glob("*.mp4"))
    if not videos:
        raise RuntimeError("OUTPUT_FATAL: final full-length video was not produced")
    state_path = base.OUT / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({
            "architecture": "v5.3-full-length",
            "target_duration_range_sec": "180-300",
            "music_backend": "ACE-Step v1.5 official Hugging Face ZeroGPU Space via Gradio Client",
            "music_api_mode": "live_gradio_positional_contract_50_inputs",
            "music_style": "LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY",
            "bpm": 128,
            "time_signature": "4/4",
            "scene_seconds": SCENE_SECONDS,
            "scene_count": target_seconds() // SCENE_SECONDS,
            "full_length_song": True,
            "zero_cost": True,
            "kaggle": False,
            "paid_services": False,
            "paid_gpu": False,
            "actions_artifacts": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STATE_OK full_length=true")
