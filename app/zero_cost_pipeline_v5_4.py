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
    return [f"Unique devotional stock-footage scene {i + 1}" for i in range(count)]


def make_srt(path: Path, seconds: int):
    # Compatibility stub only. The stock assembler never creates or burns SRT.
    path.unlink(missing_ok=True)


def assemble_without_subtitles(scene_paths: list[Path], music_path: Path, final_path: Path):
    """Assemble the stock-footage longform video with NO subtitle filter."""
    concat = base.OUT / "scenes.txt"
    concat.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_paths) + "\n", encoding="utf-8")
    visual = base.OUT / "visual.mp4"
    base.ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(visual))
    audio_filter = "loudnorm=I=-9:TP=-1.0:LRA=7"
    base.ffmpeg(
        "-i", str(visual), "-i", str(music_path),
        "-filter_complex", f"[1:a]atrim=0:{base.VIDEO_SECONDS},asetpts=N/SR/TB,{audio_filter}[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-t", str(base.VIDEO_SECONDS), "-r", str(base.FPS),
        "-s", f"{base.WIDTH}x{base.HEIGHT}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart",
        str(final_path),
    )
    dj_mp3 = base.AUDIO / "bhajan_aabha_dj_master.mp3"
    base.ffmpeg(
        "-i", str(music_path), "-af", audio_filter, "-ar", "48000",
        "-c:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3",
        "-metadata", "title=Bhajan Aabha — DJ Master",
        "-metadata", "artist=Bhajan Aabha",
        "-metadata", "genre=Devotional EDM",
        str(dj_mp3),
    )
    print("SUBTITLES_OK: Hindi subtitles disabled; no subtitle filter applied", flush=True)


def configure() -> int:
    seconds = target_seconds()
    base.VIDEO_SECONDS = seconds
    base.SCENE_SECONDS = SCENE_SECONDS
    base.PACK["lyrics"] = long_lyrics(seconds)
    base.PACK["music_prompt"] = DJ_PROMPT
    base.PACK["scene_prompts"] = scene_prompts(seconds // SCENE_SECONDS)
    base.make_srt = lambda path: make_srt(path, seconds)
    base.require_env = lambda: None
    base.assemble = assemble_without_subtitles
    print(f"ARCHITECTURE=v5.4 CLEAN_FULL_LENGTH ZERO_COST=true VIDEO_SECONDS={seconds} SCENES={seconds // SCENE_SECONDS}")
    print("VISUAL_BACKEND=Pexels stock video API")
    print("SUBTITLES=disabled")
    print("AGNES_API_KEY=not_required_for_stock_visual_pipeline")
    print("MUSIC_BACKEND=ACE-Step v1.5 on Lightning T4")
    print("MUSIC_DURATION=single full-length song; no looping")
    print("MUSIC_STYLE=LOUD_MODERN_DEVOTIONAL_EDM_READY bpm=128")
    return seconds


def generate_music_verified(session=None) -> Path:
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

    final_candidates = sorted(base.VIDEOS.glob("*_bhajan-aabha_{}_v5.mp4".format(base.PACK["slug"])))
    if len(final_candidates) != 1:
        raise RuntimeError(f"OUTPUT_FATAL: expected exactly one final named MP4, found {len(final_candidates)}")
    final_path = final_candidates[0]

    dj_master = music.make_dj_master(final_path)
    state_path = base.OUT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({
        "architecture": "v5.4-clean-full-length",
        "target_duration_range_sec": "180-300",
        "target_duration_sec": seconds,
        "visual_backend": "Pexels stock video API",
        "visual_strategy": "unique_stock_clip_per_scene_with_persistent_no_reuse_ledger",
        "pexels_no_reuse": True,
        "subtitles": False,
        "hindi_subtitles": False,
        "subtitle_burn_in": False,
        "subtitle_sidecar": False,
        "agnes_api_key_required": False,
        "agnes_video_generation": False,
        "agnes_image_generation": False,
        "music_backend": "ACE-Step v1.5 on Lightning T4",
        "music_api_mode": "lightning_ace_step_gpu_studio",
        "music_style": "LOUD_MODERN_DEVOTIONAL_EDM_READY",
        "bpm": 128,
        "time_signature": "4/4",
        "scene_seconds": SCENE_SECONDS,
        "scene_count": seconds // SCENE_SECONDS,
        "full_length_song": True,
        "music_duration_sec": seconds,
        "zero_cost": True,
        "lightning": True,
        "kaggle": False,
        "paid_services": False,
        "paid_gpu": False,
        "actions_artifacts": False,
        "dj_master": str(dj_master),
        "dj_master_bitrate": "320k",
        "dj_master_sample_rate": 48000,
        "final_video": str(final_path),
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")

    for scene in base.VIDEOS.glob("scene_*.mp4"):
        scene.unlink()
    print(f"LONGFORM_OK duration={seconds}s scenes={seconds // SCENE_SECONDS}")
    print(f"FINAL_VIDEO_OK {final_path}")
    print(f"DJ_MASTER_OK {dj_master}")


if __name__ == "__main__":
    main()
