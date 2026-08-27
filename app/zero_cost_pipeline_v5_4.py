from __future__ import annotations

import json
import time
from pathlib import Path

import app.zero_cost_pipeline_v5 as base
import app.zero_cost_pipeline_v5_3 as longform
import app.zero_cost_pipeline_v5_2 as music


def generate_music_fixed() -> Path:
    """Call the current public ACE-Step v1.5 Gradio contract safely.

    The live generation_wrapper currently exposes 48 total inputs: 4 wrapper
    inputs + 44 generation inputs. The older v5.2 mapping expected 49 and the
    v5.3 patch incorrectly added an instrumental checkbox. Build the 44-value
    generation list from the known v5.2 values while removing the obsolete
    is_format_caption_state slot (index 36).
    """
    from gradio_client import Client

    values = list(music._ORIGINAL_VALUES())
    if len(values) != 45:
        raise RuntimeError(f"MUSIC_FATAL: legacy ACE-Step value map changed: {len(values)}; expected 45")
    # Current live API has no is_format_caption_state parameter.
    del values[36]
    if len(values) != 44:
        raise RuntimeError(f"MUSIC_FATAL: corrected ACE-Step generation values={len(values)}; expected 44")

    style = music.DJ_MUSIC_PROMPT
    print("MUSIC: connecting to official ACE-Step v1.5 public ZeroGPU Space")
    print("MUSIC: LIVE CONTRACT = 4 wrapper + 44 generation = 48 total inputs")
    print("MUSIC: style=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128")

    last_error = None
    for attempt in range(1, 4):
        try:
            client = Client(music.ACE_STEP_SPACE, max_workers=1)
            api = client.view_api(print_info=False, return_format="dict")
            endpoint, info = music._find_generation_endpoint(api)
            params = list(info.get("parameters") or [])
            if len(params) != 48:
                raise RuntimeError(f"MUSIC_FATAL: live generation_wrapper exposes {len(params)} inputs, expected 48; refusing unsafe call")

            print(f"MUSIC: selected endpoint={endpoint}")
            print("MUSIC: validated live parameter count=48")
            wrapper_values = ["acestep-v15-turbo", "custom", "", "hi"] + values
            result = client.predict(*wrapper_values, api_name=endpoint)
            print("MUSIC: Gradio generation completed")
            print("MUSIC: result type=" + type(result).__name__)
            print("MUSIC: result preview=" + repr(result)[:1800])
            audio_ref = music._extract_audio(result)
            if not audio_ref:
                raise RuntimeError(f"MUSIC_FATAL: ACE-Step returned no downloadable audio: {result!r}")
            target = base.AUDIO / "bhajan_source.mp3"
            music._save_audio(audio_ref, target)
            if target.stat().st_size < 20000:
                raise RuntimeError("MUSIC_FATAL: generated audio is suspiciously small")
            print("MUSIC_OK", target, target.stat().st_size)
            return target
        except Exception as exc:
            last_error = exc
            print(f"MUSIC: generation attempt {attempt}/3 failed: {type(exc).__name__}: {exc}")
            if "GPU" not in str(exc) and "ZeroGPU" not in str(exc):
                raise
            if attempt < 3:
                wait = 25 * attempt
                print(f"MUSIC: ZeroGPU transient failure; retrying after {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"MUSIC_FATAL: ACE-Step failed after 3 attempts: {last_error}")


def main():
    longform.configure()
    # v5.3 configure establishes the 3-5 minute architecture and visual
    # generation. Replace only its broken music adapter with this live-contract
    # implementation; everything else remains unchanged.
    base.generate_music = lambda session: generate_music_fixed()
    base.ACESTEP_ROOT = "gradio://ACE-Step/Ace-Step-v1.5"
    print("ARCHITECTURE=v5.4 FULL_LENGTH ZERO_COST=true VIDEO_SECONDS=180 SCENES=12")
    print("MUSIC_BACKEND=ACE-Step v1.5 official public ZeroGPU Space via live Gradio API")
    print("MUSIC_DURATION=180s single full-length song; no looping")
    base.main()
    videos = sorted(base.VIDEOS.glob("*.mp4"))
    if not videos:
        raise RuntimeError("OUTPUT_FATAL: final full-length video was not produced")
    dj_master = music.make_dj_master(videos[0])
    state_path = base.OUT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({
        "architecture": "v5.4-full-length-live-ace-contract",
        "target_duration_range_sec": "180-300",
        "target_duration_sec": 180,
        "music_duration_sec": 180,
        "music_backend": "ACE-Step v1.5 official Hugging Face ZeroGPU Space via Gradio Client",
        "music_api_mode": "live_gradio_48_input_contract",
        "music_input_contract": "4 wrapper + 44 generation = 48 total",
        "music_style": "LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY",
        "bpm": 128,
        "time_signature": "4/4",
        "scene_seconds": 15,
        "scene_count": 12,
        "full_length_song": True,
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
    print("LONGFORM_OK target=180s music=180s scenes=12")


if __name__ == "__main__":
    main()
