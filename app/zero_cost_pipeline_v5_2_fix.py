from __future__ import annotations

import json
import time
from pathlib import Path

from gradio_client import Client

import app.zero_cost_pipeline_v5_2 as v52

ACE_STEP_SPACE = v52.ACE_STEP_SPACE
base = v52.base

# Canonical ACE-Step generation values. The public Space has exposed both
# 45- and 46-generation-input variants; compatibility is resolved from the
# live endpoint length, with only the trailing optional state argument omitted
# for the 45-input variant.
GENERATION_ARGS = [
    "captions", "lyrics", "bpm", "key_scale", "time_signature", "vocal_language",
    "inference_steps", "guidance_scale", "random_seed_checkbox", "seed", "reference_audio",
    "audio_duration", "batch_size_input", "src_audio", "text2music_audio_code_string",
    "repainting_start", "repainting_end", "instruction_display_gen", "audio_cover_strength",
    "task_type", "use_adg", "cfg_interval_start", "cfg_interval_end", "shift", "infer_method",
    "custom_timesteps", "audio_format", "lm_temperature", "think_checkbox", "lm_cfg_scale",
    "lm_top_k", "lm_top_p", "lm_negative_prompt", "use_cot_metas", "use_cot_caption",
    "use_cot_language", "is_format_caption_state", "constrained_decoding_debug", "allow_lm_batch",
    "instrumental_checkbox", "auto_score", "auto_lrc", "score_scale", "lm_batch_chunk_size",
    "track_name", "complete_track_classes",
]


def _find_generation_endpoint(api: dict):
    for name, info in (api.get("named_endpoints") or {}).items():
        if str(name).strip().lower() in ("/generation_wrapper", "generation_wrapper"):
            return str(name), info
    raise RuntimeError("MUSIC_FATAL: live ACE-Step generation_wrapper not found")


def _values() -> list[object]:
    return [
        v52.DJ_MUSIC_PROMPT, base.PACK["lyrics"], 128, "C Major", "4", "hi", 8, 7.0,
        True, "-1", None, float(base.VIDEO_SECONDS), 1, None, "", 0.0, -1.0,
        "Fill the audio semantic mask based on the given conditions:", 1.0, "text2music",
        False, 0.0, 1.0, 3.0, "ode", "", "mp3", 0.85, False, 2.0, 0, 0.9,
        "sleepy ambient, meditation, humming, spoken narration, a cappella, weak vocals, acoustic-only",
        False, False, False, False, False, False, False, False, False, 0.5, 1, None, [],
    ]


def generate_music_gradio() -> Path:
    canonical_values = _values()
    if len(canonical_values) != 46 or len(GENERATION_ARGS) != 46:
        raise RuntimeError(f"MUSIC_FATAL: canonical positional map mismatch values={len(canonical_values)} args={len(GENERATION_ARGS)}")

    print("MUSIC: connecting to official ACE-Step v1.5 public ZeroGPU Space")
    print("MUSIC: resolving LIVE generation_wrapper contract at runtime")
    print("MUSIC: style=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128 batch=1 thinking=off")

    last_error = None
    for attempt in range(1, 4):
        try:
            client = Client(ACE_STEP_SPACE, max_workers=1)
            api = client.view_api(print_info=False, return_format="dict")
            endpoint, info = _find_generation_endpoint(api)
            params = list(info.get("parameters") or [])
            total = len(params)
            if total not in (49, 50):
                raise RuntimeError(f"MUSIC_FATAL: unsupported live generation_wrapper contract: total_params={total}; supported=49 or 50")
            generation_count = total - 4
            values = canonical_values[:generation_count]
            wrapper_values = ["acestep-v15-turbo", "custom", "", "hi"] + values
            if len(wrapper_values) != total:
                raise RuntimeError(f"MUSIC_FATAL: runtime positional count mismatch call={len(wrapper_values)} endpoint={total}")
            mode = "45-generation compatibility" if generation_count == 45 else "46-generation canonical"
            print(f"MUSIC: selected endpoint={endpoint}")
            print(f"MUSIC: live contract verified: total={total}; wrapper=4; generation={generation_count}; mode={mode}")

            result = client.predict(*wrapper_values, api_name=endpoint)
            print("MUSIC: Gradio generation completed")
            print("MUSIC: result type=" + type(result).__name__)
            print("MUSIC: result preview=" + repr(result)[:1800])

            audio_ref = v52._extract_audio(result)
            if not audio_ref:
                raise RuntimeError(f"MUSIC_FATAL: generation returned no downloadable audio: {result!r}")
            target = base.AUDIO / "bhajan_source.mp3"
            v52._save_audio(audio_ref, target)
            if target.stat().st_size < 20000:
                raise RuntimeError("MUSIC_FATAL: generated audio is suspiciously small")
            print("MUSIC_OK", target, target.stat().st_size)
            return target
        except Exception as exc:
            last_error = exc
            print(f"MUSIC: generation attempt {attempt}/3 failed: {type(exc).__name__}: {exc}")
            if "GPU task aborted" not in str(exc) and "GPU" not in str(exc):
                raise
            if attempt < 3:
                wait = 20 * attempt
                print(f"MUSIC: ZeroGPU task aborted; reconnecting after {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"MUSIC_FATAL: ACE-Step ZeroGPU failed after 3 attempts: {last_error}")


base.generate_music = lambda session: generate_music_gradio()
base.ACESTEP_ROOT = "gradio://ACE-Step/Ace-Step-v1.5"

if __name__ == "__main__":
    base.main()
    videos = sorted(base.VIDEOS.glob("*.mp4"))
    if not videos:
        raise RuntimeError("OUTPUT_FATAL: final video was not produced")
    dj_master = v52.make_dj_master(videos[0])
    state_path = base.OUT / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({
            "music_backend": "ACE-Step v1.5 official Hugging Face ZeroGPU Space via Gradio Client",
            "music_api_mode": "gradio_client_live_api_runtime_contract",
            "music_style": "LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY",
            "bpm": 128,
            "time_signature": "4/4",
            "dj_master": str(dj_master),
            "dj_master_bitrate": "320k",
            "dj_master_sample_rate": 48000,
        })
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STATE_OK music_backend=ACE-Step official ZeroGPU Space via Gradio Client")
