from __future__ import annotations

import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

import app.zero_cost_pipeline_v5 as base

ACE_STEP_SPACE = "ACE-Step/Ace-Step-v1.5"

DJ_MUSIC_PROMPT = """Modern high-energy Hindi devotional bhajan made like a current YouTube DJ devotional song, 128 BPM, 4/4, loud polished commercial stereo production, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, huge memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, modern synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents, harmonium texture, short instrumental intro, strong verse build, massive chorus/drop, rhythmic instrumental break, final chorus with layered backing vocals, professional YouTube/radio loudness and DJ playback energy. NOT meditation music, NOT sleepy, NOT ambient, NOT acoustic-only, NOT spoken narration, NOT humming, NOT a cappella, NOT instrumental-only."""

# IMPORTANT: the current public ACE-Step v1.5 Space exposes exactly 49
# generation_wrapper inputs: 4 wrapper controls followed by these 45 controls.
GENERATION_ARGS = [
    "captions", "lyrics", "bpm", "key_scale", "time_signature", "vocal_language",
    "inference_steps", "guidance_scale", "random_seed_checkbox", "seed", "reference_audio",
    "audio_duration", "batch_size_input", "src_audio", "text2music_audio_code_string",
    "repainting_start", "repainting_end", "instruction_display_gen", "audio_cover_strength",
    "task_type", "use_adg", "cfg_interval_start", "cfg_interval_end", "shift", "infer_method",
    "custom_timesteps", "audio_format", "lm_temperature", "think_checkbox", "lm_cfg_scale",
    "lm_top_k", "lm_top_p", "lm_negative_prompt", "use_cot_metas", "use_cot_caption",
    "use_cot_language", "is_format_caption_state", "constrained_decoding_debug", "auto_score",
    "auto_lrc", "score_scale", "lm_batch_chunk_size", "track_name", "complete_track_classes",
    "autogen_checkbox",
]


def _norm(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _params(info: dict) -> list[dict]:
    return list(info.get("parameters") or [])


def _find_generation_endpoint(api: dict):
    candidates = []
    blocked = ("send_to_cover", "send_to_repaint", "save", "download", "score", "lrc", "format", "sample", "random", "restore", "navigate", "process_src", "transcribe")
    for name, info in (api.get("named_endpoints") or {}).items():
        n = _norm(name)
        if any(x in n for x in blocked):
            continue
        score = 0
        if n in ("/generation_wrapper", "generation_wrapper"):
            score += 1000
        elif "generation_wrapper" in n:
            score += 900
        elif "generate" in n:
            score += 100
        if score:
            candidates.append((score, str(name), info))
    if not candidates:
        raise RuntimeError("MUSIC_FATAL: live ACE-Step generation_wrapper not found")
    candidates.sort(key=lambda x: (-x[0], x[1]))
    print("MUSIC: generation candidates=" + ", ".join(f"{n}:{s}" for s, n, _ in candidates[:6]))
    return candidates[0][1], candidates[0][2]


def _extract_audio(value, seen=None) -> str | None:
    if seen is None:
        seen = set()
    if id(value) in seen:
        return None
    seen.add(id(value))
    if isinstance(value, str):
        low = value.lower()
        if low.startswith("data:audio/") or low.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")):
            return value
        return None
    if isinstance(value, dict):
        for key in ("path", "url", "name"):
            if key in value:
                found = _extract_audio(value[key], seen)
                if found:
                    return found
        for item in value.values():
            found = _extract_audio(item, seen)
            if found:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _extract_audio(item, seen)
            if found:
                return found
    return None


def _save_audio(ref: str, target: Path):
    import requests
    if ref.startswith("data:audio/"):
        target.write_bytes(base64.b64decode(ref.split(",", 1)[1]))
    else:
        source = Path(ref)
        if source.exists():
            shutil.copy2(source, target)
        elif ref.startswith("http"):
            base.download(requests.Session(), ref, target, min_bytes=20000)
        else:
            raise RuntimeError(f"MUSIC_FATAL: inaccessible returned audio reference: {ref}")


def _generation_values() -> list[object]:
    return [
        DJ_MUSIC_PROMPT,
        base.PACK["lyrics"],
        128,
        "C Major",
        "4",
        "hi",
        8,
        7.0,
        True,
        "-1",
        None,
        int(base.VIDEO_SECONDS),
        1,
        None,
        "",
        0.0,
        -1.0,
        "Fill the audio semantic mask based on the given conditions:",
        1.0,
        "text2music",
        False,
        0.0,
        1.0,
        3.0,
        "ode",
        "",
        "mp3",
        0.85,
        False,
        2.0,
        0,
        0.9,
        "sleepy ambient, meditation, humming, spoken narration, a cappella, weak vocals, acoustic-only",
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        0.5,
        8,
        None,
        [],
        False,
    ]


def _choice_values(param: dict) -> list[object]:
    found: list[object] = []
    for key in ("choices", "enum", "values"):
        value = param.get(key)
        if isinstance(value, (list, tuple)):
            found.extend(value)
    ptype = param.get("type")
    if isinstance(ptype, dict):
        for key in ("choices", "enum", "values"):
            value = ptype.get(key)
            if isinstance(value, (list, tuple)):
                found.extend(value)
    return found


def _param_text(param: dict) -> str:
    parts = []
    for key in ("parameter_name", "name", "label", "description", "component"):
        value = param.get(key)
        if value:
            parts.append(str(value))
    return _norm(" ".join(parts))


def _validate_live_contract(params: list[dict], values: list[object]) -> None:
    if len(params) != 49:
        raise RuntimeError(f"MUSIC_FATAL: unexpected live generation_wrapper parameter count={len(params)}; expected 49")
    if len(values) != 45:
        raise RuntimeError(f"MUSIC_FATAL: internal live generation payload count={len(values)}; expected 45")

    first_choices = [_choice_values(p) for p in params[:4]]
    model_choices = {_norm(x) for x in first_choices[0]}
    mode_choices = {_norm(x) for x in first_choices[1]}
    # _norm converts hyphens to underscores, so compare normalized values.
    if model_choices and "acestep_v15_turbo" not in model_choices:
        raise RuntimeError(f"MUSIC_FATAL: live model choices changed: {first_choices[0]!r}")
    if mode_choices and not {"simple", "custom"}.issubset(mode_choices):
        raise RuntimeError(f"MUSIC_FATAL: live generation-mode choices changed: {first_choices[1]!r}")

    task_choices = {_norm(x) for x in _choice_values(params[23])}
    if task_choices and "text2music" not in task_choices:
        raise RuntimeError(f"MUSIC_FATAL: live task_type moved/changed at parameter 23: {task_choices!r}")


def _resolve_live_values(params: list[dict]) -> list[object]:
    values = _generation_values()
    _validate_live_contract(params, values)
    wrapper_values = [
        "acestep-v15-turbo",
        "custom",
        "",
        "hi",
    ] + values
    if len(wrapper_values) != len(params):
        raise RuntimeError(f"MUSIC_FATAL: final live payload mismatch: payload={len(wrapper_values)} params={len(params)}")
    for index, value in enumerate(wrapper_values):
        text = _param_text(params[index])
        print(f"MUSIC: param[{index}]={text or '<unnamed>'} -> {value!r}")
    return wrapper_values


def generate_music_gradio() -> Path:
    from gradio_client import Client
    values = _generation_values()
    if len(values) != len(GENERATION_ARGS):
        raise RuntimeError(f"MUSIC_FATAL: internal ACE-Step value map mismatch: values={len(values)} expected={len(GENERATION_ARGS)}")
    print("MUSIC: connecting to official ACE-Step v1.5 public ZeroGPU Space")
    print("MUSIC: using exact current 49-parameter generation_wrapper contract")
    print("MUSIC: style=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128 batch=1 thinking=off")
    last_error = None
    for attempt in range(1, 4):
        try:
            client = Client(ACE_STEP_SPACE, max_workers=1)
            api = client.view_api(print_info=False, return_format="dict")
            endpoint, info = _find_generation_endpoint(api)
            params = _params(info)
            print(f"MUSIC: selected endpoint={endpoint}")
            print(f"MUSIC: live contract={len(params)} parameters; validating exact current schema")
            live_values = _resolve_live_values(params)
            result = client.predict(*live_values, api_name=endpoint)
            print("MUSIC: Gradio generation completed")
            print("MUSIC: result type=" + type(result).__name__)
            print("MUSIC: result preview=" + repr(result)[:1800])
            audio_ref = _extract_audio(result)
            if not audio_ref:
                raise RuntimeError(f"MUSIC_FATAL: generation returned no downloadable audio: {result!r}")
            target = base.AUDIO / "bhajan_source.mp3"
            _save_audio(audio_ref, target)
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
                print(f"MUSIC: ZeroGPU task was aborted; reconnecting after {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"MUSIC_FATAL: ACE-Step ZeroGPU failed after 3 attempts: {last_error}")


def make_dj_master(final_video: Path) -> Path:
    target = base.AUDIO / "bhajan_aabha_dj_master.mp3"
    p = subprocess.run(["ffmpeg", "-y", "-i", str(final_video), "-vn", "-af", "highpass=f=28,lowpass=f=19000,loudnorm=I=-9:TP=-1.0:LRA=7", "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", str(target)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError("DJ_MASTER_FATAL: " + p.stderr[-4000:])
    if target.stat().st_size < 50000:
        raise RuntimeError("DJ_MASTER_FATAL: master file is suspiciously small")
    print("DJ_MASTER_OK", target, target.stat().st_size)
    return target

base.generate_music = lambda session: generate_music_gradio()
base.ACESTEP_ROOT = "gradio://ACE-Step/Ace-Step-v1.5"

if __name__ == "__main__":
    base.main()
    videos = sorted(base.VIDEOS.glob("*.mp4"))
    if not videos:
        raise RuntimeError("OUTPUT_FATAL: final video was not produced")
    dj_master = make_dj_master(videos[0])
    state_path = base.OUT / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update({"music_backend": "ACE-Step v1.5 official Hugging Face ZeroGPU Space via Gradio Client", "music_api_mode": "gradio_client_live_api_exact_49", "music_style": "LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY", "bpm": 128, "time_signature": "4/4", "dj_master": str(dj_master), "dj_master_bitrate": "320k", "dj_master_sample_rate": 48000})
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STATE_OK music_backend=ACE-Step official ZeroGPU Space via Gradio Client")
