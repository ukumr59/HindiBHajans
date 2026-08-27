from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from gradio_client import Client

import app.zero_cost_pipeline_v5 as base

SPACE = os.getenv("ACESTEP_SPACE", "ACE-Step/Ace-Step-v1.5")

# The public ACE-Step deployment is a Gradio Space, not the standalone REST
# server. Discover the live Gradio API schema at runtime so UI refactors do
# not silently break the pipeline.

VALUE_MAP = {
    "captions": base.PACK["music_prompt"], "caption": base.PACK["music_prompt"],
    "lyrics": base.PACK["lyrics"], "bpm": None, "key_scale": "", "keyscale": "",
    "time_signature": "4", "timesignature": "4", "vocal_language": "hi",
    "inference_steps": 8, "guidance_scale": 7.0, "random_seed_checkbox": True,
    "use_random_seed": True, "seed": "-1", "reference_audio": None,
    "audio_duration": int(os.getenv("VIDEO_SECONDS", "45")),
    "duration": int(os.getenv("VIDEO_SECONDS", "45")), "batch_size_input": 1,
    "batch_size": 1, "src_audio": None, "text2music_audio_code_string": "",
    "audio_codes": "", "repainting_start": 0.0, "repainting_end": -1,
    "instruction_display_gen": "", "instruction": "", "audio_cover_strength": 1.0,
    "task_type": "text2music", "use_adg": False, "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0, "shift": 3.0, "infer_method": "ode",
    "custom_timesteps": "", "audio_format": "mp3", "lm_temperature": 0.85,
    "think_checkbox": False, "thinking": False, "lm_cfg_scale": 2.0,
    "lm_top_k": 0, "lm_top_p": 0.9, "lm_negative_prompt": "",
    "use_cot_metas": False, "use_cot_caption": False, "use_cot_language": False,
    "is_format_caption": False, "constrained_decoding_debug": False,
    "allow_lm_batch": False, "auto_score": False, "auto_lrc": False,
    "score_scale": 1.0, "lm_batch_chunk_size": 1,
}


def norm(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def endpoint_candidates(info: dict):
    out = []
    for name, endpoint in (info.get("named_endpoints") or {}).items():
        out.append((name, endpoint))
    for name, endpoint in (info.get("unnamed_endpoints") or {}).items():
        out.append((int(name) if str(name).isdigit() else name, endpoint))
    return out


def choose_generation_endpoint(info: dict):
    wanted = {norm(x) for x in ("captions", "lyrics", "vocal_language", "audio_duration", "inference_steps")}
    candidates = []
    for name, endpoint in endpoint_candidates(info):
        params = endpoint.get("parameters") or []
        labels = {norm(p.get("parameter_name") or p.get("label") or "") for p in params}
        overlap = len(wanted & labels)
        text = norm(name)
        # The direct generate_with_progress endpoint currently has 47 inputs.
        # Penalize much larger wrappers (batch-management endpoints) that need
        # UI state objects we do not have in a headless workflow.
        count_penalty = abs(len(params) - 47) * 3
        score = overlap * 100 + (20 if "generate" in text else 0) + (20 if "music" in text else 0)
        score += 15 if len(params) == 47 else 0
        score -= count_penalty
        if overlap >= 4:
            candidates.append((score, name, endpoint))
    if not candidates:
        raise RuntimeError("MUSIC_FATAL: could not discover the live ACE-Step Gradio generation endpoint")
    candidates.sort(reverse=True, key=lambda x: x[0])
    score, name, endpoint = candidates[0]
    print(f"MUSIC_API_ENDPOINT={name} score={score} parameter_count={len(endpoint.get('parameters') or [])}")
    print("MUSIC_API_PARAMETERS:", [p.get("parameter_name") or p.get("label") for p in endpoint.get("parameters", [])])
    return name, endpoint


def parameter_value(p: dict):
    key = norm(p.get("parameter_name") or p.get("label") or "")
    aliases = {norm(k): v for k, v in VALUE_MAP.items()}
    if key in aliases:
        return aliases[key]
    label = norm(p.get("label") or "")
    if label in aliases:
        return aliases[label]
    if p.get("parameter_has_default"):
        return p.get("parameter_default")
    optional_defaults = {"trackname": None, "completetrackclasses": [], "progress": None}
    if key in optional_defaults:
        return optional_defaults[key]
    raise RuntimeError(
        "MUSIC_FATAL: live ACE-Step endpoint has an unknown required parameter: "
        f"{p.get('parameter_name') or p.get('label')}"
    )


def extract_audio(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        s = str(value)
        if s.lower().split("?")[0].endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            return Path(s)
        return None
    if isinstance(value, dict):
        for k in ("path", "filepath", "file", "url"):
            if k in value:
                found = extract_audio(value[k])
                if found:
                    return found
        for v in value.values():
            found = extract_audio(v)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for v in value:
            found = extract_audio(v)
            if found:
                return found
    for attr in ("path", "url"):
        if hasattr(value, attr):
            found = extract_audio(getattr(value, attr))
            if found:
                return found
    return None


def generate_music_gradio(session):
    print("MUSIC: connecting to official ACE-Step 1.5 Hugging Face Gradio Space")
    client = Client(SPACE, verbose=True, download_files=str(base.AUDIO.resolve()))
    info = client.view_api(all_endpoints=True, print_info=False, return_format="dict")
    name, endpoint = choose_generation_endpoint(info)
    values = [parameter_value(p) for p in (endpoint.get("parameters") or [])]
    print("MUSIC: submitting Hindi bhajan with sung vocals + instrumental production")
    try:
        if isinstance(name, int):
            result = client.predict(*values, fn_index=name)
        else:
            api_name = name if str(name).startswith("/") else f"/{name}"
            result = client.predict(*values, api_name=api_name)
    except Exception as exc:
        raise RuntimeError(f"MUSIC_FATAL: ACE-Step Gradio generation failed: {exc}") from exc
    audio_path = extract_audio(result)
    if not audio_path or not audio_path.exists():
        raise RuntimeError(f"MUSIC_FATAL: ACE-Step completed without a downloadable audio file. Result={result!r}")
    target = base.AUDIO / "bhajan_source.mp3"
    target.parent.mkdir(parents=True, exist_ok=True)
    if audio_path.resolve() != target.resolve():
        shutil.copy2(audio_path, target)
    if target.stat().st_size < 20000:
        raise RuntimeError("MUSIC_FATAL: generated audio is suspiciously small")
    print("MUSIC_OK", target, target.stat().st_size)
    return target


base.generate_music = generate_music_gradio
base.ACESTEP_ROOT = "gradio-space:" + SPACE

if __name__ == "__main__":
    base.main()
    state_path = base.OUT / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["music_backend"] = "ACE-Step 1.5 official public Gradio Space"
        state["music_api_mode"] = "gradio_client_dynamic_endpoint"
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STATE_CORRECTED music_backend=ACE-Step 1.5 official public Gradio Space")
