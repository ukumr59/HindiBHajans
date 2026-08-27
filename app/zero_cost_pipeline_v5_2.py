from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import app.zero_cost_pipeline_v5 as base

# ACE-Step's official public HF deployment is a Gradio ZeroGPU Space, not the
# standalone FastAPI server. The standalone /release_task API exists in the
# ACE-Step source, but the public Space serves the Gradio application. Calling
# /release_task on the public .hf.space host therefore returns HTTP 405.
# Use the supported Gradio client against the actual public Space instead.
ACE_STEP_SPACE = "ACE-Step/Ace-Step-v1.5"

DJ_MUSIC_PROMPT = """Modern high-energy Hindi devotional bhajan made like a current YouTube DJ devotional song, 128 BPM, 4/4, loud polished commercial stereo production, powerful expressive Hindi male lead vocal clearly singing every lyric with natural emotion and clean pronunciation, catchy devotional melody, huge memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, modern synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents, harmonium texture, short instrumental intro, strong verse build, massive chorus/drop, rhythmic instrumental break, final chorus with layered backing vocals, professional YouTube/radio loudness and DJ playback energy. The devotional identity must remain unmistakable while the production feels contemporary, energetic and danceable. NOT meditation music, NOT sleepy, NOT ambient, NOT acoustic-only, NOT spoken narration, NOT chanting without melody, NOT humming, NOT a cappella, NOT background music, NOT an instrumental-only track."""


def _norm(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _endpoint_parameters(info: dict) -> list[dict]:
    return list(info.get("parameters") or [])


def _find_generation_endpoint(api: dict) -> tuple[str, dict]:
    named = api.get("named_endpoints") or {}
    candidates = []
    for name, info in named.items():
        text = _norm(name) + " " + " ".join(_norm(p.get("label")) for p in _endpoint_parameters(info))
        score = 0
        if "generate" in text:
            score += 10
        if "music" in text:
            score += 8
        if "caption" in text:
            score += 2
        if "lyrics" in text:
            score += 2
        if score:
            candidates.append((score, name, info))
    if not candidates:
        raise RuntimeError(f"MUSIC_FATAL: public ACE-Step Gradio API has no generation endpoint. API={api}")
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def _choose_value(param: dict):
    name = _norm(param.get("parameter_name") or param.get("label"))
    label = _norm(param.get("label"))
    text = f"{name} {label}"
    default = param.get("parameter_default")
    has_default = bool(param.get("parameter_has_default"))

    # Preserve harmless defaults for parameters that are not relevant to this
    # song. Required fields are still explicitly populated below.
    if "caption" in text or "prompt" in text and "negative" not in text:
        return DJ_MUSIC_PROMPT
    if "lyrics" in text:
        return base.PACK["lyrics"]
    if "vocal_language" in text or ("language" in text and "negative" not in text):
        return "hi"
    if text.strip() in {"bpm", "tempo"} or "bpm" in text:
        return 128
    if "key_scale" in text or "keyscale" in text:
        return "C Major"
    if "time_signature" in text or "timesignature" in text:
        return "4"
    if "audio_duration" in text or ("duration" in text and "audio" in text):
        return int(base.VIDEO_SECONDS)
    if "inference_steps" in text:
        return 8
    if "guidance_scale" in text:
        return 7.0
    if "batch_size" in text:
        return 1
    if "audio_format" in text:
        return "mp3"
    if "task_type" in text:
        return "text2music"
    if "model" in text and "lm_" not in text and "negative" not in text:
        return "acestep-v15-turbo"
    if text == "thinking" or text.endswith("_thinking") or "think" in text:
        return True
    if "use_format" in text:
        return True
    if "random_seed" in text or "use_random_seed" in text:
        return True
    if text == "seed" or text.endswith("_seed"):
        return -1
    if "shift" in text:
        return 3.0
    if "infer_method" in text:
        return "ode"
    if "instrumental" in text:
        return False
    if "negative_prompt" in text or "negative" in text:
        return "sleepy ambient, meditation, humming, spoken narration, a cappella, weak vocals, acoustic-only, muddy bass, distorted clipping"
    if "sample_mode" in text:
        return False
    if "src_audio" in text or "reference_audio" in text or "audio_file" in text:
        return None
    if "state" in text:
        return None
    if has_default:
        return default
    return None


def _extract_audio(value, seen=None) -> str | None:
    if seen is None:
        seen = set()
    if id(value) in seen:
        return None
    seen.add(id(value))
    if isinstance(value, str):
        low = value.lower()
        if low.endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            return value
        return None
    if isinstance(value, dict):
        for key in ("path", "url", "name"):
            if key in value:
                found = _extract_audio(value[key], seen)
                if found:
                    return found
        for v in value.values():
            found = _extract_audio(v, seen)
            if found:
                return found
    if isinstance(value, (list, tuple)):
        for v in value:
            found = _extract_audio(v, seen)
            if found:
                return found
    return None


def generate_music_gradio() -> Path:
    from gradio_client import Client

    print("MUSIC: connecting to official ACE-Step v1.5 public ZeroGPU Space")
    print("MUSIC: using Gradio Client API — not the Space's nonexistent REST endpoint")
    print("MUSIC: style=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128")

    client = Client(ACE_STEP_SPACE, max_workers=1)
    api = client.view_api(print_info=False, return_format="dict")
    endpoint, endpoint_info = _find_generation_endpoint(api)
    params = _endpoint_parameters(endpoint_info)
    print(f"MUSIC: selected Gradio endpoint={endpoint}")
    print("MUSIC: endpoint parameters=" + ", ".join(str(p.get("parameter_name") or p.get("label")) for p in params))

    kwargs = {}
    for p in params:
        key = p.get("parameter_name") or p.get("label")
        if key:
            kwargs[key] = _choose_value(p)

    # Let Gradio validate the live endpoint schema. This is deliberately a
    # keyword call so changes in UI input order do not silently corrupt values.
    job = client.submit(api_name=endpoint, **kwargs)
    result = job.result()
    print("MUSIC: Gradio generation completed")
    audio_ref = _extract_audio(result)
    if not audio_ref:
        raise RuntimeError(f"MUSIC_FATAL: Gradio generation returned no downloadable audio: {result}")

    target = base.AUDIO / "bhajan_source.mp3"
    source = Path(audio_ref)
    if source.exists():
        shutil.copy2(source, target)
    elif str(audio_ref).startswith("http"):
        base.download(client.session if hasattr(client, "session") else __import__("requests").Session(), str(audio_ref), target, min_bytes=20000)
    else:
        raise RuntimeError(f"MUSIC_FATAL: returned audio path is not accessible: {audio_ref}")
    if target.stat().st_size < 20000:
        raise RuntimeError("MUSIC_FATAL: generated audio is suspiciously small")
    print("MUSIC_OK", target, target.stat().st_size)
    return target


def make_dj_master(final_video: Path) -> Path:
    target = base.AUDIO / "bhajan_aabha_dj_master.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(final_video), "-vn",
        "-af", "highpass=f=28,lowpass=f=19000,loudnorm=I=-9:TP=-1.0:LRA=7",
        "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", str(target),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("DJ_MASTER_FATAL: " + p.stderr[-4000:])
    if target.stat().st_size < 50000:
        raise RuntimeError("DJ_MASTER_FATAL: master file is suspiciously small")
    print("DJ_MASTER_OK", target, target.stat().st_size)
    return target


# base.main() resolves generate_music from the base module, so patch that
# symbol rather than creating an unused local implementation.
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
        state["music_backend"] = "ACE-Step v1.5 official Hugging Face ZeroGPU Space via Gradio Client"
        state["music_api_mode"] = "gradio_client_live_api"
        state["music_style"] = "LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY"
        state["bpm"] = 128
        state["time_signature"] = "4/4"
        state["dj_master"] = str(dj_master)
        state["dj_master_bitrate"] = "320k"
        state["dj_master_sample_rate"] = 48000
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STATE_OK music_backend=ACE-Step official ZeroGPU Space via Gradio Client")
