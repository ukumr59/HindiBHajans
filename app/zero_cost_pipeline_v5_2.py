from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

import app.zero_cost_pipeline_v5 as base

# ACE-Step v1.5 now exposes a documented asynchronous HTTP API. Do not inspect
# or guess Gradio UI endpoints: UI endpoints such as /capture_current_params
# return state tuples rather than generated audio and are not a stable backend.
ACESTEP_API = base.ACESTEP_ROOT


def _result_audio_url(result: dict) -> str | None:
    if not isinstance(result, dict):
        return None
    candidates = []
    first = result.get("first_audio_path")
    if first:
        candidates.append(first)
    paths = result.get("audio_paths")
    if isinstance(paths, list):
        candidates.extend(paths)
    for value in candidates:
        if isinstance(value, str) and value:
            if value.startswith("http"):
                return value
            return urljoin(ACESTEP_API + "/", value.lstrip("/"))
    return None


def generate_music_http(session: requests.Session) -> Path:
    duration = int(base.VIDEO_SECONDS)
    print("MUSIC: ACE-Step 1.5 documented HTTP API")
    print("MUSIC: style=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128")

    payload = {
        "caption": base.PACK["music_prompt"],
        "lyrics": base.PACK["lyrics"],
        "vocal_language": "hi",
        "audio_format": "mp3",
        "bpm": 128,
        "key_scale": "C Major",
        "time_signature": "4",
        "audio_duration": duration,
        "model": "acestep-v15-turbo",
        "thinking": True,
        "use_format": True,
        "inference_steps": 8,
        "batch_size": 1,
        "use_random_seed": True,
        "task_type": "text2music",
    }

    data = base.http_json(
        session,
        "POST",
        f"{ACESTEP_API}/v1/music/generate",
        headers={"Content-Type": "application/json"},
        body=payload,
        timeout=120,
        retries=5,
    )
    job_id = data.get("job_id") if isinstance(data, dict) else None
    if not job_id:
        raise RuntimeError(f"MUSIC_FATAL: ACE-Step did not return job_id: {data}")
    print("MUSIC_JOB", job_id)

    deadline = time.time() + 30 * 60
    last_status = None
    while time.time() < deadline:
        status = base.http_json(
            session,
            "GET",
            f"{ACESTEP_API}/v1/jobs/{job_id}",
            timeout=60,
            retries=3,
        )
        state = str(status.get("status", "")).lower()
        if state != last_status:
            print(f"MUSIC: job={job_id} status={state} queue={status.get('queue_position')} eta={status.get('eta_seconds')}")
            last_status = state

        if state == "failed":
            raise RuntimeError(f"MUSIC_FATAL: ACE-Step job failed: {status.get('error', status)}")
        if state == "succeeded":
            result = status.get("result") or {}
            audio_url = _result_audio_url(result)
            if not audio_url:
                raise RuntimeError(f"MUSIC_FATAL: successful ACE-Step job has no audio path: {status}")
            target = base.AUDIO / "bhajan_source.mp3"
            base.download(session, audio_url, target, min_bytes=20000)
            print("MUSIC_OK", target, target.stat().st_size)
            return target

        time.sleep(3)

    raise RuntimeError(f"MUSIC_FATAL: ACE-Step job timed out after 30 minutes: {job_id}")


base.generate_music = generate_music_http
base.ACESTEP_ROOT = ACESTEP_API

if __name__ == "__main__":
    base.main()
    state_path = base.OUT / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["music_backend"] = "ACE-Step 1.5 documented HTTP API"
        state["music_api_mode"] = "http_async_v1_music_generate"
        state["music_style"] = "loud modern devotional EDM / DJ-ready"
        state["bpm"] = 128
        state["time_signature"] = "4/4"
        state["dj_master"] = str(base.AUDIO / "bhajan_aabha_dj_master.mp3")
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STATE_CORRECTED music_backend=ACE-Step 1.5 documented HTTP API")
