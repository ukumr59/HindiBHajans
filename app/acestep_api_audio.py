"""Generate a Hindi bhajan through the ACE-Step 1.5 hosted HTTP API.

The GitHub runner is only the control plane: submit -> poll -> download -> validate.
No Kaggle, Hugging Face ZeroGPU, or Lightning GPU is required for this stage.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from urllib.parse import urljoin
import requests

BASE = os.getenv("ACESTEP_API_BASE_URL", "https://api.acemusic.ai").rstrip("/")
API_KEY = os.getenv("ACESTEP_API_KEY", "").strip()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def headers():
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def post(path, payload, timeout=60):
    r = requests.post(BASE + path, headers=headers(), json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if body.get("code", 200) != 200:
        raise RuntimeError(f"ACE_STEP_API_ERROR: {body.get('error') or body}")
    return body.get("data", body)


def get(path, timeout=120):
    r = requests.get(BASE + path, headers=headers(), timeout=timeout)
    r.raise_for_status()
    return r


def main():
    seconds = int(os.getenv("VIDEO_SECONDS", "180"))
    if not 180 <= seconds <= 300 or seconds % 15:
        raise RuntimeError("VIDEO_SECONDS must be 180-300 and divisible by 15")
    try:
        from app.generate_bhajan_audio import LYRICS, PROMPT
    except Exception as e:
        raise RuntimeError(f"Unable to load Hindi lyrics/prompt: {e}")

    print(f"ACESTEP_API_BASE={BASE}", flush=True)
    health = requests.get(BASE + "/health", headers=headers(), timeout=20)
    health.raise_for_status()
    print("ACESTEP_HEALTH=PASS", flush=True)

    payload = {
        "prompt": PROMPT,
        "lyrics": LYRICS,
        "thinking": True,
        "vocal_language": "hi",
        "audio_format": "mp3",
        "audio_duration": float(seconds),
        "bpm": 128,
        "key_scale": "C Major",
        "time_signature": "4",
        "model": "acestep-v15-turbo",
        "inference_steps": 8,
        "use_random_seed": True,
        "batch_size": 1,
        "task_type": "text2music",
        "use_cot_caption": False,
        "use_cot_language": False,
        "constrained_decoding": True,
    }
    print("ACESTEP_SUBMITTING=TRUE", flush=True)
    data = post("/release_task", payload)
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"ACESTEP_NO_TASK_ID: {data}")
    print(f"ACESTEP_TASK_ID={task_id}", flush=True)

    deadline = time.time() + int(os.getenv("ACESTEP_TIMEOUT_SECONDS", "900"))
    poll = 10
    while time.time() < deadline:
        result = post("/query_result", {"task_id_list": [task_id]})
        items = result if isinstance(result, list) else result.get("data", [])
        item = items[0] if items else {}
        status = int(item.get("status", 0))
        print(f"ACESTEP_STATUS={status}", flush=True)
        if status == 2:
            raise RuntimeError(f"ACESTEP_GENERATION_FAILED: {item}")
        if status == 1:
            raw = item.get("result", "[]")
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if not parsed or not isinstance(parsed[0], dict):
                raise RuntimeError(f"ACESTEP_SUCCESS_WITHOUT_AUDIO: {item}")
            file_ref = parsed[0].get("file", "")
            if not file_ref:
                raise RuntimeError(f"ACESTEP_AUDIO_URL_MISSING: {parsed[0]}")
            audio_url = file_ref if file_ref.startswith("http") else urljoin(BASE + "/", file_ref.lstrip("/"))
            print(f"ACESTEP_AUDIO_URL={audio_url.split('?')[0]}", flush=True)
            audio = requests.get(audio_url, headers=headers(), timeout=180)
            audio.raise_for_status()
            dest = OUT / "bhajan_source.mp3"
            dest.write_bytes(audio.content)
            if dest.stat().st_size < 100_000:
                raise RuntimeError("ACESTEP_OUTPUT_TOO_SMALL")
            print(f"ACESTEP_AUDIO_READY={dest} BYTES={dest.stat().st_size}", flush=True)
            return
        time.sleep(poll)
        poll = min(30, poll + 5)
    raise RuntimeError("ACESTEP_TIMEOUT: generation did not finish within configured timeout")


if __name__ == "__main__":
    main()
