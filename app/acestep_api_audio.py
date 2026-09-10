"""Generate the Hindi bhajan through ACE-Step's asynchronous hosted API.

The GitHub runner is only the control plane. No GPU is used here.
The synchronous /v1/chat/completions endpoint can exceed the Cloudflare
request window for long vocal generations, so production uses the native
/release_task + /query_result workflow instead.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE = os.getenv("ACESTEP_API_BASE_URL", "https://api.acemusic.ai").rstrip("/")
API_KEY = os.getenv("ACESTEP_API_KEY", "").strip()
MODEL = os.getenv("ACESTEP_NATIVE_MODEL", "acestep-v15-turbo").strip()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def headers() -> dict[str, str]:
    h = {
        "Accept": "application/json",
        "User-Agent": "HindiBHajans/1.0",
    }
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def json_headers() -> dict[str, str]:
    h = headers()
    h["Content-Type"] = "application/json"
    return h


def checked_get(url: str, label: str) -> requests.Response:
    try:
        r = requests.get(url, headers=headers(), timeout=30)
    except requests.RequestException as e:
        raise RuntimeError(f"ACESTEP_{label}_CONNECTION_FAILED: {e}") from e
    if not r.ok:
        raise RuntimeError(f"ACESTEP_{label}_ERROR: HTTP {r.status_code}: {r.text[:1200]}")
    return r


def main() -> None:
    seconds = int(os.getenv("VIDEO_SECONDS", "180"))
    if not 180 <= seconds <= 300 or seconds % 15:
        raise RuntimeError("VIDEO_SECONDS must be 180-300 and divisible by 15")
    if not API_KEY:
        raise RuntimeError("ACESTEP_API_KEY is not configured")

    try:
        from app.generate_bhajan_audio import LYRICS, PROMPT
    except Exception as e:
        raise RuntimeError(f"Unable to load Hindi lyrics/prompt: {e}") from e

    print(f"ACESTEP_API_BASE={BASE}", flush=True)
    health = checked_get(BASE + "/health", "HEALTH")
    print(f"ACESTEP_HEALTH=PASS HTTP={health.status_code}", flush=True)
    print(f"ACESTEP_MODEL={MODEL}", flush=True)
    print("ACESTEP_MODE=ASYNC_NATIVE", flush=True)

    # ACE-Step native API: submit immediately, then poll the task. This avoids
    # keeping a Cloudflare-proxied HTTP request open for the whole generation.
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "lyrics": LYRICS,
        "task_type": "text2music",
        "audio_duration": float(seconds),
        "audio_format": "mp3",
        "vocal_language": "hi",
        "bpm": 128,
        "batch_size": 1,
        "thinking": True,
        "use_format": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "inference_steps": 8,
        "infer_method": "ode",
    }

    print("ACESTEP_SUBMITTING=TRUE", flush=True)
    # The hosted service can take longer than 30s to initialize the model/LM
    # before it returns the task id. Keep this timeout separate from polling:
    # once a task id is returned, /query_result remains a short request.
    submit_timeout = int(os.getenv("ACESTEP_SUBMIT_TIMEOUT", "240"))
    if submit_timeout < 60:
        raise RuntimeError("ACESTEP_SUBMIT_TIMEOUT must be at least 60 seconds")
    print(f"ACESTEP_SUBMIT_TIMEOUT={submit_timeout}s", flush=True)
    try:
        response = requests.post(
            BASE + "/release_task",
            headers=json_headers(),
            json=payload,
            timeout=submit_timeout,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"ACESTEP_SUBMIT_CONNECTION_FAILED: {e}") from e

    if not response.ok:
        raise RuntimeError(
            f"ACESTEP_SUBMIT_ERROR: HTTP {response.status_code}: {response.text[:2000]}"
        )

    try:
        submitted = response.json()
    except ValueError as e:
        raise RuntimeError(f"ACESTEP_SUBMIT_INVALID_JSON: {response.text[:1000]}") from e

    if submitted.get("code") not in (None, 200):
        raise RuntimeError(f"ACESTEP_SUBMIT_REJECTED: {submitted}")

    data = submitted.get("data") or {}
    task_id = data.get("task_id") if isinstance(data, dict) else None
    if not task_id:
        raise RuntimeError(f"ACESTEP_NO_TASK_ID: {submitted}")

    print(f"ACESTEP_TASK_ID={task_id}", flush=True)
    print("ACESTEP_TASK_ACCEPTED=TRUE", flush=True)

    # Allow a long-running hosted generation without holding a single request
    # open. 20 minutes is intentionally below the GitHub job's 720-minute cap.
    deadline = time.monotonic() + 20 * 60
    last_status = None
    result_items = None

    while time.monotonic() < deadline:
        try:
            q = requests.post(
                BASE + "/query_result",
                headers=json_headers(),
                json={"task_id_list": [task_id]},
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"ACESTEP_POLL_CONNECTION_RETRY={e}", flush=True)
            time.sleep(15)
            continue

        if not q.ok:
            raise RuntimeError(
                f"ACESTEP_POLL_ERROR: HTTP {q.status_code}: {q.text[:2000]}"
            )

        try:
            status_body = q.json()
        except ValueError as e:
            raise RuntimeError(f"ACESTEP_POLL_INVALID_JSON: {q.text[:1000]}") from e

        items = status_body.get("data") or []
        item = items[0] if isinstance(items, list) and items else {}
        status = item.get("status", 0)
        if status != last_status:
            print(f"ACESTEP_STATUS={status}", flush=True)
            last_status = status

        if status == 1:
            raw_result = item.get("result", "[]")
            try:
                result_items = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            except ValueError as e:
                raise RuntimeError(f"ACESTEP_RESULT_INVALID_JSON: {raw_result[:2000]}") from e
            break

        if status == 2:
            raise RuntimeError(f"ACESTEP_GENERATION_FAILED: {item}")

        time.sleep(10)

    if result_items is None:
        raise RuntimeError("ACESTEP_GENERATION_TIMEOUT: task did not finish within 20 minutes")

    if not isinstance(result_items, list) or not result_items:
        raise RuntimeError(f"ACESTEP_EMPTY_RESULT: {result_items}")

    audio_path = result_items[0].get("file", "")
    if not audio_path:
        raise RuntimeError(f"ACESTEP_RESULT_HAS_NO_AUDIO_FILE: {result_items[0]}")

    audio_url = urljoin(BASE + "/", audio_path.lstrip("/"))
    print(f"ACESTEP_DOWNLOADING_AUDIO={audio_url}", flush=True)
    try:
        audio_response = requests.get(
            audio_url,
            headers=headers(),
            timeout=180,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"ACESTEP_AUDIO_DOWNLOAD_FAILED: {e}") from e

    if not audio_response.ok:
        raise RuntimeError(
            f"ACESTEP_AUDIO_DOWNLOAD_ERROR: HTTP {audio_response.status_code}: {audio_response.text[:1200]}"
        )

    dest = OUT / "bhajan_source.mp3"
    dest.write_bytes(audio_response.content)
    if dest.stat().st_size < 100_000:
        raise RuntimeError(f"ACESTEP_OUTPUT_TOO_SMALL: {dest.stat().st_size} bytes")

    print(f"ACESTEP_AUDIO_READY={dest} BYTES={dest.stat().st_size}", flush=True)
    print("ACESTEP_GENERATION=PASS", flush=True)


if __name__ == "__main__":
    main()
