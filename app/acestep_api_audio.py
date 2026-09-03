"""Generate the Hindi bhajan through the ACE-Step official cloud completion API.

The GitHub runner is only the control plane: validate -> submit one synchronous
completion request -> decode the returned base64 MP3 -> validate.  No Kaggle,
Hugging Face ZeroGPU, or Lightning GPU is used for audio generation.

ACE-Step's current cloud interface exposes OpenAI-compatible
/v1/chat/completions.  The older /release_task endpoint is a native/local API
and is not the correct endpoint for the hosted cloud service.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

BASE = os.getenv("ACESTEP_API_BASE_URL", "https://api.acemusic.ai").rstrip("/")
API_KEY = os.getenv("ACESTEP_API_KEY", "").strip()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def headers() -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "curl/8.4.0",
    }
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


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
    health = requests.get(BASE + "/health", headers=headers(), timeout=30)
    health.raise_for_status()
    print("ACESTEP_HEALTH=PASS", flush=True)

    # The hosted service currently uses the OpenAI-compatible completion API.
    # Resolve the model id from the service rather than hard-coding a possibly
    # stale model name.
    models = requests.get(BASE + "/v1/models", headers=headers(), timeout=30)
    models.raise_for_status()
    model_body = models.json()
    model_items = model_body.get("data", []) if isinstance(model_body, dict) else []
    model_id = model_items[0].get("id") if model_items and isinstance(model_items[0], dict) else None
    model_id = model_id or "acemusic/acestep-v15-turbo"
    print(f"ACESTEP_MODEL={model_id}", flush=True)

    # Tagged mode explicitly separates the music description from the Hindi
    # lyrics. This is the documented completion-mode format for vocal songs.
    content = f"<prompt>{PROMPT}</prompt><lyrics>{LYRICS}</lyrics>"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": ["audio"],
        "thinking": True,
        "use_format": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "lyrics": LYRICS,
        "task_type": "text2music",
        "batch_size": 1,
        "audio_config": {
            "duration": float(seconds),
            "bpm": 128,
            "format": "mp3",
            "vocal_language": "hi",
            "instrumental": False,
            "key_scale": "C Major",
            "time_signature": "4/4",
        },
    }

    print("ACESTEP_SUBMITTING=TRUE", flush=True)
    response = requests.post(
        BASE + "/v1/chat/completions",
        headers=headers(),
        json=payload,
        timeout=max(900, seconds * 5),
    )
    if not response.ok:
        # Do not print request headers or the API key. The response body is
        # useful for diagnosing quota/auth/validation failures.
        raise RuntimeError(
            f"ACESTEP_API_ERROR: HTTP {response.status_code}: {response.text[:2000]}"
        )

    body = response.json()
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"ACESTEP_NO_CHOICES: {body}")
    message = choices[0].get("message", {})
    audio_items = message.get("audio", []) or []
    if not audio_items:
        raise RuntimeError(f"ACESTEP_NO_AUDIO: {message.get('content', '')[:1000]}")

    audio_url = audio_items[0].get("audio_url", {}).get("url", "")
    if not audio_url.startswith("data:audio/") or "," not in audio_url:
        raise RuntimeError("ACESTEP_INVALID_AUDIO_DATA_URL")

    try:
        encoded = audio_url.split(",", 1)[1]
        audio_bytes = base64.b64decode(encoded, validate=True)
    except Exception as e:
        raise RuntimeError(f"ACESTEP_AUDIO_DECODE_FAILED: {e}") from e

    dest = OUT / "bhajan_source.mp3"
    dest.write_bytes(audio_bytes)
    if dest.stat().st_size < 100_000:
        raise RuntimeError(f"ACESTEP_OUTPUT_TOO_SMALL: {dest.stat().st_size} bytes")

    print(f"ACESTEP_AUDIO_READY={dest} BYTES={dest.stat().st_size}", flush=True)
    print("ACESTEP_GENERATION=PASS", flush=True)


if __name__ == "__main__":
    main()
