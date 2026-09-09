"""Generate the Hindi bhajan through the ACE-Step OpenRouter-compatible API.

The GitHub runner is only the control plane. No GPU is used here.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

BASE = os.getenv("ACESTEP_API_BASE_URL", "https://api.acemusic.ai").rstrip("/")
API_KEY = os.getenv("ACESTEP_API_KEY", "").strip()
MODEL = os.getenv("ACESTEP_MODEL", "acemusic/acestep-v1.5-turbo").strip()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def headers() -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "HindiBHajans/1.0",
    }
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
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

    # Do not make generation depend on model-list discovery. The hosted
    # OpenRouter-compatible API documents this model id directly. Older code
    # incorrectly called /v1/models; the OpenRouter API exposes model listing
    # at /api/v1/models, and model discovery is unnecessary for generation.
    print(f"ACESTEP_MODEL={MODEL}", flush=True)

    # Use the documented OpenRouter-compatible request shape: prompt/lyrics
    # are tagged in the user message and generation controls are top-level.
    content = f"<prompt>{PROMPT}</prompt><lyrics>{LYRICS}</lyrics>"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "thinking": True,
        "lyrics": LYRICS,
        "duration": float(seconds),
        "bpm": 128,
        "vocal_language": "hi",
        "instrumental": False,
        "use_format": False,
        "use_cot_caption": False,
        "use_cot_language": False,
    }

    print("ACESTEP_SUBMITTING=TRUE", flush=True)
    try:
        response = requests.post(
            BASE + "/v1/chat/completions",
            headers=headers(),
            json=payload,
            timeout=max(900, seconds * 5),
        )
    except requests.RequestException as e:
        raise RuntimeError(f"ACESTEP_API_CONNECTION_FAILED: {e}") from e

    if not response.ok:
        raise RuntimeError(
            f"ACESTEP_API_ERROR: HTTP {response.status_code}: {response.text[:2000]}"
        )

    try:
        body = response.json()
    except ValueError as e:
        raise RuntimeError(f"ACESTEP_INVALID_JSON_RESPONSE: {response.text[:1000]}") from e

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
