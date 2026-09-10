"""Generate the Hindi bhajan through ACE-Step's hosted cloud API.

The official cloud endpoint at https://api.acemusic.ai exposes the
OpenAI-compatible /v1/chat/completions interface. The native /release_task
+ /query_result API is for ACE-Step server deployments and is not exposed by
the hosted cloud endpoint (it returns HTTP 404 there).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

BASE = os.getenv("ACESTEP_API_BASE_URL", "https://api.acemusic.ai").rstrip("/")
API_KEY = os.getenv("ACESTEP_API_KEY", "").strip()
# Cloud API model name uses v1.5; the self-hosted native API uses v15.
MODEL = os.getenv("ACESTEP_CLOUD_MODEL", "acemusic/acestep-v1.5-turbo").strip()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


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
    try:
        health = requests.get(
            BASE + "/health",
            headers={"Accept": "application/json", "User-Agent": "curl/8.4.0"},
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(f"ACESTEP_HEALTH_CONNECTION_FAILED: {e}") from e
    if not health.ok:
        raise RuntimeError(f"ACESTEP_HEALTH_ERROR: HTTP {health.status_code}: {health.text[:1000]}")

    print(f"ACESTEP_HEALTH=PASS HTTP={health.status_code}", flush=True)
    print(f"ACESTEP_MODEL={MODEL}", flush=True)
    print("ACESTEP_MODE=CLOUD_COMPLETIONS", flush=True)

    # The hosted acemusic.ai service exposes /v1/chat/completions, not the
    # local/server /release_task endpoint. Audio is returned as a base64 data
    # URL in choices[0].message.audio[].audio_url.url.
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"<prompt>{PROMPT}</prompt><lyrics>{LYRICS}</lyrics>",
            }
        ],
        "stream": False,
        "thinking": True,
        "use_format": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "audio_config": {
            "duration": seconds,
            "bpm": 128,
            "format": "mp3",
            "vocal_language": "hi",
        },
    }

    print("ACESTEP_GENERATING=TRUE", flush=True)
    print(f"ACESTEP_DURATION={seconds}s", flush=True)

    # Cloud generation is synchronous. Give the HTTP client a generous timeout;
    # the hosted service/proxy may take substantially longer than model health.
    request_timeout = int(os.getenv("ACESTEP_CLOUD_TIMEOUT", "660"))
    if request_timeout < 120:
        raise RuntimeError("ACESTEP_CLOUD_TIMEOUT must be at least 120 seconds")
    print(f"ACESTEP_CLOUD_TIMEOUT={request_timeout}s", flush=True)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        # The hosted service has historically handled curl-style UA reliably.
        "User-Agent": "curl/8.4.0",
    }

    try:
        response = requests.post(
            BASE + "/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=request_timeout,
        )
    except requests.Timeout as e:
        raise RuntimeError(
            "ACESTEP_CLOUD_TIMEOUT: hosted /v1/chat/completions did not return "
            f"within {request_timeout}s. This is a provider-side synchronous "
            "generation limit, not a GitHub Actions timeout."
        ) from e
    except requests.RequestException as e:
        raise RuntimeError(f"ACESTEP_CLOUD_CONNECTION_FAILED: {e}") from e

    if not response.ok:
        body = response.text[:3000]
        raise RuntimeError(
            f"ACESTEP_CLOUD_ERROR: HTTP {response.status_code}: {body}"
        )

    try:
        result = response.json()
    except ValueError as e:
        raise RuntimeError(f"ACESTEP_CLOUD_INVALID_JSON: {response.text[:2000]}") from e

    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"ACESTEP_CLOUD_NO_CHOICES: {result}")

    message = choices[0].get("message") or {}
    audio_items = message.get("audio") or []
    if not audio_items:
        raise RuntimeError(f"ACESTEP_CLOUD_NO_AUDIO: {result}")

    audio_url = ((audio_items[0].get("audio_url") or {}).get("url") or "").strip()
    if not audio_url.startswith("data:audio/"):
        raise RuntimeError(f"ACESTEP_CLOUD_INVALID_AUDIO_URL: {audio_url[:500]}")

    marker = ";base64,"
    if marker not in audio_url:
        raise RuntimeError("ACESTEP_CLOUD_AUDIO_NOT_BASE64")

    try:
        audio_bytes = base64.b64decode(audio_url.split(marker, 1)[1], validate=True)
    except Exception as e:
        raise RuntimeError(f"ACESTEP_CLOUD_AUDIO_DECODE_FAILED: {e}") from e

    dest = OUT / "bhajan_source.mp3"
    dest.write_bytes(audio_bytes)
    if dest.stat().st_size < 100_000:
        raise RuntimeError(f"ACESTEP_OUTPUT_TOO_SMALL: {dest.stat().st_size} bytes")

    print(f"ACESTEP_AUDIO_READY={dest} BYTES={dest.stat().st_size}", flush=True)
    print("ACESTEP_GENERATION=PASS", flush=True)


if __name__ == "__main__":
    main()
