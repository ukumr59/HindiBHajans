"""Headless video backend with explicit provider fallback and acceptance gates.

The production workflow must never substitute a still-image video when the
required visual transformation or lip-sync is unavailable. Providers are
ordered by VIDEO_PROVIDER_ORDER and are attempted with bounded retries. A
provider is accepted only if it produces both the required visual transformation
and audio-driven lip-sync; otherwise the next provider is tried.

This module is intentionally an adapter: provider-specific credentials/endpoints
are supplied through GitHub Secrets/Variables. It does not pretend that a
CPU slideshow satisfies the product requirements.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)
IMAGE = ROOT / "assets" / "uks model image.png"
AUDIO = OUT / "bhajan_source.mp3"
MASTER = OUT / "master.mp4"

ORDER = [x.strip().lower() for x in os.getenv(
    "VIDEO_PROVIDER_ORDER", "musetalk,wan,replicate"
).split(",") if x.strip()]
RETRIES = max(1, int(os.getenv("VIDEO_PROVIDER_RETRIES", "2")))
WAIT_SECONDS = max(1, int(os.getenv("VIDEO_PROVIDER_RETRY_WAIT_SECONDS", "30")))


def request_json(url: str, payload: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def run_adapter(provider: str) -> Path:
    """Run one configured provider adapter.

    Each adapter must expose an HTTP endpoint that accepts:
      image_path, audio_path, wardrobe_prompt, scene_prompt
    and eventually returns {"status":"completed", "video_url":"..."}.
    """
    prefix = provider.upper().replace("-", "_")
    endpoint = os.getenv(f"{prefix}_VIDEO_ENDPOINT", "").strip()
    token = os.getenv(f"{prefix}_VIDEO_TOKEN", "").strip() or None
    if not endpoint:
        raise RuntimeError(f"{provider}: endpoint not configured")

    payload = {
        "image_path": str(IMAGE),
        "audio_path": str(AUDIO),
        "wardrobe_prompt": "traditional Indian devotional clothing, modest, authentic Indian attire",
        "scene_prompt": "singer performing before a Hindu deity in a traditional Indian devotional setting",
        "require_identity_preservation": True,
        "require_lip_sync": True,
        "require_audio_sync": True,
    }
    job = request_json(endpoint, payload, token)
    job_url = job.get("status_url", endpoint)
    deadline = time.time() + int(os.getenv("VIDEO_PROVIDER_TIMEOUT_SECONDS", "900"))
    while time.time() < deadline:
        if job.get("status") == "completed" and job.get("video_url"):
            data = requests.get(job["video_url"], timeout=120)
            data.raise_for_status()
            candidate = OUT / f"candidate-{provider}.mp4"
            candidate.write_bytes(data.content)
            if candidate.stat().st_size < 100_000:
                raise RuntimeError(f"{provider}: output too small")
            return candidate
        if job.get("status") in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"{provider}: provider reported {job}")
        time.sleep(15)
        poll = requests.get(job_url, headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=60)
        poll.raise_for_status()
        job = poll.json()
    raise TimeoutError(f"{provider}: timed out")


def acceptance_gate(candidate: Path) -> None:
    """Require provider-declared transformation/lip-sync metadata plus media."""
    sidecar = candidate.with_suffix(".json")
    if not sidecar.exists():
        raise RuntimeError("ACCEPTANCE_GATE_FAIL: provider proof metadata missing")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    required = ["identity_preserved", "traditional_clothing", "deity_scene", "lip_sync"]
    missing = [k for k in required if meta.get(k) is not True]
    if missing:
        raise RuntimeError(f"ACCEPTANCE_GATE_FAIL: {missing}")


def main() -> None:
    if not IMAGE.exists() or not AUDIO.exists():
        raise RuntimeError("VIDEO_INPUT_GATE_FAIL")
    failures: list[str] = []
    for provider in ORDER:
        for attempt in range(1, RETRIES + 1):
            try:
                print(f"VIDEO_PROVIDER={provider} ATTEMPT={attempt}", flush=True)
                candidate = run_adapter(provider)
                acceptance_gate(candidate)
                candidate.replace(MASTER)
                print(f"VIDEO_ACCEPTED_PROVIDER={provider}", flush=True)
                return
            except Exception as exc:
                msg = f"{provider} attempt {attempt}: {exc}"
                failures.append(msg)
                print("VIDEO_PROVIDER_FAILURE=" + msg, flush=True)
                if attempt < RETRIES:
                    time.sleep(WAIT_SECONDS)
        print(f"VIDEO_FAILOVER_TO_NEXT={provider}", flush=True)
    raise RuntimeError("NO_VIDEO_PROVIDER_PASSED_ACCEPTANCE_GATE: " + " | ".join(failures))


if __name__ == "__main__":
    main()
