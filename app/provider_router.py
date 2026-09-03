"""Provider-neutral zero-cost GPU routing for HindiBHajans.

Providers expose a small HTTP contract. The router retries transient failures,
polls asynchronous jobs, and fails over to the next configured provider.

SATURN is supported natively because Saturn Cloud's API uses
Authorization: token <token> rather than Bearer authentication.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Provider:
    name: str
    endpoint: str
    token: str | None
    priority: int


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def load_providers() -> list[Provider]:
    providers: list[Provider] = []
    raw = _env("BH_PROVIDER_ORDER") or "SATURN,HF_SPACE,BEAM,KAGGLE"
    for priority, name in enumerate(x.strip().upper() for x in raw.split(",")):
        endpoint = _env(f"BH_{name}_ENDPOINT")
        if not endpoint:
            continue
        providers.append(Provider(name=name, endpoint=endpoint, token=_env(f"BH_{name}_TOKEN"), priority=priority))
    return providers


def _headers(provider: Provider, has_body: bool) -> dict[str, str]:
    h = {"Accept": "application/json"}
    if has_body:
        h["Content-Type"] = "application/json"
    if provider.token:
        h["Authorization"] = f"token {provider.token}" if provider.name == "SATURN" else f"Bearer {provider.token}"
    return h


def _request(url: str, provider: Provider, payload: dict[str, Any] | None = None, timeout: int = 45) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=body, headers=_headers(provider, payload is not None), method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _is_transient(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(error, (TimeoutError, urllib.error.URLError))


def _record(event: dict[str, Any]) -> None:
    with (STATE_DIR / "provider-events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def _saturn_job_urls(provider: Provider, result: dict[str, Any]) -> tuple[str | None, str | None]:
    """Map a Saturn start response into a polling URL and optional result URL."""
    job_id = result.get("job_id") or result.get("id")
    status_url = result.get("status_url")
    if not status_url and job_id:
        status_url = f"{provider.endpoint.rstrip('/')}/{job_id}"
    return status_url, job_id


def _poll(provider: Provider, result: dict[str, Any]) -> dict[str, Any]:
    status_url, job_id = _saturn_job_urls(provider, result)
    if not status_url:
        return result
    deadline = time.time() + int(os.getenv("BH_PROVIDER_TIMEOUT_SECONDS", "5400"))
    interval = max(15, int(os.getenv("BH_POLL_SECONDS", "30")))
    while time.time() < deadline:
        current = _request(str(status_url), provider)
        status = str(current.get("status", "")).lower()
        print(f"GPU_PROVIDER={provider.name} JOB={job_id} STATUS={status}", flush=True)
        if status in {"completed", "success", "succeeded", "finished"}:
            if not current.get("video_url"):
                raise RuntimeError("Provider reported success without video_url")
            return {**result, **current}
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(current.get("error") or f"Provider job failed: {status}")
        time.sleep(interval)
    raise TimeoutError(f"Provider job timed out after {os.getenv('BH_PROVIDER_TIMEOUT_SECONDS', '5400')}s")


def run_with_failover(job: dict[str, Any]) -> dict[str, Any]:
    providers = load_providers()
    if not providers:
        raise RuntimeError("NO_FREE_GPU_PROVIDER_CONFIGURED: configure at least one BH_<PROVIDER>_ENDPOINT")
    attempts = max(1, int(os.getenv("BH_PROVIDER_ATTEMPTS", "2")))
    backoff = [15, 45, 120]
    last_error: Exception | None = None
    for provider in providers:
        for attempt in range(1, attempts + 1):
            started = time.time()
            try:
                print(f"GPU_PROVIDER={provider.name} ATTEMPT={attempt}", flush=True)
                result = _request(provider.endpoint, provider, job)
                status = str(result.get("status", "")).lower()
                if status in {"queued", "running", "processing", "pending", "starting"} or (result.get("job_id") and not result.get("video_url")):
                    result = _poll(provider, result)
                if result.get("video_url"):
                    _record({"provider": provider.name, "attempt": attempt, "status": "success", "seconds": round(time.time() - started, 2)})
                    return {**result, "provider": provider.name}
                raise RuntimeError(result.get("error") or f"Provider returned status={status!r}")
            except Exception as exc:
                last_error = exc
                transient = _is_transient(exc)
                _record({"provider": provider.name, "attempt": attempt, "status": "error", "transient": transient, "error": repr(exc)})
                print(f"GPU_PROVIDER_ERROR={provider.name} transient={transient} error={exc}", flush=True)
                if not transient or attempt >= attempts:
                    break
                delay = backoff[min(attempt - 1, len(backoff) - 1)]
                print(f"GPU_PROVIDER_BACKOFF={delay}s", flush=True)
                time.sleep(delay)
        print(f"GPU_PROVIDER_FAILOVER={provider.name}", flush=True)
    raise RuntimeError(f"ALL_CONFIGURED_FREE_GPU_PROVIDERS_FAILED: {last_error}")


if __name__ == "__main__":
    print(json.dumps({"providers": [p.name for p in load_providers()]}, indent=2))
