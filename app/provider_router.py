"""Provider-neutral zero-cost GPU routing for HindiBHajans.

The pipeline never depends on one vendor. Providers expose the same HTTP contract:
POST endpoint with a JSON job -> {status, job_id, video_url, audio_url, error}.
A provider may be unavailable, rate-limited, or out of quota without breaking the
job; the router retries transient failures and fails over to the next provider.

Only endpoints explicitly configured through environment variables are eligible.
There is deliberately no paid-provider fallback.
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
    raw = _env("BH_PROVIDER_ORDER") or "MODAL,HF_SPACE,BEAM,KAGGLE"
    for priority, name in enumerate(x.strip().upper() for x in raw.split(",")):
        endpoint = _env(f"BH_{name}_ENDPOINT")
        if not endpoint:
            continue
        providers.append(
            Provider(
                name=name,
                endpoint=endpoint,
                token=_env(f"BH_{name}_TOKEN"),
                priority=priority,
            )
        )
    return providers


def _post(provider: Provider, payload: dict[str, Any], timeout: int = 45) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if provider.token:
        headers["Authorization"] = f"Bearer {provider.token}"
    req = urllib.request.Request(provider.endpoint, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _is_transient(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 425, 429, 500, 502, 503, 504}
    if isinstance(error, (TimeoutError, urllib.error.URLError)):
        return True
    return False


def _record(event: dict[str, Any]) -> None:
    path = STATE_DIR / "provider-events.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_with_failover(job: dict[str, Any]) -> dict[str, Any]:
    providers = load_providers()
    if not providers:
        raise RuntimeError(
            "NO_FREE_GPU_PROVIDER_CONFIGURED: set at least one BH_<PROVIDER>_ENDPOINT "
            "and keep BH_PROVIDER_ORDER restricted to free providers."
        )

    same_provider_attempts = max(1, int(os.getenv("BH_PROVIDER_ATTEMPTS", "2")))
    backoff = [15, 45, 120]
    last_error: Exception | None = None

    for provider in providers:
        for attempt in range(1, same_provider_attempts + 1):
            started = time.time()
            try:
                print(f"GPU_PROVIDER={provider.name} ATTEMPT={attempt}", flush=True)
                response = _post(provider, job)
                status = str(response.get("status", "")).lower()
                if status in {"completed", "success", "succeeded"} and response.get("video_url"):
                    _record({"provider": provider.name, "attempt": attempt, "status": "success", "seconds": round(time.time() - started, 2)})
                    return {**response, "provider": provider.name}
                if status in {"queued", "running", "processing"} and response.get("job_id"):
                    _record({"provider": provider.name, "attempt": attempt, "status": status, "seconds": round(time.time() - started, 2)})
                    return {**response, "provider": provider.name}
                raise RuntimeError(response.get("error") or f"Provider returned status={status!r}")
            except Exception as exc:
                last_error = exc
                transient = _is_transient(exc)
                _record({"provider": provider.name, "attempt": attempt, "status": "error", "transient": transient, "error": repr(exc)})
                print(f"GPU_PROVIDER_ERROR={provider.name} transient={transient} error={exc}", flush=True)
                if not transient:
                    break
                if attempt < same_provider_attempts:
                    delay = backoff[min(attempt - 1, len(backoff) - 1)]
                    print(f"GPU_PROVIDER_BACKOFF={delay}s", flush=True)
                    time.sleep(delay)
        print(f"GPU_PROVIDER_FAILOVER={provider.name}", flush=True)

    raise RuntimeError(f"ALL_CONFIGURED_FREE_GPU_PROVIDERS_FAILED: {last_error}")


if __name__ == "__main__":
    result = run_with_failover({"action": "health"})
    print(json.dumps(result, indent=2))
