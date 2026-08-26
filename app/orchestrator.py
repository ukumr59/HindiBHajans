"""Autonomous devotional content orchestrator.

This first-stage controller is deliberately conservative: it never spends money,
and it never uses the user's computer. Heavy generation is delegated to a remote
free compute Space. Until publishing credentials and the generator Space are
configured, it performs a dry run and exits safely.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone


def env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value else None


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    required = {
        "YOUTUBE_CLIENT_ID": env("YOUTUBE_CLIENT_ID"),
        "YOUTUBE_CLIENT_SECRET": env("YOUTUBE_CLIENT_SECRET"),
        "YOUTUBE_REFRESH_TOKEN": env("YOUTUBE_REFRESH_TOKEN"),
        "FACEBOOK_PAGE_ID": env("FACEBOOK_PAGE_ID"),
        "FACEBOOK_PAGE_ACCESS_TOKEN": env("FACEBOOK_PAGE_ACCESS_TOKEN"),
        "HF_TOKEN": env("HF_TOKEN"),
        "GENERATOR_SPACE": env("GENERATOR_SPACE"),
    }
    missing = [key for key, value in required.items() if not value]
    print(f"bhajan-automation heartbeat: {now}")
    if missing:
        print("SETUP_PENDING: missing secrets/configuration:")
        for key in missing:
            print(f"  - {key}")
        print("No paid service is called and no local compute is used.")
        return

    # Next stage: trend discovery -> opportunity scoring -> remote generation ->
    # QA -> YouTube/Facebook publishing -> analytics feedback.
    print("READY: all control-plane credentials are present.")
    print("Generator dispatch will be enabled in the next deployment stage.")


if __name__ == "__main__":
    main()
