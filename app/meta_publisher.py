"""Publish a finished Bhajan Aabha asset independently to Facebook Page and Instagram Reels."""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

GRAPH = os.getenv("META_GRAPH_VERSION", "v23.0")
BASE = f"https://graph.facebook.com/{GRAPH}"


def env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"MISSING_{name}")
    return value


def publish_facebook(video: Path, caption: str) -> str:
    page_id = env("META_PAGE_ID")
    token = env("META_PAGE_ACCESS_TOKEN")
    with video.open("rb") as fh:
        r = requests.post(
            f"{BASE}/{page_id}/videos",
            params={"access_token": token, "description": caption},
            files={"source": (video.name, fh, "video/mp4")},
            timeout=900,
        )
    r.raise_for_status()
    video_id = r.json().get("id")
    if not video_id:
        raise RuntimeError(f"FACEBOOK_PUBLISH_NO_ID: {r.text[:500]}")
    print(f"FACEBOOK_PUBLISHED={video_id}")
    return video_id


def publish_instagram(video_url: str, caption: str) -> str:
    user_id = env("META_IG_USER_ID")
    token = env("META_IG_ACCESS_TOKEN")
    create = requests.post(
        f"{BASE}/{user_id}/media",
        data={"media_type": "REELS", "video_url": video_url, "caption": caption, "share_to_feed": "true", "access_token": token},
        timeout=120,
    )
    create.raise_for_status()
    creation_id = create.json().get("id")
    if not creation_id:
        raise RuntimeError(f"INSTAGRAM_CONTAINER_NO_ID: {create.text[:500]}")

    deadline = time.time() + 900
    while time.time() < deadline:
        status = requests.get(
            f"{BASE}/{creation_id}",
            params={"fields": "status_code,status", "access_token": token},
            timeout=60,
        )
        status.raise_for_status()
        data = status.json()
        code = str(data.get("status_code", "")).upper()
        print(f"INSTAGRAM_CONTAINER_STATUS={code}")
        if code == "FINISHED":
            break
        if code in {"ERROR", "EXPIRED"}:
            raise RuntimeError(f"INSTAGRAM_CONTAINER_FAILED: {data}")
        time.sleep(20)
    else:
        raise TimeoutError("INSTAGRAM_CONTAINER_TIMEOUT")

    publish = requests.post(
        f"{BASE}/{user_id}/media_publish",
        data={"creation_id": creation_id, "access_token": token},
        timeout=120,
    )
    publish.raise_for_status()
    media_id = publish.json().get("id")
    if not media_id:
        raise RuntimeError(f"INSTAGRAM_PUBLISH_NO_ID: {publish.text[:500]}")
    print(f"INSTAGRAM_PUBLISHED={media_id}")
    return media_id


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--instagram-url", required=True)
    ap.add_argument("--caption", default="श्री राम जय राम 🙏 #BhajanAabha #Bhajan #Devotional")
    args = ap.parse_args()
    video = Path(args.video)
    if not video.exists():
        raise SystemExit("META_VIDEO_MISSING")
    publish_facebook(video, args.caption)
    publish_instagram(args.instagram_url, args.caption)


if __name__ == "__main__":
    main()
