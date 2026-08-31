from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from app.youtube_publisher import upload


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m app.youtube_publisher_seo VIDEO")
    video = Path(sys.argv[1])
    seo_path = Path(os.getenv("OUTPUT_DIR", "output")) / "youtube_seo.json"
    if seo_path.exists():
        seo = json.loads(seo_path.read_text(encoding="utf-8"))
        if seo.get("title"):
            os.environ["YOUTUBE_TITLE"] = str(seo["title"])
        if seo.get("description"):
            os.environ["YOUTUBE_DESCRIPTION"] = str(seo["description"])
        if seo.get("tags"):
            os.environ["YOUTUBE_TAGS"] = ",".join(map(str, seo["tags"]))
    credits = Path(os.getenv("OUTPUT_DIR", "output")) / "pexels_credits.txt"
    if credits.exists():
        text = credits.read_text(encoding="utf-8").strip()
        if text:
            base_description = os.getenv("YOUTUBE_DESCRIPTION", "").rstrip()
            os.environ["YOUTUBE_DESCRIPTION"] = (
                base_description
                + "\n\n📹 Visual credits\n"
                + "The video uses individually selected Pexels stock footage, edited and synchronized with original Bhajan Aabha music and lyrics.\n"
                + text
            )[:5000]
    upload(video)


if __name__ == "__main__":
    main()
