from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_URI = "https://oauth2.googleapis.com/token"

DEFAULT_TITLE = "श्री राम जय राम | जय जय राम | नया हिंदी भजन | भजन आभा"
DEFAULT_DESCRIPTION = """🙏 भजन आभा में आपका स्वागत है।

श्री राम की भक्ति, शांति और सकारात्मक ऊर्जा से भरपूर यह मौलिक हिंदी भक्ति गीत आपके मन को सुकून और भक्ति से जोड़ने के लिए प्रस्तुत है।

🎵 भजन: श्री राम जय राम
🛕 चैनल: भजन आभा
🇮🇳 भाषा: हिंदी
🎧 शैली: आधुनिक भक्ति संगीत / Devotional Music

अगर आपको यह श्री राम भजन पसंद आए तो:
👍 वीडियो को Like करें
💬 Comment में 'जय श्री राम' लिखें
📲 परिवार और मित्रों के साथ Share करें
🔔 Bhajan Aabha को Subscribe करें और नए हिंदी भजनों के लिए Bell दबाएं।

ऐसे ही नए हिंदी भजन, राम भजन, भक्ति गीत और devotional songs के लिए Bhajan Aabha से जुड़े रहें।

जय श्री राम 🙏

#भजनआभा #श्रीराम #रामभजन #श्रीरामभजन #जयश्रीराम #हिंदीभजन #भक्ति #भक्तिगीत #रामनाम #भजन #DevotionalSongs #RamBhajan #HindiBhajan #BhaktiSong
"""

# SEO keyword set: Hindi + transliterated English + high-intent devotional queries.
# Kept within YouTube's 500-character tag limit by _tags().
DEFAULT_TAGS = [
    "bhajan aabha", "भजन आभा", "shri ram", "श्री राम", "ram bhajan", "राम भजन",
    "shri ram bhajan", "श्री राम भजन", "jai shree ram", "जय श्री राम",
    "shri ram jai ram", "श्री राम जय राम", "hindi bhajan", "हिंदी भजन",
    "new hindi bhajan", "नया हिंदी भजन", "bhakti geet", "भक्ति गीत",
    "devotional song", "devotional songs", "bhakti song", "राम नाम",
    "ram naam", "sanatan bhajan", "सनातन भजन", "morning bhajan", "सुबह का भजन",
    "devotional music", "hindu devotional song", "latest bhajan", "new bhajan",
]


def _env(name: str, required: bool = True) -> str:
    value = os.getenv(name, "").strip()
    if required and not value:
        raise RuntimeError(f"SETUP_REQUIRED: missing environment variable {name}")
    return value


def _youtube_client():
    client_id = _env("YOUTUBE_CLIENT_ID")
    client_secret = _env("YOUTUBE_CLIENT_SECRET")
    refresh_token = _env("YOUTUBE_REFRESH_TOKEN")
    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    credentials.refresh(Request())
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def _safe_title(title: str) -> str:
    title = re.sub(r"\\s+", " ", title).strip()
    return title[:100]


def _tags() -> list[str]:
    raw = os.getenv("YOUTUBE_TAGS", "").strip()
    values = [x.strip() for x in raw.split(",") if x.strip()] if raw else DEFAULT_TAGS
    out: list[str] = []
    total = 0
    for tag in values:
        if total + len(tag) + (1 if out else 0) > 500:
            break
        out.append(tag)
        total += len(tag) + (1 if len(out) > 1 else 0)
    return out


def upload(video_path: Path) -> str:
    if not video_path.exists() or video_path.stat().st_size < 100_000:
        raise RuntimeError(f"UPLOAD_FATAL: video missing or suspiciously small: {video_path}")

    youtube = _youtube_client()
    title = _safe_title(os.getenv("YOUTUBE_TITLE", DEFAULT_TITLE))
    description = os.getenv("YOUTUBE_DESCRIPTION", DEFAULT_DESCRIPTION).strip()
    privacy = os.getenv("YOUTUBE_PRIVACY_STATUS", "public").strip().lower()
    if privacy not in {"public", "private", "unlisted"}:
        raise RuntimeError("CONFIG_FATAL: YOUTUBE_PRIVACY_STATUS must be public, private, or unlisted")

    body = {
        "snippet": {
            "title": title,
            "description": description[:5000],
            "tags": _tags(),
            "categoryId": "10",
            "defaultLanguage": "hi",
            "defaultAudioLanguage": "hi",
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    print(f"YOUTUBE_SEO: title={title}")
    print(f"YOUTUBE_SEO: tags={len(_tags())} keywords; description={len(description)} chars; language=hi")
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        try:
            _, response = request.next_chunk()
        except HttpError as exc:
            if exc.resp.status in {500, 502, 503, 504}:
                continue
            raise

    video_id = response.get("id")
    if not video_id:
        raise RuntimeError(f"UPLOAD_FATAL: YouTube response did not contain video id: {response}")
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"YOUTUBE_UPLOAD_OK video_id={video_id}")
    print(f"YOUTUBE_URL={url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a validated Bhajan Aabha MP4 to YouTube with SEO metadata")
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    upload(args.video)


if __name__ == "__main__":
    main()
