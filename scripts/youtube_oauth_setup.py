from __future__ import annotations

import argparse
import json
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time Google OAuth setup for Bhajan Aabha YouTube publishing")
    parser.add_argument("client_secret_json", type=Path, help="Google OAuth Desktop App client_secret JSON")
    parser.add_argument("--output", type=Path, default=Path("youtube_token.json"))
    args = parser.parse_args()

    if not args.client_secret_json.exists():
        raise SystemExit(f"Client secret file not found: {args.client_secret_json}")

    flow = InstalledAppFlow.from_client_secrets_file(str(args.client_secret_json), SCOPES)
    credentials = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    payload = {
        "refresh_token": credentials.refresh_token,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": SCOPES,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved local token bundle: {args.output.resolve()}")
    print("YOUTUBE_CLIENT_ID=" + credentials.client_id)
    print("YOUTUBE_CLIENT_SECRET=" + credentials.client_secret)
    print("YOUTUBE_REFRESH_TOKEN=" + (credentials.refresh_token or ""))
    print("Do NOT commit this file or paste these values into public chat/issues.")


if __name__ == "__main__":
    main()
