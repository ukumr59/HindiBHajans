# Bhajan Aabha — YouTube one-time OAuth setup

The repository now uses a **Google OAuth refresh token**, not a daily authorization code. The GitHub Actions runner refreshes the short-lived access token automatically when it uploads a video.

## Important

Google refresh tokens are long-lived credentials. Google documents that they can be used to obtain new access tokens without the user being present. If an OAuth consent screen is left in **Testing** status, refresh tokens can be time-limited; move the app to **In production** before relying on it for continuous unattended publishing. citehttps://developers.google.com/health/setup

## One-time setup

1. Open Google Cloud Console and create/select a project dedicated to Bhajan Aabha.
2. Enable **YouTube Data API v3**.
3. Configure the OAuth consent screen.
4. Set the publishing status to **In production**. Do not leave a production automation app in Testing mode, because Testing refresh tokens can expire after 7 days. citehttps://developers.google.com/health/setup
5. Create an OAuth client of type **Desktop app**.
6. Download the client JSON to your computer as something like `client_secret.json`.
7. From the repository root, install the requirements and run:

   `python scripts/youtube_oauth_setup.py client_secret.json`

8. A browser window will open. Sign into the Google account that owns the **Bhajan Aabha** YouTube channel and grant the requested YouTube upload permission.
9. The script creates `youtube_token.json` locally and prints three values:
   - `YOUTUBE_CLIENT_ID`
   - `YOUTUBE_CLIENT_SECRET`
   - `YOUTUBE_REFRESH_TOKEN`
10. Add those three values as **GitHub repository Actions secrets** with exactly those names.
11. Never commit `client_secret.json` or `youtube_token.json`. The repository `.gitignore` already protects both.

## What happens after setup

Every successful long-form generation will:

`Generate → Validate → GitHub release → YouTube upload`

The publisher uses the stored refresh token to obtain a fresh access token automatically. No daily or weekly OAuth-code entry is required unless Google revokes/invalidates the refresh token or the user revokes the app's access. citehttps://developers.google.com/youtube/v3/guides/auth/installed-apps

## Current YouTube defaults

- Visibility: `public`
- Category: Music (`10`)
- Language: Hindi (`hi`)
- Title/description/tags: Bhajan Aabha / श्री राम SEO defaults
- Upload method: resumable MP4 upload

The workflow skips the YouTube step safely until the three YouTube secrets are configured; it does not block video generation.
