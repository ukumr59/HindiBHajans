# Bhajan Aabha — Autonomous Devotional Channel

Zero-cost-first autonomous devotional media system for the **Bhajan Aabha** YouTube channel.

## Hard requirements

- ₹0 recurring operating cost.
- No use of the user's computer CPU/GPU for production.
- Remote GitHub/Lightning/Agnes execution.
- No human intervention after one-time account authorization/setup.
- Original devotional music, lyrics and visuals.
- Automatic QA, retries and logging.
- Never silently introduce a paid dependency.

## Current production pipeline

**GitHub Actions** → **Lightning AI T4** for ACE-Step music → **Agnes Image/Video** for visuals → **FFmpeg assembly + QA** → **GitHub release** → **YouTube Data API**.

The current long-form workflow produces one validated 180-second vertical MP4 with a single full-length generated song and 12 visual scenes. The final-output validation explicitly ignores the intermediate scene clips, preventing the previous false `13 MP4` failure.

## YouTube publishing

YouTube publishing is now implemented in `app/youtube_publisher.py` and integrated into `.github/workflows/bhajan-aabha-v5-prototype.yml`.

The intended authentication model is **one-time OAuth + persistent refresh token**. GitHub Actions receives:

- `YOUTUBE_CLIENT_ID`
- `YOUTUBE_CLIENT_SECRET`
- `YOUTUBE_REFRESH_TOKEN`

The workflow exchanges the refresh token for a fresh access token automatically during each upload. No daily/weekly authorization-code entry is required unless Google revokes/invalidates the grant.

Complete setup instructions are in `docs/YOUTUBE_ONE_TIME_SETUP.md`.

## One-time setup helper

Run locally:

`python scripts/youtube_oauth_setup.py client_secret.json`

This launches the Google authorization page, obtains offline access, and writes a local `youtube_token.json`. Secrets are protected by `.gitignore` and must never be committed.

## Publishing defaults

- Visibility: public
- Category: Music
- Language: Hindi
- SEO title/description/tags: built into the publisher and focused on Bhajan Aabha / श्री राम / Hindi devotional search terms.
- Upload: resumable MP4 upload.

If the three YouTube secrets are not configured, the generation workflow still succeeds and explicitly skips the YouTube step rather than failing the production run.

## Workflows

- `bhajan-aabha-v5-prototype.yml` — current 3–5 minute production workflow; manually triggerable while the publishing setup is being verified.
- `bhajan-automation.yml` — legacy short-form automation path; it is intentionally not used for the long-form YouTube production path until that controller is migrated.

## Zero-cost policy

The system must stop rather than incur a charge. Do not add paid APIs, paid cloud compute, or credit-card-only services as dependencies. Free quotas are allowed only where they are genuinely free and do not auto-bill.
