# Bhajan Aabha v5 — Creative Architecture

## Goal

Replace the previous procedural/eSpeak output with a real devotional-song pipeline while keeping the software/infrastructure spend at ₹0.

## Backends

1. **Canonical deity image:** Agnes Image 2.1 Flash, generated once per bhajan.
2. **Motion:** Agnes Video v2.0 image-to-video, using the same canonical deity image for all three scenes so the deity identity is preserved.
3. **Singing + music:** ACE-Step 1.5 public ZeroGPU Space. Hindi lyrics are supplied explicitly with `vocal_language=hi` and a devotional production prompt.
4. **Composition:** FFmpeg on the GitHub-hosted runner.
5. **Delivery:** GitHub Release asset, not GitHub Actions artifact storage.

## Prototype structure

- 45-second song
- 3 × 15-second vertical deity scenes
- 720 × 1280 output
- Hindi sung vocal
- harmonium + tabla + dholak + bansuri + tanpura + temple-bell texture
- synchronized Hindi lyric captions
- AAC audio / H.264 video

## Zero-cost constraints

- Kaggle: **disabled**
- GPU rental: **disabled**
- paid API: **disabled**
- GitHub Actions artifacts: **disabled**
- Agnes uses its free/default API access; the workflow requires a free `AGNES_API_KEY` repository secret.
- ACE-Step uses its public ZeroGPU Space endpoint; no paid Hugging Face inference provider is used.

## Why this is different from v4

v4 generated procedural graphics and eSpeak narration. That architecture could validate FFmpeg automation, but it could not produce a convincing devotional music video. v5 treats the song audio and deity imagery as first-class generated assets and only uses FFmpeg for assembly.

## Operational rule

The v5 workflow is **manual-only until the prototype is visually and musically approved**. The existing daily workflow is intentionally not replaced yet. Once the prototype passes review, the daily workflow can be switched to v5 without another architectural rewrite.
