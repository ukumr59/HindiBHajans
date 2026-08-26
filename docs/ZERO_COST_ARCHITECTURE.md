# Bhajan Aabha — ₹0 Architecture

## Hard constraints

- No Kaggle dependency.
- No paid GPU, AI API, storage, CDN, or automation service.
- The user's computer/GPU/CPU is never part of the production path.
- The system must fail closed rather than incur a charge.
- Trend data may be used for topic selection, but copyrighted recordings/footage are never downloaded, pitch-shifted, speed-shifted, remixed, or otherwise altered to evade copyright detection.

## Current production path

```text
GitHub Actions scheduler (23:00 Asia/Kolkata)
              |
              v
      Google Trends RSS
              |
              v
  Devotional topic ranking
              |
              v
     1–4 original concepts
              |
       +------+------+
       |             |
       v             v
 original music   procedural animation
  (pure Python)    (Pillow + FFmpeg)
       |             |
       +------+------+
              v
       vertical MP4 + SRT
              |
              v
      GitHub Actions artifact
```

## Free-GPU policy

A future optional GPU adapter may be added only when a provider has a genuinely usable free allocation. It must be an enhancement, never a dependency. If the GPU adapter is unavailable, the CPU pipeline continues. No paid fallback is permitted.

## Publishing

YouTube and Facebook publishing is intentionally separated from generation. It will be enabled only after the channel OAuth/Page credentials are configured. The publishing adapter must use official APIs and must not introduce a paid service.

## Output policy

Every run records:

- trend source and selected concepts
- number of generated videos
- generation mode
- copyright mode
- publish status

The workflow requires at least one MP4 and accepts up to four per run.
