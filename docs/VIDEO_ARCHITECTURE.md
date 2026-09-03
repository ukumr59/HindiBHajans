# Video architecture

## Production contract

A publishable master MUST contain:

1. The approved singer identity.
2. Traditional Indian clothing.
3. The approved devotional/deity setting.
4. Audio-driven lip synchronization to the generated Hindi bhajan.
5. Valid video and audio streams.

A still-image animation, slideshow, zoom/pan, or unsynchronized mouth movement is **not** an acceptable fallback.

## Pipeline

```text
Approved singer avatar/reference
        +
ACE-Step 1.5 Hindi bhajan audio
        |
        v
MuseTalk 1.5 lip-sync GPU job
        |
        v
Acceptance gate
  identity + clothing + deity + lip-sync
        |
        +---- FAIL -> retry / next GPU provider / stop
        |
       PASS
        |
        v
master.mp4 -> Shorts -> publishing
```

## Avatar-first rule

The clothing/scene transformation should be performed once to create an approved reusable singer avatar/reference. Daily jobs should perform lip-sync against that approved visual asset. This avoids spending GPU time regenerating wardrobe/scene unnecessarily.

## GPU policy

GitHub Actions is the orchestrator, not the inference engine. The production workflow must call a configured GPU provider for MuseTalk 1.5. Kaggle, Lightning AI, and Hugging Face ZeroGPU are not assumed to be reliable production dependencies.

## Fail-closed policy

If no configured provider produces a candidate with provider proof metadata confirming all four visual requirements, the workflow fails and publishing is skipped. Never manufacture a fake successful video from a still image.
