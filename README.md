# Bhajan Aabha — Autonomous Devotional Channel

Zero-cost-first autonomous devotional media system for the **Bhajan Aabha** YouTube and Facebook channels.

## Hard requirements

- ₹0 recurring operating cost.
- No use of the user's computer CPU/GPU.
- Runs remotely while the user's computer can remain switched off.
- No human intervention after one-time account authorization/setup.
- Discover devotional trends continuously.
- Generate 1–4 original devotional videos per day, subject to available free cloud compute.
- Publish automatically to YouTube and Facebook.
- Never modify copyrighted recordings to evade detection; music, lyrics and visuals must be original, public-domain, or properly licensed.
- Automatic QA, retries, logging and performance feedback.
- Never silently introduce a paid dependency; if a free quota is exhausted, the job stops safely.

## Current cloud architecture

**GitHub Actions** = control-plane scheduler and remote-worker dispatcher.

**Kaggle GPU Notebook** = remote compute worker; the user's PC is never a worker.

**Trend engine** = YouTube/public signals, with quota-aware discovery. The first production prototype uses a small seeded opportunity list; live trend discovery is the next controller stage.

**Music engine** = ACE-Step 1.5 Turbo Diffusers. Its current model documentation lists MIT licensing, optional lyrics, Hindi/50+ language support and 8-step turbo inference. citehttps://huggingface.co/docs/diffusers/api/pipelines/ace_step

**Visual engine** = CogVideoX-2B for short original devotional animation. Its current model card lists Apache 2.0 licensing. citehttps://huggingface.co/zai-org/CogVideoX-2b

**Assembly/QA** = FFmpeg with Hindi subtitles, vertical 1080x1920 output and stream/duration/file-integrity checks.

**Publishing** = YouTube Data API + Meta Page publishing APIs; publishing remains gated until one-time channel authorization is completed.

## Pipeline

1. Trend discovery
2. Opportunity scoring
3. De-duplication and content selection
4. Original Hindi devotional creative plan
5. Original music generation
6. Original visual generation
7. Animation/video assembly
8. Audio/video QA
9. Copyright/licensing gate
10. YouTube publishing
11. Facebook publishing
12. Analytics collection
13. Next-run learning

## Current implementation status

- Repository renamed to `HindiBHajans`.
- GitHub Actions cloud-worker dispatch proven successfully.
- Kaggle account phone verification completed.
- Kaggle API token connected to GitHub Actions.
- Kaggle P100 GPU execution proven successfully.
- First end-to-end generation prototype is now implemented in `worker/kaggle_worker.ipynb`.
- Prototype generates original Hindi devotional lyrics/music, a short original AI animation, a vertical MP4, subtitles and QA state.
- Publishing is deliberately gated until YouTube/Facebook authorization is configured.
- Next stage: replace seeded topic selection with live trend discovery, expand the prototype from 10 seconds to production-length videos, then enable automatic publishing and scale to 1–4/day within free compute quotas.

## Zero-cost policy

The system must stop rather than incur a charge. Do not add paid APIs, paid cloud compute, or credit-card-only services as dependencies. Free quotas are allowed only where they are genuinely free and do not auto-bill.

## One-time setup

The owner must authorize YouTube/Facebook publishing and create the free Kaggle account/API credential. Credentials belong in GitHub/Kaggle Secrets and are never committed to this repository.
