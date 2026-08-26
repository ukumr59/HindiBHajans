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

**GitHub Actions** = zero/low-cost control-plane scheduler.

**Kaggle GPU Notebook** = remote compute worker; the user's PC is never a worker.

**Trend engine** = YouTube Data API/public signals, with quota-aware discovery.

**Music engine** = planned around ACE-Step 1.5 because its current model card states commercial-ready use, MIT licensing and 50+ language support. Do not use Meta MusicGen weights for production because those weights are CC-BY-NC 4.0.

**Visual engine** = open/licensed image-generation models plus deterministic cinematic motion/animation and FFmpeg assembly.

**Publishing** = YouTube Data API + Meta Page publishing APIs.

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
- GitHub Actions control-plane smoke test completed successfully.
- Daily cloud-worker dispatcher added.
- Kaggle GPU worker scaffold added at `worker/kaggle_worker.ipynb`.
- Headless Kaggle dispatcher added at `app/dispatch_kaggle.py`.
- The worker is currently a **safe GPU/setup smoke test**; production generation is deliberately not enabled until the remote GPU path and credentials are verified.

## Zero-cost policy

The system must stop rather than incur a charge. Do not add paid APIs, paid cloud compute, or credit-card-only services as dependencies. Free quotas are allowed only where they are genuinely free and do not auto-bill.

## One-time setup

The owner must authorize YouTube/Facebook publishing and create the free Kaggle account/API credential. Credentials belong in GitHub/Kaggle Secrets and are never committed to this repository.
