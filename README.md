# Autonomous Bhajan Channel

Zero-cost-first autonomous devotional video publishing system.

## Hard requirements

- ₹0 recurring operating cost.
- No use of the user's computer CPU/GPU.
- Runs in the cloud with the user's computer allowed to be offline.
- No human intervention after one-time setup.
- Discover devotional trends continuously.
- Generate 1–4 original devotional videos per day.
- Publish automatically to YouTube and Facebook.
- Do not modify copyrighted recordings to evade detection; generated music and visuals must be original or properly licensed.
- Automatic QA, retries, logging, and performance feedback.

## Architecture

1. Trend discovery
2. Opportunity scoring
3. Original devotional creative plan
4. Remote AI generation
5. Video assembly
6. Copyright/quality gate
7. YouTube/Facebook publishing
8. Analytics feedback
9. Automatic retry/self-recovery

The GitHub repository is the orchestration/control plane. Heavy AI generation must run on free remote compute; the user's PC is never a worker.

## Zero-cost policy

The workflow must stop rather than incur a charge. Do not add paid APIs, paid cloud compute, or credit-card-only services as dependencies.

## One-time setup still required

The owner must authorize the publishing accounts and create the free remote AI execution account(s). Credentials belong in GitHub Actions Secrets and are never committed to this repository.
