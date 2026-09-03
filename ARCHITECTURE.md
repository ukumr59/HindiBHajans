# HindiBHajans — production architecture

## Only two GitHub Actions workflows

### 1. `Bhajan Aabha — Daily Production`

Runs every day at **21:30 IST** and on manual dispatch.

Pipeline:

`approved singer image -> provider router -> real singing master -> hard media gate -> 3 Shorts -> public release -> YouTube + Facebook + Instagram`

GPU selection is provider-neutral. The router retries a provider for transient errors and then fails over to the next configured free provider. Provider jobs can be synchronous or asynchronous.

### 2. `Bhajan Aabha — Maintenance & Provider Health`

Runs at **12:00 IST** and manually. It validates the repository, identity source and provider configuration without starting GPU inference, so it does not burn GPU quota.

## Production invariants

- ₹0 actual spend: only explicitly configured free/no-charge provider endpoints are allowed.
- One daily master is the target; provider failure must not require human intervention.
- Only `assets/uks model image.png` is the singer identity source.
- No face regeneration or identity replacement.
- The output must be a real singing performance with synchronized audio/lip movement.
- Traditional Indian clothing and the correct devotional deity/setting are mandatory worker requirements.
- Shorts are derived from the approved master; they do not trigger another GPU generation.
- YouTube, Facebook and Instagram are independent publishing targets; a failure on one does not regenerate the video or invalidate successful uploads on the others.
- The master is stored as a public GitHub Release asset so downstream platforms can fetch a public media URL without adding a paid object-storage service.
- Job state and provider events are recorded under `state/`.

## Daily reliability model

A free third-party GPU cannot be mathematically guaranteed to be available every day. Reliability therefore comes from **provider redundancy + bounded retry + checkpointable job state + derived Shorts + persistent published assets**. The architecture never relies on one provider being healthy.

## Provider contract

See `config/provider-contract.md`. A provider accepts the common job JSON and returns either a completed `video_url` or an asynchronous `job_id/status_url`. The application does not contain provider-specific business logic.
