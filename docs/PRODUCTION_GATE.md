# Bhajan Aabha production gate

## Current status

The control plane is implemented, but production GPU execution is intentionally **not enabled** until at least one provider passes the zero-cost acceptance test.

### Accepted model profiles

1. `LONGCAT_AVATAR_15` — image + audio singing, identity lock, full-body/long-video capable; preferred quality path.
2. `ECHOMIMIC_V3_FLASH` — image + audio, singing, identity lock, long-video capable; preferred low-VRAM path.
3. `WAN22_S2V` — image + audio singing, full-body/long-video capable; fallback.

### Hard gate

A provider is usable only when all are true:

- no payment/card requirement for the actual execution path;
- no automatic paid overage;
- API/non-interactive execution is available;
- model can accept the approved singer image and Hindi bhajan audio;
- output is a real singing performance by that singer, not a talking-head substitute;
- traditional Indian clothing and requested deity/setting are preserved;
- returned MP4 contains synchronized video + audio;
- provider can be retried and timed out safely.

The router fails closed when no such endpoint is configured. It never substitutes a paid resource silently.

## Why activation is not automatic

The previously tested GPU services have each failed a hard requirement (Kaggle API/rate-limit path, Lightning allocation, Modal payment-gated credits, Saturn paid/locked Job resources). A model being open-source does not make its GPU execution free. Therefore no provider endpoint is fabricated or treated as free without verification.

## Activation

Configure one or more genuinely free endpoints in GitHub Actions secrets/variables:

- `BH_PROVIDER_ORDER=LONGCAT,ECHOMIMIC,WAN22`
- `BH_LONGCAT_ENDPOINT` / optional `BH_LONGCAT_TOKEN`
- `BH_ECHOMIMIC_ENDPOINT` / optional `BH_ECHOMIMIC_TOKEN`
- `BH_WAN22_ENDPOINT` / optional `BH_WAN22_TOKEN`

The endpoint must implement the common JSON contract in `config/provider-contract.md`.

Until that gate is passed, a daily run must stop with `NO_CONFIGURED_ZERO_COST_GPU_PROVIDER` rather than consume money or pretend that a video was produced.
