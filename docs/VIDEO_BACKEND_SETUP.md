# Video backend — avatar-first architecture

The production requirement is: the uploaded singer must remain the singer, appear in approved traditional Indian clothing before the deity, and visibly sing the generated bhajan with synchronized lips.

## Architecture

1. **One-time avatar preparation**: create `assets/approved_singer_avatar.mp4` from the uploaded singer reference. This is the canonical visual reference and must already show the approved traditional Indian clothing and devotional/deity setting.
2. **Daily audio**: ACE-Step generates the Hindi bhajan into `output/bhajan_source.mp3`.
3. **Daily lip-sync**: MuseTalk 1.5 drives the canonical avatar with that audio. MuseTalk 1.5 is the required lip-sync engine; it is specifically audio-driven and supports real-time GPU inference.
4. **QA gate**: only accept a result when identity, traditional clothing, deity scene and lip-sync proof are all present.
5. **Shorts** are derived only from an approved master.
6. **Publishing** occurs only after the master and Shorts gates pass.

## GPU contract

GitHub Actions remains the orchestrator. It must not attempt MuseTalk inference on the CPU runner. A video execution provider must expose an authenticated HTTP endpoint and return:

- `status_url` while processing
- `video_url` when complete
- a sidecar JSON next to the returned video containing `identity_preserved: true`, `traditional_clothing: true`, `deity_scene: true`, `lip_sync: true`

Provider order is controlled by `VIDEO_PROVIDER_ORDER` and can be switched without changing the workflow.

## Zero-cost policy

No provider may require a recurring paid plan. Free quotas/credits may be used only when they are genuinely available and the account has no paid billing obligation. If no free provider is available at runtime, the workflow must fail closed and must not publish.

## Current validated model choice

MuseTalk 1.5 is the target lip-sync model. Its upstream documentation states that it modifies a face according to input audio and reports 30fps+ on an NVIDIA Tesla V100. It also documents use of a reusable avatar preparation step for repeated audio generation.

The clothing/scene transformation is deliberately **one-time**, not repeated daily. This reduces daily compute to the lip-sync stage and prevents the pipeline from silently replacing the singer with a generic generated person.
