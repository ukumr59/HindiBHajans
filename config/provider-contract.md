# HindiBHajans GPU provider contract

The production workflow has exactly one GPU entry point: `app.provider_router`.
A provider is an interchangeable worker, not part of the application design.

## Provider environment

Configure only providers that are genuinely usable at ₹0. The router reads:

- `BH_PROVIDER_ORDER` — priority order, e.g. `MODAL,HF_SPACE,BEAM,KAGGLE`
- `BH_<PROVIDER>_ENDPOINT` — HTTPS POST endpoint
- `BH_<PROVIDER>_TOKEN` — optional bearer token

No provider is silently switched to a paid tier. The operator must configure the endpoint under a free/no-charge account or free quota.

## HTTP contract

### Submit

`POST <endpoint>` with the job JSON.

Accepted immediate responses:

```json
{"status":"completed","video_url":"https://.../master.mp4"}
```

or asynchronous:

```json
{"status":"queued","job_id":"abc123","status_url":"https://.../abc123"}
```

### Status

`GET <status_url>` returns `queued`, `running`, `completed`, `failed`, or `cancelled`. A completed response must include `video_url`.

## Required generation semantics

The worker must produce a real singing performance, not a zoom/pan substitute:

1. use only `assets/uks model image.png` as the singer identity reference;
2. do not regenerate or replace the singer's face;
3. singer visibly sings the generated Hindi bhajan;
4. use traditional Indian clothing;
5. place the singer before the requested deity in a devotional setting;
6. include synchronized audio/lip movement;
7. return a valid MP4 with video and audio.

## Failure policy

- 429/408/5xx/network timeout: bounded retry, then fail over.
- 401/403/capability/input errors: no repeated hammering; fail over or stop.
- quota exhaustion: skip provider until it is manually/runtime-known available again.
- provider timeout: fail over.
- all providers unavailable: job fails cleanly; never spend money.

Shorts are always derived from the approved master, so a second GPU generation is not required.
