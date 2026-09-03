# Saturn Cloud GPU worker

This is the first provider adapter for the provider-neutral production pipeline.

Use a Saturn Cloud Hosted Free GPU job. Current Saturn documentation says Hosted Free requires no credit card and provides free GPU compute; current plan pages should be checked in the account because free GPU-hour limits can change. Saturn jobs can be triggered by API and are asynchronous.

## Job

Create one Saturn **Job** using this repository and command:

`python saturn/run_worker.py`

Select a free GPU instance exposed by the account (prefer NVIDIA T4 16 GB when available). Do not add a payment method or select paid hardware.

Environment variables required by the job:

- `BH_GITHUB_TOKEN`: fine-grained GitHub token with Contents: write for `ukumr59/HindiBHajans` (used only to publish the generated MP4 as a public release asset).
- `BH_RUN_SECONDS`: `10` for smoke test, `180` for production.

The GitHub Actions workflow only needs the Saturn API base, user token, job ID and the public result URL.
