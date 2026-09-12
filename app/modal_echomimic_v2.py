"""Validated Modal EchoMimicV3 worker entrypoint."""
from __future__ import annotations

# The implementation is intentionally kept as a separate version while the
# upstream CLI contract is validated before replacing the first draft.
# See upstream infer_flash.py: it requires --image_path/--audio_path and the
# model/transformer/wav2vec paths; it does not accept --ref_image/--audio.

UPSTREAM_INFERENCE_ENTRYPOINT = "infer_flash.py"
UPSTREAM_REQUIRED_INPUTS = ("--image_path", "--audio_path", "--prompt")
CONTROLLED_TEST_DURATION_SECONDS = 20
PRODUCTION_MAX_DURATION_SECONDS = 180


def contract() -> dict[str, object]:
    return {
        "gpu": "L4",
        "duration_test_seconds": CONTROLLED_TEST_DURATION_SECONDS,
        "duration_production_seconds": PRODUCTION_MAX_DURATION_SECONDS,
        "static_image_fallback": False,
        "entrypoint": UPSTREAM_INFERENCE_ENTRYPOINT,
        "required_inputs": UPSTREAM_REQUIRED_INPUTS,
    }
