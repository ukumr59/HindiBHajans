from __future__ import annotations

import json
from pathlib import Path

import app.zero_cost_pipeline_v5_5 as pipeline
import app.pexels_stock_visuals as stock


def _skip_generated_reference(session):
    print("IMAGE: Agnes deity reference disabled; using Pexels stock visuals", flush=True)
    return "stock://pexels"


def main() -> None:
    # Keep the existing resilient music/assembly pipeline, but replace every
    # generated deity scene with a fresh, licensed Pexels stock clip.
    pipeline.base.generate_image = _skip_generated_reference
    pipeline.base.generate_video_clip = stock.generate_stock_clip
    stock.CREDITS.parent.mkdir(parents=True, exist_ok=True)
    stock.CREDITS.unlink(missing_ok=True)
    pipeline.longform.main()

    state_path = pipeline.base.OUT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({
        "visual_backend": "Pexels stock video API",
        "visual_strategy": "unique_stock_clip_per_scene_with_persistent_no_reuse_ledger",
        "pexels_no_reuse": True,
        "pexels_credit_file": str(stock.CREDITS),
        "agnes_video_generation": False,
        "agness_image_generation": False,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (pipeline.base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STOCK_VISUALS_OK backend=pexels unique_per_scene=true persistent_no_reuse=true", flush=True)


if __name__ == "__main__":
    main()
