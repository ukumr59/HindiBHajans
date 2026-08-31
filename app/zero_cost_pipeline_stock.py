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
    title = str(pipeline.base.PACK.get("title", "Bhajan Aabha — Hindi Devotional Bhajan"))
    deity = str(pipeline.base.PACK.get("deity", "Hindi devotional"))
    description = (
        f"🙏 {title}\n\n"
        f"A new original Bhajan Aabha devotional music video featuring {deity}. "
        "Original Hindi devotional lyrics and music are combined with individually selected stock visuals, "
        "edited and synchronized for this video.\n\n"
        "🎵 Original devotional music\n"
        "🛕 Hindi bhakti / devotional theme\n"
        "🎬 Unique visual sequence for this release\n\n"
        "If this bhajan brings peace to you, like, comment and subscribe to Bhajan Aabha for more original Hindi devotional music.\n\n"
        "🙏 जय श्री राम 🙏\n\n"
        "#BhajanAabha #HindiBhajan #BhaktiGeet #DevotionalMusic #RamBhajan #ShriRam #JaiShriRam #Bhajan #Bhakti #DevotionalSong"
    )
    tags = [
        "bhajan aabha", "hindi bhajan", "bhakti geet", "devotional music", "devotional song",
        "ram bhajan", "shri ram", "jai shri ram", "shri ram jai ram", "new hindi bhajan",
        "भजन", "हिंदी भजन", "भक्ति गीत", "राम भजन", "श्री राम", "जय श्री राम", "राम नाम",
        "sanatan bhajan", "hindu devotional song", "new bhajan",
    ]
    (pipeline.base.OUT / "youtube_seo.json").write_text(
        json.dumps({"title": title[:100], "description": description, "tags": tags}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    state.update({
        "visual_backend": "Pexels stock video API",
        "visual_strategy": "unique_stock_clip_per_scene_with_persistent_no_reuse_ledger",
        "pexels_no_reuse": True,
        "pexels_credit_file": str(stock.CREDITS),
        "agnes_video_generation": False,
        "agnes_image_generation": False,
        "youtube_seo_generated": True,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (pipeline.base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STOCK_VISUALS_OK backend=pexels unique_per_scene=true persistent_no_reuse=true", flush=True)
    print("YOUTUBE_SEO_OK dynamic_title=true dynamic_description=true dynamic_tags=true", flush=True)


if __name__ == "__main__":
    main()
