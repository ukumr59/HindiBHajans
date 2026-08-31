from __future__ import annotations

import json
import re
from pathlib import Path

import app.zero_cost_pipeline_v5_5 as pipeline
import app.pexels_stock_visuals as stock


def _skip_generated_reference(session):
    print("IMAGE: Agnes deity reference disabled; using Pexels stock visuals", flush=True)
    return "stock://pexels"


def _scene_match_from_search_metadata(video: dict, plan: dict) -> tuple[bool, int, str]:
    """Use Pexels search relevance rather than requiring unavailable API tags.

    Pexels' video API does not expose titles/tags/detailed media metadata, so
    requiring every devotional keyword in the returned video object creates
    false negatives (for example a valid Rama+diya result may have no 'Rama'
    token in its URL). The query itself is the relevance signal; explicit
    forbidden religious terms still hard-reject a result. Final frame-level
    CLIP QC is the authoritative visual gate after assembly.
    """
    blob = " ".join(str(x).lower() for x in (
        video.get("url", ""), video.get("image", ""),
        (video.get("user") or {}).get("name", ""),
        (video.get("user") or {}).get("url", ""),
    ))
    forbidden = stock._forbidden(video)
    if forbidden:
        return False, -1000, f"forbidden={forbidden}"

    # Treat the search query as the semantic intent. For metadata that does
    # happen to contain devotional terms, reward those matches, but do not
    # reject a Pexels result merely because the API omits tags.
    required_hits = [term for term in plan.get("required", []) if re.search(rf"\b{re.escape(term)}\b", blob)]
    preferred_hits = [term for term in plan.get("preferred", []) if re.search(rf"\b{re.escape(term)}\b", blob)]
    score = 50 + 100 * len(required_hits) + 10 * len(preferred_hits)
    reason = f"pexels_query_relevance=true; metadata_required_hits={required_hits}; preferred={preferred_hits}"
    return True, score, reason


def main() -> None:
    # Keep the existing resilient music/assembly pipeline, but replace every
    # generated deity scene with a fresh, licensed Pexels stock clip.
    # Pexels API search is the selection signal; visual_qc.py is the mandatory
    # frame-level semantic gate before release or YouTube publication.
    pipeline.base.generate_image = _skip_generated_reference
    pipeline.base.generate_video_clip = stock.generate_stock_clip
    stock._matches_scene = _scene_match_from_search_metadata
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
        "visual_strategy": "query_relevance_plus_frame_level_clip_qc_unique_stock_clip_per_scene",
        "pexels_no_reuse": True,
        "pexels_credit_file": str(stock.CREDITS),
        "agnes_video_generation": False,
        "agnes_image_generation": False,
        "youtube_seo_generated": True,
        "visual_qc_required": True,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (pipeline.base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STOCK_VISUALS_OK backend=pexels unique_per_scene=true persistent_no_reuse=true", flush=True)
    print("VISUAL_SELECTION_OK pexels_query_relevance=true frame_level_clip_qc_required=true", flush=True)
    print("YOUTUBE_SEO_OK dynamic_title=true dynamic_description=true dynamic_tags=true", flush=True)


if __name__ == "__main__":
    main()
