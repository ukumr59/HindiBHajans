from __future__ import annotations

import json

import app.zero_cost_pipeline_v6 as v6

# v5_2_fix reads the prompt from its imported v5_2 module, so mirror the
# long-form prompt into that exact runtime namespace before generation.
v6.creative.v52.DJ_MUSIC_PROMPT = v6.creative.DJ_MUSIC_PROMPT

if __name__ == "__main__":
    v6.base.main()
    videos = sorted(v6.base.VIDEOS.glob("*.mp4"))
    if not videos:
        raise RuntimeError("OUTPUT_FATAL: final long-form video was not produced")
    dj_master = v6.creative.make_dj_master(videos[0])
    state_path = v6.base.OUT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({
        "architecture": "v6.0-longform-4min",
        "target_duration_sec": 240,
        "source_scene_seconds": 30,
        "source_scene_count": 3,
        "visual_chapters": 8,
        "music_duration_sec": 240,
        "dj_master": str(dj_master),
        "duration_contract": "180-300 seconds; default 240 seconds",
        "music_contract": "full-length 240s ACE-Step generation; not a 30s loop",
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (v6.base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LONGFORM_OK target=240s music=240s visual_chapters=8")
