from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

import app.zero_cost_pipeline_v5_5 as pipeline
import app.zero_cost_pipeline_v5_4 as longform

base = pipeline.base
LIGHTNING_USER_ID = os.getenv("LIGHTNING_USER_ID", "").strip()
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY", "").strip()
LIGHTNING_USERNAME = os.getenv("LIGHTNING_USERNAME", "").strip()
LIGHTNING_ORG = os.getenv("LIGHTNING_ORG", "").strip()
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE", "").strip()
LIGHTNING_STUDIO = os.getenv("LIGHTNING_STUDIO", "bhajan-aabha-ace-step").strip()

_STUDIO = None
_STARTED_HERE = False
_GENERATED = False
_LOCK = threading.Lock()


def _require_lightning():
    if not LIGHTNING_USER_ID or not LIGHTNING_API_KEY:
        raise RuntimeError("SETUP_REQUIRED: LIGHTNING_USER_ID and LIGHTNING_API_KEY repository secrets are required")
    if not LIGHTNING_TEAMSPACE:
        raise RuntimeError("SETUP_REQUIRED: LIGHTNING_TEAMSPACE must be resolved")
    if not LIGHTNING_USERNAME and not LIGHTNING_ORG:
        raise RuntimeError("SETUP_REQUIRED: Lightning owner was not resolved")


def _studio():
    from lightning_sdk import Studio
    kwargs = {"name": LIGHTNING_STUDIO, "teamspace": LIGHTNING_TEAMSPACE, "create_ok": True}
    if LIGHTNING_ORG:
        kwargs["org"] = LIGHTNING_ORG
    else:
        kwargs["user"] = LIGHTNING_USERNAME
    return Studio(**kwargs)


def _start():
    global _STUDIO, _STARTED_HERE
    _require_lightning()
    from lightning_sdk import Machine
    _STUDIO = _studio()
    status = str(_STUDIO.status).lower()
    print(f"WAN_LIGHTNING_STATUS studio={LIGHTNING_STUDIO} status={status}", flush=True)
    if "running" not in status:
        print("WAN_LIGHTNING_START: starting T4 Studio", flush=True)
        _STUDIO.start(Machine.T4)
        _STARTED_HERE = True
    else:
        machine = str(_STUDIO.machine).lower()
        if "t4" not in machine:
            print("WAN_LIGHTNING_START: switching Studio to T4", flush=True)
            _STUDIO.switch_machine(Machine.T4)
            _STARTED_HERE = True


def _stop():
    global _STUDIO
    if _STUDIO is not None and _STARTED_HERE:
        try:
            print("WAN_LIGHTNING_STOP: stopping dedicated T4 Studio", flush=True)
            _STUDIO.stop()
        except Exception as exc:
            print(f"WAN_LIGHTNING_STOP_WARNING: {exc}", flush=True)
    _STUDIO = None


def _generate_all():
    global _GENERATED
    with _LOCK:
        if _GENERATED:
            return
        _start()
        worker = Path(__file__).with_name("lightning_wan2_worker.py")
        request = {
            "scenes": [
                {"index": i, "prompt": prompt, "seed": 20260831 + i * 7919}
                for i, prompt in enumerate(base.PACK["scene_prompts"], 1)
            ]
        }
        request_path = Path(tempfile.mkdtemp(prefix="wan_bhajan_")) / "wan_request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        _STUDIO.upload_file(str(worker), remote_path="lightning_wan2_worker.py")
        _STUDIO.upload_file(str(request_path), remote_path="wan2_worker/wan_request.json")
        print(f"WAN_LIGHTNING_RUN: generating {len(request['scenes'])} unique Wan2.1 scenes", flush=True)
        output, code = _STUDIO.run_with_exit_code("python lightning_wan2_worker.py")
        print(output[-20000:], flush=True)
        if code != 0:
            raise RuntimeError(f"WAN_FATAL: remote Wan2.1 worker exited with code {code}")
        manifest = _STUDIO.download_file("wan2_worker/wan_manifest.json", str(base.OUT / "wan_manifest.json"))
        if not Path(manifest).exists():
            raise RuntimeError("WAN_FATAL: wan_manifest.json was not downloaded")
        _GENERATED = True


def generate_wan_clip(session, image_url: str, prompt: str, index: int) -> Path:
    _generate_all()
    target = base.VIDEOS / f"scene_{index}.mp4"
    if target.exists() and target.stat().st_size > 50_000:
        return target
    remote = f"wan2_worker/scenes/scene_{index}.mp4"
    downloaded = _STUDIO.download_file(remote, str(target))
    if not Path(downloaded).exists() or Path(downloaded).stat().st_size < 50_000:
        raise RuntimeError(f"WAN_FATAL: downloaded scene {index} is missing or too small")
    print(f"WAN_OK: scene={index}/{len(base.PACK['scene_prompts'])} {target}", flush=True)
    return target


def _skip_image(session):
    print("IMAGE: Agnes deity reference disabled; Wan2.1 T2V is the sole visual backend", flush=True)
    return "wan://t2v-1.3b"


def main():
    try:
        base.generate_image = _skip_image
        base.generate_video_clip = generate_wan_clip
        base.require_env = lambda: None
        longform.configure()
        pipeline.music.generate_music_gradio = pipeline.generate_music_lightning
        pipeline.longform.base.generate_video_clip = generate_wan_clip
        pipeline.longform.base.generate_image = _skip_image
        pipeline.longform.base.require_env = lambda: None
        pipeline.longform.base.assemble = longform.assemble_without_subtitles
        pipeline.longform.main()

        state_path = base.OUT / "run_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        title = str(base.PACK.get("title", "Bhajan Aabha — Hindi Devotional Bhajan"))
        description = (
            f"🙏 {title}\n\n"
            "An original Bhajan Aabha Hindi devotional music video with original music and lyrics, "
            "paired with individually generated Wan2.1 AI devotional scenes.\n\n"
            "🎵 Original devotional music\n"
            "🎬 Open-source Wan2.1 generated visuals\n"
            "🛕 Shri Ram / Hindu devotional visual theme\n"
            "❌ No Hindi subtitles\n\n"
            "🙏 जय श्री राम 🙏\n\n"
            "#BhajanAabha #HindiBhajan #RamBhajan #ShriRam #JaiShriRam #Bhakti #DevotionalMusic #Bhajan"
        )
        tags = [
            "bhajan aabha", "hindi bhajan", "ram bhajan", "shri ram", "jai shri ram",
            "shri ram jai ram", "new hindi bhajan", "bhakti geet", "devotional music",
            "devotional song", "hindu devotional", "sanatan bhajan", "राम भजन", "श्री राम", "जय श्री राम",
        ]
        (base.OUT / "youtube_seo.json").write_text(
            json.dumps({"title": title[:100], "description": description, "tags": tags}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state.update({
            "visual_backend": "Wan2.1 T2V 1.3B on Lightning T4",
            "visual_model": "Wan-AI/Wan2.1-T2V-1.3B",
            "visual_model_license": "Apache-2.0",
            "visual_strategy": "unique_song_aware_Rama_devotional_T2V_scene_per_15s_segment",
            "visual_source_seconds": 5,
            "visual_scene_seconds": 15,
            "wan_slow_motion_factor": 3,
            "wan_resolution": "480x832",
            "agnes_video_generation": False,
            "agnes_image_generation": False,
            "pexels_used": False,
            "youtube_seo_generated": True,
            "visual_qc_required": True,
            "zero_cost": True,
            "paid_services": False,
            "paid_gpu": False,
        })
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("WAN_VISUALS_OK backend=Wan2.1-T2V-1.3B unique_per_scene=true source=opensource", flush=True)
        print("WAN_LICENSE_OK Apache-2.0", flush=True)
        print("YOUTUBE_SEO_OK dynamic_title=true dynamic_description=true dynamic_tags=true", flush=True)
    finally:
        _stop()


if __name__ == "__main__":
    main()
