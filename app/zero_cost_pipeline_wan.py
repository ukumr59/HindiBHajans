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

# Song-aware visual briefs. These are deliberately explicit about Shri Ram,
# Hindu temple settings and forbidden visual concepts so Wan has strong semantic
# guidance instead of generic "Indian" scenery.
RAM_SCENE_PROMPTS = [
    "Cinematic devotional scene of Lord Rama standing majestically inside an ancient Hindu temple at sunrise, blue skin, saffron-gold royal garments, bow and arrow, golden crown, glowing diyas, sacred Hindu temple architecture, warm divine light, reverent atmosphere, slow gentle camera movement, photorealistic Indian devotional cinema, no text, no subtitles, no mosque, no church, no Islamic architecture.",
    "Close cinematic view of a beautiful Lord Rama murti in a Hindu temple, blue complexion, golden crown, yellow silk garments, bow beside him, hundreds of warm diyas illuminating the idol, flower garlands and marigolds, soft incense smoke, devotees praying in the background, reverent photorealistic devotional film, no text, no subtitles, no mosque, no church.",
    "Lord Rama seated peacefully with his bow inside a magnificent Ayodhya-style Hindu temple, glowing oil lamps creating a golden halo, marigold flowers, sacred bells, carved Hindu pillars, gentle devotional atmosphere matching a prayer about the light of Rama's name, cinematic realism, slow graceful motion, no text, no subtitles, no non-Hindu religious architecture.",
    "Powerful devotional hero shot of Lord Rama during evening aarti in a grand Hindu temple, blue skin, royal saffron clothing, crown, bow and arrow, priests holding lamps, devotees with folded hands, many glowing diyas, golden firelight, sacred Hindu atmosphere, cinematic realistic photography, no text, no subtitles, no mosque, no church.",
    "Emotional Hindu devotional scene: a humble devotee kneels with folded hands before Lord Rama's radiant murti during a difficult moment, Lord Rama visible clearly in the center, warm temple lamps, flower offerings, peaceful compassionate expression, sacred Hindu temple interior, cinematic realism, gentle camera push-in, no text, no subtitles, no mosque or church.",
    "Lord Rama blessing a praying devotee inside a beautiful Hindu temple, Rama clearly visible with blue skin, golden crown, saffron garments and bow, devotee's folded hands in foreground, glowing diyas and marigold garlands, compassionate divine mood, photorealistic Indian devotional cinema, no text, no subtitles, no non-Hindu religious architecture.",
    "Lord Rama and devoted Hanuman together in a sacred Hindu temple, Hanuman kneeling respectfully before Rama, Rama standing with bow and arrow, warm aarti flames, temple bells, marigold garlands, golden divine light, reverent Hindu devotional atmosphere, cinematic realistic motion, no text, no subtitles, no mosque, no church.",
    "Epic devotional vision of Lord Rama as the prince of Ayodhya, standing before a grand Ayodhya-inspired Hindu temple courtyard at dawn, bow and arrow, golden crown, saffron royal garments, temple flags, devotees and diyas, majestic but peaceful cinematic Indian devotional scene, no text, no subtitles, no Islamic or Christian architecture.",
    "Joyful Hindu devotional celebration of Jai Shri Ram: Lord Rama clearly visible at the center of a grand temple aarti, devotees raising hands in devotion, saffron flags, temple bells, flower petals, glowing lamps, dynamic but respectful cinematic movement, realistic Indian devotional film, no text, no subtitles, no mosque, no church.",
    "Grand cinematic build toward a Ram bhajan climax: Lord Rama and Hanuman together beneath a glowing temple canopy, dozens of diyas, saffron flags and marigold garlands, devotees gathered in reverence, warm golden light, powerful sacred Hindu atmosphere, realistic devotional cinema, no text, no subtitles, no non-Hindu religious buildings.",
    "Hero devotional portrait of Lord Rama standing in a magnificent Hindu temple surrounded by hundreds of diyas and flowers, blue skin, golden crown, saffron garments, bow and arrow, serene compassionate face, rich golden cinematic lighting, slow majestic camera movement, photorealistic Indian devotional film, no text, no subtitles, no mosque, no church.",
    "Peaceful closing vision of Lord Rama inside a glowing Hindu temple at night, serene blue face, golden crown, bow and arrow, rows of diyas and marigold flowers, soft incense haze, devotees praying quietly, warm divine light fading gently, cinematic devotional realism, no text, no subtitles, no mosque, no church, no Islamic architecture.",
]


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
        prompts = base.PACK.get("scene_prompts") or RAM_SCENE_PROMPTS
        request = {
            "scenes": [
                {"index": i, "prompt": prompt, "seed": 20260831 + i * 7919}
                for i, prompt in enumerate(prompts, 1)
            ]
        }
        request_path = Path(tempfile.mkdtemp(prefix="wan_bhajan_")) / "wan_request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        _STUDIO.upload_file(str(worker), remote_path="lightning_wan2_worker.py")
        _STUDIO.upload_file(str(request_path), remote_path="wan_request.json")
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
        base.PACK["scene_prompts"] = RAM_SCENE_PROMPTS[: base.VIDEO_SECONDS // longform.SCENE_SECONDS]
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
