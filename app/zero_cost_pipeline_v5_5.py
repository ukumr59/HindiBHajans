from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import app.zero_cost_pipeline_v5_4 as longform
import app.zero_cost_pipeline_v5_2 as music

base = longform.base
LIGHTNING_USER_ID = os.getenv("LIGHTNING_USER_ID", "").strip()
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY", "").strip()
LIGHTNING_USERNAME = os.getenv("LIGHTNING_USERNAME", "").strip()
LIGHTNING_ORG = os.getenv("LIGHTNING_ORG", "").strip()
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE", "").strip()
LIGHTNING_STUDIO = os.getenv("LIGHTNING_STUDIO", "bhajan-aabha-ace-step").strip()


def _require_lightning() -> None:
    if not LIGHTNING_USER_ID or not LIGHTNING_API_KEY:
        raise RuntimeError("SETUP_REQUIRED: LIGHTNING_USER_ID and LIGHTNING_API_KEY repository secrets are required")
    if not LIGHTNING_TEAMSPACE:
        raise RuntimeError("SETUP_REQUIRED: LIGHTNING_TEAMSPACE must be resolved from the Lightning membership")
    if not LIGHTNING_USERNAME and not LIGHTNING_ORG:
        raise RuntimeError("SETUP_REQUIRED: Lightning teamspace owner was not resolved; set LIGHTNING_USERNAME or LIGHTNING_ORG")
    if LIGHTNING_USERNAME and LIGHTNING_ORG:
        raise RuntimeError("SETUP_REQUIRED: set only one of LIGHTNING_USERNAME or LIGHTNING_ORG")


def _studio():
    from lightning_sdk import Studio
    kwargs = {"name": LIGHTNING_STUDIO, "teamspace": LIGHTNING_TEAMSPACE, "create_ok": True}
    if LIGHTNING_ORG:
        kwargs["org"] = LIGHTNING_ORG
    else:
        kwargs["user"] = LIGHTNING_USERNAME
    return Studio(**kwargs)


def generate_music_lightning() -> Path:
    _require_lightning()
    from lightning_sdk import Machine

    request = {
        "caption": base.PACK["music_prompt"],
        "lyrics": base.PACK["lyrics"],
        "duration": int(base.VIDEO_SECONDS),
        "bpm": 128,
        "keyscale": "C Major",
        "timesignature": "4/4",
        "vocal_language": "hi",
    }
    request_path = Path(tempfile.mkdtemp(prefix="lightning_bhajan_")) / "bhajan_request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    worker = Path(__file__).with_name("lightning_ace_step_worker.py")

    studio = _studio()
    started_here = False
    try:
        status = str(studio.status).lower()
        print(f"LIGHTNING_STATUS: studio={LIGHTNING_STUDIO} teamspace={LIGHTNING_TEAMSPACE} owner={LIGHTNING_USERNAME or LIGHTNING_ORG} status={status}", flush=True)
        if "running" not in status:
            print("LIGHTNING_LAUNCH: starting dedicated T4 GPU Studio", flush=True)
            studio.start(Machine.T4)
            started_here = True
        else:
            machine = str(studio.machine).lower()
            if "t4" not in machine:
                print("LIGHTNING_LAUNCH: switching dedicated Studio to T4", flush=True)
                studio.switch_machine(Machine.T4)
                started_here = True

        studio.upload_file(str(worker), remote_path="bhajan_ace_step_worker.py")
        studio.upload_file(str(request_path), remote_path="bhajan_request.json")
        print("LIGHTNING_LAUNCH: executing ACE-Step worker on remote T4", flush=True)
        output, code = studio.run_with_exit_code("python bhajan_ace_step_worker.py")
        print(output[-12000:], flush=True)
        if code != 0:
            raise RuntimeError(f"LIGHTNING_ACE_FATAL: remote worker exited with code {code}")

        local_output = base.AUDIO / "bhajan_source.mp3"
        local_output.parent.mkdir(parents=True, exist_ok=True)
        studio.download_file("bhajan_aabha_worker/bhajan_source.mp3", str(local_output))
        if not local_output.exists() or local_output.stat().st_size < 100_000:
            raise RuntimeError("LIGHTNING_ACE_FATAL: downloaded MP3 is missing or suspiciously small")
        print(f"LIGHTNING_ACE_OK: {local_output} {local_output.stat().st_size} bytes", flush=True)
        return local_output
    finally:
        if started_here:
            try:
                print("LIGHTNING_LAUNCH: stopping dedicated Studio", flush=True)
                studio.stop()
            except Exception as exc:
                print(f"LIGHTNING_CLEANUP_WARNING: {exc}", flush=True)


music.generate_music_gradio = generate_music_lightning
base.ACESTEP_ROOT = "lightning://ACE-Step-1.5"


if __name__ == "__main__":
    longform.main()
    state_path = base.OUT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({
        "architecture": "v5.6-lightning-gpu-longform",
        "music_backend": "ACE-Step 1.5 on Lightning AI T4 Studio",
        "music_api_mode": "lightning_ace_step_gpu_studio",
        "music_model": "acestep-v15-turbo",
        "music_lm_model": "acestep-5Hz-lm-0.6B",
        "kaggle": False,
        "huggingface_zero_gpu": False,
        "paid_services": False,
        "paid_gpu": False,
        "zero_cost": True,
        "lightning": True,
        "lightning_machine": "T4",
        "lightning_teamspace": LIGHTNING_TEAMSPACE,
        "lightning_studio": LIGHTNING_STUDIO,
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STATE_OK backend=lightning_ace_step_gpu_studio kaggle=false zero_cost=true machine=T4")
