from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests

import app.zero_cost_pipeline_v5_4 as longform
import app.zero_cost_pipeline_v5_2 as music

base = longform.base
LIGHTNING_USER_ID = os.getenv("LIGHTNING_USER_ID", "").strip()
LIGHTNING_API_KEY = os.getenv("LIGHTNING_API_KEY", "").strip()
LIGHTNING_USERNAME = os.getenv("LIGHTNING_USERNAME", "").strip()
LIGHTNING_ORG = os.getenv("LIGHTNING_ORG", "").strip()
LIGHTNING_TEAMSPACE = os.getenv("LIGHTNING_TEAMSPACE", "").strip()
LIGHTNING_STUDIO = os.getenv("LIGHTNING_STUDIO", "bhajan-aabha-ace-step").strip()

VIDEO_POLL_TIMEOUT = int(os.getenv("AGNES_VIDEO_POLL_TIMEOUT", "900"))
VIDEO_POLL_INTERVAL = int(os.getenv("AGNES_VIDEO_POLL_INTERVAL", "10"))
VIDEO_DOWNLOAD_TIMEOUT = int(os.getenv("AGNES_VIDEO_DOWNLOAD_TIMEOUT", "600"))
VIDEO_DOWNLOAD_RETRIES = int(os.getenv("AGNES_VIDEO_DOWNLOAD_RETRIES", "5"))
VIDEO_DOWNLOAD_BACKOFF = int(os.getenv("AGNES_VIDEO_DOWNLOAD_BACKOFF", "15"))


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


def _stop_studio_nonblocking(studio) -> None:
    def _stop() -> None:
        try:
            studio.stop()
            print("LIGHTNING_CLEANUP_OK: dedicated Studio stop requested", flush=True)
        except Exception as exc:
            print(f"LIGHTNING_CLEANUP_WARNING: {exc}", flush=True)
    print("LIGHTNING_LAUNCH: requesting dedicated Studio stop (non-blocking)", flush=True)
    threading.Thread(target=_stop, name="lightning-stop", daemon=True).start()


def _normalize_audio(local_output: Path, target_seconds: float) -> None:
    normalized = local_output.with_name("bhajan_source_normalized.mp3")
    print(f"LIGHTNING_DURATION_FIX: normalizing audio to {target_seconds:.1f}s", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(local_output),
        "-af", f"apad=whole_dur={target_seconds},atrim=duration={target_seconds}",
        "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", str(normalized)
    ], check=True, capture_output=True, text=True)
    normalized.replace(local_output)


def generate_music_lightning() -> Path:
    _require_lightning()
    from lightning_sdk import Machine
    request = {
        "caption": base.PACK["music_prompt"], "lyrics": base.PACK["lyrics"],
        "duration": int(base.VIDEO_SECONDS), "bpm": 128, "keyscale": "C Major",
        "timesignature": "4/4", "vocal_language": "hi",
    }
    request_path = Path(tempfile.mkdtemp(prefix="lightning_bhajan_")) / "bhajan_request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    worker = Path(__file__).with_name("lightning_ace_step_worker_v2.py")
    studio = _studio()
    started_here = False
    try:
        status = str(studio.status).lower()
        print(f"LIGHTNING_STATUS: studio={LIGHTNING_STUDIO} teamspace={LIGHTNING_TEAMSPACE} owner={LIGHTNING_USERNAME or LIGHTNING_ORG} status={status}", flush=True)
        if "running" not in status:
            print("LIGHTNING_LAUNCH: starting dedicated T4 GPU Studio", flush=True)
            studio.start(Machine.T4); started_here = True
        else:
            machine = str(studio.machine).lower()
            if "t4" not in machine:
                print("LIGHTNING_LAUNCH: switching dedicated Studio to T4", flush=True)
                studio.switch_machine(Machine.T4); started_here = True
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
        _normalize_audio(local_output, float(base.VIDEO_SECONDS))
        if not local_output.exists() or local_output.stat().st_size < 100_000:
            raise RuntimeError("LIGHTNING_ACE_FATAL: normalized MP3 is missing or suspiciously small")
        print(f"LIGHTNING_ACE_OK: {local_output} {local_output.stat().st_size} bytes", flush=True)
        return local_output
    finally:
        if started_here:
            _stop_studio_nonblocking(studio)


def _download_video_resilient(session: requests.Session, url: str, path: Path, *, min_bytes: int = 50_000) -> None:
    """Download an Agnes MP4 robustly; never leave a partial file as a valid scene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    last_error = None
    timeout = (30, VIDEO_DOWNLOAD_TIMEOUT)

    for attempt in range(1, VIDEO_DOWNLOAD_RETRIES + 1):
        try:
            if partial.exists():
                partial.unlink()
            print(
                f"VIDEO_DOWNLOAD: attempt={attempt}/{VIDEO_DOWNLOAD_RETRIES} "
                f"timeout={VIDEO_DOWNLOAD_TIMEOUT}s target={path.name}",
                flush=True,
            )
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            size = partial.stat().st_size if partial.exists() else 0
            if size < min_bytes:
                raise RuntimeError(f"downloaded file is suspiciously small: {size} bytes")
            partial.replace(path)
            print(f"VIDEO_DOWNLOAD_OK: {path} {size} bytes", flush=True)
            return
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            print(
                f"VIDEO_DOWNLOAD_WARNING: attempt={attempt}/{VIDEO_DOWNLOAD_RETRIES} "
                f"failed: {exc}",
                flush=True,
            )
            try:
                if partial.exists():
                    partial.unlink()
            except OSError:
                pass
            if attempt < VIDEO_DOWNLOAD_RETRIES:
                delay = min(120, VIDEO_DOWNLOAD_BACKOFF * attempt)
                print(f"VIDEO_DOWNLOAD_RETRY: waiting {delay}s before retry", flush=True)
                time.sleep(delay)

    raise RuntimeError(
        f"VIDEO_FATAL: Agnes video download failed after {VIDEO_DOWNLOAD_RETRIES} attempts; "
        f"last error: {last_error}"
    )


def generate_video_clip_resilient(session: requests.Session, image_url: str, prompt: str, index: int) -> Path:
    scene_count = len(base.PACK.get("scene_prompts", [])) or 1
    path = base.VIDEOS / f"scene_{index}.mp4"
    if path.exists() and path.stat().st_size >= 50_000:
        print(f"VIDEO: scene={index}/{scene_count} cached; skipping generation", flush=True)
        return path

    # IMPORTANT: Do not add an artificial per-scene delay here.
    # Agnes generation/polling is already asynchronous and rate-limit responses
    # are handled by the request/poll retry logic below. The previous fixed 70s
    # sleep made every scene after scene 1 unnecessarily slow.
    payload = {
        "model": "agnes-video-v2.0", "prompt": prompt, "image": image_url,
        "width": base.WIDTH, "height": base.HEIGHT,
        "num_frames": base.SCENE_SECONDS * base.FPS + 1, "frame_rate": base.FPS,
        "negative_prompt": "deformed face, extra fingers, extra limbs, duplicate deity, distorted hands, text, watermark, logo, flicker, jitter, cartoon, low detail",
    }
    print(f"VIDEO: submitting scene {index}/{scene_count} frames={payload['num_frames']}", flush=True)
    data = base.http_json(session, "POST", f"{base.AGNES_ROOT}/v1/videos",
                          headers={"Authorization": f"Bearer {base.AGNES_API_KEY}", "Content-Type": "application/json"},
                          body=payload, timeout=120)
    video_id = data.get("video_id")
    if not video_id:
        raise RuntimeError(f"VIDEO_FATAL: Agnes create response has no video_id: {data}")
    print(f"VIDEO: scene={index}/{scene_count} video_id={video_id} submitted", flush=True)
    deadline = time.monotonic() + VIDEO_POLL_TIMEOUT
    polls = 0
    last_progress = None
    while time.monotonic() < deadline:
        polls += 1
        try:
            r = session.get(f"{base.AGNES_ROOT}/agnesapi", params={"video_id": video_id, "model_name": "agnes-video-v2.0"},
                            headers={"Authorization": f"Bearer {base.AGNES_API_KEY}"}, timeout=45)
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"VIDEO: scene={index}/{scene_count} poll={polls} HTTP {r.status_code}; retrying", flush=True)
                time.sleep(VIDEO_POLL_INTERVAL)
                continue
            r.raise_for_status()
            status = r.json()
        except requests.RequestException as exc:
            print(f"VIDEO: scene={index}/{scene_count} poll={polls} request warning: {exc}", flush=True)
            time.sleep(VIDEO_POLL_INTERVAL)
            continue
        state = str(status.get("status", "")).lower()
        progress = status.get("progress", 0)
        if progress != last_progress or polls % 6 == 0:
            print(f"VIDEO: scene={index}/{scene_count} status={state} progress={progress}% elapsed={int(VIDEO_POLL_TIMEOUT - max(0, deadline-time.monotonic()))}s", flush=True)
            last_progress = progress
        if state == "completed":
            url = status.get("url") or status.get("video_url") or status.get("remixed_from_video_id")
            if not url:
                raise RuntimeError(f"VIDEO_FATAL: completed response has no video URL: {status}")
            _download_video_resilient(session, url, path, min_bytes=50_000)
            print(f"VIDEO_OK: scene={index}/{scene_count} downloaded={path.stat().st_size} bytes", flush=True)
            return path
        if state == "failed":
            raise RuntimeError(f"VIDEO_FATAL: scene {index} failed: {status.get('error')}")
        time.sleep(VIDEO_POLL_INTERVAL)
    raise RuntimeError(f"VIDEO_FATAL: scene {index}/{scene_count} exceeded {VIDEO_POLL_TIMEOUT}s polling timeout; no blind 25-minute wait")


music.generate_music_gradio = generate_music_lightning
base.generate_video_clip = generate_video_clip_resilient
base.ACESTEP_ROOT = "lightning://ACE-Step-1.5"


if __name__ == "__main__":
    longform.main()
    state_path = base.OUT / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update({
        "architecture": "v5.8-lightning-gpu-longform-resilient-video",
        "music_backend": "ACE-Step 1.5 on Lightning AI T4 Studio",
        "music_api_mode": "lightning_ace_step_gpu_studio",
        "music_model": "acestep-v15-turbo",
        "music_lm_model": "acestep-5Hz-lm-0.6B",
        "kaggle": False, "huggingface_zero_gpu": False, "paid_services": False,
        "paid_gpu": False, "zero_cost": True, "lightning": True,
        "lightning_machine": "T4", "lightning_teamspace": LIGHTNING_TEAMSPACE,
        "lightning_studio": LIGHTNING_STUDIO,
        "duration_normalization": "ffmpeg_apad_atrim_exact_target",
        "video_poll_timeout_sec": VIDEO_POLL_TIMEOUT,
        "video_poll_interval_sec": VIDEO_POLL_INTERVAL,
        "video_rpm_guard_sec": 0,
        "video_download_timeout_sec": VIDEO_DOWNLOAD_TIMEOUT,
        "video_download_retries": VIDEO_DOWNLOAD_RETRIES,
        "video_download_backoff_sec": VIDEO_DOWNLOAD_BACKOFF,
        "video_backend": "Agnes Video v2.0 img2video",
    })
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("STATE_OK backend=lightning_ace_step_gpu_longform_resilient_video kaggle=false zero_cost=true machine=T4 video_rpm_guard=0", flush=True)
