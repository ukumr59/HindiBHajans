from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

import app.zero_cost_pipeline_v5 as base

# IMPORTANT: the public ACE-Step 1.5 Hugging Face Space uses the classic
# asynchronous API: POST /release_task -> POST /query_result -> GET /v1/audio.
# /v1/music/generate is NOT exposed by this public Space and returns HTTP 405.
ACESTEP_API = base.ACESTEP_ROOT

DJ_MUSIC_PROMPT = """Modern high-energy Hindi devotional bhajan made like a current YouTube DJ devotional song, 128 BPM, 4/4, loud polished commercial stereo production, powerful sung Hindi male lead vocal clearly singing every lyric with natural emotion and pronunciation, catchy devotional melody, huge memorable chorus, energetic EDM arrangement, punchy four-on-the-floor kick, deep controlled sub bass, modern synth bass, bright synth leads, wide pads, electronic percussion, claps, dhol and dholak layered with tabla, cinematic risers, tasteful temple bells, bansuri accents, harmonium texture, short instrumental intro, strong verse build, massive chorus/drop, rhythmic instrumental break, final chorus with layered backing vocals, professional radio/YouTube loudness and DJ playback energy. The devotional identity must remain unmistakable while the production feels contemporary, energetic and danceable. NOT meditation music, NOT sleepy, NOT ambient, NOT acoustic-only, NOT spoken narration, NOT chanting without melody, NOT humming, NOT a cappella, NOT background music, NOT an instrumental-only track."""


def _extract_task_id(data):
    if isinstance(data, dict):
        if data.get("task_id"):
            return data["task_id"]
        inner = data.get("data")
        if isinstance(inner, dict) and inner.get("task_id"):
            return inner["task_id"]
    raise RuntimeError(f"MUSIC_FATAL: ACE-Step task id missing: {data}")


def _result_audio_ref(item):
    raw = item.get("result", "[]") if isinstance(item, dict) else "[]"
    try:
        results = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MUSIC_FATAL: invalid ACE-Step result JSON: {raw}") from exc
    if not results:
        return None
    first = results[0] if isinstance(results, list) else results
    if not isinstance(first, dict):
        return None
    return first.get("file") or first.get("url")


def generate_music_http(session: requests.Session) -> Path:
    duration = int(base.VIDEO_SECONDS)
    print("MUSIC: ACE-Step 1.5 public Hugging Face Space asynchronous API")
    print("MUSIC: endpoint=POST /release_task then POST /query_result")
    print("MUSIC: style=LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY bpm=128")

    payload = {
        "prompt": DJ_MUSIC_PROMPT,
        "lyrics": base.PACK["lyrics"],
        "vocal_language": "hi",
        "audio_duration": duration,
        "model": "acestep-v15-turbo",
        "thinking": True,
        "sample_mode": False,
        "use_format": True,
        "inference_steps": 8,
        "batch_size": 1,
        "use_random_seed": True,
        "task_type": "text2music",
        "bpm": 128,
        "time_signature": "4",
        "key_scale": "C Major",
        "audio_format": "mp3",
    }

    data = base.http_json(
        session,
        "POST",
        f"{ACESTEP_API}/release_task",
        headers={"Content-Type": "application/json"},
        body=payload,
        timeout=120,
        retries=5,
    )
    task_id = _extract_task_id(data)
    print("MUSIC_TASK", task_id)

    deadline = time.time() + 30 * 60
    last_status = None
    while time.time() < deadline:
        result = base.http_json(
            session,
            "POST",
            f"{ACESTEP_API}/query_result",
            headers={"Content-Type": "application/json"},
            body={"task_id_list": [task_id]},
            timeout=60,
            retries=3,
        )
        items = result.get("data") if isinstance(result, dict) else result
        if isinstance(items, dict):
            items = items.get("data") or items.get("results") or []
        if not isinstance(items, list) or not items:
            time.sleep(5)
            continue

        item = items[0]
        status = int(item.get("status", 0))
        if status != last_status:
            print(f"MUSIC: task={task_id} status={status}")
            last_status = status

        if status == 2:
            raise RuntimeError(f"MUSIC_FATAL: ACE-Step generation failed: {item.get('result', item)}")
        if status == 1:
            audio_ref = _result_audio_ref(item)
            if not audio_ref:
                raise RuntimeError(f"MUSIC_FATAL: successful task has no audio file: {item}")
            audio_url = audio_ref if str(audio_ref).startswith("http") else urljoin(ACESTEP_API + "/", str(audio_ref).lstrip("/"))
            target = base.AUDIO / "bhajan_source.mp3"
            base.download(session, audio_url, target, min_bytes=20000)
            print("MUSIC_OK", target, target.stat().st_size)
            return target
        time.sleep(5)

    raise RuntimeError(f"MUSIC_FATAL: ACE-Step task timed out after 30 minutes: {task_id}")


def make_dj_master(final_video: Path) -> Path:
    """Extract and master the generated song as a standalone DJ-friendly MP3."""
    target = base.AUDIO / "bhajan_aabha_dj_master.mp3"
    cmd = [
        "ffmpeg", "-y", "-i", str(final_video), "-vn",
        "-af", "highpass=f=28,lowpass=f=19000,loudnorm=I=-9:TP=-1.0:LRA=7",
        "-ar", "48000", "-ac", "2", "-c:a", "libmp3lame", "-b:a", "320k", str(target),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError("DJ_MASTER_FATAL: " + p.stderr[-4000:])
    if target.stat().st_size < 50000:
        raise RuntimeError("DJ_MASTER_FATAL: master file is suspiciously small")
    print("DJ_MASTER_OK", target, target.stat().st_size)
    return target


base.generate_music = generate_music_http
base.ACESTEP_ROOT = ACESTEP_API

if __name__ == "__main__":
    base.main()
    videos = sorted(base.VIDEOS.glob("*.mp4"))
    if not videos:
        raise RuntimeError("OUTPUT_FATAL: final video was not produced")
    dj_master = make_dj_master(videos[0])
    state_path = base.OUT / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["music_backend"] = "ACE-Step 1.5 public ZeroGPU Space /release_task + /query_result"
        state["music_api_mode"] = "http_async_release_task_query_result"
        state["music_style"] = "LOUD_MODERN_DEVOTIONAL_EDM_DJ_READY"
        state["bpm"] = 128
        state["time_signature"] = "4/4"
        state["dj_master"] = str(dj_master)
        state["dj_master_bitrate"] = "320k"
        state["dj_master_sample_rate"] = 48000
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        (base.OUT / "manifest.json").write_text(json.dumps({"videos": [state]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("STATE_OK music_backend=ACE-Step public Space release_task/query_result")
