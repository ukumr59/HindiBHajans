"""Zero-cost ACE-Step audio dispatcher using Kaggle free GPU.

Control plane: push -> verify credential read access -> poll kernel status ->
retrieve output with the supported `kaggle kernels output` command.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
WORK = ROOT / ".kaggle_audio_worker"
RAW_WORKER = "https://raw.githubusercontent.com/ukumr59/HindiBHajans/main/app/kaggle_ace_step_worker.py"
# A previously-created kernel that was successfully readable by the same
# credential. This distinguishes credential scope from newly-created-kernel
# propagation/access errors returned by Kaggle.
READ_PROBE_KERNEL = "bhajanaabha/hindibhajans-ace-step-1789124667"


def run(*args, env=None, check=True, capture=False):
    print("RUN:", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), text=True, check=check, env=env, capture_output=capture)


def permission_denied(text: str) -> bool:
    low = text.lower()
    return "permission 'kernels.get' was denied" in low or "permission kernels.get was denied" in low


def main():
    token = os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_API_TOKEN3")
    user = os.getenv("KAGGLE_USERNAME", "").strip()
    seconds = int(os.getenv("VIDEO_SECONDS", "180"))
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN secret is required")
    if not user:
        raise RuntimeError("KAGGLE_USERNAME secret is required")
    if not 180 <= seconds <= 300 or seconds % 15:
        raise RuntimeError("VIDEO_SECONDS must be 180-300 and divisible by 15")

    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)

    from app.generate_bhajan_audio import LYRICS, PROMPT
    request = {
        "duration": seconds,
        "caption": PROMPT,
        "lyrics": LYRICS,
        "bpm": 128,
        "keyscale": "C Major",
        "timesignature": "4/4",
        "vocal_language": "hi",
    }
    (WORK / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    bootstrap = f'''#!/usr/bin/env python3
import urllib.request, subprocess, sys, shutil
url={RAW_WORKER!r}
path="/kaggle/working/worker.py"
urllib.request.urlretrieve(url, path)
shutil.copy2('/kaggle/working/request.json', '/kaggle/working/bhajan_request.json')
subprocess.run([sys.executable, path], check=True)
'''
    (WORK / "kernel.py").write_text(bootstrap, encoding="utf-8")

    slug = f"hindibhajans-ace-step-{int(time.time())}"
    kernel = f"{user}/{slug}"
    meta = {
        "id": kernel,
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (WORK / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = token
    p = run(
        "kaggle", "kernels", "push", "-p", str(WORK),
        "--accelerator", "NvidiaTeslaT4", "--timeout", str(11 * 60 * 60),
        env=env, capture=True,
    )
    push_log = (p.stdout or "") + (p.stderr or "")
    print(push_log, flush=True)
    if p.returncode:
        raise RuntimeError("KAGGLE_AUDIO_PUSH_FAILED: " + push_log)

    # First prove the credential can read an already-known kernel. The
    # dedicated GitHub credential check has already demonstrated this, but we
    # verify it in the same production process so a future secret change cannot
    # be mistaken for a Kaggle propagation problem.
    probe = run("kaggle", "kernels", "status", READ_PROBE_KERNEL, env=env, capture=True, check=False)
    probe_text = (probe.stdout or "") + (probe.stderr or "")
    print("READ_PROBE:", probe_text, flush=True)
    if probe.returncode and permission_denied(probe_text):
        raise RuntimeError(
            "KAGGLE_CREDENTIAL_SCOPE_ERROR: kernels.get/read permission failed on an existing accessible kernel. "
            "Check KAGGLE_API_TOKEN3/KAGGLE_API_TOKEN."
        )
    if probe.returncode and not any(x in probe_text.lower() for x in ("queued", "running", "complete", "error", "failed")):
        raise RuntimeError("KAGGLE_READ_PROBE_FAILED: " + probe_text)
    print("KAGGLE_CREDENTIAL_READ_PROBE=PASS", flush=True)

    # Kaggle can briefly return Permission 'kernels.get' was denied for a
    # kernel immediately after `kernels push`. That is not equivalent to a
    # credential-scope failure when the independent read probe succeeds.
    # Retry only this new-kernel lookup for a bounded 5 minutes.
    deadline = time.time() + 5 * 60
    status = ""
    while time.time() < deadline:
        s = run("kaggle", "kernels", "status", kernel, env=env, capture=True, check=False)
        status = (s.stdout or "") + (s.stderr or "")
        print(status, flush=True)
        low = status.lower()
        if permission_denied(status):
            print("KAGGLE_NEW_KERNEL_ACCESS=PROPAGATING; retrying in 30s", flush=True)
            time.sleep(30)
            continue
        if "complete" in low:
            print("KAGGLE_AUDIO_STATUS=COMPLETE", flush=True)
            break
        if any(x in low for x in ("error", "failed", "cancelled", "canceled")):
            raise RuntimeError("KAGGLE_AUDIO_KERNEL_FAILED: " + status)
        if s.returncode == 0 and any(x in low for x in ("queued", "running")):
            print("KAGGLE_AUDIO_STATUS=ACTIVE", flush=True)
            break
        time.sleep(30)
    else:
        raise TimeoutError("KAGGLE_NEW_KERNEL_ACCESS_TIMEOUT: Kaggle did not expose the pushed kernel within 5 minutes")

    # Poll the actual kernel for completion, with the same bounded permission
    # propagation handling. Never turn a permanent permission failure into an
    # 11-hour blind loop.
    deadline = time.time() + 11 * 60 * 60
    while time.time() < deadline:
        s = run("kaggle", "kernels", "status", kernel, env=env, capture=True, check=False)
        status = (s.stdout or "") + (s.stderr or "")
        print(status, flush=True)
        low = status.lower()
        if permission_denied(status):
            # The credential was already proven against an existing kernel, so
            # this is a transient/new-kernel access issue. Give Kaggle a bounded
            # retry window rather than failing the run immediately.
            print("KAGGLE_NEW_KERNEL_ACCESS=TEMPORARY_DENIAL; retrying in 30s", flush=True)
            time.sleep(30)
            continue
        if "complete" in low:
            print("KAGGLE_AUDIO_STATUS=COMPLETE", flush=True)
            break
        if any(x in low for x in ("error", "failed", "cancelled", "canceled")):
            raise RuntimeError("KAGGLE_AUDIO_KERNEL_FAILED: " + status)
        time.sleep(30)
    else:
        raise TimeoutError("KAGGLE_AUDIO_KERNEL_TIMEOUT")

    outdir = OUT / "kaggle_audio_output"
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    output_deadline = time.time() + 5 * 60
    last_output = ""
    while time.time() < output_deadline:
        p = run(
            "kaggle", "kernels", "output", kernel,
            "-p", str(outdir), "--force",
            env=env, capture=True, check=False,
        )
        last_output = (p.stdout or "") + (p.stderr or "")
        print(last_output, flush=True)
        if p.returncode == 0:
            break
        if permission_denied(last_output):
            print("KAGGLE_OUTPUT_ACCESS=PROPAGATING; retrying in 30s", flush=True)
            time.sleep(30)
            continue
        raise RuntimeError("KAGGLE_AUDIO_OUTPUT_COMMAND_FAILED: " + last_output)
    else:
        raise TimeoutError("KAGGLE_AUDIO_OUTPUT_ACCESS_TIMEOUT: " + last_output)

    candidates = list(outdir.rglob("bhajan_source.mp3"))
    if not candidates:
        raise RuntimeError("KAGGLE_AUDIO_ARTIFACT_MISSING: completed kernel did not return bhajan_source.mp3")
    shutil.copy2(candidates[0], OUT / "bhajan_source.mp3")
    print("KAGGLE_AUDIO_READY", OUT / "bhajan_source.mp3", (OUT / "bhajan_source.mp3").stat().st_size, flush=True)


if __name__ == "__main__":
    main()
