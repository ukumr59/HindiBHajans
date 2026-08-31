from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "wan2_worker"
WAN = WORK / "Wan2.1"
MODEL = WORK / "Wan2.1-T2V-1.3B"
REQUEST = ROOT / "wan_request.json"
OUT = WORK / "scenes"
WAN_REPO = "https://github.com/Wan-Video/Wan2.1.git"


def run(*args, cwd=None):
    print("WAN_RUN:", " ".join(map(str, args)), flush=True)
    return subprocess.run([str(x) for x in args], cwd=cwd, check=True, text=True)


def patch_wan_attention_for_t4():
    """Make Wan2.1's direct flash_attention call use native PyTorch SDPA on T4."""
    attention = WAN / "wan" / "modules" / "attention.py"
    text = attention.read_text(encoding="utf-8")
    marker = "WAN_T4_SDPA_FALLBACK_PATCH"
    if marker in text:
        print("WAN_ATTENTION_PATCH_OK cached", flush=True)
        return

    # WanModel calls flash_attention() directly. Upstream's flash_attention()
    # asserts flash-attn 2 when FA3/FA2 are unavailable, so attention()'s own
    # fallback is never reached. Patch the function at its start instead of
    # matching the larger implementation block; this is robust to upstream
    # formatting changes.
    needle = "    half_dtypes = (torch.float16, torch.bfloat16)\n"
    if needle not in text:
        raise RuntimeError("WAN_PATCH_FATAL: flash_attention() anchor not found")

    fallback = '''    # WAN_T4_SDPA_FALLBACK_PATCH\n    # Free Lightning T4 images may not have a compatible flash-attn wheel.\n    # WanModel calls this function directly, so provide a native PyTorch SDPA\n    # path before the upstream flash-attn-only implementation.\n    if not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE:\n        if q_lens is not None or k_lens is not None:\n            warnings.warn("T4 SDPA fallback ignores variable-length padding masks.")\n        q_sdpa = q.transpose(1, 2).to(torch.float16)\n        k_sdpa = k.transpose(1, 2).to(torch.float16)\n        v_sdpa = v.transpose(1, 2).to(torch.float16)\n        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)\n        if enable_gqa and q_sdpa.size(1) % k_sdpa.size(1) != 0:\n            raise RuntimeError("WAN_T4_SDPA_FATAL: incompatible Q/K head counts")\n        out = torch.nn.functional.scaled_dot_product_attention(\n            q_sdpa, k_sdpa, v_sdpa,\n            attn_mask=None,\n            dropout_p=dropout_p,\n            is_causal=causal,\n            scale=softmax_scale,\n            enable_gqa=enable_gqa,\n        )\n        return out.transpose(1, 2).contiguous().to(q.dtype)\n\n'''
    text = text.replace(needle, fallback + needle, 1)
    attention.write_text(text, encoding="utf-8")
    print("WAN_ATTENTION_PATCH_OK applied T4 SDPA fallback", flush=True)


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    run("nvidia-smi")
    if not WAN.exists():
        run("git", "clone", "--no-tags", "--depth", "1", WAN_REPO, WAN)
    patch_wan_attention_for_t4()
    if not (WORK / ".deps_ok").exists():
        run(sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "ftfy", "imageio", "imageio-ffmpeg")
        requirements = WAN / "requirements.txt"
        filtered = WORK / "requirements-no-flash-attn.txt"
        lines = requirements.read_text(encoding="utf-8").splitlines()
        filtered.write_text(
            "\n".join(line for line in lines if line.strip().lower() != "flash_attn") + "\n",
            encoding="utf-8",
        )
        run(sys.executable, "-m", "pip", "install", "-q", "-r", str(filtered))
        (WORK / ".deps_ok").write_text("ok", encoding="utf-8")

    marker = MODEL / ".download_complete"
    if not marker.exists():
        from huggingface_hub import snapshot_download
        print("WAN_MODEL_DOWNLOAD: downloading Wan-AI/Wan2.1-T2V-1.3B", flush=True)
        snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=str(MODEL))
        marker.write_text("ok", encoding="utf-8")
        print("WAN_MODEL_DOWNLOAD_OK", flush=True)
    else:
        print("WAN_MODEL_CACHE_OK", flush=True)


def generate_scene(index: int, prompt: str, seed: int):
    raw = OUT / f"scene_{index}_raw.mp4"
    final = OUT / f"scene_{index}.mp4"
    if final.exists() and final.stat().st_size > 50_000:
        print(f"WAN_SCENE_CACHED scene={index}", flush=True)
        return
    raw.unlink(missing_ok=True)
    final.unlink(missing_ok=True)

    cmd = [
        sys.executable, "generate.py",
        "--task", "t2v-1.3B",
        "--size", "480*832",
        "--ckpt_dir", str(MODEL),
        "--offload_model", "True",
        "--t5_cpu",
        "--sample_shift", "8",
        "--sample_guide_scale", "6",
        "--sample_steps", os.getenv("WAN_SAMPLE_STEPS", "30"),
        "--frame_num", "81",
        "--base_seed", str(seed),
        "--save_file", str(raw),
        "--prompt", prompt,
    ]
    print(f"WAN_SCENE_START scene={index} seed={seed}", flush=True)
    run(*cmd, cwd=WAN)
    if not raw.exists() or raw.stat().st_size < 50_000:
        raise RuntimeError(f"WAN_FATAL: scene {index} raw MP4 missing/small")

    run(
        "ffmpeg", "-y", "-i", raw,
        "-vf", "setpts=3.0*PTS,fps=16,scale=480:832:flags=lanczos",
        "-an", "-t", "15",
        "-c:v", "libx264", "-preset", "fast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", final,
    )
    if not final.exists() or final.stat().st_size < 50_000:
        raise RuntimeError(f"WAN_FATAL: scene {index} final MP4 missing/small")
    raw.unlink(missing_ok=True)
    print(f"WAN_SCENE_OK scene={index} path={final} bytes={final.stat().st_size}", flush=True)


def main():
    prepare()
    req = json.loads(REQUEST.read_text(encoding="utf-8"))
    for item in req["scenes"]:
        generate_scene(int(item["index"]), item["prompt"], int(item["seed"]))
    manifest = {
        "backend": "Wan2.1-T2V-1.3B",
        "license": "Apache-2.0",
        "resolution": "480x832",
        "source_seconds": 5,
        "output_seconds": 15,
        "scenes": req["scenes"],
    }
    (WORK / "wan_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WAN_ALL_SCENES_OK", flush=True)


if __name__ == "__main__":
    main()
