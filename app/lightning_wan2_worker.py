from __future__ import annotations

import json
import os
import re
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
    """Patch upstream Wan2.1's hard flash-attn assertion with PyTorch SDPA.

    Wan2.1's WanModel calls flash_attention() directly, so the fallback in
    attention() is not reached when flash-attn is absent. On the free T4 image
    flash-attn is intentionally not installed because its native build is
    unreliable under the provided Python 3.12 environment. Use native PyTorch
    SDPA instead, with fp16 on T4 for tensor-core execution.
    """
    attention = WAN / "wan" / "modules" / "attention.py"
    text = attention.read_text(encoding="utf-8")
    marker = "WAN_T4_SDPA_FALLBACK_PATCH"
    if marker in text:
        print("WAN_ATTENTION_PATCH_OK cached", flush=True)
        return

    pattern = re.compile(
        r"    else:\n"
        r"        assert FLASH_ATTN_2_AVAILABLE\n"
        r"        x = flash_attn\.flash_attn_varlen_func\(.*?\n"
        r"        \)\.unflatten\(0, \(b, lq\)\)",
        re.DOTALL,
    )

    replacement = '''    else:\n        if FLASH_ATTN_2_AVAILABLE:\n            x = flash_attn.flash_attn_varlen_func(\n                q=q,\n                k=k,\n                v=v,\n                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(\n                    0, dtype=torch.int32).to(q.device, non_blocking=True),\n                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(\n                    0, dtype=torch.int32).to(q.device, non_blocking=True),\n                max_seqlen_q=lq,\n                max_seqlen_k=lk,\n                dropout_p=dropout_p,\n                softmax_scale=softmax_scale,\n                causal=causal,\n                window_size=window_size,\n                deterministic=deterministic).unflatten(0, (b, lq))\n        else:\n            # WAN_T4_SDPA_FALLBACK_PATCH\n            # The upstream WanModel calls flash_attention() directly, which\n            # otherwise asserts when flash-attn is unavailable. Run each\n            # variable-length sequence through native PyTorch SDPA. Convert\n            # to fp16 because T4 tensor cores natively accelerate fp16.\n            if window_size != (-1, -1):\n                warnings.warn('T4 SDPA fallback ignores Wan local window_size.')\n            q0 = k0 = 0\n            chunks = []\n            q_lengths = [int(z) for z in q_lens.detach().cpu().tolist()]\n            k_lengths = [int(z) for z in k_lens.detach().cpu().tolist()]\n            for qlen, klen in zip(q_lengths, k_lengths):\n                qi = q[q0:q0 + qlen].transpose(0, 1).unsqueeze(0).to(torch.float16)\n                ki = k[k0:k0 + klen].transpose(0, 1).unsqueeze(0).to(torch.float16)\n                vi = v[k0:k0 + klen].transpose(0, 1).unsqueeze(0).to(torch.float16)\n                if qi.size(1) != ki.size(1):\n                    if qi.size(1) % ki.size(1) != 0:\n                        raise RuntimeError('WAN_T4_SDPA_FATAL: incompatible Q/K head counts')\n                    repeats = qi.size(1) // ki.size(1)\n                    ki = ki.repeat_interleave(repeats, dim=1)\n                    vi = vi.repeat_interleave(repeats, dim=1)\n                yi = torch.nn.functional.scaled_dot_product_attention(\n                    qi, ki, vi,\n                    attn_mask=None,\n                    dropout_p=dropout_p,\n                    is_causal=causal,\n                    scale=softmax_scale,\n                )\n                yi = yi.squeeze(0).transpose(0, 1)\n                if qlen < lq:\n                    pad = yi.new_zeros((lq - qlen, yi.size(1), yi.size(2)))\n                    yi = torch.cat([yi, pad], dim=0)\n                chunks.append(yi)\n                q0 += qlen\n                k0 += klen\n            x = torch.stack(chunks, dim=0).to(v.dtype)'''

    patched, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("WAN_PATCH_FATAL: upstream flash-attention block not found")
    attention.write_text(patched, encoding="utf-8")
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
        # The upstream requirements include flash_attn. On the free Lightning
        # T4 Python 3.12 image it may try to build native code and abort before
        # Wan starts. Exclude only that package; the runtime patch above handles
        # WanModel's direct flash_attention() call with native PyTorch SDPA.
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
