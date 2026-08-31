from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "identity_config.json"
MODEL_ID = os.getenv("VIRTUAL_SINGER_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

MASTER_PROMPT = (
    "Photorealistic cinematic portrait of a fictional Indian male devotional singer, age 30, "
    "warm medium-brown skin, expressive deep brown eyes, distinctive oval face, straight nose, "
    "natural eyebrows, dark wavy medium-length hair, clean-shaven, slim-to-average build, "
    "calm sincere devotional expression, subtle smile, wearing a simple cream kurta and "
    "traditional beige stole, rudraksha prayer beads, standing in a softly lit ancient Hindu "
    "temple with warm diyas in the background, centered three-quarter portrait, chest-up, "
    "natural skin texture, realistic facial proportions, cinematic Indian devotional cinema, "
    "85mm portrait lens, shallow depth of field, soft volumetric temple light, highly detailed, "
    "no text, no watermark, one person only."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44031)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--output", default="virtual_singer_master.png")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    negative = config["negative_prompt"]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"VIRTUAL_SINGER_MODEL={MODEL_ID}", flush=True)
    print(f"VIRTUAL_SINGER_SEED={args.seed}", flush=True)

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    image = pipe(
        prompt=MASTER_PROMPT,
        negative_prompt=negative,
        height=1024,
        width=768,
        num_inference_steps=args.steps,
        guidance_scale=6.5,
        generator=generator,
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"VIRTUAL_SINGER_MASTER_OK path={output} bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
