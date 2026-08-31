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
    "Photorealistic cinematic portrait of a fictional Indian male devotional singer, adult and mature-looking, "
    "visually inspired by the supplied reference-derived character design. Preserve a distinctive oval-to-rectangular "
    "face, warm medium-brown Indian skin tone, dark brown expressive eyes, straight nose, defined natural jawline, "
    "short black hair neatly side-parted, clean-shaven face, and distinctive rectangular dark eyeglasses. "
    "Average-to-fit adult physique with natural shoulders and proportions. Calm, sincere, devotional expression, "
    "subtle confident smile, humble charismatic presence. Wearing a simple cream kurta and traditional beige stole, "
    "rudraksha prayer beads, standing naturally in a softly lit ancient Hindu temple with warm diyas in the background. "
    "Centered three-quarter chest-up portrait, natural skin texture, realistic facial proportions, cinematic Indian "
    "devotional cinema, 85mm portrait lens, shallow depth of field, soft volumetric temple light, highly detailed, "
    "one person only, no text, no watermark."
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
    print("VIRTUAL_SINGER_REFERENCE_PROFILE=adult_indian_male_glasses_short_side_parted_hair_average_fit", flush=True)

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
