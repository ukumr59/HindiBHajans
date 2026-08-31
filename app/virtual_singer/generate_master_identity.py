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

# The singer is fictional, while the visual design is derived from the user's
# supplied self-photo: glasses, hairstyle, facial structure and general build.
MASTER_PROMPT = (
    "Photorealistic cinematic portrait of a fictional Indian male devotional singer, age around 30, "
    "with facial features closely matching the approved self-reference design: warm medium-brown skin, "
    "short neatly combed black hair with a clean side part, broad forehead, distinctive oval-to-rectangular "
    "face, dark brown eyes, straight medium-width nose, natural medium lips, clean-shaven face, "
    "defined but natural jawline, average-to-fit adult physique, and rectangular black eyeglasses with "
    "thin dark frames. Keep the eyeglasses as a permanent identity feature. Calm sincere devotional "
    "expression, approachable and humble presence. The character is fictional and must remain one consistent "
    "person. Wearing a simple cream kurta and beige stole, standing in a softly lit ancient Hindu temple "
    "with warm diyas in the background. Centered three-quarter chest-up portrait, natural skin texture, "
    "realistic facial proportions, cinematic Indian devotional cinema, 85mm portrait lens, shallow depth "
    "of field, soft volumetric temple light, highly detailed, one person only, no text, no watermark, "
    "no tilak, no earrings, no facial hair, no ornate jewelry."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71284)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", default="virtual_singer_master.png")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    negative = config["negative_prompt"] + ", sunglasses, rimless glasses, different eyeglasses, missing glasses"
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"VIRTUAL_SINGER_MODEL={MODEL_ID}", flush=True)
    print(f"VIRTUAL_SINGER_SEED={args.seed}", flush=True)
    print("VIRTUAL_SINGER_IDENTITY_SOURCE=approved_self_reference_traits", flush=True)

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
