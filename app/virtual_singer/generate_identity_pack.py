from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

VIEWS = [
    ("front", "front-facing head and shoulders portrait, looking directly at camera"),
    ("three_quarter_left", "three-quarter view turned slightly to his left, head and shoulders"),
    ("profile", "clean natural left side profile portrait, head and shoulders"),
    ("waist_up", "waist-up portrait, standing naturally, both hands visible, centered composition"),
]

BASE = (
    "The EXACT SAME fictional Indian male devotional singer from the supplied reference image. "
    "Preserve his identity precisely: same facial proportions, same deep brown eyes, same oval face, "
    "same straight nose, same jawline, same hairline, same dark wavy medium-length hair, same warm "
    "medium-brown skin tone, same apparent age around 30, clean-shaven. Do not redesign the person. "
    "Photorealistic cinematic Indian devotional cinema, natural skin texture, realistic anatomy, "
    "single person, no text, no watermark. "
)

NEGATIVE = (
    "different person, changed identity, different face, altered facial proportions, different eyes, "
    "different nose, different jawline, different hairstyle, beard, moustache, facial hair, old, young, "
    "female, duplicate person, multiple people, deformed face, asymmetrical eyes, malformed hands, "
    "extra fingers, text, subtitles, watermark, logo, cartoon, anime, plastic skin, blurry face"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="identity_pack")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--strength", type=float, default=0.28)
    ap.add_argument("--seed", type=int, default=44031)
    args = ap.parse_args()

    src = Image.open(args.input).convert("RGB")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    for i, (name, view) in enumerate(VIEWS):
        seed = args.seed + i
        gen = torch.Generator(device="cuda").manual_seed(seed)
        prompt = BASE + view + ", wearing a simple cream kurta and beige stole, subtle rudraksha beads."
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=src,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=5.5,
            generator=gen,
        ).images[0]
        path = outdir / f"{name}.png"
        image.save(path)
        print(f"IDENTITY_VIEW_OK name={name} seed={seed} path={path} bytes={path.stat().st_size}", flush=True)

    print("IDENTITY_PACK_OK", flush=True)


if __name__ == "__main__":
    main()
