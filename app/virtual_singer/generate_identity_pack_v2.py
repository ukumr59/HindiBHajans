from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Unlike the old img2img pack, this version generates each target composition from
# text while using the master portrait as an IP-Adapter identity reference. This
# allows the framing/pose to genuinely change instead of merely nudging the crop.
VIEWS = [
    (
        "front",
        "front-facing head and shoulders portrait, looking directly into the camera, centered",
    ),
    (
        "three_quarter",
        "three-quarter portrait, body and face turned about 35 degrees to the left, natural gaze",
    ),
    (
        "profile",
        "strict clean left side profile, full side-facing head and shoulders, nose and jaw silhouette clearly visible",
    ),
    (
        "waist_up",
        "waist-up standing portrait, torso visible to the waist, both hands naturally visible, centered full upper body",
    ),
    (
        "full_body",
        "full-body standing portrait, entire person visible from head to feet, natural relaxed devotional posture, generous space around the body",
    ),
]

BASE = (
    "The exact same fictional Indian male devotional singer as the supplied identity reference. "
    "Preserve his recognizable identity: warm medium-brown Indian skin, oval face, deep brown eyes, "
    "straight nose, defined jawline, same hairline, same dark wavy medium-length hair, clean-shaven, "
    "apparent age around 30. He is a human devotional singer, not a deity. "
    "Keep facial identity stable while changing only camera framing and pose. "
    "Photorealistic cinematic Indian devotional cinema, natural skin texture, realistic anatomy, "
    "single person, simple cream kurta, beige stole, subtle rudraksha beads, plain neutral studio-like "
    "background for identity references, soft even lighting, no text, no watermark. "
)

NEGATIVE = (
    "different person, changed identity, different face, altered facial proportions, different eyes, "
    "different nose, different jawline, different hairstyle, beard, moustache, facial hair, old, young, "
    "female, duplicate person, multiple people, deity, crown, fantasy costume, cropped feet, cropped body, "
    "extra limbs, malformed hands, extra fingers, fused fingers, distorted anatomy, text, subtitles, watermark, "
    "logo, cartoon, anime, plastic skin, blurry face, extreme close-up, fisheye"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="identity_pack_v2")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--scale", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=44031)
    args = ap.parse_args()

    src = Image.open(args.input).convert("RGB")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
        image_encoder_folder="models/image_encoder",
    )
    pipe.set_ip_adapter_scale(args.scale)
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    for i, (name, view) in enumerate(VIEWS):
        seed = args.seed + i
        gen = torch.Generator(device="cuda").manual_seed(seed)
        prompt = BASE + view
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            ip_adapter_image=src,
            height=1024,
            width=768,
            num_inference_steps=args.steps,
            guidance_scale=6.0,
            generator=gen,
        ).images[0]
        path = outdir / f"{name}.png"
        image.save(path)
        print(
            f"IDENTITY_VIEW_V2_OK name={name} seed={seed} path={path} bytes={path.stat().st_size}",
            flush=True,
        )

    print("IDENTITY_PACK_V2_OK", flush=True)


if __name__ == "__main__":
    main()
