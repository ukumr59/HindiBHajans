from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

PROMPT = (
    "photorealistic Indian male devotional singer, the same person as the supplied reference photograph, "
    "preserve recognizable facial identity, same facial proportions, same hairline, same short black side-parted hair, "
    "same rectangular dark eyeglasses, clean-shaven, same adult Indian male appearance and physique, calm sincere expression, "
    "simple cream kurta and beige stole, ancient Hindu temple softly lit by diyas, cinematic devotional photography, "
    "natural skin texture, realistic anatomy, chest-up three-quarter portrait, centered subject, no text"
)
NEGATIVE = (
    "different person, changed identity, altered face, different eyes, different nose, different jaw, different hairline, "
    "no glasses, different glasses, sunglasses, beard, moustache, tilak, jewelry, deity, crown, extra people, distorted face, "
    "duplicate person, text, watermark, cartoon, anime, plastic skin, blurry face"
)


def load_reference(reference_path: str | None) -> Image.Image:
    if reference_path:
        path = Path(reference_path)
        if not path.is_file():
            raise RuntimeError(f"Reference image not found: {path}")
        return Image.open(path).convert("RGB")
    raw = os.environ.get("SINGER_REFERENCE_B64")
    if not raw:
        raise RuntimeError("No singer reference supplied")
    try:
        return Image.open(io.BytesIO(base64.b64decode(raw, validate=True))).convert("RGB")
    except Exception as exc:
        raise RuntimeError("SINGER_REFERENCE_B64 is not valid base64 image data") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=71284)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--output", default="virtual_singer_master_v3.png")
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()

    reference = load_reference(args.reference)
    print("VIRTUAL_SINGER_IDENTITY_SOURCE=actual_user_reference_photo", flush=True)
    print(f"REFERENCE_SIZE={reference.width}x{reference.height}", flush=True)

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

    # T4/CPU-offload runs can leave the CLIP vision encoder in float16 while
    # its convolution receives a CPU float16 tensor. CPU float16 convolution
    # is not reliably supported. Keep only the vision encoder in float32;
    # the SDXL denoiser/VAEs remain float16 for T4 memory efficiency.
    if getattr(pipe, "image_encoder", None) is not None:
        pipe.image_encoder = pipe.image_encoder.float()
        print("IP_ADAPTER_IMAGE_ENCODER_DTYPE=float32", flush=True)

    pipe.set_ip_adapter_scale(0.92)
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        ip_adapter_image=reference,
        height=768,
        width=576,
        num_inference_steps=args.steps,
        guidance_scale=6.0,
        generator=generator,
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"VIRTUAL_SINGER_MASTER_OK path={output} bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
