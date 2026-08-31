from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from insightface.app import FaceAnalysis
from diffusers import ControlNetModel

from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT / "checkpoints"
INSTANTID_ROOT = CHECKPOINT_ROOT / "InstantID"
FACE_MODEL_ROOT = ROOT / "models"
BASE_MODEL = os.getenv("VIRTUAL_SINGER_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

PROMPT = (
    "photorealistic Indian male devotional singer, approximately 30 years old, "
    "preserve the exact facial identity from the supplied reference photograph, "
    "short neatly side-parted black hair, rectangular dark eyeglasses, clean-shaven, "
    "natural medium-brown skin, realistic adult male physique, calm sincere expression, "
    "simple cream kurta and beige stole, ancient Hindu temple softly lit by diyas, "
    "cinematic devotional photography, natural skin texture, realistic proportions, "
    "chest-up three-quarter portrait, centered subject, 85mm lens, shallow depth of field"
)
NEGATIVE = (
    "different person, altered identity, no glasses, different glasses, sunglasses, "
    "long hair, beard, moustache, tilak, earrings, ornate jewelry, deity, crown, "
    "extra people, distorted face, duplicate person, text, watermark, cartoon, anime"
)


def load_reference(reference_path: str | None) -> Image.Image:
    if reference_path:
        path = Path(reference_path)
        if not path.is_file():
            raise RuntimeError(f"Reference image not found: {path}")
        return Image.open(path).convert("RGB")

    raw = os.environ.get("SINGER_REFERENCE_B64")
    if not raw:
        raise RuntimeError(
            "No singer reference supplied. Provide --reference or SINGER_REFERENCE_B64."
        )
    try:
        return Image.open(io.BytesIO(base64.b64decode(raw, validate=True))).convert("RGB")
    except Exception as exc:
        raise RuntimeError("SINGER_REFERENCE_B64 is not valid base64 image data") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=81273)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--output", default="virtual_singer_master_v3.png")
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    reference_path = ROOT / "reference_input.jpg"
    reference.save(reference_path, quality=95)

    print("VIRTUAL_SINGER_IDENTITY_SOURCE=actual_user_reference_photo", flush=True)
    print(f"VIRTUAL_SINGER_BASE_MODEL={BASE_MODEL}", flush=True)

    face_app = FaceAnalysis(
        name="antelopev2",
        root=str(FACE_MODEL_ROOT),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    faces = face_app.get(cv2.cvtColor(np.array(reference), cv2.COLOR_RGB2BGR))
    if not faces:
        raise RuntimeError("No face detected in the supplied reference photograph")
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    face_emb = face.embedding
    face_kps = draw_kps(reference, face.kps)

    controlnet = ControlNetModel.from_pretrained(
        str(INSTANTID_ROOT / "ControlNetModel"),
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    pipe.load_ip_adapter_instantid(str(INSTANTID_ROOT / "ip-adapter.bin"))

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        PROMPT,
        negative_prompt=NEGATIVE,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.92,
        ip_adapter_scale=0.92,
        num_inference_steps=args.steps,
        guidance_scale=5.0,
        generator=generator,
        height=768,
        width=576,
    ).images[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"VIRTUAL_SINGER_MASTER_OK path={output} bytes={output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
