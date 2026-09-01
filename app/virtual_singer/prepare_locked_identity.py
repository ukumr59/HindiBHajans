from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

from PIL import Image, ImageEnhance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-b64", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    raw = base64.b64decode(args.input_b64, validate=True)
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    # This crop matches the approved MASTER IDENTITY portrait in the user's
    # 1024x1536 identity sheet. It intentionally excludes the Identity Profile
    # text panel to the right.
    w, h = image.size
    x1, y1 = round(w * 35 / 1024), round(h * 85 / 1536)
    x2, y2 = round(w * 245 / 1024), round(h * 505 / 1536)
    master = image.crop((x1, y1, x2, y2))

    # Only a very mild exposure lift. No generative reconstruction is allowed;
    # the face, glasses and hairstyle remain the approved source pixels.
    master = ImageEnhance.Brightness(master).enhance(1.04)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    master.save(output, format="PNG", optimize=True)

    print(f"LOCKED_IDENTITY_OK path={output} size={master.size} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
