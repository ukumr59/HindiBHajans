from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL = None
PROCESSOR = None

# CLIP is used only as a conservative visual gate. It is not treated as proof
# of deity identity; failure blocks publication rather than silently accepting
# an unrelated scene.
SCENE_LABELS = {
    1: ["a Hindu devotional image of Lord Rama", "a Lord Rama idol in a Hindu temple"],
    2: ["a close-up Lord Rama Hindu devotional image", "a Shri Ram deity idol"],
    3: ["Lord Rama with a diya or aarti in a Hindu temple", "a Hindu devotional Lord Rama scene"],
    4: ["Lord Rama deity in a Hindu temple", "Shri Ram devotional worship"],
    5: ["Lord Rama and a Hindu devotee praying", "Lord Rama protecting a devotee"],
    6: ["Lord Rama with a praying Hindu devotee", "Shri Ram devotional worship"],
    7: ["Lord Rama and Hanuman in Hindu devotional worship", "Rama Hanuman devotional scene"],
    8: ["Lord Rama and Ayodhya Ram Mandir", "Shri Ram temple in Ayodhya"],
    9: ["Lord Rama with Hindu devotees in a temple", "Ram bhakti Hindu devotional worship"],
    10: ["Lord Rama and Hanuman during Hindu aarti", "Rama Hanuman devotional worship"],
    11: ["a prominent Lord Rama deity hero image", "a close-up Shri Ram idol"],
    12: ["Lord Rama idol with diya in a Hindu temple", "Shri Ram devotional closing scene"],
}

FORBIDDEN = [
    "a mosque or masjid",
    "an Islamic religious building",
    "a church or cathedral",
    "a Christian cross",
    "a synagogue",
    "a Buddhist pagoda or stupa",
    "a Sikh gurdwara",
    "a Jain temple",
]


def _load():
    global MODEL, PROCESSOR
    if MODEL is None:
        PROCESSOR = CLIPProcessor.from_pretrained(MODEL_ID)
        MODEL = CLIPModel.from_pretrained(MODEL_ID)
        MODEL.eval()


def _frames(video: Path, scene: int) -> list[Image.Image]:
    with tempfile.TemporaryDirectory() as td:
        pattern = str(Path(td) / "frame_%02d.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", "fps=1/5,scale=512:-1", "-frames:v", "3", pattern],
            check=True,
        )
        paths = sorted(Path(td).glob("frame_*.jpg"))
        images = [Image.open(p).convert("RGB").copy() for p in paths]
    if len(images) < 2:
        raise RuntimeError(f"VISUAL_QC_FATAL: scene {scene} yielded fewer than 2 inspection frames")
    return images


def _scores(images: list[Image.Image], labels: list[str]) -> list[list[float]]:
    texts = labels + FORBIDDEN
    inputs = PROCESSOR(text=texts, images=images, return_tensors="pt", padding=True)
    with torch.inference_mode():
        out = MODEL(**inputs)
        probs = out.logits_per_image.softmax(dim=1)
    return probs.tolist()


def validate_scene(video: Path, scene: int) -> dict:
    if scene not in SCENE_LABELS:
        raise RuntimeError(f"VISUAL_QC_FATAL: no scene definition for {scene}")
    images = _frames(video, scene)
    probs = _scores(images, SCENE_LABELS[scene])
    positive_count = 0
    forbidden_hits = []
    frame_results = []
    positive_n = len(SCENE_LABELS[scene])
    for idx, row in enumerate(probs, start=1):
        pos = max(row[:positive_n])
        neg = max(row[positive_n:])
        best_pos = SCENE_LABELS[scene][max(range(positive_n), key=lambda j: row[j])]
        best_neg = FORBIDDEN[max(range(len(FORBIDDEN)), key=lambda j: row[positive_n + j])]
        # Conservative gate: positive semantic match must beat every forbidden
        # religious category by a margin and have meaningful probability.
        accepted = pos >= 0.30 and pos >= neg + 0.08
        if accepted:
            positive_count += 1
        if neg >= 0.28 and neg >= pos - 0.02:
            forbidden_hits.append({"frame": idx, "label": best_neg, "score": round(neg, 4)})
        frame_results.append({"frame": idx, "positive": best_pos, "positive_score": round(pos, 4), "closest_forbidden": best_neg, "forbidden_score": round(neg, 4), "accepted": accepted})

    passed = positive_count >= 2 and not forbidden_hits
    result = {"scene": scene, "passed": passed, "accepted_frames": positive_count, "frames": frame_results, "forbidden_hits": forbidden_hits}
    if not passed:
        raise RuntimeError("VISUAL_QC_REJECT: " + json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    root = Path("output/videos")
    scenes = sorted(root.glob("scene_*.mp4"))
    # The pipeline normally deletes intermediates after assembly. This QC is
    # therefore run against the final video by extracting 12 equal time slices.
    final = next(root.glob("*.mp4"), None)
    if final is None:
        raise RuntimeError("VISUAL_QC_FATAL: final video not found")
    _load()
    # Inspect the final video at 12 scene centers. This catches obvious
    # religious-content mismatches before release/publication.
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(final)], text=True).strip())
    results = []
    for scene in range(1, 13):
        start = max(0.0, (scene - 1) * 15.0 + 4.0)
        with tempfile.TemporaryDirectory() as td:
            sample = Path(td) / f"scene_{scene}.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", str(final), "-t", "6", "-c", "copy", str(sample)], check=True)
            results.append(validate_scene(sample, scene))
    out = Path("output/visual_qc.json")
    out.write_text(json.dumps({"model": MODEL_ID, "passed": True, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VISUAL_QC_OK: all 12 scenes passed semantic devotional gate")


if __name__ == "__main__":
    main()
