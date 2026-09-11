from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

COSINE_THRESHOLD = 0.50
MIN_MATCH_RATIO = 0.80
MIN_FACE_RATIO = 0.02
MAX_FACE_RATIO = 0.65
MIN_MOTION_RATIO = 0.30


def detect(detector, frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return [f for f in faces if float(f[-1]) >= 0.85]


def feature(recognizer, image, face):
    aligned = recognizer.alignCrop(image, np.asarray(face, dtype=np.float32))
    return recognizer.feature(aligned)


def mouth_patch(frame, face):
    x, y, w, h = [float(v) for v in face[:4]]
    # YuNet landmarks: eyes, nose, right/left mouth corners.
    mx = (float(face[9]) + float(face[11])) / 2.0
    my = (float(face[10]) + float(face[12])) / 2.0
    side = max(24.0, float(np.hypot(float(face[9])-float(face[11]), float(face[10])-float(face[12])) * 2.4))
    x0 = max(0, int(mx - side)); y0 = max(0, int(my - side * 0.65))
    x1 = min(frame.shape[1], int(mx + side)); y1 = min(frame.shape[0], int(my + side * 0.65))
    if x1 <= x0 or y1 <= y0:
        return None
    patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(patch, (64, 40), interpolation=cv2.INTER_AREA)


def sample_video(video: Path, count: int = 24):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if total < count:
        count = max(8, total)
    indices = np.linspace(0, max(0, total - 1), count).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((idx / fps, frame))
    cap.release()
    if len(frames) < 8:
        raise RuntimeError("VIDEO_QA_TOO_FEW_SAMPLE_FRAMES")
    return frames


def run(video: Path, reference: Path, yunet_path: Path, sface_path: Path) -> dict:
    ref = cv2.imread(str(reference))
    if ref is None:
        raise RuntimeError(f"Cannot read identity reference: {reference}")

    detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.85, 0.30, 20)
    recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
    ref_faces = detect(detector, ref)
    if not ref_faces:
        raise RuntimeError("IDENTITY_REFERENCE_FACE_NOT_FOUND")
    ref_face = max(ref_faces, key=lambda f: float(f[2] * f[3]))
    ref_feature = feature(recognizer, ref, ref_face)

    samples = sample_video(video)
    matched = 0
    multi_match = 0
    areas = []
    mouth_patches = []
    details = []

    for t, frame in samples:
        faces = detect(detector, frame)
        candidates = []
        for f in faces:
            score = float(recognizer.match(ref_feature, feature(recognizer, frame, f), cv2.FaceRecognizerSF_FR_COSINE))
            candidates.append((score, f))
        candidates.sort(key=lambda x: x[0], reverse=True)
        matches = [(s, f) for s, f in candidates if s >= COSINE_THRESHOLD]
        if matches:
            matched += 1
            if len(matches) > 1:
                multi_match += 1
            best = matches[0][1]
            ratio = float(best[2] * best[3]) / float(frame.shape[0] * frame.shape[1])
            areas.append(ratio)
            p = mouth_patch(frame, best)
            if p is not None:
                mouth_patches.append(p)
            details.append({"t": round(t, 2), "faces": len(faces), "matched": True, "score": round(matches[0][0], 4), "face_area": round(ratio, 4)})
        else:
            details.append({"t": round(t, 2), "faces": len(faces), "matched": False})

    match_ratio = matched / len(samples)
    area_ok = bool(areas) and float(np.median(areas)) >= MIN_FACE_RATIO and float(np.median(areas)) <= MAX_FACE_RATIO

    motion_scores = []
    for a, b in zip(mouth_patches, mouth_patches[1:]):
        motion_scores.append(float(np.mean(cv2.absdiff(a, b))))
    motion_threshold = 3.0
    motion_ratio = (sum(x >= motion_threshold for x in motion_scores) / len(motion_scores)) if motion_scores else 0.0

    result = {
        "video": str(video),
        "reference": str(reference),
        "samples": len(samples),
        "identity_match_ratio": round(match_ratio, 3),
        "identity_cosine_threshold": COSINE_THRESHOLD,
        "duplicate_matched_face_samples": multi_match,
        "median_face_area_ratio": round(float(np.median(areas)), 4) if areas else 0.0,
        "mouth_motion_ratio": round(motion_ratio, 3),
        "mouth_motion_threshold": motion_threshold,
        "checks": {
            "reference_face_present": True,
            "same_person_present": match_ratio >= MIN_MATCH_RATIO,
            "face_size_valid": area_ok,
            "visible_mouth_motion": motion_ratio >= MIN_MOTION_RATIO,
            "no_duplicate_reference_identity": multi_match == 0,
        },
        "sample_details": details,
    }
    result["PASS"] = all(result["checks"].values())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["PASS"]:
        raise RuntimeError("VISUAL_QUALITY_GATE_FAILED")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--yunet", type=Path, default=Path("qa_models/face_detection_yunet_2023mar.onnx"))
    ap.add_argument("--sface", type=Path, default=Path("qa_models/face_recognition_sface_2021dec.onnx"))
    ap.add_argument("--report", type=Path, default=Path("output/visual_qa.json"))
    args = ap.parse_args()
    result = run(args.video, args.reference, args.yunet, args.sface)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VISUAL_QUALITY_GATE=PASS", flush=True)


if __name__ == "__main__":
    main()
