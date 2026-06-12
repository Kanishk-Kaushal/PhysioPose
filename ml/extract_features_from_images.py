"""
Build a labeled feature CSV from the Zenodo side-view squat image dataset
(https://zenodo.org/records/17558630) by running each image through the SAME
MediaPipe pipeline used at inference time.

This guarantees the training features and the live webcam features are produced
by identical code, eliminating the train/live mismatch that plagued the
synthetic dataset.

Expected input layout (one folder per class; folder names auto-detected):
    datasets/squat_pose/zenodo/Dataset/<ClassName>/*.jpg

Output:
    datasets/squat_pose/squat_features_real.csv
"""

import os
import glob

import cv2
import pandas as pd

from pose_estimation.pose_detector import PoseDetector
from utils.angle_utils import extract_squat_features_sideview

IMAGE_ROOT = "datasets/squat_pose/zenodo"
OUT_PATH = "datasets/squat_pose/squat_features_real.csv"

# Map the dataset's folder names to integer labels. Filled in after we see the
# actual folder names; keys are matched case-insensitively on a substring basis.
CLASS_KEYWORDS = {
    "good": ("Good form", 0),
    "back": ("Bad back", 1),
    "heel": ("Bad heel", 2),
}

# Side view occludes the far-side limbs, so MediaPipe reports lower visibility
# for them. Relaxed from the live default (0.6) so we don't drop most images.
MIN_VISIBILITY = 0.3

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def classify_folder(name):
    low = name.lower()
    for kw, (desc, label) in CLASS_KEYWORDS.items():
        if kw in low:
            return desc, label
    return None, None


def find_class_dirs(root):
    """Return [(dir_path, description, label, split)] for each class folder."""
    found = []
    for dirpath, _, files in os.walk(root):
        if not any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
            continue
        desc, label = classify_folder(os.path.basename(dirpath))
        if label is None:
            continue
        low = dirpath.lower()
        split = "test" if os.sep + "test" in low else "train"
        found.append((dirpath, desc, label, split))
    return found


def images_in(dirpath):
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(dirpath, ext)))
    return sorted(set(paths))


def main():
    # static_image_mode=True is required for unrelated still images (no tracking).
    detector = PoseDetector(static_image_mode=True, min_detection_confidence=0.4)

    class_dirs = find_class_dirs(IMAGE_ROOT)
    if not class_dirs:
        raise SystemExit(f"No recognized class folders under {IMAGE_ROOT}. "
                         f"Inspect the unzipped layout and update CLASS_KEYWORDS.")

    print("Detected class folders:")
    for d, desc, lbl, split in class_dirs:
        print(f"  [{lbl}] {split:5s} {desc:10s} <- {d} ({len(images_in(d))} images)")

    rows = []
    for dirpath, desc, label, split in class_dirs:
        imgs = images_in(dirpath)
        ok = 0
        for path in imgs:
            img = cv2.imread(path)
            if img is None:
                continue
            _, results = detector.detect_pose(img, draw=False)
            landmarks = detector.get_landmarks(results, img.shape)
            feats = extract_squat_features_sideview(landmarks, min_visibility=MIN_VISIBILITY)
            if feats is None:
                continue
            feats["image_file"] = os.path.basename(path)
            feats["label"] = label
            feats["split"] = split
            rows.append(feats)
            ok += 1
        print(f"  {split:5s} {desc:10s}: extracted {ok}/{len(imgs)} "
              f"({100*ok/max(len(imgs),1):.0f}% yield)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}: {df.shape}")
    if "label" in df:
        print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
