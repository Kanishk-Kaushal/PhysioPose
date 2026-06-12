"""
Record labeled squat data from YOUR webcam, so the classifier trains on the same
camera/geometry it runs on (fixing the domain gap from the Zenodo photos).

Usage (run once per class, doing several reps each time):
    PYTHONPATH=. ./venv/bin/python ml/record_squat_data.py good
    PYTHONPATH=. ./venv/bin/python ml/record_squat_data.py back
    PYTHONPATH=. ./venv/bin/python ml/record_squat_data.py heel

Stand SIDE-ON. Frames are only recorded while you're actually in a squat
(knee bent past ~150 deg) so standing/transition frames don't pollute the data.
Press Ctrl+C (or 'q' in the window) to stop; rows are appended to
datasets/squat_pose/my_squats.csv. You can run a class multiple times -- each
run is tagged as its own session so the trainer can split without leakage.
"""

import sys
import time
import os

import cv2
import pandas as pd

from pose_estimation.pose_detector import PoseDetector
from utils.angle_utils import extract_squat_features_sideview

OUT_CSV = "datasets/squat_pose/my_squats.csv"
SQUAT_KNEE_MAX = 150  # only record frames where the knee is bent (a real squat)

ALIASES = {
    "good": 0, "0": 0,
    "back": 1, "bad_back": 1, "badback": 1, "1": 1,
    "heel": 2, "bad_heel": 2, "badheel": 2, "2": 2,
}
NAMES = {0: "Good form", 1: "Bad back", 2: "Bad heel"}


def resolve_label(arg):
    key = arg.strip().lower()
    if key not in ALIASES:
        raise SystemExit(f"Unknown class '{arg}'. Use one of: good | back | heel")
    return ALIASES[key]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: record_squat_data.py <good|back|heel>")

    label = resolve_label(sys.argv[1])
    session = str(int(time.time()))  # unique id per recording run

    detector = PoseDetector()
    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    rows = []
    print(f"Recording class '{NAMES[label]}' (session {session}).")
    print("Stand side-on and do reps. Recording only while squatting. Ctrl+C/'q' to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, results = detector.detect_pose(frame, draw=True)
            landmarks = detector.get_landmarks(results, frame.shape)
            feats = extract_squat_features_sideview(landmarks, min_visibility=0.2)

            recording = feats is not None and feats["knee_angle"] < SQUAT_KNEE_MAX
            if recording:
                feats["label"] = label
                feats["session"] = session
                rows.append(feats)

            color = (0, 0, 255) if recording else (0, 200, 200)
            status = f"REC {NAMES[label]}  frames:{len(rows)}" if recording else "stand by (squat to record)"
            cv2.putText(frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("record", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not rows:
        print("No squat frames captured (did you bend your knees past ~150 deg?).")
        return

    df = pd.DataFrame(rows)
    header = not os.path.exists(OUT_CSV)
    df.to_csv(OUT_CSV, mode="a", header=header, index=False)
    print(f"Appended {len(df)} frames for '{NAMES[label]}' to {OUT_CSV}")

    if os.path.exists(OUT_CSV):
        total = pd.read_csv(OUT_CSV)
        print("Totals so far:",
              {NAMES[k]: int(v) for k, v in total["label"].value_counts().items()})


if __name__ == "__main__":
    main()
