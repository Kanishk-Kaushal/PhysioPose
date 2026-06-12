"""
Diagnostic: capture live webcam features and compare them to the training data.

Run it, do a few squats side-on, then press Ctrl+C in the terminal to stop.
It saves two files:
  - datasets/squat_pose/live_features.csv   (every captured frame's features)
  - datasets/squat_pose/live_report.txt     (live range vs training mean per class)

so we can see WHICH feature is out of distribution and pushing every prediction
to one label.
"""

import cv2
import pandas as pd

from pose_estimation.pose_detector import PoseDetector
from utils.angle_utils import extract_squat_features_sideview
from ml.predict_squat import SquatClassifier

TRAIN_CSV = "datasets/squat_pose/squat_features_real.csv"
LIVE_CSV = "datasets/squat_pose/live_features.csv"
REPORT_TXT = "datasets/squat_pose/live_report.txt"


def save_results(captured, preds, cols):
    if not captured:
        print("No frames with a full-body pose were captured; nothing to save.")
        return

    live = pd.DataFrame(captured)
    live["prediction"] = preds
    live.to_csv(LIVE_CSV, index=False)

    train = pd.read_csv(TRAIN_CSV)
    lines = []
    lines.append(f"Captured {len(live)} frames.")
    lines.append(f"Prediction spread: {pd.Series(preds).value_counts().to_dict()}")
    lines.append("")
    lines.append("{:<20}{:>16}{:>10}{:>10}{:>10}".format(
        "feature", "LIVE range", "Good", "BadBack", "BadHeel"))
    for c in cols:
        lo, hi = live[c].min(), live[c].max()
        g = train[train.label == 0][c].mean()
        b = train[train.label == 1][c].mean()
        h = train[train.label == 2][c].mean()
        flag = ""
        if hi < min(g, b, h) - 5 or lo > max(g, b, h) + 5:
            flag = "  <-- OUT OF RANGE"
        lines.append("{:<20}{:>16}{:>10.1f}{:>10.1f}{:>10.1f}{}".format(
            c, f"{lo:.0f}..{hi:.0f}", g, b, h, flag))

    report = "\n".join(lines)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nSaved {LIVE_CSV} and {REPORT_TXT}")


def main():
    detector = PoseDetector()
    clf = SquatClassifier()
    cols = clf.feature_columns

    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam.")

    captured = []
    preds = []
    print("Recording... do squats side-on. Press Ctrl+C in the terminal to stop and save.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, results = detector.detect_pose(frame, draw=True)
            landmarks = detector.get_landmarks(results, frame.shape)
            feats = extract_squat_features_sideview(landmarks, min_visibility=0.2)
            if feats:
                captured.append(feats)
                p = clf.predict(feats)
                preds.append(p["description"])
                cv2.putText(frame, f"{p['description']} {p['confidence']*100:.0f}%",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("debug", frame)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_results(captured, preds, cols)


if __name__ == "__main__":
    main()
