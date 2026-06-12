# PhysioPose

**Real-time squat form classification from a standard webcam.**

PhysioPose uses pose estimation and a lightweight machine-learning classifier to
watch you squat and tell you, in real time, whether your form is **good** or
which common fault you're making — using nothing but a webcam. It's aimed at
making at-home rehabilitation and strength training feedback accessible without
wearables or special hardware.

---

## What it does

For each webcam frame, PhysioPose:

1. Detects 33 body landmarks with **MediaPipe Pose**.
2. Computes four resolution-invariant **joint angles** from the camera-facing
   side of the body: `knee_angle`, `hip_angle`, `ankle_angle`, `spine_angle`.
3. Feeds those angles to a **Random Forest** classifier that predicts one of:

   | Label | Meaning |
   |-------|---------|
   | **Good form** | Correct squat |
   | **Bad back**  | Excessive forward/back torso lean |
   | **Bad heel**  | Heels lifting off the ground |

4. Overlays the predicted label and confidence on the live video.

> **Stand side-on to the camera.** The model is trained on a side ("sagittal")
> view, which is what lets it judge torso lean and heel lift. A front-facing
> squat will not classify correctly.

---

## How it works

```
webcam frame
   │
   ▼
PoseDetector (MediaPipe Pose)        pose_estimation/pose_detector.py
   │  33 landmarks (x, y, visibility)
   ▼
extract_squat_features_sideview()    utils/angle_utils.py
   │  knee / hip / ankle / spine angles (visible-side leg)
   ▼
SquatClassifier (Random Forest)      ml/predict_squat.py  +  models/squat_classifier.pkl
   │  label + confidence
   ▼
overlay on video                     main.py
```

Angle features are used instead of raw pixel coordinates so the model is
independent of camera resolution and the subject's distance from the camera.

---

## Tech stack

- **Python 3.10**
- **OpenCV** — webcam capture & video overlay
- **MediaPipe Pose** — body landmark detection
- **scikit-learn** — Random Forest classifier
- **NumPy / pandas** — feature handling

---

## Project structure

```
PhysioPose/
├── main.py                          # Live webcam app (run this)
├── dependency_test.py               # Quick webcam sanity check
├── requirements.txt
│
├── pose_estimation/
│   └── pose_detector.py             # MediaPipe Pose wrapper
│
├── utils/
│   ├── angle_utils.py               # Joint-angle feature extraction
│   └── fps_utils.py                 # FPS counter
│
├── ml/
│   ├── predict_squat.py             # Loads the model, runs predictions
│   ├── train_squat_classifier.py    # Trains the Random Forest
│   ├── record_squat_data.py         # Record labeled squats from your webcam
│   ├── extract_features_from_images.py  # Build features from an image dataset
│   └── debug_live_features.py       # Diagnostic: live features vs training data
│
├── models/
│   └── squat_classifier.pkl         # Trained model (model + features + labels)
│
└── datasets/squat_pose/
    ├── my_squats.csv                # Features recorded from your webcam
    └── squat_features_real.csv      # Features extracted from the image dataset
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/Kanishk-Kaushal/PhysioPose.git
cd PhysioPose
```

### 2. Create a virtual environment (Python 3.10)

```bash
# Linux / macOS
python3.10 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check your webcam works

```bash
python dependency_test.py     # opens the webcam; press 'q' to close
```

---

## Run the live app

With the virtual environment activated:

```bash
python main.py
```

- Stand **side-on** to the camera with your whole body in frame.
- The prediction + confidence appear once you begin squatting.
- Press **`q`** to quit.

---

## Retrain on your own webcam (recommended)

The shipped model was trained on a public image dataset, and webcams differ
enough that accuracy improves a lot when the model is trained on **your** camera.
The `ml/` scripts import project modules, so run them with `PYTHONPATH=.`.

### 1. Record a few sets of each class

Run each command, do several reps side-on, then press **Ctrl+C** (or `q`) to
stop. Frames are only recorded while you're actually squatting. **Record each
class in at least two separate runs** so the trainer can build a real test set.

```bash
PYTHONPATH=. python ml/record_squat_data.py good   # correct form
PYTHONPATH=. python ml/record_squat_data.py back    # intentional forward lean
PYTHONPATH=. python ml/record_squat_data.py heel    # intentional heel lift
```

Recordings are appended to `datasets/squat_pose/my_squats.csv`.

### 2. Train

```bash
PYTHONPATH=. python ml/train_squat_classifier.py
```

The trainer automatically uses `my_squats.csv` if it exists (otherwise it falls
back to the image-derived dataset), prints accuracy + a confusion matrix, and
saves the model to `models/squat_classifier.pkl`. Re-run `main.py` to use it.

---

## Optional: rebuild features from an image dataset

If you have a folder of side-view squat images organized into `Good` / `Bad
back` / `Bad heel` subfolders (e.g. the
[Zenodo squat dataset](https://zenodo.org/records/17558630)), extract features
from them with:

```bash
PYTHONPATH=. python ml/extract_features_from_images.py
```

This writes `datasets/squat_pose/squat_features_real.csv`, which the trainer can
use as a fallback when no personal recordings exist.

---

## Troubleshooting

If predictions look stuck on one label, run the diagnostic — it captures your
live features and compares them to the training distribution:

```bash
PYTHONPATH=. python ml/debug_live_features.py    # Ctrl+C to stop & save a report
```

It saves `live_features.csv` and `live_report.txt` under `datasets/squat_pose/`,
flagging any feature that is out of the training range. The usual fixes are:
standing side-on, getting your whole body in frame, and retraining on your own
webcam recordings (above).

---

## Limitations

- **Side view only** — front-facing squats won't classify correctly.
- **Three faults** — detects good form, back lean, and heel lift; it does not
  cover knee valgus ("knees caving in"), which isn't reliably visible from the
  side.
- **Single person**, full body in frame, reasonable lighting.

---

## Authors

- Kanishk Kaushal
- Kaushal Jha

---

## References

- C. Lugaresi et al., *MediaPipe: A Framework for Building Perception Pipelines*, Google Research, 2019.
- C. Teng, *Squat Posture Image Dataset (Good / Bad Back / Bad Heel)*, Zenodo, 2025 — record [17558630](https://zenodo.org/records/17558630).
