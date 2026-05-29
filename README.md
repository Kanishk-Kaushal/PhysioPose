# PhysioPose

AI-Based Rehabilitation Exercise Monitoring using Webcam

---

# Overview

PhysioPose is an AI-powered rehabilitation exercise monitoring system that uses a standard webcam to analyze human body posture and movement in real time.

The system helps users perform physiotherapy exercises correctly at home by:
- Detecting body landmarks
- Tracking exercise motion
- Counting repetitions
- Evaluating posture correctness
- Providing real-time corrective feedback

The project uses computer vision and machine learning techniques to make rehabilitation more accessible and affordable without requiring wearable sensors or special hardware.

---

# Features

- Real-time webcam pose estimation
- Human skeleton tracking
- Joint angle calculation
- Exercise repetition counting
- Posture correction feedback
- Multiple rehabilitation exercises
- Lightweight webcam-only setup

---

# Tech Stack

## Computer Vision
- OpenCV
- MediaPipe Pose

## Machine Learning
- NumPy
- Scikit-learn

## Language
- Python 3.10

---

# Recommended Environment

- Ubuntu 22.04 (Recommended)
- Windows 10/11
- macOS

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Kanishk-Kaushal/PhysioPose.git
cd PhysioPose
```

---

## 2. Create Virtual Environment

### Linux/macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dependency Test

To verify all dependencies are installed correctly:

```bash
python dependency_test.py
```

If everything is installed correctly, the script should run without errors.

---

# Run the Project (Runs pose estimation for now)

```bash
python main.py
```

Press:
```text
q
```

to quit the webcam window.

---

# Project Structure

```bash
PhysioPose/
│
├── venv/
├── datasets/
├── models/
├── pose_estimation/
├── posture_analysis/
├── ui/
├── utils/
├── dependency_test.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Current Progress

- [x] GitHub repository setup
- [x] Python virtual environment setup
- [x] Webcam pipeline
- [x] Pose estimation
- [ ] Joint angle calculation
- [ ] Squat detection
- [ ] Rep counting
- [ ] Posture feedback system
- [ ] Machine learning integration

---

# Future Improvements

- LSTM-based motion analysis
- Personalized rehabilitation plans
- Mobile application
- Cloud-based monitoring
- Physiotherapist dashboard
- Voice feedback assistance

---

# Authors

- Kanishk Kaushal
- Kaushal Jha

---

# References

- MediaPipe Pose — Google
- OpenPose — CMU
- Artificial Intelligence and Machine Learning in Rehabilitation (npj Digital Medicine, 2020)
