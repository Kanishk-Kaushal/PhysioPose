# PhysioPose

AI-Based Rehabilitation Exercise Monitoring using Webcam

---

## Overview

PhysioPose is a computer vision and AI-based system that monitors rehabilitation exercises through a webcam and provides real-time posture correction feedback. The system helps users perform physiotherapy exercises accurately at home by analyzing body movements and detecting incorrect form.

The project uses pose estimation techniques to extract body landmarks and applies machine learning methods to classify exercises and evaluate movement quality.

---

## Features

- Real-time webcam-based monitoring
- Human pose estimation using body keypoints
- Exercise classification
- Joint angle analysis
- Incorrect posture detection
- Repetition counting
- Real-time corrective feedback
- Lightweight and accessible system
- No wearable sensors required

---

## Proposed Pipeline

1. Capture webcam video
2. Detect body landmarks using pose estimation
3. Track movement and joint angles
4. Classify rehabilitation exercise
5. Detect posture errors or incomplete movement
6. Display live feedback and repetition count

---

## Technologies Used

### Computer Vision
- OpenCV
- MediaPipe Pose

### Machine Learning
- NumPy
- Scikit-learn
- LSTM / Sequence Models

### Programming Language
- Python

### Visualization
- Matplotlib

---

## Example Exercises

- Squats
- Arm Raises
- Knee Rehabilitation
- Shoulder Mobility Exercises
- Leg Raises

---

## System Architecture

Webcam Input  
↓  
Pose Estimation  
↓  
Landmark Extraction  
↓  
Exercise Classification  
↓  
Posture Analysis  
↓  
Real-Time Feedback

---

## Future Improvements

- Personalized rehabilitation plans
- Mobile application support
- Cloud-based patient monitoring
- Physiotherapist dashboard
- Exercise quality scoring system
- Multi-exercise support
- Voice feedback assistance

---

## Installation

```bash
git clone https://github.com/yourusername/PhysioPose.git
cd PhysioPose
pip install -r requirements.txt
```

---

## Run the Project

```bash
python main.py
```

---

## Folder Structure

```bash
PhysioPose/
│
├── data/
├── models/
├── pose_estimation/
├── exercise_classifier/
├── posture_analysis/
├── utils/
├── main.py
├── requirements.txt
└── README.md
```

---

## Applications

- Home physiotherapy
- Sports rehabilitation
- Elderly mobility assistance
- Post-surgery recovery monitoring
- AI-assisted healthcare systems

---

## Motivation

Many rehabilitation patients perform exercises incorrectly while practicing at home without supervision. Incorrect posture can slow recovery or lead to further injury. PhysioPose aims to provide an affordable and accessible AI-powered rehabilitation assistant using only a webcam.

---

## Authors

- Kanishk Kaushal
- Kaushal Jha

---

## References

- MediaPipe Pose — Google
- OpenPose — CMU
- Artificial Intelligence and Machine Learning in Rehabilitation, npj Digital Medicine, 2020
