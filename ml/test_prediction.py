from predict_squat import SquatClassifier


sample_features = {
    "left_knee_angle": 90,
    "right_knee_angle": 92,
    "left_hip_angle": 80,
    "right_hip_angle": 82,
    "left_ankle_angle": 70,
    "right_ankle_angle": 72,
    "spine_angle": 170,
    "torso_lean": 10,
    "left_knee_lateral": 0.05,
    "right_knee_lateral": 0.04,
    "symmetry_score": 5,
    "hip_depth": 0.45,
}


classifier = SquatClassifier()
result = classifier.predict(sample_features)

print(result)