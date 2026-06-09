import joblib
import pandas as pd


MODEL_PATH = "models/squat_classifier.pkl"


class SquatClassifier:
    def __init__(self, model_path=MODEL_PATH):
        model_bundle = joblib.load(model_path)

        self.model = model_bundle["model"]
        self.feature_columns = model_bundle["feature_columns"]
        self.label_map = model_bundle["label_map"]

    def predict(self, features_dict):
        """
        features_dict example:
        {
            "left_knee_angle": 90,
            "right_knee_angle": 92,
            "left_hip_angle": 80,
            ...
        }
        """

        input_df = pd.DataFrame([features_dict])

        input_df = input_df[self.feature_columns]

        prediction = self.model.predict(input_df)[0]

        probabilities = self.model.predict_proba(input_df)[0]

        confidence = max(probabilities)

        return {
            "label": int(prediction),
            "description": self.label_map[int(prediction)],
            "confidence": float(confidence),
        }