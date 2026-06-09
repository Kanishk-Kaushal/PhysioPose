import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


DATASET_PATH = "datasets/squat_pose/squat_features_augmented.csv"
MODEL_DIR = "models"
MODEL_PATH = "models/squat_classifier.pkl"


LABEL_MAP = {
    0: "Correct squat form",
    1: "Shallow squat",
    2: "Forward lean",
    3: "Knees caving in",
    4: "Heels off ground",
    5: "Asymmetric squat",
}


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    return df


def prepare_data(df):
    drop_columns = ["label"]

    if "video_file" in df.columns:
        drop_columns.append("video_file")

    if "frame" in df.columns:
        drop_columns.append("frame")

    X = df.drop(columns=drop_columns)
    y = df["label"]

    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[
        LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())
    ]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model, feature_columns):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "label_map": LABEL_MAP,
    }

    joblib.dump(model_bundle, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def main():
    df = load_dataset()

    X, y = prepare_data(df)

    feature_columns = X.columns.tolist()

    print("\nFeatures used for training:")
    print(feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, feature_columns)


if __name__ == "__main__":
    main()