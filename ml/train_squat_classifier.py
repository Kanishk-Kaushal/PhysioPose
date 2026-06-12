import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Train on data recorded from YOUR webcam if it exists (matches the live camera
# geometry); otherwise fall back to the Zenodo side-view photo dataset.
OWN_DATA_PATH = "datasets/squat_pose/my_squats.csv"
ZENODO_PATH = "datasets/squat_pose/squat_features_real.csv"
DATASET_PATH = OWN_DATA_PATH if os.path.exists(OWN_DATA_PATH) else ZENODO_PATH
MODEL_DIR = "models"
MODEL_PATH = "models/squat_classifier.pkl"


LABEL_MAP = {
    0: "Good form",
    1: "Bad back",
    2: "Bad heel",
}


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    source = "YOUR webcam recordings" if DATASET_PATH == OWN_DATA_PATH else "Zenodo photos"
    print(f"Dataset loaded from {DATASET_PATH} ({source})")
    print("Shape:", df.shape)
    return df


def split_data(df):
    non_features = {"label", "split", "session", "image_file", "video_file", "frame"}

    # Legacy pixel-based features (resolution-dependent) are excluded if present.
    non_features |= {"hip_depth", "left_knee_lateral", "right_knee_lateral"}

    feature_columns = [c for c in df.columns if c not in non_features]

    if "split" in df.columns:
        # Zenodo data ships an official train/test split.
        train, test = df[df["split"] == "train"], df[df["split"] == "test"]
        train_idx, test_idx = train.index, test.index
    else:
        # Own recordings: split by session, per class, so (a) near-identical
        # consecutive frames from one run never straddle train/test, and (b)
        # every class still appears in both sets. A class recorded in only one
        # session can't be split, so it goes entirely to train (with a warning).
        train_idx, test_idx = [], []
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for lbl, grp in df.groupby("label"):
            sessions = grp["session"] if "session" in grp.columns else grp.index
            if sessions.nunique() < 2:
                print(f"  WARN: class {lbl} has 1 session; all of it goes to train. "
                      f"Record it in 2+ separate runs for a real test score.")
                train_idx.extend(grp.index)
                continue
            tr, te = next(splitter.split(grp, grp["label"], sessions))
            train_idx.extend(grp.index[tr])
            test_idx.extend(grp.index[te])
        train_idx, test_idx = pd.Index(train_idx), pd.Index(test_idx)

    X = df[feature_columns]
    y = df["label"]
    return (X.loc[train_idx], X.loc[test_idx],
            y.loc[train_idx], y.loc[test_idx], feature_columns)


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

    labels = sorted(LABEL_MAP.keys())
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, labels=labels,
        target_names=[LABEL_MAP[i] for i in labels],
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))


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
    X_train, X_test, y_train, y_test, feature_columns = split_data(df)

    print("\nFeatures used for training:")
    print(feature_columns)
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, feature_columns)


if __name__ == "__main__":
    main()
