from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .config import MODEL_PATH, MODELS_DIR, TRAIN_FEATURES, TRAIN_LABELS
from .data_loader import load_feature_table, split_features_and_target


def main() -> None:
    features = load_feature_table(TRAIN_FEATURES)
    labels = load_feature_table(TRAIN_LABELS)
    x, y = split_features_and_target(features, labels)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_valid)
    score = f1_score(y_valid, preds, average="macro")
    print(f"Validation Macro-F1: {score:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
