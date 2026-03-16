from __future__ import annotations

import joblib

from .config import MODEL_PATH, SUBMISSION_PATH, SUBMISSIONS_DIR, TEST_FEATURES, TEST_SKELETON
from .data_loader import load_feature_table


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train_baseline` first."
        )

    model = joblib.load(MODEL_PATH)
    test_features = load_feature_table(TEST_FEATURES)
    skeleton = load_feature_table(TEST_SKELETON)

    x_test = test_features.drop(columns=["ID"])
    preds = model.predict(x_test)

    submission = skeleton.copy()
    submission["impairment_type"] = preds

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
