from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"

TRAIN_FEATURES = DATA_DIR / "train_features.csv"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
TEST_FEATURES = DATA_DIR / "test_features.csv"
TEST_SKELETON = DATA_DIR / "student_skeleton.csv"

MODEL_PATH = MODELS_DIR / "baseline_model.joblib"
SUBMISSION_PATH = SUBMISSIONS_DIR / "submission_baseline.csv"
