from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "heart.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "heart_clean.csv"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
OUTPUT_FIGURES = ROOT / "outputs" / "figures"

TARGET = "target"
PREDICTORS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# These columns are stored as numeric codes but represent categories.
CATEGORICAL_PREDICTORS = ["cp", "restecg", "slope", "thal"]
BINARY_PREDICTORS = ["sex", "fbs", "exang"]
CONTINUOUS_OR_COUNT_PREDICTORS = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca"
]

RANDOM_STATE = 2026
TEST_SIZE = 0.20
CLASSIFICATION_THRESHOLD = 0.50
