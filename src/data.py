from __future__ import annotations

import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, PREDICTORS, TARGET

EXPECTED_COLUMNS = PREDICTORS + [TARGET]


def load_raw_data(path=DATA_RAW) -> pd.DataFrame:
    """Load the raw CSV and validate the expected schema."""
    df = pd.read_csv(path)
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")
    return df[EXPECTED_COLUMNS].copy()


def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact audit table for reproducibility and data-quality review."""
    records = [
        ("rows_raw", len(df)),
        ("columns", df.shape[1]),
        ("missing_cells", int(df.isna().sum().sum())),
        ("exact_duplicate_rows", int(df.duplicated().sum())),
        ("target_0", int((df[TARGET] == 0).sum())),
        ("target_1", int((df[TARGET] == 1).sum())),
    ]
    return pd.DataFrame(records, columns=["metric", "value"])


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove exact duplicate observations and validate the binary target.

    Exact duplicates are removed because treating repeated rows as independent
    observations would artificially increase the effective sample size and can
    make inferential uncertainty look smaller than it really is.
    """
    before = len(df)
    clean = df.drop_duplicates().reset_index(drop=True)
    after = len(clean)

    target_levels = sorted(clean[TARGET].dropna().unique().tolist())
    if target_levels != [0, 1]:
        raise ValueError(f"Expected binary target [0, 1], got {target_levels}")

    if clean.isna().any().any():
        raise ValueError("Unexpected missing values remain in the cleaned data.")

    audit = pd.DataFrame(
        {
            "metric": [
                "rows_before_deduplication",
                "rows_after_deduplication",
                "duplicates_removed",
                "missing_cells_after_cleaning",
                "target_0_after_cleaning",
                "target_1_after_cleaning",
            ],
            "value": [
                before,
                after,
                before - after,
                int(clean.isna().sum().sum()),
                int((clean[TARGET] == 0).sum()),
                int((clean[TARGET] == 1).sum()),
            ],
        }
    )
    return clean, audit


def save_clean_data(df: pd.DataFrame, path=DATA_PROCESSED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def descriptive_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.describe().T
    summary["missing"] = df.isna().sum()
    summary["unique"] = df.nunique()
    return summary
