from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def split_features_and_target(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    id_col: str = "ID",
    target_col: str = "impairment_type",
) -> Tuple[pd.DataFrame, pd.Series]:
    merged = features.merge(labels[[id_col, target_col]], on=id_col, how="inner")
    x = merged.drop(columns=[id_col, target_col])
    y = merged[target_col]
    return x, y
