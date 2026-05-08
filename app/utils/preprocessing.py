from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

NormMethod = Literal["none", "min-max", "z-score"]


def normalize_dataframe(df: pd.DataFrame, columns: list[str], method: NormMethod) -> pd.DataFrame:
    normalized = df.copy()
    if method == "none":
        return normalized

    for column in columns:
        series = normalized[column].astype(float)
        if method == "min-max":
            denominator = (series.max() - series.min()) or 1.0
            normalized[column] = (series - series.min()) / denominator
        elif method == "z-score":
            std = series.std(ddof=0) or 1.0
            normalized[column] = (series - series.mean()) / std
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    return normalized


def split_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int = 42,
) -> dict[str, pd.DataFrame | pd.Series]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    x = df[feature_columns]
    y = df[target_column]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
    )

    temp_total = val_ratio + test_ratio
    test_share_in_temp = 0.5 if temp_total == 0 else test_ratio / temp_total

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_share_in_temp,
        random_state=random_state,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose()
