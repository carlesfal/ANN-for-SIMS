from __future__ import annotations

import pandas as pd

from app.utils.preprocessing import normalize_dataframe, split_xy, summary_statistics


class DataController:
    def __init__(self) -> None:
        self.raw_data: pd.DataFrame | None = None
        self.processed_data: pd.DataFrame | None = None
        self.splits: dict[str, pd.DataFrame | pd.Series] | None = None

    def set_data(self, dataframe: pd.DataFrame) -> None:
        self.raw_data = dataframe
        self.processed_data = dataframe
        self.splits = None

    def preprocess(
        self,
        feature_columns: list[str],
        target_column: str,
        method: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | pd.Series], pd.DataFrame]:
        if self.raw_data is None:
            raise ValueError("No data loaded")

        all_columns = list(dict.fromkeys(feature_columns + [target_column]))
        self.processed_data = normalize_dataframe(self.raw_data, all_columns, method)  # type: ignore[arg-type]
        self.splits = split_xy(
            self.processed_data,
            feature_columns=feature_columns,
            target_column=target_column,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        stats = summary_statistics(self.processed_data)
        return self.processed_data, self.splits, stats
