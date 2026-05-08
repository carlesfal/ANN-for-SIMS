from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.inference import predict, regression_metrics


class PredictController:
    def __init__(self) -> None:
        self.input_data: pd.DataFrame | None = None
        self.results: pd.DataFrame | None = None

    def set_input_data(self, dataframe: pd.DataFrame) -> None:
        self.input_data = dataframe

    def run_prediction(self, model, feature_columns: list[str], target_column: str | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
        if self.input_data is None:
            raise ValueError("No prediction input loaded")

        features = np.asarray(self.input_data[feature_columns], dtype=float)
        preds = predict(model, features)

        result_df = self.input_data.copy()
        result_df["prediction"] = preds

        metrics: dict[str, float] = {}
        if target_column and target_column in result_df.columns:
            y_true = np.asarray(result_df[target_column], dtype=float)
            metrics = regression_metrics(y_true, np.asarray(result_df["prediction"], dtype=float))

        self.results = result_df
        return result_df, metrics
