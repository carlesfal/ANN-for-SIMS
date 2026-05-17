"""New-data prediction with calibrated empirical prediction intervals."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .config import PipelineConfig
from .data import load_table, upload_or_find_file
from .metrics import compute_basic_metrics
from .utils import detect_colab, show


# ---------------------------------------------------------------------------
# Calibration residuals
# ---------------------------------------------------------------------------

def get_calibration_residuals(
    source: str,
    y_val_inv: NDArray[np.floating],
    y_val_pred: NDArray[np.floating],
    y_oof_true: NDArray[np.floating],
    y_oof_pred: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return the residual vector used for empirical PI calibration."""
    source = source.strip().lower()
    if source == "val":
        return y_val_inv.reshape(-1) - y_val_pred.reshape(-1)
    if source == "oof":
        return y_oof_true.reshape(-1) - y_oof_pred.reshape(-1)
    raise ValueError(f"pi_calibration must be 'val' or 'oof', got '{source}'")


# ---------------------------------------------------------------------------
# New-data prediction
# ---------------------------------------------------------------------------

def predict_new_data(
    model: keras.Model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    cfg: PipelineConfig,
    input_columns: list[str],
    X_train_orig: NDArray[np.floating],
    y_train_orig: NDArray[np.floating],
    q_low: float,
    q_high: float,
    export_dir: str,
) -> list[str]:
    """Load new data, generate predictions with PI, and export results.

    Returns a list of exported file paths.
    """
    # --- Upload / locate new data ---
    print("\nUpload new data for predictions (optional).")
    try:
        if detect_colab():
            file_name = upload_or_find_file()
        else:
            candidates = ["new_data.tsv", "new_data.csv", "new_data.xlsx"]
            found = next((p for p in candidates if os.path.exists(p)), None)
            if found is None:
                print("No new_data.* found. Skipping predictions.")
                return []
            file_name = found
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"No new data: {exc}")
        return []

    new_df = load_table(file_name, sep=cfg.sep)
    print(f"New data loaded. Shape: {new_df.shape}")
    show(new_df.head())

    n_cols = new_df.shape[1]
    new_labels = new_df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()

    if n_cols < cfg.n_labels + cfg.n_inputs:
        raise ValueError(f"Insufficient columns: {n_cols} < {cfg.n_labels + cfg.n_inputs}")
    new_inputs = new_df.iloc[:, cfg.n_labels:cfg.n_labels + cfg.n_inputs]

    if new_inputs.shape[1] != len(input_columns):
        if set(new_inputs.columns).issubset(set(input_columns)):
            new_inputs = new_inputs[input_columns]
        else:
            raise ValueError("Column mismatch between training and new data inputs.")

    # Impute NaN features with TRAIN means
    if new_inputs.isna().sum().sum() > 0:
        print("Imputing NaN features with TRAIN means…")
        imp = SimpleImputer(strategy="mean").fit(X_train_orig)
        new_inputs = pd.DataFrame(imp.transform(new_inputs), columns=new_inputs.columns)

    new_X = new_inputs.values
    new_X_scaled = scaler_X.transform(new_X)

    print(f"Generating predictions for {len(new_X)} samples…")
    y_pred = scaler_y.inverse_transform(model.predict(new_X_scaled, verbose=0)).reshape(-1)
    pi_lower = y_pred + q_low
    pi_upper = y_pred + q_high

    # --- Compact export ---
    compact = pd.DataFrame(index=range(len(new_X)))
    if new_labels.shape[0] == len(new_X):
        compact = pd.concat([compact, new_labels.reset_index(drop=True)], axis=1)
    compact = pd.concat([compact, new_inputs.reset_index(drop=True)], axis=1)
    compact["Predicted_Value"] = y_pred
    compact["PI_Lower_95%"] = pi_lower
    compact["PI_Upper_95%"] = pi_upper
    compact["PI_Width"] = pi_upper - pi_lower
    compact["PI_Calibration_Source"] = cfg.pi_calibration

    # --- If actual values exist ---
    has_actual = False
    if n_cols > cfg.n_labels + cfg.n_inputs:
        try:
            y_actual = new_df.iloc[:, cfg.n_labels + cfg.n_inputs].values.reshape(-1, 1)
            if not np.isnan(y_actual).all():
                if np.isnan(y_actual).any():
                    print("Imputing NaN targets with TRAIN target mean…")
                    imp_y = SimpleImputer(strategy="mean").fit(y_train_orig)
                    y_actual = imp_y.transform(y_actual)
                y_flat = y_actual.reshape(-1)
                has_actual = True

                residual = y_flat - y_pred
                abs_err = np.abs(residual)
                with np.errstate(divide="ignore", invalid="ignore"):
                    abs_pct = np.where(np.abs(y_flat) > 1e-12, 100.0 * abs_err / np.abs(y_flat), np.nan)

                compact["Actual_Value"] = y_flat
                compact["Prediction_Error"] = residual
                compact["Abs_Error"] = abs_err
                compact["Abs_Percent_Error"] = abs_pct

                within_pi = (y_flat >= pi_lower) & (y_flat <= pi_upper)
                compact["Within_95%_PI"] = within_pi.astype(int)

                metrics = compute_basic_metrics(y_flat, y_pred)
                print("\nMetrics on new data:")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
                coverage = within_pi.sum() / len(within_pi) * 100
                print(f"Empirical 95% PI coverage: {within_pi.sum()}/{len(within_pi)} ({coverage:.1f}%)")
        except Exception as exc:
            print(f"Could not extract target: {exc}")

    if not has_actual:
        print("No actual values. Predictions + PI only.")

    # --- Full-row export ---
    full = new_df.copy()
    full["Predicted_Value"] = y_pred
    full["PI_Lower_95%"] = pi_lower
    full["PI_Upper_95%"] = pi_upper
    full["PI_Width"] = pi_upper - pi_lower
    full["PI_Calibration_Source"] = cfg.pi_calibration
    if has_actual:
        full["Actual_Value"] = y_flat  # type: ignore[possibly-undefined]
        full["Prediction_Error"] = residual  # type: ignore[possibly-undefined]
        full["Abs_Error"] = abs_err  # type: ignore[possibly-undefined]
        full["Abs_Percent_Error"] = abs_pct  # type: ignore[possibly-undefined]
        full["Within_95%_PI"] = within_pi.astype(int)  # type: ignore[possibly-undefined]

    # --- Write files ---
    os.makedirs(export_dir, exist_ok=True)
    paths: list[str] = []
    for name, df in [("new_data_predictions", compact), ("new_data_full_with_all_columns", full)]:
        xlsx = os.path.join(export_dir, f"{name}.xlsx")
        csv = os.path.join(export_dir, f"{name}.csv")
        df.to_excel(xlsx, index=False, engine="openpyxl")
        df.to_csv(csv, index=False)
        paths.extend([xlsx, csv])

    print("Predictions exported.")
    show(compact.head(10))
    return paths
