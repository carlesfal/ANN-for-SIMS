"""Result export: DataFrames, Excel/CSV files, and ZIP packaging."""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from .utils import detect_colab


# ---------------------------------------------------------------------------
# DataFrame assembly
# ---------------------------------------------------------------------------

def make_export_df(
    labels_df: Optional[pd.DataFrame],
    inputs_df: pd.DataFrame,
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> pd.DataFrame:
    """Build an export DataFrame with actual, predicted, residual, and error columns."""
    actual = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    residual = actual - pred
    abs_err = np.abs(residual)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_pct = np.where(np.abs(actual) > 1e-12, 100.0 * abs_err / np.abs(actual), np.nan)

    df = pd.DataFrame(index=range(len(actual)))
    if labels_df is not None and labels_df.shape[0] == len(actual):
        df = pd.concat([df, labels_df.reset_index(drop=True)], axis=1)
    df = pd.concat([df, inputs_df.reset_index(drop=True)], axis=1)
    df["Actual_Value"] = actual
    df["Predicted_Value"] = pred
    df["Residual"] = residual
    df["Abs_Error"] = abs_err
    df["Abs_Percent_Error"] = abs_pct
    return df


def _move_pred_cols_to_end(df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = ["Actual_Value", "Predicted_Value", "Residual", "Abs_Error", "Abs_Percent_Error"]
    other = [c for c in df.columns if c not in pred_cols]
    return df[other + [c for c in pred_cols if c in df.columns]]


def make_full_row_export(
    original_df: pd.DataFrame,
    row_indices: NDArray[np.intp],
    y_true_inv: NDArray[np.floating],
    y_pred_inv: NDArray[np.floating],
) -> pd.DataFrame:
    """Create a full-row export with all original columns plus predictions."""
    full = original_df.iloc[row_indices].reset_index(drop=True).copy()
    y_t = np.asarray(y_true_inv).reshape(-1)
    y_p = np.asarray(y_pred_inv).reshape(-1)
    residuals = y_t - y_p
    abs_err = np.abs(residuals)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_pct = np.where(np.abs(y_t) > 1e-12, 100.0 * abs_err / np.abs(y_t), np.nan)

    full["Actual_Value"] = y_t
    full["Predicted_Value"] = y_p
    full["Residual"] = residuals
    full["Abs_Error"] = abs_err
    full["Abs_Percent_Error"] = abs_pct
    return _move_pred_cols_to_end(full)


# ---------------------------------------------------------------------------
# Save to Excel / CSV
# ---------------------------------------------------------------------------

def save_results(
    export_dir: str,
    results_train: pd.DataFrame,
    results_val: pd.DataFrame,
    results_test: pd.DataFrame,
    full_train: Optional[pd.DataFrame] = None,
    full_val: Optional[pd.DataFrame] = None,
    full_test: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Write all result DataFrames to Excel + CSV and return file paths."""
    os.makedirs(export_dir, exist_ok=True)
    files: list[str] = []

    for name, df in [
        ("train_predictions", results_train),
        ("val_predictions", results_val),
        ("test_predictions", results_test),
    ]:
        xlsx = os.path.join(export_dir, f"{name}.xlsx")
        csv = os.path.join(export_dir, f"{name}.csv")
        df.to_excel(xlsx, index=False, engine="openpyxl")
        df.to_csv(csv, index=False)
        files.extend([xlsx, csv])

    for name, df in [
        ("train_full_with_all_columns", full_train),
        ("val_full_with_all_columns", full_val),
        ("test_full_with_all_columns", full_test),
    ]:
        if df is None:
            continue
        xlsx = os.path.join(export_dir, f"{name}.xlsx")
        csv = os.path.join(export_dir, f"{name}.csv")
        df.to_excel(xlsx, index=False, engine="openpyxl")
        df.to_csv(csv, index=False)
        files.extend([xlsx, csv])

    print(f"Results exported to {export_dir}/")
    return files


# ---------------------------------------------------------------------------
# Model / scaler persistence
# ---------------------------------------------------------------------------

def save_model_and_scalers(
    model: keras.Model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    export_dir: str,
) -> list[str]:
    """Persist the trained model and fitted scalers."""
    import joblib

    os.makedirs(export_dir, exist_ok=True)
    files: list[str] = []

    keras_path = os.path.join(export_dir, "final_model.keras")
    model.save(keras_path)
    files.append(keras_path)

    try:
        h5_path = os.path.join(export_dir, "final_model.h5")
        model.save(h5_path)
        files.append(h5_path)
    except Exception as exc:
        print(f"Could not save .h5 (ok to ignore): {exc}")

    sx_path = os.path.join(export_dir, "scaler_X.pkl")
    sy_path = os.path.join(export_dir, "scaler_y.pkl")
    joblib.dump(scaler_X, sx_path)
    joblib.dump(scaler_y, sy_path)
    files.extend([sx_path, sy_path])

    print(f"Model and scalers saved to {export_dir}/")
    return files


# ---------------------------------------------------------------------------
# Model architecture summary
# ---------------------------------------------------------------------------

def save_model_scheme(model: keras.Model, export_dir: str) -> Optional[str]:
    """Print and save a summary of Dense layers."""
    from tensorflow.keras.layers import Dense

    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    if not dense_layers:
        return None

    print("\n=== Final Dense Layers ===")
    lines: list[str] = []
    for i, layer in enumerate(dense_layers, start=1):
        units = layer.units
        act_name = layer.activation.__name__ if layer.activation else "N/A"
        reg = layer.kernel_regularizer
        reg_str = ""
        if reg is not None:
            try:
                reg_str = f", regularizer=L2={reg.l2}" if hasattr(reg, "l2") else f", regularizer={reg}"
            except Exception:
                reg_str = f", regularizer={reg}"
        line = f"Layer {i}: name='{layer.name}', units={units}, activation={act_name}{reg_str}"
        print(line)
        lines.append(line)

    path = os.path.join(export_dir, "model_scheme.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Final Dense Layers (in model order)\n")
        fh.write("=" * 35 + "\n")
        for ln in lines:
            fh.write(ln + "\n")
    print(f"Model scheme saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# ZIP packaging + download
# ---------------------------------------------------------------------------

def create_zip(zip_name: str, files: list[str], base_dir: str = ".") -> Optional[str]:
    """Create a ZIP archive from *files* and return the path (or None on failure)."""
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        print("No files to zip.")
        return None

    if os.path.exists(zip_name):
        os.remove(zip_name)

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in existing:
            zf.write(fp, arcname=os.path.relpath(fp, base_dir))

    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"Created ZIP: {zip_name} ({size_mb:.2f} MB, {len(existing)} files)")
    return zip_name


def download_zip(zip_path: str) -> None:
    """Trigger a download in Colab or copy to ~/Downloads locally."""
    if detect_colab():
        print(f"\nIn Colab: find '{zip_path}' in the Files sidebar and download it.")
    else:
        dl_dir = os.path.expanduser("~/Downloads")
        os.makedirs(dl_dir, exist_ok=True)
        dest = os.path.join(dl_dir, os.path.basename(zip_path))
        shutil.copy2(zip_path, dest)
        print(f"Saved to {dest}")
