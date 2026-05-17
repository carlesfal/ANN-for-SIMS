"""Data loading, column selection, splitting, and scaling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .utils import detect_colab


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_table(path: str, sep: str = "\t") -> pd.DataFrame:
    """Load a tabular file (Excel or delimited text) with encoding fallback."""
    _, ext = os.path.splitext(path.lower())

    if ext in (".xlsx", ".xls", ".xlsm"):
        print(f"Detected Excel file: {path}")
        return pd.read_excel(path, engine="openpyxl")

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            print(f"Trying read_csv: sep={sep!r} encoding={enc}")
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as exc:
            last_err = exc

    for enc in encodings:
        try:
            print(f"Fallback sniff: sep=None encoding={enc} (python engine)")
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"Failed to read '{path}'. Last error: {last_err}")


def upload_or_find_file(default_name: str = "data.tsv") -> str:
    """Upload a file via Colab or fall back to a local path."""
    if detect_colab():
        from google.colab import files as colab_files  # type: ignore[import-untyped]
        uploaded = colab_files.upload()
        if not uploaded:
            raise RuntimeError("No file uploaded.")
        return list(uploaded.keys())[0]

    if not os.path.exists(default_name):
        raise FileNotFoundError(
            f"Not in Colab and '{default_name}' not found. Upload or change the path."
        )
    return default_name


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

@dataclass
class ColumnSelection:
    """Result of selecting label / input / target columns from a DataFrame."""

    labels_df: pd.DataFrame
    inputs_df: pd.DataFrame
    y_full: NDArray[np.floating]


def select_columns(df: pd.DataFrame, cfg: PipelineConfig) -> ColumnSelection:
    """Partition *df* into label columns, input columns, and target vector."""
    n_cols = df.shape[1]

    if cfg.target_col is not None:
        if cfg.target_col < 0 or cfg.target_col >= n_cols:
            raise IndexError("target_col out of range.")
        labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
        input_cols = list(range(cfg.n_labels, n_cols))
        input_cols.remove(cfg.target_col)
        inputs_df = df.iloc[:, input_cols]
        y_full = df.iloc[:, cfg.target_col].values.reshape(-1, 1)
    elif cfg.n_labels + cfg.n_inputs >= n_cols:
        print(
            "n_labels + n_inputs >= total columns. "
            "Using last column as target and middle columns as inputs."
        )
        labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, cfg.n_labels:-1]
        y_full = df.iloc[:, -1].values.reshape(-1, 1)
    else:
        labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, cfg.n_labels:cfg.n_labels + cfg.n_inputs]
        y_full = df.iloc[:, cfg.n_labels + cfg.n_inputs].values.reshape(-1, 1)

    print(f"Labels shape: {labels_df.shape}, Inputs shape: {inputs_df.shape}, y shape: {y_full.shape}")
    return ColumnSelection(labels_df=labels_df, inputs_df=inputs_df, y_full=y_full)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

@dataclass
class SplitData:
    """All arrays produced by the train / val / test split."""

    # Original (unscaled) arrays
    X_train_orig: NDArray[np.floating]
    X_val_orig: NDArray[np.floating]
    X_test_orig: NDArray[np.floating]
    y_train_orig: NDArray[np.floating]
    y_val_orig: NDArray[np.floating]
    y_test_orig: NDArray[np.floating]

    # Label DataFrames (for export)
    labels_train: pd.DataFrame
    labels_val: pd.DataFrame
    labels_test: pd.DataFrame

    # Row indices into the original DataFrame
    idx_train: NDArray[np.intp]
    idx_val: NDArray[np.intp]
    idx_test: NDArray[np.intp]

    # Input column names
    input_columns: list[str]

    # Convenience DataFrames of unscaled inputs
    X_train_df: pd.DataFrame
    X_val_df: pd.DataFrame
    X_test_df: pd.DataFrame


def split_data(
    X_full: NDArray[np.floating],
    y_full: NDArray[np.floating],
    labels_full: pd.DataFrame,
    inputs_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> SplitData:
    """Perform a configurable train / val / test split."""
    row_pos = np.arange(len(X_full))

    X_temp, X_test, y_temp, y_test, lbl_temp, lbl_test, idx_temp, idx_test = (
        train_test_split(
            X_full, y_full, labels_full, row_pos,
            test_size=cfg.test_frac,
            random_state=cfg.random_seed,
        )
    )
    val_ratio = cfg.val_frac / (cfg.train_frac + cfg.val_frac)
    X_train, X_val, y_train, y_val, lbl_train, lbl_val, idx_train, idx_val = (
        train_test_split(
            X_temp, y_temp, lbl_temp, idx_temp,
            test_size=val_ratio,
            random_state=cfg.random_seed,
        )
    )

    cols = list(inputs_df.columns)
    n_total = len(X_full)
    print(f"\nSplit sizes: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(
        f"Split %: train={100*len(X_train)/n_total:.1f}% "
        f"val={100*len(X_val)/n_total:.1f}% "
        f"test={100*len(X_test)/n_total:.1f}%"
    )

    return SplitData(
        X_train_orig=X_train, X_val_orig=X_val, X_test_orig=X_test,
        y_train_orig=y_train, y_val_orig=y_val, y_test_orig=y_test,
        labels_train=lbl_train, labels_val=lbl_val, labels_test=lbl_test,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        input_columns=cols,
        X_train_df=pd.DataFrame(X_train, columns=cols).reset_index(drop=True),
        X_val_df=pd.DataFrame(X_val, columns=cols).reset_index(drop=True),
        X_test_df=pd.DataFrame(X_test, columns=cols).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

@dataclass
class ScaledData:
    """Scaled arrays together with the fitted scalers."""

    X_train: NDArray[np.floating]
    X_val: NDArray[np.floating]
    X_test: NDArray[np.floating]
    y_train: NDArray[np.floating]
    y_val: NDArray[np.floating]
    y_test: NDArray[np.floating]
    scaler_X: StandardScaler
    scaler_y: StandardScaler


def fit_scalers(split: SplitData) -> ScaledData:
    """Fit scalers on *train* only (no leakage) and transform all splits."""
    scaler_X = StandardScaler().fit(split.X_train_orig)
    scaler_y = StandardScaler().fit(split.y_train_orig)

    return ScaledData(
        X_train=scaler_X.transform(split.X_train_orig),
        X_val=scaler_X.transform(split.X_val_orig),
        X_test=scaler_X.transform(split.X_test_orig),
        y_train=scaler_y.transform(split.y_train_orig),
        y_val=scaler_y.transform(split.y_val_orig),
        y_test=scaler_y.transform(split.y_test_orig),
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )
