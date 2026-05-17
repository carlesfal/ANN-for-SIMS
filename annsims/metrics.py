"""Metric computation utilities."""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import r2_score


def compute_basic_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> dict[str, Any]:
    """Compute regression metrics for a single split.

    Returns a dict with keys: n, SSE, MSE, RMSE, MAE, SEP, MRPD_percent, R2.
    """
    actual = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    n = len(actual)
    residuals = actual - pred

    sse = float(np.sum(residuals ** 2))
    mse = sse / n if n > 0 else float("nan")
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")
    mae = float(np.mean(np.abs(residuals))) if n > 0 else float("nan")
    sep = float(np.sqrt(sse / (n - 1))) if n > 1 else float("nan")

    mean_abs = float(np.mean(np.abs(actual))) if n > 0 else float("nan")
    mrpd = (
        100.0 * float(np.sum(np.abs(residuals))) / (n * mean_abs)
        if (n > 0 and not np.isclose(mean_abs, 0.0))
        else float("nan")
    )
    r2 = float(r2_score(actual, pred)) if n > 0 else float("nan")

    return {
        "n": n,
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SEP": sep,
        "MRPD_percent": mrpd,
        "R2": r2,
    }


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Compute adjusted R² given *n* observations and *p* predictors."""
    denom = n - p - 1
    if denom <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / denom


def print_metrics(
    metrics_train: dict[str, Any],
    metrics_val: dict[str, Any],
    metrics_test: dict[str, Any],
    train_pct: int,
    val_pct: int,
    test_pct: int,
) -> None:
    """Pretty-print final metrics for all three splits."""
    print("\n=== Final Metrics ===")
    for label, pct, m in [
        ("Training", train_pct, metrics_train),
        ("Validation", val_pct, metrics_val),
        ("Test", test_pct, metrics_test),
    ]:
        print(f"{label} set ({pct}%):")
        for k, v in m.items():
            print(f"  {k}: {v}")


def save_statistics_file(
    path: str,
    *,
    p: int,
    train_pct: int,
    val_pct: int,
    test_pct: int,
    best_hp_values: dict[str, Any],
    metrics_train: dict[str, Any],
    metrics_val: dict[str, Any],
    metrics_test: dict[str, Any],
    pred_r2_train: float,
    best_epoch: int,
    do_optional_retrain: bool,
    pi_calibration: str,
    pi_alpha: float,
) -> None:
    """Write a human-readable statistics summary to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Model statistics summary\n")
        fh.write("========================\n\n")
        fh.write(f"Current date: {_dt.date.today().isoformat()}\n")
        fh.write(f"Number of predictors (p): {p}\n")
        fh.write(f"Data split: {train_pct}% train / {val_pct}% val / {test_pct}% test\n\n")
        fh.write("Hyperparameters (best):\n")
        for k, v in best_hp_values.items():
            fh.write(f"  {k}: {v}\n")
        for label, pct, m in [
            ("Training", train_pct, metrics_train),
            ("Validation", val_pct, metrics_val),
            ("Test", test_pct, metrics_test),
        ]:
            fh.write(f"\n{label} set metrics ({pct}%):\n")
            for k, v in m.items():
                fh.write(f"  {k}: {v}\n")
        fh.write(f"\nPredicted R2 (Q2) on training (OOF, no leakage): {pred_r2_train}\n")
        fh.write(f"Best epoch (VAL loss): {best_epoch}\n")
        fh.write(f"Optional retrain on train+val: {do_optional_retrain}\n")
        fh.write(f"PI calibration source: {pi_calibration}, alpha={pi_alpha}\n")
    print(f"Statistics saved to: {path}")
