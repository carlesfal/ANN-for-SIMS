"""Diagnostic and 3D surface plotting functions."""

from __future__ import annotations

import os
from itertools import combinations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from .config import SurfacePlotConfig


# ---------------------------------------------------------------------------
# MSE evolution
# ---------------------------------------------------------------------------

def plot_mse_evolution(
    history: keras.callbacks.History,
    save_path: str,
    in_notebook: bool = False,
) -> None:
    """Plot train/val MSE across epochs."""
    hist = history.history
    loss = hist.get("loss")
    val_loss = hist.get("val_loss")
    if loss is None:
        return

    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Train MSE", marker="o")
    if val_loss is not None:
        plt.plot(epochs, val_loss, label="Val MSE", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (loss)")
    plt.title("Evolution of MSE during final training")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if in_notebook:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Predicted vs Actual
# ---------------------------------------------------------------------------

def plot_predicted_vs_actual(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    title: str,
    save_path: Optional[str] = None,
    in_notebook: bool = False,
) -> None:
    """Scatter plot of predicted vs actual values with identity line."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    plt.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    try:
        sns.regplot(x=y_true, y=y_pred, scatter=False, color="blue", ci=None)
    except Exception:
        pass
    plt.title(title)
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    if in_notebook:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Residual plots
# ---------------------------------------------------------------------------

def _plot_residual_pair(
    residuals: NDArray[np.floating],
    title_prefix: str,
    hist_title_suffix: str,
    hist_xlabel: str,
    hist_color: str,
    save_path: Optional[str] = None,
    in_notebook: bool = False,
) -> None:
    """Internal helper: histogram + time-series of residuals."""
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(residuals, bins=25, kde=True, color=hist_color, edgecolor="black")
    plt.title(f"{title_prefix} — {hist_title_suffix}")
    plt.xlabel(hist_xlabel)
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(residuals, marker="o", linestyle="-")
    plt.title(f"{title_prefix} — {hist_title_suffix} (time series)")
    plt.xlabel("Observation index")
    plt.ylabel(hist_xlabel)
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if in_notebook:
        plt.show()
    else:
        plt.close()


def plot_standardized_residuals(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    title_prefix: str,
    save_path: Optional[str] = None,
    in_notebook: bool = False,
) -> None:
    """Histogram + time-series of standardised residuals."""
    residuals = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    _plot_residual_pair(
        std_res, title_prefix,
        "Standardized residuals distribution",
        "Standardized residuals",
        "gray",
        save_path=save_path,
        in_notebook=in_notebook,
    )


def plot_raw_residuals(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    title_prefix: str,
    save_path: Optional[str] = None,
    in_notebook: bool = False,
) -> None:
    """Histogram + time-series of raw residuals."""
    residuals = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)
    _plot_residual_pair(
        residuals, title_prefix,
        "Raw residuals distribution",
        "Residuals (Actual - Predicted)",
        "salmon",
        save_path=save_path,
        in_notebook=in_notebook,
    )


def generate_diagnostic_plots(
    y_train_inv: NDArray[np.floating],
    y_train_pred: NDArray[np.floating],
    y_val_inv: NDArray[np.floating],
    y_val_pred: NDArray[np.floating],
    y_test_inv: NDArray[np.floating],
    y_test_pred: NDArray[np.floating],
    plots_dir: str,
    in_notebook: bool = False,
) -> None:
    """Generate and save all diagnostic plots for the three splits."""
    os.makedirs(plots_dir, exist_ok=True)

    for tag, y_t, y_p in [
        ("train", y_train_inv, y_train_pred),
        ("val", y_val_inv, y_val_pred),
        ("test", y_test_inv, y_test_pred),
    ]:
        label = tag.capitalize()
        plot_predicted_vs_actual(
            y_t, y_p, f"Predicted vs Actual ({label})",
            save_path=os.path.join(plots_dir, f"pred_vs_actual_{tag}.png"),
            in_notebook=in_notebook,
        )
        plot_standardized_residuals(
            y_t, y_p, label,
            save_path=os.path.join(plots_dir, f"{tag}_std_residuals_hist_ts.png"),
            in_notebook=in_notebook,
        )
        plot_raw_residuals(
            y_t, y_p, label,
            save_path=os.path.join(plots_dir, f"{tag}_raw_residuals_hist_ts.png"),
            in_notebook=in_notebook,
        )
    print(f"Diagnostic plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# 3D surface helpers (shared)
# ---------------------------------------------------------------------------

def _build_reference_vector(
    X_train_orig: NDArray[np.floating],
    hold_mode: str,
    row_index: int = 0,
) -> NDArray[np.floating]:
    """Build the held-constant reference vector in original feature space."""
    if hold_mode == "median_train":
        return np.median(X_train_orig, axis=0)
    if hold_mode == "mean_train":
        return np.mean(X_train_orig, axis=0)
    if hold_mode == "row":
        if row_index < 0 or row_index >= len(X_train_orig):
            raise IndexError(f"row_index out of range: {row_index}")
        return X_train_orig[row_index].copy()
    raise ValueError(f"hold_mode must be 'median_train', 'mean_train', or 'row', got '{hold_mode}'")


def _grid_values(
    X_train_orig: NDArray[np.floating],
    col_idx: int,
    grid_n: int,
    range_mode: str,
    q_low: float,
    q_high: float,
) -> NDArray[np.floating]:
    """Return *grid_n* linearly-spaced values for column *col_idx*."""
    v = X_train_orig[:, col_idx]
    if range_mode == "minmax":
        lo, hi = float(np.min(v)), float(np.max(v))
    elif range_mode == "quantile":
        lo, hi = float(np.quantile(v, q_low)), float(np.quantile(v, q_high))
    else:
        raise ValueError(f"range_mode must be 'quantile' or 'minmax', got '{range_mode}'")
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, grid_n)


def _predict_from_orig(
    X_orig: NDArray[np.floating],
    model: keras.Model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
) -> NDArray[np.floating]:
    """Predict in original (unscaled) space."""
    X_scaled = scaler_X.transform(X_orig)
    y_scaled = model.predict(X_scaled, verbose=0)
    return scaler_y.inverse_transform(y_scaled).reshape(-1)


def _safe_filename(s: str) -> str:
    """Sanitise a string for use in a file name."""
    for ch in " /\\:;|()[]{}%":
        s = s.replace(ch, "_")
    return s


# ---------------------------------------------------------------------------
# 3D surface: all pairs
# ---------------------------------------------------------------------------

def generate_3d_surfaces(
    X_train_orig: NDArray[np.floating],
    input_columns: list[str],
    model: keras.Model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    output_dir: str,
    scfg: SurfacePlotConfig,
    in_notebook: bool = False,
) -> None:
    """Generate 3D predicted-surface plots for feature pairs.

    When ``scfg.features_to_use`` is ``None`` every pair (up to
    ``scfg.max_pairs``) is plotted; otherwise only the specified
    features are combined.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_features = X_train_orig.shape[1]

    # Resolve feature indices
    if scfg.features_to_use is None:
        feat_indices = list(range(n_features))
    else:
        feat_indices = []
        for f in scfg.features_to_use:
            if f not in input_columns:
                raise ValueError(f"Feature '{f}' not in input_columns.")
            feat_indices.append(input_columns.index(f))

    pairs = list(combinations(feat_indices, 2))[: scfg.max_pairs]
    xref = _build_reference_vector(X_train_orig, scfg.hold_mode, scfg.row_index)

    print(f"Generating {len(pairs)} 3D surface(s) into {output_dir}")

    for i, j in pairs:
        xi = _grid_values(X_train_orig, i, scfg.grid_n, scfg.range_mode, scfg.q_low, scfg.q_high)
        xj = _grid_values(X_train_orig, j, scfg.grid_n, scfg.range_mode, scfg.q_low, scfg.q_high)
        XI, XJ = np.meshgrid(xi, xj)

        Xgrid = np.tile(xref.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)

        Z = _predict_from_orig(Xgrid, model, scaler_X, scaler_y).reshape(XI.shape)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        if scfg.overlay_train_scatter:
            z_train = _predict_from_orig(X_train_orig, model, scaler_X, scaler_y)
            ax.scatter(
                X_train_orig[:, i], X_train_orig[:, j], z_train,
                c="k", s=scfg.scatter_size, alpha=scfg.scatter_alpha,
            )

        col_i = str(input_columns[i])
        col_j = str(input_columns[j])
        ax.set_title(f"Predicted surface: {col_i} vs {col_j}\n(others held: {scfg.hold_mode})")
        ax.set_xlabel(col_i)
        ax.set_ylabel(col_j)
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()

        out_png = os.path.join(
            output_dir,
            f"surface_{_safe_filename(col_i)}_vs_{_safe_filename(col_j)}_hold_{scfg.hold_mode}.png",
        )
        plt.savefig(out_png, dpi=scfg.dpi)
        if in_notebook:
            plt.show()
        else:
            plt.close(fig)

    n_saved = len([f for f in os.listdir(output_dir) if f.lower().endswith(".png")])
    print(f"3D surfaces saved. Count: {n_saved}")
