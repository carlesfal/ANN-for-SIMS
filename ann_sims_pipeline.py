"""
ANN-for-SIMS: Unified Pipeline
===============================

Single pipeline that integrates Hill pre-training with the full ANN
regression workflow for SIMS data.

Phases
------
  0. Environment setup (seeds, GPU, mixed precision)
  1. (Optional) Hill pre-training — synthetic Hill-function data to learn
     sigmoidal priors, producing a ``warmup_model`` for weight transfer.
  2. Data loading & column selection
  3. Train / Val / Test split + scaling (fit on TRAIN only — no leakage)
  4. Keras-Tuner hyperparameter search on real data
  5. Weight transfer from Hill model (if Phase 1 ran)
  6. Strict no-leakage K-fold CV inside TRAIN with fold-fitted scalers;
     optionally retains fold models for an **ensemble**.
  7. Final single-model training (TRAIN → validate on VAL)
  8. Optional retrain on TRAIN+VAL for ``best_epoch`` epochs
  9. **Ensemble evaluation** (CV-fold ensemble on TEST)
 10. Metrics, diagnostic plots, 3-D surface plots
 11. Export results (Excel + CSV) and download zips
 12. New-data prediction with empirical prediction intervals

Usage — Colab / Jupyter
-----------------------
    exec(open("ann_sims_pipeline.py").read())        # defaults
    cfg = PipelineConfig(do_hill_pretrain=True)       # with Hill warmup
    main(cfg=cfg)                                     # custom config

Usage — CLI
------------
    python ann_sims_pipeline.py                        # defaults
    python ann_sims_pipeline.py --hill                 # enable Hill pre-training
    python ann_sims_pipeline.py --trials 30 --hill     # more tuner trials + Hill
    python ann_sims_pipeline.py --mixed-precision      # enable FP16 training
"""

from __future__ import annotations

# ── Detect environment ───────────────────────────────────────────────────────
try:
    get_ipython  # type: ignore[name-defined]  # noqa: F821
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False

use_colab = False
try:
    from google.colab import files as colab_files  # type: ignore[import-untyped]

    use_colab = True
except Exception:
    pass

# ── Install dependencies ─────────────────────────────────────────────────────
if IN_NOTEBOOK:
    print("Installing dependencies…")
    get_ipython().run_line_magic(  # type: ignore[name-defined]  # noqa: F821
        "pip", "install -q keras-tuner seaborn openpyxl joblib"
    )
else:
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "keras-tuner",
            "seaborn",
            "openpyxl",
            "joblib",
        ]
    )

# ── Imports ──────────────────────────────────────────────────────────────────
import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, List, Optional, Tuple  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend (overridden below for notebooks)
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import callbacks, layers, regularizers  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

try:
    import keras_tuner as kt
except ImportError:
    try:
        import kerastuner as kt  # type: ignore[import-untyped]
    except ImportError:
        subprocess.check_call(  # type: ignore[name-defined]
            [sys.executable, "-m", "pip", "install", "keras-tuner"]  # type: ignore[name-defined]
        )
        import keras_tuner as kt  # type: ignore[import-untyped]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if IN_NOTEBOOK:
    matplotlib.use("module://matplotlib_inline.backend_inline")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------


def _show(obj: Any, n: int = 5) -> None:
    if IN_NOTEBOOK:
        try:
            display(obj)  # type: ignore[name-defined]  # noqa: F821
        except Exception:
            print(obj)
    else:
        print(obj.head(n) if hasattr(obj, "head") else obj)


# =============================================================================
#  UNIFIED CONFIGURATION
# =============================================================================
@dataclass
class PipelineConfig:
    """Central configuration for the entire ANN-for-SIMS pipeline.

    Includes settings for data loading, splitting, tuning, CV, training,
    ensemble, Hill pre-training, plots, and exports.
    """

    # ── Data ──────────────────────────────────────────────────────────────
    train_percent: int = 60
    val_percent: int = 20
    test_percent: int = 20

    n_labels: int = 7
    n_inputs: int = 9
    target_col: Optional[int] = None
    sep: str = "\t"

    # ── Device ────────────────────────────────────────────────────────────
    disable_gpu: bool = True
    mixed_precision: bool = False

    # ── Tuner ─────────────────────────────────────────────────────────────
    tuner_trials: int = 30
    tuner_algorithm: str = "bayesian"  # "bayesian" or "random"
    tuner_epochs: int = 300
    tuner_dir: str = "tuner_results"

    # ── Architecture search space ─────────────────────────────────────────
    num_layers_min: int = 1
    num_layers_max: int = 8
    units_min: int = 32
    units_max: int = 1024
    units_step: int = 32
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    dropout_step: float = 0.05
    l2_choices: List[float] = field(
        default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2]
    )
    lr_choices: List[float] = field(
        default_factory=lambda: [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    )
    activation_choices: List[str] = field(
        default_factory=lambda: ["relu", "elu", "selu", "swish"]
    )
    use_batch_norm: bool = True

    # ── CV ────────────────────────────────────────────────────────────────
    k_folds: int = 15
    cv_epochs: int = 300

    # ── Final training ────────────────────────────────────────────────────
    final_epochs: int = 500
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 8

    do_optional_retrain: bool = True

    # ── Ensemble ──────────────────────────────────────────────────────────
    use_cv_ensemble: bool = True

    # ── Prediction intervals ──────────────────────────────────────────────
    pi_calibration: str = "val"  # "val" or "oof"
    pi_alpha: float = 0.05  # 95 % PI

    # ── Hill pre-training (optional Phase 1) ──────────────────────────────
    do_hill_pretrain: bool = False
    hill_n_synthetic: int = 500
    hill_v_max: float = 200.0
    hill_k: float = 0.10
    hill_n: float = 1.20
    hill_x_min: float = 0.1
    hill_x_max: float = 100.0
    hill_pretrain_epochs: int = 50
    hill_pretrain_batch_size: int = 32
    hill_pretrain_patience: int = 5
    hill_pretrain_lr: float = 1e-3
    hill_tuner_trials: int = 10
    hill_tuner_epochs: int = 50

    # ── Reproducibility ───────────────────────────────────────────────────
    random_seed: int = 42

    # ── Export / plots ────────────────────────────────────────────────────
    export_dir: str = "optimized_model"

    surface_grid_n: int = 35
    surface_range_mode: str = "quantile"
    surface_q_low: float = 0.02
    surface_q_high: float = 0.98
    surface_hold_mode: str = "median_train"
    surface_row_index: int = 0
    surface_overlay_scatter: bool = True
    surface_scatter_alpha: float = 0.25
    surface_scatter_size: float = 8
    surface_max_pairs: int = 12
    surface_dpi: int = 160

    def __post_init__(self) -> None:
        total = self.train_percent + self.val_percent + self.test_percent
        if abs(total - 100) > 0.01:
            raise ValueError(
                f"Split percentages must sum to 100. "
                f"Got {self.train_percent}+{self.val_percent}"
                f"+{self.test_percent}={total}"
            )


# ---------------------------------------------------------------------------
# Reproducibility & GPU helpers
# ---------------------------------------------------------------------------


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_device(cfg: PipelineConfig) -> None:
    if cfg.disable_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            logger.info("GPU disabled. Using CPU.")
        except Exception as exc:
            logger.warning("Could not disable GPU: %s", exc)
    else:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    if cfg.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed-precision (FP16) enabled")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_table(path: str, sep: str = "\t") -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext in (".xlsx", ".xls", ".xlsm"):
        logger.info("Detected Excel file: %s", path)
        return pd.read_excel(path, engine="openpyxl")

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            logger.info("Trying read_csv: sep=%r encoding=%s", sep, enc)
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as exc:
            last_err = exc
    for enc in encodings:
        try:
            logger.info("Fallback sniff: sep=None encoding=%s", enc)
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read '{path}'. Last error: {last_err}")


# ---------------------------------------------------------------------------
# Model builder (single, shared across Hill pre-training & main pipeline)
# ---------------------------------------------------------------------------


def make_model_builder(
    n_feat: int,
    cfg: PipelineConfig,
) -> Any:
    """Return a ``build_model(hp)`` closure for Keras-Tuner.

    The **same** search space is used for the optional Hill pre-training and
    the real-data tuner, so auto-generated Keras layer names match and weight
    transfer is always compatible.
    """

    def build_model(hp: Any) -> keras.Sequential:
        model = keras.Sequential()
        model.add(layers.Input(shape=(n_feat,)))

        n_layers = hp.Int(
            "num_layers", cfg.num_layers_min, cfg.num_layers_max, step=1
        )
        l2_val = hp.Choice("l2_reg", cfg.l2_choices)
        activation = hp.Choice("activation", cfg.activation_choices)
        do_bn = hp.Boolean("batch_norm") if cfg.use_batch_norm else False

        for i in range(n_layers):
            units = hp.Int(
                f"units_{i}",
                cfg.units_min,
                cfg.units_max,
                step=cfg.units_step,
            )
            model.add(
                layers.Dense(
                    units,
                    activation=activation,
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(l2_val),
                )
            )
            if do_bn:
                model.add(layers.BatchNormalization())
            model.add(
                layers.Dropout(
                    hp.Float(
                        f"dropout_{i}",
                        cfg.dropout_min,
                        cfg.dropout_max,
                        step=cfg.dropout_step,
                    )
                )
            )

        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", cfg.lr_choices)
            ),
            loss="mse",
            metrics=["mae"],
        )
        return model

    return build_model


# =============================================================================
#  PHASE 1 — Hill Pre-Training (optional)
# =============================================================================


def generate_hill_data(
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, y) pairs governed by a Hill equation."""
    rng = np.random.default_rng(cfg.random_seed)

    X = rng.uniform(
        cfg.hill_x_min,
        cfg.hill_x_max,
        (cfg.hill_n_synthetic, cfg.n_inputs),
    ).astype(np.float32)

    x1 = X[:, 0]
    y = cfg.hill_v_max * (x1 ** cfg.hill_n) / (
        cfg.hill_k ** cfg.hill_n + x1 ** cfg.hill_n
    )
    y += rng.normal(0, 0.05 * cfg.hill_v_max, cfg.hill_n_synthetic).astype(
        np.float32
    )

    span = cfg.hill_x_max - cfg.hill_x_min
    for i in range(1, min(cfg.n_inputs, 3)):
        y += 0.05 * cfg.hill_v_max * (X[:, i] - cfg.hill_x_min) / span

    return X, y.reshape(-1, 1).astype(np.float32)


def hill_pretrain(
    cfg: PipelineConfig,
) -> Optional[keras.Sequential]:
    """Run Hill pre-training: tune on synthetic data, then pre-train best arch.

    Returns the pre-trained ``warmup_model``, or *None* on failure.
    """
    logger.info("=" * 70)
    logger.info("HILL PRE-TRAINING PHASE")
    logger.info("=" * 70)

    X_raw, y_raw = generate_hill_data(cfg)
    logger.info("Hill data: X=%s  y range [%.2f, %.2f]", X_raw.shape, y_raw.min(), y_raw.max())

    scaler_X = StandardScaler().fit(X_raw)
    scaler_y = StandardScaler().fit(y_raw)
    X_scaled = scaler_X.transform(X_raw)
    y_scaled = scaler_y.transform(y_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=cfg.random_seed
    )

    # --- Tune on Hill data ---
    logger.info("Hill tuner: %d trials, %d epochs", cfg.hill_tuner_trials, cfg.hill_tuner_epochs)
    tf.keras.backend.clear_session()

    build_fn = make_model_builder(cfg.n_inputs, cfg)

    tuner = kt.RandomSearch(
        build_fn,
        objective="val_loss",
        max_trials=cfg.hill_tuner_trials,
        executions_per_trial=1,
        directory=cfg.tuner_dir,
        project_name="hill_pretrain",
        seed=cfg.random_seed,
        overwrite=True,
    )
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.hill_tuner_epochs,
        batch_size=cfg.hill_pretrain_batch_size,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=1,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    logger.info("Hill best HP: %s", best_hp.values)

    # --- Pre-train ---
    tf.keras.backend.clear_session()
    model = make_model_builder(cfg.n_inputs, cfg)(best_hp)
    logger.info(
        "Hill warmup model: %d layers, %s params",
        len(model.layers),
        f"{model.count_params():,}",
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.hill_pretrain_epochs,
        batch_size=cfg.hill_pretrain_batch_size,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.hill_pretrain_patience,
                restore_best_weights=True,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
        ],
        verbose=1,
    )

    # Save warmup artefacts
    save_path = Path(cfg.export_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(save_path / "warmup_model.keras")
    joblib.dump(scaler_X, save_path / "hill_scaler_X.joblib")
    joblib.dump(scaler_y, save_path / "hill_scaler_y.joblib")
    logger.info("Hill warmup model saved to %s", save_path)

    return model


# ---------------------------------------------------------------------------
# Weight transfer helper
# ---------------------------------------------------------------------------


def transfer_weights(
    target_model: keras.Sequential,
    source_model: keras.Sequential,
) -> int:
    """Transfer compatible weights from *source_model* to *target_model*.

    Returns the number of layers transferred.
    """
    n_transferred = 0
    for lf, lw in zip(target_model.layers, source_model.layers):
        if isinstance(lf, layers.InputLayer) or isinstance(lw, layers.InputLayer):
            continue
        if not lf.weights or not lw.weights:
            continue
        try:
            if lf.name.split("_")[0] == lw.name.split("_")[0]:
                lf.set_weights(lw.get_weights())
                n_transferred += 1
                logger.info("  %s → %s", lw.name, lf.name)
        except Exception as exc:
            logger.debug("  Skipping %s: %s", lw.name, exc)
    return n_transferred


# =============================================================================
#  Metrics helpers
# =============================================================================


def compute_basic_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    actual = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    n = len(actual)
    res = actual - pred
    sse = float(np.sum(res ** 2))
    mse = sse / n if n else np.nan
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else np.nan
    mae = float(np.mean(np.abs(res))) if n else np.nan
    sep = float(np.sqrt(sse / (n - 1))) if n > 1 else np.nan
    mean_abs = float(np.mean(np.abs(actual))) if n else np.nan
    mrpd = (
        100 * float(np.sum(np.abs(res))) / (n * mean_abs)
        if (n and not np.isclose(mean_abs, 0))
        else np.nan
    )
    r2 = float(r2_score(actual, pred)) if n else np.nan
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
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else np.nan


# =============================================================================
#  Plot helpers
# =============================================================================


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_t, y_p, alpha=0.6)
    lo, hi = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    try:
        sns.regplot(x=y_t, y=y_p, scatter=False, color="blue", ci=None)
    except Exception:
        pass
    plt.title(title)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
    save_prefix: Optional[str] = None,
    standardize: bool = False,
) -> None:
    res = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)
    if standardize:
        res = (res - np.mean(res)) / (np.std(res) + 1e-12)
    color = "gray" if standardize else "salmon"
    label = "Standardized residuals" if standardize else "Raw residuals"
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(res, bins=25, kde=True, color=color, edgecolor="black")
    plt.title(f"{prefix} — {label} distribution")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(res, marker="o", linestyle="-")
    plt.title(f"{prefix} — {label} (time series)")
    plt.xlabel("Index")
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    suffix = (
        "_std_residuals_hist_ts.png" if standardize else "_raw_residuals_hist_ts.png"
    )
    if save_prefix:
        plt.savefig(save_prefix + suffix, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


# =============================================================================
#  3-D Surface helpers
# =============================================================================


def _build_ref_vector(
    X_orig: np.ndarray, hold_mode: str, row_idx: int = 0
) -> np.ndarray:
    if hold_mode == "median_train":
        return np.median(X_orig, axis=0)
    if hold_mode == "mean_train":
        return np.mean(X_orig, axis=0)
    if hold_mode == "row":
        return X_orig[int(row_idx)].copy()
    raise ValueError(
        f"hold_mode must be 'median_train', 'mean_train', or 'row', got '{hold_mode}'"
    )


def _grid_vals(
    X_orig: np.ndarray,
    col: int,
    grid_n: int,
    mode: str,
    q_lo: float,
    q_hi: float,
) -> np.ndarray:
    v = X_orig[:, col]
    if mode == "minmax":
        lo, hi = float(v.min()), float(v.max())
    elif mode == "quantile":
        lo, hi = float(np.quantile(v, q_lo)), float(np.quantile(v, q_hi))
    else:
        raise ValueError(
            f"range_mode must be 'quantile' or 'minmax', got '{mode}'"
        )
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, grid_n)


def _predict_orig(
    X_orig: np.ndarray,
    mdl: keras.Model,
    sx: StandardScaler,
    sy: StandardScaler,
) -> np.ndarray:
    return sy.inverse_transform(
        mdl.predict(sx.transform(X_orig), verbose=0)
    ).reshape(-1)


def _safe_name(s: str) -> str:
    for ch in " /\\:;|()[]{}%":
        s = s.replace(ch, "_")
    return s


# =============================================================================
#  Export helpers
# =============================================================================


def make_export_df(
    lbl: Optional[pd.DataFrame],
    inp: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    actual = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    res = actual - pred
    abs_e = np.abs(res)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_pct = np.where(
            np.abs(actual) > 1e-12, 100 * abs_e / np.abs(actual), np.nan
        )
    out = pd.DataFrame(index=range(len(actual)))
    if lbl is not None and lbl.shape[0] == len(actual):
        out = pd.concat([out, lbl.reset_index(drop=True)], axis=1)
    out = pd.concat([out, inp.reset_index(drop=True)], axis=1)
    out["Actual_Value"] = actual
    out["Predicted_Value"] = pred
    out["Residual"] = res
    out["Abs_Error"] = abs_e
    out["Abs_Percent_Error"] = abs_pct
    return out


# =============================================================================
#  CLI / notebook argument parsing
# =============================================================================


def parse_args(
    argv: Optional[List[str]] = None,
) -> PipelineConfig:
    if IN_NOTEBOOK:
        return PipelineConfig()

    parser = argparse.ArgumentParser(
        description="ANN-for-SIMS: unified Hill-pretrain + training pipeline"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=30, help="Tuner trials")
    parser.add_argument("--tuner-epochs", type=int, default=300)
    parser.add_argument(
        "--tuner-algorithm",
        choices=["bayesian", "random"],
        default="bayesian",
    )
    parser.add_argument("--k-folds", type=int, default=15)
    parser.add_argument("--cv-epochs", type=int, default=300)
    parser.add_argument("--final-epochs", type=int, default=500)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--export-dir", default="optimized_model")
    parser.add_argument("--tuner-dir", default="tuner_results")
    # Hill options
    parser.add_argument("--hill", action="store_true", help="Enable Hill pre-training")
    parser.add_argument("--hill-synthetic", type=int, default=500)
    parser.add_argument("--hill-pretrain-epochs", type=int, default=50)
    parser.add_argument("--hill-tuner-trials", type=int, default=10)
    # Ensemble
    parser.add_argument(
        "--no-ensemble", action="store_true", help="Disable CV ensemble"
    )
    args = parser.parse_args(argv)

    return PipelineConfig(
        random_seed=args.seed,
        tuner_trials=args.trials,
        tuner_epochs=args.tuner_epochs,
        tuner_algorithm=args.tuner_algorithm,
        tuner_dir=args.tuner_dir,
        k_folds=args.k_folds,
        cv_epochs=args.cv_epochs,
        final_epochs=args.final_epochs,
        mixed_precision=args.mixed_precision,
        disable_gpu=not args.enable_gpu,
        export_dir=args.export_dir,
        do_hill_pretrain=args.hill,
        hill_n_synthetic=args.hill_synthetic,
        hill_pretrain_epochs=args.hill_pretrain_epochs,
        hill_tuner_trials=args.hill_tuner_trials,
        use_cv_ensemble=not args.no_ensemble,
    )


# =============================================================================
#  MAIN PIPELINE
# =============================================================================


def main(
    argv: Optional[List[str]] = None,
    *,
    cfg: Optional[PipelineConfig] = None,
) -> None:
    """Run the full ANN-for-SIMS pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if cfg is None:
        cfg = parse_args(argv)

    print(
        f"Data split: TRAIN={cfg.train_percent}%  "
        f"VAL={cfg.val_percent}%  TEST={cfg.test_percent}%"
    )

    # ── Phase 0: Environment ─────────────────────────────────────────────
    set_global_seeds(cfg.random_seed)
    configure_device(cfg)

    # ── Phase 1: Hill pre-training (optional) ────────────────────────────
    warmup_model: Optional[keras.Sequential] = None
    if cfg.do_hill_pretrain:
        warmup_model = hill_pretrain(cfg)
        if warmup_model is not None:
            logger.info("Hill warmup model ready (%s params)", f"{warmup_model.count_params():,}")
    else:
        logger.info("Hill pre-training: SKIPPED (do_hill_pretrain=False)")

    # ── Phase 2: Data loading ────────────────────────────────────────────
    print("\nUpload your dataset.")
    if use_colab:
        uploaded = colab_files.upload()
        if not uploaded:
            raise RuntimeError("No file uploaded.")
        file_name = list(uploaded.keys())[0]
    else:
        file_name = "data.tsv"
        if not os.path.exists(file_name):
            raise FileNotFoundError("Not in Colab and 'data.tsv' not found.")

    df = load_table(file_name, sep=cfg.sep)
    print(f"Data loaded. Shape: {df.shape}")
    _show(df.head())

    # ── Column selection ─────────────────────────────────────────────────
    n_cols = df.shape[1]

    if cfg.target_col is not None:
        target_idx = cfg.target_col
        if target_idx < 0 or target_idx >= n_cols:
            raise IndexError("target_col out of range.")
        labels_df = df.iloc[:, : cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
        input_cols = list(range(cfg.n_labels, n_cols))
        input_cols.remove(target_idx)
        inputs_df = df.iloc[:, input_cols]
        y_full = df.iloc[:, target_idx].values.reshape(-1, 1)
    elif cfg.n_labels + cfg.n_inputs >= n_cols:
        print("n_labels + n_inputs >= total columns. Using last column as target.")
        labels_df = df.iloc[:, : cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, cfg.n_labels : -1]
        y_full = df.iloc[:, -1].values.reshape(-1, 1)
    else:
        labels_df = df.iloc[:, : cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, cfg.n_labels : cfg.n_labels + cfg.n_inputs]
        y_full = df.iloc[:, cfg.n_labels + cfg.n_inputs].values.reshape(-1, 1)

    print(f"Labels: {labels_df.shape}, Inputs: {inputs_df.shape}, y: {y_full.shape}")

    X_full = inputs_df.values
    y_full_arr = y_full
    labels_full = labels_df
    input_columns = list(inputs_df.columns)
    row_pos = np.arange(len(df))

    # ── Phase 3: Train / Val / Test split ────────────────────────────────
    test_frac = cfg.test_percent / 100.0
    val_frac = cfg.val_percent / 100.0
    train_frac = cfg.train_percent / 100.0

    (
        X_temp,
        X_test_orig,
        y_temp,
        y_test_orig,
        lbl_temp,
        labels_test,
        idx_temp,
        idx_test,
    ) = train_test_split(
        X_full, y_full_arr, labels_full, row_pos,
        test_size=test_frac, random_state=cfg.random_seed,
    )
    val_ratio = val_frac / (train_frac + val_frac)
    (
        X_train_orig,
        X_val_orig,
        y_train_orig,
        y_val_orig,
        labels_train,
        labels_val,
        idx_train,
        idx_val,
    ) = train_test_split(
        X_temp, y_temp, lbl_temp, idx_temp,
        test_size=val_ratio, random_state=cfg.random_seed,
    )

    X_train_df = pd.DataFrame(X_train_orig, columns=input_columns).reset_index(drop=True)
    X_val_df = pd.DataFrame(X_val_orig, columns=input_columns).reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test_orig, columns=input_columns).reset_index(drop=True)

    n_total = len(df)
    print(
        f"\nSplit sizes: train={len(X_train_orig)} "
        f"val={len(X_val_orig)} test={len(X_test_orig)}"
    )
    print(
        f"Split %: train={100*len(X_train_orig)/n_total:.1f}%  "
        f"val={100*len(X_val_orig)/n_total:.1f}%  "
        f"test={100*len(X_test_orig)/n_total:.1f}%"
    )

    # ── Scaling (fit on TRAIN only — no leakage) ─────────────────────────
    scaler_X = StandardScaler().fit(X_train_orig)
    scaler_y = StandardScaler().fit(y_train_orig)

    X_train = scaler_X.transform(X_train_orig)
    X_val = scaler_X.transform(X_val_orig)
    X_test = scaler_X.transform(X_test_orig)
    y_train = scaler_y.transform(y_train_orig)
    y_val = scaler_y.transform(y_val_orig)
    y_test = scaler_y.transform(y_test_orig)

    n_features = X_train.shape[1]

    # ── Phase 4: Keras-Tuner HP search ───────────────────────────────────
    build_fn = make_model_builder(n_features, cfg)

    tuner_cls = (
        kt.BayesianOptimization
        if cfg.tuner_algorithm == "bayesian"
        else kt.RandomSearch
    )
    algo_name = (
        "BayesianOptimization"
        if cfg.tuner_algorithm == "bayesian"
        else "RandomSearch"
    )

    tuner = tuner_cls(
        build_fn,
        objective="val_loss",
        max_trials=cfg.tuner_trials,
        executions_per_trial=1,
        directory=cfg.tuner_dir,
        project_name=f"ann_{cfg.train_percent}_{cfg.val_percent}_{cfg.test_percent}",
    )
    print(
        f"Starting {algo_name} hyperparameter search ({cfg.tuner_trials} trials)…"
    )
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.tuner_epochs,
        batch_size=32,
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters:")
    for k, v in best_hp.values.items():
        print(f"  {k}: {v}")

    # ── Phase 5: Weight transfer ─────────────────────────────────────────
    _pretrained_model: Optional[keras.Sequential] = None

    print("\n" + "=" * 70)
    print("WEIGHT TRANSFER: Checking for pre-trained model…")
    print("=" * 70)

    if warmup_model is not None:
        print("Found pre-trained warmup_model!")
        try:
            model_for_transfer = build_fn(best_hp)
            n_transferred = transfer_weights(model_for_transfer, warmup_model)
            if n_transferred > 0:
                print(f"Transferred {n_transferred} layer(s)!")
                _pretrained_model = model_for_transfer
            else:
                print("No compatible weights (architecture mismatch).")
        except Exception as exc:
            print(f"Weight transfer failed: {exc}")
    else:
        print("No pre-trained model found.")
    print("=" * 70 + "\n")

    # ── Phase 6: Strict no-leakage CV inside TRAIN ───────────────────────
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)

    r2_scores: List[float] = []
    rmse_scores: List[float] = []
    mae_scores: List[float] = []
    y_oof_pred_inv = np.full(len(y_train_orig), np.nan)
    y_oof_true_inv = np.full(len(y_train_orig), np.nan)

    cv_fold_models: List[keras.Sequential] = []
    cv_fold_scalers_X: List[StandardScaler] = []
    cv_fold_scalers_y: List[StandardScaler] = []

    cv_cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
    ]
    if cfg.use_lr_scheduler:
        cv_cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.lr_scheduler_factor,
                patience=cfg.lr_scheduler_patience,
                min_lr=1e-6,
                verbose=1,
            )
        )

    print(
        f"{cfg.k_folds}-fold CV on TRAIN with fold-fitted scalers (no leakage)…"
    )

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig), start=1):
        X_tr, X_va = X_train_orig[tr_idx], X_train_orig[va_idx]
        y_tr, y_va = y_train_orig[tr_idx], y_train_orig[va_idx]

        fold_sx = StandardScaler().fit(X_tr)
        fold_sy = StandardScaler().fit(y_tr)

        X_tr_s = fold_sx.transform(X_tr)
        X_va_s = fold_sx.transform(X_va)
        y_tr_s = fold_sy.transform(y_tr)
        y_va_s = fold_sy.transform(y_va)

        model_fold = build_fn(best_hp)
        if _pretrained_model is not None:
            try:
                model_fold.set_weights(_pretrained_model.get_weights())
            except Exception:
                pass

        model_fold.fit(
            X_tr_s,
            y_tr_s,
            validation_data=(X_va_s, y_va_s),
            epochs=cfg.cv_epochs,
            batch_size=32,
            callbacks=cv_cbs,
            verbose=1,
        )

        va_pred = fold_sy.inverse_transform(
            model_fold.predict(X_va_s, verbose=0)
        ).reshape(-1)
        va_true = y_va.reshape(-1)

        y_oof_pred_inv[va_idx] = va_pred
        y_oof_true_inv[va_idx] = va_true

        if cfg.use_cv_ensemble:
            cv_fold_models.append(model_fold)
            cv_fold_scalers_X.append(fold_sx)
            cv_fold_scalers_y.append(fold_sy)

        r2 = r2_score(va_true, va_pred)
        rmse = float(np.sqrt(mean_squared_error(va_true, va_pred)))
        mae = float(np.mean(np.abs(va_true - va_pred)))
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        print(f"Fold {fold}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    if np.isnan(y_oof_pred_inv).any():
        raise RuntimeError("OOF predictions contain NaNs.")

    print(
        f"\nCV (mean±std):  R²={np.mean(r2_scores):.4f}±{np.std(r2_scores):.4f}  "
        f"RMSE={np.mean(rmse_scores):.4f}±{np.std(rmse_scores):.4f}  "
        f"MAE={np.mean(mae_scores):.4f}±{np.std(mae_scores):.4f}"
    )

    ss_res = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
    ss_tot = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
    pred_R2_train = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    print(f"Predicted R² (Q², OOF/PRESS): {pred_R2_train:.4f}")

    # ── Phase 7: Final training (TRAIN → validate on VAL) ────────────────
    print("\nFinal training: fit on TRAIN, validate on VAL.")

    os.makedirs(cfg.export_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.export_dir, "best_model.keras")

    final_cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]
    if cfg.use_lr_scheduler:
        final_cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.lr_scheduler_factor,
                patience=cfg.lr_scheduler_patience,
                min_lr=1e-6,
                verbose=1,
            )
        )

    tf.keras.backend.clear_session()
    model = build_fn(best_hp)

    if _pretrained_model is not None:
        try:
            model.set_weights(_pretrained_model.get_weights())
            logger.info("Initialized final model with pre-trained weights.")
        except Exception:
            pass

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.final_epochs,
        batch_size=32,
        callbacks=final_cbs,
        verbose=1,
    )

    if os.path.exists(ckpt_path):
        model = keras.models.load_model(ckpt_path, compile=False)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print(f"Best epoch (by VAL loss): {best_epoch}")

    # Evaluate once on TEST
    print("\nEvaluating once on TEST.")
    y_test_pred_eval = scaler_y.inverse_transform(
        model.predict(X_test, verbose=0)
    ).reshape(-1)
    y_test_inv_eval = scaler_y.inverse_transform(y_test).reshape(-1)
    print(
        f"TEST: R²={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}  "
        f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}"
    )

    # ── Phase 9: CV fold ensemble evaluation ─────────────────────────────
    if cfg.use_cv_ensemble and cv_fold_models:
        print("\n" + "=" * 70)
        print(f"ENSEMBLE EVALUATION ({len(cv_fold_models)} CV-fold models)")
        print("=" * 70)

        ensemble_preds_test = np.zeros(len(X_test_orig))
        for _fm, _fsx, _fsy in zip(
            cv_fold_models, cv_fold_scalers_X, cv_fold_scalers_y
        ):
            _pred_i = _fsy.inverse_transform(
                _fm.predict(_fsx.transform(X_test_orig), verbose=0)
            ).reshape(-1)
            ensemble_preds_test += _pred_i
        ensemble_preds_test /= len(cv_fold_models)
        _y_test_true = y_test_orig.reshape(-1)
        print(
            f"TEST (ensemble of {len(cv_fold_models)} folds): "
            f"R²={r2_score(_y_test_true, ensemble_preds_test):.4f}  "
            f"RMSE={np.sqrt(mean_squared_error(_y_test_true, ensemble_preds_test)):.4f}  "
            f"MAE={np.mean(np.abs(_y_test_true - ensemble_preds_test)):.4f}"
        )

        # Ensemble on TRAIN (for comparison)
        ensemble_preds_train = np.zeros(len(X_train_orig))
        for _fm, _fsx, _fsy in zip(
            cv_fold_models, cv_fold_scalers_X, cv_fold_scalers_y
        ):
            _pred_i = _fsy.inverse_transform(
                _fm.predict(_fsx.transform(X_train_orig), verbose=0)
            ).reshape(-1)
            ensemble_preds_train += _pred_i
        ensemble_preds_train /= len(cv_fold_models)
        _y_train_true = y_train_orig.reshape(-1)
        print(
            f"TRAIN (ensemble): "
            f"R²={r2_score(_y_train_true, ensemble_preds_train):.4f}  "
            f"RMSE={np.sqrt(mean_squared_error(_y_train_true, ensemble_preds_train)):.4f}"
        )

        # Ensemble on VAL
        ensemble_preds_val = np.zeros(len(X_val_orig))
        for _fm, _fsx, _fsy in zip(
            cv_fold_models, cv_fold_scalers_X, cv_fold_scalers_y
        ):
            _pred_i = _fsy.inverse_transform(
                _fm.predict(_fsx.transform(X_val_orig), verbose=0)
            ).reshape(-1)
            ensemble_preds_val += _pred_i
        ensemble_preds_val /= len(cv_fold_models)
        _y_val_true = y_val_orig.reshape(-1)
        print(
            f"VAL (ensemble): "
            f"R²={r2_score(_y_val_true, ensemble_preds_val):.4f}  "
            f"RMSE={np.sqrt(mean_squared_error(_y_val_true, ensemble_preds_val)):.4f}"
        )
        print("=" * 70)

    # ── Phase 8: Optional retrain on TRAIN+VAL ───────────────────────────
    if cfg.do_optional_retrain:
        print(
            "\nOptional retrain: refit scalers on TRAIN+VAL, "
            "retrain, evaluate on TEST."
        )
        X_tv = np.vstack([X_train_orig, X_val_orig])
        y_tv = np.vstack([y_train_orig, y_val_orig])

        scaler_X_tv = StandardScaler().fit(X_tv)
        scaler_y_tv = StandardScaler().fit(y_tv)

        tf.keras.backend.clear_session()
        model_rt = build_fn(best_hp)

        if _pretrained_model is not None:
            try:
                model_rt.set_weights(_pretrained_model.get_weights())
            except Exception:
                pass

        model_rt.fit(
            scaler_X_tv.transform(X_tv),
            scaler_y_tv.transform(y_tv),
            epochs=best_epoch,
            batch_size=32,
            verbose=1,
        )

        y_pred_rt = scaler_y_tv.inverse_transform(
            model_rt.predict(scaler_X_tv.transform(X_test_orig), verbose=0)
        ).reshape(-1)
        y_true_rt = y_test_orig.reshape(-1)
        print(
            f"TEST (retrained): R²={r2_score(y_true_rt, y_pred_rt):.4f}  "
            f"RMSE={np.sqrt(mean_squared_error(y_true_rt, y_pred_rt)):.4f}  "
            f"MAE={np.mean(np.abs(y_true_rt - y_pred_rt)):.4f}"
        )

        model = model_rt
        scaler_X = scaler_X_tv
        scaler_y = scaler_y_tv
        X_train = scaler_X.transform(X_train_orig)
        X_val = scaler_X.transform(X_val_orig)
        X_test = scaler_X.transform(X_test_orig)
        y_train = scaler_y.transform(y_train_orig)
        y_val = scaler_y.transform(y_val_orig)
        y_test = scaler_y.transform(y_test_orig)
        print("Using retrained model + train+val-fitted scalers.")

    # ── Save model architecture ──────────────────────────────────────────
    try:
        from tensorflow.keras.layers import Dense

        dense_layers = [
            layer for layer in model.layers if isinstance(layer, Dense)
        ]
        if dense_layers:
            print("\n=== Final Dense Layers ===")
            scheme_lines = []
            for i, layer in enumerate(dense_layers, start=1):
                act_name = (
                    layer.activation.__name__ if layer.activation else "N/A"
                )
                reg = layer.kernel_regularizer
                reg_str = ""
                if reg is not None:
                    try:
                        reg_str = (
                            f", regularizer=L2={reg.l2}"
                            if hasattr(reg, "l2")
                            else f", regularizer={reg}"
                        )
                    except Exception:
                        reg_str = f", regularizer={reg}"
                line = (
                    f"Layer {i}: name='{layer.name}', units={layer.units}, "
                    f"activation={act_name}{reg_str}"
                )
                print(line)
                scheme_lines.append(line)
            scheme_path = os.path.join(cfg.export_dir, "model_scheme.txt")
            with open(scheme_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "Final Dense Layers (in model order)\n"
                    "===================================\n"
                )
                for ln in scheme_lines:
                    fh.write(ln + "\n")
            print(f"Model scheme saved to: {scheme_path}")
    except Exception as exc:
        print(f"Error saving model scheme: {exc}")

    # ── Save model & scalers ─────────────────────────────────────────────
    model.save(os.path.join(cfg.export_dir, "final_model.keras"))
    try:
        model.save(os.path.join(cfg.export_dir, "final_model.h5"))
    except Exception as exc:
        print(f"Could not save .h5 (ok to ignore): {exc}")

    joblib.dump(scaler_X, os.path.join(cfg.export_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(cfg.export_dir, "scaler_y.pkl"))

    # Save ensemble fold models
    if cfg.use_cv_ensemble and cv_fold_models:
        ensemble_dir = os.path.join(cfg.export_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        for idx, (fm, fsx, fsy) in enumerate(
            zip(cv_fold_models, cv_fold_scalers_X, cv_fold_scalers_y)
        ):
            fm.save(os.path.join(ensemble_dir, f"fold_{idx}.keras"))
            joblib.dump(fsx, os.path.join(ensemble_dir, f"fold_{idx}_scaler_X.pkl"))
            joblib.dump(fsy, os.path.join(ensemble_dir, f"fold_{idx}_scaler_y.pkl"))
        print(f"Ensemble ({len(cv_fold_models)} folds) saved to {ensemble_dir}/")

    print(f"Model and scalers saved to {cfg.export_dir}/")

    # ── Predictions (inverse-scaled) ─────────────────────────────────────
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
    y_val_pred = scaler_y.inverse_transform(model.predict(X_val, verbose=0))
    y_test_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
    y_train_inv = scaler_y.inverse_transform(y_train)
    y_val_inv = scaler_y.inverse_transform(y_val)
    y_test_inv = scaler_y.inverse_transform(y_test)

    # ── Phase 10: Metrics ────────────────────────────────────────────────
    p = n_features
    m_train = compute_basic_metrics(y_train_inv, y_train_pred)
    m_val = compute_basic_metrics(y_val_inv, y_val_pred)
    m_test = compute_basic_metrics(y_test_inv, y_test_pred)

    m_train["R2_adj"] = adjusted_r2(m_train["R2"], m_train["n"], p)
    m_val["R2_adj"] = adjusted_r2(m_val["R2"], m_val["n"], p)
    m_test["R2_adj"] = adjusted_r2(m_test["R2"], m_test["n"], p)
    m_train["Predicted_R2_Q2"] = pred_R2_train
    m_val["Predicted_R2_Q2"] = np.nan
    m_test["Predicted_R2_Q2"] = np.nan

    print("\n=== Final Metrics (single model) ===")
    for label, pct, m in [
        ("Training", cfg.train_percent, m_train),
        ("Validation", cfg.val_percent, m_val),
        ("Test", cfg.test_percent, m_test),
    ]:
        print(f"{label} set ({pct}%):")
        for k, v in m.items():
            print(f"  {k}: {v}")

    # Ensemble metrics
    if cfg.use_cv_ensemble and cv_fold_models:
        m_ens_test = compute_basic_metrics(
            y_test_orig.reshape(-1), ensemble_preds_test
        )
        m_ens_test["R2_adj"] = adjusted_r2(m_ens_test["R2"], m_ens_test["n"], p)
        print(f"\n=== Ensemble Metrics (TEST, {len(cv_fold_models)} folds) ===")
        for k, v in m_ens_test.items():
            print(f"  {k}: {v}")

    # Save statistics
    stats_path = os.path.join(cfg.export_dir, "model_statistics.txt")
    try:
        with open(stats_path, "w", encoding="utf-8") as fh:
            fh.write(
                "Model statistics summary\n========================\n\n"
            )
            fh.write(f"Current date: {_dt.date.today().isoformat()}\n")
            fh.write(f"Number of predictors (p): {p}\n")
            fh.write(
                f"Data split: {cfg.train_percent}% / {cfg.val_percent}% / {cfg.test_percent}%\n"
            )
            fh.write(f"Hill pre-training: {cfg.do_hill_pretrain}\n")
            fh.write(f"Ensemble: {cfg.use_cv_ensemble} ({len(cv_fold_models)} folds)\n\n")
            fh.write("Hyperparameters (best):\n")
            for k, v in best_hp.values.items():
                fh.write(f"  {k}: {v}\n")
            for label, pct, m in [
                ("Training", cfg.train_percent, m_train),
                ("Validation", cfg.val_percent, m_val),
                ("Test", cfg.test_percent, m_test),
            ]:
                fh.write(f"\n{label} set metrics ({pct}%):\n")
                for k, v in m.items():
                    fh.write(f"  {k}: {v}\n")
            if cfg.use_cv_ensemble and cv_fold_models:
                fh.write(f"\nEnsemble TEST metrics ({len(cv_fold_models)} folds):\n")
                for k, v in m_ens_test.items():
                    fh.write(f"  {k}: {v}\n")
            fh.write(f"\nPredicted R2 (Q2, OOF, no leakage): {pred_R2_train}\n")
            fh.write(f"Best epoch (VAL loss): {best_epoch}\n")
            fh.write(f"Optional retrain: {cfg.do_optional_retrain}\n")
            fh.write(
                f"PI calibration: {cfg.pi_calibration}, alpha={cfg.pi_alpha}\n"
            )
        print(f"Statistics saved to: {stats_path}")
    except Exception as exc:
        print(f"Could not save statistics: {exc}")

    # ── Plots: MSE evolution ─────────────────────────────────────────────
    try:
        mse_path = os.path.join(cfg.export_dir, "mse_evolution.png")
        loss = history.history.get("loss")
        val_loss = history.history.get("val_loss")
        if loss:
            epochs_range = range(1, len(loss) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs_range, loss, label="Train MSE", marker="o")
            if val_loss:
                plt.plot(epochs_range, val_loss, label="Val MSE", marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("MSE Evolution")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(mse_path, dpi=150)
            if IN_NOTEBOOK:
                plt.show()
            else:
                plt.close()
    except Exception as exc:
        print(f"Error plotting MSE: {exc}")

    # ── Plots: diagnostics ───────────────────────────────────────────────
    try:
        plots_dir = os.path.join(cfg.export_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for tag, yt, yp in [
            ("train", y_train_inv, y_train_pred),
            ("val", y_val_inv, y_val_pred),
            ("test", y_test_inv, y_test_pred),
        ]:
            label = tag.capitalize()
            plot_pred_vs_actual(
                yt,
                yp,
                f"Predicted vs Actual ({label})",
                save_path=os.path.join(plots_dir, f"pred_vs_actual_{tag}.png"),
            )
            plot_residuals(
                yt,
                yp,
                label,
                save_prefix=os.path.join(plots_dir, tag),
                standardize=True,
            )
            plot_residuals(
                yt,
                yp,
                label,
                save_prefix=os.path.join(plots_dir, tag),
                standardize=False,
            )

        # Ensemble diagnostic plots
        if cfg.use_cv_ensemble and cv_fold_models:
            plot_pred_vs_actual(
                y_test_orig.reshape(-1),
                ensemble_preds_test,
                f"Ensemble Predicted vs Actual (Test, {len(cv_fold_models)} folds)",
                save_path=os.path.join(plots_dir, "pred_vs_actual_ensemble_test.png"),
            )
            plot_pred_vs_actual(
                y_train_orig.reshape(-1),
                ensemble_preds_train,
                f"Ensemble Predicted vs Actual (Train, {len(cv_fold_models)} folds)",
                save_path=os.path.join(plots_dir, "pred_vs_actual_ensemble_train.png"),
            )

        print(f"Diagnostic plots saved to: {plots_dir}")
    except Exception as exc:
        print(f"Error creating diagnostic plots: {exc}")
        traceback.print_exc()

    # ── Export DataFrames ────────────────────────────────────────────────
    results_train = make_export_df(
        labels_train.reset_index(drop=True), X_train_df, y_train_inv, y_train_pred
    )
    results_val = make_export_df(
        labels_val.reset_index(drop=True), X_val_df, y_val_inv, y_val_pred
    )
    results_test = make_export_df(
        labels_test.reset_index(drop=True), X_test_df, y_test_inv, y_test_pred
    )

    for name, res_df in [
        ("train_predictions", results_train),
        ("val_predictions", results_val),
        ("test_predictions", results_test),
    ]:
        res_df.to_excel(
            os.path.join(cfg.export_dir, f"{name}.xlsx"),
            index=False,
            engine="openpyxl",
        )
        res_df.to_csv(
            os.path.join(cfg.export_dir, f"{name}.csv"), index=False
        )

    # Full-row exports
    try:

        def _full_row_export(
            orig_df: pd.DataFrame,
            indices: np.ndarray,
            y_t: np.ndarray,
            y_p: np.ndarray,
        ) -> pd.DataFrame:
            full = orig_df.iloc[indices].reset_index(drop=True).copy()
            yt = np.asarray(y_t).reshape(-1)
            yp = np.asarray(y_p).reshape(-1)
            res = yt - yp
            abs_e = np.abs(res)
            with np.errstate(divide="ignore", invalid="ignore"):
                abs_pct = np.where(
                    np.abs(yt) > 1e-12, 100 * abs_e / np.abs(yt), np.nan
                )
            full["Actual_Value"] = yt
            full["Predicted_Value"] = yp
            full["Residual"] = res
            full["Abs_Error"] = abs_e
            full["Abs_Percent_Error"] = abs_pct
            pred_cols = [
                "Actual_Value",
                "Predicted_Value",
                "Residual",
                "Abs_Error",
                "Abs_Percent_Error",
            ]
            other = [c for c in full.columns if c not in pred_cols]
            return full[other + pred_cols]

        for tag, idx, yt, yp in [
            ("train", idx_train, y_train_inv, y_train_pred),
            ("val", idx_val, y_val_inv, y_val_pred),
            ("test", idx_test, y_test_inv, y_test_pred),
        ]:
            full_df = _full_row_export(df, idx, yt, yp)
            full_df.to_excel(
                os.path.join(cfg.export_dir, f"{tag}_full_with_all_columns.xlsx"),
                index=False,
                engine="openpyxl",
            )
            full_df.to_csv(
                os.path.join(cfg.export_dir, f"{tag}_full_with_all_columns.csv"),
                index=False,
            )
        print("Full-row exports saved.")
    except Exception as exc:
        print(f"Error creating full-row exports: {exc}")
        traceback.print_exc()

    _show(results_train.head())
    _show(results_val.head())
    _show(results_test.head())

    # ── 3-D surface plots ────────────────────────────────────────────────
    try:
        plots_3d = os.path.join(cfg.export_dir, "plots_3d")
        os.makedirs(plots_3d, exist_ok=True)

        xref = _build_ref_vector(
            X_train_orig, cfg.surface_hold_mode, cfg.surface_row_index
        )
        pairs = list(combinations(range(n_features), 2))[: cfg.surface_max_pairs]
        print(f"\nGenerating {len(pairs)} 3D surface(s)…")

        for i, j in pairs:
            xi = _grid_vals(
                X_train_orig,
                i,
                cfg.surface_grid_n,
                cfg.surface_range_mode,
                cfg.surface_q_low,
                cfg.surface_q_high,
            )
            xj = _grid_vals(
                X_train_orig,
                j,
                cfg.surface_grid_n,
                cfg.surface_range_mode,
                cfg.surface_q_low,
                cfg.surface_q_high,
            )
            XI, XJ = np.meshgrid(xi, xj)
            Xgrid = np.tile(xref.reshape(1, -1), (XI.size, 1))
            Xgrid[:, i] = XI.reshape(-1)
            Xgrid[:, j] = XJ.reshape(-1)
            Z = _predict_orig(Xgrid, model, scaler_X, scaler_y).reshape(
                XI.shape
            )

            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(
                XI,
                XJ,
                Z,
                cmap="viridis",
                linewidth=0,
                antialiased=True,
                alpha=0.92,
            )
            fig.colorbar(
                surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output"
            )

            if cfg.surface_overlay_scatter:
                z_tr = _predict_orig(
                    X_train_orig, model, scaler_X, scaler_y
                )
                ax.scatter(
                    X_train_orig[:, i],
                    X_train_orig[:, j],
                    z_tr,
                    c="k",
                    s=cfg.surface_scatter_size,
                    alpha=cfg.surface_scatter_alpha,
                )

            ci_name = str(input_columns[i])
            cj_name = str(input_columns[j])
            ax.set_title(
                f"Predicted: {ci_name} vs {cj_name}\n"
                f"(others held: {cfg.surface_hold_mode})"
            )
            ax.set_xlabel(ci_name)
            ax.set_ylabel(cj_name)
            ax.set_zlabel("Predicted output")
            ax.view_init(elev=25, azim=-135)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    plots_3d,
                    f"surface_{_safe_name(ci_name)}_vs_{_safe_name(cj_name)}.png",
                ),
                dpi=cfg.surface_dpi,
            )
            if IN_NOTEBOOK:
                plt.show()
            else:
                plt.close(fig)

        n_saved = len([f for f in os.listdir(plots_3d) if f.endswith(".png")])
        print(f"3D surfaces saved: {n_saved}")
    except Exception as exc:
        print(f"Error generating 3D surfaces: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED!")
    print("=" * 80)

    # ── DOWNLOAD #1: TRAINING RESULTS ────────────────────────────────────
    print("\nDOWNLOAD #1: TRAINING RESULTS")
    print("-" * 80)

    training_files = []
    for fn in os.listdir(cfg.export_dir):
        fp = os.path.join(cfg.export_dir, fn)
        if os.path.isfile(fp):
            training_files.append(fp)
    for subdir in ("plots", "plots_3d", "ensemble"):
        d = os.path.join(cfg.export_dir, subdir)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                training_files.append(os.path.join(d, fn))

    try:
        zip_train_path = "training_results.zip"
        if os.path.exists(zip_train_path):
            os.remove(zip_train_path)
        with zipfile.ZipFile(zip_train_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in training_files:
                zf.write(fp, arcname=os.path.relpath(fp, "."))
        size_mb = os.path.getsize(zip_train_path) / (1024 * 1024)
        print(
            f"Created: {zip_train_path} ({size_mb:.2f} MB, "
            f"{len(training_files)} files)"
        )

        if use_colab:
            print(
                "Find 'training_results.zip' in Colab Files sidebar "
                "→ right-click → Download"
            )
        else:
            dl = os.path.expanduser("~/Downloads")
            os.makedirs(dl, exist_ok=True)
            shutil.copy2(zip_train_path, os.path.join(dl, zip_train_path))
            print(f"Saved to {os.path.join(dl, zip_train_path)}")
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()

    # ── NEW DATA PREDICTION ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("NEW DATA PREDICTION SECTION")
    print("=" * 80)

    # Calibration residuals for PI
    if cfg.pi_calibration == "val":
        cal_res = y_val_inv.reshape(-1) - y_val_pred.reshape(-1)
    elif cfg.pi_calibration == "oof":
        cal_res = y_oof_true_inv.reshape(-1) - y_oof_pred_inv.reshape(-1)
    else:
        raise ValueError(
            f"pi_calibration must be 'val' or 'oof', got '{cfg.pi_calibration}'"
        )

    q_low = float(np.quantile(cal_res, cfg.pi_alpha / 2.0))
    q_high = float(np.quantile(cal_res, 1.0 - cfg.pi_alpha / 2.0))
    print(
        f"Empirical PI ({cfg.pi_calibration}): q_low={q_low:.4f}  "
        f"q_high={q_high:.4f}  alpha={cfg.pi_alpha}"
    )

    print("\nUpload new data for predictions (optional).")
    new_data_loaded = False
    new_df: Optional[pd.DataFrame] = None

    if use_colab:
        try:
            uploaded_new = colab_files.upload()
            if uploaded_new:
                new_file_name = list(uploaded_new.keys())[0]
                new_df = load_table(new_file_name, sep=cfg.sep)
                new_data_loaded = True
                print(f"New data loaded. Shape: {new_df.shape}")
            else:
                print("No new data uploaded. Skipping.")
        except Exception as exc:
            print(f"Upload failed: {exc}")
    else:
        for candidate in ["new_data.tsv", "new_data.csv", "new_data.xlsx"]:
            if os.path.exists(candidate):
                new_df = load_table(candidate, sep=cfg.sep)
                new_data_loaded = True
                print(f"New data loaded from {candidate}. Shape: {new_df.shape}")
                break
        if not new_data_loaded:
            print("No new_data.* found. Skipping predictions.")

    if new_data_loaded and new_df is not None:
        try:
            _show(new_df.head())
            n_cols_new = new_df.shape[1]
            new_labels = (
                new_df.iloc[:, : cfg.n_labels]
                if cfg.n_labels > 0
                else pd.DataFrame()
            )

            if n_cols_new < cfg.n_labels + cfg.n_inputs:
                raise ValueError(
                    f"Insufficient columns: {n_cols_new} < "
                    f"{cfg.n_labels + cfg.n_inputs}"
                )
            new_inputs = new_df.iloc[
                :, cfg.n_labels : cfg.n_labels + cfg.n_inputs
            ]

            if new_inputs.shape[1] != n_features:
                if set(new_inputs.columns).issubset(set(input_columns)):
                    new_inputs = new_inputs[input_columns]
                else:
                    raise ValueError(
                        "Column mismatch between training and new data inputs."
                    )

            if new_inputs.isna().sum().sum() > 0:
                print("Imputing NaN features with TRAIN means…")
                imp = SimpleImputer(strategy="mean").fit(X_train_orig)
                new_inputs = pd.DataFrame(
                    imp.transform(new_inputs), columns=new_inputs.columns
                )

            new_X = new_inputs.values
            print(f"Generating predictions for {len(new_X)} samples…")

            # Single-model prediction
            new_y_pred = scaler_y.inverse_transform(
                model.predict(scaler_X.transform(new_X), verbose=0)
            ).reshape(-1)

            # Ensemble prediction (if available)
            new_y_pred_ensemble: Optional[np.ndarray] = None
            if cfg.use_cv_ensemble and cv_fold_models:
                new_y_pred_ensemble = np.zeros(len(new_X))
                for _fm, _fsx, _fsy in zip(
                    cv_fold_models, cv_fold_scalers_X, cv_fold_scalers_y
                ):
                    _pred_i = _fsy.inverse_transform(
                        _fm.predict(_fsx.transform(new_X), verbose=0)
                    ).reshape(-1)
                    new_y_pred_ensemble += _pred_i
                new_y_pred_ensemble /= len(cv_fold_models)

            pi_lower = new_y_pred + q_low
            pi_upper = new_y_pred + q_high

            # Compact export
            compact = pd.DataFrame(index=range(len(new_X)))
            if new_labels.shape[0] == len(new_X):
                compact = pd.concat(
                    [compact, new_labels.reset_index(drop=True)], axis=1
                )
            compact = pd.concat(
                [compact, new_inputs.reset_index(drop=True)], axis=1
            )
            compact["Predicted_Value"] = new_y_pred
            if new_y_pred_ensemble is not None:
                compact["Predicted_Value_Ensemble"] = new_y_pred_ensemble
            compact["PI_Lower_95%"] = pi_lower
            compact["PI_Upper_95%"] = pi_upper
            compact["PI_Width"] = pi_upper - pi_lower
            compact["PI_Calibration_Source"] = cfg.pi_calibration

            # If actual values exist
            has_actual = False
            if n_cols_new > cfg.n_labels + cfg.n_inputs:
                try:
                    y_actual = new_df.iloc[
                        :, cfg.n_labels + cfg.n_inputs
                    ].values.reshape(-1, 1)
                    if not np.isnan(y_actual).all():
                        if np.isnan(y_actual).any():
                            print("Imputing NaN targets with TRAIN target mean…")
                            imp_y = SimpleImputer(strategy="mean").fit(
                                y_train_orig
                            )
                            y_actual = imp_y.transform(y_actual)
                        y_flat = y_actual.reshape(-1)
                        has_actual = True
                        residual_new = y_flat - new_y_pred
                        abs_err_new = np.abs(residual_new)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            abs_pct_new = np.where(
                                np.abs(y_flat) > 1e-12,
                                100 * abs_err_new / np.abs(y_flat),
                                np.nan,
                            )
                        compact["Actual_Value"] = y_flat
                        compact["Prediction_Error"] = residual_new
                        compact["Abs_Error"] = abs_err_new
                        compact["Abs_Percent_Error"] = abs_pct_new
                        within_pi = (y_flat >= pi_lower) & (
                            y_flat <= pi_upper
                        )
                        compact["Within_95%_PI"] = within_pi.astype(int)

                        new_metrics = compute_basic_metrics(y_flat, new_y_pred)
                        print("\nMetrics on new data (single model):")
                        for k, v in new_metrics.items():
                            print(f"  {k}: {v}")
                        coverage = within_pi.sum() / len(within_pi) * 100
                        print(
                            f"Empirical 95% PI coverage: "
                            f"{within_pi.sum()}/{len(within_pi)} ({coverage:.1f}%)"
                        )

                        if new_y_pred_ensemble is not None:
                            ens_metrics = compute_basic_metrics(
                                y_flat, new_y_pred_ensemble
                            )
                            print("\nMetrics on new data (ensemble):")
                            for k, v in ens_metrics.items():
                                print(f"  {k}: {v}")
                except Exception as exc:
                    print(f"Could not extract target: {exc}")

            if not has_actual:
                print("No actual values. Predictions + PI only.")

            # Full-row export
            full_new = new_df.copy()
            full_new["Predicted_Value"] = new_y_pred
            if new_y_pred_ensemble is not None:
                full_new["Predicted_Value_Ensemble"] = new_y_pred_ensemble
            full_new["PI_Lower_95%"] = pi_lower
            full_new["PI_Upper_95%"] = pi_upper
            full_new["PI_Width"] = pi_upper - pi_lower
            full_new["PI_Calibration_Source"] = cfg.pi_calibration
            if has_actual:
                full_new["Actual_Value"] = y_flat  # type: ignore[possibly-undefined]
                full_new["Prediction_Error"] = residual_new  # type: ignore[possibly-undefined]
                full_new["Abs_Error"] = abs_err_new  # type: ignore[possibly-undefined]
                full_new["Abs_Percent_Error"] = abs_pct_new  # type: ignore[possibly-undefined]
                full_new["Within_95%_PI"] = within_pi.astype(int)  # type: ignore[possibly-undefined]

            compact.to_excel(
                os.path.join(cfg.export_dir, "new_data_predictions.xlsx"),
                index=False,
                engine="openpyxl",
            )
            compact.to_csv(
                os.path.join(cfg.export_dir, "new_data_predictions.csv"),
                index=False,
            )
            full_new.to_excel(
                os.path.join(
                    cfg.export_dir, "new_data_full_with_all_columns.xlsx"
                ),
                index=False,
                engine="openpyxl",
            )
            full_new.to_csv(
                os.path.join(
                    cfg.export_dir, "new_data_full_with_all_columns.csv"
                ),
                index=False,
            )
            print("Predictions exported.")
            _show(compact.head(10))

        except Exception as exc:
            print(f"Error processing new data: {exc}")
            traceback.print_exc()

    # ── DOWNLOAD #2: NEW DATA PREDICTIONS ────────────────────────────────
    print("\n" + "=" * 80)
    print("DOWNLOAD #2: NEW DATA PREDICTIONS")
    print("=" * 80)

    pred_files = [
        os.path.join(cfg.export_dir, "new_data_predictions.xlsx"),
        os.path.join(cfg.export_dir, "new_data_predictions.csv"),
        os.path.join(cfg.export_dir, "new_data_full_with_all_columns.xlsx"),
        os.path.join(cfg.export_dir, "new_data_full_with_all_columns.csv"),
    ]
    pred_files = [f for f in pred_files if os.path.exists(f)]

    if pred_files:
        try:
            zip_pred = "new_data_predictions.zip"
            if os.path.exists(zip_pred):
                os.remove(zip_pred)
            with zipfile.ZipFile(zip_pred, "w", zipfile.ZIP_DEFLATED) as zf:
                for fp in pred_files:
                    zf.write(fp, arcname=os.path.relpath(fp, "."))
            size_mb = os.path.getsize(zip_pred) / (1024 * 1024)
            print(f"Created: {zip_pred} ({size_mb:.2f} MB)")

            if use_colab:
                print(
                    "Find 'new_data_predictions.zip' in Colab Files sidebar "
                    "→ right-click → Download"
                )
            else:
                dl = os.path.expanduser("~/Downloads")
                shutil.copy2(zip_pred, os.path.join(dl, zip_pred))
                print(f"Saved to {os.path.join(dl, zip_pred)}")
        except Exception as exc:
            print(f"Error: {exc}")
            traceback.print_exc()
    else:
        print("No new-data predictions available.")

    print("\n" + "=" * 80)
    print("ALL OPERATIONS COMPLETED!")
    print("=" * 80)
    print("\nDownloads Summary:")
    print("  #1: training_results.zip — Ready")
    print("  #2: new_data_predictions.zip — Ready (if data uploaded)")
    if cfg.use_cv_ensemble and cv_fold_models:
        print(f"  Ensemble: {len(cv_fold_models)} fold models saved")
    if cfg.do_hill_pretrain:
        print("  Hill pre-training: warmup_model.keras saved")
    print("\nFiles are ready to use!")


if __name__ == "__main__":
    main()
