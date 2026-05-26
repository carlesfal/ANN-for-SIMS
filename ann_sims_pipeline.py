# =============================================================================
# Three-Phase Pipeline:
#   Hill Warmup -> Phase A (Dataset 1) -> Phase B (Dataset 2)
#
# DESIGN:
#
# HILL WARMUP — Synthetic Hill-function pre-training.
#   Generates synthetic data, tunes dropout/L2/LR, trains warmup model.
#   Snapshots weights for transfer into Phase A.
#
# PHASE A — FULL TRAINING ON DATASET 1
#   Upload Dataset 1 (n_labels_a + n_inputs_a + target).
#   Weight transfer from Hill warmup.
#   L2-SP regularisation toward warmup weights.
#   Three-stage progressive unfreezing + L-BFGS-B.
#   K-fold CV, final training, diagnostics.
#
# PHASE B — FINE-TUNE ON DATASET 2 (can differ in n_labels/n_inputs)
#   Upload Dataset 2 (n_labels_b + n_inputs_b + target).
#   Weight transfer from Phase A model.
#   L2-SP regularisation toward Phase A weights.
#   Three-stage progressive unfreezing + L-BFGS-B.
#   K-fold CV, final training, diagnostics.
#   Optional new-data prediction.
#
# TECHNIQUES:
#   Fixed shared architecture (tuner only searches dropout/L2/LR).
#   L2-SP (Li et al., 2018): penalise deviation from pre-trained weights.
#   L-BFGS-B (Broyden-Fletcher-Goldfarb-Shanno) second-order refinement.
#   ReduceLROnPlateau at every training stage.
#   Unified standardisation (fit on TRAIN only; inverse at output).
# =============================================================================

# --- Dependency installation ---
try:
    get_ipython  # type: ignore
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False

use_colab = False
try:
    from google.colab import files as colab_files  # type: ignore
    use_colab = True
except Exception:
    pass

if IN_NOTEBOOK:
    print("Installing dependencies...")
    get_ipython().run_line_magic("pip", "install -q keras-tuner seaborn openpyxl joblib")  # type: ignore
else:
    import subprocess, sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install",
         "keras-tuner", "seaborn", "openpyxl", "joblib"]
    )

# --- Imports ---
import os, shutil, traceback, zipfile
import datetime as _dt
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize as scipy_minimize
import matplotlib
matplotlib.use("Agg") if not IN_NOTEBOOK else None
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers, backend as K
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

try:
    import keras_tuner as kt
except ImportError:
    try:
        import kerastuner as kt  # type: ignore
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tuner"])
        import keras_tuner as kt  # type: ignore


def _show(obj, n=5):
    if IN_NOTEBOOK:
        try:
            display(obj)  # type: ignore
        except Exception:
            print(obj)
    else:
        print(obj.head(n) if hasattr(obj, "head") else obj)


# =============================================================================
#  L2-SP REGULARISER
# =============================================================================
class L2SP(keras.regularizers.Regularizer):
    """L2-SP: Starting-Point L2 (Li et al., 2018).
    ``alpha * sum((w - w0)**2)``; falls back to standard L2 when w0 is None.
    """
    def __init__(self, alpha: float = 1e-3, w0: Optional[np.ndarray] = None):
        self.alpha = alpha
        self.w0 = w0

    def __call__(self, w):
        if self.w0 is not None:
            ref = tf.constant(self.w0, dtype=w.dtype)
            return self.alpha * tf.reduce_sum(tf.square(w - ref))
        return self.alpha * tf.reduce_sum(tf.square(w))

    def get_config(self):
        return {"alpha": float(self.alpha),
                "w0": self.w0.tolist() if self.w0 is not None else None}


# =============================================================================
#  CONFIGURATION
# =============================================================================
@dataclass
class PipelineConfig:
    train_percent: int = 60
    val_percent: int = 20
    test_percent: int = 20

    # Phase A dataset layout
    n_labels_a: int = 7
    n_inputs_a: int = 10

    # Phase B dataset layout (can differ from Phase A)
    n_labels_b: int = 7
    n_inputs_b: int = 10

    target_col: Optional[int] = None
    sep: str = "\t"

    disable_gpu: bool = True

    # Fixed shared architecture (Hill warmup + Phase A + Phase B)
    architecture: List[int] = field(default_factory=lambda: [256, 512, 128, 64, 32])

    # Hill warmup settings
    n_synthetic: int = 500
    hill_v_max: float = 200.0
    hill_k: float = 0.10
    hill_n: float = 1.20
    hill_x_min: float = 0.1
    hill_x_max: float = 100.0
    pretrain_epochs: int = 100
    pretrain_tuner_trials: int = 15

    # Training
    tuner_trials: int = 15
    k_folds: int = 15
    random_seed: int = 42
    tuner_epochs: int = 200
    cv_epochs: int = 200
    final_epochs: int = 200

    do_optional_retrain: bool = True

    # Three-stage fine-tuning
    ft_stage1_epochs: int = 20
    ft_stage1_lr: float = 1e-3
    ft_stage2_epochs: int = 40
    ft_stage2_lr: float = 5e-4
    ft_stage2_unfreeze_last_n: int = 3
    ft_stage3_epochs: int = 40
    ft_stage3_lr: float = 1e-4

    # L2-SP strength
    l2sp_alpha: float = 1e-3

    # L-BFGS-B
    lbfgs_maxiter: int = 200
    lbfgs_enabled: bool = True

    pi_calibration: str = "val"
    pi_alpha: float = 0.05

    export_dir_a: str = "results_dataset1"
    export_dir_b: str = "results_dataset2"

    # 3D surface plot settings
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

    def __post_init__(self):
        total = self.train_percent + self.val_percent + self.test_percent
        if abs(total - 100) > 0.01:
            raise ValueError(
                f"Split percentages must sum to 100. "
                f"Got {self.train_percent}+{self.val_percent}+"
                f"{self.test_percent}={total}"
            )


cfg = PipelineConfig()

print(f"Data split: TRAIN={cfg.train_percent}%  VAL={cfg.val_percent}%  "
      f"TEST={cfg.test_percent}%")
print(f"Shared architecture: {cfg.architecture}")
print(f"Phase A: n_labels={cfg.n_labels_a}  n_inputs={cfg.n_inputs_a}")
print(f"Phase B: n_labels={cfg.n_labels_b}  n_inputs={cfg.n_inputs_b}")

np.random.seed(cfg.random_seed)
tf.random.set_seed(cfg.random_seed)

if cfg.disable_gpu:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("GPU disabled. Using CPU.")
    except Exception as e:
        print(f"Could not change GPU visibility: {e}")


# =============================================================================
#  L-BFGS-B REFINEMENT
# =============================================================================
def lbfgs_refine(model, X_tr, y_tr, X_va=None, y_va=None,
                 maxiter=200, verbose=True):
    """Polish weights with L-BFGS-B (second-order optimiser)."""
    if not cfg.lbfgs_enabled:
        return model

    trainable_vars = model.trainable_variables
    shapes = [v.shape.as_list() for v in trainable_vars]
    sizes = [int(np.prod(s)) for s in shapes]

    X_tensor = tf.constant(X_tr, dtype=tf.float32)
    y_tensor = tf.constant(y_tr, dtype=tf.float32)

    loss_before = float(tf.reduce_mean(tf.square(
        model(X_tensor, training=False) - y_tensor)).numpy())

    def _unflatten(flat):
        tensors, offset = [], 0
        for shape, sz in zip(shapes, sizes):
            tensors.append(tf.constant(
                flat[offset:offset+sz].reshape(shape), dtype=tf.float32))
            offset += sz
        return tensors

    @tf.function
    def _loss_and_grad(weight_tensors):
        for var, val in zip(trainable_vars, weight_tensors):
            var.assign(val)
        with tf.GradientTape() as tape:
            pred = model(X_tensor, training=False)
            loss = tf.reduce_mean(tf.square(pred - y_tensor))
        grads = tape.gradient(loss, trainable_vars)
        return loss, grads

    call_count = [0]

    def func(flat_weights):
        wt = _unflatten(flat_weights)
        loss, grads = _loss_and_grad(wt)
        flat_grad = np.concatenate([g.numpy().ravel() for g in grads])
        call_count[0] += 1
        return float(loss.numpy()), flat_grad.astype(np.float64)

    w0 = np.concatenate([v.numpy().ravel() for v in trainable_vars])

    result = scipy_minimize(
        func, w0.astype(np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8,
                 "disp": False},
    )

    opt_tensors = _unflatten(result.x)
    for var, val in zip(trainable_vars, opt_tensors):
        var.assign(val)

    loss_after = float(tf.reduce_mean(tf.square(
        model(X_tensor, training=False) - y_tensor)).numpy())

    if verbose:
        msg = (f"  [L-BFGS-B] {result.nit} iters, {call_count[0]} f-evals | "
               f"MSE {loss_before:.6f} -> {loss_after:.6f}")
        if X_va is not None and y_va is not None:
            val_pred = model.predict(X_va, verbose=0)
            val_mse = float(np.mean((val_pred - y_va) ** 2))
            msg += f" | val_MSE={val_mse:.6f}"
        if result.success:
            msg += " [converged]"
        else:
            msg += f" [{result.message}]"
        print(msg)

    return model


# =============================================================================
#  FIXED-ARCHITECTURE MODEL BUILDER
# =============================================================================
def make_fixed_builder(n_feat, arch, kernel_regularizers=None):
    """Dense widths locked to *arch*; tuner only picks dropout / L2 / LR."""
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(n_feat,)))
        l2_val = hp.Choice("l2_reg", [1e-4, 1e-3, 1e-2])
        for idx, units in enumerate(arch):
            kreg = (kernel_regularizers[idx]
                    if kernel_regularizers is not None
                    else regularizers.l2(l2_val))
            model.add(layers.Dense(
                units, activation="relu", kernel_regularizer=kreg,
                name=f"dense_{idx}",
            ))
            model.add(layers.Dropout(
                hp.Float(f"dropout_{idx}", 0.0, 0.5, step=0.1),
                name=f"drop_{idx}",
            ))
        out_kreg = (kernel_regularizers[-1]
                    if kernel_regularizers is not None else None)
        model.add(layers.Dense(1, activation="linear",
                               kernel_regularizer=out_kreg, name="output"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", [1e-4, 5e-4, 1e-3, 5e-3])
            ),
            loss="mse", metrics=["mae"],
        )
        return model
    return build_model


# =============================================================================
#  HELPER: load, split, standardise a dataset
# =============================================================================
def load_table(path, sep="\t"):
    _, ext = os.path.splitext(path.lower())
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, engine="openpyxl")
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as e:
            last_err = e
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read '{path}'. Last error: {last_err}")


def upload_dataset(prompt_label=""):
    """Upload or locate a dataset. Returns (file_name, df)."""
    print(f"\nUpload your {prompt_label}dataset.")
    if use_colab:
        uploaded = colab_files.upload()
        if not uploaded:
            raise RuntimeError("No file uploaded.")
        file_name = list(uploaded.keys())[0]
    else:
        candidates = [f"data{'_' + prompt_label.strip().lower().replace(' ','_') if prompt_label.strip() else ''}.tsv",
                      "data.tsv"]
        file_name = None
        for c in candidates:
            if os.path.exists(c):
                file_name = c
                break
        if file_name is None:
            raise FileNotFoundError(
                f"Not in Colab and none of {candidates} found.")
    df = load_table(file_name, sep=cfg.sep)
    print(f"Data loaded ({prompt_label}). Shape: {df.shape}")
    _show(df.head())
    return file_name, df


def select_columns(df, n_labels, n_inputs):
    """Return (labels_df, inputs_df, y_full, input_columns)."""
    n_cols = df.shape[1]
    if cfg.target_col is not None:
        target_idx = cfg.target_col
        if target_idx < 0 or target_idx >= n_cols:
            raise IndexError("target_col out of range.")
        labels_df = df.iloc[:, :n_labels] if n_labels > 0 else pd.DataFrame()
        input_cols = list(range(n_labels, n_cols))
        input_cols.remove(target_idx)
        inputs_df = df.iloc[:, input_cols]
        y_full = df.iloc[:, target_idx].values.reshape(-1, 1)
    elif n_labels + n_inputs >= n_cols:
        print("n_labels + n_inputs >= total columns. Using last column as target.")
        labels_df = df.iloc[:, :n_labels] if n_labels > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, n_labels:-1]
        y_full = df.iloc[:, -1].values.reshape(-1, 1)
    else:
        labels_df = df.iloc[:, :n_labels] if n_labels > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, n_labels:n_labels + n_inputs]
        y_full = df.iloc[:, n_labels + n_inputs].values.reshape(-1, 1)
    print(f"Labels: {labels_df.shape}, Inputs: {inputs_df.shape}, y: {y_full.shape}")
    return labels_df, inputs_df, y_full, list(inputs_df.columns)


def split_and_scale(X_full, y_full, labels_full, row_pos):
    """Train/Val/Test split + standardise. Returns dict of arrays."""
    test_frac = cfg.test_percent / 100.0
    val_frac = cfg.val_percent / 100.0
    train_frac = cfg.train_percent / 100.0

    X_temp, X_test_o, y_temp, y_test_o, lbl_temp, lbl_test, idx_temp, idx_test = (
        train_test_split(X_full, y_full, labels_full, row_pos,
                         test_size=test_frac, random_state=cfg.random_seed)
    )
    val_ratio = val_frac / (train_frac + val_frac)
    X_train_o, X_val_o, y_train_o, y_val_o, lbl_train, lbl_val, idx_train, idx_val = (
        train_test_split(X_temp, y_temp, lbl_temp, idx_temp,
                         test_size=val_ratio, random_state=cfg.random_seed)
    )

    sx = StandardScaler().fit(X_train_o)
    sy = StandardScaler().fit(y_train_o)

    d = dict(
        X_train_orig=X_train_o, X_val_orig=X_val_o, X_test_orig=X_test_o,
        y_train_orig=y_train_o, y_val_orig=y_val_o, y_test_orig=y_test_o,
        labels_train=lbl_train, labels_val=lbl_val, labels_test=lbl_test,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        X_train=sx.transform(X_train_o), X_val=sx.transform(X_val_o),
        X_test=sx.transform(X_test_o),
        y_train=sy.transform(y_train_o), y_val=sy.transform(y_val_o),
        y_test=sy.transform(y_test_o),
        scaler_X=sx, scaler_y=sy,
    )
    print(f"Split: train={len(X_train_o)} val={len(X_val_o)} test={len(X_test_o)}")
    return d


# =============================================================================
#  HELPER: compute metrics, plots, exports
# =============================================================================
def compute_basic_metrics(y_true, y_pred):
    a = np.asarray(y_true).reshape(-1)
    p_ = np.asarray(y_pred).reshape(-1)
    n = len(a)
    res = a - p_
    sse = float(np.sum(res**2))
    mse = sse / n if n else np.nan
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else np.nan
    mae = float(np.mean(np.abs(res))) if n else np.nan
    sep_ = float(np.sqrt(sse / (n - 1))) if n > 1 else np.nan
    mean_abs = float(np.mean(np.abs(a))) if n else np.nan
    mrpd = (100 * float(np.sum(np.abs(res))) / (n * mean_abs)
            if (n and not np.isclose(mean_abs, 0)) else np.nan)
    r2 = float(r2_score(a, p_)) if n else np.nan
    return {"n": n, "SSE": sse, "MSE": mse, "RMSE": rmse, "MAE": mae,
            "SEP": sep_, "MRPD%": mrpd, "R2": r2}


def adj_r2(r2, n, p_):
    return 1 - (1 - r2) * (n - 1) / (n - p_ - 1) if (n - p_ - 1) > 0 else np.nan


def _eval_model(mdl, X_v, y_v, label=""):
    loss, mae = mdl.evaluate(X_v, y_v, verbose=0)
    print(f"  [{label}] val_loss={loss:.6f}  val_mae={mae:.6f}")
    return loss


def plot_pred_vs_actual(y_true, y_pred, title, save_path=None):
    yt, yp = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, alpha=0.6)
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Ideal")
    try:
        sns.regplot(x=yt, y=yp, scatter=False, color="blue", ci=None)
    except Exception:
        pass
    plt.title(title); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.grid(True); plt.legend()
    if save_path:
        plt.tight_layout(); plt.savefig(save_path, dpi=150)
    plt.show() if IN_NOTEBOOK else plt.close()


def plot_residuals(y_true, y_pred, prefix, save_prefix=None, standardize=False):
    res = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)
    if standardize:
        res = (res - np.mean(res)) / (np.std(res) + 1e-12)
    color = "gray" if standardize else "salmon"
    lab = "Std residuals" if standardize else "Raw residuals"
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(res, bins=25, kde=True, color=color, edgecolor="black")
    plt.title(f"{prefix} - {lab} dist"); plt.xlabel(lab); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(res, marker="o", linestyle="-")
    plt.title(f"{prefix} - {lab} series"); plt.xlabel("Index"); plt.grid(True)
    plt.tight_layout()
    suffix = "_std_res.png" if standardize else "_raw_res.png"
    if save_prefix:
        plt.savefig(save_prefix + suffix, dpi=150)
    plt.show() if IN_NOTEBOOK else plt.close()


def make_export_df(lbl, inp, y_true, y_pred):
    a = np.asarray(y_true).reshape(-1)
    p_ = np.asarray(y_pred).reshape(-1)
    res = a - p_
    abs_e = np.abs(res)
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_pct = np.where(np.abs(a) > 1e-12, 100 * abs_e / np.abs(a), np.nan)
    out = pd.DataFrame(index=range(len(a)))
    if lbl is not None and lbl.shape[0] == len(a):
        out = pd.concat([out, lbl.reset_index(drop=True)], axis=1)
    out = pd.concat([out, inp.reset_index(drop=True)], axis=1)
    out["Actual_Value"] = a
    out["Predicted_Value"] = p_
    out["Residual"] = res
    out["Abs_Error"] = abs_e
    out["Abs_Percent_Error"] = abs_pct
    return out


def run_diagnostics(model, d, df_orig, input_columns, export_dir,
                    history, best_epoch, pred_R2_train, best_hp, phase_label,
                    n_transferred=0):
    """Run all diagnostics, metrics, plots, exports for a trained model."""
    os.makedirs(export_dir, exist_ok=True)
    scaler_X = d["scaler_X"]; scaler_y = d["scaler_y"]
    n_features = d["X_train"].shape[1]

    y_train_pred = scaler_y.inverse_transform(model.predict(d["X_train"], verbose=0))
    y_val_pred = scaler_y.inverse_transform(model.predict(d["X_val"], verbose=0))
    y_test_pred = scaler_y.inverse_transform(model.predict(d["X_test"], verbose=0))
    y_train_inv = scaler_y.inverse_transform(d["y_train"])
    y_val_inv = scaler_y.inverse_transform(d["y_val"])
    y_test_inv = scaler_y.inverse_transform(d["y_test"])

    p = n_features
    m_train = compute_basic_metrics(y_train_inv, y_train_pred)
    m_val = compute_basic_metrics(y_val_inv, y_val_pred)
    m_test = compute_basic_metrics(y_test_inv, y_test_pred)
    for m in (m_train, m_val, m_test):
        m["R2_adj"] = adj_r2(m["R2"], m["n"], p)
    m_train["Q2"] = pred_R2_train
    m_val["Q2"] = np.nan
    m_test["Q2"] = np.nan

    print(f"\n=== {phase_label} Final Metrics ===")
    for label, pct, m in [("Train", cfg.train_percent, m_train),
                           ("Val", cfg.val_percent, m_val),
                           ("Test", cfg.test_percent, m_test)]:
        print(f"{label} ({pct}%):")
        for kk, vv in m.items():
            print(f"  {kk}: {vv}")

    # Save statistics
    stats_path = os.path.join(export_dir, "model_statistics.txt")
    try:
        with open(stats_path, "w", encoding="utf-8") as fh:
            fh.write(f"{phase_label} Statistics\n{'='*40}\n\n")
            fh.write(f"Date: {_dt.date.today().isoformat()}\n")
            fh.write(f"Architecture: {cfg.architecture}\n")
            fh.write(f"Split: {cfg.train_percent}/{cfg.val_percent}/{cfg.test_percent}\n\n")
            fh.write("Best HPs:\n")
            for kk, vv in best_hp.values.items():
                fh.write(f"  {kk}: {vv}\n")
            fh.write(f"\nL-BFGS-B enabled: {cfg.lbfgs_enabled}  "
                     f"maxiter: {cfg.lbfgs_maxiter}\n")
            if n_transferred > 0:
                fh.write(f"L2-SP alpha: {cfg.l2sp_alpha}\n")
                fh.write(f"Layers transferred: {n_transferred}\n")
                fh.write(f"Fine-tune stages: "
                         f"S1({cfg.ft_stage1_epochs}ep,LR={cfg.ft_stage1_lr}) "
                         f"S2({cfg.ft_stage2_epochs}ep,LR={cfg.ft_stage2_lr}) "
                         f"S3({cfg.ft_stage3_epochs}ep,LR={cfg.ft_stage3_lr})\n")
            for label, pct, m in [("Train", cfg.train_percent, m_train),
                                   ("Val", cfg.val_percent, m_val),
                                   ("Test", cfg.test_percent, m_test)]:
                fh.write(f"\n{label} ({pct}%):\n")
                for kk, vv in m.items():
                    fh.write(f"  {kk}: {vv}\n")
            fh.write(f"\nQ2 (OOF): {pred_R2_train}\n")
            fh.write(f"Best epoch: {best_epoch}\n")
        print(f"Statistics saved: {stats_path}")
    except Exception as e:
        print(f"Could not save statistics: {e}")

    # Save model architecture
    try:
        from tensorflow.keras.layers import Dense
        dense_layers = [l for l in model.layers if isinstance(l, Dense)]
        if dense_layers:
            scheme_lines = []
            for i, layer in enumerate(dense_layers, start=1):
                act = layer.activation.__name__ if layer.activation else "N/A"
                reg = layer.kernel_regularizer
                reg_str = ""
                if reg is not None:
                    try:
                        reg_str = (f", reg=L2={reg.l2}" if hasattr(reg, "l2")
                                   else f", reg={reg}")
                    except Exception:
                        reg_str = f", reg={reg}"
                line = (f"Layer {i}: '{layer.name}' units={layer.units} "
                        f"act={act}{reg_str}")
                scheme_lines.append(line)
            with open(os.path.join(export_dir, "model_scheme.txt"),
                      "w", encoding="utf-8") as fh:
                fh.write("Dense Layers\n============\n")
                for ln in scheme_lines:
                    fh.write(ln + "\n")
    except Exception as e:
        print(f"Error saving model scheme: {e}")

    # Save model + scalers
    model.save(os.path.join(export_dir, "final_model.keras"))
    try:
        model.save(os.path.join(export_dir, "final_model.h5"))
    except Exception:
        pass
    joblib.dump(scaler_X, os.path.join(export_dir, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(export_dir, "scaler_y.pkl"))
    print(f"Model and scalers saved to {export_dir}/")

    # Plots
    try:
        mse_path = os.path.join(export_dir, "mse_evolution.png")
        loss_hist = history.history.get("loss")
        val_loss_hist = history.history.get("val_loss")
        if loss_hist:
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, len(loss_hist)+1), loss_hist, label="Train MSE", marker="o")
            if val_loss_hist:
                plt.plot(range(1, len(val_loss_hist)+1), val_loss_hist,
                         label="Val MSE", marker="o")
            plt.xlabel("Epoch"); plt.ylabel("MSE")
            plt.title(f"{phase_label} MSE Evolution")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(mse_path, dpi=150)
            plt.show() if IN_NOTEBOOK else plt.close()
    except Exception as e:
        print(f"Error plotting MSE: {e}")

    try:
        plots_dir = os.path.join(export_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        for tag, yt, yp in [("train", y_train_inv, y_train_pred),
                             ("val", y_val_inv, y_val_pred),
                             ("test", y_test_inv, y_test_pred)]:
            plot_pred_vs_actual(yt, yp, f"Pred vs Actual ({tag.title()}) - {phase_label}",
                                os.path.join(plots_dir, f"pred_vs_actual_{tag}.png"))
            plot_residuals(yt, yp, tag.title(),
                           os.path.join(plots_dir, tag), standardize=True)
            plot_residuals(yt, yp, tag.title(),
                           os.path.join(plots_dir, tag), standardize=False)
        print(f"Diagnostic plots saved: {plots_dir}")
    except Exception as e:
        print(f"Error creating plots: {e}")
        traceback.print_exc()

    # Export DataFrames
    X_train_df = pd.DataFrame(d["X_train_orig"], columns=input_columns).reset_index(drop=True)
    X_val_df = pd.DataFrame(d["X_val_orig"], columns=input_columns).reset_index(drop=True)
    X_test_df = pd.DataFrame(d["X_test_orig"], columns=input_columns).reset_index(drop=True)

    results_train = make_export_df(d["labels_train"].reset_index(drop=True),
                                    X_train_df, y_train_inv, y_train_pred)
    results_val = make_export_df(d["labels_val"].reset_index(drop=True),
                                  X_val_df, y_val_inv, y_val_pred)
    results_test = make_export_df(d["labels_test"].reset_index(drop=True),
                                   X_test_df, y_test_inv, y_test_pred)

    for name, rdf in [("train_predictions", results_train),
                       ("val_predictions", results_val),
                       ("test_predictions", results_test)]:
        rdf.to_excel(os.path.join(export_dir, f"{name}.xlsx"),
                     index=False, engine="openpyxl")
        rdf.to_csv(os.path.join(export_dir, f"{name}.csv"), index=False)

    try:
        def _full_export(orig_df, indices, yt, yp):
            full = orig_df.iloc[indices].reset_index(drop=True).copy()
            a = np.asarray(yt).reshape(-1)
            p_ = np.asarray(yp).reshape(-1)
            res = a - p_
            abs_e = np.abs(res)
            with np.errstate(divide="ignore", invalid="ignore"):
                abs_pct = np.where(np.abs(a) > 1e-12, 100*abs_e/np.abs(a), np.nan)
            full["Actual_Value"] = a
            full["Predicted_Value"] = p_
            full["Residual"] = res
            full["Abs_Error"] = abs_e
            full["Abs_Percent_Error"] = abs_pct
            pred_cols = ["Actual_Value","Predicted_Value","Residual",
                         "Abs_Error","Abs_Percent_Error"]
            other = [c for c in full.columns if c not in pred_cols]
            return full[other + pred_cols]

        for tag, idx, yt, yp in [("train", d["idx_train"], y_train_inv, y_train_pred),
                                  ("val", d["idx_val"], y_val_inv, y_val_pred),
                                  ("test", d["idx_test"], y_test_inv, y_test_pred)]:
            fdf = _full_export(df_orig, idx, yt, yp)
            fdf.to_excel(os.path.join(export_dir,
                         f"{tag}_full_with_all_columns.xlsx"),
                         index=False, engine="openpyxl")
            fdf.to_csv(os.path.join(export_dir,
                       f"{tag}_full_with_all_columns.csv"), index=False)
        print("Full-row exports saved.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    _show(results_train.head())
    _show(results_test.head())

    # 3D surface plots
    try:
        d3 = os.path.join(export_dir, "plots_3d")
        os.makedirs(d3, exist_ok=True)

        def _ref_vec(X_orig, mode, idx=0):
            if mode == "median_train": return np.median(X_orig, axis=0)
            if mode == "mean_train":   return np.mean(X_orig, axis=0)
            if mode == "row":          return X_orig[int(idx)].copy()
            raise ValueError(f"Unknown mode '{mode}'")

        def _grid(X_orig, col, n, mode, qlo, qhi):
            v = X_orig[:, col]
            if mode == "minmax":    lo, hi = float(v.min()), float(v.max())
            elif mode == "quantile": lo, hi = float(np.quantile(v, qlo)), float(np.quantile(v, qhi))
            else: raise ValueError(f"Unknown mode '{mode}'")
            if np.isclose(lo, hi): lo, hi = lo - 1, hi + 1
            return np.linspace(lo, hi, n)

        def _pred_orig(X, mdl, sx, sy):
            return sy.inverse_transform(mdl.predict(sx.transform(X), verbose=0)).reshape(-1)

        def _safe(s):
            for c in " /\\:;|()[]{}%": s = s.replace(c, "_")
            return s

        xr = _ref_vec(d["X_train_orig"], cfg.surface_hold_mode, cfg.surface_row_index)
        pairs = list(combinations(range(n_features), 2))[:cfg.surface_max_pairs]
        print(f"\nGenerating {len(pairs)} 3D surface(s)...")
        for i, j in pairs:
            xi = _grid(d["X_train_orig"], i, cfg.surface_grid_n,
                        cfg.surface_range_mode, cfg.surface_q_low, cfg.surface_q_high)
            xj = _grid(d["X_train_orig"], j, cfg.surface_grid_n,
                        cfg.surface_range_mode, cfg.surface_q_low, cfg.surface_q_high)
            XI, XJ = np.meshgrid(xi, xj)
            Xg = np.tile(xr.reshape(1, -1), (XI.size, 1))
            Xg[:, i] = XI.reshape(-1)
            Xg[:, j] = XJ.reshape(-1)
            Z = _pred_orig(Xg, model, scaler_X, scaler_y).reshape(XI.shape)
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0,
                            antialiased=True, alpha=0.92)
            if cfg.surface_overlay_scatter:
                zt = _pred_orig(d["X_train_orig"], model, scaler_X, scaler_y)
                ax.scatter(d["X_train_orig"][:, i], d["X_train_orig"][:, j], zt,
                           c="k", s=cfg.surface_scatter_size,
                           alpha=cfg.surface_scatter_alpha)
            ci, cj = str(input_columns[i]), str(input_columns[j])
            ax.set_title(f"{ci} vs {cj} ({phase_label})")
            ax.set_xlabel(ci); ax.set_ylabel(cj); ax.set_zlabel("Predicted")
            ax.view_init(elev=25, azim=-135); plt.tight_layout()
            plt.savefig(os.path.join(d3, f"surface_{_safe(ci)}_vs_{_safe(cj)}.png"),
                        dpi=cfg.surface_dpi)
            plt.show() if IN_NOTEBOOK else plt.close(fig)
        print(f"3D surfaces saved: {len([f for f in os.listdir(d3) if f.endswith('.png')])}")
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()

    return dict(y_train_inv=y_train_inv, y_val_inv=y_val_inv, y_test_inv=y_test_inv,
                y_train_pred=y_train_pred, y_val_pred=y_val_pred, y_test_pred=y_test_pred,
                m_train=m_train, m_val=m_val, m_test=m_test)


def make_zip(export_dir, zip_name):
    """Zip all files in export_dir."""
    files = []
    for root, dirs, fnames in os.walk(export_dir):
        for fn in fnames:
            files.append(os.path.join(root, fn))
    try:
        if os.path.exists(zip_name):
            os.remove(zip_name)
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in files:
                zf.write(fp, arcname=os.path.relpath(fp, "."))
        print(f"Created: {zip_name} ({os.path.getsize(zip_name)/1024/1024:.2f} MB)")
        if use_colab:
            print(f"Download via: from google.colab import files; "
                  f"files.download('{zip_name}')")
        else:
            dl = os.path.expanduser("~/Downloads"); os.makedirs(dl, exist_ok=True)
            shutil.copy2(zip_name, os.path.join(dl, zip_name))
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()


def build_l2sp_regularisers(arch, ptw, alpha):
    """Create L2-SP regulariser per Dense layer from stored weights."""
    regs = []
    layer_names = [f"dense_{i}" for i in range(len(arch))] + ["output"]
    for name in layer_names:
        if name in ptw:
            regs.append(L2SP(alpha=alpha, w0=ptw[name][0]))
        else:
            regs.append(L2SP(alpha=alpha, w0=None))
    return regs


def snapshot_weights(model):
    """Capture per-layer Dense weights and full model weights."""
    per_layer = {}
    for layer in model.layers:
        if isinstance(layer, layers.Dense) and layer.weights:
            per_layer[layer.name] = [w.numpy() for w in layer.weights]
    full = [np.array(w) for w in model.get_weights()]
    return per_layer, full


def run_three_stage_finetune(model, d, pretrained_weights):
    """Run 3-stage progressive unfreezing with L-BFGS-B."""
    n_transferred = 0
    for layer in model.layers:
        if layer.name in pretrained_weights:
            try:
                layer.set_weights(pretrained_weights[layer.name])
                n_transferred += 1
            except Exception as e:
                print(f"  Could not transfer {layer.name}: {e}")
    print(f"Transferred weights for {n_transferred} layer(s).")
    _eval_model(model, d["X_val"], d["y_val"], "after transfer, before fine-tune")

    # Stage 1: output-only
    print(f"\n--- Stage 1: output-only ({cfg.ft_stage1_epochs} ep, "
          f"LR={cfg.ft_stage1_lr}) ---")
    for layer in model.layers:
        if isinstance(layer, layers.Dense) and layer.name != "output":
            layer.trainable = False
        if isinstance(layer, layers.Dropout):
            layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.ft_stage1_lr),
                  loss="mse", metrics=["mae"])
    model.fit(d["X_train"], d["y_train"],
              validation_data=(d["X_val"], d["y_val"]),
              epochs=cfg.ft_stage1_epochs, batch_size=32,
              callbacks=[
                  callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                          restore_best_weights=True),
                  callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=4, min_lr=1e-5, verbose=1),
              ], verbose=1)
    _eval_model(model, d["X_val"], d["y_val"], "after Stage 1 (Adam)")
    model = lbfgs_refine(model, d["X_train"], d["y_train"],
                          X_va=d["X_val"], y_va=d["y_val"],
                          maxiter=cfg.lbfgs_maxiter)

    # Stage 2: unfreeze last N
    n_unfreeze = cfg.ft_stage2_unfreeze_last_n
    dense_names = [l.name for l in model.layers if isinstance(l, layers.Dense)]
    unfreeze_names = set(dense_names[-n_unfreeze:])
    print(f"\n--- Stage 2: unfreeze {unfreeze_names} ({cfg.ft_stage2_epochs} ep, "
          f"LR={cfg.ft_stage2_lr}) ---")

    for layer in model.layers:
        if layer.name in unfreeze_names:
            layer.trainable = True
        drop_name = layer.name.replace("dense_", "drop_")
        if isinstance(layer, layers.Dropout) and drop_name in unfreeze_names:
            layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.ft_stage2_lr),
                  loss="mse", metrics=["mae"])
    model.fit(d["X_train"], d["y_train"],
              validation_data=(d["X_val"], d["y_val"]),
              epochs=cfg.ft_stage2_epochs, batch_size=32,
              callbacks=[
                  callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                          restore_best_weights=True),
                  callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=5, min_lr=1e-5, verbose=1),
              ], verbose=1)
    _eval_model(model, d["X_val"], d["y_val"], "after Stage 2 (Adam)")
    model = lbfgs_refine(model, d["X_train"], d["y_train"],
                          X_va=d["X_val"], y_va=d["y_val"],
                          maxiter=cfg.lbfgs_maxiter)

    # Stage 3: full unfreeze
    print(f"\n--- Stage 3: full unfreeze ({cfg.ft_stage3_epochs} ep, "
          f"LR={cfg.ft_stage3_lr}) ---")
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.ft_stage3_lr),
                  loss="mse", metrics=["mae"])
    model.fit(d["X_train"], d["y_train"],
              validation_data=(d["X_val"], d["y_val"]),
              epochs=cfg.ft_stage3_epochs, batch_size=32,
              callbacks=[
                  callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                          restore_best_weights=True),
                  callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=5, min_lr=1e-6, verbose=1),
              ], verbose=1)
    _eval_model(model, d["X_val"], d["y_val"], "after Stage 3 (Adam)")
    model = lbfgs_refine(model, d["X_train"], d["y_train"],
                          X_va=d["X_val"], y_va=d["y_val"],
                          maxiter=cfg.lbfgs_maxiter)
    _eval_model(model, d["X_val"], d["y_val"], "after Stage 3 (L-BFGS-B, final)")

    return model, n_transferred


def run_cv(d, build_fn, best_hp, init_weights=None):
    """K-fold CV on TRAIN (no leakage). Returns (q2, r2_list, rmse_list, mae_list, oof_pred, oof_true)."""
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)
    r2_list, rmse_list, mae_list = [], [], []
    y_oof_pred = np.full(len(d["y_train_orig"]), np.nan)
    y_oof_true = np.full(len(d["y_train_orig"]), np.nan)

    print(f"{cfg.k_folds}-fold CV (no leakage)...")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(d["X_train_orig"]), start=1):
        X_tr = d["X_train_orig"][tr_idx]; X_va = d["X_train_orig"][va_idx]
        y_tr = d["y_train_orig"][tr_idx]; y_va = d["y_train_orig"][va_idx]

        fsx = StandardScaler().fit(X_tr); fsy = StandardScaler().fit(y_tr)
        X_tr_s, X_va_s = fsx.transform(X_tr), fsx.transform(X_va)
        y_tr_s, y_va_s = fsy.transform(y_tr), fsy.transform(y_va)

        mf = build_fn(best_hp)
        if init_weights is not None:
            try:
                mf.set_weights(init_weights)
            except Exception:
                pass
        mf.fit(X_tr_s, y_tr_s, validation_data=(X_va_s, y_va_s),
               epochs=cfg.cv_epochs, batch_size=32,
               callbacks=[
                   callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                           restore_best_weights=True),
                   callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                               patience=5, min_lr=1e-5, verbose=0),
               ], verbose=1)
        mf = lbfgs_refine(mf, X_tr_s, y_tr_s, X_va=X_va_s, y_va=y_va_s,
                           maxiter=cfg.lbfgs_maxiter, verbose=False)

        vp = fsy.inverse_transform(mf.predict(X_va_s, verbose=0)).reshape(-1)
        vt = y_va.reshape(-1)
        y_oof_pred[va_idx] = vp; y_oof_true[va_idx] = vt

        r2_list.append(r2_score(vt, vp))
        rmse_list.append(float(np.sqrt(mean_squared_error(vt, vp))))
        mae_list.append(float(np.mean(np.abs(vt - vp))))
        print(f"Fold {fold}: R2={r2_list[-1]:.4f}  RMSE={rmse_list[-1]:.4f}  MAE={mae_list[-1]:.4f}")

    ss_res = np.sum((y_oof_true - y_oof_pred)**2)
    ss_tot = np.sum((y_oof_true - np.mean(y_oof_true))**2)
    q2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    print(f"\nCV:  R2={np.mean(r2_list):.4f}+/-{np.std(r2_list):.4f}  "
          f"RMSE={np.mean(rmse_list):.4f}  MAE={np.mean(mae_list):.4f}  Q2={q2:.4f}")
    return q2, r2_list, rmse_list, mae_list, y_oof_pred, y_oof_true


def run_final_training(d, build_fn, best_hp, export_dir, init_weights=None):
    """Final training + optional retrain. Returns (model, history, best_epoch, d_updated)."""
    os.makedirs(export_dir, exist_ok=True)
    ckpt = os.path.join(export_dir, "best_model.keras")

    tf.keras.backend.clear_session()
    model = build_fn(best_hp)
    if init_weights is not None:
        try:
            model.set_weights(init_weights)
            print("Initialised from pre-trained weights.")
        except Exception as e:
            print(f"Could not init: {e}")

    history = model.fit(
        d["X_train"], d["y_train"],
        validation_data=(d["X_val"], d["y_val"]),
        epochs=cfg.final_epochs, batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                    restore_best_weights=True),
            callbacks.ModelCheckpoint(ckpt, monitor="val_loss",
                                      save_best_only=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=8, min_lr=1e-6, verbose=1),
        ], verbose=1,
    )

    if os.path.exists(ckpt):
        model = keras.models.load_model(ckpt, compile=False)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print(f"Best epoch: {best_epoch}")

    print("Refining with L-BFGS-B...")
    model.compile(loss="mse", metrics=["mae"])
    model = lbfgs_refine(model, d["X_train"], d["y_train"],
                          X_va=d["X_val"], y_va=d["y_val"],
                          maxiter=cfg.lbfgs_maxiter)

    if cfg.do_optional_retrain:
        print("\nRetrain on TRAIN+VAL.")
        X_tv = np.vstack([d["X_train_orig"], d["X_val_orig"]])
        y_tv = np.vstack([d["y_train_orig"], d["y_val_orig"]])
        sx_tv = StandardScaler().fit(X_tv)
        sy_tv = StandardScaler().fit(y_tv)

        tf.keras.backend.clear_session()
        model = build_fn(best_hp)
        if init_weights is not None:
            try:
                model.set_weights(init_weights)
            except Exception:
                pass
        X_tv_s = sx_tv.transform(X_tv); y_tv_s = sy_tv.transform(y_tv)
        model.fit(X_tv_s, y_tv_s, epochs=best_epoch, batch_size=32,
                  callbacks=[callbacks.ReduceLROnPlateau(
                      monitor="loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1)],
                  verbose=1)
        model = lbfgs_refine(model, X_tv_s, y_tv_s, maxiter=cfg.lbfgs_maxiter)

        d["scaler_X"] = sx_tv; d["scaler_y"] = sy_tv
        d["X_train"] = sx_tv.transform(d["X_train_orig"])
        d["X_val"] = sx_tv.transform(d["X_val_orig"])
        d["X_test"] = sx_tv.transform(d["X_test_orig"])
        d["y_train"] = sy_tv.transform(d["y_train_orig"])
        d["y_val"] = sy_tv.transform(d["y_val_orig"])
        d["y_test"] = sy_tv.transform(d["y_test_orig"])

        y_pred_rt = sy_tv.inverse_transform(
            model.predict(sx_tv.transform(d["X_test_orig"]), verbose=0)).reshape(-1)
        y_true_rt = d["y_test_orig"].reshape(-1)
        print(f"TEST (retrained): R2={r2_score(y_true_rt, y_pred_rt):.4f}  "
              f"RMSE={np.sqrt(mean_squared_error(y_true_rt, y_pred_rt)):.4f}")

    return model, history, best_epoch, d


# #############################################################################
#
#  HILL WARMUP — SYNTHETIC PRE-TRAINING
#
# #############################################################################
print("\n" + "=" * 80)
print("HILL WARMUP: SYNTHETIC DATA PRE-TRAINING")
print("=" * 80)


def _generate_hill_data(n_samples, n_features, v_max, k, n_hill, x_min, x_max):
    X = np.random.uniform(x_min, x_max, (n_samples, n_features)).astype(np.float32)
    x1 = X[:, 0]
    y = v_max * (x1 ** n_hill) / (k ** n_hill + x1 ** n_hill)
    y += np.random.normal(0, 0.05 * v_max, n_samples)
    for i in range(1, min(n_features, 3)):
        y += 0.05 * v_max * (X[:, i] - x_min) / (x_max - x_min)
    return X, y.reshape(-1, 1)


# Use the larger of n_inputs_a and n_inputs_b for synthetic features
hill_n_features = max(cfg.n_inputs_a, cfg.n_inputs_b)
X_h, y_h = _generate_hill_data(
    cfg.n_synthetic, hill_n_features, cfg.hill_v_max,
    cfg.hill_k, cfg.hill_n, cfg.hill_x_min, cfg.hill_x_max,
)

scaler_Xh = StandardScaler().fit(X_h)
scaler_yh = StandardScaler().fit(y_h)
X_h_scaled = scaler_Xh.transform(X_h)
y_h_scaled = scaler_yh.transform(y_h)

X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
    X_h_scaled, y_h_scaled, test_size=0.2, random_state=cfg.random_seed
)
print(f"Synthetic data: {X_h.shape[0]} samples, {X_h.shape[1]} features (standardised)")

# Tune on synthetic
print("\n" + "-" * 60)
print("HILL WARMUP: Tuning dropout/L2/LR (fixed architecture)")
print("-" * 60)

build_fn_hill = make_fixed_builder(hill_n_features, cfg.architecture)

tuner_hill = kt.RandomSearch(
    build_fn_hill, objective="val_loss",
    max_trials=cfg.pretrain_tuner_trials, executions_per_trial=1,
    directory="tuner_results", project_name="hill_warmup",
)
tuner_hill.search(
    X_h_train, y_h_train, validation_data=(X_h_val, y_h_val),
    epochs=min(cfg.tuner_epochs, 100), batch_size=32, verbose=1,
)

best_hp_hill = tuner_hill.get_best_hyperparameters(1)[0]
print("\nBest HPs (Hill warmup):")
for k_hp, v_hp in best_hp_hill.values.items():
    print(f"  {k_hp}: {v_hp}")

# Train warmup model
print("\n" + "-" * 60)
print("HILL WARMUP: Training warmup model")
print("-" * 60)

tf.keras.backend.clear_session()
warmup_model = build_fn_hill(best_hp_hill)

es_w = callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                restore_best_weights=True)
rlr_w = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=7, min_lr=1e-6, verbose=1)
warmup_model.fit(
    X_h_train, y_h_train, validation_data=(X_h_val, y_h_val),
    epochs=cfg.pretrain_epochs, batch_size=32,
    callbacks=[es_w, rlr_w], verbose=1,
)

warmup_val = warmup_model.evaluate(X_h_val, y_h_val, verbose=0)
print(f"\nWarmup (Adam):  val_loss={warmup_val[0]:.6f}  val_mae={warmup_val[1]:.6f}")

print("Refining warmup with L-BFGS-B...")
warmup_model = lbfgs_refine(warmup_model, X_h_train, y_h_train,
                             X_va=X_h_val, y_va=y_h_val,
                             maxiter=cfg.lbfgs_maxiter)

hill_weights, hill_full_weights = snapshot_weights(warmup_model)
print(f"Captured Hill warmup weights: {list(hill_weights.keys())}")

print("\n" + "=" * 80)
print("HILL WARMUP COMPLETE.")
print("=" * 80)


# #############################################################################
#
#  PHASE A: TRAIN ON DATASET 1
#
# #############################################################################
print("\n" + "=" * 80)
print("PHASE A: TRAINING ON DATASET 1")
print("=" * 80)

file_name_a, df_a = upload_dataset("Dataset 1 (Phase A) ")
labels_a, inputs_a, y_a, input_columns_a = select_columns(
    df_a, cfg.n_labels_a, cfg.n_inputs_a)
d_a = split_and_scale(inputs_a.values, y_a, labels_a, np.arange(len(df_a)))
n_features_a = d_a["X_train"].shape[1]

# Build L2-SP model using Hill warmup weights
l2sp_regs_a = build_l2sp_regularisers(cfg.architecture, hill_weights,
                                       cfg.l2sp_alpha)
build_fn_a_l2sp = make_fixed_builder(n_features_a, cfg.architecture,
                                      kernel_regularizers=l2sp_regs_a)
build_fn_a = make_fixed_builder(n_features_a, cfg.architecture)

# Tune on Dataset 1
print("\n" + "-" * 60)
print("PHASE A: Hyperparameter tuning (fixed architecture)")
print("-" * 60)

tuner_a = kt.RandomSearch(
    build_fn_a, objective="val_loss",
    max_trials=cfg.tuner_trials, executions_per_trial=1,
    directory="tuner_results", project_name="phase_a_dataset1",
)
tuner_a.search(d_a["X_train"], d_a["y_train"],
               validation_data=(d_a["X_val"], d_a["y_val"]),
               epochs=cfg.tuner_epochs, batch_size=32, verbose=1)

best_hp_a = tuner_a.get_best_hyperparameters(1)[0]
print("\nBest HPs (Dataset 1):")
for k_hp, v_hp in best_hp_a.values.items():
    print(f"  {k_hp}: {v_hp}")

# Three-stage fine-tuning with Hill warmup weights
print("\n" + "-" * 60)
print("PHASE A: Weight transfer from Hill warmup + 3-stage fine-tuning")
print("-" * 60)

tf.keras.backend.clear_session()
model_a_ft = build_fn_a_l2sp(best_hp_a)
model_a_ft, n_transferred_a = run_three_stage_finetune(
    model_a_ft, d_a, hill_weights)
ft_weights_a = [w.numpy() for w in model_a_ft.get_weights()]
print("\nPhase A fine-tuning complete.")

# CV on Dataset 1
print("\n" + "-" * 60)
print(f"PHASE A: {cfg.k_folds}-fold CV")
print("-" * 60)

q2_a, r2_a, rmse_a, mae_a, y_oof_pred_a, y_oof_true_a = run_cv(
    d_a, build_fn_a, best_hp_a, init_weights=ft_weights_a)

# Final training
print("\n" + "-" * 60)
print("PHASE A: Final training on Dataset 1")
print("-" * 60)

model_a, history_a, best_epoch_a, d_a = run_final_training(
    d_a, build_fn_a, best_hp_a, cfg.export_dir_a, init_weights=ft_weights_a)

# Diagnostics
diag_a = run_diagnostics(
    model_a, d_a, df_a, input_columns_a, cfg.export_dir_a,
    history_a, best_epoch_a, q2_a, best_hp_a, "Phase A (Dataset 1)",
    n_transferred=n_transferred_a)

# Snapshot Phase A weights for transfer to Phase B
phase_a_weights, phase_a_full_weights = snapshot_weights(model_a)
print(f"\nCaptured Phase A weights: {list(phase_a_weights.keys())}")
make_zip(cfg.export_dir_a, "dataset1_results.zip")

print("\n" + "=" * 80)
print("PHASE A COMPLETE.")
print("=" * 80)


# #############################################################################
#
#  PHASE B: FINE-TUNE ON DATASET 2
#
# #############################################################################
print("\n" + "=" * 80)
print("PHASE B: FINE-TUNING ON DATASET 2")
print("=" * 80)

file_name_b, df_b = upload_dataset("Dataset 2 (Phase B) ")
labels_b, inputs_b, y_b, input_columns_b = select_columns(
    df_b, cfg.n_labels_b, cfg.n_inputs_b)
d_b = split_and_scale(inputs_b.values, y_b, labels_b, np.arange(len(df_b)))
n_features_b = d_b["X_train"].shape[1]

# Build L2-SP model using Phase A weights
l2sp_regs_b = build_l2sp_regularisers(cfg.architecture, phase_a_weights,
                                       cfg.l2sp_alpha)
build_fn_b_l2sp = make_fixed_builder(n_features_b, cfg.architecture,
                                      kernel_regularizers=l2sp_regs_b)
build_fn_b = make_fixed_builder(n_features_b, cfg.architecture)

# Tune on Dataset 2
print("\n" + "-" * 60)
print("PHASE B: Hyperparameter tuning on Dataset 2")
print("-" * 60)

tuner_b = kt.RandomSearch(
    build_fn_b, objective="val_loss",
    max_trials=cfg.tuner_trials, executions_per_trial=1,
    directory="tuner_results", project_name="phase_b_dataset2",
)
tuner_b.search(d_b["X_train"], d_b["y_train"],
               validation_data=(d_b["X_val"], d_b["y_val"]),
               epochs=cfg.tuner_epochs, batch_size=32, verbose=1)

best_hp_b = tuner_b.get_best_hyperparameters(1)[0]
print("\nBest HPs (Dataset 2):")
for k_hp, v_hp in best_hp_b.values.items():
    print(f"  {k_hp}: {v_hp}")

# Three-stage fine-tuning with Phase A weights
print("\n" + "-" * 60)
print("PHASE B: Weight transfer from Phase A + 3-stage fine-tuning")
print("-" * 60)

tf.keras.backend.clear_session()
model_b_ft = build_fn_b_l2sp(best_hp_b)
model_b_ft, n_transferred_b = run_three_stage_finetune(
    model_b_ft, d_b, phase_a_weights)
ft_weights_b = [w.numpy() for w in model_b_ft.get_weights()]
print("\nPhase B fine-tuning complete.")

# CV on Dataset 2
print("\n" + "-" * 60)
print(f"PHASE B: {cfg.k_folds}-fold CV on Dataset 2")
print("-" * 60)

q2_b, r2_b, rmse_b, mae_b, y_oof_pred_b, y_oof_true_b = run_cv(
    d_b, build_fn_b, best_hp_b, init_weights=ft_weights_b)

# Final training
print("\n" + "-" * 60)
print("PHASE B: Final training on Dataset 2")
print("-" * 60)

model_b, history_b, best_epoch_b, d_b = run_final_training(
    d_b, build_fn_b, best_hp_b, cfg.export_dir_b, init_weights=ft_weights_b)

# Diagnostics
diag_b = run_diagnostics(
    model_b, d_b, df_b, input_columns_b, cfg.export_dir_b,
    history_b, best_epoch_b, q2_b, best_hp_b, "Phase B (Dataset 2)",
    n_transferred=n_transferred_b)

make_zip(cfg.export_dir_b, "dataset2_results.zip")

print("\n" + "=" * 80)
print("PHASE B COMPLETE.")
print("=" * 80)


# =============================================================================
#  NEW DATA PREDICTION (optional)
# =============================================================================
print("\n" + "=" * 80)
print("NEW DATA PREDICTION (optional)")
print("=" * 80)

model = model_b
scaler_X = d_b["scaler_X"]; scaler_y = d_b["scaler_y"]
export_dir = cfg.export_dir_b

if cfg.pi_calibration == "val":
    cal_res = (diag_b["y_val_inv"].reshape(-1)
               - diag_b["y_val_pred"].reshape(-1))
elif cfg.pi_calibration == "oof":
    cal_res = y_oof_true_b.reshape(-1) - y_oof_pred_b.reshape(-1)
else:
    raise ValueError(f"pi_calibration: '{cfg.pi_calibration}'")

q_low = float(np.quantile(cal_res, cfg.pi_alpha / 2.0))
q_high = float(np.quantile(cal_res, 1.0 - cfg.pi_alpha / 2.0))
print(f"PI ({cfg.pi_calibration}): q_low={q_low:.4f}  q_high={q_high:.4f}")

print("\nUpload new data (optional).")
new_data_loaded = False; new_df = None

if use_colab:
    try:
        uploaded_new = colab_files.upload()
        if uploaded_new:
            new_file_name = list(uploaded_new.keys())[0]
            new_df = load_table(new_file_name, sep=cfg.sep)
            new_data_loaded = True
    except Exception:
        pass
else:
    for c in ["new_data.tsv", "new_data.csv", "new_data.xlsx"]:
        if os.path.exists(c):
            new_df = load_table(c, sep=cfg.sep)
            new_data_loaded = True; break
    if not new_data_loaded:
        print("No new_data.* found. Skipping.")

if new_data_loaded and new_df is not None:
    try:
        _show(new_df.head())
        ncn = new_df.shape[1]
        nl = (new_df.iloc[:, :cfg.n_labels_b]
              if cfg.n_labels_b > 0 else pd.DataFrame())
        if ncn < cfg.n_labels_b + cfg.n_inputs_b:
            raise ValueError(f"Too few columns: {ncn}")
        ni = new_df.iloc[:, cfg.n_labels_b:cfg.n_labels_b + cfg.n_inputs_b]
        n_features_new = ni.shape[1]
        if n_features_new != n_features_b:
            if set(ni.columns).issubset(set(input_columns_b)):
                ni = ni[input_columns_b]
            else:
                raise ValueError("Column mismatch.")
        if ni.isna().sum().sum() > 0:
            ni = pd.DataFrame(
                SimpleImputer(strategy="mean").fit(d_b["X_train_orig"]).transform(ni),
                columns=ni.columns)
        nX = ni.values
        nyp = scaler_y.inverse_transform(
            model.predict(scaler_X.transform(nX), verbose=0)).reshape(-1)
        pil, piu = nyp + q_low, nyp + q_high
        comp = pd.DataFrame(index=range(len(nX)))
        if nl.shape[0] == len(nX):
            comp = pd.concat([comp, nl.reset_index(drop=True)], axis=1)
        comp = pd.concat([comp, ni.reset_index(drop=True)], axis=1)
        comp["Predicted_Value"] = nyp
        comp["PI_Lower_95%"] = pil; comp["PI_Upper_95%"] = piu
        comp["PI_Width"] = piu - pil
        comp["PI_Calibration_Source"] = cfg.pi_calibration

        has_actual = False
        if ncn > cfg.n_labels_b + cfg.n_inputs_b:
            try:
                ya = new_df.iloc[:, cfg.n_labels_b+cfg.n_inputs_b].values.reshape(-1, 1)
                if not np.isnan(ya).all():
                    if np.isnan(ya).any():
                        ya = SimpleImputer(strategy="mean").fit(
                            d_b["y_train_orig"]).transform(ya)
                    yf = ya.reshape(-1); has_actual = True
                    rn = yf - nyp; ae = np.abs(rn)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ap = np.where(np.abs(yf) > 1e-12, 100*ae/np.abs(yf), np.nan)
                    comp["Actual_Value"] = yf; comp["Prediction_Error"] = rn
                    comp["Abs_Error"] = ae; comp["Abs_Percent_Error"] = ap
                    wpi = (yf >= pil) & (yf <= piu)
                    comp["Within_95%_PI"] = wpi.astype(int)
                    nm = compute_basic_metrics(yf, nyp)
                    print("\nNew data metrics:")
                    for kk, vv in nm.items(): print(f"  {kk}: {vv}")
                    print(f"PI coverage: {wpi.sum()}/{len(wpi)} "
                          f"({100*wpi.sum()/len(wpi):.1f}%)")
            except Exception as exc:
                print(f"Could not extract target: {exc}")

        fn_new = new_df.copy()
        fn_new["Predicted_Value"] = nyp
        fn_new["PI_Lower_95%"] = pil; fn_new["PI_Upper_95%"] = piu
        fn_new["PI_Width"] = piu - pil
        if has_actual:
            fn_new["Actual_Value"] = yf; fn_new["Prediction_Error"] = rn
            fn_new["Abs_Error"] = ae; fn_new["Abs_Percent_Error"] = ap
            fn_new["Within_95%_PI"] = wpi.astype(int)

        comp.to_excel(os.path.join(export_dir, "new_data_predictions.xlsx"),
                      index=False, engine="openpyxl")
        comp.to_csv(os.path.join(export_dir, "new_data_predictions.csv"),
                    index=False)
        fn_new.to_excel(os.path.join(export_dir,
                        "new_data_full_with_all_columns.xlsx"),
                        index=False, engine="openpyxl")
        fn_new.to_csv(os.path.join(export_dir,
                      "new_data_full_with_all_columns.csv"), index=False)
        print("Predictions exported.")
        _show(comp.head(10))
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()

# Download new-data predictions
pf = [os.path.join(export_dir, f) for f in
      ["new_data_predictions.xlsx", "new_data_predictions.csv",
       "new_data_full_with_all_columns.xlsx",
       "new_data_full_with_all_columns.csv"]]
pf = [f for f in pf if os.path.exists(f)]
if pf:
    try:
        zp2 = "new_data_predictions.zip"
        if os.path.exists(zp2): os.remove(zp2)
        with zipfile.ZipFile(zp2, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in pf: zf.write(fp, arcname=os.path.relpath(fp, "."))
        print(f"Created: {zp2}")
        if use_colab:
            print(f"Download: from google.colab import files; "
                  f"files.download('{zp2}')")
        else:
            dl = os.path.expanduser("~/Downloads"); os.makedirs(dl, exist_ok=True)
            shutil.copy2(zp2, os.path.join(dl, zp2))
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()

print("\nAll done.")
