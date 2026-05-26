# =============================================================================
# Merged Pipeline: Hill Pre-Training -> Weight Transfer -> ANN-for-SIMS
#
# ZERO-LOSS WEIGHT TRANSFER DESIGN:
#
# 1. FIXED SHARED ARCHITECTURE — The same Dense layer sizes (256->512->128->
#    64->32->1) are used for the warmup model AND the real-data model.
#    The tuner only searches dropout, L2, and LR — never layer counts or
#    widths.  This guarantees 100 % weight-shape compatibility.
#
# 2. L2-SP REGULARISATION — During fine-tuning on real data, each Dense
#    layer's kernel is penalised toward the pre-trained (warmup) value:
#        loss += alpha * sum( ||W_i - W_i^pretrain||^2 )
#    This is the "Starting Point" variant of L2 (Li et al., 2018) that
#    prevents catastrophic forgetting while still allowing adaptation.
#
# 3. THREE-STAGE PROGRESSIVE UNFREEZING —
#    Stage 1 (output-only):  freeze every Dense except the output layer.
#    Stage 2 (last-N):       unfreeze the last N Dense layers.
#    Stage 3 (full):         unfreeze everything, very low LR + cosine decay.
#    Each stage uses ReduceLROnPlateau for adaptive scheduling.
#
# 4. UNIFIED STANDARDISATION — Every array entering the model is z-scored;
#    inverse-transform is applied only when producing human-readable outputs.
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
#  L2-SP REGULARISER  (penalise toward pre-trained weights, not toward zero)
# =============================================================================
class L2SP(keras.regularizers.Regularizer):
    """L2-SP: Starting-Point L2 regularisation (Li et al., 2018).

    Adds ``alpha * sum((w - w0)**2)`` where *w0* are the pre-trained
    (warmup) weights.  Falls back to standard L2 when *w0* is None.
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

    n_labels: int = 7
    n_inputs: int = 10
    target_col: Optional[int] = None
    sep: str = "\t"

    disable_gpu: bool = True

    # Fixed shared architecture  (warmup + main model)
    architecture: List[int] = field(default_factory=lambda: [256, 512, 128, 64, 32])

    # Synthetic pre-training
    n_synthetic: int = 500
    hill_v_max: float = 200.0
    hill_k: float = 0.10
    hill_n: float = 1.20
    hill_x_min: float = 0.1
    hill_x_max: float = 100.0
    pretrain_epochs: int = 100
    pretrain_tuner_trials: int = 15

    # Main tuning & training
    tuner_trials: int = 15
    k_folds: int = 15
    random_seed: int = 42
    tuner_epochs: int = 200
    cv_epochs: int = 200
    final_epochs: int = 200

    do_optional_retrain: bool = True

    # Three-stage fine-tuning schedule
    ft_stage1_epochs: int = 20
    ft_stage1_lr: float = 1e-3
    ft_stage2_epochs: int = 40
    ft_stage2_lr: float = 5e-4
    ft_stage2_unfreeze_last_n: int = 3
    ft_stage3_epochs: int = 40
    ft_stage3_lr: float = 1e-4

    # L2-SP strength during fine-tuning (0 = standard L2)
    l2sp_alpha: float = 1e-3

    pi_calibration: str = "val"
    pi_alpha: float = 0.05

    export_dir: str = "optimized_model"

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

np.random.seed(cfg.random_seed)
tf.random.set_seed(cfg.random_seed)

if cfg.disable_gpu:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("GPU disabled. Using CPU.")
    except Exception as e:
        print(f"Could not change GPU visibility: {e}")


# =============================================================================
#  FIXED-ARCHITECTURE MODEL BUILDER
#  Architecture is ALWAYS cfg.architecture; tuner only picks dropout / L2 / LR
# =============================================================================
def make_fixed_builder(n_feat, arch, kernel_regularizers=None):
    """Return a build function whose Dense widths are locked to *arch*.

    Parameters
    ----------
    kernel_regularizers : list[Regularizer] | None
        If supplied, one regulariser per Dense layer (len == len(arch)+1,
        the last being the output layer).  Used to inject L2-SP after
        pre-training.
    """
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(n_feat,)))
        l2_val = hp.Choice("l2_reg", [1e-4, 1e-3, 1e-2])
        for idx, units in enumerate(arch):
            if kernel_regularizers is not None:
                kreg = kernel_regularizers[idx]
            else:
                kreg = regularizers.l2(l2_val)
            model.add(layers.Dense(
                units, activation="relu", kernel_regularizer=kreg,
                name=f"dense_{idx}",
            ))
            model.add(layers.Dropout(
                hp.Float(f"dropout_{idx}", 0.0, 0.5, step=0.1),
                name=f"drop_{idx}",
            ))
        # Output layer
        out_kreg = (kernel_regularizers[-1]
                    if kernel_regularizers is not None else None)
        model.add(layers.Dense(1, activation="linear",
                               kernel_regularizer=out_kreg,
                               name="output"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", [1e-4, 5e-4, 1e-3, 5e-3])
            ),
            loss="mse", metrics=["mae"],
        )
        return model
    return build_model


# =============================================================================
#  COSINE DECAY SCHEDULE HELPER
# =============================================================================
def make_cosine_schedule(initial_lr, total_epochs, warmup_epochs=5):
    """Returns a LearningRateScheduler callback with linear warmup + cosine."""
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return initial_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
    return callbacks.LearningRateScheduler(schedule, verbose=0)


# ============================================================
# PHASE 1: SYNTHETIC DATA + STANDARDISATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: HILL SYNTHETIC DATA GENERATION")
print("=" * 70)


def _generate_hill_data(n_samples, n_features, v_max, k, n_hill, x_min, x_max):
    X = np.random.uniform(x_min, x_max, (n_samples, n_features)).astype(np.float32)
    x1 = X[:, 0]
    y = v_max * (x1 ** n_hill) / (k ** n_hill + x1 ** n_hill)
    y += np.random.normal(0, 0.05 * v_max, n_samples)
    for i in range(1, min(n_features, 3)):
        y += 0.05 * v_max * (X[:, i] - x_min) / (x_max - x_min)
    return X, y.reshape(-1, 1)


X_h, y_h = _generate_hill_data(
    cfg.n_synthetic, cfg.n_inputs, cfg.hill_v_max,
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


# ============================================================
# PHASE 2: TUNE DROPOUT / L2 / LR ON SYNTHETIC DATA
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: HYPERPARAMETER TUNING ON SYNTHETIC (fixed architecture)")
print("=" * 70)

build_fn_synth = make_fixed_builder(cfg.n_inputs, cfg.architecture)

tuner_synth = kt.RandomSearch(
    build_fn_synth,
    objective="val_loss",
    max_trials=cfg.pretrain_tuner_trials,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="ann_synth_fixed_arch",
)
tuner_synth.search(
    X_h_train, y_h_train,
    validation_data=(X_h_val, y_h_val),
    epochs=min(cfg.tuner_epochs, 100), batch_size=32, verbose=1,
)

best_hp_synth = tuner_synth.get_best_hyperparameters(1)[0]
print("\nBest HPs (synthetic, fixed arch):")
for k_hp, v_hp in best_hp_synth.values.items():
    print(f"  {k_hp}: {v_hp}")


# ============================================================
# PHASE 3: TRAIN WARMUP MODEL (full convergence)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: PRE-TRAINING WARMUP MODEL")
print("=" * 70)

tf.keras.backend.clear_session()
warmup_model = build_fn_synth(best_hp_synth)

es_w = callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
rlr_w = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
)
warmup_model.fit(
    X_h_train, y_h_train,
    validation_data=(X_h_val, y_h_val),
    epochs=cfg.pretrain_epochs, batch_size=32,
    callbacks=[es_w, rlr_w], verbose=1,
)

warmup_val = warmup_model.evaluate(X_h_val, y_h_val, verbose=0)
print(f"\nWarmup model converged.  val_loss={warmup_val[0]:.6f}  "
      f"val_mae={warmup_val[1]:.6f}")

# Snapshot pre-trained weights (needed for L2-SP)
pretrained_weights = {}
for layer in warmup_model.layers:
    if isinstance(layer, layers.Dense) and layer.weights:
        pretrained_weights[layer.name] = [w.numpy() for w in layer.weights]

print(f"Captured pre-trained weights for {len(pretrained_weights)} Dense layer(s): "
      f"{list(pretrained_weights.keys())}")


# =============================================================================
#  REAL DATA LOADING
# =============================================================================
print("\n" + "=" * 70)
print("LOADING REAL SIMS DATA")
print("=" * 70)


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


print("Upload your dataset.")
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


# =============================================================================
#  COLUMN SELECTION
# =============================================================================
n_cols = df.shape[1]
if cfg.target_col is not None:
    target_idx = cfg.target_col
    if target_idx < 0 or target_idx >= n_cols:
        raise IndexError("target_col out of range.")
    labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
    input_cols = list(range(cfg.n_labels, n_cols))
    input_cols.remove(target_idx)
    inputs_df = df.iloc[:, input_cols]
    y_full = df.iloc[:, target_idx].values.reshape(-1, 1)
elif cfg.n_labels + cfg.n_inputs >= n_cols:
    print("n_labels + n_inputs >= total columns. Using last column as target.")
    labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
    inputs_df = df.iloc[:, cfg.n_labels:-1]
    y_full = df.iloc[:, -1].values.reshape(-1, 1)
else:
    labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
    inputs_df = df.iloc[:, cfg.n_labels:cfg.n_labels + cfg.n_inputs]
    y_full = df.iloc[:, cfg.n_labels + cfg.n_inputs].values.reshape(-1, 1)

print(f"Labels: {labels_df.shape}, Inputs: {inputs_df.shape}, y: {y_full.shape}")

X_full = inputs_df.values
y_full_arr = y_full
labels_full = labels_df
input_columns = list(inputs_df.columns)
row_pos = np.arange(len(df))


# =============================================================================
#  TRAIN / VAL / TEST SPLIT
# =============================================================================
test_frac = cfg.test_percent / 100.0
val_frac = cfg.val_percent / 100.0
train_frac = cfg.train_percent / 100.0

X_temp, X_test_orig, y_temp, y_test_orig, lbl_temp, labels_test, idx_temp, idx_test = (
    train_test_split(X_full, y_full_arr, labels_full, row_pos,
                     test_size=test_frac, random_state=cfg.random_seed)
)
val_ratio = val_frac / (train_frac + val_frac)
(X_train_orig, X_val_orig, y_train_orig, y_val_orig,
 labels_train, labels_val, idx_train, idx_val) = (
    train_test_split(X_temp, y_temp, lbl_temp, idx_temp,
                     test_size=val_ratio, random_state=cfg.random_seed)
)

X_train_df = pd.DataFrame(X_train_orig, columns=input_columns).reset_index(drop=True)
X_val_df = pd.DataFrame(X_val_orig, columns=input_columns).reset_index(drop=True)
X_test_df = pd.DataFrame(X_test_orig, columns=input_columns).reset_index(drop=True)

n_total = len(df)
print(f"\nSplit sizes: train={len(X_train_orig)} val={len(X_val_orig)} "
      f"test={len(X_test_orig)}")


# =============================================================================
#  STANDARDISATION (fit on TRAIN only)
# =============================================================================
scaler_X = StandardScaler().fit(X_train_orig)
scaler_y = StandardScaler().fit(y_train_orig)

X_train = scaler_X.transform(X_train_orig)
X_val = scaler_X.transform(X_val_orig)
X_test = scaler_X.transform(X_test_orig)
y_train = scaler_y.transform(y_train_orig)
y_val = scaler_y.transform(y_val_orig)
y_test = scaler_y.transform(y_test_orig)

n_features = X_train.shape[1]
print(f"Standardised (fit on TRAIN). n_features={n_features}")


# =============================================================================
#  TUNE DROPOUT / L2 / LR ON REAL DATA  (architecture stays fixed)
# =============================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING ON REAL DATA (fixed architecture)")
print("=" * 70)

build_fn = make_fixed_builder(n_features, cfg.architecture)

tuner = kt.RandomSearch(
    build_fn,
    objective="val_loss",
    max_trials=cfg.tuner_trials,
    executions_per_trial=1,
    directory="tuner_results",
    project_name=f"ann_real_{cfg.train_percent}_{cfg.val_percent}_{cfg.test_percent}",
)
tuner.search(X_train, y_train, validation_data=(X_val, y_val),
             epochs=cfg.tuner_epochs, batch_size=32, verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest HPs (real data):")
for k_hp, v_hp in best_hp.values.items():
    print(f"  {k_hp}: {v_hp}")


# =============================================================================
#  BUILD L2-SP REGULARISED MODEL BUILDER
# =============================================================================
def build_l2sp_regularisers(arch, pretrained_w, alpha):
    """Create one L2SP regulariser per Dense layer using stored weights."""
    regs = []
    layer_names = [f"dense_{i}" for i in range(len(arch))] + ["output"]
    for name in layer_names:
        if name in pretrained_w:
            kernel_w0 = pretrained_w[name][0]
            regs.append(L2SP(alpha=alpha, w0=kernel_w0))
        else:
            regs.append(L2SP(alpha=alpha, w0=None))
    return regs


l2sp_regs = build_l2sp_regularisers(cfg.architecture, pretrained_weights,
                                     cfg.l2sp_alpha)
build_fn_l2sp = make_fixed_builder(n_features, cfg.architecture,
                                    kernel_regularizers=l2sp_regs)


# =============================================================================
#  THREE-STAGE PROGRESSIVE FINE-TUNING
# =============================================================================
print("\n" + "=" * 70)
print("WEIGHT TRANSFER + THREE-STAGE PROGRESSIVE FINE-TUNING")
print("=" * 70)


def _eval_model(mdl, X_v, y_v, label=""):
    loss, mae = mdl.evaluate(X_v, y_v, verbose=0)
    print(f"  [{label}] val_loss={loss:.6f}  val_mae={mae:.6f}")
    return loss


# --- Build real-data model with L2-SP and load warmup weights ----------------
tf.keras.backend.clear_session()
model_ft = build_fn_l2sp(best_hp)

# Transfer ALL weights from warmup (architecture is identical)
n_transferred = 0
for layer in model_ft.layers:
    if layer.name in pretrained_weights:
        try:
            layer.set_weights(pretrained_weights[layer.name])
            n_transferred += 1
        except Exception as e:
            print(f"  Could not transfer {layer.name}: {e}")

print(f"Transferred weights for {n_transferred} layer(s).")
_eval_model(model_ft, X_val, y_val, "after transfer, before fine-tune")

# --- Stage 1: Freeze all Dense except output ---------------------------------
print(f"\n--- Stage 1: output-only ({cfg.ft_stage1_epochs} epochs, "
      f"LR={cfg.ft_stage1_lr}) ---")
for layer in model_ft.layers:
    if isinstance(layer, layers.Dense) and layer.name != "output":
        layer.trainable = False
    if isinstance(layer, layers.Dropout):
        layer.trainable = False

model_ft.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cfg.ft_stage1_lr),
    loss="mse", metrics=["mae"],
)
model_ft.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=cfg.ft_stage1_epochs, batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=4, min_lr=1e-5, verbose=1),
    ],
    verbose=1,
)
_eval_model(model_ft, X_val, y_val, "after Stage 1")

# --- Stage 2: Unfreeze last N Dense layers ------------------------------------
n_unfreeze = cfg.ft_stage2_unfreeze_last_n
dense_names = [l.name for l in model_ft.layers if isinstance(l, layers.Dense)]
unfreeze_names = set(dense_names[-n_unfreeze:])
print(f"\n--- Stage 2: unfreeze {unfreeze_names} ({cfg.ft_stage2_epochs} epochs, "
      f"LR={cfg.ft_stage2_lr}) ---")

for layer in model_ft.layers:
    if layer.name in unfreeze_names:
        layer.trainable = True
    drop_name = layer.name.replace("dense_", "drop_")
    if isinstance(layer, layers.Dropout) and drop_name in unfreeze_names:
        layer.trainable = True

model_ft.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cfg.ft_stage2_lr),
    loss="mse", metrics=["mae"],
)
model_ft.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=cfg.ft_stage2_epochs, batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-5, verbose=1),
    ],
    verbose=1,
)
_eval_model(model_ft, X_val, y_val, "after Stage 2")

# --- Stage 3: Unfreeze everything, cosine decay, very low LR -----------------
print(f"\n--- Stage 3: full unfreeze ({cfg.ft_stage3_epochs} epochs, "
      f"LR={cfg.ft_stage3_lr}, cosine decay) ---")
for layer in model_ft.layers:
    layer.trainable = True

model_ft.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cfg.ft_stage3_lr),
    loss="mse", metrics=["mae"],
)
cosine_cb = make_cosine_schedule(cfg.ft_stage3_lr, cfg.ft_stage3_epochs,
                                  warmup_epochs=3)
model_ft.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=cfg.ft_stage3_epochs, batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                restore_best_weights=True),
        cosine_cb,
    ],
    verbose=1,
)
_eval_model(model_ft, X_val, y_val, "after Stage 3 (final)")

# The fine-tuned model becomes the pretrained initialiser for CV + final
_pretrained_model = model_ft
_pretrained_model_weights = [w.numpy() for w in _pretrained_model.get_weights()]

print("\n" + "=" * 70)
print("Fine-tuning complete. Pre-trained weights locked for CV & final.")
print("=" * 70 + "\n")


# =============================================================================
#  NO-LEAKAGE CV INSIDE TRAIN
# =============================================================================
kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)

r2_scores, rmse_scores, mae_scores = [], [], []
y_oof_pred_inv = np.full(len(y_train_orig), np.nan)
y_oof_true_inv = np.full(len(y_train_orig), np.nan)

print(f"{cfg.k_folds}-fold CV on TRAIN (fold-fitted scalers, no leakage)...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig), start=1):
    X_tr, X_va = X_train_orig[tr_idx], X_train_orig[va_idx]
    y_tr, y_va = y_train_orig[tr_idx], y_train_orig[va_idx]

    fold_sx = StandardScaler().fit(X_tr)
    fold_sy = StandardScaler().fit(y_tr)
    X_tr_s, X_va_s = fold_sx.transform(X_tr), fold_sx.transform(X_va)
    y_tr_s, y_va_s = fold_sy.transform(y_tr), fold_sy.transform(y_va)

    model_fold = build_fn(best_hp)
    try:
        model_fold.set_weights(_pretrained_model_weights)
    except Exception:
        pass

    model_fold.fit(
        X_tr_s, y_tr_s, validation_data=(X_va_s, y_va_s),
        epochs=cfg.cv_epochs, batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=5, min_lr=1e-5, verbose=0),
        ],
        verbose=1,
    )

    va_pred = fold_sy.inverse_transform(
        model_fold.predict(X_va_s, verbose=0)
    ).reshape(-1)
    va_true = y_va.reshape(-1)

    y_oof_pred_inv[va_idx] = va_pred
    y_oof_true_inv[va_idx] = va_true

    r2 = r2_score(va_true, va_pred)
    rmse = float(np.sqrt(mean_squared_error(va_true, va_pred)))
    mae = float(np.mean(np.abs(va_true - va_pred)))
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    print(f"Fold {fold}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

if np.isnan(y_oof_pred_inv).any():
    raise RuntimeError("OOF predictions contain NaNs.")

print(f"\nCV mean+/-std:  R2={np.mean(r2_scores):.4f}+/-{np.std(r2_scores):.4f}  "
      f"RMSE={np.mean(rmse_scores):.4f}+/-{np.std(rmse_scores):.4f}  "
      f"MAE={np.mean(mae_scores):.4f}+/-{np.std(mae_scores):.4f}")

ss_res = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
ss_tot = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
pred_R2_train = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
print(f"Predicted R2 (Q2): {pred_R2_train:.4f}")


# =============================================================================
#  FINAL TRAINING
# =============================================================================
print("\nFinal training: fit on TRAIN, validate on VAL.")

os.makedirs(cfg.export_dir, exist_ok=True)
ckpt_path = os.path.join(cfg.export_dir, "best_model.keras")

tf.keras.backend.clear_session()
model = build_fn(best_hp)
try:
    model.set_weights(_pretrained_model_weights)
    print("Initialised from fine-tuned pre-trained weights.")
except Exception as e:
    print(f"Could not init from pretrained: {e}")

history = model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=cfg.final_epochs, batch_size=32,
    callbacks=[
        callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                  save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=8, min_lr=1e-6, verbose=1),
    ],
    verbose=1,
)

if os.path.exists(ckpt_path):
    model = keras.models.load_model(ckpt_path, compile=False)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print(f"Best epoch (VAL loss): {best_epoch}")

# Evaluate on TEST
y_test_pred_eval = scaler_y.inverse_transform(
    model.predict(X_test, verbose=0)).reshape(-1)
y_test_inv_eval = scaler_y.inverse_transform(y_test).reshape(-1)
print(f"TEST: R2={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}  "
      f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}")


# =============================================================================
#  OPTIONAL RETRAIN ON TRAIN+VAL
# =============================================================================
if cfg.do_optional_retrain:
    print("\nRetrain on TRAIN+VAL, evaluate TEST.")
    X_tv = np.vstack([X_train_orig, X_val_orig])
    y_tv = np.vstack([y_train_orig, y_val_orig])

    scaler_X_tv = StandardScaler().fit(X_tv)
    scaler_y_tv = StandardScaler().fit(y_tv)

    tf.keras.backend.clear_session()
    model_rt = build_fn(best_hp)
    try:
        model_rt.set_weights(_pretrained_model_weights)
    except Exception:
        pass
    model_rt.fit(
        scaler_X_tv.transform(X_tv), scaler_y_tv.transform(y_tv),
        epochs=best_epoch, batch_size=32,
        callbacks=[callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1)],
        verbose=1,
    )

    y_pred_rt = scaler_y_tv.inverse_transform(
        model_rt.predict(scaler_X_tv.transform(X_test_orig), verbose=0)
    ).reshape(-1)
    y_true_rt = y_test_orig.reshape(-1)
    print(f"TEST (retrained): R2={r2_score(y_true_rt, y_pred_rt):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_true_rt, y_pred_rt)):.4f}  "
          f"MAE={np.mean(np.abs(y_true_rt - y_pred_rt)):.4f}")

    model = model_rt
    scaler_X = scaler_X_tv
    scaler_y = scaler_y_tv
    X_train = scaler_X.transform(X_train_orig)
    X_val = scaler_X.transform(X_val_orig)
    X_test = scaler_X.transform(X_test_orig)
    y_train = scaler_y.transform(y_train_orig)
    y_val = scaler_y.transform(y_val_orig)
    y_test = scaler_y.transform(y_test_orig)
    print("Using retrained model + train+val scalers.")


# =============================================================================
#  SAVE MODEL ARCHITECTURE
# =============================================================================
try:
    from tensorflow.keras.layers import Dense
    dense_layers = [l for l in model.layers if isinstance(l, Dense)]
    if dense_layers:
        print("\n=== Final Dense Layers ===")
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
            print(line)
            scheme_lines.append(line)
        with open(os.path.join(cfg.export_dir, "model_scheme.txt"),
                  "w", encoding="utf-8") as fh:
            fh.write("Final Dense Layers\n==================\n")
            for ln in scheme_lines:
                fh.write(ln + "\n")
except Exception as e:
    print(f"Error saving model scheme: {e}")


# =============================================================================
#  SAVE MODEL & SCALERS
# =============================================================================
model.save(os.path.join(cfg.export_dir, "final_model.keras"))
try:
    model.save(os.path.join(cfg.export_dir, "final_model.h5"))
except Exception:
    pass
joblib.dump(scaler_X, os.path.join(cfg.export_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(cfg.export_dir, "scaler_y.pkl"))
print(f"Model and scalers saved to {cfg.export_dir}/")


# =============================================================================
#  PREDICTIONS (inverse-transformed)
# =============================================================================
y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
y_val_pred = scaler_y.inverse_transform(model.predict(X_val, verbose=0))
y_test_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
y_train_inv = scaler_y.inverse_transform(y_train)
y_val_inv = scaler_y.inverse_transform(y_val)
y_test_inv = scaler_y.inverse_transform(y_test)


# =============================================================================
#  METRICS
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
    sep = float(np.sqrt(sse / (n - 1))) if n > 1 else np.nan
    mean_abs = float(np.mean(np.abs(a))) if n else np.nan
    mrpd = (100 * float(np.sum(np.abs(res))) / (n * mean_abs)
            if (n and not np.isclose(mean_abs, 0)) else np.nan)
    r2 = float(r2_score(a, p_)) if n else np.nan
    return {"n": n, "SSE": sse, "MSE": mse, "RMSE": rmse, "MAE": mae,
            "SEP": sep, "MRPD%": mrpd, "R2": r2}


def adj_r2(r2, n, p_):
    return 1 - (1 - r2) * (n - 1) / (n - p_ - 1) if (n - p_ - 1) > 0 else np.nan


m_train = compute_basic_metrics(y_train_inv, y_train_pred)
m_val = compute_basic_metrics(y_val_inv, y_val_pred)
m_test = compute_basic_metrics(y_test_inv, y_test_pred)
p = n_features
for m in (m_train, m_val, m_test):
    m["R2_adj"] = adj_r2(m["R2"], m["n"], p)
m_train["Q2"] = pred_R2_train
m_val["Q2"] = np.nan
m_test["Q2"] = np.nan

print("\n=== Final Metrics (inverse-transformed) ===")
for label, pct, m in [("Train", cfg.train_percent, m_train),
                       ("Val", cfg.val_percent, m_val),
                       ("Test", cfg.test_percent, m_test)]:
    print(f"{label} ({pct}%):")
    for kk, vv in m.items():
        print(f"  {kk}: {vv}")

stats_path = os.path.join(cfg.export_dir, "model_statistics.txt")
try:
    with open(stats_path, "w", encoding="utf-8") as fh:
        fh.write("Model statistics\n================\n\n")
        fh.write(f"Date: {_dt.date.today().isoformat()}\n")
        fh.write(f"Architecture: {cfg.architecture}\n")
        fh.write(f"Split: {cfg.train_percent}/{cfg.val_percent}/{cfg.test_percent}\n\n")
        fh.write("Best HPs (real):\n")
        for kk, vv in best_hp.values.items():
            fh.write(f"  {kk}: {vv}\n")
        fh.write(f"\nL2-SP alpha: {cfg.l2sp_alpha}\n")
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


# =============================================================================
#  PLOTS
# =============================================================================
try:
    mse_path = os.path.join(cfg.export_dir, "mse_evolution.png")
    loss = history.history.get("loss")
    val_loss = history.history.get("val_loss")
    if loss:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss)+1), loss, label="Train MSE", marker="o")
        if val_loss:
            plt.plot(range(1, len(val_loss)+1), val_loss, label="Val MSE", marker="o")
        plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("MSE Evolution")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(mse_path, dpi=150)
        plt.show() if IN_NOTEBOOK else plt.close()
except Exception as e:
    print(f"Error plotting MSE: {e}")


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


try:
    plots_dir = os.path.join(cfg.export_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for tag, yt, yp in [("train", y_train_inv, y_train_pred),
                         ("val", y_val_inv, y_val_pred),
                         ("test", y_test_inv, y_test_pred)]:
        plot_pred_vs_actual(yt, yp, f"Pred vs Actual ({tag.title()})",
                            os.path.join(plots_dir, f"pred_vs_actual_{tag}.png"))
        plot_residuals(yt, yp, tag.title(),
                       os.path.join(plots_dir, tag), standardize=True)
        plot_residuals(yt, yp, tag.title(),
                       os.path.join(plots_dir, tag), standardize=False)
    print(f"Diagnostic plots saved: {plots_dir}")
except Exception as e:
    print(f"Error creating plots: {e}")
    traceback.print_exc()


# =============================================================================
#  EXPORT DataFrames
# =============================================================================
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


results_train = make_export_df(labels_train.reset_index(drop=True),
                                X_train_df, y_train_inv, y_train_pred)
results_val = make_export_df(labels_val.reset_index(drop=True),
                              X_val_df, y_val_inv, y_val_pred)
results_test = make_export_df(labels_test.reset_index(drop=True),
                               X_test_df, y_test_inv, y_test_pred)

for name, rdf in [("train_predictions", results_train),
                   ("val_predictions", results_val),
                   ("test_predictions", results_test)]:
    rdf.to_excel(os.path.join(cfg.export_dir, f"{name}.xlsx"),
                 index=False, engine="openpyxl")
    rdf.to_csv(os.path.join(cfg.export_dir, f"{name}.csv"), index=False)

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

    for tag, idx, yt, yp in [("train", idx_train, y_train_inv, y_train_pred),
                              ("val", idx_val, y_val_inv, y_val_pred),
                              ("test", idx_test, y_test_inv, y_test_pred)]:
        fdf = _full_export(df, idx, yt, yp)
        fdf.to_excel(os.path.join(cfg.export_dir,
                     f"{tag}_full_with_all_columns.xlsx"),
                     index=False, engine="openpyxl")
        fdf.to_csv(os.path.join(cfg.export_dir,
                   f"{tag}_full_with_all_columns.csv"), index=False)
    print("Full-row exports saved.")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

_show(results_train.head())
_show(results_val.head())
_show(results_test.head())


# =============================================================================
#  3D SURFACE PLOTS
# =============================================================================
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


try:
    d3 = os.path.join(cfg.export_dir, "plots_3d")
    os.makedirs(d3, exist_ok=True)
    xr = _ref_vec(X_train_orig, cfg.surface_hold_mode, cfg.surface_row_index)
    pairs = list(combinations(range(n_features), 2))[:cfg.surface_max_pairs]
    print(f"\nGenerating {len(pairs)} 3D surface(s)...")
    for i, j in pairs:
        xi = _grid(X_train_orig, i, cfg.surface_grid_n,
                    cfg.surface_range_mode, cfg.surface_q_low, cfg.surface_q_high)
        xj = _grid(X_train_orig, j, cfg.surface_grid_n,
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
            zt = _pred_orig(X_train_orig, model, scaler_X, scaler_y)
            ax.scatter(X_train_orig[:, i], X_train_orig[:, j], zt,
                       c="k", s=cfg.surface_scatter_size,
                       alpha=cfg.surface_scatter_alpha)
        ci, cj = str(input_columns[i]), str(input_columns[j])
        ax.set_title(f"{ci} vs {cj} (hold={cfg.surface_hold_mode})")
        ax.set_xlabel(ci); ax.set_ylabel(cj); ax.set_zlabel("Predicted")
        ax.view_init(elev=25, azim=-135); plt.tight_layout()
        plt.savefig(os.path.join(d3, f"surface_{_safe(ci)}_vs_{_safe(cj)}.png"),
                    dpi=cfg.surface_dpi)
        plt.show() if IN_NOTEBOOK else plt.close(fig)
    print(f"3D surfaces saved: {len([f for f in os.listdir(d3) if f.endswith('.png')])}")
except Exception as e:
    print(f"Error: {e}"); traceback.print_exc()

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED!")
print("=" * 80)


# =============================================================================
#  DOWNLOAD #1: TRAINING RESULTS
# =============================================================================
print("\nDOWNLOAD #1: TRAINING RESULTS")
training_files = []
for fn in os.listdir(cfg.export_dir):
    fp = os.path.join(cfg.export_dir, fn)
    if os.path.isfile(fp): training_files.append(fp)
for sub in ("plots", "plots_3d"):
    d = os.path.join(cfg.export_dir, sub)
    if os.path.isdir(d):
        for fn in os.listdir(d): training_files.append(os.path.join(d, fn))
try:
    zp = "training_results.zip"
    if os.path.exists(zp): os.remove(zp)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in training_files:
            zf.write(fp, arcname=os.path.relpath(fp, "."))
    print(f"Created: {zp} ({os.path.getsize(zp)/1024/1024:.2f} MB)")
    if use_colab:
        print("Download via Colab sidebar or: "
              "from google.colab import files; files.download('training_results.zip')")
    else:
        dl = os.path.expanduser("~/Downloads"); os.makedirs(dl, exist_ok=True)
        shutil.copy2(zp, os.path.join(dl, zp))
except Exception as e:
    print(f"Error: {e}"); traceback.print_exc()


# =============================================================================
#  NEW DATA PREDICTION
# =============================================================================
print("\n" + "=" * 80)
print("NEW DATA PREDICTION")
print("=" * 80)

if cfg.pi_calibration == "val":
    cal_res = y_val_inv.reshape(-1) - y_val_pred.reshape(-1)
elif cfg.pi_calibration == "oof":
    cal_res = y_oof_true_inv.reshape(-1) - y_oof_pred_inv.reshape(-1)
else:
    raise ValueError(f"pi_calibration: '{cfg.pi_calibration}'")

q_low = float(np.quantile(cal_res, cfg.pi_alpha / 2.0))
q_high = float(np.quantile(cal_res, 1.0 - cfg.pi_alpha / 2.0))
print(f"PI ({cfg.pi_calibration}): q_low={q_low:.4f}  q_high={q_high:.4f}")

print("\nUpload new data (optional).")
new_data_loaded = False
new_df = None

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
        nl = new_df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
        if ncn < cfg.n_labels + cfg.n_inputs:
            raise ValueError(f"Too few columns: {ncn}")
        ni = new_df.iloc[:, cfg.n_labels:cfg.n_labels + cfg.n_inputs]
        if ni.shape[1] != n_features:
            if set(ni.columns).issubset(set(input_columns)):
                ni = ni[input_columns]
            else:
                raise ValueError("Column mismatch.")
        if ni.isna().sum().sum() > 0:
            ni = pd.DataFrame(
                SimpleImputer(strategy="mean").fit(X_train_orig).transform(ni),
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
        if ncn > cfg.n_labels + cfg.n_inputs:
            try:
                ya = new_df.iloc[:, cfg.n_labels+cfg.n_inputs].values.reshape(-1, 1)
                if not np.isnan(ya).all():
                    if np.isnan(ya).any():
                        ya = SimpleImputer(strategy="mean").fit(y_train_orig).transform(ya)
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

        comp.to_excel(os.path.join(cfg.export_dir, "new_data_predictions.xlsx"),
                      index=False, engine="openpyxl")
        comp.to_csv(os.path.join(cfg.export_dir, "new_data_predictions.csv"),
                    index=False)
        fn_new.to_excel(os.path.join(cfg.export_dir,
                        "new_data_full_with_all_columns.xlsx"),
                        index=False, engine="openpyxl")
        fn_new.to_csv(os.path.join(cfg.export_dir,
                      "new_data_full_with_all_columns.csv"), index=False)
        print("Predictions exported.")
        _show(comp.head(10))
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()

# DOWNLOAD #2
pf = [os.path.join(cfg.export_dir, f) for f in
      ["new_data_predictions.xlsx","new_data_predictions.csv",
       "new_data_full_with_all_columns.xlsx","new_data_full_with_all_columns.csv"]]
pf = [f for f in pf if os.path.exists(f)]
if pf:
    try:
        zp2 = "new_data_predictions.zip"
        if os.path.exists(zp2): os.remove(zp2)
        with zipfile.ZipFile(zp2, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in pf: zf.write(fp, arcname=os.path.relpath(fp, "."))
        print(f"Created: {zp2}")
        if use_colab:
            print("Download via sidebar or: "
                  "from google.colab import files; "
                  "files.download('new_data_predictions.zip')")
        else:
            dl = os.path.expanduser("~/Downloads"); os.makedirs(dl, exist_ok=True)
            shutil.copy2(zp2, os.path.join(dl, zp2))
    except Exception as e:
        print(f"Error: {e}"); traceback.print_exc()

print("\nAll done.")
