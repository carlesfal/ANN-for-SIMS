# =============================================================================
# Merged Pipeline: Hill Pre-Training -> Hyperparameter Tuning -> Weight Transfer
#                  (Frozen Layers + Low-LR Fine-Tuning)
#                                  +
#  ANN-for-SIMS Single-Cell Evaluator (No-Leakage)
#
#  KEY DESIGN:
#  - Single model builder used for BOTH synthetic pre-training and real data,
#    guaranteeing architecture compatibility for weight transfer.
#  - Tuner runs on synthetic data first to find architecture, then warmup model
#    trains with that exact architecture. Weights transfer 1:1 to the real model.
#  - Gradual unfreezing: freeze early layers, fine-tune deeper layers at low LR,
#    then unfreeze all at very low LR with ReduceLROnPlateau.
#  - All data (synthetic + real) standardised; inverse-transform at the end only.
# =============================================================================

# --- Dependency installation (Colab/Notebook/Script-friendly) ---
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

# --- All imports ---
import os
import shutil
import traceback
import zipfile
import datetime as _dt
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers
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
#  USER CONFIGURATION
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

    # Synthetic pre-training
    n_synthetic: int = 500
    hill_v_max: float = 200
    hill_k: float = 0.10
    hill_n: float = 1.20
    hill_x_min: float = 0.1
    hill_x_max: float = 100
    pretrain_epochs: int = 80
    pretrain_tuner_trials: int = 15

    # Main tuning & training
    tuner_trials: int = 15
    k_folds: int = 15
    random_seed: int = 42
    tuner_epochs: int = 200
    cv_epochs: int = 200
    final_epochs: int = 200

    do_optional_retrain: bool = True

    # Weight transfer & fine-tuning
    n_layers_to_freeze: int = 2
    finetune_lr: float = 5e-4
    finetune_epochs: int = 50
    unfreeze_all_after: bool = True
    unfreeze_lr: float = 1e-4
    unfreeze_epochs: int = 40

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

np.random.seed(cfg.random_seed)
tf.random.set_seed(cfg.random_seed)

if cfg.disable_gpu:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("GPU disabled. Using CPU.")
    except Exception as e:
        print(f"Could not change GPU visibility: {e}")


# =============================================================================
#  UNIFIED MODEL BUILDER (used for BOTH synthetic and real data)
# =============================================================================
def make_model_builder(n_feat):
    """Single model builder used across the entire pipeline."""
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(n_feat,)))
        n_layers = hp.Int("num_layers", 2, 6, step=1)
        l2_val = hp.Choice("l2_reg", [1e-4, 1e-3, 1e-2])
        for i in range(n_layers):
            units = hp.Int(f"units_{i}", 64, 512, step=64)
            model.add(layers.Dense(
                units, activation="relu",
                kernel_regularizer=regularizers.l2(l2_val),
            ))
            model.add(layers.Dropout(
                hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
            ))
        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", [1e-4, 5e-4, 1e-3, 5e-3])
            ),
            loss="mse", metrics=["mae"],
        )
        return model
    return build_model


# ============================================================
# PHASE 1: SYNTHETIC DATA GENERATION + STANDARDISATION
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

# Standardise synthetic data
scaler_Xh = StandardScaler().fit(X_h)
scaler_yh = StandardScaler().fit(y_h)
X_h_scaled = scaler_Xh.transform(X_h)
y_h_scaled = scaler_yh.transform(y_h)

X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
    X_h_scaled, y_h_scaled, test_size=0.2, random_state=cfg.random_seed
)

print(f"Synthetic data: {X_h.shape[0]} samples, {X_h.shape[1]} features")
print(f"Standardised: X mean~{X_h_scaled.mean():.4f}, std~{X_h_scaled.std():.4f}")


# ============================================================
# PHASE 2: TUNER ON SYNTHETIC DATA (finds best architecture)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: HYPERPARAMETER TUNING ON SYNTHETIC DATA")
print("=" * 70)

build_fn_synth = make_model_builder(cfg.n_inputs)

tuner_synth = kt.RandomSearch(
    build_fn_synth,
    objective="val_loss",
    max_trials=cfg.pretrain_tuner_trials,
    executions_per_trial=1,
    directory="tuner_results",
    project_name="ann_synthetic_arch_search",
)
print(f"Tuning on synthetic data ({cfg.pretrain_tuner_trials} trials)...")
tuner_synth.search(
    X_h_train, y_h_train,
    validation_data=(X_h_val, y_h_val),
    epochs=cfg.tuner_epochs, batch_size=32, verbose=1,
)

best_hp_synth = tuner_synth.get_best_hyperparameters(1)[0]
print("\nBest architecture (synthetic):")
for k, v in best_hp_synth.values.items():
    print(f"  {k}: {v}")


# ============================================================
# PHASE 3: TRAIN WARMUP MODEL WITH BEST ARCHITECTURE
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: PRE-TRAINING WARMUP MODEL (best architecture from Phase 2)")
print("=" * 70)

tf.keras.backend.clear_session()
warmup_model = build_fn_synth(best_hp_synth)

es_warmup = callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
rlr_warmup = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
)

history_warmup = warmup_model.fit(
    X_h_train, y_h_train,
    validation_data=(X_h_val, y_h_val),
    epochs=cfg.pretrain_epochs, batch_size=32,
    callbacks=[es_warmup, rlr_warmup], verbose=1,
)

warmup_val_loss = min(history_warmup.history["val_loss"])
print(f"\nPHASE 3 DONE: warmup model trained. Best val_loss={warmup_val_loss:.6f}")
print(f"Architecture: {best_hp_synth.values}")


# =============================================================================
#  DATA LOADING (Real SIMS data)
# =============================================================================
print("\n" + "=" * 70)
print("LOADING REAL SIMS DATA")
print("=" * 70)


def load_table(path, sep="\t"):
    _, ext = os.path.splitext(path.lower())
    if ext in (".xlsx", ".xls", ".xlsm"):
        print(f"Detected Excel file: {path}")
        return pd.read_excel(path, engine="openpyxl")
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            print(f"Trying read_csv: sep={sep!r} encoding={enc}")
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as e:
            last_err = e
    for enc in encodings:
        try:
            print(f"Fallback sniff: sep=None encoding={enc}")
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
print(f"Split %: train={100*len(X_train_orig)/n_total:.1f}%  "
      f"val={100*len(X_val_orig)/n_total:.1f}%  "
      f"test={100*len(X_test_orig)/n_total:.1f}%")


# =============================================================================
#  STANDARDISATION (fit on TRAIN only -- no leakage)
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
print(f"\nStandardisation (fit on TRAIN only):")
print(f"  X_train: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
print(f"  y_train: mean={y_train.mean():.6f}, std={y_train.std():.6f}")


# =============================================================================
#  KERAS-TUNER ON REAL DATA (same model builder as synthetic)
# =============================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING ON REAL DATA")
print("=" * 70)

build_fn = make_model_builder(n_features)

tuner = kt.RandomSearch(
    build_fn,
    objective="val_loss",
    max_trials=cfg.tuner_trials,
    executions_per_trial=1,
    directory="tuner_results",
    project_name=f"ann_{cfg.train_percent}_{cfg.val_percent}_{cfg.test_percent}",
)
print(f"Starting hyperparameter search ({cfg.tuner_trials} trials)...")
tuner.search(X_train, y_train, validation_data=(X_val, y_val),
             epochs=cfg.tuner_epochs, batch_size=32, verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest hyperparameters (real data):")
for k, v in best_hp.values.items():
    print(f"  {k}: {v}")


# =============================================================================
#  WEIGHT TRANSFER WITH PARTIAL FREEZING + GRADUAL UNFREEZE
# =============================================================================
print("\n" + "=" * 70)
print("WEIGHT TRANSFER: warmup -> real model (partial freeze + fine-tune)")
print("=" * 70)

_pretrained_model = None

tf.keras.backend.clear_session()
model_transfer = build_fn(best_hp)

# Get Dense layers from both models
warmup_dense = [l for l in warmup_model.layers if isinstance(l, layers.Dense)]
target_dense = [l for l in model_transfer.layers if isinstance(l, layers.Dense)]

n_transferred = 0
transferred_layer_names = []

for lw, lt in zip(warmup_dense, target_dense):
    if not lw.weights or not lt.weights:
        continue
    if len(lw.weights) != len(lt.weights):
        print(f"  Skip {lw.name}->{lt.name}: tensor count mismatch")
        continue
    shapes_ok = all(
        ws.shape == wt.shape for ws, wt in zip(lw.weights, lt.weights)
    )
    if shapes_ok:
        lt.set_weights(lw.get_weights())
        n_transferred += 1
        transferred_layer_names.append(lt.name)
        print(f"  Transferred: {lw.name} -> {lt.name} "
              f"(shapes: {[w.shape for w in lt.weights]})")
    else:
        print(f"  Skip {lw.name}->{lt.name}: shape mismatch "
              f"({[w.shape for w in lw.weights]} vs {[w.shape for w in lt.weights]})")

print(f"\nTotal layers transferred: {n_transferred}/{len(target_dense)}")

# Freeze the first N transferred layers (not all)
n_to_freeze = min(cfg.n_layers_to_freeze, n_transferred)
frozen_names = []
if n_to_freeze > 0 and n_transferred > 0:
    freeze_count = 0
    for layer in model_transfer.layers:
        if isinstance(layer, layers.Dense) and layer.name in transferred_layer_names:
            if freeze_count < n_to_freeze:
                layer.trainable = False
                frozen_names.append(layer.name)
                freeze_count += 1
    print(f"Frozen {len(frozen_names)} early layer(s): {frozen_names}")

if n_transferred > 0:
    _pretrained_model = model_transfer
    print("Weight transfer successful.")
else:
    print("No weights transferred. Model will train from scratch.")
    _pretrained_model = None

# ── Fine-tune phase 1: low LR, frozen early layers ──────────────────────────
if _pretrained_model is not None and frozen_names:
    print(f"\nFine-tune Phase 1: {cfg.finetune_epochs} epochs, "
          f"LR={cfg.finetune_lr} (early layers frozen)")
    _pretrained_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.finetune_lr),
        loss="mse", metrics=["mae"],
    )
    es_ft1 = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    rlr_ft1 = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
    )
    _pretrained_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.finetune_epochs, batch_size=32,
        callbacks=[es_ft1, rlr_ft1], verbose=1,
    )

    # ── Fine-tune phase 2: unfreeze all layers, very low LR ─────────────────
    if cfg.unfreeze_all_after:
        print(f"\nFine-tune Phase 2: unfreeze all, {cfg.unfreeze_epochs} epochs, "
              f"LR={cfg.unfreeze_lr}")
        for layer in _pretrained_model.layers:
            layer.trainable = True
        _pretrained_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.unfreeze_lr),
            loss="mse", metrics=["mae"],
        )
        es_ft2 = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        rlr_ft2 = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
        _pretrained_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.unfreeze_epochs, batch_size=32,
            callbacks=[es_ft2, rlr_ft2], verbose=1,
        )
    print("Fine-tuning complete.\n")

print("=" * 70 + "\n")


# =============================================================================
#  STRICT NO-LEAKAGE CV INSIDE TRAIN
# =============================================================================
kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)

r2_scores, rmse_scores, mae_scores = [], [], []
y_oof_pred_inv = np.full(len(y_train_orig), np.nan)
y_oof_true_inv = np.full(len(y_train_orig), np.nan)

print(f"{cfg.k_folds}-fold CV on TRAIN with fold-fitted scalers (no leakage)...")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig), start=1):
    X_tr, X_va = X_train_orig[tr_idx], X_train_orig[va_idx]
    y_tr, y_va = y_train_orig[tr_idx], y_train_orig[va_idx]

    fold_sx = StandardScaler().fit(X_tr)
    fold_sy = StandardScaler().fit(y_tr)

    X_tr_s, X_va_s = fold_sx.transform(X_tr), fold_sx.transform(X_va)
    y_tr_s, y_va_s = fold_sy.transform(y_tr), fold_sy.transform(y_va)

    model_fold = build_fn(best_hp)
    if _pretrained_model is not None:
        try:
            model_fold.set_weights(_pretrained_model.get_weights())
        except Exception:
            pass

    es_cv = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    rlr_cv = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=0
    )
    model_fold.fit(
        X_tr_s, y_tr_s,
        validation_data=(X_va_s, y_va_s),
        epochs=cfg.cv_epochs, batch_size=32,
        callbacks=[es_cv, rlr_cv], verbose=1,
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

print(f"\nCV (mean +/- std):  R2={np.mean(r2_scores):.4f}+/-{np.std(r2_scores):.4f}  "
      f"RMSE={np.mean(rmse_scores):.4f}+/-{np.std(rmse_scores):.4f}  "
      f"MAE={np.mean(mae_scores):.4f}+/-{np.std(mae_scores):.4f}")

ss_res = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
ss_tot = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
pred_R2_train = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
print(f"Predicted R2 (Q2, OOF/PRESS): {pred_R2_train:.4f}")


# =============================================================================
#  FINAL TRAINING (TRAIN -> validate on VAL)
# =============================================================================
print("\nFinal training: fit on TRAIN (standardised), validate on VAL.")

os.makedirs(cfg.export_dir, exist_ok=True)
ckpt_path = os.path.join(cfg.export_dir, "best_model.keras")

mc = callbacks.ModelCheckpoint(
    ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
)
es_final = callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)
rlr_final = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
)

tf.keras.backend.clear_session()
model = build_fn(best_hp)

if _pretrained_model is not None:
    try:
        model.set_weights(_pretrained_model.get_weights())
        print("Initialised final model with fine-tuned pre-trained weights.")
    except Exception as e:
        print(f"Could not init from pretrained: {e}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=cfg.final_epochs, batch_size=32,
    callbacks=[es_final, mc, rlr_final], verbose=1,
)

if os.path.exists(ckpt_path):
    model = keras.models.load_model(ckpt_path, compile=False)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print(f"Best epoch (by VAL loss): {best_epoch}")

# Evaluate on TEST (inverse-transformed)
print("\nEvaluating on TEST (inverse-transformed).")
y_test_pred_eval = scaler_y.inverse_transform(
    model.predict(X_test, verbose=0)
).reshape(-1)
y_test_inv_eval = scaler_y.inverse_transform(y_test).reshape(-1)
print(f"TEST: R2={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}  "
      f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}")


# =============================================================================
#  OPTIONAL RETRAIN ON TRAIN+VAL
# =============================================================================
if cfg.do_optional_retrain:
    print("\nOptional retrain: refit scalers on TRAIN+VAL, retrain, evaluate TEST.")
    X_tv = np.vstack([X_train_orig, X_val_orig])
    y_tv = np.vstack([y_train_orig, y_val_orig])

    scaler_X_tv = StandardScaler().fit(X_tv)
    scaler_y_tv = StandardScaler().fit(y_tv)

    X_tv_s = scaler_X_tv.transform(X_tv)
    y_tv_s = scaler_y_tv.transform(y_tv)

    tf.keras.backend.clear_session()
    model_rt = build_fn(best_hp)
    if _pretrained_model is not None:
        try:
            model_rt.set_weights(_pretrained_model.get_weights())
        except Exception:
            pass

    rlr_rt = callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
    )
    model_rt.fit(X_tv_s, y_tv_s, epochs=best_epoch, batch_size=32,
                 callbacks=[rlr_rt], verbose=1)

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
    print("Using retrained model + train+val-fitted scalers.")


# =============================================================================
#  SAVE MODEL ARCHITECTURE
# =============================================================================
try:
    from tensorflow.keras.layers import Dense
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    if dense_layers:
        print("\n=== Final Dense Layers ===")
        scheme_lines = []
        for i, layer in enumerate(dense_layers, start=1):
            act_name = layer.activation.__name__ if layer.activation else "N/A"
            reg = layer.kernel_regularizer
            reg_str = ""
            if reg is not None:
                try:
                    reg_str = (f", regularizer=L2={reg.l2}"
                               if hasattr(reg, "l2") else f", regularizer={reg}")
                except Exception:
                    reg_str = f", regularizer={reg}"
            line = (f"Layer {i}: name='{layer.name}', units={layer.units}, "
                    f"activation={act_name}{reg_str}")
            print(line)
            scheme_lines.append(line)
        scheme_path = os.path.join(cfg.export_dir, "model_scheme.txt")
        with open(scheme_path, "w", encoding="utf-8") as fh:
            fh.write("Final Dense Layers (in model order)\n"
                     "===================================\n")
            for ln in scheme_lines:
                fh.write(ln + "\n")
        print(f"Model scheme saved to: {scheme_path}")
except Exception as e:
    print(f"Error saving model scheme: {e}")


# =============================================================================
#  SAVE MODEL & SCALERS
# =============================================================================
model.save(os.path.join(cfg.export_dir, "final_model.keras"))
try:
    model.save(os.path.join(cfg.export_dir, "final_model.h5"))
except Exception as e:
    print(f"Could not save .h5 (ok to ignore): {e}")

joblib.dump(scaler_X, os.path.join(cfg.export_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(cfg.export_dir, "scaler_y.pkl"))
print(f"Model and scalers saved to {cfg.export_dir}/")


# =============================================================================
#  PREDICTIONS (inverse-transformed for all results)
# =============================================================================
y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
y_val_pred = scaler_y.inverse_transform(model.predict(X_val, verbose=0))
y_test_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
y_train_inv = scaler_y.inverse_transform(y_train)
y_val_inv = scaler_y.inverse_transform(y_val)
y_test_inv = scaler_y.inverse_transform(y_test)


# =============================================================================
#  METRICS (all on inverse-transformed values)
# =============================================================================
def compute_basic_metrics(y_true, y_pred):
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
    mrpd = (100 * float(np.sum(np.abs(res))) / (n * mean_abs)
            if (n and not np.isclose(mean_abs, 0)) else np.nan)
    r2 = float(r2_score(actual, pred)) if n else np.nan
    return {"n": n, "SSE": sse, "MSE": mse, "RMSE": rmse, "MAE": mae,
            "SEP": sep, "MRPD_percent": mrpd, "R2": r2}


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else np.nan


m_train = compute_basic_metrics(y_train_inv, y_train_pred)
m_val = compute_basic_metrics(y_val_inv, y_val_pred)
m_test = compute_basic_metrics(y_test_inv, y_test_pred)

p = n_features
m_train["R2_adj"] = adjusted_r2(m_train["R2"], m_train["n"], p)
m_val["R2_adj"] = adjusted_r2(m_val["R2"], m_val["n"], p)
m_test["R2_adj"] = adjusted_r2(m_test["R2"], m_test["n"], p)
m_train["Predicted_R2_Q2"] = pred_R2_train
m_val["Predicted_R2_Q2"] = np.nan
m_test["Predicted_R2_Q2"] = np.nan

print("\n=== Final Metrics (inverse-transformed) ===")
for label, pct, m in [("Training", cfg.train_percent, m_train),
                       ("Validation", cfg.val_percent, m_val),
                       ("Test", cfg.test_percent, m_test)]:
    print(f"{label} set ({pct}%):")
    for k, v in m.items():
        print(f"  {k}: {v}")

# Save statistics
stats_path = os.path.join(cfg.export_dir, "model_statistics.txt")
try:
    with open(stats_path, "w", encoding="utf-8") as fh:
        fh.write("Model statistics summary\n========================\n\n")
        fh.write(f"Current date: {_dt.date.today().isoformat()}\n")
        fh.write(f"Number of predictors (p): {p}\n")
        fh.write(f"Data split: {cfg.train_percent}% / {cfg.val_percent}% / "
                 f"{cfg.test_percent}%\n\n")
        fh.write("Hyperparameters (best, real data):\n")
        for k, v in best_hp.values.items():
            fh.write(f"  {k}: {v}\n")
        fh.write(f"\nWeight transfer: n_frozen={cfg.n_layers_to_freeze}, "
                 f"finetune_lr={cfg.finetune_lr}, unfreeze_lr={cfg.unfreeze_lr}\n")
        fh.write(f"Layers transferred: {n_transferred}, frozen: {frozen_names}\n")
        for label, pct, m in [("Training", cfg.train_percent, m_train),
                               ("Validation", cfg.val_percent, m_val),
                               ("Test", cfg.test_percent, m_test)]:
            fh.write(f"\n{label} set metrics ({pct}%):\n")
            for k, v in m.items():
                fh.write(f"  {k}: {v}\n")
        fh.write(f"\nPredicted R2 (Q2, OOF, no leakage): {pred_R2_train}\n")
        fh.write(f"Best epoch (VAL loss): {best_epoch}\n")
        fh.write(f"Optional retrain: {cfg.do_optional_retrain}\n")
        fh.write(f"PI calibration: {cfg.pi_calibration}, alpha={cfg.pi_alpha}\n")
    print(f"Statistics saved to: {stats_path}")
except Exception as e:
    print(f"Could not save statistics: {e}")


# =============================================================================
#  PLOTS: MSE EVOLUTION
# =============================================================================
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
except Exception as e:
    print(f"Error plotting MSE: {e}")


# =============================================================================
#  PLOTS: DIAGNOSTICS
# =============================================================================
def plot_pred_vs_actual(y_true, y_pred, title, save_path=None):
    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_t, y_p, alpha=0.6)
    lo = min(y_t.min(), y_p.min())
    hi = max(y_t.max(), y_p.max())
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


def plot_residuals(y_true, y_pred, prefix, save_prefix=None, standardize=False):
    res = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)
    if standardize:
        res = (res - np.mean(res)) / (np.std(res) + 1e-12)
    color = "gray" if standardize else "salmon"
    label = "Standardized residuals" if standardize else "Raw residuals"
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(res, bins=25, kde=True, color=color, edgecolor="black")
    plt.title(f"{prefix} -- {label} distribution")
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(res, marker="o", linestyle="-")
    plt.title(f"{prefix} -- {label} (time series)")
    plt.xlabel("Index")
    plt.ylabel(label)
    plt.grid(True)
    plt.tight_layout()
    suffix = ("_std_residuals_hist_ts.png" if standardize
              else "_raw_residuals_hist_ts.png")
    if save_prefix:
        plt.savefig(save_prefix + suffix, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


try:
    plots_dir = os.path.join(cfg.export_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for tag, yt, yp in [("train", y_train_inv, y_train_pred),
                         ("val", y_val_inv, y_val_pred),
                         ("test", y_test_inv, y_test_pred)]:
        label = tag.capitalize()
        plot_pred_vs_actual(
            yt, yp, f"Predicted vs Actual ({label})",
            save_path=os.path.join(plots_dir, f"pred_vs_actual_{tag}.png"),
        )
        plot_residuals(yt, yp, label,
                       save_prefix=os.path.join(plots_dir, f"{tag}"),
                       standardize=True)
        plot_residuals(yt, yp, label,
                       save_prefix=os.path.join(plots_dir, f"{tag}"),
                       standardize=False)
    print(f"Diagnostic plots saved to: {plots_dir}")
except Exception as e:
    print(f"Error creating diagnostic plots: {e}")
    traceback.print_exc()


# =============================================================================
#  EXPORT DataFrames (inverse-transformed)
# =============================================================================
def make_export_df(lbl, inp, y_true, y_pred):
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


results_train = make_export_df(
    labels_train.reset_index(drop=True), X_train_df, y_train_inv, y_train_pred
)
results_val = make_export_df(
    labels_val.reset_index(drop=True), X_val_df, y_val_inv, y_val_pred
)
results_test = make_export_df(
    labels_test.reset_index(drop=True), X_test_df, y_test_inv, y_test_pred
)

for name, res_df in [("train_predictions", results_train),
                      ("val_predictions", results_val),
                      ("test_predictions", results_test)]:
    res_df.to_excel(
        os.path.join(cfg.export_dir, f"{name}.xlsx"),
        index=False, engine="openpyxl",
    )
    res_df.to_csv(os.path.join(cfg.export_dir, f"{name}.csv"), index=False)

# Full-row exports
try:
    def _full_row_export(orig_df, indices, y_t, y_p):
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
            "Actual_Value", "Predicted_Value", "Residual",
            "Abs_Error", "Abs_Percent_Error",
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
            index=False, engine="openpyxl",
        )
        full_df.to_csv(
            os.path.join(cfg.export_dir, f"{tag}_full_with_all_columns.csv"),
            index=False,
        )
    print("Full-row exports saved.")
except Exception as e:
    print(f"Error creating full-row exports: {e}")
    traceback.print_exc()

_show(results_train.head())
_show(results_val.head())
_show(results_test.head())


# =============================================================================
#  3D SURFACE PLOTS
# =============================================================================
def _build_ref_vector(X_orig, hold_mode, row_idx=0):
    if hold_mode == "median_train":
        return np.median(X_orig, axis=0)
    if hold_mode == "mean_train":
        return np.mean(X_orig, axis=0)
    if hold_mode == "row":
        return X_orig[int(row_idx)].copy()
    raise ValueError(
        f"hold_mode must be 'median_train', 'mean_train', or 'row', "
        f"got '{hold_mode}'"
    )


def _grid_vals(X_orig, col, grid_n, mode, q_lo, q_hi):
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


def _predict_orig(X_orig, mdl, sx, sy):
    return sy.inverse_transform(
        mdl.predict(sx.transform(X_orig), verbose=0)
    ).reshape(-1)


def _safe_name(s):
    for ch in " /\\:;|()[]{}%":
        s = s.replace(ch, "_")
    return s


try:
    plots_3d = os.path.join(cfg.export_dir, "plots_3d")
    os.makedirs(plots_3d, exist_ok=True)

    xref = _build_ref_vector(
        X_train_orig, cfg.surface_hold_mode, cfg.surface_row_index
    )
    pairs = list(combinations(range(n_features), 2))[:cfg.surface_max_pairs]
    print(f"\nGenerating {len(pairs)} 3D surface(s)...")

    for i, j in pairs:
        xi = _grid_vals(
            X_train_orig, i, cfg.surface_grid_n,
            cfg.surface_range_mode, cfg.surface_q_low, cfg.surface_q_high,
        )
        xj = _grid_vals(
            X_train_orig, j, cfg.surface_grid_n,
            cfg.surface_range_mode, cfg.surface_q_low, cfg.surface_q_high,
        )
        XI, XJ = np.meshgrid(xi, xj)
        Xgrid = np.tile(xref.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)
        Z = _predict_orig(Xgrid, model, scaler_X, scaler_y).reshape(XI.shape)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            XI, XJ, Z, cmap="viridis", linewidth=0,
            antialiased=True, alpha=0.92,
        )
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        if cfg.surface_overlay_scatter:
            z_tr = _predict_orig(X_train_orig, model, scaler_X, scaler_y)
            ax.scatter(
                X_train_orig[:, i], X_train_orig[:, j], z_tr,
                c="k", s=cfg.surface_scatter_size,
                alpha=cfg.surface_scatter_alpha,
            )

        ci, cj = str(input_columns[i]), str(input_columns[j])
        ax.set_title(
            f"Predicted: {ci} vs {cj}\n(others held: {cfg.surface_hold_mode})"
        )
        ax.set_xlabel(ci)
        ax.set_ylabel(cj)
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                plots_3d,
                f"surface_{_safe_name(ci)}_vs_{_safe_name(cj)}.png",
            ),
            dpi=cfg.surface_dpi,
        )
        if IN_NOTEBOOK:
            plt.show()
        else:
            plt.close(fig)

    n_saved = len([f for f in os.listdir(plots_3d) if f.endswith(".png")])
    print(f"3D surfaces saved: {n_saved}")
except Exception as e:
    print(f"Error generating 3D surfaces: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED!")
print("=" * 80)


# =============================================================================
#  DOWNLOAD #1: TRAINING RESULTS
# =============================================================================
print("\nDOWNLOAD #1: TRAINING RESULTS")
print("-" * 80)

training_files = []
for fn in os.listdir(cfg.export_dir):
    fp = os.path.join(cfg.export_dir, fn)
    if os.path.isfile(fp):
        training_files.append(fp)
for subdir in ("plots", "plots_3d"):
    d = os.path.join(cfg.export_dir, subdir)
    if os.path.isdir(d):
        for fn in os.listdir(d):
            training_files.append(os.path.join(d, fn))

try:
    zip_train = "training_results.zip"
    if os.path.exists(zip_train):
        os.remove(zip_train)
    with zipfile.ZipFile(zip_train, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in training_files:
            zf.write(fp, arcname=os.path.relpath(fp, "."))
    size_mb = os.path.getsize(zip_train) / (1024 * 1024)
    print(f"Created: {zip_train} ({size_mb:.2f} MB, {len(training_files)} files)")

    if use_colab:
        print("Find 'training_results.zip' in Colab Files sidebar -> "
              "right-click -> Download")
        print("Or run:  from google.colab import files; "
              "files.download('training_results.zip')")
    else:
        dl = os.path.expanduser("~/Downloads")
        os.makedirs(dl, exist_ok=True)
        shutil.copy2(zip_train, os.path.join(dl, zip_train))
        print(f"Saved to {os.path.join(dl, zip_train)}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()


# =============================================================================
#  NEW DATA PREDICTION (standardise -> predict -> inverse-transform)
# =============================================================================
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
print(f"Empirical PI ({cfg.pi_calibration}): q_low={q_low:.4f}  "
      f"q_high={q_high:.4f}  alpha={cfg.pi_alpha}")

print("\nUpload new data for predictions (optional).")
new_data_loaded = False
new_df = None

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
    except Exception as e:
        print(f"Upload failed: {e}")
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
        new_labels = (new_df.iloc[:, :cfg.n_labels]
                      if cfg.n_labels > 0 else pd.DataFrame())

        if n_cols_new < cfg.n_labels + cfg.n_inputs:
            raise ValueError(
                f"Insufficient columns: {n_cols_new} < "
                f"{cfg.n_labels + cfg.n_inputs}"
            )
        new_inputs = new_df.iloc[:, cfg.n_labels:cfg.n_labels + cfg.n_inputs]

        if new_inputs.shape[1] != n_features:
            if set(new_inputs.columns).issubset(set(input_columns)):
                new_inputs = new_inputs[input_columns]
            else:
                raise ValueError(
                    "Column mismatch between training and new data inputs."
                )

        if new_inputs.isna().sum().sum() > 0:
            print("Imputing NaN features with TRAIN means...")
            imp = SimpleImputer(strategy="mean").fit(X_train_orig)
            new_inputs = pd.DataFrame(
                imp.transform(new_inputs), columns=new_inputs.columns
            )

        new_X = new_inputs.values
        print(f"Generating predictions for {len(new_X)} samples...")

        # Standardise -> predict -> inverse-transform
        new_X_scaled = scaler_X.transform(new_X)
        new_y_pred = scaler_y.inverse_transform(
            model.predict(new_X_scaled, verbose=0)
        ).reshape(-1)

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
        compact["PI_Lower_95%"] = pi_lower
        compact["PI_Upper_95%"] = pi_upper
        compact["PI_Width"] = pi_upper - pi_lower
        compact["PI_Calibration_Source"] = cfg.pi_calibration

        has_actual = False
        if n_cols_new > cfg.n_labels + cfg.n_inputs:
            try:
                y_actual = (
                    new_df.iloc[:, cfg.n_labels + cfg.n_inputs]
                    .values.reshape(-1, 1)
                )
                if not np.isnan(y_actual).all():
                    if np.isnan(y_actual).any():
                        print("Imputing NaN targets with TRAIN target mean...")
                        imp_y = SimpleImputer(strategy="mean").fit(y_train_orig)
                        y_actual = imp_y.transform(y_actual)
                    y_flat = y_actual.reshape(-1)
                    has_actual = True
                    residual_new = y_flat - new_y_pred
                    abs_err_new = np.abs(residual_new)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        abs_pct_new = np.where(
                            np.abs(y_flat) > 1e-12,
                            100 * abs_err_new / np.abs(y_flat), np.nan,
                        )
                    compact["Actual_Value"] = y_flat
                    compact["Prediction_Error"] = residual_new
                    compact["Abs_Error"] = abs_err_new
                    compact["Abs_Percent_Error"] = abs_pct_new
                    within_pi = (y_flat >= pi_lower) & (y_flat <= pi_upper)
                    compact["Within_95%_PI"] = within_pi.astype(int)

                    new_metrics = compute_basic_metrics(y_flat, new_y_pred)
                    print("\nMetrics on new data (inverse-transformed):")
                    for k, v in new_metrics.items():
                        print(f"  {k}: {v}")
                    coverage = within_pi.sum() / len(within_pi) * 100
                    print(f"Empirical 95% PI coverage: "
                          f"{within_pi.sum()}/{len(within_pi)} ({coverage:.1f}%)")
            except Exception as exc:
                print(f"Could not extract target: {exc}")

        if not has_actual:
            print("No actual values. Predictions + PI only.")

        # Full-row export
        full_new = new_df.copy()
        full_new["Predicted_Value"] = new_y_pred
        full_new["PI_Lower_95%"] = pi_lower
        full_new["PI_Upper_95%"] = pi_upper
        full_new["PI_Width"] = pi_upper - pi_lower
        full_new["PI_Calibration_Source"] = cfg.pi_calibration
        if has_actual:
            full_new["Actual_Value"] = y_flat
            full_new["Prediction_Error"] = residual_new
            full_new["Abs_Error"] = abs_err_new
            full_new["Abs_Percent_Error"] = abs_pct_new
            full_new["Within_95%_PI"] = within_pi.astype(int)

        # Save
        compact.to_excel(
            os.path.join(cfg.export_dir, "new_data_predictions.xlsx"),
            index=False, engine="openpyxl",
        )
        compact.to_csv(
            os.path.join(cfg.export_dir, "new_data_predictions.csv"),
            index=False,
        )
        full_new.to_excel(
            os.path.join(cfg.export_dir, "new_data_full_with_all_columns.xlsx"),
            index=False, engine="openpyxl",
        )
        full_new.to_csv(
            os.path.join(cfg.export_dir, "new_data_full_with_all_columns.csv"),
            index=False,
        )
        print("Predictions exported (all inverse-transformed).")
        _show(compact.head(10))

    except Exception as e:
        print(f"Error processing new data: {e}")
        traceback.print_exc()


# =============================================================================
#  DOWNLOAD #2: NEW DATA PREDICTIONS
# =============================================================================
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
        print(f"Created: {zip_pred} ({size_mb:.2f} MB, "
              f"{len(pred_files)} files)")

        if use_colab:
            print("Find 'new_data_predictions.zip' in Colab Files sidebar -> "
                  "right-click -> Download")
            print("Or run:  from google.colab import files; "
                  "files.download('new_data_predictions.zip')")
        else:
            dl = os.path.expanduser("~/Downloads")
            os.makedirs(dl, exist_ok=True)
            shutil.copy2(zip_pred, os.path.join(dl, zip_pred))
            print(f"Saved to {os.path.join(dl, zip_pred)}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
else:
    print("No prediction files found. Skipping download.")

print("\nAll done.")
