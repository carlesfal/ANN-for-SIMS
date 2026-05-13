# ============================================================
# MERGED CELL: HILL PRE-TRAINING + TUNING + WEIGHT TRANSFER + FULL PIPELINE
# ============================================================
#
# This single cell combines:
#   Phase 1: Hill-function pre-training (synthetic data warmup)
#   Phase 2: Real data loading, splitting, scaling
#   Phase 3: Hyperparameter tuning (fixed 128→64→32→1 architecture)
#   Phase 4: Refined weight transfer (shape-aware, applied at every stage)
#   Phase 5: K-fold cross-validation with pre-trained weights
#   Phase 6: Final training with pre-trained weights
#   Phase 7: Optional retrain on train+val with pre-trained weights
#   Phase 8: Evaluation, metrics, diagnostic plots, exports
#   Phase 9: 3D surface plots
#   Phase 10: New data predictions with empirical prediction intervals
#
# WEIGHT TRANSFER REFINEMENTS vs. original two-cell approach:
#   1. Centralised transfer_weights() with per-layer shape validation
#   2. Pre-trained weights propagated to CV folds, final model, and retrain
#   3. Hill pre-training kept in its own namespace (X_h_*, y_h_*) to avoid
#      variable collisions with the real data pipeline
#   4. Tuning runs on real data only (Hill data is for warmup, not tuning)
#   5. Graceful fallback to random init if architectures diverge
#
# STRICT NO-LEAKAGE evaluation:
#   - Scalers fit only on TRAIN (or TRAIN+VAL for optional retrain)
#   - CV inside TRAIN with fold-fitted scalers
#   - TEST evaluated once, never used for selection
#
# Paste this entire cell into Colab and run. Adjust USER CONFIG as needed.
# ============================================================

# --- Detect notebook / Colab ---
try:
    get_ipython  # type: ignore
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False

# --- Install dependencies ---
if IN_NOTEBOOK:
    print("Installing (if missing) tensorflow, keras-tuner, seaborn, openpyxl, joblib...")
    %pip install -q tensorflow "keras-tuner" seaborn openpyxl joblib
else:
    import subprocess, sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install",
         "tensorflow", "keras-tuner", "seaborn", "openpyxl", "joblib"]
    )

# --- Imports ---
import os
import shutil
import traceback
import zipfile
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as _dt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

try:
    import keras_tuner as kt
except Exception:
    try:
        import kerastuner as kt
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tuner"])
        import importlib as _importlib
        _importlib.invalidate_caches()
        import keras_tuner as kt

use_colab = False
try:
    from google.colab import files as colab_files  # type: ignore
    use_colab = True
except Exception:
    pass

def _show(obj, n=5):
    if IN_NOTEBOOK:
        try:
            display(obj)  # type: ignore
        except Exception:
            print(obj)
    else:
        try:
            print(obj.head(n))
        except Exception:
            print(obj)

# ======================== USER CONFIGURATION ========================
# Data layout
N_LABELS = 7
N_INPUTS = 10
TARGET_COL = None
SEP = "\t"

# Split percentages
TRAIN_PERCENT = 60
VAL_PERCENT = 20
TEST_PERCENT = 20

DISABLE_GPU = True

# Tuner / training
TUNER_TRIALS = 15
K_FOLDS = 15
RANDOM_SEED = 42
TUNER_EPOCHS = 200
CV_EPOCHS = 200
FINAL_EPOCHS = 200

# Optional retrain on TRAIN+VAL
DO_OPTIONAL_RETRAIN = True

# Prediction-interval calibration
PI_CALIBRATION = "val"   # "val" or "oof"
PI_ALPHA = 0.05          # 95 % PI

EXPORT_DIR = "optimized_model"

# Hill pre-training config
HILL_V_MAX = 200
HILL_K = 0.10
HILL_N = 1.20
HILL_X_MIN = 0.1
HILL_X_MAX = 100
N_SYNTHETIC = 500
PRETRAIN_EPOCHS = 50

# 3D surface plots
FEATURE_X = "MnCO3"
FEATURE_Y = "FeCO3"
GRID_N_3D = 50
RANGE_MODE_3D = "quantile"
Q_LOW_3D, Q_HIGH_3D = 0.02, 0.98
HOLD_MODE_3D = "median_train"
ROW_INDEX_3D = 0
OVERLAY_TRAIN_SCATTER = True
SCATTER_ALPHA = 0.25
SCATTER_SIZE = 10
MAX_PAIRS_3D = 12
# =====================================================================

# Validate split
total_percent = TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT
if abs(total_percent - 100) > 0.01:
    raise ValueError(
        f"Split percentages must sum to 100. "
        f"Got: {TRAIN_PERCENT}% + {VAL_PERCENT}% + {TEST_PERCENT}% = {total_percent}%"
    )

print(f"✓ Data split: TRAIN={TRAIN_PERCENT}%, VAL={VAL_PERCENT}%, TEST={TEST_PERCENT}%")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if DISABLE_GPU:
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled (if present). Using CPU.")
    except Exception as e:
        print("Could not change GPU visibility:", e)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def transfer_weights(source_model, target_model, verbose=True):
    """
    Transfer weights from source_model → target_model layer-by-layer.

    Matches layers by position among weight-bearing layers (Dense, etc.),
    validates shape compatibility for each weight tensor. Skips layers
    whose shapes do not match, with detailed logging.

    Returns
    -------
    n_transferred : int
        Number of weight-bearing layers whose weights were transferred.
    """
    src_layers = [l for l in source_model.layers if l.weights]
    tgt_layers = [l for l in target_model.layers if l.weights]

    n_transferred = 0
    for src, tgt in zip(src_layers, tgt_layers):
        src_w = src.get_weights()
        tgt_w = tgt.get_weights()

        if len(src_w) != len(tgt_w):
            if verbose:
                print(f"  ⊘ {src.name} → {tgt.name}: "
                      f"different number of weight tensors ({len(src_w)} vs {len(tgt_w)})")
            continue

        shapes_match = all(
            sw.shape == tw.shape for sw, tw in zip(src_w, tgt_w)
        )

        if shapes_match:
            tgt.set_weights(src_w)
            n_transferred += 1
            if verbose:
                print(f"  ✓ {src.name} {src_w[0].shape} → {tgt.name}")
        else:
            if verbose:
                s_shapes = [w.shape for w in src_w]
                t_shapes = [w.shape for w in tgt_w]
                print(f"  ✗ {src.name} → {tgt.name}: "
                      f"shape mismatch {s_shapes} vs {t_shapes}")

    if verbose:
        total_src = len(src_layers)
        total_tgt = len(tgt_layers)
        if total_src != total_tgt:
            print(f"  ℹ  Source has {total_src} weight layers, "
                  f"target has {total_tgt}; matched first {min(total_src, total_tgt)}")

    return n_transferred


def _generate_hill_data(n_samples, n_features, v_max, k, n_hill, x_min, x_max):
    """Generate synthetic Hill-function regression data."""
    X = np.random.uniform(x_min, x_max, (n_samples, n_features)).astype(np.float32)
    x1 = X[:, 0]
    y = v_max * (x1 ** n_hill) / (k ** n_hill + x1 ** n_hill)
    y += np.random.normal(0, 0.05 * v_max, n_samples)
    for i in range(1, min(n_features, 3)):
        y += 0.05 * v_max * (X[:, i] - x_min) / (x_max - x_min)
    return X, y.reshape(-1, 1)


def load_table(path, sep=SEP):
    """Robust table loader: Excel + common text encodings + separator sniff."""
    _, ext = os.path.splitext(path.lower())
    if ext in [".xlsx", ".xls", ".xlsm"]:
        print(f"Detected Excel file: {path}")
        return pd.read_excel(path, engine="openpyxl")

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings_to_try:
        try:
            print(f"Trying read_csv: sep={repr(sep)} encoding={enc}")
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e
    for enc in encodings_to_try:
        try:
            print(f"Fallback sniff: sep=None encoding={enc} (python engine)")
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read '{path}'. Last error: {last_err}")


def compute_basic_metrics(y_true, y_pred):
    actual = np.array(y_true).reshape(-1)
    pred = np.array(y_pred).reshape(-1)
    n = len(actual)
    residuals = actual - pred
    SSE = np.sum(residuals ** 2)
    MSE = SSE / n if n > 0 else np.nan
    RMSE = np.sqrt(MSE) if not np.isnan(MSE) else np.nan
    MAE = np.mean(np.abs(residuals)) if n > 0 else np.nan
    SEP_metric = np.sqrt(SSE / (n - 1)) if n > 1 else np.nan
    mean_abs_actual = np.mean(np.abs(actual)) if n > 0 else np.nan
    MRPD = (100.0 * np.sum(np.abs(residuals)) / (n * mean_abs_actual)
            if (n > 0 and not np.isclose(mean_abs_actual, 0.0)) else np.nan)
    R2 = r2_score(actual, pred) if n > 0 else np.nan
    return {"n": n, "SSE": SSE, "MSE": MSE, "RMSE": RMSE,
            "MAE": MAE, "SEP": SEP_metric, "MRPD_percent": MRPD, "R2": R2}


# =====================================================================
# PHASE 1: HILL PRE-TRAINING
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 1: HILL PRE-TRAINING (synthetic data warmup)")
print("=" * 70)

print("\n[1/4] Generating synthetic Hill-function data...")
X_h, y_h = _generate_hill_data(
    N_SYNTHETIC, N_INPUTS, HILL_V_MAX, HILL_K, HILL_N, HILL_X_MIN, HILL_X_MAX
)
print(f"✓ Generated: X {X_h.shape}, y {y_h.shape}")
print(f"  y range: [{y_h.min():.2f}, {y_h.max():.2f}]")

print("\n[2/4] Scaling Hill data...")
scaler_Xh = StandardScaler().fit(X_h)
scaler_yh = StandardScaler().fit(y_h)
X_h_scaled = scaler_Xh.transform(X_h)
y_h_scaled = scaler_yh.transform(y_h)

X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
    X_h_scaled, y_h_scaled, test_size=0.2, random_state=RANDOM_SEED
)
print(f"✓ Hill train: {X_h_train.shape}, Hill val: {X_h_val.shape}")

print("\n[3/4] Building warmup model (128→64→32→1)...")
tf.keras.backend.clear_session()

warmup_model = keras.Sequential([
    layers.Input(shape=(N_INPUTS,)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.Dropout(0.1),
    layers.Dense(1, activation='linear')
])
warmup_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
print(f"✓ Model built: {len(warmup_model.layers)} layers, "
      f"{warmup_model.count_params():,} parameters")

print(f"\n[4/4] Pre-training on Hill data ({PRETRAIN_EPOCHS} epochs)...")
es_hill = callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
history_hill = warmup_model.fit(
    X_h_train, y_h_train,
    validation_data=(X_h_val, y_h_val),
    epochs=PRETRAIN_EPOCHS,
    batch_size=32,
    callbacks=[es_hill],
    verbose=1
)
print(f"\n✓ Pre-training complete!")
print(f"  Final train loss: {history_hill.history['loss'][-1]:.6f}")
print(f"  Final val loss:   {history_hill.history['val_loss'][-1]:.6f}")
print("\n" + "=" * 70)
print("✓✓✓ PHASE 1 DONE — warmup_model ready for weight transfer ✓✓✓")
print("=" * 70)


# =====================================================================
# PHASE 2: REAL DATA LOADING & PREPARATION
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2: REAL DATA LOADING & PREPARATION")
print("=" * 70)

print("📂 Upload your dataset (dialog appears if running in Colab).")
if use_colab:
    uploaded = colab_files.upload()
    if len(uploaded) == 0:
        raise RuntimeError("No file uploaded.")
    file_name = list(uploaded.keys())[0]
else:
    file_name = "data.tsv"
    if not os.path.exists(file_name):
        raise FileNotFoundError(
            "Not in Colab and 'data.tsv' not found. Upload or change the path."
        )

df = load_table(file_name, sep=SEP)
print(f"✅ Data loaded. Shape: {df.shape}")
_show(df.head())

# --- Column selection ---
n_cols = df.shape[1]
if TARGET_COL is not None:
    target_idx = TARGET_COL
    if target_idx < 0 or target_idx >= n_cols:
        raise IndexError("TARGET_COL out of range.")
    labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
    input_cols = list(range(N_LABELS, n_cols))
    input_cols.remove(target_idx)
    inputs_df = df.iloc[:, input_cols]
    y_full = df.iloc[:, target_idx].values.reshape(-1, 1)
else:
    if N_LABELS + N_INPUTS >= n_cols:
        print("N_LABELS + N_INPUTS >= total columns. "
              "Using last column as target and the middle columns as inputs.")
        labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, N_LABELS:-1]
        y_full = df.iloc[:, -1].values.reshape(-1, 1)
    else:
        labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, N_LABELS:N_LABELS + N_INPUTS]
        y_full = df.iloc[:, N_LABELS + N_INPUTS].values.reshape(-1, 1)

print(f"Labels shape: {labels_df.shape}, Inputs shape: {inputs_df.shape}, "
      f"y shape: {y_full.shape}")

X_full = inputs_df.values
y_full_arr = y_full
labels_full = labels_df
row_pos = np.arange(len(df))

# --- Configurable split ---
test_frac = TEST_PERCENT / 100.0
val_frac = VAL_PERCENT / 100.0
train_frac = TRAIN_PERCENT / 100.0

X_temp, X_test_orig, y_temp, y_test_orig, labels_temp, labels_test, idx_temp, idx_test = \
    train_test_split(X_full, y_full_arr, labels_full, row_pos,
                     test_size=test_frac, random_state=RANDOM_SEED)

val_split_ratio = val_frac / (train_frac + val_frac)
X_train_orig, X_val_orig, y_train_orig, y_val_orig, labels_train, labels_val, idx_train, idx_val = \
    train_test_split(X_temp, y_temp, labels_temp, idx_temp,
                     test_size=val_split_ratio, random_state=RANDOM_SEED)

inputs_columns = list(inputs_df.columns)
X_train_df = pd.DataFrame(X_train_orig, columns=inputs_columns).reset_index(drop=True)
X_val_df   = pd.DataFrame(X_val_orig,   columns=inputs_columns).reset_index(drop=True)
X_test_df  = pd.DataFrame(X_test_orig,  columns=inputs_columns).reset_index(drop=True)

print(f"\nSplit sizes (rows): train={len(X_train_orig)} "
      f"val={len(X_val_orig)} test={len(X_test_orig)}")
print(f"Split percentages: "
      f"train={100*len(X_train_orig)/len(df):.1f}% "
      f"val={100*len(X_val_orig)/len(df):.1f}% "
      f"test={100*len(X_test_orig)/len(df):.1f}%")

# --- Scaling (fit on TRAIN only — no leakage) ---
scaler_X = StandardScaler().fit(X_train_orig)
scaler_y = StandardScaler().fit(y_train_orig)

X_train = scaler_X.transform(X_train_orig)
X_val   = scaler_X.transform(X_val_orig)
X_test  = scaler_X.transform(X_test_orig)

y_train = scaler_y.transform(y_train_orig)
y_val   = scaler_y.transform(y_val_orig)
y_test  = scaler_y.transform(y_test_orig)

# Validate that Hill pre-training input dimension matches real data
_real_n_inputs = X_train.shape[1]
if _real_n_inputs != N_INPUTS:
    print(f"\n⚠️  Real data has {_real_n_inputs} input features but "
          f"N_INPUTS={N_INPUTS}. Weight transfer from Hill model may "
          f"fail on the first Dense layer.")

print("\n" + "=" * 70)
print("✓✓✓ PHASE 2 DONE — real data ready ✓✓✓")
print("=" * 70)


# =====================================================================
# PHASE 3: HYPERPARAMETER TUNING (fixed architecture on real data)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 3: HYPERPARAMETER TUNING (fixed 128→64→32→1 on real data)")
print("=" * 70)


def build_model_fixed(hp):
    """
    Fixed architecture matching the warmup model: 128 → 64 → 32 → 1.
    Only tunes: L2 regularisation, dropout rates, learning rate.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    l2_val = hp.Choice('l2_reg', [1e-4, 1e-3, 1e-2])

    model.add(layers.Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_val)))
    model.add(layers.Dropout(hp.Float('dropout_1', 0.0, 0.3, step=0.05)))

    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_val)))
    model.add(layers.Dropout(hp.Float('dropout_2', 0.0, 0.3, step=0.05)))

    model.add(layers.Dense(32, activation='relu',
                           kernel_regularizer=regularizers.l2(l2_val)))
    model.add(layers.Dropout(hp.Float('dropout_3', 0.0, 0.2, step=0.05)))

    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('lr', [1e-4, 5e-4, 1e-3, 5e-3])
        ),
        loss='mse',
        metrics=['mae']
    )
    return model


print(f"\n[1/2] Running hyperparameter search ({TUNER_TRIALS} trials)...")

tuner = kt.RandomSearch(
    build_model_fixed,
    objective='val_loss',
    max_trials=TUNER_TRIALS,
    executions_per_trial=1,
    directory='tuner_results',
    project_name=f'ann_{TRAIN_PERCENT}_{VAL_PERCENT}_{TEST_PERCENT}'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=TUNER_EPOCHS,
    batch_size=32,
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]

print("\n[2/2] Best hyperparameters found:")
for k, v in best_hp.values.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("✓✓✓ PHASE 3 DONE — best hyperparameters found ✓✓✓")
print("=" * 70)


# =====================================================================
# PHASE 4: REFINED WEIGHT TRANSFER
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 4: WEIGHT TRANSFER FROM HILL PRE-TRAINED MODEL")
print("=" * 70)

_model_with_pretrain = None
_pretrain_available = False

print("\n[1/2] Building model from best hyperparameters...")
model_for_transfer = build_model_fixed(best_hp)
print(f"✓ Target model: {model_for_transfer.count_params():,} parameters")

print("\n[2/2] Transferring weights from warmup_model → tuned model...")
print(f"  Source (warmup_model) layers: "
      f"{[l.name for l in warmup_model.layers if l.weights]}")
print(f"  Target (tuned model) layers:  "
      f"{[l.name for l in model_for_transfer.layers if l.weights]}")

n_transferred = transfer_weights(warmup_model, model_for_transfer, verbose=True)

if n_transferred > 0:
    print(f"\n✓ Successfully transferred {n_transferred} weight-bearing layers!")
    _model_with_pretrain = model_for_transfer
    _pretrain_available = True
    print("  → _model_with_pretrain will be used for CV, final training, and retrain")
else:
    print("\n⚠️  No weights transferred (architecture mismatch between Hill "
          "model and tuned model). Proceeding with random initialisation.")
    _model_with_pretrain = None

print("\n" + "=" * 70)
if _pretrain_available:
    print("✓✓✓ PHASE 4 DONE — pre-trained weights ready ✓✓✓")
else:
    print("ℹ️  PHASE 4 DONE — no pre-trained weights (will use random init)")
print("=" * 70)


# =====================================================================
# PHASE 5: K-FOLD CROSS-VALIDATION (with pre-trained weights)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 5: K-FOLD CROSS-VALIDATION (TRAIN split only, no leakage)")
print("=" * 70)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

r2_scores, rmse_scores, mae_scores = [], [], []
y_oof_pred_inv = np.full(len(y_train_orig), np.nan, dtype=float)
y_oof_true_inv = np.full(len(y_train_orig), np.nan, dtype=float)

if _pretrain_available:
    print("✓ Will initialise each CV fold with pre-trained weights")
else:
    print("ℹ  Will initialise each CV fold from scratch (no pre-trained weights)")

print(f"\nRunning {K_FOLDS}-Fold CV with fold-fitted scalers...\n")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig), start=1):
    X_tr_orig, X_va_orig = X_train_orig[tr_idx], X_train_orig[va_idx]
    y_tr_orig, y_va_orig = y_train_orig[tr_idx], y_train_orig[va_idx]

    fold_scaler_X = StandardScaler().fit(X_tr_orig)
    fold_scaler_y = StandardScaler().fit(y_tr_orig)

    X_tr = fold_scaler_X.transform(X_tr_orig)
    X_va = fold_scaler_X.transform(X_va_orig)
    y_tr = fold_scaler_y.transform(y_tr_orig)
    y_va = fold_scaler_y.transform(y_va_orig)

    model_fold = build_model_fixed(best_hp)

    if _pretrain_available:
        transfer_weights(_model_with_pretrain, model_fold, verbose=False)

    es_cv = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model_fold.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=CV_EPOCHS,
        batch_size=32,
        callbacks=[es_cv],
        verbose=1
    )

    y_va_pred_scaled = model_fold.predict(X_va, verbose=0)
    y_va_pred_inv_fold = fold_scaler_y.inverse_transform(y_va_pred_scaled).reshape(-1)
    y_va_true_inv_fold = y_va_orig.reshape(-1)

    y_oof_pred_inv[va_idx] = y_va_pred_inv_fold
    y_oof_true_inv[va_idx] = y_va_true_inv_fold

    r2 = r2_score(y_va_true_inv_fold, y_va_pred_inv_fold)
    rmse = np.sqrt(mean_squared_error(y_va_true_inv_fold, y_va_pred_inv_fold))
    mae = np.mean(np.abs(y_va_true_inv_fold - y_va_pred_inv_fold))

    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    print(f"Fold {fold}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

print("\nCV Metrics (Mean ± Std) on TRAIN split (no leakage):")
print(f"R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")

if np.isnan(y_oof_pred_inv).any():
    raise RuntimeError("OOF predictions contain NaNs. Check fold logic.")

ss_res_press = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
ss_tot_train = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
pred_R2_train = 1.0 - ss_res_press / ss_tot_train if ss_tot_train != 0 else np.nan
print(f"\nPredicted R² (Q²) on TRAIN split (OOF/PRESS, no leakage): {pred_R2_train:.4f}")


# =====================================================================
# PHASE 6: FINAL TRAINING (with pre-trained weights)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 6: FINAL TRAINING (fit on TRAIN, validate on VAL)")
print("=" * 70)

os.makedirs(EXPORT_DIR, exist_ok=True)
checkpoint_path = os.path.join(EXPORT_DIR, "best_model.keras")
mc = callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1
)
es_final = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

tf.keras.backend.clear_session()
model = build_model_fixed(best_hp)

# Apply pre-trained weights before final training
if _pretrain_available:
    print("Initialising final model with pre-trained weights...")
    n_tx = transfer_weights(warmup_model, model, verbose=True)
    print(f"✓ Transferred {n_tx} layers to final model\n")
else:
    print("Training final model from random initialisation\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=FINAL_EPOCHS,
    batch_size=32,
    callbacks=[es_final, mc],
    verbose=1
)

if os.path.exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path, compile=False)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print(f"\nBest epoch selected by VAL loss: {best_epoch}")

# Evaluate ONCE on TEST
print("\nEvaluating once on TEST (no selection/tuning on test).")
y_test_pred_eval = scaler_y.inverse_transform(
    model.predict(X_test, verbose=0)
).reshape(-1)
y_test_inv_eval = scaler_y.inverse_transform(y_test).reshape(-1)
print(f"TEST: R²={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}, "
      f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}")


# =====================================================================
# PHASE 7: OPTIONAL RETRAIN ON TRAIN+VAL (with pre-trained weights)
# =====================================================================
if DO_OPTIONAL_RETRAIN:
    print("\n" + "=" * 70)
    print("PHASE 7: OPTIONAL RETRAIN (clean: refit scalers on TRAIN+VAL)")
    print("=" * 70)

    X_trainval_orig = np.vstack([X_train_orig, X_val_orig])
    y_trainval_orig = np.vstack([y_train_orig, y_val_orig])

    scaler_X_tv = StandardScaler().fit(X_trainval_orig)
    scaler_y_tv = StandardScaler().fit(y_trainval_orig)

    X_trainval_tv = scaler_X_tv.transform(X_trainval_orig)
    y_trainval_tv = scaler_y_tv.transform(y_trainval_orig)

    X_test_tv = scaler_X_tv.transform(X_test_orig)
    y_test_tv = scaler_y_tv.transform(y_test_orig)

    tf.keras.backend.clear_session()
    model_retrain = build_model_fixed(best_hp)

    # Apply pre-trained weights before retraining
    if _pretrain_available:
        print("Initialising retrain model with pre-trained weights...")
        n_tx = transfer_weights(warmup_model, model_retrain, verbose=True)
        print(f"✓ Transferred {n_tx} layers to retrain model\n")
    else:
        print("Retraining from random initialisation\n")

    model_retrain.fit(
        X_trainval_tv, y_trainval_tv,
        epochs=best_epoch,
        batch_size=32,
        verbose=1
    )

    y_test_pred_rt = scaler_y_tv.inverse_transform(
        model_retrain.predict(X_test_tv, verbose=0)
    ).reshape(-1)
    y_test_inv_rt = y_test_orig.reshape(-1)

    print(f"TEST (retrained): "
          f"R²={r2_score(y_test_inv_rt, y_test_pred_rt):.4f}, "
          f"RMSE={np.sqrt(mean_squared_error(y_test_inv_rt, y_test_pred_rt)):.4f}, "
          f"MAE={np.mean(np.abs(y_test_inv_rt - y_test_pred_rt)):.4f}")

    # Use retrained artifacts for exports/predictions
    model = model_retrain
    scaler_X = scaler_X_tv
    scaler_y = scaler_y_tv
    X_train = scaler_X.transform(X_train_orig)
    X_val   = scaler_X.transform(X_val_orig)
    X_test  = scaler_X.transform(X_test_orig)
    y_train = scaler_y.transform(y_train_orig)
    y_val   = scaler_y.transform(y_val_orig)
    y_test  = scaler_y.transform(y_test_orig)
    print("Using retrained model + train+val-fitted scalers for exports/predictions.")


# =====================================================================
# PHASE 8: EVALUATION, METRICS, EXPORTS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 8: EVALUATION & EXPORTS")
print("=" * 70)

# --- Print & save final Dense architecture ---
try:
    from tensorflow.keras.layers import Dense
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    if dense_layers:
        print("\n=== Final Dense Layers (in model order) ===")
        scheme_lines = []
        for i, layer in enumerate(dense_layers, start=1):
            units = getattr(layer, "units", None)
            activation = getattr(layer, "activation", None)
            act_name = activation.__name__ if activation is not None else "N/A"
            reg = getattr(layer, "kernel_regularizer", None)
            reg_str = None
            try:
                if reg is not None and hasattr(reg, "l2"):
                    reg_str = f"L2={getattr(reg, 'l2')}"
                elif reg is not None:
                    reg_str = str(reg)
            except Exception:
                reg_str = str(reg)
            line = f"Layer {i}: name='{layer.name}', units={units}, activation={act_name}"
            if reg_str:
                line += f", regularizer={reg_str}"
            print(line)
            scheme_lines.append(line)
        last = dense_layers[-1]
        final_line = (
            f"Final Dense (output) -> name='{last.name}', "
            f"units={getattr(last, 'units', None)}, "
            f"activation={getattr(last, 'activation').__name__ if getattr(last, 'activation', None) else 'N/A'}"
        )
        print("\n" + final_line)
        scheme_lines.append("")
        scheme_lines.append(final_line)
        scheme_path = os.path.join(EXPORT_DIR, "model_scheme.txt")
        with open(scheme_path, "w", encoding="utf-8") as f:
            f.write("Final Dense Layers (in model order)\n")
            f.write("===============================\n")
            for ln in scheme_lines:
                f.write(ln + "\n")
        print(f"Model scheme saved to: {scheme_path}")
except Exception as e:
    print("Error while printing/saving model architecture:", e)

# --- Save model & scalers ---
model.save(os.path.join(EXPORT_DIR, "final_model.keras"))
try:
    model.save(os.path.join(EXPORT_DIR, "final_model.h5"))
except Exception as e:
    print("Could not save .h5 model (ok to ignore):", e)

joblib.dump(scaler_X, os.path.join(EXPORT_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(EXPORT_DIR, "scaler_y.pkl"))
print(f"Model and scalers saved to {EXPORT_DIR}/")

# --- Predictions (inverse scaled) ---
y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
y_val_pred   = scaler_y.inverse_transform(model.predict(X_val,   verbose=0))
y_test_pred  = scaler_y.inverse_transform(model.predict(X_test,  verbose=0))

y_train_inv = scaler_y.inverse_transform(y_train)
y_val_inv   = scaler_y.inverse_transform(y_val)
y_test_inv  = scaler_y.inverse_transform(y_test)

# --- Metrics ---
metrics_train = compute_basic_metrics(y_train_inv, y_train_pred)
metrics_val   = compute_basic_metrics(y_val_inv,   y_val_pred)
metrics_test  = compute_basic_metrics(y_test_inv,  y_test_pred)

p = X_train.shape[1]
n_in  = metrics_train["n"]
n_val = metrics_val["n"]
n_out = metrics_test["n"]
r2_in  = metrics_train["R2"]
r2_val = metrics_val["R2"]
r2_out = metrics_test["R2"]
r2_adj_in  = 1 - (1 - r2_in)  * (n_in  - 1) / (n_in  - p - 1) if (n_in  - p - 1) > 0 else np.nan
r2_adj_val = 1 - (1 - r2_val) * (n_val - 1) / (n_val - p - 1) if (n_val - p - 1) > 0 else np.nan
r2_adj_out = 1 - (1 - r2_out) * (n_out - 1) / (n_out - p - 1) if (n_out - p - 1) > 0 else np.nan

metrics_train.update({"R2_adj": r2_adj_in,  "Predicted_R2_Q2": pred_R2_train})
metrics_val.update(  {"R2_adj": r2_adj_val, "Predicted_R2_Q2": np.nan})
metrics_test.update( {"R2_adj": r2_adj_out, "Predicted_R2_Q2": np.nan})

print("\n=== Final Metrics ===")
print(f"Training set ({TRAIN_PERCENT}%):")
for k, v in metrics_train.items():
    print(f"  {k}: {v}")
print(f"Validation set ({VAL_PERCENT}%):")
for k, v in metrics_val.items():
    print(f"  {k}: {v}")
print(f"Test set ({TEST_PERCENT}%):")
for k, v in metrics_test.items():
    print(f"  {k}: {v}")

# --- Save summary statistics ---
stats_path = os.path.join(EXPORT_DIR, "model_statistics.txt")
try:
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Model statistics summary\n")
        f.write("========================\n\n")
        f.write(f"Current date: {_dt.date.today().isoformat()}\n")
        f.write(f"Number of predictors (p): {p}\n")
        f.write(f"Data split: {TRAIN_PERCENT}% train / {VAL_PERCENT}% validation / {TEST_PERCENT}% test\n")
        f.write(f"Hill pre-training: V_max={HILL_V_MAX}, K={HILL_K}, n={HILL_N}, "
                f"epochs={PRETRAIN_EPOCHS}, samples={N_SYNTHETIC}\n")
        f.write(f"Pre-trained weight transfer: {'YES' if _pretrain_available else 'NO'}\n\n")
        f.write("Hyperparameters (best):\n")
        for k, v in best_hp.values.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTraining set metrics ({TRAIN_PERCENT}%):\n")
        for k, v in metrics_train.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nValidation set metrics ({VAL_PERCENT}%):\n")
        for k, v in metrics_val.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTest set metrics ({TEST_PERCENT}%):\n")
        for k, v in metrics_test.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nPredicted R2 (Q2) on training (OOF aggregated, no leakage): {pred_R2_train}\n")
        f.write(f"Best epoch selected on VAL loss: {best_epoch}\n")
        f.write(f"Optional retrain on train+val: {DO_OPTIONAL_RETRAIN}\n")
        f.write(f"PI calibration source: {PI_CALIBRATION}, alpha={PI_ALPHA}\n")
    print(f"Model statistics saved to: {stats_path}")
except Exception as e:
    print("Could not save model statistics file:", e)

# --- Plot MSE evolution ---
mse_img_path = os.path.join(EXPORT_DIR, "mse_evolution.png")
try:
    if 'history' in globals() and hasattr(history, "history"):
        hist = history.history
        loss = hist.get('loss', None)
        val_loss = hist.get('val_loss', None)
        if loss is not None:
            epochs_range = range(1, len(loss) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(epochs_range, loss, label='Train MSE (loss)', marker='o')
            if val_loss is not None:
                plt.plot(epochs_range, val_loss, label='Validation MSE (val_loss)', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('MSE (loss)')
            plt.title('Evolution of MSE during final training')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(mse_img_path, dpi=150)
            if IN_NOTEBOOK:
                try:
                    from IPython.display import Image, display as ipy_display
                    ipy_display(Image(mse_img_path))
                except Exception:
                    pass
            plt.show()
except Exception as e:
    print("Error while plotting MSE evolution:", e)

# --- Diagnostic plotting functions ---
def plot_predicted_vs_actual(y_true, y_pred, title, save_path=None):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = np.min([np.min(y_true), np.min(y_pred)])
    mx = np.max([np.max(y_true), np.max(y_pred)])
    plt.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
    try:
        sns.regplot(x=y_true, y=y_pred, scatter=False, color='blue', ci=None)
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
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


def plot_residuals_std_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1)
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(std_res, bins=25, kde=True, color='gray', edgecolor='black')
    plt.title(f"{title_prefix} - Standardized residuals distribution")
    plt.xlabel("Standardized residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(std_res, marker='o', linestyle='-')
    plt.title(f"{title_prefix} - Standardized residuals (time series)")
    plt.xlabel("Observation index")
    plt.ylabel("Standardized residual")
    plt.grid(True)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(save_prefix + "_std_residuals_hist_ts.png", dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


def plot_residuals_raw_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1)
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, bins=25, kde=True, color='salmon', edgecolor='black')
    plt.title(f"{title_prefix} - Raw residuals distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(residuals, marker='o', linestyle='-')
    plt.title(f"{title_prefix} - Raw residuals (time series)")
    plt.xlabel("Observation index")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    if save_prefix:
        plt.savefig(save_prefix + "_raw_residuals_hist_ts.png", dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()


try:
    plots_dir = os.path.join(EXPORT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_predicted_vs_actual(y_train_inv, y_train_pred, "Predicted vs Actual (Train)",
                            save_path=os.path.join(plots_dir, "pred_vs_actual_train.png"))
    plot_residuals_std_and_timeseries(y_train_inv, y_train_pred, "Train",
                                     save_prefix=os.path.join(plots_dir, "train_std"))
    plot_residuals_raw_and_timeseries(y_train_inv, y_train_pred, "Train",
                                     save_prefix=os.path.join(plots_dir, "train_raw"))

    plot_predicted_vs_actual(y_val_inv, y_val_pred, "Predicted vs Actual (Validation)",
                            save_path=os.path.join(plots_dir, "pred_vs_actual_val.png"))
    plot_residuals_std_and_timeseries(y_val_inv, y_val_pred, "Validation",
                                     save_prefix=os.path.join(plots_dir, "val_std"))
    plot_residuals_raw_and_timeseries(y_val_inv, y_val_pred, "Validation",
                                     save_prefix=os.path.join(plots_dir, "val_raw"))

    plot_predicted_vs_actual(y_test_inv, y_test_pred, "Predicted vs Actual (Test)",
                            save_path=os.path.join(plots_dir, "pred_vs_actual_test.png"))
    plot_residuals_std_and_timeseries(y_test_inv, y_test_pred, "Test",
                                     save_prefix=os.path.join(plots_dir, "test_std"))
    plot_residuals_raw_and_timeseries(y_test_inv, y_test_pred, "Test",
                                     save_prefix=os.path.join(plots_dir, "test_raw"))
    print("Diagnostic plots saved to:", plots_dir)
except Exception as e:
    print("Error while creating diagnostic plots:", e)
    traceback.print_exc()

# --- Prepare export DataFrames ---
def make_export_df(labels_df_part, inputs_df_part, y_true, y_pred):
    actual = np.array(y_true).reshape(-1)
    pred = np.array(y_pred).reshape(-1)
    residual = actual - pred
    abs_err = np.abs(residual)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_pct = np.where(np.abs(actual) > 1e-12,
                           100.0 * abs_err / np.abs(actual), np.nan)
    df_export = pd.DataFrame(index=range(len(actual)))
    if labels_df_part is not None and labels_df_part.shape[0] == len(actual):
        df_export = pd.concat([df_export, labels_df_part.reset_index(drop=True)], axis=1)
    df_export = pd.concat([df_export, inputs_df_part.reset_index(drop=True)], axis=1)
    df_export["Actual_Value"] = actual
    df_export["Predicted_Value"] = pred
    df_export["Residual"] = residual
    df_export["Abs_Error"] = abs_err
    df_export["Abs_Percent_Error"] = abs_pct
    return df_export

results_train = make_export_df(labels_train.reset_index(drop=True), X_train_df, y_train_inv, y_train_pred)
results_val   = make_export_df(labels_val.reset_index(drop=True),   X_val_df,   y_val_inv,   y_val_pred)
results_test  = make_export_df(labels_test.reset_index(drop=True),  X_test_df,  y_test_inv,  y_test_pred)

# --- Save standard exports ---
train_path = os.path.join(EXPORT_DIR, "train_predictions.xlsx")
val_path   = os.path.join(EXPORT_DIR, "val_predictions.xlsx")
test_path  = os.path.join(EXPORT_DIR, "test_predictions.xlsx")
results_train.to_excel(train_path, index=False, engine='openpyxl')
results_val.to_excel(val_path, index=False, engine='openpyxl')
results_test.to_excel(test_path, index=False, engine='openpyxl')
results_train.to_csv(os.path.join(EXPORT_DIR, "train_predictions.csv"), index=False)
results_val.to_csv(os.path.join(EXPORT_DIR, "val_predictions.csv"), index=False)
results_test.to_csv(os.path.join(EXPORT_DIR, "test_predictions.csv"), index=False)
print(f"Results exported to {train_path}, {val_path}, {test_path} and CSV equivalents")

# --- Create full-row exports ---
try:
    train_full_rows = df.iloc[idx_train].reset_index(drop=True)
    val_full_rows   = df.iloc[idx_val].reset_index(drop=True)
    test_full_rows  = df.iloc[idx_test].reset_index(drop=True)

    y_train_pred_flat = np.array(y_train_pred).reshape(-1)
    y_val_pred_flat   = np.array(y_val_pred).reshape(-1)
    y_test_pred_flat  = np.array(y_test_pred).reshape(-1)
    y_train_inv_flat  = np.array(y_train_inv).reshape(-1)
    y_val_inv_flat    = np.array(y_val_inv).reshape(-1)
    y_test_inv_flat   = np.array(y_test_inv).reshape(-1)

    residuals_train = y_train_inv_flat - y_train_pred_flat
    residuals_val   = y_val_inv_flat   - y_val_pred_flat
    residuals_test  = y_test_inv_flat  - y_test_pred_flat
    abs_err_train = np.abs(residuals_train)
    abs_err_val   = np.abs(residuals_val)
    abs_err_test  = np.abs(residuals_test)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_pct_train = np.where(np.abs(y_train_inv_flat) > 1e-12,
                                 100.0 * abs_err_train / np.abs(y_train_inv_flat), np.nan)
        abs_pct_val   = np.where(np.abs(y_val_inv_flat) > 1e-12,
                                 100.0 * abs_err_val / np.abs(y_val_inv_flat), np.nan)
        abs_pct_test  = np.where(np.abs(y_test_inv_flat) > 1e-12,
                                 100.0 * abs_err_test / np.abs(y_test_inv_flat), np.nan)

    train_full_rows = train_full_rows.copy()
    val_full_rows   = val_full_rows.copy()
    test_full_rows  = test_full_rows.copy()

    for fr, actual, pred, res, aerr, apct in [
        (train_full_rows, y_train_inv_flat, y_train_pred_flat, residuals_train, abs_err_train, abs_pct_train),
        (val_full_rows,   y_val_inv_flat,   y_val_pred_flat,   residuals_val,   abs_err_val,   abs_pct_val),
        (test_full_rows,  y_test_inv_flat,  y_test_pred_flat,  residuals_test,  abs_err_test,  abs_pct_test),
    ]:
        fr["Actual_Value"]      = actual
        fr["Predicted_Value"]   = pred
        fr["Residual"]          = res
        fr["Abs_Error"]         = aerr
        fr["Abs_Percent_Error"] = apct

    def move_pred_cols_to_end(df_full):
        pred_cols = ["Actual_Value", "Predicted_Value", "Residual",
                     "Abs_Error", "Abs_Percent_Error"]
        cols = [c for c in df_full.columns if c not in pred_cols] + pred_cols
        return df_full[cols]

    train_full_rows = move_pred_cols_to_end(train_full_rows)
    val_full_rows   = move_pred_cols_to_end(val_full_rows)
    test_full_rows  = move_pred_cols_to_end(test_full_rows)

    for name, fr in [("train", train_full_rows), ("val", val_full_rows), ("test", test_full_rows)]:
        fr.to_excel(os.path.join(EXPORT_DIR, f"{name}_full_with_all_columns.xlsx"),
                    index=False, engine='openpyxl')
        fr.to_csv(os.path.join(EXPORT_DIR, f"{name}_full_with_all_columns.csv"), index=False)
    print("Full-row exports saved")
except Exception as e:
    print("Error while creating/saving full-row exports:", e)
    traceback.print_exc()

_show(results_train.head())
_show(results_val.head())
_show(results_test.head())

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED!")
print("Training results are ready for download.")
print("=" * 80)


# =====================================================================
# PHASE 9: 3D SURFACE PLOTS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 9: 3D SURFACE PLOTS")
print("=" * 70)

from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def predict_from_origX(X_orig_2d):
    """Predict from unscaled original feature space."""
    X_scaled = scaler_X.transform(X_orig_2d)
    y_scaled = model.predict(X_scaled, verbose=0)
    return scaler_y.inverse_transform(y_scaled).reshape(-1)


def _grid_vals(col_idx, grid_n=GRID_N_3D, range_mode=RANGE_MODE_3D,
               q_low=Q_LOW_3D, q_high=Q_HIGH_3D):
    v = X_train_orig[:, col_idx]
    if range_mode == "minmax":
        lo, hi = float(np.min(v)), float(np.max(v))
    elif range_mode == "quantile":
        lo, hi = float(np.quantile(v, q_low)), float(np.quantile(v, q_high))
    else:
        raise ValueError("range_mode must be 'quantile' or 'minmax'")
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, grid_n)


def _build_Xref(hold_mode=HOLD_MODE_3D, row_index=ROW_INDEX_3D):
    if hold_mode == "median_train":
        return np.median(X_train_orig, axis=0)
    elif hold_mode == "mean_train":
        return np.mean(X_train_orig, axis=0)
    elif hold_mode == "row":
        if row_index < 0 or row_index >= len(X_train_orig):
            raise IndexError(f"ROW_INDEX out of range: {row_index}")
        return X_train_orig[row_index].copy()
    else:
        raise ValueError("HOLD_MODE must be: median_train, mean_train, or row")


# --- Single featured pair surface (FEATURE_X vs FEATURE_Y) ---
try:
    if FEATURE_X in inputs_columns and FEATURE_Y in inputs_columns:
        ix = inputs_columns.index(FEATURE_X)
        iy = inputs_columns.index(FEATURE_Y)

        Xref = _build_Xref()
        x_vals = _grid_vals(ix)
        y_vals = _grid_vals(iy)
        XX, YY = np.meshgrid(x_vals, y_vals)

        Xgrid = np.tile(Xref.reshape(1, -1), (XX.size, 1))
        Xgrid[:, ix] = XX.reshape(-1)
        Xgrid[:, iy] = YY.reshape(-1)
        Z = predict_from_origX(Xgrid).reshape(XX.shape)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XX, YY, Z, cmap="viridis", linewidth=0,
                               antialiased=True, alpha=0.92)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        if OVERLAY_TRAIN_SCATTER:
            z_train_pred = predict_from_origX(X_train_orig)
            ax.scatter(X_train_orig[:, ix], X_train_orig[:, iy], z_train_pred,
                       c="k", s=SCATTER_SIZE, alpha=SCATTER_ALPHA)

        ax.set_title(f"Predicted surface: {FEATURE_X} vs {FEATURE_Y}\n"
                     f"(others held constant: {HOLD_MODE_3D})")
        ax.set_xlabel(FEATURE_X)
        ax.set_ylabel(FEATURE_Y)
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()
        plt.show()
    else:
        missing = []
        if FEATURE_X not in inputs_columns:
            missing.append(FEATURE_X)
        if FEATURE_Y not in inputs_columns:
            missing.append(FEATURE_Y)
        print(f"⚠️  Feature(s) {missing} not found in inputs_columns. "
              f"Skipping single-pair surface plot.")
except Exception as e:
    print(f"⚠️  Error in single-pair surface: {e}")
    traceback.print_exc()

# --- Multi-pair 3D surfaces ---
try:
    n_features = X_train_orig.shape[1]

    def _colname(i):
        return inputs_columns[i] if isinstance(inputs_columns[i], (str, int)) else str(inputs_columns[i])

    feat_indices = list(range(n_features))
    pairs = list(combinations(feat_indices, 2))[:MAX_PAIRS_3D]

    Xref_mp = _build_Xref()

    print(f"\nPlotting {len(pairs)} 3D surfaces; "
          f"HOLD_MODE={HOLD_MODE_3D}, GRID_N={GRID_N_3D}, RANGE={RANGE_MODE_3D}")

    for (i, j) in pairs:
        xi_vals = _grid_vals(i)
        xj_vals = _grid_vals(j)
        XI, XJ = np.meshgrid(xi_vals, xj_vals)

        Xgrid = np.tile(Xref_mp.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)
        ygrid = predict_from_origX(Xgrid)
        Y = ygrid.reshape(XI.shape)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XI, XJ, Y, cmap="viridis", linewidth=0,
                               antialiased=True, alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        if OVERLAY_TRAIN_SCATTER:
            y_train_pred_vis = predict_from_origX(X_train_orig)
            ax.scatter(X_train_orig[:, i], X_train_orig[:, j], y_train_pred_vis,
                       c="k", s=SCATTER_SIZE, alpha=SCATTER_ALPHA)

        ax.set_title(f"Predicted surface: {_colname(i)} vs {_colname(j)}\n"
                     f"(others held constant: {HOLD_MODE_3D})")
        ax.set_xlabel(str(_colname(i)))
        ax.set_ylabel(str(_colname(j)))
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"⚠️  Error in multi-pair surfaces: {e}")
    traceback.print_exc()

# --- Export all 3D surfaces to files ---
try:
    GRID_N_EXPORT = 30
    DPI_EXPORT = 160

    plots_dir_3d = os.path.join(EXPORT_DIR, "plots_3d")
    os.makedirs(plots_dir_3d, exist_ok=True)

    def _safe_name(s):
        s = str(s)
        for ch in [" ", "/", "\\", ":", ";", "|", "(", ")", "[", "]", "{", "}", "%"]:
            s = s.replace(ch, "_")
        return s

    Xref_export = _build_Xref()
    all_pairs = list(combinations(range(n_features), 2))
    print(f"\nExporting {len(all_pairs)} 3D surfaces to: {plots_dir_3d}")

    for (i, j) in all_pairs:
        xi = _grid_vals(i, grid_n=GRID_N_EXPORT)
        xj = _grid_vals(j, grid_n=GRID_N_EXPORT)
        XI, XJ = np.meshgrid(xi, xj)

        Xgrid = np.tile(Xref_export.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)
        Z = predict_from_origX(Xgrid).reshape(XI.shape)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0,
                               antialiased=True, alpha=0.95)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        ax.set_title(f"Predicted surface: {inputs_columns[i]} vs {inputs_columns[j]}\n"
                     f"(others held constant: {HOLD_MODE_3D})")
        ax.set_xlabel(str(inputs_columns[i]))
        ax.set_ylabel(str(inputs_columns[j]))
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()

        out_png = os.path.join(
            plots_dir_3d,
            f"surface_{_safe_name(inputs_columns[i])}_vs_"
            f"{_safe_name(inputs_columns[j])}_hold_{HOLD_MODE_3D}.png"
        )
        plt.savefig(out_png, dpi=DPI_EXPORT)
        plt.close(fig)

    print(f"3D surfaces saved. Count: "
          f"{len([f for f in os.listdir(plots_dir_3d) if f.lower().endswith('.png')])}")
except Exception as e:
    print("⚠️  Error while generating 3D surfaces:", e)
    traceback.print_exc()


# =====================================================================
# DOWNLOAD #1: TRAINING RESULTS
# =====================================================================
print("\nDOWNLOAD #1: TRAINING RESULTS")
print("-" * 80)
print("Creating ZIP archive of training files...")

training_files = [
    os.path.join(EXPORT_DIR, "train_predictions.xlsx"),
    os.path.join(EXPORT_DIR, "val_predictions.xlsx"),
    os.path.join(EXPORT_DIR, "test_predictions.xlsx"),
    os.path.join(EXPORT_DIR, "train_full_with_all_columns.xlsx"),
    os.path.join(EXPORT_DIR, "val_full_with_all_columns.xlsx"),
    os.path.join(EXPORT_DIR, "test_full_with_all_columns.xlsx"),
    os.path.join(EXPORT_DIR, "train_predictions.csv"),
    os.path.join(EXPORT_DIR, "val_predictions.csv"),
    os.path.join(EXPORT_DIR, "test_predictions.csv"),
    os.path.join(EXPORT_DIR, "train_full_with_all_columns.csv"),
    os.path.join(EXPORT_DIR, "val_full_with_all_columns.csv"),
    os.path.join(EXPORT_DIR, "test_full_with_all_columns.csv"),
    os.path.join(EXPORT_DIR, "model_statistics.txt"),
    os.path.join(EXPORT_DIR, "model_scheme.txt"),
    os.path.join(EXPORT_DIR, "mse_evolution.png"),
    os.path.join(EXPORT_DIR, "final_model.keras"),
    os.path.join(EXPORT_DIR, "final_model.h5"),
    os.path.join(EXPORT_DIR, "scaler_X.pkl"),
    os.path.join(EXPORT_DIR, "scaler_y.pkl"),
]

plots_dir = os.path.join(EXPORT_DIR, "plots")
if os.path.isdir(plots_dir):
    for fname in os.listdir(plots_dir):
        training_files.append(os.path.join(plots_dir, fname))

plots_dir_3d = os.path.join(EXPORT_DIR, "plots_3d")
if os.path.isdir(plots_dir_3d):
    for fname in os.listdir(plots_dir_3d):
        training_files.append(os.path.join(plots_dir_3d, fname))

training_files = [f for f in training_files if os.path.exists(f)]
print(f"Found {len(training_files)} training files")

try:
    zip_path_train = "training_results.zip"
    if os.path.exists(zip_path_train):
        os.remove(zip_path_train)

    with zipfile.ZipFile(zip_path_train, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in training_files:
            arcname = os.path.relpath(file_path, ".")
            zipf.write(file_path, arcname=arcname)

    if os.path.exists(zip_path_train):
        file_size_mb = os.path.getsize(zip_path_train) / (1024 * 1024)
        print(f"Created ZIP: training_results.zip ({file_size_mb:.2f} MB)")

        if use_colab:
            print("\nFile location in Colab:")
            print("  1. Click the 'Files' folder icon on the left sidebar")
            print("  2. Look for 'training_results.zip'")
            print("  3. Right-click it and select 'Download'")
            print("\nOr run:")
            print("  from google.colab import files")
            print("  files.download('training_results.zip')")
            print("\nDownload #1 Ready!")
        else:
            downloads_dir = os.path.expanduser("~/Downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            dest_path = os.path.join(downloads_dir, "training_results.zip")
            shutil.copy2(zip_path_train, dest_path)
            print(f"Download #1 Complete! Saved to {dest_path}")
    else:
        print("ZIP creation failed")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 80 + "\n")


# =====================================================================
# PHASE 10: NEW DATA PREDICTIONS
# =====================================================================
print("=" * 80)
print("PHASE 10: NEW DATA PREDICTIONS")
print("=" * 80)

# --- Empirical prediction interval calibration ---
def get_calibration_residuals(source="val"):
    source = str(source).strip().lower()
    if source == "val":
        return y_val_inv.reshape(-1) - y_val_pred.reshape(-1)
    elif source == "oof":
        return y_oof_true_inv.reshape(-1) - y_oof_pred_inv.reshape(-1)
    else:
        raise ValueError("PI_CALIBRATION must be 'val' or 'oof'")

cal_residuals = get_calibration_residuals(PI_CALIBRATION)
q_low = np.quantile(cal_residuals, PI_ALPHA / 2.0)
q_high = np.quantile(cal_residuals, 1.0 - PI_ALPHA / 2.0)
print(f"Empirical PI calibration ({PI_CALIBRATION}): "
      f"q_low={q_low:.4f}, q_high={q_high:.4f} (alpha={PI_ALPHA})")

print("\n📂 Upload new data for predictions (optional).")
new_data_loaded = False
new_df = None

if use_colab:
    try:
        uploaded_new = colab_files.upload()
        if len(uploaded_new) > 0:
            new_file_name = list(uploaded_new.keys())[0]
            new_df = load_table(new_file_name, sep=SEP)
            new_data_loaded = True
            print(f"✅ New data loaded. Shape: {new_df.shape}")
        else:
            print("⏭️  No new data uploaded. Skipping predictions.")
    except Exception as e:
        print(f"⚠️ Upload failed: {e}")
else:
    candidate_paths = ["new_data.tsv", "new_data.csv", "new_data.xlsx"]
    found = None
    for pth in candidate_paths:
        if os.path.exists(pth):
            found = pth
            break
    if found is not None:
        try:
            new_df = load_table(found, sep=SEP)
            new_data_loaded = True
            print(f"✅ New data loaded from {found}. Shape: {new_df.shape}")
        except Exception as e:
            print(f"⚠️ Load failed: {e}")
    else:
        print("⏭️  No new_data.* found. Skipping predictions.")

if new_data_loaded and new_df is not None:
    try:
        _show(new_df.head())

        n_cols_new = new_df.shape[1]
        new_labels_df = new_df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()

        if n_cols_new >= N_LABELS + N_INPUTS:
            new_inputs_df = new_df.iloc[:, N_LABELS:N_LABELS + N_INPUTS]
        else:
            raise ValueError(f"Insufficient columns: {n_cols_new} < {N_LABELS + N_INPUTS}")

        if new_inputs_df.shape[1] != X_train.shape[1]:
            if set(new_inputs_df.columns).issubset(set(inputs_columns)):
                new_inputs_df = new_inputs_df[inputs_columns]
            else:
                raise ValueError("Column mismatch between training inputs and new data inputs.")

        nan_in_features = new_inputs_df.isna().sum().sum()
        if nan_in_features > 0:
            print(f"NaNs in new features: {nan_in_features}. "
                  "Imputing with TRAIN means (fit on TRAIN only)...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(X_train_orig)
            new_inputs_imputed = imputer.transform(new_inputs_df)
            new_inputs_df = pd.DataFrame(new_inputs_imputed, columns=new_inputs_df.columns)

        new_X = new_inputs_df.values
        new_X_scaled = scaler_X.transform(new_X)

        print(f"Generating predictions for {len(new_X)} samples...")
        new_y_pred_scaled = model.predict(new_X_scaled, verbose=0)
        new_y_pred = scaler_y.inverse_transform(new_y_pred_scaled).reshape(-1)

        pi_lower = new_y_pred + q_low
        pi_upper = new_y_pred + q_high

        new_results_compact = pd.DataFrame(index=range(len(new_X)))
        if new_labels_df is not None and new_labels_df.shape[0] == len(new_X):
            new_results_compact = pd.concat(
                [new_results_compact, new_labels_df.reset_index(drop=True)], axis=1
            )
        new_results_compact = pd.concat(
            [new_results_compact, new_inputs_df.reset_index(drop=True)], axis=1
        )
        new_results_compact["Predicted_Value"] = new_y_pred
        new_results_compact["PI_Lower_95%"] = pi_lower
        new_results_compact["PI_Upper_95%"] = pi_upper
        new_results_compact["PI_Width"] = pi_upper - pi_lower
        new_results_compact["PI_Calibration_Source"] = PI_CALIBRATION

        new_has_actual = False
        if n_cols_new > N_LABELS + N_INPUTS:
            try:
                new_y_actual = new_df.iloc[:, N_LABELS + N_INPUTS].values.reshape(-1, 1)
                if np.isnan(new_y_actual).all():
                    print("Target column is entirely NaN. Skipping error calculations.")
                else:
                    if np.isnan(new_y_actual).any():
                        print("NaNs in new target. Imputing with TRAIN target mean...")
                        from sklearn.impute import SimpleImputer
                        imputer_y = SimpleImputer(strategy='mean')
                        imputer_y.fit(y_train_orig)
                        new_y_actual = imputer_y.transform(new_y_actual)
                    new_y_actual_flat = new_y_actual.reshape(-1)
                    new_has_actual = True
            except Exception as e:
                print(f"Could not extract target: {e}")

        if new_has_actual:
            residual = new_y_actual_flat - new_y_pred
            abs_err = np.abs(residual)
            with np.errstate(divide='ignore', invalid='ignore'):
                abs_pct = np.where(np.abs(new_y_actual_flat) > 1e-12,
                                   100.0 * abs_err / np.abs(new_y_actual_flat), np.nan)

            new_results_compact["Actual_Value"] = new_y_actual_flat
            new_results_compact["Prediction_Error"] = residual
            new_results_compact["Abs_Error"] = abs_err
            new_results_compact["Abs_Percent_Error"] = abs_pct

            within_pi = (new_y_actual_flat >= pi_lower) & (new_y_actual_flat <= pi_upper)
            new_results_compact["Within_95%_PI"] = within_pi.astype(int)

            new_metrics = compute_basic_metrics(new_y_actual_flat, new_y_pred)
            print("\nMetrics on new data:")
            for k, v in new_metrics.items():
                print(f"  {k}: {v}")
            pi_coverage = (within_pi.sum() / len(within_pi)) * 100
            print(f"\nEmpirical 95% PI Coverage: "
                  f"{within_pi.sum()}/{len(within_pi)} ({pi_coverage:.1f}%)")
        else:
            print("\nNo actual values. Predictions + PI only.")

        new_full_rows = new_df.copy()
        new_full_rows["Predicted_Value"] = new_y_pred
        new_full_rows["PI_Lower_95%"] = pi_lower
        new_full_rows["PI_Upper_95%"] = pi_upper
        new_full_rows["PI_Width"] = pi_upper - pi_lower
        new_full_rows["PI_Calibration_Source"] = PI_CALIBRATION
        if new_has_actual:
            new_full_rows["Actual_Value"] = new_y_actual_flat
            new_full_rows["Prediction_Error"] = residual
            new_full_rows["Abs_Error"] = abs_err
            new_full_rows["Abs_Percent_Error"] = abs_pct
            new_full_rows["Within_95%_PI"] = within_pi.astype(int)

        new_pred_path = os.path.join(EXPORT_DIR, "new_data_predictions.xlsx")
        new_pred_csv  = os.path.join(EXPORT_DIR, "new_data_predictions.csv")
        new_results_compact.to_excel(new_pred_path, index=False, engine='openpyxl')
        new_results_compact.to_csv(new_pred_csv, index=False)

        new_full_path = os.path.join(EXPORT_DIR, "new_data_full_with_all_columns.xlsx")
        new_full_csv  = os.path.join(EXPORT_DIR, "new_data_full_with_all_columns.csv")
        new_full_rows.to_excel(new_full_path, index=False, engine='openpyxl')
        new_full_rows.to_csv(new_full_csv, index=False)

        print("\nPredictions exported")
        print("\nPreview of predictions:")
        _show(new_results_compact.head(10))

    except Exception as e:
        print(f"Error processing new data: {e}")
        traceback.print_exc()


# =====================================================================
# DOWNLOAD #2: NEW DATA PREDICTIONS
# =====================================================================
print("\n" + "=" * 80)
print("DOWNLOAD #2: NEW DATA PREDICTIONS (if new data was uploaded)")
print("=" * 80)

prediction_files = [
    os.path.join(EXPORT_DIR, "new_data_predictions.xlsx"),
    os.path.join(EXPORT_DIR, "new_data_predictions.csv"),
    os.path.join(EXPORT_DIR, "new_data_full_with_all_columns.xlsx"),
    os.path.join(EXPORT_DIR, "new_data_full_with_all_columns.csv"),
]
prediction_files = [f for f in prediction_files if os.path.exists(f)]

if len(prediction_files) > 0:
    print(f"\nFound {len(prediction_files)} prediction files")
    print("Creating ZIP archive of prediction files...")

    try:
        zip_path_pred = "new_data_predictions.zip"
        if os.path.exists(zip_path_pred):
            os.remove(zip_path_pred)

        with zipfile.ZipFile(zip_path_pred, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in prediction_files:
                arcname = os.path.relpath(file_path, ".")
                zipf.write(file_path, arcname=arcname)

        if os.path.exists(zip_path_pred):
            file_size_mb = os.path.getsize(zip_path_pred) / (1024 * 1024)
            print(f"Created ZIP: new_data_predictions.zip ({file_size_mb:.2f} MB)")

            if use_colab:
                print("\nFile location in Colab:")
                print("  1. Click the 'Files' folder icon on the left sidebar")
                print("  2. Look for 'new_data_predictions.zip'")
                print("  3. Right-click it and select 'Download'")
                print("\nDownload #2 Ready!")
            else:
                downloads_dir = os.path.expanduser("~/Downloads")
                os.makedirs(downloads_dir, exist_ok=True)
                dest_path = os.path.join(downloads_dir, "new_data_predictions.zip")
                shutil.copy2(zip_path_pred, dest_path)
                print(f"Download #2 Complete! Saved to {dest_path}")
        else:
            print("ZIP creation failed")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
else:
    print("\nNo new data predictions available.")
    print("(Upload new data in the prediction section above to generate predictions)")

print("\n" + "=" * 80)
print("ALL OPERATIONS COMPLETED!")
print("=" * 80)
print("\nDownloads Summary:")
print("  Download #1: training_results.zip - Ready")
print("  Download #2: new_data_predictions.zip - Ready (if data uploaded)")
print("\nTo download files from Colab:")
print("  1. Click the 'Files' folder on the left sidebar")
print("  2. Find the ZIP file you need")
print("  3. Right-click and select 'Download'")
print("\nFiles are ready to use!")
