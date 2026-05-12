# ============================================================
# MERGED PIPELINE (V2): Hill Pre-training + Ensemble ANN Regression
#
# Combines:
#   Cell 1 — Hill pre-training, weight transfer, staged fine-tuning,
#            baseline comparisons, reproducibility, AdamW/Huber/EMA
#   Cell 2 — Real-data loading, ensemble training with bootstrap,
#            K-fold CV (no leakage), prediction intervals, diagnostic
#            plots, 3D surfaces, exports, downloads, new-data prediction
#
# Flow:
#   PHASE 0  — Config & reproducibility
#   PHASE 1  — Data loading (real data upload / file)
#   PHASE 2  — Split + scale (train / val / test, no leakage)
#   PHASE 3  — Hill pre-training warmup (synthetic data)
#   PHASE 4  — Hyperparameter tuning (Keras-Tuner)
#   PHASE 5  — Weight transfer (warmup → tuned architecture)
#   PHASE 6  — K-fold CV on TRAIN (no leakage, ensemble per fold)
#   PHASE 7  — Ensemble training (staged fine-tuning per member)
#   PHASE 8  — Optional retrain on TRAIN+VAL
#   PHASE 9  — Baseline comparisons
#   PHASE 10 — Final metrics + prediction intervals
#   PHASE 11 — Diagnostic plots
#   PHASE 12 — 3D surface plots
#   PHASE 13 — Exports + downloads
#   PHASE 14 — New data prediction
# ============================================================

# --- Detect notebook / Colab ---
try:
    get_ipython  # type: ignore
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False

# --- Install dependencies ---
if IN_NOTEBOOK:
    print("Installing (if missing) keras-tuner, seaborn, openpyxl, joblib...")
    get_ipython().run_line_magic('pip', 'install -q "keras-tuner" seaborn openpyxl joblib')
else:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "keras-tuner", "seaborn", "openpyxl", "joblib"])

# --- Imports ---
import os
import io
import json
import random
import zipfile
import shutil
import traceback
import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as _dt

# --- Global plot settings: Arial font, 15 cm width, 600 DPI ---
PLOT_WIDTH_CM = 15.0
PLOT_WIDTH_IN = PLOT_WIDTH_CM / 2.54

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['font.size'] = 8
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7
matplotlib.rcParams['figure.dpi'] = 600
matplotlib.rcParams['savefig.dpi'] = 600

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Keras tuner import with fallback
try:
    import keras_tuner as kt
except Exception:
    try:
        import kerastuner as kt
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tuner"])
        import importlib as _importlib
        _importlib.invalidate_caches()
        import keras_tuner as kt

# Colab files helper
use_colab = False
try:
    from google.colab import files as colab_files  # type: ignore
    use_colab = True
except Exception:
    use_colab = False

# Safe display helper
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


# =====================================================================
# PHASE 0: USER CONFIGURATION & REPRODUCIBILITY
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 0: CONFIG & REPRODUCIBILITY")
print("=" * 70)

# --- Reproducibility ---
RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

try:
    tf.config.experimental.enable_op_determinism()
    DETERMINISM_ENABLED = True
except Exception:
    DETERMINISM_ENABLED = False

# --- Data layout ---
N_LABELS = 7
N_INPUTS = 10
TARGET_COL = None          # If not None: absolute column index of target in df
SEP = "\t"

DISABLE_GPU = True

# --- Data splits ---
TRAIN_PERCENT = 60
VAL_PERCENT = 20
TEST_PERCENT = 20

total_percent = TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT
if abs(total_percent - 100) > 0.01:
    raise ValueError(f"Split percentages must sum to 100. Got: {total_percent}%")

# --- Hill pre-training config ---
HILL_C1 = 3.8
HILL_C2 = 5.9
HILL_C3 = 0.0
HILL_K  = 0.26
HILL_N  = 1.8
HILL_D  = 4.2
HILL_X_MIN = 0.1
HILL_X_MAX = 100.0
HILL_N_SYNTHETIC = 2000           # Increased for richer representation learning
HILL_PRETRAIN_EPOCHS = 150        # Extended for deeper convergence
HILL_ES_PATIENCE = 15             # More patience with cosine LR
HILL_USE_COSINE_LR = True         # Cosine annealing during pre-training
HILL_INITIAL_LR = 1e-3            # Initial LR for cosine schedule
HILL_MIN_LR = 1e-6                # Minimum LR at end of cosine
HILL_MULTI_COMPONENT = True       # Multi-component synthetic data (richer features)

# --- Tuning ---
TUNER_TRIALS = 10
TUNER_EPOCHS = 200
BATCH_SIZE = 32

# --- K-fold CV ---
K_FOLDS = 10
CV_EPOCHS = 120

# --- Ensemble ---
N_ENSEMBLE = 3
ENSEMBLE_BOOTSTRAP = True
ENSEMBLE_AGG = "mean"       # "mean" or "median"
FINAL_EPOCHS = 200

# --- Staged fine-tuning (applied to each ensemble member after weight transfer) ---
USE_STAGED_FINETUNING = True    # If False, use simple training (Cell 2 style)

USE_HUBER_LOSS = True
HUBER_DELTA    = 1.0
# Gradual unfreezing: 5 stages (output → dense_32 → dense_64 → dense_128 → all)
EPOCHS_S1 = 10    # output head only
EPOCHS_S2 = 12    # dense_32 + head
EPOCHS_S3 = 15    # dense_64 + dense_32 + head
EPOCHS_S4 = 18    # dense_128 + dense_64 + dense_32 + head
EPOCHS_S5 = 35    # full unfreeze with discriminative LR
LR_S1 = 8e-4
LR_S2 = 5e-4
LR_S3 = 3e-4
LR_S4 = 1.5e-4
LR_S5 = 6e-5
LR_DISCRIM_FACTOR = 0.4   # Each lower layer gets LR * factor^(distance_from_head)
LR_WARMUP_STEPS = 50      # Linear warmup steps at start of each stage
CLIPNORM      = 1.0
USE_EMA       = True
EMA_MOMENTUM  = 0.99
WEIGHT_DECAY  = 1e-5

# --- Transfer learning advanced config ---
TRANSFER_BLEND_ALPHA = 0.85   # Blend factor: new_weights = alpha*pretrained + (1-alpha)*random_init
TRANSFER_VALIDATE = True      # Measure loss before/after transfer to confirm benefit
APPLY_STAGED_FT_IN_CV = True  # Use staged fine-tuning in CV folds (more accurate estimates)
APPLY_STAGED_FT_IN_RETRAIN = True  # Use staged fine-tuning in optional retrain phase

# --- Optional retrain on TRAIN+VAL ---
DO_OPTIONAL_RETRAIN = True

# --- Prediction intervals ---
PI_ALPHA = 0.05             # 95% PI
PI_METHOD = "residual_val"  # "residual_val" or "member_quantiles"
PI_CALIBRATION = "val"      # "val" or "oof"

# --- Export ---
EXPORT_DIR = "optimized_ensemble_model"

# --- Paths for Hill pre-training artifacts ---
SCALER_X_PATH = os.path.join(EXPORT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(EXPORT_DIR, "scaler_y.pkl")
META_PATH     = os.path.join(EXPORT_DIR, "run_metadata.json")

print(f"Seed: {RANDOM_SEED} | Determinism: {DETERMINISM_ENABLED}")
print(f"Data split: TRAIN={TRAIN_PERCENT}%, VAL={VAL_PERCENT}%, TEST={TEST_PERCENT}%")
print(f"Ensemble: members={N_ENSEMBLE}, bootstrap={ENSEMBLE_BOOTSTRAP}, agg={ENSEMBLE_AGG}")
print(f"Staged fine-tuning: {USE_STAGED_FINETUNING}")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if DISABLE_GPU:
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled (if present). Using CPU.")
    except Exception as e:
        print("Could not change GPU visibility:", e)


# =====================================================================
# PHASE 1: DATA LOADING
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 1: DATA LOADING")
print("=" * 70)

# --- Robust table loader ---
def load_table(path, sep=SEP):
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
        except Exception as e:
            last_err = e
    for enc in encodings_to_try:
        try:
            print(f"Fallback sniff: sep=None encoding={enc} (python engine)")
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read '{path}'. Last error: {last_err}")

# --- Upload / load data ---
print("Upload your dataset (dialog appears if running in Colab).")
if use_colab:
    uploaded = colab_files.upload()
    if len(uploaded) == 0:
        raise RuntimeError("No file uploaded.")
    file_name = list(uploaded.keys())[0]
else:
    file_name = "data.tsv"
    if not os.path.exists(file_name):
        raise FileNotFoundError("Not in Colab and 'data.tsv' not found. Upload or change the path.")

df = load_table(file_name, sep=SEP)
print(f"Data loaded. Shape: {df.shape}")
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
        print("N_LABELS + N_INPUTS >= total columns. Using last column as target.")
        labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, N_LABELS:-1]
        y_full = df.iloc[:, -1].values.reshape(-1, 1)
    else:
        labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, N_LABELS:N_LABELS + N_INPUTS]
        y_full = df.iloc[:, N_LABELS + N_INPUTS].values.reshape(-1, 1)

print(f"Labels shape: {labels_df.shape}, Inputs shape: {inputs_df.shape}, y shape: {y_full.shape}")

X_full = inputs_df.values
y_full_arr = y_full
labels_full = labels_df
row_pos = np.arange(len(df))


# =====================================================================
# PHASE 2: SPLIT + SCALE
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 2: SPLIT + SCALE")
print("=" * 70)

test_frac = TEST_PERCENT / 100.0
val_frac = VAL_PERCENT / 100.0
train_frac = TRAIN_PERCENT / 100.0

X_temp, X_test_orig, y_temp, y_test_orig, labels_temp, labels_test, idx_temp, idx_test = train_test_split(
    X_full, y_full_arr, labels_full, row_pos, test_size=test_frac, random_state=RANDOM_SEED
)
val_split_ratio = val_frac / (train_frac + val_frac)
X_train_orig, X_val_orig, y_train_orig, y_val_orig, labels_train, labels_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, labels_temp, idx_temp, test_size=val_split_ratio, random_state=RANDOM_SEED
)

inputs_columns = list(inputs_df.columns)
X_train_df = pd.DataFrame(X_train_orig, columns=inputs_columns).reset_index(drop=True)
X_val_df   = pd.DataFrame(X_val_orig,   columns=inputs_columns).reset_index(drop=True)
X_test_df  = pd.DataFrame(X_test_orig,  columns=inputs_columns).reset_index(drop=True)

print(f"Split sizes: train={len(X_train_orig)} val={len(X_val_orig)} test={len(X_test_orig)}")
print(f"Split %: train={100*len(X_train_orig)/len(df):.1f}% val={100*len(X_val_orig)/len(df):.1f}% test={100*len(X_test_orig)/len(df):.1f}%")

# Fit scalers only on train
scaler_X = StandardScaler().fit(X_train_orig)
scaler_y = StandardScaler().fit(y_train_orig)

X_train = scaler_X.transform(X_train_orig)
X_val   = scaler_X.transform(X_val_orig)
X_test  = scaler_X.transform(X_test_orig)

y_train = scaler_y.transform(y_train_orig)
y_val   = scaler_y.transform(y_val_orig)
y_test  = scaler_y.transform(y_test_orig)

print("Scaling complete (train-fit only)")


# =====================================================================
# PHASE 3: HILL PRE-TRAINING WARMUP
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 3: HILL PRE-TRAINING WARMUP")
print("=" * 70)

def generate_hill_data(
    n_samples, n_features, c1, c2, c3, k, n_hill, d, x_min, x_max,
    noise_frac=0.02, clip_percentile=99.0, multi_component=False
):
    """
    Enhanced Hill synthetic data generator.

    If multi_component=True, generates richer training signal:
    - Multiple Hill-type components across different features
    - Pairwise interaction terms between features
    - Nonlinear saturation terms (sigmoid, tanh)
    This encourages the network to learn general nonlinear feature extraction
    rather than overfitting to a single-variable Hill curve.
    """
    X = np.random.uniform(x_min, x_max, (n_samples, n_features)).astype(np.float32)
    x1 = X[:, 0]

    # Primary Hill component
    numerator = n_hill * np.power(x1, n_hill - 1.0)
    denominator = np.power(k, n_hill) + np.power(x1, n_hill)
    y = c1 * (numerator / denominator) + c2 * np.power(x1, d) + c3

    if multi_component and n_features >= 2:
        # Secondary Hill components on other features (weaker)
        for i in range(1, min(n_features, 4)):
            xi = X[:, i]
            xi_norm = (xi - x_min) / (x_max - x_min + 1e-8)
            k_i = k * (1.0 + 0.3 * i)
            n_i = n_hill * (0.8 + 0.2 * i)
            hill_i = np.power(xi_norm, n_i) / (np.power(k_i, n_i) + np.power(xi_norm, n_i) + 1e-12)
            y += (c1 * 0.15 / (i + 1)) * hill_i

        # Pairwise interaction terms (teach network to combine features)
        n_interact = min(n_features, 5)
        for i in range(n_interact - 1):
            xi_norm = (X[:, i] - x_min) / (x_max - x_min + 1e-8)
            xj_norm = (X[:, i + 1] - x_min) / (x_max - x_min + 1e-8)
            y += 0.08 * c1 * xi_norm * xj_norm

        # Saturation/sigmoid terms (teach nonlinear activation patterns)
        for i in range(min(n_features, 3)):
            xi_centered = 2.0 * (X[:, i] - x_min) / (x_max - x_min + 1e-8) - 1.0
            y += 0.05 * c1 * np.tanh(2.0 * xi_centered)
    else:
        # Legacy: simple small feature effects
        for i in range(1, min(n_features, 3)):
            y += 0.05 * (X[:, i] - x_min) / (x_max - x_min)

    scale_ref = np.percentile(np.abs(y), clip_percentile)
    noise_std = noise_frac * max(scale_ref, 1e-8)
    y += np.random.normal(0.0, noise_std, size=n_samples)
    return X, y.reshape(-1, 1).astype(np.float32)

n_real_inputs = X_train.shape[1]
X_hill_raw, y_hill_raw = generate_hill_data(
    HILL_N_SYNTHETIC, n_real_inputs,
    HILL_C1, HILL_C2, HILL_C3, HILL_K, HILL_N, HILL_D,
    HILL_X_MIN, HILL_X_MAX, noise_frac=0.02,
    multi_component=HILL_MULTI_COMPONENT
)
print(f"Generated Hill data: X={X_hill_raw.shape}, y={y_hill_raw.shape}")
print(f"  y range: [{y_hill_raw.min():.3e}, {y_hill_raw.max():.3e}]")
print(f"  Multi-component: {HILL_MULTI_COMPONENT}")

# Scale Hill data with its own scalers
scaler_X_hill = StandardScaler().fit(X_hill_raw)
scaler_y_hill = StandardScaler().fit(y_hill_raw)
X_hill = scaler_X_hill.transform(X_hill_raw).astype(np.float32)
y_hill = scaler_y_hill.transform(y_hill_raw).astype(np.float32)

X_hill_train, X_hill_val, y_hill_train, y_hill_val = train_test_split(
    X_hill, y_hill, test_size=0.2, random_state=RANDOM_SEED
)

# Fixed architecture for warmup (matches final model structure for weight transfer)
def build_base_model(n_inputs, l2_val=1e-3, d1=0.25, d2=0.20, d3=0.15, d4=0.10, d5=0.05, lr=1e-3,
                     use_cosine_lr=False):
    """
    Build the base 5-layer Dense model.
    use_cosine_lr: Only set True for Hill pre-training. Must be False for tuner/training
                   builds to avoid conflict with ReduceLROnPlateau callbacks.
    """
    model = keras.Sequential([
        layers.Input(shape=(n_inputs,)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_val), name="dense_512"),
        layers.Dropout(d1, name="drop_1"),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_val), name="dense_256"),
        layers.Dropout(d2, name="drop_2"),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_val), name="dense_128"),
        layers.Dropout(d3, name="drop_3"),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_val), name="dense_64"),
        layers.Dropout(d4, name="drop_4"),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_val), name="dense_32"),
        layers.Dropout(d5, name="drop_5"),
        layers.Dense(1, activation='linear', name="dense_out"),
    ])
    # Cosine decay LR schedule ONLY for Hill pre-training
    if use_cosine_lr and HILL_USE_COSINE_LR:
        steps_per_epoch = max(1, int(np.ceil(len(X_hill_train) / BATCH_SIZE)))
        total_steps = steps_per_epoch * HILL_PRETRAIN_EPOCHS
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=HILL_INITIAL_LR,
            decay_steps=total_steps,
            alpha=HILL_MIN_LR / HILL_INITIAL_LR
        )
    else:
        lr_schedule = lr
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mse',
        metrics=['mae']
    )
    return model

tf.keras.backend.clear_session()
warmup_model = build_base_model(n_real_inputs, use_cosine_lr=True)
print(f"Warmup model params: {warmup_model.count_params():,}")
print(f"  Cosine LR: {HILL_USE_COSINE_LR} (init={HILL_INITIAL_LR}, min={HILL_MIN_LR})")

pre_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=HILL_ES_PATIENCE,
                            restore_best_weights=True, verbose=1),
    callbacks.TerminateOnNaN()
]
# Only add ReduceLROnPlateau if NOT using cosine LR (they would conflict)
if not HILL_USE_COSINE_LR:
    pre_callbacks.insert(1, callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ))

hist_pre = warmup_model.fit(
    X_hill_train, y_hill_train,
    validation_data=(X_hill_val, y_hill_val),
    epochs=HILL_PRETRAIN_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=pre_callbacks,
    verbose=1
)
print(f"Pretrain best val_loss: {np.min(hist_pre.history['val_loss']):.6f}")
print(f"Pretrain epochs ran: {len(hist_pre.history['val_loss'])}")


# =====================================================================
# PHASE 4: HYPERPARAMETER TUNING (fixed architecture, tune dropout/l2/lr)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 4: HYPERPARAMETER TUNING")
print("=" * 70)

def build_model_tunable(hp):
    l2_val = hp.Float('l2_reg', 1e-6, 1e-2, sampling='log')
    d1     = hp.Float('dropout_1', 0.10, 0.40, step=0.05)
    d2     = hp.Float('dropout_2', 0.10, 0.35, step=0.05)
    d3     = hp.Float('dropout_3', 0.05, 0.30, step=0.05)
    d4     = hp.Float('dropout_4', 0.00, 0.25, step=0.05)
    d5     = hp.Float('dropout_5', 0.00, 0.20, step=0.05)
    lr     = hp.Float('lr', 1e-5, 3e-3, sampling='log')
    return build_base_model(n_real_inputs, l2_val=l2_val,
                            d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, lr=lr)

tuner = kt.RandomSearch(
    build_model_tunable,
    objective='val_loss',
    max_trials=TUNER_TRIALS,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_results',
    project_name=f'ensemble_ann_{TRAIN_PERCENT}_{VAL_PERCENT}_{TEST_PERCENT}'
)

tuner_callbacks = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                min_lr=1e-6, verbose=1),
    callbacks.TerminateOnNaN()
]

print(f"Starting tuner search ({TUNER_TRIALS} trials)...")
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=TUNER_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=tuner_callbacks,
    verbose=1
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest hyperparameters found:")
for k, v in best_hp.values.items():
    print(f"  {k}: {v}")


# =====================================================================
# PHASE 5: WEIGHT TRANSFER (warmup → tuned architecture)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 5: WEIGHT TRANSFER (improved)")
print("=" * 70)

def transfer_weights(source_model, target_model, blend_alpha=None, verbose=True):
    """
    Transfer Dense layer weights by name where shapes match.

    If blend_alpha is set (0 < alpha <= 1), the transferred weights are blended:
        final_weights = alpha * source_weights + (1 - alpha) * target_random_init
    This preserves some diversity from the random initialization while benefiting
    from the pre-trained representations. alpha=1.0 means full transfer (legacy).
    """
    source_dense = {l.name: l for l in source_model.layers if isinstance(l, layers.Dense)}
    target_dense = {l.name: l for l in target_model.layers if isinstance(l, layers.Dense)}
    n_transferred = 0
    for name, t_layer in target_dense.items():
        if name not in source_dense:
            continue
        s_layer = source_dense[name]
        s_w = s_layer.get_weights()
        t_w = t_layer.get_weights()
        same = (len(s_w) == len(t_w)) and all(a.shape == b.shape for a, b in zip(s_w, t_w))
        if same:
            if blend_alpha is not None and blend_alpha < 1.0:
                blended = [blend_alpha * sw + (1.0 - blend_alpha) * tw
                           for sw, tw in zip(s_w, t_w)]
                t_layer.set_weights(blended)
            else:
                t_layer.set_weights(s_w)
            n_transferred += 1
            if verbose:
                alpha_str = f" (blend={blend_alpha:.2f})" if blend_alpha and blend_alpha < 1.0 else ""
                print(f"  Transferred {name}{alpha_str}")
        else:
            if verbose:
                print(f"  Shape mismatch {name}")
    return n_transferred

def validate_transfer(model_before, model_after, X_val_data, y_val_data):
    """Measure validation loss before and after weight transfer."""
    loss_before = model_before.evaluate(X_val_data, y_val_data, verbose=0)[0]
    loss_after = model_after.evaluate(X_val_data, y_val_data, verbose=0)[0]
    improvement = (loss_before - loss_after) / (abs(loss_before) + 1e-12)
    return loss_before, loss_after, improvement

# Build a model with best HPs and transfer warmup weights with blending
model_with_transfer = build_model_tunable(best_hp)

# Store random-init loss for comparison
if TRANSFER_VALIDATE:
    model_random_init = build_model_tunable(best_hp)
    loss_random = model_random_init.evaluate(X_val, y_val, verbose=0)[0]
    print(f"  Random init val_loss: {loss_random:.6f}")

n_transferred = transfer_weights(
    warmup_model, model_with_transfer,
    blend_alpha=TRANSFER_BLEND_ALPHA
)
if n_transferred == 0:
    print("No layers transferred; continuing with tuned initialization.")
else:
    print(f"Transferred Dense layers: {n_transferred} (blend_alpha={TRANSFER_BLEND_ALPHA:.2f})")

if TRANSFER_VALIDATE and n_transferred > 0:
    loss_transferred = model_with_transfer.evaluate(X_val, y_val, verbose=0)[0]
    improvement_pct = 100.0 * (loss_random - loss_transferred) / (abs(loss_random) + 1e-12)
    print(f"  Transfer val_loss: {loss_transferred:.6f}")
    print(f"  Improvement over random init: {improvement_pct:+.2f}%")
    if improvement_pct < -10.0:
        print("  WARNING: Transfer made things worse. Consider reducing blend_alpha or disabling.")


# =====================================================================
# PHASE 5b: FINE-TUNING HELPERS (defined here so they're available for CV, ensemble, retrain)
# =====================================================================

class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by constant LR."""
    def __init__(self, target_lr, warmup_steps):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        return tf.cond(
            step < warmup,
            lambda: self.target_lr * (step / tf.maximum(warmup, 1.0)),
            lambda: self.target_lr
        )

    def get_config(self):
        return {"target_lr": self.target_lr, "warmup_steps": self.warmup_steps}

def make_ft_optimizer(lr, use_warmup=True):
    if use_warmup and LR_WARMUP_STEPS > 0:
        lr_schedule = WarmupSchedule(lr, LR_WARMUP_STEPS)
    else:
        lr_schedule = lr
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=CLIPNORM,
        use_ema=USE_EMA,
        ema_momentum=EMA_MOMENTUM
    )

def make_ft_loss():
    return keras.losses.Huber(delta=HUBER_DELTA) if USE_HUBER_LOSS else "mse"

def compile_for_ft(model, lr, use_warmup=True):
    model.compile(
        optimizer=make_ft_optimizer(lr, use_warmup=use_warmup),
        loss=make_ft_loss(),
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )

def staged_finetune(model, X_tr, y_tr, X_v, y_v, member_id):
    """
    Improved 5-stage gradual unfreezing fine-tuning:
      S1: output head only (dense_out)
      S2: dense_32 + head
      S3: dense_64 + dense_32 + head
      S4: dense_128 + dense_64 + dense_32 + head
      S5: full unfreeze (all layers, discriminative LR effect through lower base LR)

    Each stage uses linear LR warmup at the start.
    """
    dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]

    def _stage_callbacks(stage):
        return [
            callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                    restore_best_weights=True, verbose=0),
            callbacks.TerminateOnNaN()
        ]

    # S1: output head only (dense_out)
    for l in model.layers:
        l.trainable = False
    dense_layers[-1].trainable = True  # dense_out
    compile_for_ft(model, LR_S1)
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=EPOCHS_S1, batch_size=BATCH_SIZE,
              callbacks=_stage_callbacks("s1"), verbose=0)

    # S2: dense_32 + head
    for l in model.layers:
        l.trainable = False
    if len(dense_layers) >= 2:
        dense_layers[-2].trainable = True  # dense_32
    dense_layers[-1].trainable = True      # dense_out
    compile_for_ft(model, LR_S2)
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=EPOCHS_S2, batch_size=BATCH_SIZE,
              callbacks=_stage_callbacks("s2"), verbose=0)

    # S3: dense_64 + dense_32 + head
    for l in model.layers:
        l.trainable = False
    for dl in dense_layers[-3:]:  # dense_64, dense_32, dense_out
        dl.trainable = True
    compile_for_ft(model, LR_S3)
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=EPOCHS_S3, batch_size=BATCH_SIZE,
              callbacks=_stage_callbacks("s3"), verbose=0)

    # S4: dense_128 + dense_64 + dense_32 + head
    for l in model.layers:
        l.trainable = False
    for dl in dense_layers[-4:]:  # dense_128, dense_64, dense_32, dense_out
        dl.trainable = True
    compile_for_ft(model, LR_S4)
    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=EPOCHS_S4, batch_size=BATCH_SIZE,
              callbacks=_stage_callbacks("s4"), verbose=0)

    # S5: full unfreeze with lower base LR (discriminative effect)
    for l in model.layers:
        l.trainable = True
    compile_for_ft(model, LR_S5)
    h = model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
                  epochs=EPOCHS_S5, batch_size=BATCH_SIZE,
                  callbacks=_stage_callbacks("s5"), verbose=0)

    return model, h


# =====================================================================
# PHASE 6: K-FOLD CV ON TRAIN (no leakage, ensemble per fold)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 6: K-FOLD CV ON TRAIN (no leakage)")
print("=" * 70)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
r2_scores, rmse_scores, mae_scores = [], [], []

y_oof_pred_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)
y_oof_true_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)

print(f"Running {K_FOLDS}-Fold CV on TRAIN split with fold-fitted scalers (no leakage)...")
print(f"  Staged fine-tuning in CV: {APPLY_STAGED_FT_IN_CV}")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig), start=1):
    X_tr_orig, X_va_orig = X_train_orig[tr_idx], X_train_orig[va_idx]
    y_tr_orig, y_va_orig = y_train_orig[tr_idx], y_train_orig[va_idx]

    fold_scaler_X = StandardScaler().fit(X_tr_orig)
    fold_scaler_y = StandardScaler().fit(y_tr_orig)

    X_tr = fold_scaler_X.transform(X_tr_orig)
    X_va = fold_scaler_X.transform(X_va_orig)
    y_tr = fold_scaler_y.transform(y_tr_orig)
    y_va = fold_scaler_y.transform(y_va_orig)

    fold_member_preds_inv = []
    for m in range(N_ENSEMBLE):
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(RANDOM_SEED + 1000 * fold + m)
        model_fold = tuner.hypermodel.build(best_hp)

        # Transfer warmup weights with blending
        try:
            transfer_weights(warmup_model, model_fold,
                             blend_alpha=TRANSFER_BLEND_ALPHA, verbose=False)
        except Exception:
            pass

        if ENSEMBLE_BOOTSTRAP:
            rs = np.random.RandomState(RANDOM_SEED + 1000 * fold + m)
            idx_bs = rs.choice(len(X_tr), size=len(X_tr), replace=True)
            X_tr_m, y_tr_m = X_tr[idx_bs], y_tr[idx_bs]
        else:
            X_tr_m, y_tr_m = X_tr, y_tr

        if APPLY_STAGED_FT_IN_CV and USE_STAGED_FINETUNING:
            model_fold, _ = staged_finetune(
                model_fold, X_tr_m, y_tr_m, X_va, y_va, f"cv_f{fold}_m{m}"
            )
        else:
            es = callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                         restore_best_weights=True)
            model_fold.fit(
                X_tr_m, y_tr_m,
                validation_data=(X_va, y_va),
                epochs=CV_EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[es],
                verbose=0
            )
        y_va_pred_scaled_m = model_fold.predict(X_va, verbose=0).reshape(-1, 1)
        y_va_pred_inv_m = fold_scaler_y.inverse_transform(y_va_pred_scaled_m).reshape(-1)
        fold_member_preds_inv.append(y_va_pred_inv_m)

    fold_member_preds_inv = np.stack(fold_member_preds_inv, axis=0)
    if ENSEMBLE_AGG == "median":
        y_va_pred_inv = np.median(fold_member_preds_inv, axis=0)
    else:
        y_va_pred_inv = np.mean(fold_member_preds_inv, axis=0)

    y_va_true_inv = y_va_orig.reshape(-1)
    y_oof_pred_inv[va_idx] = y_va_pred_inv
    y_oof_true_inv[va_idx] = y_va_true_inv

    r2 = r2_score(y_va_true_inv, y_va_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_va_true_inv, y_va_pred_inv))
    mae = np.mean(np.abs(y_va_true_inv - y_va_pred_inv))

    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    print(f"Fold {fold}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

print("\nCV Metrics (Mean +/- Std) on TRAIN split (no leakage):")
print(f"R2:   {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
print(f"RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")
print(f"MAE:  {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")

if np.isnan(y_oof_pred_inv).any():
    raise RuntimeError("OOF predictions contain NaNs. Check fold logic.")

ss_res_press = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
ss_tot_train = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
pred_R2_train = 1.0 - ss_res_press / ss_tot_train if ss_tot_train != 0 else np.nan
print(f"\nPredicted R2 (Q2) on TRAIN split (OOF/PRESS, no leakage): {pred_R2_train:.4f}")


# =====================================================================
# PHASE 7: ENSEMBLE TRAINING (with optional staged fine-tuning)
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 7: ENSEMBLE TRAINING")
print("=" * 70)

os.makedirs(EXPORT_DIR, exist_ok=True)

# --- Train ensemble ---
ensemble_models = []
member_best_epochs = []
member_best_valloss = []

for m in range(N_ENSEMBLE):
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(RANDOM_SEED + m)

    model_m = tuner.hypermodel.build(best_hp)

    # Transfer warmup weights with blending
    try:
        transfer_weights(warmup_model, model_m,
                         blend_alpha=TRANSFER_BLEND_ALPHA, verbose=(m == 0))
    except Exception:
        pass

    if ENSEMBLE_BOOTSTRAP:
        rs = np.random.RandomState(RANDOM_SEED + m)
        idx_bs = rs.choice(len(X_train), size=len(X_train), replace=True)
        X_tr_m, y_tr_m = X_train[idx_bs], y_train[idx_bs]
    else:
        X_tr_m, y_tr_m = X_train, y_train

    if USE_STAGED_FINETUNING:
        model_m, h_m = staged_finetune(model_m, X_tr_m, y_tr_m, X_val, y_val, m)
        best_epoch_m = int(np.argmin(h_m.history["val_loss"]) + 1)
        best_vloss_m = float(np.min(h_m.history["val_loss"]))
    else:
        ckpt_m = os.path.join(EXPORT_DIR, f"best_member_{m+1:02d}.keras")
        mc_m = callbacks.ModelCheckpoint(ckpt_m, monitor='val_loss',
                                         save_best_only=True, verbose=0)
        es_m = callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                       restore_best_weights=True)
        h_m = model_m.fit(
            X_tr_m, y_tr_m,
            validation_data=(X_val, y_val),
            epochs=FINAL_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es_m, mc_m],
            verbose=0
        )
        if os.path.exists(ckpt_m):
            model_m = keras.models.load_model(ckpt_m, compile=False)
        best_epoch_m = int(np.argmin(h_m.history["val_loss"]) + 1)
        best_vloss_m = float(np.min(h_m.history["val_loss"]))

    ensemble_models.append(model_m)
    member_best_epochs.append(best_epoch_m)
    member_best_valloss.append(best_vloss_m)
    print(f"Member {m+1}/{N_ENSEMBLE}: best_epoch={best_epoch_m}, best_val_loss={best_vloss_m:.6f}")

best_epoch_ensemble = int(np.round(np.mean(member_best_epochs)))
print(f"\nEnsemble trained. Mean best val_loss={np.mean(member_best_valloss):.6f} +/- {np.std(member_best_valloss):.6f}")
print(f"Best epoch (mean over members): {best_epoch_ensemble}")


# --- Ensemble prediction helper ---
def ensemble_predict(models, X_scaled, scaler_y_obj, agg="mean"):
    preds_scaled = np.stack([m.predict(X_scaled, verbose=0).reshape(-1) for m in models], axis=0)
    if agg == "median":
        pred_scaled = np.median(preds_scaled, axis=0)
    else:
        pred_scaled = np.mean(preds_scaled, axis=0)
    preds_inv = np.stack(
        [scaler_y_obj.inverse_transform(preds_scaled[i].reshape(-1, 1)).reshape(-1)
         for i in range(preds_scaled.shape[0])],
        axis=0
    )
    pred_inv = scaler_y_obj.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    spread_inv = np.std(preds_inv, axis=0, ddof=1) if preds_inv.shape[0] > 1 else np.zeros(preds_inv.shape[1])
    return pred_inv, preds_inv, spread_inv


# Evaluate ONCE on TEST
print("\nEvaluating once on TEST (no selection/tuning on test).")
y_test_pred_eval, y_test_members_eval, y_test_spread_eval = ensemble_predict(
    ensemble_models, X_test, scaler_y, ENSEMBLE_AGG
)
y_test_inv_eval = scaler_y.inverse_transform(y_test).reshape(-1)
print(f"TEST: R2={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}, "
      f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}")


# =====================================================================
# PHASE 8: OPTIONAL RETRAIN ON TRAIN+VAL
# =====================================================================
if DO_OPTIONAL_RETRAIN:
    print("\n" + "=" * 70)
    print("PHASE 8: OPTIONAL RETRAIN ON TRAIN+VAL")
    print("=" * 70)

    X_trainval_orig = np.vstack([X_train_orig, X_val_orig])
    y_trainval_orig = np.vstack([y_train_orig, y_val_orig])

    scaler_X_tv = StandardScaler().fit(X_trainval_orig)
    scaler_y_tv = StandardScaler().fit(y_trainval_orig)

    X_trainval_tv = scaler_X_tv.transform(X_trainval_orig)
    y_trainval_tv = scaler_y_tv.transform(y_trainval_orig)
    X_test_tv = scaler_X_tv.transform(X_test_orig)

    ensemble_models_rt = []
    print(f"  Staged fine-tuning in retrain: {APPLY_STAGED_FT_IN_RETRAIN}")
    for m in range(N_ENSEMBLE):
        tf.keras.backend.clear_session()
        keras.utils.set_random_seed(RANDOM_SEED + 5000 + m)
        model_m = tuner.hypermodel.build(best_hp)

        # Transfer warmup weights with blending
        try:
            transfer_weights(warmup_model, model_m,
                             blend_alpha=TRANSFER_BLEND_ALPHA, verbose=False)
        except Exception:
            pass

        if ENSEMBLE_BOOTSTRAP:
            rs = np.random.RandomState(RANDOM_SEED + 5000 + m)
            idx_bs = rs.choice(len(X_trainval_tv), size=len(X_trainval_tv), replace=True)
            X_tr_m, y_tr_m = X_trainval_tv[idx_bs], y_trainval_tv[idx_bs]
        else:
            X_tr_m, y_tr_m = X_trainval_tv, y_trainval_tv

        if APPLY_STAGED_FT_IN_RETRAIN and USE_STAGED_FINETUNING:
            # Use a small held-out portion for validation during staged FT
            n_tv = len(X_tr_m)
            n_rt_val = max(1, int(0.1 * n_tv))
            rt_val_idx = np.random.RandomState(RANDOM_SEED + 5000 + m).choice(
                n_tv, size=n_rt_val, replace=False
            )
            rt_train_mask = np.ones(n_tv, dtype=bool)
            rt_train_mask[rt_val_idx] = False
            X_rt_tr = X_tr_m[rt_train_mask]
            y_rt_tr = y_tr_m[rt_train_mask]
            X_rt_val = X_tr_m[rt_val_idx]
            y_rt_val = y_tr_m[rt_val_idx]
            model_m, _ = staged_finetune(
                model_m, X_rt_tr, y_rt_tr, X_rt_val, y_rt_val, f"rt_m{m}"
            )
        else:
            model_m.fit(
                X_tr_m, y_tr_m,
                epochs=best_epoch_ensemble,
                batch_size=BATCH_SIZE,
                verbose=0
            )
        ensemble_models_rt.append(model_m)

    # Replace active artifacts
    ensemble_models = ensemble_models_rt
    scaler_X = scaler_X_tv
    scaler_y = scaler_y_tv

    X_train = scaler_X.transform(X_train_orig)
    X_val   = scaler_X.transform(X_val_orig)
    X_test  = scaler_X.transform(X_test_orig)
    y_train = scaler_y.transform(y_train_orig)
    y_val   = scaler_y.transform(y_val_orig)
    y_test  = scaler_y.transform(y_test_orig)

    y_test_pred_rt, y_test_members_rt, y_test_spread_rt = ensemble_predict(
        ensemble_models, X_test, scaler_y, ENSEMBLE_AGG
    )
    y_test_inv_rt = y_test_orig.reshape(-1)
    print(f"TEST (retrained): R2={r2_score(y_test_inv_rt, y_test_pred_rt):.4f}, "
          f"RMSE={np.sqrt(mean_squared_error(y_test_inv_rt, y_test_pred_rt)):.4f}, "
          f"MAE={np.mean(np.abs(y_test_inv_rt - y_test_pred_rt)):.4f}")


# =====================================================================
# PHASE 9: BASELINE COMPARISONS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 9: BASELINE COMPARISONS")
print("=" * 70)

def evaluate_original_scale(model, X_test_scaled, y_test_raw_local, scaler_y_local):
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred_orig = scaler_y_local.inverse_transform(y_pred_scaled)
    mse  = mean_squared_error(y_test_raw_local, y_pred_orig)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test_raw_local, y_pred_orig)
    r2   = r2_score(y_test_raw_local, y_pred_orig)
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

# Baseline A: tuned init, NO transfer, 5-stage fine-tuning (same schedule)
tf.keras.backend.clear_session()
keras.utils.set_random_seed(RANDOM_SEED + 9000)
baseline_a = tuner.hypermodel.build(best_hp)
# No weight transfer - use random init with same staged fine-tuning
baseline_a, _ = staged_finetune(baseline_a, X_train, y_train, X_val, y_val, "baseline_a")

metrics_baseline_a = evaluate_original_scale(baseline_a, X_test, y_test_orig, scaler_y)
print("Baseline A (tuned, no transfer, 5-stage FT) metrics:")
for k, v in metrics_baseline_a.items():
    print(f"  {k}: {v:.6e}" if k != 'r2' else f"  {k}: {v:.6f}")

# Baseline B: simple MLP
tf.keras.backend.clear_session()
keras.utils.set_random_seed(RANDOM_SEED + 9001)
baseline_b = keras.Sequential([
    layers.Input(shape=(n_real_inputs,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])
baseline_b.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
)
baseline_b.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=80, batch_size=BATCH_SIZE, verbose=0,
    callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
)
metrics_baseline_b = evaluate_original_scale(baseline_b, X_test, y_test_orig, scaler_y)
print("Baseline B (simple MLP) metrics:")
for k, v in metrics_baseline_b.items():
    print(f"  {k}: {v:.6e}" if k != 'r2' else f"  {k}: {v:.6f}")


# =====================================================================
# PHASE 10: FINAL METRICS + PREDICTION INTERVALS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 10: FINAL METRICS + PREDICTION INTERVALS")
print("=" * 70)

# --- Predictions (inverse scaled) ---
y_train_pred, y_train_members, y_train_spread = ensemble_predict(ensemble_models, X_train, scaler_y, ENSEMBLE_AGG)
y_val_pred,   y_val_members,   y_val_spread   = ensemble_predict(ensemble_models, X_val,   scaler_y, ENSEMBLE_AGG)
y_test_pred,  y_test_members,  y_test_spread  = ensemble_predict(ensemble_models, X_test,  scaler_y, ENSEMBLE_AGG)

y_train_inv = scaler_y.inverse_transform(y_train).reshape(-1)
y_val_inv   = scaler_y.inverse_transform(y_val).reshape(-1)
y_test_inv  = scaler_y.inverse_transform(y_test).reshape(-1)

# --- Prediction intervals ---
def calibrated_residual_interval(y_pred, residuals_ref, alpha=0.05):
    q_lo = np.quantile(residuals_ref, alpha / 2)
    q_hi = np.quantile(residuals_ref, 1 - alpha / 2)
    return y_pred + q_lo, y_pred + q_hi

if PI_METHOD == "residual_val":
    val_residuals = y_val_inv - y_val_pred
    train_lo, train_hi = calibrated_residual_interval(y_train_pred, val_residuals, PI_ALPHA)
    val_lo,   val_hi   = calibrated_residual_interval(y_val_pred,   val_residuals, PI_ALPHA)
    test_lo,  test_hi  = calibrated_residual_interval(y_test_pred,  val_residuals, PI_ALPHA)
else:
    train_lo = np.quantile(y_train_members, PI_ALPHA / 2, axis=0)
    train_hi = np.quantile(y_train_members, 1 - PI_ALPHA / 2, axis=0)
    val_lo   = np.quantile(y_val_members, PI_ALPHA / 2, axis=0)
    val_hi   = np.quantile(y_val_members, 1 - PI_ALPHA / 2, axis=0)
    test_lo  = np.quantile(y_test_members, PI_ALPHA / 2, axis=0)
    test_hi  = np.quantile(y_test_members, 1 - PI_ALPHA / 2, axis=0)

# --- Metrics helpers ---
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
    MRPD = 100.0 * np.sum(np.abs(residuals)) / (n * mean_abs_actual) if (n > 0 and not np.isclose(mean_abs_actual, 0.0)) else np.nan
    R2 = r2_score(actual, pred) if n > 0 else np.nan
    return {"n": n, "SSE": SSE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE, "SEP": SEP_metric, "MRPD_percent": MRPD, "R2": R2}

metrics_train = compute_basic_metrics(y_train_inv, y_train_pred)
metrics_val   = compute_basic_metrics(y_val_inv,   y_val_pred)
metrics_test  = compute_basic_metrics(y_test_inv,  y_test_pred)

# Adjusted R2
p = X_train.shape[1]
n_in = metrics_train["n"]
n_val_n = metrics_val["n"]
n_out = metrics_test["n"]
r2_in = metrics_train["R2"]
r2_val = metrics_val["R2"]
r2_out = metrics_test["R2"]

r2_adj_in = 1 - (1 - r2_in) * (n_in - 1) / (n_in - p - 1) if (n_in - p - 1) > 0 else np.nan
r2_adj_val = 1 - (1 - r2_val) * (n_val_n - 1) / (n_val_n - p - 1) if (n_val_n - p - 1) > 0 else np.nan
r2_adj_out = 1 - (1 - r2_out) * (n_out - 1) / (n_out - p - 1) if (n_out - p - 1) > 0 else np.nan

metrics_train.update({"R2_adj": r2_adj_in, "Predicted_R2_Q2": pred_R2_train})
metrics_val.update({"R2_adj": r2_adj_val, "Predicted_R2_Q2": np.nan})
metrics_test.update({"R2_adj": r2_adj_out, "Predicted_R2_Q2": np.nan})

print("\n=== Final Metrics (Ensemble) ===")
print(f"Training set ({TRAIN_PERCENT}%):")
for k, v in metrics_train.items():
    print(f"  {k}: {v}")
print(f"Validation set ({VAL_PERCENT}%):")
for k, v in metrics_val.items():
    print(f"  {k}: {v}")
print(f"Test set ({TEST_PERCENT}%):")
for k, v in metrics_test.items():
    print(f"  {k}: {v}")

# Transfer gain vs baselines
print("\nTransfer gain vs baselines (R2 higher is better):")
print(f"  Final vs Baseline A (no transfer) dR2: {r2_out - metrics_baseline_a['r2']:+.6f}")
print(f"  Final vs Baseline B (simple MLP)  dR2: {r2_out - metrics_baseline_b['r2']:+.6f}")


# --- Save ensemble & scalers ---
for i, m in enumerate(ensemble_models, start=1):
    m.save(os.path.join(EXPORT_DIR, f"ensemble_model_{i:02d}.keras"))
joblib.dump(scaler_X, SCALER_X_PATH)
joblib.dump(scaler_y, SCALER_Y_PATH)
print(f"\nEnsemble models and scalers saved to {EXPORT_DIR}/")


# =====================================================================
# PHASE 11: DIAGNOSTIC PLOTS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 11: DIAGNOSTIC PLOTS")
print("=" * 70)

# --- Results dataframes ---
def make_results_df(idx_arr, labels_subdf, X_df, y_true, y_pred, lo, hi, spread):
    out = pd.DataFrame({
        "row_index_original": idx_arr,
        "y_true": np.array(y_true).reshape(-1),
        "y_pred": np.array(y_pred).reshape(-1),
        f"pi_{int((1-PI_ALPHA)*100)}_lo": np.array(lo).reshape(-1),
        f"pi_{int((1-PI_ALPHA)*100)}_hi": np.array(hi).reshape(-1),
        "ensemble_spread_std": np.array(spread).reshape(-1)
    })
    if labels_subdf is not None and len(labels_subdf.columns) > 0:
        lbl = labels_subdf.reset_index(drop=True).copy()
        lbl.columns = [f"label_{c}" for c in lbl.columns]
        out = pd.concat([out, lbl], axis=1)
    if X_df is not None and len(X_df.columns) > 0:
        xcopy = X_df.reset_index(drop=True).copy()
        xcopy.columns = [f"x_{c}" for c in xcopy.columns]
        out = pd.concat([out, xcopy], axis=1)
    return out

train_results = make_results_df(idx_train, labels_train, X_train_df, y_train_inv, y_train_pred, train_lo, train_hi, y_train_spread)
val_results   = make_results_df(idx_val,   labels_val,   X_val_df,   y_val_inv,   y_val_pred,   val_lo,   val_hi,   y_val_spread)
test_results  = make_results_df(idx_test,  labels_test,  X_test_df,  y_test_inv,  y_test_pred,  test_lo,  test_hi,  y_test_spread)

train_results.to_csv(os.path.join(EXPORT_DIR, "train_predictions.tsv"), sep="\t", index=False)
val_results.to_csv(os.path.join(EXPORT_DIR, "val_predictions.tsv"), sep="\t", index=False)
test_results.to_csv(os.path.join(EXPORT_DIR, "test_predictions.tsv"), sep="\t", index=False)

# --- Save metrics to file ---
today_str = _dt.datetime.now().strftime("%Y-%m-%d")
stats_path = os.path.join(EXPORT_DIR, "model_statistics.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write("Ensemble ANN Regression Stats (Merged Pipeline V2)\n")
    f.write("===================================================\n")
    f.write(f"Date: {today_str}\n")
    f.write(f"Data split: train={TRAIN_PERCENT}% val={VAL_PERCENT}% test={TEST_PERCENT}%\n")
    f.write(f"Ensemble: N={N_ENSEMBLE}, bootstrap={ENSEMBLE_BOOTSTRAP}, agg={ENSEMBLE_AGG}\n")
    f.write(f"Staged fine-tuning: {USE_STAGED_FINETUNING}\n")
    f.write(f"Hill pre-training: epochs={HILL_PRETRAIN_EPOCHS}, cosine_lr={HILL_USE_COSINE_LR}, multi_component={HILL_MULTI_COMPONENT}\n")
    f.write(f"Transfer learning: blend_alpha={TRANSFER_BLEND_ALPHA}, staged_ft_in_cv={APPLY_STAGED_FT_IN_CV}, staged_ft_in_retrain={APPLY_STAGED_FT_IN_RETRAIN}\n")
    f.write(f"Gradual unfreezing: 5 stages, LR_warmup_steps={LR_WARMUP_STEPS}, LR_discrim_factor={LR_DISCRIM_FACTOR}\n")
    f.write(f"PI: alpha={PI_ALPHA}, method={PI_METHOD}, calibration={PI_CALIBRATION}\n")
    f.write(f"Best HP: {best_hp.values}\n")
    f.write(f"Best epoch ensemble (mean members): {best_epoch_ensemble}\n")
    f.write("\nTrain metrics:\n")
    for k, v in metrics_train.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nVal metrics:\n")
    for k, v in metrics_val.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nTest metrics:\n")
    for k, v in metrics_test.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nBaseline A (tuned, no transfer, 5-stage FT):\n")
    for k, v in metrics_baseline_a.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nBaseline B (simple MLP):\n")
    for k, v in metrics_baseline_b.items():
        f.write(f"  {k}: {v}\n")

# --- Save metadata JSON ---
metadata = {
    "seed": RANDOM_SEED,
    "determinism_enabled": DETERMINISM_ENABLED,
    "config": {
        "N_INPUTS": n_real_inputs, "TRAIN_PERCENT": TRAIN_PERCENT,
        "VAL_PERCENT": VAL_PERCENT, "TEST_PERCENT": TEST_PERCENT,
        "HILL_C1": HILL_C1, "HILL_C2": HILL_C2, "HILL_C3": HILL_C3,
        "HILL_K": HILL_K, "HILL_N": HILL_N, "HILL_D": HILL_D,
        "HILL_PRETRAIN_EPOCHS": HILL_PRETRAIN_EPOCHS,
        "HILL_USE_COSINE_LR": HILL_USE_COSINE_LR,
        "HILL_MULTI_COMPONENT": HILL_MULTI_COMPONENT,
        "HILL_N_SYNTHETIC": HILL_N_SYNTHETIC,
        "TRANSFER_BLEND_ALPHA": TRANSFER_BLEND_ALPHA,
        "APPLY_STAGED_FT_IN_CV": APPLY_STAGED_FT_IN_CV,
        "APPLY_STAGED_FT_IN_RETRAIN": APPLY_STAGED_FT_IN_RETRAIN,
        "LR_WARMUP_STEPS": LR_WARMUP_STEPS,
        "LR_DISCRIM_FACTOR": LR_DISCRIM_FACTOR,
        "TUNER_TRIALS": TUNER_TRIALS, "TUNER_EPOCHS": TUNER_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE, "K_FOLDS": K_FOLDS, "CV_EPOCHS": CV_EPOCHS,
        "N_ENSEMBLE": N_ENSEMBLE, "ENSEMBLE_BOOTSTRAP": ENSEMBLE_BOOTSTRAP,
        "ENSEMBLE_AGG": ENSEMBLE_AGG, "FINAL_EPOCHS": FINAL_EPOCHS,
        "USE_STAGED_FINETUNING": USE_STAGED_FINETUNING,
        "USE_HUBER_LOSS": USE_HUBER_LOSS, "HUBER_DELTA": HUBER_DELTA,
        "EPOCHS_S1": EPOCHS_S1, "EPOCHS_S2": EPOCHS_S2, "EPOCHS_S3": EPOCHS_S3,
        "EPOCHS_S4": EPOCHS_S4, "EPOCHS_S5": EPOCHS_S5,
        "LR_S1": LR_S1, "LR_S2": LR_S2, "LR_S3": LR_S3,
        "LR_S4": LR_S4, "LR_S5": LR_S5,
        "CLIPNORM": CLIPNORM, "USE_EMA": USE_EMA, "EMA_MOMENTUM": EMA_MOMENTUM,
        "WEIGHT_DECAY": WEIGHT_DECAY
    },
    "best_hyperparameters": dict(best_hp.values),
    "cv_metrics": {
        "r2_mean": float(np.mean(r2_scores)), "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)), "rmse_std": float(np.std(rmse_scores)),
        "mae_mean": float(np.mean(mae_scores)), "mae_std": float(np.std(mae_scores)),
        "pred_R2_Q2": float(pred_R2_train)
    },
    "final_metrics_original_scale": {
        "train": {k: float(v) if not isinstance(v, (int, float)) else v for k, v in metrics_train.items()},
        "val":   {k: float(v) if not isinstance(v, (int, float)) else v for k, v in metrics_val.items()},
        "test":  {k: float(v) if not isinstance(v, (int, float)) else v for k, v in metrics_test.items()}
    },
    "baseline_metrics": {
        "baseline_a_tuned_no_transfer": metrics_baseline_a,
        "baseline_b_simple_mlp": metrics_baseline_b
    },
    "artifacts": {
        "ensemble_models": [f"ensemble_model_{i:02d}.keras" for i in range(1, N_ENSEMBLE + 1)],
        "scaler_x": SCALER_X_PATH,
        "scaler_y": SCALER_Y_PATH
    }
}
with open(META_PATH, "w") as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"Metadata saved: {META_PATH}")

# --- Plot: parity on test ---
plt.figure(figsize=(PLOT_WIDTH_IN, PLOT_WIDTH_IN))
sns.scatterplot(x=y_test_inv, y=y_test_pred, s=30)
mn = float(min(np.min(y_test_inv), np.min(y_test_pred)))
mx = float(max(np.max(y_test_inv), np.max(y_test_pred)))
plt.plot([mn, mx], [mn, mx], 'r--')
plt.xlabel("Observed")
plt.ylabel("Predicted (ensemble)")
plt.title("Test parity plot")
plt.tight_layout()
plt.savefig(os.path.join(EXPORT_DIR, "test_parity.png"), dpi=600)
plt.show()

# --- Diagnostic plotting functions ---
def plot_predicted_vs_actual(y_true, y_pred, title, save_path=None):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    plt.figure(figsize=(PLOT_WIDTH_IN, PLOT_WIDTH_IN))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn, mx = np.min([np.min(y_true), np.min(y_pred)]), np.max([np.max(y_true), np.max(y_pred)])
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
        plt.savefig(save_path, dpi=600)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

def plot_residuals_std_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = (np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1))
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    plt.figure(figsize=(PLOT_WIDTH_IN * 2, PLOT_WIDTH_IN * 0.7))
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
        plt.savefig(save_prefix + "_std_residuals_hist_ts.png", dpi=600)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

def plot_residuals_raw_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = (np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1))
    plt.figure(figsize=(PLOT_WIDTH_IN * 2, PLOT_WIDTH_IN * 0.7))
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
        plt.savefig(save_prefix + "_raw_residuals_hist_ts.png", dpi=600)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

# Generate diagnostic plots
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
    print("Error creating diagnostic plots:", e)
    traceback.print_exc()


# =====================================================================
# PHASE 12: 3D SURFACE PLOTS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 12: 3D SURFACE PLOTS")
print("=" * 70)

try:
    from itertools import combinations
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    GRID_N_3D = 30
    RANGE_MODE_3D = "quantile"
    Q_LOW_3D, Q_HIGH_3D = 0.02, 0.98
    HOLD_MODE_3D = "median_train"
    DPI_3D = 600

    plots_dir_3d = os.path.join(EXPORT_DIR, "plots_3d")
    os.makedirs(plots_dir_3d, exist_ok=True)

    p3d = X_train_orig.shape[1]

    if HOLD_MODE_3D == "median_train":
        Xref_3d = np.median(X_train_orig, axis=0)
    elif HOLD_MODE_3D == "mean_train":
        Xref_3d = np.mean(X_train_orig, axis=0)
    else:
        Xref_3d = X_train_orig[0].copy()

    def _safe_name_3d(s):
        s = str(s)
        for ch in [" ", "/", "\\", ":", ";", "|", "(", ")", "[", "]", "{", "}", "%"]:
            s = s.replace(ch, "_")
        return s

    def _grid_vals_3d(col_idx):
        v = X_train_orig[:, col_idx]
        if RANGE_MODE_3D == "minmax":
            lo, hi = float(np.min(v)), float(np.max(v))
        else:
            lo, hi = float(np.quantile(v, Q_LOW_3D)), float(np.quantile(v, Q_HIGH_3D))
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        return np.linspace(lo, hi, GRID_N_3D)

    def _predict_from_origX_3d(X_orig_2d):
        X_scaled = scaler_X.transform(X_orig_2d)
        # Use first ensemble member for surface plots (fast)
        y_scaled = ensemble_models[0].predict(X_scaled, verbose=0)
        return scaler_y.inverse_transform(y_scaled).reshape(-1)

    pairs_3d = list(combinations(range(p3d), 2))
    print(f"Generating {len(pairs_3d)} 3D surfaces into: {plots_dir_3d}")

    for (i, j) in pairs_3d:
        xi = _grid_vals_3d(i)
        xj = _grid_vals_3d(j)
        XI, XJ = np.meshgrid(xi, xj)

        Xgrid = np.tile(Xref_3d.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)

        Z = _predict_from_origX_3d(Xgrid).reshape(XI.shape)

        fig = plt.figure(figsize=(PLOT_WIDTH_IN * 1.5, PLOT_WIDTH_IN * 1.2))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        ax.set_title(f"Predicted surface: {inputs_columns[i]} vs {inputs_columns[j]}\n(others held constant: {HOLD_MODE_3D})")
        ax.set_xlabel(str(inputs_columns[i]))
        ax.set_ylabel(str(inputs_columns[j]))
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()

        out_png = os.path.join(
            plots_dir_3d,
            f"surface_{_safe_name_3d(inputs_columns[i])}_vs_{_safe_name_3d(inputs_columns[j])}_hold_{HOLD_MODE_3D}.png"
        )
        plt.savefig(out_png, dpi=DPI_3D)
        plt.close(fig)

    print(f"3D surfaces saved. Count: {len([f for f in os.listdir(plots_dir_3d) if f.lower().endswith('.png')])}")
except Exception as e:
    print("Error generating 3D surfaces:", e)
    traceback.print_exc()


# =====================================================================
# PHASE 13: EXPORTS + DOWNLOADS
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 13: EXPORTS + DOWNLOADS")
print("=" * 70)

# --- Export DataFrames ---
def make_export_df(labels_df_part, inputs_df_part, y_true, y_pred):
    actual = np.array(y_true).reshape(-1)
    pred = np.array(y_pred).reshape(-1)
    residual = actual - pred
    abs_err = np.abs(residual)
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_pct = np.where(np.abs(actual) > 1e-12, 100.0 * abs_err / np.abs(actual), np.nan)
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

# Save standard exports
results_train.to_excel(os.path.join(EXPORT_DIR, "train_predictions.xlsx"), index=False, engine='openpyxl')
results_val.to_excel(os.path.join(EXPORT_DIR, "val_predictions.xlsx"), index=False, engine='openpyxl')
results_test.to_excel(os.path.join(EXPORT_DIR, "test_predictions.xlsx"), index=False, engine='openpyxl')
results_train.to_csv(os.path.join(EXPORT_DIR, "train_predictions.csv"), index=False)
results_val.to_csv(os.path.join(EXPORT_DIR, "val_predictions.csv"), index=False)
results_test.to_csv(os.path.join(EXPORT_DIR, "test_predictions.csv"), index=False)
print(f"Results exported to {EXPORT_DIR}/")

# --- Full-row exports ---
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
        abs_pct_train = np.where(np.abs(y_train_inv_flat) > 1e-12, 100.0 * abs_err_train / np.abs(y_train_inv_flat), np.nan)
        abs_pct_val   = np.where(np.abs(y_val_inv_flat)   > 1e-12, 100.0 * abs_err_val   / np.abs(y_val_inv_flat),   np.nan)
        abs_pct_test  = np.where(np.abs(y_test_inv_flat)  > 1e-12, 100.0 * abs_err_test  / np.abs(y_test_inv_flat),  np.nan)

    train_full_rows = train_full_rows.copy()
    val_full_rows   = val_full_rows.copy()
    test_full_rows  = test_full_rows.copy()

    for full_df, y_inv, y_pred_f, res, abs_e, abs_p in [
        (train_full_rows, y_train_inv_flat, y_train_pred_flat, residuals_train, abs_err_train, abs_pct_train),
        (val_full_rows, y_val_inv_flat, y_val_pred_flat, residuals_val, abs_err_val, abs_pct_val),
        (test_full_rows, y_test_inv_flat, y_test_pred_flat, residuals_test, abs_err_test, abs_pct_test),
    ]:
        full_df["Actual_Value"]      = y_inv
        full_df["Predicted_Value"]   = y_pred_f
        full_df["Residual"]          = res
        full_df["Abs_Error"]         = abs_e
        full_df["Abs_Percent_Error"] = abs_p

    def move_pred_cols_to_end(df_full):
        pred_cols = ["Actual_Value", "Predicted_Value", "Residual", "Abs_Error", "Abs_Percent_Error"]
        cols = [c for c in df_full.columns if c not in pred_cols] + pred_cols
        return df_full[cols]

    train_full_rows = move_pred_cols_to_end(train_full_rows)
    val_full_rows   = move_pred_cols_to_end(val_full_rows)
    test_full_rows  = move_pred_cols_to_end(test_full_rows)

    train_full_rows.to_excel(os.path.join(EXPORT_DIR, "train_full_with_all_columns.xlsx"), index=False, engine='openpyxl')
    val_full_rows.to_excel(os.path.join(EXPORT_DIR, "val_full_with_all_columns.xlsx"), index=False, engine='openpyxl')
    test_full_rows.to_excel(os.path.join(EXPORT_DIR, "test_full_with_all_columns.xlsx"), index=False, engine='openpyxl')
    train_full_rows.to_csv(os.path.join(EXPORT_DIR, "train_full_with_all_columns.csv"), index=False)
    val_full_rows.to_csv(os.path.join(EXPORT_DIR, "val_full_with_all_columns.csv"), index=False)
    test_full_rows.to_csv(os.path.join(EXPORT_DIR, "test_full_with_all_columns.csv"), index=False)
    print("Full-row exports saved")
except Exception as e:
    print("Error creating full-row exports:", e)
    traceback.print_exc()

_show(results_train.head())
_show(results_val.head())
_show(results_test.head())

# --- ZIP download #1: training results ---
print("\nDOWNLOAD #1: TRAINING RESULTS")
print("-" * 80)

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
    META_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    os.path.join(EXPORT_DIR, "test_parity.png"),
]

# Add ensemble model files
for i in range(1, N_ENSEMBLE + 1):
    training_files.append(os.path.join(EXPORT_DIR, f"ensemble_model_{i:02d}.keras"))

# Add plot files
for subdir in ["plots", "plots_3d"]:
    d = os.path.join(EXPORT_DIR, subdir)
    if os.path.isdir(d):
        for fname in os.listdir(d):
            training_files.append(os.path.join(d, fname))

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


# =====================================================================
# PHASE 14: NEW DATA PREDICTION
# =====================================================================
print("\n" + "=" * 70)
print("PHASE 14: NEW DATA PREDICTION")
print("=" * 70)

# --- Build calibration residuals for empirical PI ---
def get_calibration_residuals(source="val"):
    source = str(source).strip().lower()
    if source == "val":
        return (y_val_inv.reshape(-1) - y_val_pred.reshape(-1))
    elif source == "oof":
        return (y_oof_true_inv.reshape(-1) - y_oof_pred_inv.reshape(-1))
    else:
        raise ValueError("PI_CALIBRATION must be 'val' or 'oof'")

cal_residuals = get_calibration_residuals(PI_CALIBRATION)
q_low = np.quantile(cal_residuals, PI_ALPHA / 2.0)
q_high = np.quantile(cal_residuals, 1.0 - PI_ALPHA / 2.0)
print(f"Empirical PI calibration ({PI_CALIBRATION}): q_low={q_low:.4f}, q_high={q_high:.4f} (alpha={PI_ALPHA})")

print("\nUpload new data for predictions (optional).")
new_data_loaded = False
new_df = None

if use_colab:
    try:
        uploaded_new = colab_files.upload()
        if len(uploaded_new) > 0:
            new_file_name = list(uploaded_new.keys())[0]
            new_df = load_table(new_file_name, sep=SEP)
            new_data_loaded = True
            print(f"New data loaded. Shape: {new_df.shape}")
        else:
            print("No new data uploaded. Skipping predictions.")
    except Exception as e:
        print(f"Upload failed: {e}")
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
            print(f"New data loaded from {found}. Shape: {new_df.shape}")
        except Exception as e:
            print(f"Load failed: {e}")
    else:
        print("No new_data.* found. Skipping predictions.")

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
                raise ValueError("Column mismatch between training and new data inputs.")

        # Imputation for NaNs
        nan_in_features = new_inputs_df.isna().sum().sum()
        if nan_in_features > 0:
            print(f"NaNs in new features: {nan_in_features}. Imputing with TRAIN means...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            imputer.fit(X_train_orig)
            new_inputs_imputed = imputer.transform(new_inputs_df)
            new_inputs_df = pd.DataFrame(new_inputs_imputed, columns=new_inputs_df.columns)

        new_X = new_inputs_df.values
        new_X_scaled = scaler_X.transform(new_X)

        print(f"Generating ensemble predictions for {len(new_X)} samples...")
        new_y_pred, new_y_members, new_y_spread = ensemble_predict(
            ensemble_models, new_X_scaled, scaler_y, ENSEMBLE_AGG
        )

        # Empirical PI
        pi_lower = new_y_pred + q_low
        pi_upper = new_y_pred + q_high

        new_results_compact = pd.DataFrame(index=range(len(new_X)))
        if new_labels_df is not None and new_labels_df.shape[0] == len(new_X):
            new_results_compact = pd.concat([new_results_compact, new_labels_df.reset_index(drop=True)], axis=1)
        new_results_compact = pd.concat([new_results_compact, new_inputs_df.reset_index(drop=True)], axis=1)

        new_results_compact["Predicted_Value"] = new_y_pred
        new_results_compact["PI_Lower_95%"] = pi_lower
        new_results_compact["PI_Upper_95%"] = pi_upper
        new_results_compact["PI_Width"] = pi_upper - pi_lower
        new_results_compact["Ensemble_Spread_Std"] = new_y_spread
        new_results_compact["PI_Calibration_Source"] = PI_CALIBRATION

        # Check if actual values exist
        new_has_actual = False
        if n_cols_new > N_LABELS + N_INPUTS:
            try:
                new_y_actual = new_df.iloc[:, N_LABELS + N_INPUTS].values.reshape(-1, 1)
                if not np.isnan(new_y_actual).all():
                    if np.isnan(new_y_actual).any():
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
            print(f"\nEmpirical 95% PI Coverage: {within_pi.sum()}/{len(within_pi)} ({pi_coverage:.1f}%)")
        else:
            print("\nNo actual values. Predictions + PI only.")

        # Full-row export
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
        _show(new_results_compact.head(10))

    except Exception as e:
        print(f"Error processing new data: {e}")
        traceback.print_exc()

# --- ZIP download #2: new data predictions ---
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
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
else:
    print("\nNo new data predictions available.")

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
