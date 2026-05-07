# ============================================================
# MERGED CELL: Hill Pre-training + Tuning + Weight Transfer +
#              CV + Final Training + Metrics + Export
#
# FIXES APPLIED:
#   #1/#8 — build_model_fixed → build_model (all occurrences)
#   #2    — shape-aware Dense-only weight transfer (Cell 1 approach)
#           IMPROVED: partial/slice-based transfer when shapes differ
#   #3    — CV loop uses CV_EPOCHS, not TUNER_EPOCHS
#   #4    — set_weights in CV folds guarded with try/except
#           IMPROVED: per-layer transfer with partial matching
#   #5    — overwrite=True added to tuner
#   #6    — PI_CALIBRATION / PI_ALPHA now fully implemented
#   #7    — scaler reassignment in retrain block is documented
#   #9    — NEW: smart weight transfer function with partial matching
#   #10   — NEW: best fold weights fed into final training
#   #11   — NEW: optimized Hill pre-training (curriculum, cosine LR,
#                 wider warmup, multi-feature Hill surface)
#   #12   — NEW: scale-aware weight transfer preserving activation
#                 variance + warm-start fine-tuning after transfer
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
import traceback
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
        import subprocess, sys
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

# =====================================================================
# USER CONFIGURATION
# =====================================================================

# --- Hill pre-training ---
N_INPUTS        = 10
N_SYNTHETIC     = 2000         # FIX #11: more synthetic data for richer pre-training
HILL_V_MAX      = 200
HILL_K          = 0.10
HILL_N          = 1.20
HILL_X_MIN      = 0.1
HILL_X_MAX      = 100
PRETRAIN_EPOCHS = 300          # FIX #11: longer budget (cosine LR + curriculum)
HILL_CURRICULUM  = True         # FIX #11: staged noise curriculum
HILL_MULTI_FEAT  = True         # FIX #11: multi-feature Hill interactions

# --- Data split ---
TRAIN_PERCENT = 60
VAL_PERCENT   = 20
TEST_PERCENT  = 20

N_LABELS   = 7
TARGET_COL = None   # absolute column index of target, or None
SEP        = "\t"

# --- Training ---
DISABLE_GPU         = True
TUNER_TRIALS        = 15
K_FOLDS             = 15
RANDOM_SEED         = 42
TUNER_EPOCHS        = 200
CV_EPOCHS           = 200   # FIX #3: dedicated budget for CV folds
FINAL_EPOCHS        = 200
DO_OPTIONAL_RETRAIN = True

# --- Weight Transfer ---
TRANSFER_MODE       = "smart"  # "smart" (partial/slice), "strict" (exact match only)
TRANSFER_BEST_FOLD  = True     # seed final model from best CV fold weights
TRANSFER_SCALE_AWARE = True    # FIX #12: scale weights by fan ratio on partial transfer
WARMSTART_EPOCHS     = 10      # FIX #12: brief low-LR fine-tune after weight transfer
WARMSTART_LR         = 1e-4    # FIX #12: learning rate for warm-start phase

# --- Prediction intervals ---
PI_CALIBRATION = "val"   # "val" → calibrate on val residuals | "oof" → OOF residuals
PI_ALPHA       = 0.05    # 95% PI (2.5% / 97.5% empirical quantiles)

# --- Output ---
EXPORT_DIR = "optimized_model"

# =====================================================================

total_percent = TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT
if abs(total_percent - 100) > 0.01:
    raise ValueError(
        f"Split percentages must sum to 100. "
        f"Got: {TRAIN_PERCENT}+{VAL_PERCENT}+{TEST_PERCENT}={total_percent}"
    )

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if DISABLE_GPU:
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled. Using CPU.")
    except Exception as e:
        print("Could not change GPU visibility:", e)


# =====================================================================
# SMART WEIGHT TRANSFER UTILITIES (FIX #9)
# =====================================================================

def _scale_kernel_slice(ws, wt, slice_kernel, scale_aware=True):
    """
    FIX #12: When transferring a partial kernel slice, scale the values
    to preserve activation variance (He-style fan-in correction).

    If source kernel is (in_s, out_s) and target is (in_t, out_t),
    the transferred slice covers min(in_s,in_t) inputs. But the target
    neuron expects in_t inputs. Scale by sqrt(min_in / in_t) so the
    variance of the pre-activation stays roughly the same.
    """
    if not scale_aware or ws.ndim != 2 or wt.ndim != 2:
        return slice_kernel
    in_src, _ = ws.shape
    in_tgt, _ = wt.shape
    min_in = min(in_src, in_tgt)
    if min_in == in_tgt:
        return slice_kernel
    scale = np.sqrt(float(min_in) / float(in_tgt))
    return slice_kernel * scale


def transfer_weights_smart(source_model, target_model, mode="smart",
                           scale_aware=None, verbose=True):
    """
    Transfer weights from source_model to target_model layer-by-layer.

    Modes:
      - "strict": only transfer when shapes match exactly (original behavior).
      - "smart":  transfer overlapping slices when shapes partially match.
                  For a Dense layer with kernel (in, out):
                    - transfer min(in_src, in_tgt) x min(out_src, out_tgt) slice
                    - transfer min(out_src, out_tgt) bias entries
                  FIX #12: optionally scales partial kernels by fan-in ratio
                  to preserve activation variance (He initialization).

    Returns:
      dict with keys:
        "full"    — count of layers with exact shape match (full transfer)
        "partial" — count of layers with partial/slice transfer
        "skipped" — count of layers skipped entirely
        "details" — list of per-layer info strings
    """
    if scale_aware is None:
        scale_aware = TRANSFER_SCALE_AWARE

    dense_src = [l for l in source_model.layers if isinstance(l, layers.Dense)]
    dense_tgt = [l for l in target_model.layers if isinstance(l, layers.Dense)]

    stats = {"full": 0, "partial": 0, "skipped": 0, "details": []}
    n_pairs = min(len(dense_src), len(dense_tgt))

    for i, (ls, lt) in enumerate(zip(dense_src, dense_tgt)):
        ws_list = ls.get_weights()  # [kernel, bias] or [kernel]
        wt_list = lt.get_weights()

        if len(ws_list) != len(wt_list):
            info = f"  Layer {i} ({ls.name} -> {lt.name}): skipped (different # weight arrays)"
            stats["skipped"] += 1
            stats["details"].append(info)
            if verbose:
                print(info)
            continue

        all_match = all(ws.shape == wt.shape for ws, wt in zip(ws_list, wt_list))

        if all_match:
            lt.set_weights(ws_list)
            info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                    f"FULL transfer {[w.shape for w in ws_list]}")
            stats["full"] += 1
            stats["details"].append(info)
            if verbose:
                print(info)
        elif mode == "smart":
            new_weights = []
            transferred_something = False

            for ws, wt in zip(ws_list, wt_list):
                wt_current = np.array(wt)

                if ws.ndim == 2 and wt.ndim == 2:
                    min_in = min(ws.shape[0], wt.shape[0])
                    min_out = min(ws.shape[1], wt.shape[1])
                    if min_in > 0 and min_out > 0:
                        wt_new = wt_current.copy()
                        raw_slice = ws[:min_in, :min_out]
                        scaled_slice = _scale_kernel_slice(
                            ws, wt, raw_slice, scale_aware=scale_aware
                        )
                        wt_new[:min_in, :min_out] = scaled_slice
                        new_weights.append(wt_new)
                        transferred_something = True
                    else:
                        new_weights.append(wt_current)
                elif ws.ndim == 1 and wt.ndim == 1:
                    min_dim = min(ws.shape[0], wt.shape[0])
                    if min_dim > 0:
                        wt_new = wt_current.copy()
                        wt_new[:min_dim] = ws[:min_dim]
                        new_weights.append(wt_new)
                        transferred_something = True
                    else:
                        new_weights.append(wt_current)
                else:
                    new_weights.append(wt_current)

            if transferred_something:
                lt.set_weights(new_weights)
                src_shapes = [w.shape for w in ws_list]
                tgt_shapes = [w.shape for w in wt_list]
                scale_tag = "+scaled" if scale_aware else ""
                info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                        f"PARTIAL{scale_tag} transfer (src={src_shapes}, tgt={tgt_shapes})")
                stats["partial"] += 1
            else:
                info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                        f"skipped (no overlap)")
                stats["skipped"] += 1
            stats["details"].append(info)
            if verbose:
                print(info)
        else:
            info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                    f"skipped (shape mismatch, strict mode)")
            stats["skipped"] += 1
            stats["details"].append(info)
            if verbose:
                print(info)

    if len(dense_src) > n_pairs:
        info = f"  {len(dense_src) - n_pairs} extra source layer(s) not transferred"
        stats["details"].append(info)
        if verbose:
            print(info)
    if len(dense_tgt) > n_pairs:
        info = f"  {len(dense_tgt) - n_pairs} extra target layer(s) kept random init"
        stats["details"].append(info)
        if verbose:
            print(info)

    return stats


def transfer_weights_to_fold(source_model, fold_model, mode="smart",
                             scale_aware=None, verbose=False):
    """
    Convenience wrapper for transferring weights into a CV fold model.
    Returns True if at least one layer was transferred (full or partial).
    """
    stats = transfer_weights_smart(source_model, fold_model, mode=mode,
                                   scale_aware=scale_aware, verbose=verbose)
    return (stats["full"] + stats["partial"]) > 0


def warmstart_finetune(model_to_tune, X_tr, y_tr, X_va, y_va,
                       epochs=None, lr=None):
    """
    FIX #12: Brief low-LR fine-tuning phase after weight transfer.
    Gently adapts transferred weights to the new data distribution
    before full training begins.
    """
    if epochs is None:
        epochs = WARMSTART_EPOCHS
    if lr is None:
        lr = WARMSTART_LR
    if epochs <= 0:
        return None

    model_to_tune.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse', metrics=['mae']
    )
    es_ws = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    hist = model_to_tune.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=32,
        callbacks=[es_ws],
        verbose=0,
    )
    ws_best = min(hist.history['val_loss'])
    print(f"    Warm-start fine-tune: {len(hist.history['loss'])} epochs, "
          f"best val_loss={ws_best:.6f}")
    return hist


# ============================================================
# PHASE 1: HILL PRE-TRAINING
# ============================================================
print("\n" + "="*70)
print("PHASE 1: HILL PRE-TRAINING")
print("="*70)

def _generate_hill_data(n_samples, n_features, v_max, k, n_hill, x_min, x_max,
                        noise_frac=0.05, multi_feature=False):
    """
    FIX #11: Enhanced Hill data generation.
    - noise_frac controls the noise level (for curriculum learning)
    - multi_feature=True adds pairwise interaction terms so the warmup
      model learns richer feature representations (not just x1-dominated).
    """
    X  = np.random.uniform(x_min, x_max, (n_samples, n_features)).astype(np.float32)
    x1 = X[:, 0]
    y  = v_max * (x1 ** n_hill) / (k ** n_hill + x1 ** n_hill)

    if multi_feature and n_features >= 2:
        for i in range(1, min(n_features, 5)):
            xi_norm = (X[:, i] - x_min) / (x_max - x_min)
            y += 0.08 * v_max * xi_norm
        for i in range(min(n_features - 1, 3)):
            xi = (X[:, i] - x_min) / (x_max - x_min)
            xj = (X[:, i + 1] - x_min) / (x_max - x_min)
            y += 0.03 * v_max * xi * xj
    else:
        for i in range(1, min(n_features, 3)):
            y += 0.05 * v_max * (X[:, i] - x_min) / (x_max - x_min)

    y += np.random.normal(0, noise_frac * v_max, n_samples)
    return X, y.reshape(-1, 1)


class CosineAnnealingSchedule(callbacks.Callback):
    """FIX #11: Cosine annealing LR schedule with warm restarts."""
    def __init__(self, lr_max=1e-3, lr_min=1e-5, T_0=50, T_mult=2):
        super().__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self._cycle_epoch = 0
        self._current_T = T_0

    def on_epoch_begin(self, epoch, logs=None):
        if self._cycle_epoch >= self._current_T:
            self._cycle_epoch = 0
            self._current_T = int(self._current_T * self.T_mult)
        frac = self._cycle_epoch / max(self._current_T, 1)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * frac))
        self.model.optimizer.learning_rate.assign(lr)
        self._cycle_epoch += 1


print("[1/5] Generating synthetic Hill data...")
X_h, y_h = _generate_hill_data(
    N_SYNTHETIC, N_INPUTS, HILL_V_MAX, HILL_K, HILL_N, HILL_X_MIN, HILL_X_MAX,
    noise_frac=0.05 if not HILL_CURRICULUM else 0.01,
    multi_feature=HILL_MULTI_FEAT,
)
print(f"  X {X_h.shape} | y {y_h.shape} | y range [{y_h.min():.2f}, {y_h.max():.2f}]")

print("\n[2/5] Scaling Hill data (fit on full synthetic set)...")
scaler_Xh = StandardScaler().fit(X_h)
scaler_yh = StandardScaler().fit(y_h)
X_h_sc    = scaler_Xh.transform(X_h)
y_h_sc    = scaler_yh.transform(y_h)

X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
    X_h_sc, y_h_sc, test_size=0.2, random_state=RANDOM_SEED
)
print(f"  Hill train {X_h_train.shape} | Hill val {X_h_val.shape}")

# FIX #11: Wider warmup model (256->128->64->32->1) to maximize weight overlap
print("\n[3/5] Building warmup model (256 -> 128 -> 64 -> 32 -> 1)...")
tf.keras.backend.clear_session()
warmup_model = keras.Sequential([
    layers.Input(shape=(N_INPUTS,)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.Dropout(0.2),
    layers.Dense(64,  activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.Dropout(0.2),
    layers.Dense(32,  activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
    layers.Dropout(0.1),
    layers.Dense(1,   activation='linear'),
])
warmup_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
print(f"  {warmup_model.count_params():,} parameters")

# FIX #11: Curriculum learning — train in stages with increasing noise
if HILL_CURRICULUM:
    print(f"\n[4/5] Curriculum pre-training ({PRETRAIN_EPOCHS} total epochs)...")
    noise_stages = [0.01, 0.03, 0.05, 0.10]
    stage_epochs = PRETRAIN_EPOCHS // len(noise_stages)

    cosine_cb = CosineAnnealingSchedule(lr_max=1e-3, lr_min=1e-5, T_0=stage_epochs // 2)
    es_warmup = callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    for stage_i, nf in enumerate(noise_stages, start=1):
        print(f"\n  Stage {stage_i}/{len(noise_stages)}: noise_frac={nf}")
        X_stage, y_stage = _generate_hill_data(
            N_SYNTHETIC, N_INPUTS, HILL_V_MAX, HILL_K, HILL_N,
            HILL_X_MIN, HILL_X_MAX, noise_frac=nf, multi_feature=HILL_MULTI_FEAT,
        )
        X_stage_sc = scaler_Xh.transform(X_stage)
        y_stage_sc = scaler_yh.transform(y_stage)

        X_st_tr, X_st_va, y_st_tr, y_st_va = train_test_split(
            X_stage_sc, y_stage_sc, test_size=0.2, random_state=RANDOM_SEED + stage_i
        )

        cosine_cb._cycle_epoch = 0
        cosine_cb._current_T = cosine_cb.T_0

        hist_stage = warmup_model.fit(
            X_st_tr, y_st_tr,
            validation_data=(X_st_va, y_st_va),
            epochs=stage_epochs,
            batch_size=32,
            callbacks=[es_warmup, cosine_cb],
            verbose=0,
        )
        print(f"    -> {len(hist_stage.history['loss'])} epochs, "
              f"train_loss={hist_stage.history['loss'][-1]:.6f}, "
              f"val_loss={hist_stage.history['val_loss'][-1]:.6f}")

    print("\n  Curriculum pre-training complete")
else:
    print(f"\n[4/5] Pre-training ({PRETRAIN_EPOCHS} epochs, cosine LR + early-stop)...")
    cosine_cb = CosineAnnealingSchedule(lr_max=1e-3, lr_min=1e-5, T_0=50)
    es_warmup = callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    hist_warmup = warmup_model.fit(
        X_h_train, y_h_train,
        validation_data=(X_h_val, y_h_val),
        epochs=PRETRAIN_EPOCHS,
        batch_size=32,
        callbacks=[es_warmup, cosine_cb],
        verbose=1,
    )
    print(f"\n  Pre-training done | "
          f"train loss {hist_warmup.history['loss'][-1]:.6f} | "
          f"val loss   {hist_warmup.history['val_loss'][-1]:.6f}")

# [5/5] Validate warmup quality
print("\n[5/5] Warmup model validation...")
y_h_val_pred = scaler_yh.inverse_transform(warmup_model.predict(X_h_val, verbose=0))
y_h_val_true = scaler_yh.inverse_transform(y_h_val)
warmup_r2 = r2_score(y_h_val_true.reshape(-1), y_h_val_pred.reshape(-1))
warmup_rmse = np.sqrt(mean_squared_error(y_h_val_true.reshape(-1), y_h_val_pred.reshape(-1)))
print(f"  Warmup model on Hill val: R2={warmup_r2:.4f}, RMSE={warmup_rmse:.4f}")

print("\n" + "="*70)
print("PHASE 1 DONE - warmup_model ready")
print("="*70)

# ============================================================
# PHASE 2: DATA LOADING & SPLITTING
# ============================================================
print("\n" + "="*70)
print("PHASE 2: DATA LOADING & SPLITTING")
print("="*70)

def load_table(path, sep=SEP):
    _, ext = os.path.splitext(path.lower())
    if ext in [".xlsx", ".xls", ".xlsm"]:
        print(f"Detected Excel file: {path}")
        return pd.read_excel(path, engine="openpyxl")
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err  = None
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

print("Upload your dataset...")
if use_colab:
    uploaded = colab_files.upload()
    if not uploaded:
        raise RuntimeError("No file uploaded.")
    file_name = list(uploaded.keys())[0]
else:
    file_name = "data.tsv"
    if not os.path.exists(file_name):
        raise FileNotFoundError("Not in Colab and 'data.tsv' not found.")

df = load_table(file_name, sep=SEP)
print(f"Data loaded. Shape: {df.shape}")
_show(df.head())

n_cols = df.shape[1]
if TARGET_COL is not None:
    if not (0 <= TARGET_COL < n_cols):
        raise IndexError("TARGET_COL out of range.")
    labels_df  = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
    input_cols = [c for c in range(N_LABELS, n_cols) if c != TARGET_COL]
    inputs_df  = df.iloc[:, input_cols]
    y_full     = df.iloc[:, TARGET_COL].values.reshape(-1, 1)
else:
    if N_LABELS + N_INPUTS >= n_cols:
        print("N_LABELS + N_INPUTS >= total columns -> using last column as target.")
        labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, N_LABELS:-1]
        y_full    = df.iloc[:, -1].values.reshape(-1, 1)
    else:
        labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
        inputs_df = df.iloc[:, N_LABELS:N_LABELS + N_INPUTS]
        y_full    = df.iloc[:, N_LABELS + N_INPUTS].values.reshape(-1, 1)

print(f"Labels {labels_df.shape} | Inputs {inputs_df.shape} | y {y_full.shape}")

X_full      = inputs_df.values
labels_full = labels_df
row_pos     = np.arange(len(df))

test_frac       = TEST_PERCENT  / 100.0
val_frac        = VAL_PERCENT   / 100.0
train_frac      = TRAIN_PERCENT / 100.0
val_split_ratio = val_frac / (train_frac + val_frac)

(X_temp, X_test_orig,
 y_temp, y_test_orig,
 labels_temp, labels_test,
 idx_temp, idx_test) = train_test_split(
    X_full, y_full, labels_full, row_pos,
    test_size=test_frac, random_state=RANDOM_SEED,
)
(X_train_orig, X_val_orig,
 y_train_orig, y_val_orig,
 labels_train, labels_val,
 idx_train, idx_val) = train_test_split(
    X_temp, y_temp, labels_temp, idx_temp,
    test_size=val_split_ratio, random_state=RANDOM_SEED,
)

inputs_columns = list(inputs_df.columns)
X_train_df = pd.DataFrame(X_train_orig, columns=inputs_columns).reset_index(drop=True)
X_val_df   = pd.DataFrame(X_val_orig,   columns=inputs_columns).reset_index(drop=True)
X_test_df  = pd.DataFrame(X_test_orig,  columns=inputs_columns).reset_index(drop=True)

print(f"\nSplit (rows): train={len(X_train_orig)} | val={len(X_val_orig)} | test={len(X_test_orig)}")
print(f"Split (%):    train={100*len(X_train_orig)/len(df):.1f}% | "
      f"val={100*len(X_val_orig)/len(df):.1f}% | "
      f"test={100*len(X_test_orig)/len(df):.1f}%")

# Scalers fit on TRAIN only — no leakage
scaler_X = StandardScaler().fit(X_train_orig)
scaler_y = StandardScaler().fit(y_train_orig)

X_train = scaler_X.transform(X_train_orig)
X_val   = scaler_X.transform(X_val_orig)
X_test  = scaler_X.transform(X_test_orig)
y_train = scaler_y.transform(y_train_orig)
y_val   = scaler_y.transform(y_val_orig)
y_test  = scaler_y.transform(y_test_orig)

# ============================================================
# PHASE 3: HYPERPARAMETER TUNING
# ============================================================
print("\n" + "="*70)
print("PHASE 3: HYPERPARAMETER TUNING")
print("="*70)

def build_model(hp):
    """
    Variable-depth MLP; units, dropout, L2, and lr are all tunable.
    The warmup architecture (128->64->32->1) is a reachable point in this space.
    """
    model    = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    n_layers = hp.Int('num_layers', 2, 6, step=1)
    l2_val   = hp.Choice('l2_reg', [1e-4, 1e-3, 1e-2])

    for i in range(n_layers):
        model.add(layers.Dense(
            hp.Int(f'units_{i}', 64, 512, step=64),
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_val),
        ))
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('lr', [1e-4, 5e-4, 1e-3, 5e-3])
        ),
        loss='mse',
        metrics=['mae'],
    )
    return model

# FIX #5: overwrite=True so every run starts a clean search
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=TUNER_TRIALS,
    executions_per_trial=1,
    overwrite=True,
    directory='tuner_results',
    project_name=f'ann_{TRAIN_PERCENT}_{VAL_PERCENT}_{TEST_PERCENT}',
)

print(f"Starting search ({TUNER_TRIALS} trials)...")
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=TUNER_EPOCHS,
    batch_size=32,
    verbose=1,
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\nBest hyperparameters:")
for k, v in best_hp.values.items():
    print(f"  {k}: {v}")

print("\n" + "="*70)
print("PHASE 3 DONE")
print("="*70)

# ============================================================
# PHASE 4: WEIGHT TRANSFER FROM WARMUP MODEL (IMPROVED)
# ============================================================
print("\n" + "="*70)
print("PHASE 4: WEIGHT TRANSFER FROM WARMUP MODEL")
print("="*70)

_model_with_pretrain = None
model_for_transfer   = None

try:
    print("[1/4] Building model from best HPs...")
    model_for_transfer = tuner.hypermodel.build(best_hp)
    print(f"  {model_for_transfer.count_params():,} parameters")

    print(f"\n[2/4] Smart weight transfer (mode='{TRANSFER_MODE}', "
          f"scale_aware={TRANSFER_SCALE_AWARE})...")
    stats = transfer_weights_smart(
        warmup_model, model_for_transfer,
        mode=TRANSFER_MODE, verbose=True
    )

    print(f"\n[3/4] Transfer summary: "
          f"{stats['full']} full, {stats['partial']} partial, "
          f"{stats['skipped']} skipped")

    if (stats["full"] + stats["partial"]) > 0:
        _model_with_pretrain = model_for_transfer

        # FIX #12: warm-start fine-tuning after weight transfer
        if WARMSTART_EPOCHS > 0:
            print(f"\n[4/4] Warm-start fine-tuning ({WARMSTART_EPOCHS} epochs, "
                  f"lr={WARMSTART_LR})...")
            warmstart_finetune(
                _model_with_pretrain, X_train, y_train, X_val, y_val,
                epochs=WARMSTART_EPOCHS, lr=WARMSTART_LR
            )
        else:
            print("\n[4/4] Warm-start skipped (WARMSTART_EPOCHS=0)")

        print("-> _model_with_pretrain is ready (with warm-start)")
    else:
        print("WARNING: No weights transferred (architecture mismatch)")

except Exception as e:
    print(f"Weight transfer failed: {e}")
    traceback.print_exc()
    _model_with_pretrain = None

print("\n" + "="*70)
if _model_with_pretrain is not None:
    print("PHASE 4 DONE - pre-trained weights loaded + warm-started")
else:
    print("PHASE 4: no weight transfer")
    if model_for_transfer is not None:
        _model_with_pretrain = model_for_transfer
        print("    Falling back to tuned model without transferred weights")
    else:
        print("    model_for_transfer unavailable - re-run from Phase 3")
print("="*70)

# ============================================================
# PHASE 5: K-FOLD CV ON TRAIN SPLIT (STRICT NO-LEAKAGE)
# ============================================================
print("\n" + "="*70)
print("PHASE 5: K-FOLD CV ON TRAIN SPLIT")
print("="*70)

kf             = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
r2_scores      = []
rmse_scores    = []
mae_scores     = []
y_oof_pred_inv = np.full(len(y_train_orig), np.nan)
y_oof_true_inv = np.full(len(y_train_orig), np.nan)

# Track best fold for weight transfer to final model (FIX #10)
best_fold_r2      = -np.inf
best_fold_weights = None

if _model_with_pretrain is not None:
    print(f"  Will seed each fold from pre-trained weights (mode='{TRANSFER_MODE}')")
    use_pretrain = True
    source_model_for_folds = _model_with_pretrain
else:
    print("  No pre-trained model - folds start from random init")
    use_pretrain = False
    source_model_for_folds = None

print(f"\nRunning {K_FOLDS}-fold CV "
      f"(each fold's scaler fit on fold-train only, no leakage)...\n")

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig), start=1):

    X_tr_orig, X_va_orig = X_train_orig[tr_idx], X_train_orig[va_idx]
    y_tr_orig, y_va_orig = y_train_orig[tr_idx], y_train_orig[va_idx]

    fold_scaler_X = StandardScaler().fit(X_tr_orig)
    fold_scaler_y = StandardScaler().fit(y_tr_orig)

    X_tr = fold_scaler_X.transform(X_tr_orig)
    X_va = fold_scaler_X.transform(X_va_orig)
    y_tr = fold_scaler_y.transform(y_tr_orig)
    y_va = fold_scaler_y.transform(y_va_orig)

    model_fold = build_model(best_hp)

    # FIX #4/#9: per-layer smart weight transfer instead of bulk set_weights
    if use_pretrain:
        transferred = transfer_weights_to_fold(
            source_model_for_folds, model_fold,
            mode=TRANSFER_MODE, verbose=False
        )
        if not transferred and fold == 1:
            print(f"  NOTE: Fold {fold} weight seeding skipped (no compatible layers)")
        # FIX #12: warm-start fine-tune per fold after weight transfer
        elif transferred and WARMSTART_EPOCHS > 0:
            warmstart_finetune(model_fold, X_tr, y_tr, X_va, y_va,
                               epochs=WARMSTART_EPOCHS, lr=WARMSTART_LR)

    es_fold = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    model_fold.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=CV_EPOCHS,       # FIX #3: CV_EPOCHS, not TUNER_EPOCHS
        batch_size=32,
        callbacks=[es_fold],
        verbose=0,
    )

    y_va_pred_inv = fold_scaler_y.inverse_transform(
        model_fold.predict(X_va, verbose=0)
    ).reshape(-1)
    y_va_true_inv = y_va_orig.reshape(-1)

    y_oof_pred_inv[va_idx] = y_va_pred_inv
    y_oof_true_inv[va_idx] = y_va_true_inv

    r2   = r2_score(y_va_true_inv, y_va_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_va_true_inv, y_va_pred_inv))
    mae  = np.mean(np.abs(y_va_true_inv - y_va_pred_inv))
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    print(f"  Fold {fold:2d}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # FIX #10: track best fold weights for later transfer to final model
    if r2 > best_fold_r2:
        best_fold_r2 = r2
        best_fold_weights = model_fold.get_weights()

print(f"\nCV summary (TRAIN split, no leakage):")
print(f"  R2   {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
print(f"  RMSE {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")
print(f"  MAE  {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")
print(f"\n  Best fold R2: {best_fold_r2:.4f}")

if np.isnan(y_oof_pred_inv).any():
    raise RuntimeError("OOF predictions contain NaNs - check fold logic.")

ss_res  = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
ss_tot  = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
pred_R2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
print(f"\n  Predicted R2 (Q2/PRESS): {pred_R2:.4f}")

print("\n" + "="*70)
print("PHASE 5 DONE")
print("="*70)

# ============================================================
# PHASE 6: FINAL TRAINING (TRAIN -> VALIDATE ON VAL)
# ============================================================
print("\n" + "="*70)
print("PHASE 6: FINAL TRAINING")
print("="*70)

os.makedirs(EXPORT_DIR, exist_ok=True)
checkpoint_path = os.path.join(EXPORT_DIR, "best_model.keras")

tf.keras.backend.clear_session()
model    = tuner.hypermodel.build(best_hp)

# FIX #10: seed final model from best CV fold weights if available
_final_transferred = False
if TRANSFER_BEST_FOLD and best_fold_weights is not None:
    print("[Weight Transfer] Seeding final model from best CV fold weights...")
    try:
        model.set_weights(best_fold_weights)
        print("  Full weight transfer from best fold successful")
        _final_transferred = True
    except ValueError:
        print("  Direct transfer failed, using smart per-layer transfer...")
        temp_source = tuner.hypermodel.build(best_hp)
        temp_source.set_weights(best_fold_weights)
        stats = transfer_weights_smart(temp_source, model, mode=TRANSFER_MODE, verbose=True)
        _final_transferred = (stats["full"] + stats["partial"]) > 0
        del temp_source
elif use_pretrain and _model_with_pretrain is not None:
    print("[Weight Transfer] Seeding final model from warmup pre-trained weights...")
    stats = transfer_weights_smart(_model_with_pretrain, model, mode=TRANSFER_MODE, verbose=True)
    _final_transferred = (stats["full"] + stats["partial"]) > 0
else:
    print("[Weight Transfer] No weight seeding for final model (random init)")

# FIX #12: warm-start fine-tune the final model after weight transfer
if _final_transferred and WARMSTART_EPOCHS > 0:
    print(f"[Warm-start] Fine-tuning final model ({WARMSTART_EPOCHS} epochs, lr={WARMSTART_LR})...")
    warmstart_finetune(model, X_train, y_train, X_val, y_val,
                       epochs=WARMSTART_EPOCHS, lr=WARMSTART_LR)

mc       = callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1
)
es_final = callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=FINAL_EPOCHS,
    batch_size=32,
    callbacks=[es_final, mc],
    verbose=1,
)

if os.path.exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path, compile=False)

best_epoch = int(np.argmin(history.history['val_loss']) + 1)
print(f"\n  Best epoch (by val loss): {best_epoch}")

print("\nEvaluating ONCE on TEST (no tuning on test):")
_y_test_pred_ph6 = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).reshape(-1)
_y_test_true_ph6 = scaler_y.inverse_transform(y_test).reshape(-1)
print(f"  R2={r2_score(_y_test_true_ph6, _y_test_pred_ph6):.4f}  "
      f"RMSE={np.sqrt(mean_squared_error(_y_test_true_ph6, _y_test_pred_ph6)):.4f}  "
      f"MAE={np.mean(np.abs(_y_test_true_ph6 - _y_test_pred_ph6)):.4f}")

print("\n" + "="*70)
print("PHASE 6 DONE")
print("="*70)

# ============================================================
# PHASE 7: OPTIONAL RETRAIN ON TRAIN+VAL (CLEAN)
# ============================================================
if DO_OPTIONAL_RETRAIN:
    print("\n" + "="*70)
    print("PHASE 7: OPTIONAL RETRAIN ON TRAIN+VAL")
    print("="*70)

    X_trainval_orig = np.vstack([X_train_orig, X_val_orig])
    y_trainval_orig = np.vstack([y_train_orig, y_val_orig])

    # FIX #7 (documented): scalers are intentionally refit on TRAIN+VAL here.
    scaler_X = StandardScaler().fit(X_trainval_orig)
    scaler_y = StandardScaler().fit(y_trainval_orig)

    X_trainval = scaler_X.transform(X_trainval_orig)
    y_trainval = scaler_y.transform(y_trainval_orig)

    tf.keras.backend.clear_session()
    model_retrain = tuner.hypermodel.build(best_hp)

    # FIX #10: seed retrain model from best fold weights too
    if TRANSFER_BEST_FOLD and best_fold_weights is not None:
        print("[Weight Transfer] Seeding retrain model from best CV fold...")
        try:
            model_retrain.set_weights(best_fold_weights)
            print("  Full weight transfer successful")
        except ValueError:
            print("  Direct transfer failed, using smart per-layer transfer...")
            temp_source = tuner.hypermodel.build(best_hp)
            temp_source.set_weights(best_fold_weights)
            transfer_weights_smart(temp_source, model_retrain, mode=TRANSFER_MODE, verbose=False)
            del temp_source

    model_retrain.fit(
        X_trainval, y_trainval,
        epochs=best_epoch,
        batch_size=32,
        verbose=1,
    )

    X_test_rt      = scaler_X.transform(X_test_orig)
    y_test_pred_rt = scaler_y.inverse_transform(
        model_retrain.predict(X_test_rt, verbose=0)
    ).reshape(-1)
    y_test_inv_rt  = y_test_orig.reshape(-1)

    print(f"\nTEST (retrained model, TRAIN+VAL scalers):")
    print(f"  R2={r2_score(y_test_inv_rt, y_test_pred_rt):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_test_inv_rt, y_test_pred_rt)):.4f}  "
          f"MAE={np.mean(np.abs(y_test_inv_rt - y_test_pred_rt)):.4f}")

    # Promote retrained artifacts
    model   = model_retrain
    X_train = scaler_X.transform(X_train_orig)
    X_val   = scaler_X.transform(X_val_orig)
    X_test  = scaler_X.transform(X_test_orig)
    y_train = scaler_y.transform(y_train_orig)
    y_val   = scaler_y.transform(y_val_orig)
    y_test  = scaler_y.transform(y_test_orig)
    print("\nRetrained model and TRAIN+VAL scalers are now active for export/predictions.")

    print("\n" + "="*70)
    print("PHASE 7 DONE")
    print("="*70)

# ============================================================
# PHASE 8: METRICS, PREDICTION INTERVALS & EXPORT
# ============================================================
print("\n" + "="*70)
print("PHASE 8: METRICS, PREDICTION INTERVALS & EXPORT")
print("="*70)

# --- Inverse-scaled predictions ---
y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
y_val_pred   = scaler_y.inverse_transform(model.predict(X_val,   verbose=0))
y_test_pred  = scaler_y.inverse_transform(model.predict(X_test,  verbose=0))

y_train_inv  = scaler_y.inverse_transform(y_train)
y_val_inv    = scaler_y.inverse_transform(y_val)
y_test_inv   = scaler_y.inverse_transform(y_test)

# --- Metrics helper ---
def compute_basic_metrics(y_true, y_pred):
    actual    = np.array(y_true).reshape(-1)
    pred      = np.array(y_pred).reshape(-1)
    n         = len(actual)
    residuals = actual - pred
    SSE       = np.sum(residuals ** 2)
    MSE       = SSE / n if n > 0 else np.nan
    RMSE      = np.sqrt(MSE) if not np.isnan(MSE) else np.nan
    MAE       = np.mean(np.abs(residuals)) if n > 0 else np.nan
    SEP_val   = np.sqrt(SSE / (n - 1)) if n > 1 else np.nan
    mean_abs  = np.mean(np.abs(actual))
    MRPD      = (100.0 * np.sum(np.abs(residuals)) / (n * mean_abs)
                 if n > 0 and not np.isclose(mean_abs, 0.0) else np.nan)
    R2        = r2_score(actual, pred) if n > 0 else np.nan
    return {"n": n, "SSE": SSE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE,
            "SEP": SEP_val, "MRPD_percent": MRPD, "R2": R2}

metrics_train = compute_basic_metrics(y_train_inv, y_train_pred)
metrics_val   = compute_basic_metrics(y_val_inv,   y_val_pred)
metrics_test  = compute_basic_metrics(y_test_inv,  y_test_pred)

# Adjusted R2
p = X_train.shape[1]
for m in (metrics_train, metrics_val, metrics_test):
    n, r2 = m["n"], m["R2"]
    m["R2_adj"] = (1 - (1 - r2) * (n - 1) / (n - p - 1)
                   if (n - p - 1) > 0 else np.nan)

metrics_train["Predicted_R2_Q2"] = pred_R2
metrics_val["Predicted_R2_Q2"]   = np.nan
metrics_test["Predicted_R2_Q2"]  = np.nan

print("\n=== Final Metrics ===")
for label, m in [(f"Train ({TRAIN_PERCENT}%)", metrics_train),
                 (f"Val   ({VAL_PERCENT}%)",   metrics_val),
                 (f"Test  ({TEST_PERCENT}%)",  metrics_test)]:
    print(f"\n{label}:")
    for k, v in m.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

# FIX #6: PI_CALIBRATION and PI_ALPHA are now fully implemented
print(f"\n=== Empirical Prediction Intervals ({int((1 - PI_ALPHA) * 100)}% PI) ===")
if PI_CALIBRATION == "oof":
    residuals_cal = y_oof_true_inv - y_oof_pred_inv
    cal_label     = "OOF residuals (train CV)"
else:   # "val"
    residuals_cal = y_val_inv.reshape(-1) - y_val_pred.reshape(-1)
    cal_label     = "val residuals"

pi_lo_q = np.quantile(residuals_cal, PI_ALPHA / 2)
pi_hi_q = np.quantile(residuals_cal, 1 - PI_ALPHA / 2)
print(f"Calibrated on {cal_label}: offset = [{pi_lo_q:+.4f}, {pi_hi_q:+.4f}]")

def add_pi(pred_arr):
    p = pred_arr.reshape(-1)
    return p + pi_lo_q, p + pi_hi_q

y_train_pi_lo, y_train_pi_hi = add_pi(y_train_pred)
y_val_pi_lo,   y_val_pi_hi   = add_pi(y_val_pred)
y_test_pi_lo,  y_test_pi_hi  = add_pi(y_test_pred)

for label, y_true, lo, hi in [
    ("Train", y_train_inv.reshape(-1), y_train_pi_lo, y_train_pi_hi),
    ("Val",   y_val_inv.reshape(-1),   y_val_pi_lo,   y_val_pi_hi),
    ("Test",  y_test_inv.reshape(-1),  y_test_pi_lo,  y_test_pi_hi),
]:
    cov = np.mean((y_true >= lo) & (y_true <= hi))
    print(f"  {label} empirical coverage: {100 * cov:.1f}%")

# --- Save model & scalers ---
model.save(os.path.join(EXPORT_DIR, "final_model.keras"))
try:
    model.save(os.path.join(EXPORT_DIR, "final_model.h5"))
except Exception as e:
    print(f"Could not save .h5 (non-fatal): {e}")
joblib.dump(scaler_X, os.path.join(EXPORT_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(EXPORT_DIR, "scaler_y.pkl"))
print(f"\nModel & scalers saved to '{EXPORT_DIR}/'")

# --- Dense architecture summary ---
try:
    dense_layers = [l for l in model.layers if isinstance(l, layers.Dense)]
    if dense_layers:
        print("\n=== Final Dense Architecture ===")
        scheme_lines = []
        for i, layer in enumerate(dense_layers, start=1):
            act_name = layer.activation.__name__ if layer.activation else "N/A"
            reg      = layer.kernel_regularizer
            reg_str  = (f"L2={reg.l2}" if reg is not None and hasattr(reg, 'l2')
                        else str(reg))
            line = (f"  Layer {i}: '{layer.name}' | units={layer.units} | "
                    f"act={act_name} | reg={reg_str}")
            print(line)
            scheme_lines.append(line)
        scheme_path = os.path.join(EXPORT_DIR, "model_scheme.txt")
        with open(scheme_path, "w", encoding="utf-8") as f:
            f.write("Final Dense Architecture\n========================\n")
            f.write("\n".join(scheme_lines) + "\n")
        print(f"Architecture saved to '{scheme_path}'")
except Exception as e:
    print(f"Error saving architecture: {e}")

# --- Build and export result DataFrames (with PI columns) ---
pi_col = int((1 - PI_ALPHA) * 100)

def make_result_df(lbl_df, inp_df, y_true, y_pred, lo, hi):
    out = pd.concat(
        [lbl_df.reset_index(drop=True), inp_df.reset_index(drop=True)], axis=1
    )
    out["y_true"]           = np.array(y_true).reshape(-1)
    out["y_pred"]           = np.array(y_pred).reshape(-1)
    out[f"PI_{pi_col}_lo"]  = lo
    out[f"PI_{pi_col}_hi"]  = hi
    return out

df_train_res = make_result_df(labels_train.reset_index(drop=True), X_train_df,
                               y_train_inv, y_train_pred, y_train_pi_lo, y_train_pi_hi)
df_val_res   = make_result_df(labels_val.reset_index(drop=True),   X_val_df,
                               y_val_inv,   y_val_pred,   y_val_pi_lo,   y_val_pi_hi)
df_test_res  = make_result_df(labels_test.reset_index(drop=True),  X_test_df,
                               y_test_inv,  y_test_pred,  y_test_pi_lo,  y_test_pi_hi)

for name, frame in [("results_train.csv", df_train_res),
                    ("results_val.csv",   df_val_res),
                    ("results_test.csv",  df_test_res)]:
    path = os.path.join(EXPORT_DIR, name)
    frame.to_csv(path, index=False)
    print(f"Saved: {path}")

if use_colab:
    for fname in ["results_train.csv", "results_val.csv", "results_test.csv",
                  "final_model.keras", "scaler_X.pkl", "scaler_y.pkl"]:
        fpath = os.path.join(EXPORT_DIR, fname)
        try:
            colab_files.download(fpath)
        except Exception as e:
            print(f"Download failed for {fpath}: {e}")

print("\n" + "="*70)
print("ALL PHASES COMPLETE")
print("="*70)

# ============================================================
# PHASE 9: 3D SURFACE PLOTS + EXPORT
# ============================================================
print("\n" + "="*70)
print("PHASE 9: 3D SURFACE PLOTS + EXPORT")
print("="*70)

from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------- PLOT USER SETTINGS ----------------
FEATURE_X = "MnCO3"
FEATURE_Y = "FeCO3"

GRID_N_SINGLE = 50
RANGE_MODE = "quantile"
Q_LOW, Q_HIGH = 0.02, 0.98
HOLD_MODE = "median_train"
ROW_INDEX = 0

OVERLAY_TRAIN_SCATTER = True
SCATTER_ALPHA = 0.25
SCATTER_SIZE = 10

# Multi-pair settings
FEATURES_TO_USE = None   # e.g. ["X1","X2","X3"] or [0,1,2] or None
MAX_PAIRS = 12
GRID_N_MULTI = 35
GRID_RANGE_MODE = "train_quantile"
Q_LOW_MULTI, Q_HIGH_MULTI = 0.02, 0.98
HOLD_MODE_MULTI = "median_train"
ROW_INDEX_MULTI = 0
SCATTER_ALPHA_MULTI = 0.25
SCATTER_SIZE_MULTI = 8

# Export settings
GRID_N_EXPORT = 30
RANGE_MODE_EXPORT = "quantile"
Q_LOW_EXPORT, Q_HIGH_EXPORT = 0.02, 0.98
HOLD_MODE_EXPORT = "median_train"
ROW_INDEX_EXPORT = 0
DPI = 600
FIG_WIDTH = 8.3 / 2.54  # 8.3 cm -> inches
# ------------------------------------------------

if "inputs_columns" not in globals():
    if "inputs_df" in globals():
        inputs_columns = list(inputs_df.columns)
    else:
        inputs_columns = list(range(X_train_orig.shape[1]))

def _grid_vals(col_idx, grid_n, range_mode, q_low, q_high):
    v = X_train_orig[:, col_idx]
    if range_mode == "minmax":
        lo, hi = float(np.min(v)), float(np.max(v))
    elif range_mode in ("quantile", "train_quantile"):
        lo, hi = float(np.quantile(v, q_low)), float(np.quantile(v, q_high))
    else:
        raise ValueError(f"range_mode must be 'quantile', 'train_quantile', or 'minmax', got: {range_mode}")
    if np.isclose(lo, hi):
        lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, grid_n)

def predict_from_origX(X_orig_2d):
    X_scaled = scaler_X.transform(X_orig_2d)
    y_scaled = model.predict(X_scaled, verbose=0)
    return scaler_y.inverse_transform(y_scaled).reshape(-1)

def _build_xref(hold_mode, row_index):
    if hold_mode == "median_train":
        return np.median(X_train_orig, axis=0)
    elif hold_mode == "mean_train":
        return np.mean(X_train_orig, axis=0)
    elif hold_mode == "row":
        if row_index < 0 or row_index >= len(X_train_orig):
            raise IndexError(f"ROW_INDEX out of range: {row_index}")
        return X_train_orig[row_index].copy()
    else:
        raise ValueError("HOLD_MODE must be one of: median_train, mean_train, row")

def _colname(i):
    return inputs_columns[i] if isinstance(inputs_columns[i], (str, int)) else str(inputs_columns[i])

def _safe_name(s):
    s = str(s)
    for ch in [" ", "/", "\\", ":", ";", "|", "(", ")", "[", "]", "{", "}", "%"]:
        s = s.replace(ch, "_")
    return s

# ---- Single surface: MnCO3 vs FeCO3 ----
if FEATURE_X in inputs_columns and FEATURE_Y in inputs_columns:
    ix = inputs_columns.index(FEATURE_X)
    iy = inputs_columns.index(FEATURE_Y)
    Xref = _build_xref(HOLD_MODE, ROW_INDEX)

    x_vals = _grid_vals(ix, GRID_N_SINGLE, RANGE_MODE, Q_LOW, Q_HIGH)
    y_vals = _grid_vals(iy, GRID_N_SINGLE, RANGE_MODE, Q_LOW, Q_HIGH)
    XX, YY = np.meshgrid(x_vals, y_vals)

    Xgrid = np.tile(Xref.reshape(1, -1), (XX.size, 1))
    Xgrid[:, ix] = XX.reshape(-1)
    Xgrid[:, iy] = YY.reshape(-1)

    Z = predict_from_origX(Xgrid).reshape(XX.shape)

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.85))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XX, YY, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

    if OVERLAY_TRAIN_SCATTER:
        z_train_pred = predict_from_origX(X_train_orig)
        ax.scatter(X_train_orig[:, ix], X_train_orig[:, iy], z_train_pred, c="k", s=SCATTER_SIZE, alpha=SCATTER_ALPHA)

    ax.set_title(f"Predicted surface: {FEATURE_X} vs {FEATURE_Y}\n(others held constant: {HOLD_MODE})")
    ax.set_xlabel(FEATURE_X)
    ax.set_ylabel(FEATURE_Y)
    ax.set_zlabel("Predicted output")
    ax.view_init(elev=25, azim=-135)
    plt.tight_layout()
    plt.show()
    plt.close()
else:
    print(f"Skipping MnCO3 vs FeCO3 plot: features not found in {inputs_columns}")

# ---- Multi-pair surfaces ----
n_features = X_train_orig.shape[1]
if FEATURES_TO_USE is None:
    feat_indices = list(range(n_features))
else:
    feat_indices = []
    for f in FEATURES_TO_USE:
        if isinstance(f, int):
            feat_indices.append(f)
        else:
            if f not in inputs_columns:
                raise ValueError(f"Feature name '{f}' not in inputs_columns.")
            feat_indices.append(inputs_columns.index(f))

feat_indices = [i for i in feat_indices if 0 <= i < n_features]
if len(feat_indices) >= 2:
    pairs = list(combinations(feat_indices, 2))[:MAX_PAIRS]
    Xref_multi = _build_xref(HOLD_MODE_MULTI, ROW_INDEX_MULTI)

    print(f"Plotting {len(pairs)} 3D surfaces; HOLD_MODE={HOLD_MODE_MULTI}, GRID_N={GRID_N_MULTI}, RANGE={GRID_RANGE_MODE}")

    if OVERLAY_TRAIN_SCATTER:
        y_train_pred_vis = predict_from_origX(X_train_orig)

    for (i, j) in pairs:
        xi_vals = _grid_vals(i, GRID_N_MULTI, GRID_RANGE_MODE, Q_LOW_MULTI, Q_HIGH_MULTI)
        xj_vals = _grid_vals(j, GRID_N_MULTI, GRID_RANGE_MODE, Q_LOW_MULTI, Q_HIGH_MULTI)
        XI, XJ = np.meshgrid(xi_vals, xj_vals)

        Xgrid = np.tile(Xref_multi.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)

        Y = predict_from_origX(Xgrid).reshape(XI.shape)

        fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.85))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XI, XJ, Y, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        if OVERLAY_TRAIN_SCATTER:
            ax.scatter(X_train_orig[:, i], X_train_orig[:, j], y_train_pred_vis, c="k", s=SCATTER_SIZE_MULTI, alpha=SCATTER_ALPHA_MULTI)

        ax.set_title(f"Predicted surface: {_colname(i)} vs {_colname(j)}\n(others held constant: {HOLD_MODE_MULTI})")
        ax.set_xlabel(str(_colname(i)))
        ax.set_ylabel(str(_colname(j)))
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()
        plt.show()
        plt.close()
else:
    print("Need at least 2 features to plot multi-pair 3D surfaces.")

# ---- Export ALL 3D surfaces to PNG ----
try:
    plots_dir_3d = os.path.join(EXPORT_DIR, "plots_3d")
    os.makedirs(plots_dir_3d, exist_ok=True)

    p3d = X_train_orig.shape[1]
    Xref_3d = _build_xref(HOLD_MODE_EXPORT, ROW_INDEX_EXPORT)
    pairs_3d = list(combinations(range(p3d), 2))
    print(f"\nGenerating {len(pairs_3d)} 3D surfaces into: {plots_dir_3d}")

    for (i, j) in pairs_3d:
        xi = _grid_vals(i, GRID_N_EXPORT, RANGE_MODE_EXPORT, Q_LOW_EXPORT, Q_HIGH_EXPORT)
        xj = _grid_vals(j, GRID_N_EXPORT, RANGE_MODE_EXPORT, Q_LOW_EXPORT, Q_HIGH_EXPORT)
        XI, XJ = np.meshgrid(xi, xj)

        Xgrid = np.tile(Xref_3d.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)

        Z = predict_from_origX(Xgrid).reshape(XI.shape)

        fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.85))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

        ax.set_title(f"Predicted surface: {inputs_columns[i]} vs {inputs_columns[j]}\n(others held constant: {HOLD_MODE_EXPORT})")
        ax.set_xlabel(str(inputs_columns[i]))
        ax.set_ylabel(str(inputs_columns[j]))
        ax.set_zlabel("Predicted output")
        ax.view_init(elev=25, azim=-135)
        plt.tight_layout()

        out_tiff = os.path.join(
            plots_dir_3d,
            f"surface_{_safe_name(inputs_columns[i])}_vs_{_safe_name(inputs_columns[j])}_hold_{HOLD_MODE_EXPORT}.tiff"
        )
        plt.savefig(out_tiff, dpi=DPI, format='tiff')
        plt.close(fig)

    print(f"3D surfaces saved as TIFF ({DPI} dpi, {FIG_WIDTH}in width). Count: {len([f for f in os.listdir(plots_dir_3d) if f.lower().endswith('.tiff')])}")
except Exception as e:
    print("Error while generating 3D surfaces:", e)
    traceback.print_exc()

print("\n" + "="*70)
print("PHASE 9 DONE - 3D surface plots generated and exported")
print("="*70)
