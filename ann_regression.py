# Colab cell: Optimized ANN regression with configurable train/val/test split
# STRICT NO-LEAKAGE evaluation:
# - Split train/val/test
# - Tuner on train, validate on val
# - CV inside train with fold-fitted scalers (no leakage)
# - Final training on train, validate on val (early stop), then evaluate once on test
# - Optional retrain on train+val for best_epoch (val-chosen), with scalers refit on train+val, then evaluate once on test
#
# FIXES INCORPORATED:
# 1) "Clean" optional retrain: refit X/y scalers on train+val before retraining + test eval
# 2) Better uncertainty bands: calibrated using VAL residuals (or OOF residuals from CV), producing empirical quantile interval
#
# PATCH FIXES (2026-02-24):
# - Fixed typo: reshape(--1) -> reshape(-1)
# - Fixed date written to stats file (now uses current date dynamically)
# - Safer metric evaluation: flatten arrays before sklearn metrics
#
# ACTIVATION CHANGES (2026-05-11):
# - Hidden layers: ReLU -> LeakyReLU (alpha=0.1)
# - Output layer: linear -> sigmoid
# - Fixed architecture to match pretrained Hill model (128->64->32->1)
# - Name-based weight transfer for compatibility with warmup_model
#
# INCLUDES TWO DOWNLOAD SECTIONS:
# 1. After model training - downloads train/val/test results
# 2. After new data predictions - downloads new data predictions
#
# Paste this entire cell into Colab and run. Adjust USER CONFIG as needed.

# --- Detect notebook / Colab ---
try:
    get_ipython  # type: ignore
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False

# --- Install dependencies (if running in notebook) ---
if IN_NOTEBOOK:
    print("Installing (if missing) keras-tuner, seaborn, openpyxl, joblib...")
    get_ipython().run_line_magic('pip', 'install -q "keras-tuner" seaborn openpyxl joblib')
else:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tuner", "seaborn", "openpyxl", "joblib"])

# --- Imports ---
import os
import shutil
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

# ----------------- USER CONFIGURATION -----------------
TRAIN_PERCENT = 60
VAL_PERCENT = 20
TEST_PERCENT = 20

N_LABELS = 7
N_INPUTS = 10
TARGET_COL = None          # If not None: absolute column index of target in df
SEP = "\t"

DISABLE_GPU = True

TUNER_TRIALS = 15
K_FOLDS = 15
RANDOM_SEED = 42
TUNER_EPOCHS = 250
CV_EPOCHS = 200
FINAL_EPOCHS = 250

DO_OPTIONAL_RETRAIN = True  # retrain on train+val for val-selected best_epoch (with scalers refit on train+val)
PI_CALIBRATION = "val"      # "val" or "oof" for empirical prediction interval calibration
PI_ALPHA = 0.05             # 95% PI => alpha=0.05 (2.5%/97.5% quantiles)

EXPORT_DIR = "optimized_model"

# Activation configuration
HIDDEN_ACTIVATION = "leaky_relu"   # LeakyReLU for hidden layers
LEAKY_ALPHA = 0.1                  # negative slope for LeakyReLU
OUTPUT_ACTIVATION = "sigmoid"      # sigmoid for output layer
# ------------------------------------------------------

total_percent = TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT
if abs(total_percent - 100) > 0.01:
    raise ValueError(f"Split percentages must sum to 100. Got: {TRAIN_PERCENT}% + {VAL_PERCENT}% + {TEST_PERCENT}% = {total_percent}%")

print(f"✓ Data split configuration: TRAIN={TRAIN_PERCENT}%, VAL={VAL_PERCENT}%, TEST={TEST_PERCENT}%")
print(f"✓ Activation configuration: hidden={HIDDEN_ACTIVATION}(alpha={LEAKY_ALPHA}), output={OUTPUT_ACTIVATION}")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if DISABLE_GPU:
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled (if present). Using CPU.")
    except Exception as e:
        print("Could not change GPU visibility:", e)

# --- Robust table loader (Excel + common text encodings + separator sniff) ---
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

# --- Upload / load data ---
print("📂 Upload your dataset (dialog appears if running in Colab).")
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

print(f"✅ Data loaded. Shape: {df.shape}")
_show(df.head())

# --- Column selection (flexible) ---
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
        print("N_LABELS + N_INPUTS >= total columns. Using last column as target and the middle columns as inputs.")
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

# ============ CONFIGURABLE SPLIT ============
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

print(f"\nSplit sizes (rows): train={len(X_train_orig)} val={len(X_val_orig)} test={len(X_test_orig)}")
print(f"Split percentages: train={100*len(X_train_orig)/len(df):.1f}% val={100*len(X_val_orig)/len(df):.1f}% test={100*len(X_test_orig)/len(df):.1f}%")

# --- Scaling for tuning + final training (fit only on TRAIN to avoid leakage) ---
scaler_X = StandardScaler().fit(X_train_orig)
scaler_y = StandardScaler().fit(y_train_orig)

X_train = scaler_X.transform(X_train_orig)
X_val   = scaler_X.transform(X_val_orig)
X_test  = scaler_X.transform(X_test_orig)

y_train = scaler_y.transform(y_train_orig)
y_val   = scaler_y.transform(y_val_orig)
y_test  = scaler_y.transform(y_test_orig)

# --- Fixed-architecture model builder (matches pretrained Hill model) ---
# Architecture: 128 -> 64 -> 32 -> 1 (same layer names as warmup_model)
# Hidden layers: LeakyReLU | Output layer: Sigmoid
def build_base_model(l2_val=1e-3, d1=0.2, d2=0.2, d3=0.1, lr=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation=None, kernel_regularizer=regularizers.l2(l2_val), name="dense_128"),
        layers.LeakyReLU(negative_slope=LEAKY_ALPHA),
        layers.Dropout(d1, name="drop_1"),
        layers.Dense(64, activation=None, kernel_regularizer=regularizers.l2(l2_val), name="dense_64"),
        layers.LeakyReLU(negative_slope=LEAKY_ALPHA),
        layers.Dropout(d2, name="drop_2"),
        layers.Dense(32, activation=None, kernel_regularizer=regularizers.l2(l2_val), name="dense_32"),
        layers.LeakyReLU(negative_slope=LEAKY_ALPHA),
        layers.Dropout(d3, name="drop_3"),
        layers.Dense(1, activation=OUTPUT_ACTIVATION, name="dense_out"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model

def build_model_fixed(hp):
    l2_val = hp.Choice('l2_reg', [1e-4, 1e-3, 1e-2])
    d1     = hp.Float('dropout_1', 0.0, 0.3, step=0.05)
    d2     = hp.Float('dropout_2', 0.0, 0.3, step=0.05)
    d3     = hp.Float('dropout_3', 0.0, 0.2, step=0.05)
    lr     = hp.Choice('lr', [1e-4, 5e-4, 1e-3, 5e-3])
    return build_base_model(l2_val=l2_val, d1=d1, d2=d2, d3=d3, lr=lr)

# --- Keras-Tuner search (RandomSearch) ---
tuner = kt.RandomSearch(
    build_model_fixed,
    objective='val_loss',
    max_trials=TUNER_TRIALS,
    executions_per_trial=1,
    directory='tuner_results',
    project_name=f'ann_{TRAIN_PERCENT}_{VAL_PERCENT}_{TEST_PERCENT}'
)

print(f"Starting hyperparameter search ({TUNER_TRIALS} trials). This may take a while.")
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=TUNER_EPOCHS, batch_size=32, verbose=1)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best hyperparameters found:")
for k, v in best_hp.values.items():
    print(f"  {k}: {v}")

# ---- Weight transfer from warmup_model (name-based Dense matching) ----
print("\n" + "="*70)
print("WEIGHT TRANSFER: Checking for pre-trained Hill model...")
print("="*70)

_model_with_pretrain = None

if 'warmup_model' in globals():
    print("✓ Found pre-trained warmup_model!")
    try:
        model_for_transfer = build_model_fixed(best_hp)

        warmup_dense = {l.name: l for l in warmup_model.layers if isinstance(l, layers.Dense)}
        target_dense = {l.name: l for l in model_for_transfer.layers if isinstance(l, layers.Dense)}

        n_transferred = 0
        for name, t_layer in target_dense.items():
            if name not in warmup_dense:
                print(f"  - Skip {name}: not found in warmup")
                continue
            w_layer = warmup_dense[name]
            w_w = w_layer.get_weights()
            t_w = t_layer.get_weights()

            same = (len(w_w) == len(t_w)) and all(a.shape == b.shape for a, b in zip(w_w, t_w))
            if same:
                t_layer.set_weights(w_w)
                n_transferred += 1
                print(f"  ✓ Transferred {name}")
            else:
                print(f"  ✗ Shape mismatch {name}")

        if n_transferred > 0:
            print(f"✓ Transferred Dense layers: {n_transferred}")
            _model_with_pretrain = model_for_transfer
        else:
            print("⚠️ No layers transferred; continuing with tuned initialization.")
    except Exception as e:
        print(f"✗ Weight transfer failed: {e}")
else:
    print("ℹ️  No pre-trained model found (run the Hill pre-training cell first)")

print("="*70 + "\n")

# ======================================================
# STRICT NO-LEAKAGE CV inside TRAIN with fold-fitted scalers
# ======================================================
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

r2_scores, rmse_scores, mae_scores = [], [], []
y_oof_pred_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)
y_oof_true_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)

print(f"\nRunning {K_FOLDS}-Fold CV on TRAIN split ONLY with fold-fitted scalers (no leakage)...")

# Check for pre-trained model once (before loop)
if '_model_with_pretrain' in globals() and _model_with_pretrain is not None:
    print("✓ Using pre-trained model from Cell B")
    base_model = _model_with_pretrain
    use_pretrain = True
else:
    print("⚠️  Pre-trained model not found, building fresh model...")
    use_pretrain = False

print(f"✓ Model ready for k-fold CV\n")

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

    if use_pretrain:
        model_fold.set_weights(base_model.get_weights())

    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model_fold.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=TUNER_EPOCHS,
        batch_size=32,
        callbacks=[es],
        verbose=1
    )

    y_va_pred_scaled = model_fold.predict(X_va, verbose=0)
    y_va_pred_inv = fold_scaler_y.inverse_transform(y_va_pred_scaled).reshape(-1)
    y_va_true_inv = y_va_orig.reshape(-1)

    y_oof_pred_inv[va_idx] = y_va_pred_inv
    y_oof_true_inv[va_idx] = y_va_true_inv

    r2 = r2_score(y_va_true_inv, y_va_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_va_true_inv, y_va_pred_inv))
    mae = np.mean(np.abs(y_va_true_inv - y_va_pred_inv))

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

# ======================================================
# Final training: TRAIN -> validate on VAL (no test leakage)
# ======================================================
print("\nFinal training: fit on TRAIN, validate on VAL (test untouched).")

os.makedirs(EXPORT_DIR, exist_ok=True)
checkpoint_path = os.path.join(EXPORT_DIR, "best_model.keras")
mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
es_final = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

tf.keras.backend.clear_session()
model = build_model_fixed(best_hp)
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
print(f"Best epoch selected by VAL loss: {best_epoch}")

# Evaluate ONCE on TEST (no tuning on test)
print("\nEvaluating once on TEST (no selection/tuning on test).")
y_test_pred_eval = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).reshape(-1)
y_test_inv_eval  = scaler_y.inverse_transform(y_test).reshape(-1)
print(f"TEST: R²={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}, "
      f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}")

# ======================================================
# Optional retrain on TRAIN+VAL for best_epoch (VAL-informed),
# with scalers refit on TRAIN+VAL (clean retrain), then evaluate once on TEST.
# ======================================================
if DO_OPTIONAL_RETRAIN:
    print("\nOptional retrain (clean): refit scalers on TRAIN+VAL, retrain for best_epoch, evaluate once on TEST.")
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
    model_retrain.fit(
        X_trainval_tv, y_trainval_tv,
        epochs=best_epoch,
        batch_size=32,
        verbose=1
    )

    y_test_pred_rt = scaler_y_tv.inverse_transform(model_retrain.predict(X_test_tv, verbose=0)).reshape(-1)
    y_test_inv_rt  = y_test_orig.reshape(-1)

    print(f"TEST (retrained): R²={r2_score(y_test_inv_rt, y_test_pred_rt):.4f}, "
          f"RMSE={np.sqrt(mean_squared_error(y_test_inv_rt, y_test_pred_rt)):.4f}, "
          f"MAE={np.mean(np.abs(y_test_inv_rt - y_test_pred_rt)):.4f}")

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
                    reg_str = f"L2={getattr(reg,'l2')}"
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
        scheme_path = os.path.join(EXPORT_DIR, "model_scheme.txt")
        with open(scheme_path, "w") as f:
            f.write("Final Dense Layer Scheme\n")
            f.write("========================\n\n")
            for line in scheme_lines:
                f.write(line + "\n")
        print(f"Model scheme saved to: {scheme_path}")
except Exception as e:
    print("Error printing model scheme:", e)
    traceback.print_exc()

# --- Compute metrics on train / val / test ---
def compute_basic_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    n = len(y_true)
    p = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else np.nan
    return {"R2": round(r2, 6), "Adj_R2": round(adj_r2, 6),
            "RMSE": round(rmse, 6), "MAE": round(mae, 6), "n": n, "p": p}

y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0)).reshape(-1)
y_val_pred   = scaler_y.inverse_transform(model.predict(X_val,   verbose=0)).reshape(-1)
y_test_pred  = scaler_y.inverse_transform(model.predict(X_test,  verbose=0)).reshape(-1)

y_train_inv = scaler_y.inverse_transform(y_train).reshape(-1)
y_val_inv   = scaler_y.inverse_transform(y_val).reshape(-1)
y_test_inv  = scaler_y.inverse_transform(y_test).reshape(-1)

p = X_train.shape[1]
metrics_train = compute_basic_metrics(y_train_inv, y_train_pred)
metrics_val   = compute_basic_metrics(y_val_inv,   y_val_pred)
metrics_test  = compute_basic_metrics(y_test_inv,  y_test_pred)

print(f"\nTrain set ({TRAIN_PERCENT}%):")
for k, v in metrics_train.items():
    print(f"  {k}: {v}")
print(f"Validation set ({VAL_PERCENT}%):")
for k, v in metrics_val.items():
    print(f"  {k}: {v}")
print(f"Test set ({TEST_PERCENT}%):")
for k, v in metrics_test.items():
    print(f"  {k}: {v}")

# --- Save summary statistics file ---
stats_path = os.path.join(EXPORT_DIR, "model_statistics.txt")
try:
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Model statistics summary\n")
        f.write("========================\n\n")
        f.write(f"Current date: {_dt.date.today().isoformat()}\n")
        f.write(f"Number of predictors (p): {p}\n")
        f.write(f"Data split: {TRAIN_PERCENT}% train / {VAL_PERCENT}% validation / {TEST_PERCENT}% test\n")
        f.write(f"Hidden activation: {HIDDEN_ACTIVATION} (alpha={LEAKY_ALPHA})\n")
        f.write(f"Output activation: {OUTPUT_ACTIVATION}\n\n")
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

# --- Plot evolution of MSE during final training ---
mse_img_path = os.path.join(EXPORT_DIR, "mse_evolution.png")
try:
    if 'history' in globals() and hasattr(history, "history"):
        hist = history.history
        loss = hist.get('loss', None)
        val_loss = hist.get('val_loss', None)
        if loss is not None:
            epochs = range(1, len(loss) + 1)
            plt.figure(figsize=(8,5))
            plt.plot(epochs, loss, label='Train MSE (loss)', marker='o')
            if val_loss is not None:
                plt.plot(epochs, val_loss, label='Validation MSE (val_loss)', marker='o')
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
    plt.figure(figsize=(6,6))
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
        plt.savefig(save_path, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

def plot_residuals_std_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = (np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1))
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    sns.histplot(std_res, bins=25, kde=True, color='gray', edgecolor='black')
    plt.title(f"{title_prefix} - Standardized residuals distribution")
    plt.xlabel("Standardized residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(std_res, marker='o', linestyle='-')
    plt.title(f"{title_prefix} - Standardized residuals (time series)")
    plt.xlabel("Observation index")
    plt.ylabel("Standardized residual")
    plt.grid(True)
    plt.tight_layout()
    if save_prefix:
        path = save_prefix + "_std_residuals_hist_ts.png"
        plt.savefig(path, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

def plot_residuals_raw_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = (np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1))
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    sns.histplot(residuals, bins=25, kde=True, color='salmon', edgecolor='black')
    plt.title(f"{title_prefix} - Raw residuals distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(residuals, marker='o', linestyle='-')
    plt.title(f"{title_prefix} - Raw residuals (time series)")
    plt.xlabel("Observation index")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    if save_prefix:
        path = save_prefix + "_raw_residuals_hist_ts.png"
        plt.savefig(path, dpi=150)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close()

# Generate and save diagnostic plots
try:
    plots_dir = os.path.join(EXPORT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_predicted_vs_actual(y_train_inv, y_train_pred, "Predicted vs Actual (Train)", save_path=os.path.join(plots_dir,"pred_vs_actual_train.png"))
    plot_residuals_std_and_timeseries(y_train_inv, y_train_pred, "Train", save_prefix=os.path.join(plots_dir,"train_std"))
    plot_residuals_raw_and_timeseries(y_train_inv, y_train_pred, "Train", save_prefix=os.path.join(plots_dir,"train_raw"))

    plot_predicted_vs_actual(y_val_inv, y_val_pred, "Predicted vs Actual (Validation)", save_path=os.path.join(plots_dir,"pred_vs_actual_val.png"))
    plot_residuals_std_and_timeseries(y_val_inv, y_val_pred, "Validation", save_prefix=os.path.join(plots_dir,"val_std"))
    plot_residuals_raw_and_timeseries(y_val_inv, y_val_pred, "Validation", save_prefix=os.path.join(plots_dir,"val_raw"))

    plot_predicted_vs_actual(y_test_inv, y_test_pred, "Predicted vs Actual (Test)", save_path=os.path.join(plots_dir,"pred_vs_actual_test.png"))
    plot_residuals_std_and_timeseries(y_test_inv, y_test_pred, "Test", save_prefix=os.path.join(plots_dir,"test_std"))
    plot_residuals_raw_and_timeseries(y_test_inv, y_test_pred, "Test", save_prefix=os.path.join(plots_dir,"test_raw"))

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
        abs_pct_train = np.where(np.abs(y_train_inv_flat) > 1e-12, 100.0 * abs_err_train / np.abs(y_train_inv_flat), np.nan)
        abs_pct_val   = np.where(np.abs(y_val_inv_flat)   > 1e-12, 100.0 * abs_err_val   / np.abs(y_val_inv_flat),   np.nan)
        abs_pct_test  = np.where(np.abs(y_test_inv_flat)  > 1e-12, 100.0 * abs_err_test  / np.abs(y_test_inv_flat),  np.nan)

    for full_rows, pred_flat, inv_flat, res, ae, apct, suffix in [
        (train_full_rows, y_train_pred_flat, y_train_inv_flat, residuals_train, abs_err_train, abs_pct_train, "train"),
        (val_full_rows,   y_val_pred_flat,   y_val_inv_flat,   residuals_val,   abs_err_val,   abs_pct_val,   "val"),
        (test_full_rows,  y_test_pred_flat,  y_test_inv_flat,  residuals_test,  abs_err_test,  abs_pct_test,  "test"),
    ]:
        full_rows["Predicted_Value"] = pred_flat
        full_rows["Actual_Value"] = inv_flat
        full_rows["Residual"] = res
        full_rows["Abs_Error"] = ae
        full_rows["Abs_Percent_Error"] = apct
        full_path_xlsx = os.path.join(EXPORT_DIR, f"{suffix}_full_with_all_columns.xlsx")
        full_path_csv  = os.path.join(EXPORT_DIR, f"{suffix}_full_with_all_columns.csv")
        full_rows.to_excel(full_path_xlsx, index=False, engine='openpyxl')
        full_rows.to_csv(full_path_csv, index=False)

    print("Full-row exports saved.")
except Exception as e:
    print("Error creating full-row exports:", e)
    traceback.print_exc()

# --- Save model + scalers ---
try:
    model.save(os.path.join(EXPORT_DIR, "final_model.keras"))
    model.save(os.path.join(EXPORT_DIR, "final_model.h5"))
    joblib.dump(scaler_X, os.path.join(EXPORT_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(EXPORT_DIR, "scaler_y.pkl"))
    print("Model and scalers saved.")
except Exception as e:
    print("Error saving model/scalers:", e)
    traceback.print_exc()

# ======================================================
# EXPORT ALL 3D SURFACES (predicted output vs every pair of inputs)
# Saved into: optimized_model/plots_3d/
# ======================================================
try:
    from itertools import combinations
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    GRID_N_3D = 30
    RANGE_MODE_3D = "quantile"
    Q_LOW_3D, Q_HIGH_3D = 0.02, 0.98
    HOLD_MODE_3D = "median_train"
    ROW_INDEX_3D = 0
    DPI_3D = 160

    plots_dir_3d = os.path.join(EXPORT_DIR, "plots_3d")
    os.makedirs(plots_dir_3d, exist_ok=True)

    if "inputs_columns" not in globals():
        try:
            inputs_columns = list(inputs_df.columns)
        except Exception:
            inputs_columns = [f"X{i}" for i in range(X_train_orig.shape[1])]

    p3d = X_train_orig.shape[1]

    if HOLD_MODE_3D == "median_train":
        Xref_3d = np.median(X_train_orig, axis=0)
    elif HOLD_MODE_3D == "mean_train":
        Xref_3d = np.mean(X_train_orig, axis=0)
    elif HOLD_MODE_3D == "row":
        Xref_3d = X_train_orig[int(ROW_INDEX_3D)].copy()
    else:
        raise ValueError("HOLD_MODE must be one of: median_train, mean_train, row")

    def _safe_name_3d(col):
        return str(col).replace("/", "_").replace(" ", "_").replace("\\", "_")

    def _grid_vals_3d(col_idx):
        v = X_train_orig[:, col_idx]
        if RANGE_MODE_3D == "quantile":
            lo, hi = float(np.quantile(v, Q_LOW_3D)), float(np.quantile(v, Q_HIGH_3D))
        else:
            lo, hi = float(np.min(v)), float(np.max(v))
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        return np.linspace(lo, hi, GRID_N_3D)

    def _predict_from_origX_3d(X_orig_2d):
        X_scaled = scaler_X.transform(X_orig_2d)
        y_scaled = model.predict(X_scaled, verbose=0)
        return scaler_y.inverse_transform(y_scaled).reshape(-1)

    pairs_3d = list(combinations(range(p3d), 2))
    print(f"\nGenerating {len(pairs_3d)} 3D surfaces into: {plots_dir_3d}")

    for (i, j) in pairs_3d:
        xi = _grid_vals_3d(i)
        xj = _grid_vals_3d(j)
        XI, XJ = np.meshgrid(xi, xj)

        Xgrid = np.tile(Xref_3d.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)

        Z = _predict_from_origX_3d(Xgrid).reshape(XI.shape)

        fig = plt.figure(figsize=(9, 7))
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
    print("⚠️  Error while generating 3D surfaces:", e)
    traceback.print_exc()

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

    import zipfile
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

print("\n" + "="*80 + "\n")

# ======================================================
# NEW DATA PREDICTION SECTION
# - Uses model + scaler_X/scaler_y currently in memory (after optional retrain, these are train+val-fit)
# - Computes calibrated empirical prediction interval using residuals from:
#   * VAL residuals (recommended) OR
#   * OOF residuals (from CV) as a robust alternative
# ======================================================
print("="*80)
print("🔮 NEW DATA PREDICTION SECTION")
print("="*80)

# --- Build calibration residuals for empirical PI ---
def get_calibration_residuals(source="val"):
    source = str(source).strip().lower()
    if source == "val":
        res = (y_val_inv.reshape(-1) - y_val_pred.reshape(-1))
        return res
    elif source == "oof":
        res = (y_oof_true_inv.reshape(-1) - y_oof_pred_inv.reshape(-1))
        return res
    else:
        raise ValueError("PI_CALIBRATION must be 'val' or 'oof'")

cal_residuals = get_calibration_residuals(PI_CALIBRATION)
q_low = np.quantile(cal_residuals, PI_ALPHA / 2.0)
q_high = np.quantile(cal_residuals, 1.0 - PI_ALPHA / 2.0)
print(f"Empirical PI calibration ({PI_CALIBRATION}): q_low={q_low:.4f}, q_high={q_high:.4f} (alpha={PI_ALPHA})")

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
            print(f"NaNs in new features: {nan_in_features}. Imputing with TRAIN means (fit on TRAIN only)...")
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
            new_results_compact = pd.concat([new_results_compact, new_labels_df.reset_index(drop=True)], axis=1)
        new_results_compact = pd.concat([new_results_compact, new_inputs_df.reset_index(drop=True)], axis=1)

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
                        print("NaNs in new target. Imputing with TRAIN target mean (fit on TRAIN only)...")
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

# ============ SECOND DOWNLOAD SECTION: NEW DATA PREDICTIONS ONLY ============
print("\n" + "="*80)
print("DOWNLOAD #2: NEW DATA PREDICTIONS (if new data was uploaded)")
print("="*80)

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

        import zipfile
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

print("\n" + "="*80)
print("ALL OPERATIONS COMPLETED!")
print("="*80)
print("\nDownloads Summary:")
print("  Download #1: training_results.zip - Ready")
print("  Download #2: new_data_predictions.zip - Ready (if data uploaded)")
print("\nTo download files from Colab:")
print("  1. Click the 'Files' folder on the left sidebar")
print("  2. Find the ZIP file you need")
print("  3. Right-click and select 'Download'")
print("\nFiles are ready to use!")
