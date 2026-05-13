# ============================================================
# MERGED CELL: HILL PRE-TRAINING + TUNING + WEIGHT TRANSFER
# ============================================================

# --- Install dependencies ---
try:
    get_ipython  # type: ignore
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False

if IN_NOTEBOOK:
    get_ipython().run_line_magic('pip', 'install -q "keras-tuner"')  # type: ignore
else:
    import subprocess, sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "keras-tuner"],
    )

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt

print("\n" + "=" * 70)
print("PHASE 1: HILL PRE-TRAINING")
print("=" * 70)

# Configuration
N_INPUTS = 10
N_SYNTHETIC = 500
HILL_V_MAX = 200
HILL_K = 0.10
HILL_N = 1.20
HILL_X_MIN = 0.1
HILL_X_MAX = 100
RANDOM_SEED = 42
PRETRAIN_EPOCHS = 50

print("[1/5] Setting seeds...")
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
print("done")


# Generate Hill data
print("\n[2/5] Generating synthetic Hill-function data...")


def _generate_hill_data(n_samples, n_features, v_max, k, n_hill, x_min, x_max):
    X = np.random.uniform(x_min, x_max, (n_samples, n_features)).astype(np.float32)
    x1 = X[:, 0]
    y = v_max * (x1**n_hill) / (k**n_hill + x1**n_hill)
    y += np.random.normal(0, 0.05 * v_max, n_samples)
    for i in range(1, min(n_features, 3)):
        y += 0.05 * v_max * (X[:, i] - x_min) / (x_max - x_min)
    return X, y.reshape(-1, 1)


X_h, y_h = _generate_hill_data(
    N_SYNTHETIC, N_INPUTS, HILL_V_MAX, HILL_K, HILL_N, HILL_X_MIN, HILL_X_MAX
)
print(f"  Generated: X {X_h.shape}, y {y_h.shape}")
print(f"  y range: [{y_h.min():.2f}, {y_h.max():.2f}]")

# Scale
print("\n[3/5] Scaling data...")
scaler_Xh = StandardScaler().fit(X_h)
scaler_yh = StandardScaler().fit(y_h)
X_h_scaled = scaler_Xh.transform(X_h)
y_h_scaled = scaler_yh.transform(y_h)

# Split into train/val
X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
    X_h_scaled, y_h_scaled, test_size=0.2, random_state=RANDOM_SEED
)
print(f"  Train: {X_h_train.shape}, Val: {X_h_val.shape}")

# Build warmup model
print("\n[4/5] Building warmup_model...")
tf.keras.backend.clear_session()

warmup_model = keras.Sequential(
    [
        layers.Input(shape=(N_INPUTS,)),
        layers.Dense(
            128, activation="relu", kernel_regularizer=regularizers.l2(1e-3)
        ),
        layers.Dropout(0.2),
        layers.Dense(
            64, activation="relu", kernel_regularizer=regularizers.l2(1e-3)
        ),
        layers.Dropout(0.2),
        layers.Dense(
            32, activation="relu", kernel_regularizer=regularizers.l2(1e-3)
        ),
        layers.Dropout(0.1),
        layers.Dense(1, activation="linear"),
    ]
)

warmup_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

print(
    f"  Model built: {len(warmup_model.layers)} layers, "
    f"{warmup_model.count_params():,} parameters"
)

# Pre-train
print(f"\n[5/5] Pre-training on Hill data ({PRETRAIN_EPOCHS} epochs)...")

es = callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = warmup_model.fit(
    X_h_train,
    y_h_train,
    validation_data=(X_h_val, y_h_val),
    epochs=PRETRAIN_EPOCHS,
    batch_size=32,
    callbacks=[es],
    verbose=1,
)

print(f"\n  Pre-training complete!")
print(f"  Final train loss: {history.history['loss'][-1]:.6f}")
print(f"  Final val loss:   {history.history['val_loss'][-1]:.6f}")

print("\n" + "=" * 70)
print("PHASE 1 DONE - warmup_model ready")
print("=" * 70)

# ============================================================
# PHASE 2: FLEXIBLE HYPERMODEL + TUNING
# ============================================================

X_train = X_h_train
y_train = y_h_train
X_val = X_h_val
y_val = y_h_val

print("\n" + "=" * 70)
print("PHASE 2: HYPERPARAMETER TUNING (FLEXIBLE ARCHITECTURE)")
print("=" * 70)

# Configuration
TUNER_TRIALS = 10
TUNER_EPOCHS = 50
TRAIN_PERCENT = 70
VAL_PERCENT = 15
TEST_PERCENT = 15

print("\n[1/3] Preparing training data...")
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")

# Define FLEXIBLE hypermodel
print("\n[2/3] Defining flexible hypermodel...")

N_FEATURES = X_train.shape[1]


def build_model(hp):
    """
    Flexible architecture: the number of hidden layers (1-5) and the
    number of units per layer are tunable, along with L2 regularisation,
    per-layer dropout, and learning rate.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(N_FEATURES,)))

    l2_val = hp.Choice("l2_reg", [1e-4, 1e-3, 1e-2])
    n_layers = hp.Int("n_layers", min_value=1, max_value=5, default=3)

    for i in range(n_layers):
        units = hp.Int(f"units_{i}", min_value=16, max_value=256, step=16, default=128 // (2 ** min(i, 2)))
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_val),
            )
        )
        dropout = hp.Float(f"dropout_{i}", 0.0, 0.4, step=0.05, default=0.1)
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("lr", [1e-4, 5e-4, 1e-3, 5e-3])
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model


print("  Flexible hypermodel defined (1-5 layers, 16-256 units each)")

# Create tuner
print("\n[3/3] Running hyperparameter search...")
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=TUNER_TRIALS,
    executions_per_trial=1,
    overwrite=True,
    directory="tuner_results",
    project_name=f"ann_{TRAIN_PERCENT}_{VAL_PERCENT}_{TEST_PERCENT}",
)

print(f"Starting hyperparameter search ({TUNER_TRIALS} trials)...")
tuner.search(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=TUNER_EPOCHS,
    batch_size=32,
    verbose=1,
)

best_hp = tuner.get_best_hyperparameters(1)[0]
print("\n  Best hyperparameters found:")
for k, v in best_hp.values.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("PHASE 2 DONE - Best hyperparameters found")
print("=" * 70)

# ============================================================
# PHASE 3: WEIGHT TRANSFER
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3: WEIGHT TRANSFER FROM WARMUP MODEL")
print("=" * 70)

_model_with_pretrain = None
model_for_transfer = None

try:
    print("\n[1/3] Building model from best hyperparameters...")
    model_for_transfer = build_model(best_hp)
    print(f"  Model built: {model_for_transfer.count_params():,} parameters")

    print("\n[2/3] Comparing architectures...")
    dense_warmup = [l for l in warmup_model.layers if isinstance(l, layers.Dense)]
    dense_final = [l for l in model_for_transfer.layers if isinstance(l, layers.Dense)]
    print(f"  Warmup Dense layers: {len(dense_warmup)}  sizes: {[l.units for l in dense_warmup]}")
    print(f"  Tuned  Dense layers: {len(dense_final)}  sizes: {[l.units for l in dense_final]}")

    print("\n[3/3] Transferring weights (shape-matched layers only)...")

    n_transferred = 0
    for idx, (lw, lf) in enumerate(zip(dense_warmup, dense_final)):
        try:
            if len(lf.weights) != len(lw.weights):
                print(f"  - layer {idx} ({lw.name}): different number of weight tensors, skipped")
                continue
            shapes_match = all(
                wf.shape == ww.shape for wf, ww in zip(lf.weights, lw.weights)
            )
            if shapes_match:
                lf.set_weights(lw.get_weights())
                n_transferred += 1
                print(f"  + layer {idx} ({lw.name} {lw.weights[0].shape} -> {lf.name}): transferred")
            else:
                print(
                    f"  - layer {idx} ({lw.name}): shape mismatch "
                    f"({[w.shape for w in lw.weights]} vs "
                    f"{[w.shape for w in lf.weights]}), skipped"
                )
        except Exception as e:
            print(f"  - layer {idx} ({lw.name}): error: {e}")

    if n_transferred > 0:
        print(f"\n  Transferred {n_transferred}/{min(len(dense_warmup), len(dense_final))} Dense layers")
        _model_with_pretrain = model_for_transfer
        print("  -> Use _model_with_pretrain for further training")
    else:
        print("\n  No weights transferred (architectures are incompatible)")

except Exception as e:
    print(f"  Weight transfer failed: {e}")
    import traceback

    traceback.print_exc()
    _model_with_pretrain = None

print("\n" + "=" * 70)
if _model_with_pretrain is not None:
    print("WEIGHT TRANSFER COMPLETE")
    print("Model is pre-trained and ready for fine-tuning")
else:
    print("Proceeding without pre-training")
    if model_for_transfer is not None:
        _model_with_pretrain = model_for_transfer
        print("  Falling back to tuned model without transferred weights")
    else:
        print("  model_for_transfer unavailable -- re-run Phase 2 first")
print("=" * 70 + "\n")
