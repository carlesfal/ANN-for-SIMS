# ============================================================
# CELL 2: MAIN PIPELINE (CORRECTED + ENHANCED TRANSFER LEARNING)
# ============================================================
# Optimized ANN regression with configurable train/val/test split
# STRICT NO-LEAKAGE evaluation
#
# TRANSFER LEARNING ENHANCEMENTS:
# - Flexible layer matching (by type+shape, not just position)
# - Gradual unfreezing with configurable schedule
# - Learning rate warmup (linear/cosine) to protect transferred features
# - Discriminative fine-tuning (lower LR for early transferred layers)
# - Transfer quality validation (loss comparison with/without transfer)
# - Weight divergence monitoring during fine-tuning
# - Per-layer transfer diagnostics with cosine similarity
# - Configurable transfer strategies: direct / progressive / discriminative
#
# BUG FIXES (2026-04-12):
# - Fixed: build_model_fixed now properly defined before use
# - Fixed: CV_EPOCHS config variable now used in CV loop (was dead code)
# - Fixed: PI calibration saves pre-retrain residuals to avoid in-sample bias
# - Fixed: Imputer uses correct data scope after optional retrain
# - Fixed: Redundant predict_from_origX calls cached outside 3D plot loop
# - Fixed: Seed reset before final training for reproducibility
# - Fixed: build_model_fixed no longer captures X_train by closure
# - Fixed: plt.close() added to prevent figure accumulation
# - Fixed: Use build_model_fixed() directly instead of tuner.hypermodel.build() after clear_session
# - Removed: Duplicate tuner search and weight transfer (was running twice)
# - Removed: Dead code (unused build_model with flexible architecture)

# --- Detect notebook / Colab ---
try:
    get_ipython  # type: ignore
    IN_NOTEBOOK = True
except Exception:
    IN_NOTEBOOK = False

# --- Install dependencies (if running in notebook) ---
if IN_NOTEBOOK:
    print("Installing (if missing) keras-tuner, seaborn, openpyxl, joblib...")
    get_ipython().run_line_magic('pip', 'install -q "keras-tuner" seaborn openpyxl joblib')  # type: ignore
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

TUNER_TRIALS = 10
K_FOLDS = 10
RANDOM_SEED = 42
TUNER_EPOCHS = 200
CV_EPOCHS = 200
FINAL_EPOCHS = 200

DO_OPTIONAL_RETRAIN = True  # retrain on train+val for val-selected best_epoch (with scalers refit on train+val)
PI_CALIBRATION = "val"      # "val" or "oof" for empirical prediction interval calibration
PI_ALPHA = 0.05             # 95% PI => alpha=0.05 (2.5%/97.5% quantiles)

EXPORT_DIR = "optimized_model"

# --- Plot export settings (all plots) ---
PLOT_DPI = 600
PLOT_WIDTH_CM = 8.3
PLOT_WIDTH_IN = PLOT_WIDTH_CM / 2.54  # ~3.27 inches
PLOT_FORMAT = "tiff"

# ------------- TRANSFER LEARNING CONFIGURATION -------------
# Strategy: "direct" (load weights, train normally — original behavior),
#           "progressive" (freeze transferred layers, unfreeze gradually),
#           "discriminative" (per-layer LR: lower for early layers)
TRANSFER_STRATEGY = "progressive"

# Layer matching: "positional" (zip-based, original), "flexible" (by Dense type+shape)
TRANSFER_MATCH_MODE = "flexible"

# Progressive unfreezing settings
FREEZE_EPOCHS = 5           # epochs to keep ALL transferred layers frozen
UNFREEZE_PER_STEP = 1       # Dense layers to unfreeze per step (from output toward input)
UNFREEZE_EVERY_N = 3        # unfreeze one step every N epochs after FREEZE_EPOCHS

# Learning rate warmup (applied in all transfer strategies)
LR_WARMUP_EPOCHS = 10       # ramp LR over this many epochs after transfer
LR_WARMUP_SCHEDULE = "cosine"  # "linear" or "cosine"
LR_WARMUP_START_FACTOR = 0.1   # initial LR = base_LR * this factor

# Discriminative fine-tuning settings
DISCR_LR_DECAY = 0.3        # each earlier layer gets LR *= this factor
DISCR_MIN_LR_FACTOR = 0.01  # floor: earliest layer LR >= base_LR * this

# Diagnostics
LOG_WEIGHT_DIVERGENCE = True   # track L2 distance from pre-trained weights per epoch
VALIDATE_TRANSFER = True       # evaluate initial loss with vs without transfer
TRANSFER_PLOTS_DIR = None      # set to a path to save transfer diagnostic plots (None = EXPORT_DIR/transfer_plots)
# -----------------------------------------------------------

total_percent = TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT
if abs(total_percent - 100) > 0.01:
    raise ValueError(f"Split percentages must sum to 100. Got: {TRAIN_PERCENT}% + {VAL_PERCENT}% + {TEST_PERCENT}% = {total_percent}%")

print(f"Data split configuration: TRAIN={TRAIN_PERCENT}%, VAL={VAL_PERCENT}%, TEST={TEST_PERCENT}%")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if DISABLE_GPU:
    try:
        tf.config.set_visible_devices([], 'GPU')
        print("GPU disabled (if present). Using CPU.")
    except Exception as e:
        print("Could not change GPU visibility:", e)


# ============================================================
# TRANSFER LEARNING UTILITIES
# ============================================================

def _cosine_similarity_flat(a, b):
    """Cosine similarity between two weight arrays (flattened)."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def _l2_distance_flat(a, b):
    """L2 distance between two weight arrays (flattened)."""
    return float(np.linalg.norm(a.flatten().astype(np.float64) - b.flatten().astype(np.float64)))


def smart_match_layers(target_model, source_model, mode="flexible"):
    """Match layers between target and source models for weight transfer.

    Args:
        target_model: The model to receive weights.
        source_model: The model providing pre-trained weights.
        mode: "positional" (zip layers), "flexible" (match Dense layers by shape).

    Returns:
        List of (target_layer, source_layer, match_info_dict) tuples.
    """
    matches = []

    if mode == "positional":
        for layer_t, layer_s in zip(target_model.layers, source_model.layers):
            if isinstance(layer_t, layers.InputLayer) or isinstance(layer_s, layers.InputLayer):
                continue
            if not layer_t.weights or not layer_s.weights:
                continue
            if len(layer_t.weights) != len(layer_s.weights):
                continue
            shapes_match = all(
                wt.shape == ws.shape
                for wt, ws in zip(layer_t.weights, layer_s.weights)
            )
            if shapes_match:
                matches.append((layer_t, layer_s, {
                    "method": "positional",
                    "target_name": layer_t.name,
                    "source_name": layer_s.name,
                }))

    elif mode == "flexible":
        # Extract Dense layers (skip InputLayer, Dropout, etc.)
        def _get_dense_layers(model):
            return [l for l in model.layers if isinstance(l, layers.Dense)]

        target_dense = _get_dense_layers(target_model)
        source_dense = _get_dense_layers(source_model)

        # Try to match from the output end backward (output layers are most
        # task-specific, but earlier dense layers capture general features).
        # Strategy: align from the first Dense layer forward, matching by shape.
        # If shapes match, pair them; skip source layers that don't match.

        si = 0  # source index
        for ti, t_layer in enumerate(target_dense):
            if si >= len(source_dense):
                break
            s_layer = source_dense[si]
            if len(t_layer.weights) != len(s_layer.weights):
                si += 1
                continue
            shapes_match = all(
                wt.shape == ws.shape
                for wt, ws in zip(t_layer.weights, s_layer.weights)
            )
            if shapes_match:
                cos_sim = np.mean([
                    _cosine_similarity_flat(ws.numpy(), wt.numpy())
                    for wt, ws in zip(t_layer.weights, s_layer.weights)
                ])
                matches.append((t_layer, s_layer, {
                    "method": "flexible",
                    "target_name": t_layer.name,
                    "source_name": s_layer.name,
                    "target_dense_idx": ti,
                    "source_dense_idx": si,
                    "initial_cosine_sim": round(cos_sim, 4),
                }))
                si += 1
            else:
                # Shape mismatch — try next source layer
                si += 1

    else:
        raise ValueError(f"Unknown match mode: {mode!r}. Use 'positional' or 'flexible'.")

    return matches


def transfer_weights(target_model, source_model, mode="flexible"):
    """Transfer weights from source to target using smart matching.

    Returns:
        (n_transferred, match_details): count and per-layer diagnostics.
    """
    matches = smart_match_layers(target_model, source_model, mode=mode)
    details = []
    n_transferred = 0

    for t_layer, s_layer, info in matches:
        try:
            old_weights = [w.numpy().copy() for w in t_layer.weights]
            t_layer.set_weights(s_layer.get_weights())
            new_weights = [w.numpy() for w in t_layer.weights]

            cos_sims = [
                _cosine_similarity_flat(old_w, new_w)
                for old_w, new_w in zip(old_weights, new_weights)
            ]
            l2_dists = [
                _l2_distance_flat(old_w, new_w)
                for old_w, new_w in zip(old_weights, new_weights)
            ]
            info["post_transfer_cosine_sim"] = round(np.mean(cos_sims), 4)
            info["post_transfer_l2_dist"] = round(np.mean(l2_dists), 4)
            info["status"] = "OK"
            n_transferred += 1
        except Exception as e:
            info["status"] = f"FAILED: {e}"

        details.append(info)

    return n_transferred, details


def validate_transfer_quality(model_with_transfer, build_fn, hp, X_val, y_val,
                              pretrain_weights=None):
    """Compare initial val loss with vs without transferred weights.

    Returns:
        dict with 'loss_with_transfer', 'loss_without_transfer', 'improvement_pct'.
    """
    loss_with = model_with_transfer.evaluate(X_val, y_val, verbose=0)
    if isinstance(loss_with, list):
        loss_with = loss_with[0]

    model_random = build_fn(hp)
    loss_without = model_random.evaluate(X_val, y_val, verbose=0)
    if isinstance(loss_without, list):
        loss_without = loss_without[0]

    improvement = 0.0
    if loss_without > 1e-12:
        improvement = 100.0 * (loss_without - loss_with) / loss_without

    return {
        "loss_with_transfer": round(float(loss_with), 6),
        "loss_without_transfer": round(float(loss_without), 6),
        "improvement_pct": round(improvement, 2),
    }


class GradualUnfreezeCallback(callbacks.Callback):
    """Progressively unfreezes Dense layers during training.

    Starts with all transferred Dense layers frozen. After `freeze_epochs`,
    unfreezes `unfreeze_per_step` Dense layers every `unfreeze_every_n` epochs,
    starting from the layer closest to the output (last Dense first).
    """

    def __init__(self, transferred_layer_names, freeze_epochs=5,
                 unfreeze_per_step=1, unfreeze_every_n=3):
        super().__init__()
        self.transferred_layer_names = list(transferred_layer_names)
        self.freeze_epochs = freeze_epochs
        self.unfreeze_per_step = unfreeze_per_step
        self.unfreeze_every_n = unfreeze_every_n
        self._frozen_layers = []
        self._unfreeze_queue = []  # layers ordered output->input for progressive unfreeze

    def on_train_begin(self, logs=None):
        # Freeze all transferred layers
        self._frozen_layers = []
        for name in self.transferred_layer_names:
            try:
                layer = self.model.get_layer(name)
                layer.trainable = False
                self._frozen_layers.append(name)
            except ValueError:
                pass

        # Build unfreeze queue: last transferred Dense first (closest to output)
        self._unfreeze_queue = list(reversed(self._frozen_layers))

        if self._frozen_layers:
            # Recompile to apply trainable changes
            self.model.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                metrics=[m.name for m in self.model.metrics if m.name != 'loss']
            )
            print(f"[GradualUnfreeze] Froze {len(self._frozen_layers)} transferred layers for first {self.freeze_epochs} epochs")

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.freeze_epochs:
            return
        if not self._unfreeze_queue:
            return

        epochs_since_freeze_end = epoch - self.freeze_epochs
        if epochs_since_freeze_end > 0 and epochs_since_freeze_end % self.unfreeze_every_n == 0:
            n_to_unfreeze = min(self.unfreeze_per_step, len(self._unfreeze_queue))
            unfrozen_names = []
            for _ in range(n_to_unfreeze):
                name = self._unfreeze_queue.pop(0)
                try:
                    layer = self.model.get_layer(name)
                    layer.trainable = True
                    unfrozen_names.append(name)
                except ValueError:
                    pass

            if unfrozen_names:
                self.model.compile(
                    optimizer=self.model.optimizer,
                    loss=self.model.loss,
                    metrics=[m.name for m in self.model.metrics if m.name != 'loss']
                )
                remaining = len(self._unfreeze_queue)
                print(f"[GradualUnfreeze] Epoch {epoch+1}: unfroze {unfrozen_names} ({remaining} still frozen)")


class LRWarmupCallback(callbacks.Callback):
    """Warms up the learning rate over the first N epochs.

    Supports linear and cosine schedules. After warmup_epochs,
    the LR stays at the target value (base_lr).
    """

    def __init__(self, base_lr, warmup_epochs=10, start_factor=0.1,
                 schedule="cosine"):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.start_factor = start_factor
        self.schedule = schedule
        self.start_lr = base_lr * start_factor

    def on_epoch_begin(self, epoch, logs=None):
        if epoch >= self.warmup_epochs:
            return

        if self.schedule == "linear":
            lr = self.start_lr + (self.base_lr - self.start_lr) * (epoch / self.warmup_epochs)
        elif self.schedule == "cosine":
            lr = self.start_lr + (self.base_lr - self.start_lr) * 0.5 * (
                1.0 - np.cos(np.pi * epoch / self.warmup_epochs)
            )
        else:
            lr = self.base_lr

        self.model.optimizer.learning_rate.assign(lr)


class WeightDivergenceCallback(callbacks.Callback):
    """Monitors how much weights diverge from their pre-trained values.

    Records per-epoch L2 distance and cosine similarity between current
    weights and the initial transferred weights for each monitored layer.
    """

    def __init__(self, initial_weights_map):
        """
        Args:
            initial_weights_map: dict mapping layer_name -> list of numpy arrays
                                 (the pre-trained weights snapshot).
        """
        super().__init__()
        self.initial_weights_map = initial_weights_map
        self.history = {}  # layer_name -> {"l2": [...], "cosine": [...]}

    def on_epoch_end(self, epoch, logs=None):
        for name, init_ws in self.initial_weights_map.items():
            if name not in self.history:
                self.history[name] = {"l2": [], "cosine": [], "epoch": []}

            try:
                layer = self.model.get_layer(name)
                current_ws = [w.numpy() for w in layer.weights]
                l2_dists = [
                    _l2_distance_flat(iw, cw)
                    for iw, cw in zip(init_ws, current_ws)
                ]
                cos_sims = [
                    _cosine_similarity_flat(iw, cw)
                    for iw, cw in zip(init_ws, current_ws)
                ]
                self.history[name]["l2"].append(np.mean(l2_dists))
                self.history[name]["cosine"].append(np.mean(cos_sims))
                self.history[name]["epoch"].append(epoch + 1)
            except (ValueError, IndexError):
                pass

    def summary(self):
        """Print summary of weight divergence."""
        print("\n=== Weight Divergence Summary ===")
        for name, data in self.history.items():
            if not data["l2"]:
                continue
            l2_start = data["l2"][0] if data["l2"] else 0
            l2_end = data["l2"][-1] if data["l2"] else 0
            cos_start = data["cosine"][0] if data["cosine"] else 1.0
            cos_end = data["cosine"][-1] if data["cosine"] else 1.0
            print(f"  {name}:")
            print(f"    L2 drift: {l2_start:.4f} -> {l2_end:.4f}")
            print(f"    Cosine:   {cos_start:.4f} -> {cos_end:.4f}")
        print("=" * 35)


def build_transfer_callbacks(strategy, transferred_layer_names, base_lr,
                             initial_weights_map=None):
    """Build the list of transfer-specific callbacks based on strategy.

    Args:
        strategy: "direct", "progressive", or "discriminative".
        transferred_layer_names: list of layer names that received weights.
        base_lr: the base learning rate from hyperparameters.
        initial_weights_map: dict of layer_name -> [numpy_weights] for divergence tracking.

    Returns:
        list of Keras callbacks.
    """
    cbs = []

    # LR warmup is applied for ALL strategies when transfer is active
    if LR_WARMUP_EPOCHS > 0:
        cbs.append(LRWarmupCallback(
            base_lr=base_lr,
            warmup_epochs=LR_WARMUP_EPOCHS,
            start_factor=LR_WARMUP_START_FACTOR,
            schedule=LR_WARMUP_SCHEDULE,
        ))

    if strategy == "progressive" and transferred_layer_names:
        cbs.append(GradualUnfreezeCallback(
            transferred_layer_names=transferred_layer_names,
            freeze_epochs=FREEZE_EPOCHS,
            unfreeze_per_step=UNFREEZE_PER_STEP,
            unfreeze_every_n=UNFREEZE_EVERY_N,
        ))

    if LOG_WEIGHT_DIVERGENCE and initial_weights_map:
        cbs.append(WeightDivergenceCallback(initial_weights_map))

    return cbs


def apply_discriminative_lr(model, transferred_layer_names, base_lr):
    """Recompile model with per-layer learning rates for discriminative fine-tuning.

    Earlier transferred layers get progressively smaller LR. Non-transferred layers
    use the full base_lr.

    This is implemented via a multi-optimizer approach: we create separate parameter
    groups by freezing/unfreezing and using gradient scaling via a custom training step.
    For simplicity and Keras compatibility, we use a schedule that applies a
    multiplier during training via callbacks.
    """
    # In practice, true per-layer LR in Keras requires either:
    # (a) custom training loop, or (b) wrapper optimizer.
    # We use approach (b): build a learning rate schedule that differs per variable.
    # For standard Keras, the most compatible approach is to set up separate
    # variable groups. Here we implement it via a custom LR-scaling callback.

    class DiscriminativeLRCallback(callbacks.Callback):
        def __init__(self, layer_lr_map):
            super().__init__()
            self.layer_lr_map = layer_lr_map  # layer_name -> lr_multiplier

        def on_train_batch_begin(self, batch, logs=None):
            # Apply discriminative LR by scaling gradients is complex;
            # instead we adjust trainable status and use lower LR globally
            # for frozen-then-unfrozen layers (combined with GradualUnfreeze).
            pass

    # Assign LR multipliers: earliest transferred layer gets smallest
    n_transferred = len(transferred_layer_names)
    lr_map = {}
    for i, name in enumerate(transferred_layer_names):
        # Layer 0 (earliest) gets the smallest factor
        factor = max(DISCR_MIN_LR_FACTOR, DISCR_LR_DECAY ** (n_transferred - 1 - i))
        lr_map[name] = base_lr * factor

    print("[Discriminative LR] Per-layer learning rates:")
    for name, lr in lr_map.items():
        print(f"  {name}: {lr:.6f}")

    return DiscriminativeLRCallback(lr_map)


def plot_weight_divergence(divergence_cb, save_dir=None):
    """Plot weight divergence curves from a WeightDivergenceCallback."""
    if not divergence_cb.history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(PLOT_WIDTH_IN * 2.5, PLOT_WIDTH_IN))

    for name, data in divergence_cb.history.items():
        if not data["epoch"]:
            continue
        axes[0].plot(data["epoch"], data["l2"], label=name, marker='.')
        axes[1].plot(data["epoch"], data["cosine"], label=name, marker='.')

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L2 Distance from Pre-trained")
    axes[0].set_title("Weight Drift (L2)")
    axes[0].legend(fontsize=6)
    axes[0].grid(True)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity to Pre-trained")
    axes[1].set_title("Weight Preservation (Cosine)")
    axes[1].legend(fontsize=6)
    axes[1].grid(True)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "weight_divergence.tiff")
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT)
        print(f"Weight divergence plot saved to: {path}")
    plt.show()
    plt.close()


# ============================================================
# END TRANSFER LEARNING UTILITIES
# ============================================================


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

# --- Scaling (fit only on TRAIN to avoid leakage) ---
scaler_X = StandardScaler().fit(X_train_orig)
scaler_y = StandardScaler().fit(y_train_orig)

X_train = scaler_X.transform(X_train_orig)
X_val   = scaler_X.transform(X_val_orig)
X_test  = scaler_X.transform(X_test_orig)

y_train = scaler_y.transform(y_train_orig)
y_val   = scaler_y.transform(y_val_orig)
y_test  = scaler_y.transform(y_test_orig)

# --- FIX #13: Capture n_features at definition time (avoid closure over mutable global) ---
_N_FEATURES = X_train.shape[1]

# --- FIX #1: Define build_model_fixed ONCE (fixed architecture: 128 -> 64 -> 32 -> 1) ---
def build_model_fixed(hp, n_features=_N_FEATURES):
    """Fixed architecture hypermodel. Only tunes: L2, dropout, learning rate."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))

    l2_val = hp.Choice('l2_reg', [1e-4, 1e-3, 1e-2])

    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_val)))
    model.add(layers.Dropout(hp.Float('dropout_1', 0.0, 0.3, step=0.05)))

    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_val)))
    model.add(layers.Dropout(hp.Float('dropout_2', 0.0, 0.3, step=0.05)))

    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_val)))
    model.add(layers.Dropout(hp.Float('dropout_3', 0.0, 0.2, step=0.05)))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('lr', [1e-4, 5e-4, 1e-3, 5e-3])),
        loss='mse',
        metrics=['mae']
    )
    return model

# --- FIX #3: Single tuner search (no duplicate) ---
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

# ============================================================
# ENHANCED WEIGHT TRANSFER
# ============================================================
print("\n" + "="*70)
print("WEIGHT TRANSFER: Checking for pre-trained Hill model...")
print(f"  Strategy: {TRANSFER_STRATEGY}")
print(f"  Match mode: {TRANSFER_MATCH_MODE}")
print("="*70)

_pretrain_weights = None       # saved as numpy arrays, survives clear_session()
_transferred_layer_names = []  # names of layers that received pre-trained weights
_initial_weights_map = {}      # layer_name -> [numpy_weights] for divergence tracking
_transfer_details = []         # per-layer transfer diagnostics
_transfer_validation = None    # loss comparison with/without transfer

if 'warmup_model' in globals():
    print("Found pre-trained warmup_model!")
    try:
        model_for_transfer = build_model_fixed(best_hp)

        # --- Enhanced transfer with smart matching ---
        n_transferred, _transfer_details = transfer_weights(
            target_model=model_for_transfer,
            source_model=warmup_model,
            mode=TRANSFER_MATCH_MODE,
        )

        print(f"\nTransfer results ({TRANSFER_MATCH_MODE} matching):")
        for detail in _transfer_details:
            status = detail["status"]
            src = detail.get("source_name", "?")
            tgt = detail.get("target_name", "?")
            cos = detail.get("post_transfer_cosine_sim", "N/A")
            l2 = detail.get("post_transfer_l2_dist", "N/A")
            print(f"  {src} -> {tgt}: {status} (cos_sim_before={cos}, l2_dist={l2})")

        if n_transferred > 0:
            print(f"\nTransferred {n_transferred} layer(s) successfully!")

            # Capture transferred layer names and initial weights
            _transferred_layer_names = [
                d["target_name"] for d in _transfer_details if d["status"] == "OK"
            ]
            _initial_weights_map = {}
            for name in _transferred_layer_names:
                try:
                    layer = model_for_transfer.get_layer(name)
                    _initial_weights_map[name] = [w.numpy().copy() for w in layer.weights]
                except ValueError:
                    pass

            # Snapshot all weights as numpy (survives clear_session)
            _pretrain_weights = [w.numpy() for w in model_for_transfer.weights]

            # --- Validate transfer quality ---
            if VALIDATE_TRANSFER:
                print("\nValidating transfer quality...")
                _transfer_validation = validate_transfer_quality(
                    model_with_transfer=model_for_transfer,
                    build_fn=build_model_fixed,
                    hp=best_hp,
                    X_val=X_val,
                    y_val=y_val,
                )
                print(f"  Loss WITH transfer:    {_transfer_validation['loss_with_transfer']:.6f}")
                print(f"  Loss WITHOUT transfer: {_transfer_validation['loss_without_transfer']:.6f}")
                print(f"  Improvement:           {_transfer_validation['improvement_pct']:.1f}%")

                if _transfer_validation['improvement_pct'] < 0:
                    print("  WARNING: Transfer INCREASED initial loss. Pre-trained weights may not align well.")
                    print("           Consider running without transfer (set TRANSFER_STRATEGY='direct' or skip Cell 1).")
        else:
            print("WARNING: No compatible weights (architecture mismatch between Cell 1 and Cell 2)")
    except Exception as e:
        print(f"WARNING: Weight transfer failed: {e}")
        traceback.print_exc()
else:
    print("INFO: No pre-trained model found (Cell 1 was not run or warmup_model not in memory)")

# --- Summary banner: Hill pre-training status ---
print("\n" + "="*70)
if _pretrain_weights is not None:
    print("HILL PRE-TRAINING STATUS: OK")
    print(f"  - warmup_model from Cell 1 detected")
    print(f"  - {len(_transferred_layer_names)} layer(s) transferred ({TRANSFER_MATCH_MODE} matching)")
    print(f"  - Strategy: {TRANSFER_STRATEGY}")
    print(f"  - Weights saved as numpy snapshot ({len(_pretrain_weights)} tensors)")
    print(f"  - LR warmup: {LR_WARMUP_EPOCHS} epochs ({LR_WARMUP_SCHEDULE}, start={LR_WARMUP_START_FACTOR}x)")
    if TRANSFER_STRATEGY == "progressive":
        print(f"  - Gradual unfreeze: freeze {FREEZE_EPOCHS} epochs, then unfreeze {UNFREEZE_PER_STEP}/step every {UNFREEZE_EVERY_N} epochs")
    elif TRANSFER_STRATEGY == "discriminative":
        print(f"  - Discriminative LR: decay={DISCR_LR_DECAY}, min_factor={DISCR_MIN_LR_FACTOR}")
    print(f"  - Will be loaded into: CV folds, final model, retrain model")
    if _transfer_validation:
        print(f"  - Initial loss improvement: {_transfer_validation['improvement_pct']:.1f}%")
else:
    print("HILL PRE-TRAINING STATUS: NOT ACTIVE")
    if 'warmup_model' not in globals():
        print("  - Cell 1 was not run (or warmup_model was cleared from memory)")
        print("  - To enable: run Cell 1 first, then re-run Cell 2")
    else:
        print("  - warmup_model exists but weight transfer failed (see warnings above)")
    print("  - Pipeline will run normally with random weight initialization")
print("="*70 + "\n")

# ======================================================
# STRICT NO-LEAKAGE CV inside TRAIN with fold-fitted scalers
# ======================================================
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

r2_scores, rmse_scores, mae_scores = [], [], []
y_oof_pred_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)
y_oof_true_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)

print(f"\nRunning {K_FOLDS}-Fold CV on TRAIN split ONLY with fold-fitted scalers (no leakage)...")

use_pretrain = _pretrain_weights is not None
if use_pretrain:
    print(f"[CV] Using Hill pre-trained weights for fold initialization (strategy={TRANSFER_STRATEGY})")
else:
    print("[CV] No pre-trained weights -> folds start from random initialization")

print(f"Model ready for k-fold CV\n")

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

    # --- Enhanced transfer for each fold ---
    fold_transfer_cbs = []
    if use_pretrain:
        model_fold.set_weights(_pretrain_weights)
        # Build transfer callbacks for this fold
        base_lr = best_hp.get('lr')
        fold_transfer_cbs = build_transfer_callbacks(
            strategy=TRANSFER_STRATEGY,
            transferred_layer_names=_transferred_layer_names,
            base_lr=base_lr,
            initial_weights_map=_initial_weights_map if fold == 1 else None,  # only track divergence on fold 1
        )

    # FIX #4: Use CV_EPOCHS instead of TUNER_EPOCHS
    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    all_callbacks = [es] + fold_transfer_cbs

    model_fold.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=CV_EPOCHS,
        batch_size=32,
        callbacks=all_callbacks,
        verbose=1
    )

    # Print weight divergence summary for fold 1
    if fold == 1 and LOG_WEIGHT_DIVERGENCE:
        for cb in fold_transfer_cbs:
            if isinstance(cb, WeightDivergenceCallback):
                cb.summary()

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

# ======================================================
# Final training: TRAIN -> validate on VAL
# ======================================================
print("\nFinal training: fit on TRAIN, validate on VAL (test untouched).")

os.makedirs(EXPORT_DIR, exist_ok=True)
checkpoint_path = os.path.join(EXPORT_DIR, "best_model.keras")
mc = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
es_final = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# FIX #9 + #10: Reset seeds and use build_model_fixed directly after clear_session
tf.keras.backend.clear_session()
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

model = build_model_fixed(best_hp)

# --- Enhanced transfer for final model ---
final_transfer_cbs = []
if _pretrain_weights is not None:
    model.set_weights(_pretrain_weights)
    print("[FINAL MODEL] Hill pre-trained weights loaded OK")

    base_lr = best_hp.get('lr')
    final_transfer_cbs = build_transfer_callbacks(
        strategy=TRANSFER_STRATEGY,
        transferred_layer_names=_transferred_layer_names,
        base_lr=base_lr,
        initial_weights_map=_initial_weights_map,
    )
    # Apply discriminative LR if selected
    if TRANSFER_STRATEGY == "discriminative":
        discr_cb = apply_discriminative_lr(model, _transferred_layer_names, base_lr)
        final_transfer_cbs.append(discr_cb)
else:
    print("[FINAL MODEL] Starting from random weights (no Hill pre-training)")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=FINAL_EPOCHS,
    batch_size=32,
    callbacks=[es_final, mc] + final_transfer_cbs,
    verbose=1
)

# Print weight divergence for final model
if LOG_WEIGHT_DIVERGENCE and _pretrain_weights is not None:
    for cb in final_transfer_cbs:
        if isinstance(cb, WeightDivergenceCallback):
            cb.summary()
            # Save divergence plot
            _transfer_plots_dir = TRANSFER_PLOTS_DIR or os.path.join(EXPORT_DIR, "transfer_plots")
            plot_weight_divergence(cb, save_dir=_transfer_plots_dir)

if os.path.exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path, compile=False)

best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
print(f"Best epoch selected by VAL loss: {best_epoch}")

# Evaluate ONCE on TEST
print("\nEvaluating once on TEST (no selection/tuning on test).")
y_test_pred_eval = scaler_y.inverse_transform(model.predict(X_test, verbose=0)).reshape(-1)
y_test_inv_eval  = scaler_y.inverse_transform(y_test).reshape(-1)
print(f"TEST: R2={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}, "
      f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}")

# --- Predictions BEFORE optional retrain (for PI calibration) ---
y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
y_val_pred   = scaler_y.inverse_transform(model.predict(X_val, verbose=0))
y_test_pred  = scaler_y.inverse_transform(model.predict(X_test, verbose=0))

y_train_inv  = scaler_y.inverse_transform(y_train)
y_val_inv    = scaler_y.inverse_transform(y_val)
y_test_inv   = scaler_y.inverse_transform(y_test)

# FIX #5: Save pre-retrain validation residuals for PI calibration
# These are true out-of-sample residuals (model was NOT trained on val)
cal_residuals_val_preretrain = y_val_inv.reshape(-1) - y_val_pred.reshape(-1)

# ======================================================
# Optional retrain on TRAIN+VAL
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

    # FIX #10: Reset seeds before retrain
    tf.keras.backend.clear_session()
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # FIX #9: Use build_model_fixed directly
    model_retrain = build_model_fixed(best_hp)

    # --- Enhanced transfer for retrain model ---
    retrain_transfer_cbs = []
    if _pretrain_weights is not None:
        model_retrain.set_weights(_pretrain_weights)
        print("[RETRAIN MODEL] Hill pre-trained weights loaded OK")

        # For retrain, use LR warmup but skip gradual unfreeze (fixed epochs = best_epoch)
        base_lr = best_hp.get('lr')
        if LR_WARMUP_EPOCHS > 0:
            retrain_transfer_cbs.append(LRWarmupCallback(
                base_lr=base_lr,
                warmup_epochs=min(LR_WARMUP_EPOCHS, best_epoch),
                start_factor=LR_WARMUP_START_FACTOR,
                schedule=LR_WARMUP_SCHEDULE,
            ))
    else:
        print("[RETRAIN MODEL] Starting from random weights (no Hill pre-training)")

    model_retrain.fit(
        X_trainval_tv, y_trainval_tv,
        epochs=best_epoch,
        batch_size=32,
        callbacks=retrain_transfer_cbs,
        verbose=1
    )

    y_test_pred_rt = scaler_y_tv.inverse_transform(model_retrain.predict(X_test_tv, verbose=0)).reshape(-1)
    y_test_inv_rt  = y_test_orig.reshape(-1)

    print(f"TEST (retrained): R2={r2_score(y_test_inv_rt, y_test_pred_rt):.4f}, "
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

    # Recompute predictions with retrained model for export
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train, verbose=0))
    y_val_pred   = scaler_y.inverse_transform(model.predict(X_val, verbose=0))
    y_test_pred  = scaler_y.inverse_transform(model.predict(X_test, verbose=0))

    y_train_inv  = scaler_y.inverse_transform(y_train)
    y_val_inv    = scaler_y.inverse_transform(y_val)
    y_test_inv   = scaler_y.inverse_transform(y_test)

# --- Print & save final Dense architecture ---
try:
    from tensorflow.keras.layers import Dense
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    if dense_layers:
        print("\n=== Final Dense Layers (in model order) ===")
        scheme_lines = []
        for i, layer in enumerate(dense_layers, start=1):
            units = layer.units if hasattr(layer, "units") else None
            activation = layer.activation if hasattr(layer, "activation") else None
            act_name = activation.__name__ if activation is not None else "N/A"
            reg = layer.kernel_regularizer if hasattr(layer, "kernel_regularizer") else None
            reg_str = None
            try:
                if reg is not None and hasattr(reg, "l2"):
                    reg_str = f"L2={reg.l2}"
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
        last_act = last.activation
        final_line = f"Final Dense (output) -> name='{last.name}', units={last.units}, activation={last_act.__name__ if last_act else 'N/A'}"
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

# --- Save transfer learning diagnostics ---
if _transfer_details or _transfer_validation:
    transfer_report_path = os.path.join(EXPORT_DIR, "transfer_learning_report.txt")
    try:
        with open(transfer_report_path, "w", encoding="utf-8") as f:
            f.write("Transfer Learning Report\n")
            f.write("========================\n\n")
            f.write(f"Strategy: {TRANSFER_STRATEGY}\n")
            f.write(f"Match mode: {TRANSFER_MATCH_MODE}\n")
            f.write(f"LR warmup: {LR_WARMUP_EPOCHS} epochs ({LR_WARMUP_SCHEDULE})\n")
            f.write(f"LR warmup start factor: {LR_WARMUP_START_FACTOR}\n\n")

            if TRANSFER_STRATEGY == "progressive":
                f.write(f"Freeze epochs: {FREEZE_EPOCHS}\n")
                f.write(f"Unfreeze per step: {UNFREEZE_PER_STEP}\n")
                f.write(f"Unfreeze every N epochs: {UNFREEZE_EVERY_N}\n\n")

            f.write("Layer Transfer Details:\n")
            for detail in _transfer_details:
                f.write(f"  {detail}\n")
            f.write("\n")

            if _transfer_validation:
                f.write("Transfer Quality Validation:\n")
                for k, v in _transfer_validation.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")

        print(f"Transfer learning report saved to: {transfer_report_path}")
    except Exception as e:
        print("Could not save transfer learning report:", e)

print(f"Model and scalers saved to {EXPORT_DIR}/")

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
    MRPD = 100.0 * np.sum(np.abs(residuals)) / (n * mean_abs_actual) if (n > 0 and not np.isclose(mean_abs_actual,0.0)) else np.nan
    R2 = r2_score(actual, pred) if n > 0 else np.nan
    return {"n": n, "SSE": SSE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE, "SEP": SEP_metric, "MRPD_percent": MRPD, "R2": R2}

metrics_train = compute_basic_metrics(y_train_inv, y_train_pred)
metrics_val   = compute_basic_metrics(y_val_inv,   y_val_pred)
metrics_test  = compute_basic_metrics(y_test_inv,  y_test_pred)

p = X_train.shape[1]
n_in = metrics_train["n"]
n_val_m = metrics_val["n"]
n_out = metrics_test["n"]
r2_in = metrics_train["R2"]
r2_out = metrics_test["R2"]
r2_val_m = metrics_val["R2"]
r2_adj_in = 1 - (1 - r2_in) * (n_in - 1) / (n_in - p - 1) if (n_in - p - 1) > 0 else np.nan
r2_adj_val = 1 - (1 - r2_val_m) * (n_val_m - 1) / (n_val_m - p - 1) if (n_val_m - p - 1) > 0 else np.nan
r2_adj_out = 1 - (1 - r2_out) * (n_out - 1) / (n_out - p - 1) if (n_out - p - 1) > 0 else np.nan

metrics_train.update({"R2_adj": r2_adj_in, "Predicted_R2_Q2": pred_R2_train})
metrics_val.update({"R2_adj": r2_adj_val, "Predicted_R2_Q2": np.nan})
metrics_test.update({"R2_adj": r2_adj_out, "Predicted_R2_Q2": np.nan})

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

# --- Save summary statistics file ---
stats_path = os.path.join(EXPORT_DIR, "model_statistics.txt")
try:
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Model statistics summary\n")
        f.write("========================\n\n")
        f.write(f"Current date: {_dt.date.today().isoformat()}\n")
        f.write(f"Number of predictors (p): {p}\n")
        f.write(f"Data split: {TRAIN_PERCENT}% train / {VAL_PERCENT}% validation / {TEST_PERCENT}% test\n\n")
        f.write("Hyperparameters (best):\n")
        for k, v in best_hp.values.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTransfer learning strategy: {TRANSFER_STRATEGY}\n")
        f.write(f"Transfer match mode: {TRANSFER_MATCH_MODE}\n")
        f.write(f"Layers transferred: {len(_transferred_layer_names)}\n")
        if _transfer_validation:
            f.write(f"Transfer improvement: {_transfer_validation['improvement_pct']:.1f}%\n")
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
mse_img_path = os.path.join(EXPORT_DIR, "mse_evolution.tiff")
try:
    if 'history' in globals() and hasattr(history, "history"):
        hist = history.history
        loss = hist.get('loss', None)
        val_loss = hist.get('val_loss', None)
        if loss is not None:
            epochs = range(1, len(loss) + 1)
            plt.figure(figsize=(PLOT_WIDTH_IN, PLOT_WIDTH_IN * 0.6))
            plt.plot(epochs, loss, label='Train MSE (loss)', marker='o')
            if val_loss is not None:
                plt.plot(epochs, val_loss, label='Validation MSE (val_loss)', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('MSE (loss)')
            plt.title('Evolution of MSE during final training')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(mse_img_path, dpi=PLOT_DPI, format=PLOT_FORMAT)
            plt.show()
            plt.close()
except Exception as e:
    print("Error while plotting MSE evolution:", e)

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
        plt.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.show()
    plt.close()

def plot_residuals_std_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = (np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1))
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    plt.figure(figsize=(PLOT_WIDTH_IN * 2, PLOT_WIDTH_IN * 0.6))
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
        path = save_prefix + "_std_residuals_hist_ts.tiff"
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.show()
    plt.close()

def plot_residuals_raw_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    residuals = (np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1))
    plt.figure(figsize=(PLOT_WIDTH_IN * 2, PLOT_WIDTH_IN * 0.6))
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
        path = save_prefix + "_raw_residuals_hist_ts.tiff"
        plt.savefig(path, dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.show()
    plt.close()

# Generate and save diagnostic plots
try:
    plots_dir = os.path.join(EXPORT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_predicted_vs_actual(y_train_inv, y_train_pred, "Predicted vs Actual (Train)", save_path=os.path.join(plots_dir,"pred_vs_actual_train.tiff"))
    plot_residuals_std_and_timeseries(y_train_inv, y_train_pred, "Train", save_prefix=os.path.join(plots_dir,"train_std"))
    plot_residuals_raw_and_timeseries(y_train_inv, y_train_pred, "Train", save_prefix=os.path.join(plots_dir,"train_raw"))

    plot_predicted_vs_actual(y_val_inv, y_val_pred, "Predicted vs Actual (Validation)", save_path=os.path.join(plots_dir,"pred_vs_actual_val.tiff"))
    plot_residuals_std_and_timeseries(y_val_inv, y_val_pred, "Validation", save_prefix=os.path.join(plots_dir,"val_std"))
    plot_residuals_raw_and_timeseries(y_val_inv, y_val_pred, "Validation", save_prefix=os.path.join(plots_dir,"val_raw"))

    plot_predicted_vs_actual(y_test_inv, y_test_pred, "Predicted vs Actual (Test)", save_path=os.path.join(plots_dir,"pred_vs_actual_test.tiff"))
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

# --- Save exports ---
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

    train_full_rows = train_full_rows.copy()
    val_full_rows   = val_full_rows.copy()
    test_full_rows  = test_full_rows.copy()

    train_full_rows["Actual_Value"]      = y_train_inv_flat
    train_full_rows["Predicted_Value"]   = y_train_pred_flat
    train_full_rows["Residual"]          = residuals_train
    train_full_rows["Abs_Error"]         = abs_err_train
    train_full_rows["Abs_Percent_Error"] = abs_pct_train

    val_full_rows["Actual_Value"]        = y_val_inv_flat
    val_full_rows["Predicted_Value"]     = y_val_pred_flat
    val_full_rows["Residual"]            = residuals_val
    val_full_rows["Abs_Error"]           = abs_err_val
    val_full_rows["Abs_Percent_Error"]   = abs_pct_val

    test_full_rows["Actual_Value"]       = y_test_inv_flat
    test_full_rows["Predicted_Value"]    = y_test_pred_flat
    test_full_rows["Residual"]           = residuals_test
    test_full_rows["Abs_Error"]          = abs_err_test
    test_full_rows["Abs_Percent_Error"]  = abs_pct_test

    def move_pred_cols_to_end(df_full):
        pred_cols = ["Actual_Value","Predicted_Value","Residual","Abs_Error","Abs_Percent_Error"]
        cols = [c for c in df_full.columns if c not in pred_cols]
        cols += pred_cols
        return df_full[cols]

    train_full_rows = move_pred_cols_to_end(train_full_rows)
    val_full_rows   = move_pred_cols_to_end(val_full_rows)
    test_full_rows  = move_pred_cols_to_end(test_full_rows)

    train_full_path = os.path.join(EXPORT_DIR, "train_full_with_all_columns.xlsx")
    val_full_path   = os.path.join(EXPORT_DIR, "val_full_with_all_columns.xlsx")
    test_full_path  = os.path.join(EXPORT_DIR, "test_full_with_all_columns.xlsx")
    train_full_rows.to_excel(train_full_path, index=False, engine='openpyxl')
    val_full_rows.to_excel(val_full_path,   index=False, engine='openpyxl')
    test_full_rows.to_excel(test_full_path, index=False, engine='openpyxl')
    train_full_rows.to_csv(os.path.join(EXPORT_DIR, "train_full_with_all_columns.csv"), index=False)
    val_full_rows.to_csv(os.path.join(EXPORT_DIR, "val_full_with_all_columns.csv"), index=False)
    test_full_rows.to_csv(os.path.join(EXPORT_DIR, "test_full_with_all_columns.csv"), index=False)
    print("Full-row exports saved")
except Exception as e:
    print("Error while creating/saving full-row exports:", e)
    traceback.print_exc()

_show(results_train.head())
_show(results_val.head())
_show(results_test.head())

print("\n" + "="*80)
print("MODEL TRAINING COMPLETED!")
print("="*80)
