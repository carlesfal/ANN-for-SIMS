"""Model building, hyperparameter tuning, cross-validation, and training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import keras_tuner as kt
except ImportError:
    try:
        import kerastuner as kt  # type: ignore[import-untyped]
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keras-tuner"])
        import keras_tuner as kt  # type: ignore[import-untyped]

from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def make_model_builder(n_features: int):
    """Return a Keras-Tuner–compatible model-builder function.

    The returned callable accepts an ``hp`` object and produces a compiled
    ``keras.Sequential`` model.  The *n_features* argument is captured via
    closure so the builder does not depend on global state.
    """

    def build_model(hp: Any) -> keras.Model:
        model = keras.Sequential()
        model.add(layers.Input(shape=(n_features,)))

        n_layers = hp.Int("num_layers", 2, 6, step=1)
        l2_val = hp.Choice("l2_reg", [1e-4, 1e-3, 1e-2])

        for i in range(n_layers):
            units = hp.Int(f"units_{i}", 64, 512, step=64)
            model.add(
                layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_val),
                )
            )
            model.add(layers.Dropout(hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)))

        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", [1e-4, 5e-4, 1e-3, 5e-3])
            ),
            loss="mse",
            metrics=["mae"],
        )
        return model

    return build_model


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def run_tuner(
    build_fn,
    X_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    X_val: NDArray[np.floating],
    y_val: NDArray[np.floating],
    cfg: PipelineConfig,
) -> tuple[Any, Any]:
    """Run Keras-Tuner ``RandomSearch`` and return ``(tuner, best_hp)``."""
    tuner = kt.RandomSearch(
        build_fn,
        objective="val_loss",
        max_trials=cfg.tuner_trials,
        executions_per_trial=1,
        directory="tuner_results",
        project_name=f"ann_{cfg.train_percent}_{cfg.val_percent}_{cfg.test_percent}",
    )
    print(f"Starting hyperparameter search ({cfg.tuner_trials} trials)…")
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.tuner_epochs,
        batch_size=32,
        verbose=1,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters:")
    for k, v in best_hp.values.items():
        print(f"  {k}: {v}")
    return tuner, best_hp


# ---------------------------------------------------------------------------
# Weight transfer
# ---------------------------------------------------------------------------

def try_weight_transfer(
    build_fn,
    best_hp: Any,
    warmup_model: Optional[keras.Model],
) -> Optional[keras.Model]:
    """Attempt to transfer weights from *warmup_model* into a fresh model.

    Returns the weight-initialised model on success, or ``None``.
    """
    print("\n" + "=" * 70)
    print("WEIGHT TRANSFER: Checking for pre-trained model…")
    print("=" * 70)

    if warmup_model is None:
        print("No pre-trained model found.")
        print("=" * 70 + "\n")
        return None

    print("Found pre-trained warmup_model!")
    try:
        model = build_fn(best_hp)
        n_transferred = 0
        for layer_final, layer_warmup in zip(model.layers, warmup_model.layers):
            if isinstance(layer_final, layers.InputLayer) or isinstance(
                layer_warmup, layers.InputLayer
            ):
                continue
            if not layer_final.weights or not layer_warmup.weights:
                continue
            try:
                if layer_final.name.split("_")[0] == layer_warmup.name.split("_")[0]:
                    layer_final.set_weights(layer_warmup.get_weights())
                    n_transferred += 1
                    print(f"  {layer_warmup.name} -> {layer_final.name}")
            except Exception as exc:
                print(f"  Skipping {layer_warmup.name}: {exc}")

        if n_transferred > 0:
            print(f"Transferred {n_transferred} layer(s)!")
            print("=" * 70 + "\n")
            return model
        print("No compatible weights (architecture mismatch).")
    except Exception as exc:
        print(f"Weight transfer failed: {exc}")

    print("=" * 70 + "\n")
    return None


# ---------------------------------------------------------------------------
# Cross-validation (strict no-leakage)
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Aggregated cross-validation metrics and OOF predictions."""

    r2_scores: list[float]
    rmse_scores: list[float]
    mae_scores: list[float]
    y_oof_pred: NDArray[np.floating]
    y_oof_true: NDArray[np.floating]
    pred_r2: float


def run_cv(
    build_fn,
    best_hp: Any,
    X_train_orig: NDArray[np.floating],
    y_train_orig: NDArray[np.floating],
    cfg: PipelineConfig,
    pretrained_model: Optional[keras.Model] = None,
) -> CVResult:
    """K-fold CV on the TRAIN split with per-fold fitted scalers (no leakage)."""
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)

    r2_list: list[float] = []
    rmse_list: list[float] = []
    mae_list: list[float] = []
    oof_pred = np.full(len(y_train_orig), np.nan)
    oof_true = np.full(len(y_train_orig), np.nan)

    print(f"\n{cfg.k_folds}-fold CV on TRAIN with fold-fitted scalers (no leakage)…")

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
        if pretrained_model is not None:
            try:
                model_fold.set_weights(pretrained_model.get_weights())
            except Exception:
                pass

        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        model_fold.fit(
            X_tr_s, y_tr_s,
            validation_data=(X_va_s, y_va_s),
            epochs=cfg.cv_epochs,
            batch_size=32,
            callbacks=[es],
            verbose=1,
        )

        va_pred_inv = fold_sy.inverse_transform(
            model_fold.predict(X_va_s, verbose=0)
        ).reshape(-1)
        va_true_inv = y_va.reshape(-1)

        oof_pred[va_idx] = va_pred_inv
        oof_true[va_idx] = va_true_inv

        r2 = r2_score(va_true_inv, va_pred_inv)
        rmse = float(np.sqrt(mean_squared_error(va_true_inv, va_pred_inv)))
        mae = float(np.mean(np.abs(va_true_inv - va_pred_inv)))
        r2_list.append(r2)
        rmse_list.append(rmse)
        mae_list.append(mae)
        print(f"Fold {fold}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    if np.isnan(oof_pred).any():
        raise RuntimeError("OOF predictions contain NaNs — check fold logic.")

    ss_res = float(np.sum((oof_true - oof_pred) ** 2))
    ss_tot = float(np.sum((oof_true - np.mean(oof_true)) ** 2))
    pred_r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    print(f"\nCV (mean +/- std):  R²={np.mean(r2_list):.4f}+/-{np.std(r2_list):.4f}  "
          f"RMSE={np.mean(rmse_list):.4f}+/-{np.std(rmse_list):.4f}  "
          f"MAE={np.mean(mae_list):.4f}+/-{np.std(mae_list):.4f}")
    print(f"Predicted R² (Q², OOF/PRESS): {pred_r2:.4f}")

    return CVResult(
        r2_scores=r2_list,
        rmse_scores=rmse_list,
        mae_scores=mae_list,
        y_oof_pred=oof_pred,
        y_oof_true=oof_true,
        pred_r2=pred_r2,
    )


# ---------------------------------------------------------------------------
# Final training
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Outputs of the final model training step."""

    model: keras.Model
    history: keras.callbacks.History
    best_epoch: int


def train_final_model(
    build_fn,
    best_hp: Any,
    X_train: NDArray[np.floating],
    y_train: NDArray[np.floating],
    X_val: NDArray[np.floating],
    y_val: NDArray[np.floating],
    cfg: PipelineConfig,
) -> TrainResult:
    """Train on TRAIN, validate on VAL (test untouched)."""
    print("\nFinal training: fit on TRAIN, validate on VAL.")

    os.makedirs(cfg.export_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.export_dir, "best_model.keras")

    mc = callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    tf.keras.backend.clear_session()
    model = build_fn(best_hp)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.final_epochs,
        batch_size=32,
        callbacks=[es, mc],
        verbose=1,
    )

    if os.path.exists(ckpt_path):
        model = keras.models.load_model(ckpt_path, compile=False)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    print(f"Best epoch (by VAL loss): {best_epoch}")
    return TrainResult(model=model, history=history, best_epoch=best_epoch)


# ---------------------------------------------------------------------------
# Optional retrain on train+val
# ---------------------------------------------------------------------------

@dataclass
class RetrainResult:
    """Artefacts produced by the optional retrain on train+val."""

    model: keras.Model
    scaler_X: StandardScaler
    scaler_y: StandardScaler


def retrain_on_trainval(
    build_fn,
    best_hp: Any,
    X_train_orig: NDArray[np.floating],
    X_val_orig: NDArray[np.floating],
    y_train_orig: NDArray[np.floating],
    y_val_orig: NDArray[np.floating],
    X_test_orig: NDArray[np.floating],
    y_test_orig: NDArray[np.floating],
    best_epoch: int,
) -> RetrainResult:
    """Retrain on TRAIN+VAL for *best_epoch* epochs (scalers refit)."""
    print("\nOptional retrain: refit scalers on TRAIN+VAL, retrain, evaluate on TEST.")

    X_tv = np.vstack([X_train_orig, X_val_orig])
    y_tv = np.vstack([y_train_orig, y_val_orig])

    sx = StandardScaler().fit(X_tv)
    sy = StandardScaler().fit(y_tv)

    tf.keras.backend.clear_session()
    model = build_fn(best_hp)
    model.fit(
        sx.transform(X_tv), sy.transform(y_tv),
        epochs=best_epoch,
        batch_size=32,
        verbose=1,
    )

    y_pred = sy.inverse_transform(model.predict(sx.transform(X_test_orig), verbose=0)).reshape(-1)
    y_true = y_test_orig.reshape(-1)
    print(
        f"TEST (retrained): R²={r2_score(y_true, y_pred):.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f}  "
        f"MAE={np.mean(np.abs(y_true - y_pred)):.4f}"
    )
    return RetrainResult(model=model, scaler_X=sx, scaler_y=sy)
