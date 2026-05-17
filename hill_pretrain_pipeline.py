"""
Hill Pre-Training + Hyperparameter Tuning + Weight Transfer Pipeline
====================================================================

Three-phase pipeline for training neural networks on SIMS data:
  Phase 1: Pre-train on synthetic Hill-function data to learn sigmoidal priors.
  Phase 2: Hyperparameter search (Keras Tuner) over regularisation & learning rate.
  Phase 3: Transfer pre-trained weights into the best-tuned architecture.

Usage
-----
    python hill_pretrain_pipeline.py                   # defaults
    python hill_pretrain_pipeline.py --trials 20       # more tuner trials
    python hill_pretrain_pipeline.py --mixed-precision  # enable FP16 training
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import keras_tuner

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import callbacks, layers, regularizers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Central, immutable configuration for the full pipeline."""

    # Hill-function parameters
    n_inputs: int = 10
    n_synthetic: int = 500
    hill_v_max: float = 200.0
    hill_k: float = 0.10
    hill_n: float = 1.20
    hill_x_min: float = 0.1
    hill_x_max: float = 100.0

    # Reproducibility
    random_seed: int = 42

    # Pre-training (Phase 1)
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 32
    pretrain_patience: int = 5
    pretrain_lr: float = 1e-3

    # Tuner (Phase 2)
    tuner_trials: int = 10
    tuner_epochs: int = 50
    tuner_batch_size: int = 32
    tuner_dir: str = "tuner_results"

    # Data split ratios (used for naming only; actual split is 80/20)
    train_pct: int = 70
    val_pct: int = 15
    test_pct: int = 15

    # Performance
    mixed_precision: bool = False

    # Tuner search-space
    l2_choices: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    lr_choices: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3])
    dropout_max_hidden: float = 0.3
    dropout_max_output: float = 0.2
    dropout_step: float = 0.05

    # Architecture (layer widths, top → bottom)
    layer_units: Tuple[int, ...] = (128, 64, 32)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_gpu() -> None:
    """Allow GPU memory growth so TF doesn't pre-allocate all VRAM."""
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_hill_data(
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce synthetic (X, y) pairs governed by a Hill equation on the first
    feature, with small linear contributions from features 2 and 3.

    Returns float32 arrays: X (n_synthetic, n_inputs), y (n_synthetic, 1).
    """
    rng = np.random.default_rng(cfg.random_seed)

    X = rng.uniform(
        cfg.hill_x_min, cfg.hill_x_max, (cfg.n_synthetic, cfg.n_inputs)
    ).astype(np.float32)

    x1 = X[:, 0]
    y = cfg.hill_v_max * (x1 ** cfg.hill_n) / (cfg.hill_k ** cfg.hill_n + x1 ** cfg.hill_n)
    y += rng.normal(0, 0.05 * cfg.hill_v_max, cfg.n_synthetic).astype(np.float32)

    span = cfg.hill_x_max - cfg.hill_x_min
    for i in range(1, min(cfg.n_inputs, 3)):
        y += 0.05 * cfg.hill_v_max * (X[:, i] - cfg.hill_x_min) / span

    return X, y.reshape(-1, 1).astype(np.float32)


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    StandardScaler, StandardScaler,
]:
    """Scale and split into train / validation sets (80 / 20)."""
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=cfg.random_seed
    )
    return X_train, X_val, y_train, y_val, scaler_X, scaler_y


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_warmup_model(cfg: PipelineConfig) -> keras.Sequential:
    """
    Deterministic 128 → 64 → 32 → 1 architecture used for Hill pre-training.
    """
    model = keras.Sequential(name="warmup_model")
    model.add(layers.Input(shape=(cfg.n_inputs,)))

    for i, units in enumerate(cfg.layer_units):
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-3),
                name=f"dense_{i}",
            )
        )
        drop_rate = 0.1 if i == len(cfg.layer_units) - 1 else 0.2
        model.add(layers.Dropout(drop_rate, name=f"dropout_{i}"))

    model.add(layers.Dense(1, activation="linear", name="output"))
    return model


def build_tunable_model(
    hp: "keras_tuner.HyperParameters",
    n_features: int,
    cfg: PipelineConfig,
) -> keras.Sequential:
    """
    Same fixed architecture as the warmup model, but with tunable
    regularisation, dropout, and learning-rate hyperparameters.
    """
    l2_val = hp.Choice("l2_reg", cfg.l2_choices)
    lr = hp.Choice("lr", cfg.lr_choices)

    model = keras.Sequential(name="tuned_model")
    model.add(layers.Input(shape=(n_features,)))

    for i, units in enumerate(cfg.layer_units):
        model.add(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_val),
                name=f"dense_{i}",
            )
        )
        max_drop = cfg.dropout_max_output if i == len(cfg.layer_units) - 1 else cfg.dropout_max_hidden
        model.add(
            layers.Dropout(
                hp.Float(f"dropout_{i}", 0.0, max_drop, step=cfg.dropout_step),
                name=f"dropout_{i}",
            )
        )

    model.add(layers.Dense(1, activation="linear", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ---------------------------------------------------------------------------
# Phase 1 – Hill pre-training
# ---------------------------------------------------------------------------

def phase1_pretrain(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: PipelineConfig,
) -> keras.Sequential:
    """Pre-train the warmup model on synthetic Hill data."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Hill Pre-Training")
    logger.info("=" * 60)

    tf.keras.backend.clear_session()

    model = build_warmup_model(cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.pretrain_lr),
        loss="mse",
        metrics=["mae"],
    )
    logger.info(
        "Warmup model: %d layers, %s parameters",
        len(model.layers),
        f"{model.count_params():,}",
    )

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.pretrain_patience,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.pretrain_epochs,
        batch_size=cfg.pretrain_batch_size,
        callbacks=cb,
        verbose=1,
    )

    logger.info(
        "Pre-training done  —  train_loss=%.6f  val_loss=%.6f",
        history.history["loss"][-1],
        history.history["val_loss"][-1],
    )
    return model


# ---------------------------------------------------------------------------
# Phase 2 – Hyperparameter tuning
# ---------------------------------------------------------------------------

def phase2_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: PipelineConfig,
) -> "keras_tuner.HyperParameters":
    """Run Keras Tuner RandomSearch and return best hyperparameters."""
    import keras_tuner as kt

    logger.info("=" * 60)
    logger.info("PHASE 2: Hyperparameter Tuning (%d trials)", cfg.tuner_trials)
    logger.info("=" * 60)

    n_features = X_train.shape[1]

    def _build(hp: kt.HyperParameters) -> keras.Sequential:
        return build_tunable_model(hp, n_features, cfg)

    project = f"ann_{cfg.train_pct}_{cfg.val_pct}_{cfg.test_pct}"

    tuner = kt.RandomSearch(
        _build,
        objective="val_loss",
        max_trials=cfg.tuner_trials,
        executions_per_trial=1,
        directory=cfg.tuner_dir,
        project_name=project,
        seed=cfg.random_seed,
        overwrite=True,
    )

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.tuner_epochs,
        batch_size=cfg.tuner_batch_size,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
        ],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    logger.info("Best hyperparameters:")
    for k, v in best_hp.values.items():
        logger.info("  %s: %s", k, v)

    return best_hp


# ---------------------------------------------------------------------------
# Phase 3 – Weight transfer
# ---------------------------------------------------------------------------

def phase3_transfer(
    warmup_model: keras.Sequential,
    best_hp: "keras_tuner.HyperParameters",
    cfg: PipelineConfig,
) -> keras.Sequential:
    """
    Build the tuned model from *best_hp* and transfer compatible weights
    from *warmup_model* layer-by-layer, matching by name.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Weight Transfer")
    logger.info("=" * 60)

    target = build_tunable_model(best_hp, cfg.n_inputs, cfg)
    logger.info(
        "Target model: %s params", f"{target.count_params():,}"
    )

    warmup_map: Dict[str, keras.layers.Layer] = {
        layer.name: layer for layer in warmup_model.layers
    }
    transferred = 0

    for layer in target.layers:
        if not layer.weights:
            continue

        source = warmup_map.get(layer.name)
        if source is None:
            logger.debug("  skip %s (no matching source layer)", layer.name)
            continue

        if len(layer.weights) != len(source.weights):
            logger.warning(
                "  %s: weight-tensor count mismatch (%d vs %d)",
                layer.name,
                len(source.weights),
                len(layer.weights),
            )
            continue

        shapes_match = all(
            ws.shape == wt.shape
            for ws, wt in zip(source.weights, layer.weights)
        )
        if not shapes_match:
            logger.warning(
                "  %s: shape mismatch  src=%s  tgt=%s",
                layer.name,
                [w.shape for w in source.weights],
                [w.shape for w in layer.weights],
            )
            continue

        layer.set_weights(source.get_weights())
        transferred += 1
        logger.info(
            "  transferred  %s  %s", layer.name, source.weights[0].shape
        )

    logger.info("Transferred %d / %d weight-bearing layers", transferred, len(target.layers))

    if transferred == 0:
        logger.warning("No weights transferred — architecture may have diverged")

    return target


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _running_in_notebook() -> bool:
    """Detect Jupyter / Colab kernel environments."""
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]
        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except ImportError:
        return False


def parse_args(argv: Optional[List[str]] = None) -> "tuple[PipelineConfig, str]":
    if _running_in_notebook():
        return PipelineConfig(), "saved_models"

    parser = argparse.ArgumentParser(
        description="Hill pre-train → tune → transfer pipeline"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-synthetic", type=int, default=500)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--tuner-epochs", type=int, default=50)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--save-dir", type=str, default="saved_models")
    parser.add_argument("--tuner-dir", type=str, default="tuner_results")
    args = parser.parse_args(argv)

    return PipelineConfig(
        random_seed=args.seed,
        n_synthetic=args.n_synthetic,
        pretrain_epochs=args.pretrain_epochs,
        tuner_trials=args.trials,
        tuner_epochs=args.tuner_epochs,
        mixed_precision=args.mixed_precision,
        tuner_dir=args.tuner_dir,
    ), args.save_dir


def main(argv: Optional[List[str]] = None) -> keras.Sequential:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg, save_dir = parse_args(argv)

    # -- Reproducibility & GPU setup --
    set_global_seeds(cfg.random_seed)
    configure_gpu()

    if cfg.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed-precision (FP16) enabled")

    # -- Data --
    logger.info("Generating synthetic Hill data (%d samples)…", cfg.n_synthetic)
    X_raw, y_raw = generate_hill_data(cfg)
    logger.info("y range: [%.2f, %.2f]", y_raw.min(), y_raw.max())

    X_train, X_val, y_train, y_val, scaler_X, scaler_y = prepare_data(
        X_raw, y_raw, cfg
    )
    logger.info("Train: %s   Val: %s", X_train.shape, X_val.shape)

    # -- Phase 1: Pre-train --
    warmup = phase1_pretrain(X_train, y_train, X_val, y_val, cfg)

    # -- Phase 2: Tune --
    best_hp = phase2_tune(X_train, y_train, X_val, y_val, cfg)

    # -- Phase 3: Transfer --
    final_model = phase3_transfer(warmup, best_hp, cfg)

    # -- Persist artefacts --
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    final_model.save(save_path / "model_with_pretrain.keras")
    logger.info("Model saved to %s", save_path / "model_with_pretrain.keras")

    import joblib  # noqa: E402 — deferred import avoids hard dep at module level

    joblib.dump(scaler_X, save_path / "scaler_X.joblib")
    joblib.dump(scaler_y, save_path / "scaler_y.joblib")
    logger.info("Scalers saved to %s", save_path)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return final_model


if __name__ == "__main__":
    main()
