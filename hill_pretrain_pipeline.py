"""
Hill Pre-Training + Hyperparameter Tuning Pipeline
===================================================

Three-phase pipeline for training neural networks on SIMS data:
  Phase 1: Hyperparameter search (Keras Tuner) with the **same variable
           architecture** used by the downstream training cell (2-6 layers,
           64-512 units) so pre-trained weights are directly compatible.
  Phase 2: Pre-train the best architecture on synthetic Hill-function data
           to learn sigmoidal priors.
  Phase 3: Expose ``warmup_model`` in the caller's globals so the downstream
           training cell can pick it up for weight transfer.

Usage -- Colab / Jupyter
------------------------
    main()                           # defaults
    cfg = HillPipelineConfig(pretrain_epochs=80, tuner_trials=20)
    main(cfg=cfg)                    # custom

Usage -- CLI
------------
    python hill_pretrain_pipeline.py                    # defaults
    python hill_pretrain_pipeline.py --trials 20        # more tuner trials
    python hill_pretrain_pipeline.py --mixed-precision  # enable FP16 training
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

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
class HillPipelineConfig:
    """Central configuration for the Hill pre-training pipeline.

    The tuner HP search-space mirrors the downstream training cell's
    ``make_model_builder`` so that layer counts, widths, and auto-generated
    Keras layer names are compatible for weight transfer.
    """

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

    # Pre-training (Phase 2)
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 32
    pretrain_patience: int = 5
    pretrain_lr: float = 1e-3

    # Tuner (Phase 1)
    tuner_trials: int = 10
    tuner_epochs: int = 50
    tuner_batch_size: int = 32
    tuner_dir: str = "tuner_results"

    # Data split ratios (used for tuner project naming)
    train_pct: int = 70
    val_pct: int = 15
    test_pct: int = 15

    # Performance
    mixed_precision: bool = False

    # Tuner search-space -- matches downstream cell's make_model_builder
    num_layers_min: int = 2
    num_layers_max: int = 6
    units_min: int = 64
    units_max: int = 512
    units_step: int = 64
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    dropout_step: float = 0.1
    l2_choices: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    lr_choices: List[float] = field(
        default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3]
    )


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
    cfg: HillPipelineConfig,
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
    y = cfg.hill_v_max * (x1 ** cfg.hill_n) / (
        cfg.hill_k ** cfg.hill_n + x1 ** cfg.hill_n
    )
    y += rng.normal(0, 0.05 * cfg.hill_v_max, cfg.n_synthetic).astype(
        np.float32
    )

    span = cfg.hill_x_max - cfg.hill_x_min
    for i in range(1, min(cfg.n_inputs, 3)):
        y += 0.05 * cfg.hill_v_max * (X[:, i] - cfg.hill_x_min) / span

    return X, y.reshape(-1, 1).astype(np.float32)


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    cfg: HillPipelineConfig,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    StandardScaler,
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
# Model builder (variable architecture -- mirrors downstream cell)
# ---------------------------------------------------------------------------


def make_model_builder(
    n_feat: int,
    cfg: HillPipelineConfig,
) -> Callable:
    """Return a Keras-Tuner ``build_model(hp)`` closure whose HP search-space
    is **identical** to the downstream training cell's ``make_model_builder``:

    * ``num_layers``: Int  2-6
    * ``units_i``:    Int  64-512  (step 64)
    * ``l2_reg``:     Choice [1e-4, 1e-3, 1e-2]
    * ``dropout_i``:  Float 0.0-0.5 (step 0.1)
    * ``lr``:         Choice [1e-4, 5e-4, 1e-3, 5e-3]

    Keras auto-generates layer names (``dense``, ``dense_1``, ...) which
    match the downstream cell's naming, enabling name-prefix weight transfer.
    """

    def build_model(hp):  # type: ignore[no-untyped-def]
        model = keras.Sequential()
        model.add(layers.Input(shape=(n_feat,)))

        n_layers = hp.Int(
            "num_layers",
            cfg.num_layers_min,
            cfg.num_layers_max,
            step=1,
        )
        l2_val = hp.Choice("l2_reg", cfg.l2_choices)

        for i in range(n_layers):
            units = hp.Int(
                f"units_{i}",
                cfg.units_min,
                cfg.units_max,
                step=cfg.units_step,
            )
            model.add(
                layers.Dense(
                    units,
                    activation="relu",
                    kernel_regularizer=regularizers.l2(l2_val),
                )
            )
            model.add(
                layers.Dropout(
                    hp.Float(
                        f"dropout_{i}",
                        cfg.dropout_min,
                        cfg.dropout_max,
                        step=cfg.dropout_step,
                    ),
                )
            )

        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("lr", cfg.lr_choices),
            ),
            loss="mse",
            metrics=["mae"],
        )
        return model

    return build_model


# ---------------------------------------------------------------------------
# Phase 1 -- Hyperparameter tuning (find best architecture on Hill data)
# ---------------------------------------------------------------------------


def phase1_tune(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: HillPipelineConfig,
) -> "keras_tuner.HyperParameters":
    """Run Keras Tuner RandomSearch and return best hyperparameters."""
    import keras_tuner as kt

    logger.info("=" * 60)
    logger.info(
        "PHASE 1: Hyperparameter Tuning (%d trials)", cfg.tuner_trials
    )
    logger.info("=" * 60)

    tf.keras.backend.clear_session()

    n_features = X_train.shape[1]
    build_fn = make_model_builder(n_features, cfg)

    project = f"ann_{cfg.train_pct}_{cfg.val_pct}_{cfg.test_pct}"

    tuner = kt.RandomSearch(
        build_fn,
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
# Phase 2 -- Pre-train the best architecture on Hill data
# ---------------------------------------------------------------------------


def phase2_pretrain(
    best_hp: "keras_tuner.HyperParameters",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: HillPipelineConfig,
) -> keras.Sequential:
    """Build the model from *best_hp* and pre-train on synthetic Hill data."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Hill Pre-Training (best architecture)")
    logger.info("=" * 60)

    tf.keras.backend.clear_session()

    build_fn = make_model_builder(cfg.n_inputs, cfg)
    model = build_fn(best_hp)

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
        "Pre-training done -- train_loss=%.6f  val_loss=%.6f",
        history.history["loss"][-1],
        history.history["val_loss"][-1],
    )
    return model


# ---------------------------------------------------------------------------
# Globals injection (for Colab / Jupyter)
# ---------------------------------------------------------------------------


def _inject_into_caller_globals(**variables: object) -> None:
    """Push *variables* into the caller's (notebook cell) global namespace.

    This makes ``warmup_model`` visible to a subsequent Colab cell that
    checks ``if "warmup_model" in globals():``.
    """
    frame = inspect.currentframe()
    try:
        caller_globals = frame.f_back.f_back.f_globals  # type: ignore[union-attr]
        caller_globals.update(variables)
    finally:
        del frame


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
        return shell.__class__.__name__ in (
            "ZMQInteractiveShell",
            "Shell",
        )
    except ImportError:
        return False


def parse_args(
    argv: Optional[List[str]] = None,
) -> Tuple[HillPipelineConfig, str]:
    if _running_in_notebook():
        return HillPipelineConfig(), "saved_models"

    parser = argparse.ArgumentParser(
        description="Hill tune -> pre-train -> expose pipeline"
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

    return HillPipelineConfig(
        random_seed=args.seed,
        n_synthetic=args.n_synthetic,
        pretrain_epochs=args.pretrain_epochs,
        tuner_trials=args.trials,
        tuner_epochs=args.tuner_epochs,
        mixed_precision=args.mixed_precision,
        tuner_dir=args.tuner_dir,
    ), args.save_dir


def main(
    argv: Optional[List[str]] = None,
    *,
    cfg: Optional[HillPipelineConfig] = None,
) -> keras.Sequential:
    """Run the full Hill pre-training pipeline.

    Parameters
    ----------
    argv : list[str] | None
        CLI arguments (ignored in notebook mode).
    cfg : HillPipelineConfig | None
        If provided, skip CLI parsing and use this config directly.
        Useful for calling ``main(cfg=HillPipelineConfig(...))`` in a
        notebook.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if cfg is None:
        cfg, save_dir = parse_args(argv)
    else:
        save_dir = "saved_models"

    # -- Reproducibility & GPU setup --
    set_global_seeds(cfg.random_seed)
    configure_gpu()

    if cfg.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed-precision (FP16) enabled")

    # -- Data --
    logger.info(
        "Generating synthetic Hill data (%d samples)...", cfg.n_synthetic
    )
    X_raw, y_raw = generate_hill_data(cfg)
    logger.info("y range: [%.2f, %.2f]", y_raw.min(), y_raw.max())

    X_train, X_val, y_train, y_val, scaler_X, scaler_y = prepare_data(
        X_raw, y_raw, cfg
    )
    logger.info("Train: %s   Val: %s", X_train.shape, X_val.shape)

    # -- Phase 1: Tune (find best variable architecture) --
    best_hp = phase1_tune(X_train, y_train, X_val, y_val, cfg)

    # -- Phase 2: Pre-train the best architecture on Hill data --
    warmup_model = phase2_pretrain(
        best_hp, X_train, y_train, X_val, y_val, cfg
    )

    # -- Persist artefacts --
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    warmup_model.save(save_path / "warmup_model.keras")
    logger.info("Model saved to %s", save_path / "warmup_model.keras")

    import joblib

    joblib.dump(scaler_X, save_path / "scaler_X.joblib")
    joblib.dump(scaler_y, save_path / "scaler_y.joblib")
    logger.info("Scalers saved to %s", save_path)

    # -- Phase 3: Expose warmup_model to downstream notebook cell --
    logger.info("=" * 60)
    logger.info("PHASE 3: Exposing warmup_model to globals")
    logger.info("=" * 60)
    _inject_into_caller_globals(
        warmup_model=warmup_model,
        _model_with_pretrain=warmup_model,
    )
    logger.info("warmup_model injected into caller globals")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return warmup_model


if __name__ == "__main__":
    main()
