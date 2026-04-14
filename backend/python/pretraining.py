"""
Hill function pre-training module.

Generates synthetic data based on the Hill equation:
  y = x^n / (K^n + x^n)

Trains a small dense network on this synthetic data and saves the weights
so they can be used to warm-start the main training pipeline.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow imports from this directory
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from utils import log_status, log_progress


def generate_hill_data(n_samples=cfg.HILL_N_SAMPLES, random_seed=cfg.RANDOM_SEED):
    """
    Generate synthetic Hill-function dataset with multiple n and K values.
    Returns X (n_samples, 3) and y (n_samples, 1).
    Features: [x, n, K]
    """
    np.random.seed(random_seed)
    X_list, y_list = [], []

    samples_per_combo = n_samples // (len(cfg.HILL_N_VALUES) * 4)

    for n_hill in cfg.HILL_N_VALUES:
        for _ in range(4):
            K = np.random.uniform(*cfg.HILL_K_RANGE)
            x = np.random.uniform(*cfg.HILL_X_RANGE, size=samples_per_combo)
            y = (x ** n_hill) / (K ** n_hill + x ** n_hill)
            noise = np.random.normal(0, cfg.HILL_NOISE_STD, size=samples_per_combo)
            y = np.clip(y + noise, 0, 1)

            X_chunk = np.column_stack([x, np.full(samples_per_combo, n_hill), np.full(samples_per_combo, K)])
            X_list.append(X_chunk)
            y_list.append(y)

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.float32).reshape(-1, 1)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def build_pretrain_model(input_dim, n_units=cfg.HILL_PRETRAIN_UNITS, n_layers=cfg.HILL_PRETRAIN_LAYERS):
    """Build a dense neural network for pre-training."""
    tf.random.set_seed(cfg.RANDOM_SEED)
    model = keras.Sequential(name="hill_pretrain")
    model.add(keras.layers.Input(shape=(input_dim,)))

    for i in range(n_layers):
        model.add(keras.layers.Dense(n_units, activation='relu', name=f'dense_{i}'))
        model.add(keras.layers.BatchNormalization(name=f'bn_{i}'))

    model.add(keras.layers.Dense(1, activation='sigmoid', name='output'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def pretrain(output_dir, n_units=cfg.HILL_PRETRAIN_UNITS, n_layers=cfg.HILL_PRETRAIN_LAYERS):
    """
    Run Hill function pre-training and save the weights.
    Returns the path to saved weights file.
    """
    log_status("Starting Hill function pre-training...")

    X, y = generate_hill_data()
    log_status(f"Generated {len(X)} synthetic Hill-function samples")

    # Normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    model = build_pretrain_model(input_dim=X_scaled.shape[1], n_units=n_units, n_layers=n_layers)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        X_scaled, y_scaled,
        epochs=cfg.HILL_PRETRAIN_EPOCHS,
        batch_size=64,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
    )

    final_loss = float(history.history['val_loss'][-1])
    log_status(f"Pre-training complete. Final val_loss={final_loss:.4f}")
    log_progress(5, metrics={'pretrain_val_loss': final_loss})

    weights_path = os.path.join(output_dir, cfg.PRETRAIN_WEIGHTS_FILENAME)
    model.save_weights(weights_path)
    log_status(f"Pre-trained weights saved to {weights_path}")

    return weights_path, model.get_weights(), n_units, n_layers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hill function pre-training')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--units', type=int, default=cfg.HILL_PRETRAIN_UNITS)
    parser.add_argument('--layers', type=int, default=cfg.HILL_PRETRAIN_LAYERS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path, _, _, _ = pretrain(args.output_dir, args.units, args.layers)
    print(f"Weights saved to: {weights_path}")
