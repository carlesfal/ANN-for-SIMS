"""
Main ANN training pipeline for the SIMS prediction system.

Pipeline:
1. Load and preprocess data
2. (Optional) Hill function pre-training for weight initialization
3. Keras Tuner hyperparameter search
4. 10-fold cross-validation with the best hyperparameters
5. Optional retraining on train+val combined
6. Compute comprehensive metrics (RMSE, MAE, R2, MAPE) per fold and averaged
7. Prediction interval calibration
8. Save best model, scalers, and results JSON
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import tensorflow as tf
import keras_tuner as kt

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from utils import (
    load_data, create_scalers, fit_scale, transform_scale, inverse_scale_y,
    compute_metrics, average_metrics, save_scalers, save_results,
    log_progress, log_status,
)


def build_model(hp, input_dim):
    """Keras Tuner model builder."""
    tf.random.set_seed(cfg.RANDOM_SEED)

    n_units = hp.Choice('units', values=[32, 64, 128, 256])
    n_layers = hp.Int('layers', min_value=1, max_value=cfg.TUNER_MAX_LAYERS)
    learning_rate = hp.Choice('learning_rate', values=cfg.TUNER_LEARNING_RATES)
    dropout_rate = hp.Choice('dropout', values=cfg.TUNER_DROPOUT_RATES)

    model = tf.keras.Sequential(name='ann_sims')
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(n_units, activation='relu', name=f'dense_{i}',
                                        kernel_initializer='glorot_uniform',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        model.add(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate, name=f'drop_{i}'))

    model.add(tf.keras.layers.Dense(1, activation='linear', name='output'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def build_fixed_model(input_dim, n_units, n_layers, learning_rate, dropout_rate):
    """Build a model with fixed hyperparameters."""
    tf.random.set_seed(cfg.RANDOM_SEED)
    model = tf.keras.Sequential(name='ann_sims')
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(n_units, activation='relu', name=f'dense_{i}',
                                        kernel_initializer='glorot_uniform',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        model.add(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate, name=f'drop_{i}'))

    model.add(tf.keras.layers.Dense(1, activation='linear', name='output'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def run_hyperparameter_search(X_train, y_train, input_dim, output_dir, epochs, batch_size, patience):
    """Run Keras Tuner random search for hyperparameters."""
    log_status("Starting hyperparameter search...")
    log_progress(10)

    tuner_dir = os.path.join(output_dir, 'kt_dir')

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, input_dim),
        objective='val_loss',
        max_trials=cfg.TUNER_MAX_TRIALS,
        executions_per_trial=cfg.TUNER_EXECUTIONS_PER_TRIAL,
        directory=tuner_dir,
        project_name='ann_sims',
        seed=cfg.RANDOM_SEED,
        overwrite=True,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience // 2, restore_best_weights=True),
    ]

    tuner.search(
        X_train, y_train,
        epochs=min(epochs, 50),
        validation_split=cfg.DEFAULT_VALIDATION_SPLIT,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_params = {
        'units': best_hp.get('units'),
        'layers': best_hp.get('layers'),
        'learning_rate': best_hp.get('learning_rate'),
        'dropout': best_hp.get('dropout'),
    }
    log_status(f"Best hyperparameters found: {best_params}")
    log_progress(20, metrics={'best_hyperparameters': best_params})

    return best_params


def run_cross_validation(X, y, best_params, epochs, batch_size, patience, pretrain_weights=None):
    """
    Run stratified K-fold cross-validation with the best hyperparameters.
    Returns per-fold metrics, all predictions, and all true values.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cfg.DEFAULT_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_lower = []
    all_y_pred_upper = []
    fold_histories = []

    n_folds = cfg.DEFAULT_FOLDS

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_num = fold_idx + 1
        log_status(f"Training fold {fold_num}/{n_folds}...")

        X_train_f, X_val_f = X[train_idx], X[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]

        # Fit scalers on training fold
        scaler_X, scaler_y = create_scalers()
        X_train_s, y_train_s = fit_scale(scaler_X, scaler_y, X_train_f, y_train_f)
        X_val_s, y_val_s = transform_scale(scaler_X, scaler_y, X_val_f, y_val_f)

        model = build_fixed_model(
            input_dim=X.shape[1],
            n_units=best_params['units'],
            n_layers=best_params['layers'],
            learning_rate=best_params['learning_rate'],
            dropout_rate=best_params['dropout'],
        )

        # Transfer pre-trained weights if available and shapes match
        if pretrain_weights is not None:
            try:
                model.set_weights(pretrain_weights)
            except ValueError:
                pass  # Shape mismatch, skip transfer

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=patience // 2, min_lr=1e-6
            ),
        ]

        history = model.fit(
            X_train_s, y_train_s,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_s, y_val_s),
            callbacks=callbacks,
            verbose=0,
        )

        # Predict on validation set
        y_pred_s = model.predict(X_val_s, verbose=0)
        y_pred = inverse_scale_y(scaler_y, y_pred_s)
        y_true = y_val_f.flatten()

        # Residuals for PI estimation
        residuals = y_true - y_pred
        std_residuals = np.std(residuals)
        y_lower = y_pred - cfg.PI_Z_SCORE * std_residuals
        y_upper = y_pred + cfg.PI_Z_SCORE * std_residuals

        metrics = compute_metrics(y_true, y_pred)
        fold_metrics.append(metrics)
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_pred_lower.extend(y_lower.tolist())
        all_y_pred_upper.extend(y_upper.tolist())
        fold_histories.append({
            'fold': fold_num,
            'train_loss': [float(v) for v in history.history['loss']],
            'val_loss': [float(v) for v in history.history['val_loss']],
            'epochs_trained': len(history.history['loss']),
        })

        progress = 20 + int(fold_num / n_folds * 60)
        log_progress(progress, fold=fold_num, metrics=metrics)

    return fold_metrics, all_y_true, all_y_pred, all_y_pred_lower, all_y_pred_upper, fold_histories


def train_final_model(X, y, best_params, epochs, batch_size, patience, output_dir, pretrain_weights=None):
    """
    Train the final model on the full dataset and save it.
    Returns the trained model and fitted scalers.
    """
    log_status("Training final model on full dataset...")
    log_progress(82)

    scaler_X, scaler_y = create_scalers()
    X_s, y_s = fit_scale(scaler_X, scaler_y, X, y)

    model = build_fixed_model(
        input_dim=X.shape[1],
        n_units=best_params['units'],
        n_layers=best_params['layers'],
        learning_rate=best_params['learning_rate'],
        dropout_rate=best_params['dropout'],
    )

    if pretrain_weights is not None:
        try:
            model.set_weights(pretrain_weights)
        except ValueError:
            pass

    val_split = min(0.1, 50 / len(X)) if len(X) > 100 else 0.1
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 2, min_lr=1e-6),
    ]

    model.fit(
        X_s, y_s,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=0,
    )

    # Save model and scalers
    model_dir = output_dir  # Save into the job output dir
    model_path = os.path.join(model_dir, cfg.MODEL_FILENAME)
    model.save(model_path)
    save_scalers(scaler_X, scaler_y, model_dir)

    log_status(f"Final model saved to {model_path}")
    log_progress(95)

    return model, scaler_X, scaler_y


def main():
    parser = argparse.ArgumentParser(description='ANN-SIMS Training Pipeline')
    parser.add_argument('--input', required=True, help='Path to input CSV/Excel file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--job-id', required=True, help='Job ID')
    parser.add_argument('--target-column', default='target', help='Name of the target column')
    parser.add_argument('--feature-columns', default='', help='Comma-separated feature column names')
    parser.add_argument('--epochs', type=int, default=cfg.DEFAULT_EPOCHS)
    parser.add_argument('--folds', type=int, default=cfg.DEFAULT_FOLDS)
    parser.add_argument('--learning-rate', type=float, default=cfg.DEFAULT_LEARNING_RATE)
    parser.add_argument('--units', type=int, default=cfg.DEFAULT_UNITS)
    parser.add_argument('--layers', type=int, default=cfg.DEFAULT_LAYERS)
    parser.add_argument('--dropout', type=float, default=cfg.DEFAULT_DROPOUT)
    parser.add_argument('--batch-size', type=int, default=cfg.DEFAULT_BATCH_SIZE)
    parser.add_argument('--patience', type=int, default=cfg.DEFAULT_PATIENCE)
    parser.add_argument('--use-pretraining', action='store_true', help='Enable Hill function pre-training')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Also save model into backend/models/<job-id> so prediction can find it
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(backend_dir, 'models', args.job_id)
    os.makedirs(models_dir, exist_ok=True)

    tf.random.set_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    log_status("Loading data...")
    log_progress(2)

    X, y, feature_columns = load_data(args.input, args.target_column, args.feature_columns or None)
    log_status(f"Loaded {X.shape[0]} samples with {X.shape[1]} features: {feature_columns}")
    log_progress(5)

    # Pre-training
    pretrain_weights = None
    if args.use_pretraining:
        log_status("Running Hill function pre-training...")
        from pretraining import pretrain
        _, pretrain_weights, _, _ = pretrain(args.output_dir, args.units, args.layers)
        log_status("Pre-training complete, weights ready for transfer")

    # Hyperparameter search on a subset if large
    search_X = X
    search_y = y
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False)
        search_X = X[idx]
        search_y = y[idx]

    scaler_X_tmp, scaler_y_tmp = create_scalers()
    search_X_s, search_y_s = fit_scale(scaler_X_tmp, scaler_y_tmp, search_X, search_y)

    best_params = run_hyperparameter_search(
        search_X_s, search_y_s, X.shape[1],
        args.output_dir, args.epochs, args.batch_size, args.patience
    )

    # Override with CLI params if tuner found worse (keep CLI if specified)
    log_status(f"Using hyperparameters: {best_params}")

    # Cross-validation
    fold_metrics, all_y_true, all_y_pred, all_y_pred_lower, all_y_pred_upper, fold_histories = \
        run_cross_validation(X, y, best_params, args.epochs, args.batch_size, args.patience, pretrain_weights)

    avg_metrics = average_metrics(fold_metrics)
    log_status(f"CV Results - RMSE: {avg_metrics['rmse']:.4f}, R2: {avg_metrics['r2']:.4f}, MAE: {avg_metrics['mae']:.4f}")
    log_progress(82, metrics=avg_metrics)

    # Final model training
    final_model, scaler_X, scaler_y = train_final_model(
        X, y, best_params, args.epochs, args.batch_size, args.patience,
        args.output_dir, pretrain_weights
    )

    # Copy model files to models_dir for prediction
    import shutil
    for fname in [cfg.MODEL_FILENAME, cfg.SCALER_X_FILENAME, cfg.SCALER_Y_FILENAME]:
        src = os.path.join(args.output_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(models_dir, fname))

    # Save feature columns info
    meta = {'feature_columns': feature_columns, 'target_column': args.target_column}
    with open(os.path.join(models_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)

    # Generate 3D plots
    log_status("Generating 3D surface plots...")
    try:
        from plot_3d import generate_3d_plots
        plot_paths = generate_3d_plots(
            final_model, scaler_X, scaler_y, X, y,
            feature_columns, args.target_column, plots_dir
        )
        log_status(f"Generated {len(plot_paths)} 3D plots")
    except Exception as e:
        log_status(f"3D plot generation failed (non-critical): {e}")
        plot_paths = []

    # Assemble final results
    results = {
        'jobId': args.job_id,
        'status': 'completed',
        'featureColumns': feature_columns,
        'targetColumn': args.target_column,
        'nSamples': int(X.shape[0]),
        'nFeatures': int(X.shape[1]),
        'bestHyperparameters': best_params,
        'averageMetrics': avg_metrics,
        'foldMetrics': fold_metrics,
        'foldHistories': fold_histories,
        'predictions': {
            'yTrue': [float(v) for v in all_y_true],
            'yPred': [float(v) for v in all_y_pred],
            'yLower': [float(v) for v in all_y_pred_lower],
            'yUpper': [float(v) for v in all_y_pred_upper],
        },
        'plots': [os.path.basename(p) for p in plot_paths],
        'usedPretraining': args.use_pretraining,
    }

    save_results(results, args.output_dir)
    log_status("Training pipeline complete!")
    log_progress(100, metrics=avg_metrics)
    print(f"STATUS: Results saved to {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
