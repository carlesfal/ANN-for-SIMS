"""
Prediction module for the ANN-SIMS system.

Loads a saved model and scalers, runs predictions on new data,
and returns predictions with 95% prediction intervals.

Supports two inference backends:
  1. **Local** — loads the Keras .h5 model directly (default).
  2. **TF Serving** — sends REST requests to a TensorFlow Serving endpoint
     (enabled with ``--use-tf-serving``).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from utils import load_scalers, log_status, inverse_scale_y


# ---------------------------------------------------------------------------
# TF Serving REST client
# ---------------------------------------------------------------------------

def _predict_via_tf_serving(X_scaled, model_name, tf_serving_url=None):
    """
    Call TF Serving REST ``v1/models/<model_name>:predict`` endpoint.
    Returns a numpy array of shape (n_samples, 1).
    """
    import requests  # imported here so local-only usage has no extra dep

    url = (tf_serving_url or cfg.TF_SERVING_URL).rstrip("/")
    endpoint = f"{url}/v1/models/{model_name}:predict"
    payload = {"instances": X_scaled.tolist()}
    resp = requests.post(endpoint, json=payload, timeout=cfg.TF_SERVING_TIMEOUT)
    resp.raise_for_status()
    preds = np.array(resp.json()["predictions"], dtype=np.float32)
    return preds


def load_model_and_scalers(model_dir, skip_model=False):
    """Load the saved Keras model and scalers.

    When *skip_model* is True the Keras model is **not** loaded (useful when
    inference will be delegated to TF Serving).
    """
    model = None
    if not skip_model:
        model_path = os.path.join(model_dir, cfg.MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = tf.keras.models.load_model(model_path)

    scaler_X, scaler_y = load_scalers(model_dir)

    # Load metadata if available
    meta_path = os.path.join(model_dir, 'meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return model, scaler_X, scaler_y, meta


def predict_with_intervals(model, scaler_X, scaler_y, X,
                           n_bootstrap=cfg.BOOTSTRAP_N_ITERATIONS,
                           model_name=None, tf_serving_url=None):
    """
    Predict using the model and compute 95% prediction intervals via
    MC Dropout (if dropout layers present) or bootstrap estimation.

    When *model_name* is provided (and *tf_serving_url* is set or the default
    is reachable), inference is delegated to TensorFlow Serving.  MC Dropout
    is **not** available through TF Serving, so residual-based intervals are
    used in that case.

    Returns arrays: y_pred, y_lower, y_upper
    """
    X_scaled = scaler_X.transform(X)

    use_tf_serving = model_name is not None

    if use_tf_serving:
        # ── TF Serving path ──────────────────────────────────────────────
        y_pred_scaled = _predict_via_tf_serving(X_scaled, model_name, tf_serving_url)
        y_pred = inverse_scale_y(scaler_y, y_pred_scaled)

        # Residual-based PI (MC Dropout not available via REST)
        pred_range = y_pred.max() - y_pred.min() if y_pred.max() != y_pred.min() else 1.0
        uncertainty = pred_range * cfg.FALLBACK_UNCERTAINTY_FACTOR
        y_lower = y_pred - cfg.PI_Z_SCORE * uncertainty
        y_upper = y_pred + cfg.PI_Z_SCORE * uncertainty
    else:
        # ── Local model path ─────────────────────────────────────────────
        # Check if model has dropout layers
        has_dropout = any(isinstance(layer, tf.keras.layers.Dropout) for layer in model.layers)

        if has_dropout:
            # MC Dropout inference
            preds = []
            for _ in range(n_bootstrap):
                # Enable training mode for dropout
                pred = model(X_scaled, training=True).numpy()
                preds.append(inverse_scale_y(scaler_y, pred))
            preds = np.array(preds)  # (n_bootstrap, n_samples)
            y_pred = np.mean(preds, axis=0)
            y_lower = np.percentile(preds, 2.5, axis=0)
            y_upper = np.percentile(preds, 97.5, axis=0)
        else:
            # Point prediction with residual-based PI
            y_pred_scaled = model.predict(X_scaled, verbose=0)
            y_pred = inverse_scale_y(scaler_y, y_pred_scaled)

        # Estimate uncertainty as a fraction of prediction range
            pred_range = y_pred.max() - y_pred.min() if y_pred.max() != y_pred.min() else 1.0
            uncertainty = pred_range * cfg.FALLBACK_UNCERTAINTY_FACTOR
            y_lower = y_pred - cfg.PI_Z_SCORE * uncertainty
            y_upper = y_pred + cfg.PI_Z_SCORE * uncertainty

    return y_pred.flatten(), y_lower.flatten(), y_upper.flatten()


def batch_predict(model, scaler_X, scaler_y, filepath, feature_columns=None, target_column=None,
                  model_name=None, tf_serving_url=None):
    """
    Load a file and run batch predictions.
    Returns a dict with predictions and metadata.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    if feature_columns:
        if isinstance(feature_columns, str):
            feature_columns = [c.strip() for c in feature_columns.split(',') if c.strip()]
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        X = df[feature_columns].values.astype(np.float32)
    else:
        # Exclude target column if present
        cols = [c for c in df.columns if c != target_column] if target_column else list(df.columns)
        feature_columns = cols
        X = df[cols].values.astype(np.float32)

    y_true = None
    if target_column and target_column in df.columns:
        y_true = df[target_column].values.astype(np.float32)

    y_pred, y_lower, y_upper = predict_with_intervals(
        model, scaler_X, scaler_y, X,
        model_name=model_name, tf_serving_url=tf_serving_url,
    )

    rows = []
    for i in range(len(y_pred)):
        row = {col: float(X[i, j]) for j, col in enumerate(feature_columns)}
        row['prediction'] = float(y_pred[i])
        row['lower_95'] = float(y_lower[i])
        row['upper_95'] = float(y_upper[i])
        if y_true is not None:
            row['actual'] = float(y_true[i])
        rows.append(row)

    result = {
        'predictions': rows,
        'n_samples': int(len(y_pred)),
        'feature_columns': feature_columns,
        'summary': {
            'mean_prediction': float(np.mean(y_pred)),
            'std_prediction': float(np.std(y_pred)),
            'min_prediction': float(np.min(y_pred)),
            'max_prediction': float(np.max(y_pred)),
        },
    }

    if y_true is not None:
        from utils import compute_metrics
        metrics = compute_metrics(y_true, y_pred)
        result['metrics'] = metrics

    return result


def single_predict(model, scaler_X, scaler_y, features_dict, feature_columns,
                   model_name=None, tf_serving_url=None):
    """Predict a single sample from a dictionary of feature values."""
    X = np.array([[features_dict[col] for col in feature_columns]], dtype=np.float32)
    y_pred, y_lower, y_upper = predict_with_intervals(
        model, scaler_X, scaler_y, X, n_bootstrap=50,
        model_name=model_name, tf_serving_url=tf_serving_url,
    )

    return {
        'prediction': float(y_pred[0]),
        'lower_95': float(y_lower[0]),
        'upper_95': float(y_upper[0]),
        'features': features_dict,
    }


def main():
    parser = argparse.ArgumentParser(description='ANN-SIMS Prediction')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', help='Directory to save predictions JSON')
    parser.add_argument('--input', help='Path to input CSV/Excel for batch prediction')
    parser.add_argument('--target-column', default='target')
    parser.add_argument('--feature-columns', default='')
    parser.add_argument('--single', action='store_true', help='Single sample prediction mode')
    parser.add_argument('--input-json', help='JSON string of feature values for single prediction')
    parser.add_argument('--use-tf-serving', action='store_true',
                        help='Delegate inference to TensorFlow Serving')
    parser.add_argument('--tf-serving-url', default=None,
                        help='TF Serving REST endpoint (default: from config / env)')
    parser.add_argument('--model-name', default=None,
                        help='Model name registered in TF Serving (defaults to job ID)')
    args = parser.parse_args()

    # Determine TF Serving parameters
    use_serving = args.use_tf_serving
    tf_serving_url = args.tf_serving_url
    model_name = args.model_name
    if use_serving and model_name is None:
        # Derive model name from the model-dir basename (which is the jobId)
        model_name = os.path.basename(os.path.normpath(args.model_dir))

    model, scaler_X, scaler_y, meta = load_model_and_scalers(
        args.model_dir, skip_model=use_serving
    )
    feature_columns = meta.get('feature_columns', None)
    if args.feature_columns:
        feature_columns = [c.strip() for c in args.feature_columns.split(',') if c.strip()]

    if args.single and args.input_json:
        features = json.loads(args.input_json)
        if feature_columns is None:
            feature_columns = list(features.keys())
        result = single_predict(
            model, scaler_X, scaler_y, features, feature_columns,
            model_name=model_name if use_serving else None,
            tf_serving_url=tf_serving_url,
        )
        print(json.dumps(result))

    elif args.input:
        log_status(f"Running batch prediction on {args.input}")
        result = batch_predict(
            model, scaler_X, scaler_y, args.input,
            feature_columns=feature_columns,
            target_column=args.target_column,
            model_name=model_name if use_serving else None,
            tf_serving_url=tf_serving_url,
        )

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, cfg.PREDICTIONS_FILENAME)
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            log_status(f"Predictions saved to {out_path}")

            # Also save CSV
            import csv
            csv_path = os.path.join(args.output_dir, f'results_{os.path.basename(args.output_dir)}.csv')
            if result['predictions']:
                with open(csv_path, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=result['predictions'][0].keys())
                    writer.writeheader()
                    writer.writerows(result['predictions'])
        else:
            print(json.dumps(result))
    else:
        parser.error("Must provide --input for batch prediction or --single with --input-json")


if __name__ == '__main__':
    main()
