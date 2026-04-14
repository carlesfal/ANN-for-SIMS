"""
Shared utilities for the ANN-SIMS ML pipeline.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(filepath, target_column, feature_columns=None):
    """
    Load CSV or Excel file and return features X and target y as numpy arrays.
    Also returns column names.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    df = df.dropna()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    if feature_columns:
        if isinstance(feature_columns, str):
            feature_columns = [c.strip() for c in feature_columns.split(',') if c.strip()]
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
    else:
        feature_columns = [c for c in df.columns if c != target_column]

    X = df[feature_columns].values.astype(np.float32)
    y = df[target_column].values.astype(np.float32).reshape(-1, 1)

    return X, y, feature_columns


def create_scalers():
    """Create MinMaxScaler instances for X and y."""
    return MinMaxScaler(), MinMaxScaler()


def fit_scale(scaler_X, scaler_y, X_train, y_train):
    """Fit scalers and return scaled data."""
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_y.fit_transform(y_train)
    return X_scaled, y_scaled


def transform_scale(scaler_X, scaler_y, X, y=None):
    """Transform data using fitted scalers."""
    X_scaled = scaler_X.transform(X)
    if y is not None:
        y_scaled = scaler_y.transform(y)
        return X_scaled, y_scaled
    return X_scaled


def inverse_scale_y(scaler_y, y_scaled):
    """Inverse transform predictions."""
    return scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def compute_metrics(y_true, y_pred):
    """Compute RMSE, MAE, R2, MAPE for a fold or overall."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE - avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float('nan')

    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


def average_metrics(metrics_list):
    """Average a list of metric dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    avg = {}
    std_dict = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m.get(k, float('nan')))]
        avg[k] = float(np.mean(vals)) if vals else float('nan')
        std_dict[k + '_std'] = float(np.std(vals)) if vals else float('nan')
    avg.update(std_dict)
    return avg


def save_scalers(scaler_X, scaler_y, output_dir):
    """Save fitted scalers to output_dir."""
    from config import SCALER_X_FILENAME, SCALER_Y_FILENAME
    joblib.dump(scaler_X, os.path.join(output_dir, SCALER_X_FILENAME))
    joblib.dump(scaler_y, os.path.join(output_dir, SCALER_Y_FILENAME))


def load_scalers(model_dir):
    """Load fitted scalers from model_dir."""
    from config import SCALER_X_FILENAME, SCALER_Y_FILENAME
    scaler_X = joblib.load(os.path.join(model_dir, SCALER_X_FILENAME))
    scaler_y = joblib.load(os.path.join(model_dir, SCALER_Y_FILENAME))
    return scaler_X, scaler_y


def log_progress(progress, fold=None, epoch=None, metrics=None):
    """Print structured progress for the Node.js controller to parse."""
    data = {'progress': progress}
    if fold is not None:
        data['fold'] = fold
    if epoch is not None:
        data['epoch'] = epoch
    if metrics is not None:
        data['metrics'] = metrics
    print(f"PROGRESS:{json.dumps(data)}", flush=True)


def log_status(message):
    """Print a status message."""
    print(f"STATUS: {message}", flush=True)


def save_results(results, output_dir):
    """Save results dict as JSON."""
    from config import RESULTS_FILENAME
    path = os.path.join(output_dir, RESULTS_FILENAME)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_serializer)
    return path


def _json_serializer(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")
