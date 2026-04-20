"""
ML Pipeline Runner - Wraps the ANN regression pipeline for web API use.
Runs the pipeline in a background thread and reports progress via a shared state dict.
Uses scikit-learn MLPRegressor for lightweight deployment (no TensorFlow dependency).
"""
import os
import sys
import shutil
import traceback
import threading
import uuid
import zipfile
import io
import base64
import datetime as _dt
import random
from typing import Optional

import numpy as np
import pandas as pd


# ---- In-memory job store ----
_jobs: dict = {}


def get_job(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


def list_jobs() -> list:
    return [
        {
            "job_id": jid,
            "status": j["status"],
            "progress": j["progress"],
            "created_at": j.get("created_at", ""),
        }
        for jid, j in _jobs.items()
    ]


def _log(job: dict, msg: str):
    job["logs"].append(msg)
    print(f"[{job['job_id'][:8]}] {msg}", flush=True)


def _set_progress(job: dict, pct: int, stage: str):
    job["progress"] = pct
    job["stage"] = stage
    _log(job, f"[{pct}%] {stage}")


# ---- Pipeline runner ----
def start_pipeline(
    data_path: str,
    config: dict,
    new_data_path: Optional[str] = None,
) -> str:
    """Start pipeline in background thread. Returns job_id."""
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "running",
        "progress": 0,
        "stage": "initializing",
        "logs": [],
        "metrics": {},
        "plots": {},  # name -> base64 PNG
        "plot_files": {},  # name -> file path
        "results_zip": None,
        "predictions_zip": None,
        "error": None,
        "created_at": _dt.datetime.utcnow().isoformat(),
        "config": config,
        "export_dir": None,
    }
    _jobs[job_id] = job

    t = threading.Thread(target=_run_pipeline, args=(job, data_path, config, new_data_path), daemon=True)
    t.start()
    return job_id


def _run_pipeline(job: dict, data_path: str, config: dict, new_data_path: Optional[str]):
    try:
        _run_pipeline_inner(job, data_path, config, new_data_path)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        _log(job, f"FAILED: {e}")


def _run_pipeline_inner(job: dict, data_path: str, config: dict, new_data_path: Optional[str]):
    # ---- Lazy imports (loaded only when training starts) ----
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from itertools import combinations

    job_id = job["job_id"]

    # ---- Config ----
    TRAIN_PERCENT = config.get("train_percent", 60)
    VAL_PERCENT = config.get("val_percent", 20)
    TEST_PERCENT = config.get("test_percent", 20)
    N_LABELS = config.get("n_labels", 7)
    N_INPUTS = config.get("n_inputs", 10)
    TARGET_COL = config.get("target_col", None)
    SEP = config.get("sep", "\t")
    TUNER_TRIALS = config.get("tuner_trials", 10)
    K_FOLDS = config.get("k_folds", 10)
    RANDOM_SEED = config.get("random_seed", 42)
    TUNER_EPOCHS = config.get("tuner_epochs", 200)
    CV_EPOCHS = config.get("cv_epochs", 200)
    FINAL_EPOCHS = config.get("final_epochs", 200)
    DO_OPTIONAL_RETRAIN = config.get("do_optional_retrain", True)
    PI_CALIBRATION = config.get("pi_calibration", "val")
    PI_ALPHA = config.get("pi_alpha", 0.05)

    # Plot settings
    PLOT_DPI = 150
    PLOT_WIDTH_CM = 8.3
    PLOT_WIDTH_IN = PLOT_WIDTH_CM / 2.54
    PLOT_FORMAT = "png"

    EXPORT_DIR = os.path.join("/tmp", f"pipeline_{job_id}")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    job["export_dir"] = EXPORT_DIR

    total_percent = TRAIN_PERCENT + VAL_PERCENT + TEST_PERCENT
    if abs(total_percent - 100) > 0.01:
        raise ValueError(f"Split percentages must sum to 100. Got {total_percent}%")

    _set_progress(job, 2, "Setting seeds and configuring model")

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # ---- Load data ----
    _set_progress(job, 5, "Loading dataset")

    _template_categories = None

    def _detect_template_format(df_raw):
        first_row = [str(v).strip().lower() for v in df_raw.iloc[0]]
        categories = {"label", "input", "output"}
        return all(v in categories for v in first_row if v)

    def load_table(path, sep=SEP):
        nonlocal _template_categories
        _, ext = os.path.splitext(path.lower())
        if ext in [".xlsx", ".xls", ".xlsm"]:
            df_raw = pd.read_excel(path, engine="openpyxl", header=None)
            if len(df_raw) >= 3 and _detect_template_format(df_raw):
                _log(job, "Detected template format (Row 1=categories, Row 2=names, Row 3+=data)")
                _template_categories = [str(v).strip().lower() for v in df_raw.iloc[0]]
                col_names = [str(v).strip() for v in df_raw.iloc[1]]
                df_data = df_raw.iloc[2:].reset_index(drop=True)
                df_data.columns = col_names
                for col in df_data.columns:
                    try:
                        df_data[col] = pd.to_numeric(df_data[col])
                    except (ValueError, TypeError):
                        pass
                return df_data
            else:
                return pd.read_excel(path, engine="openpyxl")
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        for enc in encodings:
            try:
                return pd.read_csv(path, sep=sep, encoding=enc)
            except Exception:
                pass
        for enc in encodings:
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc)
            except Exception:
                pass
        raise RuntimeError(f"Failed to read '{path}'")

    df = load_table(data_path, sep=SEP)
    _log(job, f"Data loaded. Shape: {df.shape}")

    # ---- Column selection ----
    n_cols = df.shape[1]
    if _template_categories is not None:
        label_indices = [i for i, c in enumerate(_template_categories) if c == "label"]
        input_indices = [i for i, c in enumerate(_template_categories) if c == "input"]
        output_indices = [i for i, c in enumerate(_template_categories) if c == "output"]
        labels_df = df.iloc[:, label_indices] if label_indices else pd.DataFrame()
        inputs_df = df.iloc[:, input_indices]
        y_full = df.iloc[:, output_indices[0]].values.reshape(-1, 1)
        N_LABELS = len(label_indices)
        N_INPUTS = len(input_indices)
        _log(job, f"Template categories: {N_LABELS} labels, {N_INPUTS} inputs, {len(output_indices)} output")
    elif TARGET_COL is not None:
        target_idx = TARGET_COL
        labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
        input_cols = list(range(N_LABELS, n_cols))
        input_cols.remove(target_idx)
        inputs_df = df.iloc[:, input_cols]
        y_full = df.iloc[:, target_idx].values.reshape(-1, 1)
    else:
        if N_LABELS + N_INPUTS >= n_cols:
            labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
            inputs_df = df.iloc[:, N_LABELS:-1]
            y_full = df.iloc[:, -1].values.reshape(-1, 1)
        else:
            labels_df = df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()
            inputs_df = df.iloc[:, N_LABELS:N_LABELS + N_INPUTS]
            y_full = df.iloc[:, N_LABELS + N_INPUTS].values.reshape(-1, 1)

    _log(job, f"Labels: {labels_df.shape}, Inputs: {inputs_df.shape}, y: {y_full.shape}")

    X_full = inputs_df.values
    y_full_arr = y_full
    labels_full = labels_df
    row_pos = np.arange(len(df))
    inputs_columns = list(inputs_df.columns)

    # ---- Split ----
    _set_progress(job, 8, "Splitting data")

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

    X_train_df = pd.DataFrame(X_train_orig, columns=inputs_columns).reset_index(drop=True)
    X_val_df = pd.DataFrame(X_val_orig, columns=inputs_columns).reset_index(drop=True)
    X_test_df = pd.DataFrame(X_test_orig, columns=inputs_columns).reset_index(drop=True)

    _log(job, f"Split: train={len(X_train_orig)} val={len(X_val_orig)} test={len(X_test_orig)}")

    # ---- Scaling ----
    scaler_X = StandardScaler().fit(X_train_orig)
    scaler_y = StandardScaler().fit(y_train_orig)

    X_train = scaler_X.transform(X_train_orig)
    X_val = scaler_X.transform(X_val_orig)
    X_test = scaler_X.transform(X_test_orig)
    y_train = scaler_y.transform(y_train_orig)
    y_val = scaler_y.transform(y_val_orig)
    y_test = scaler_y.transform(y_test_orig)

    _N_FEATURES = X_train.shape[1]

    # ---- Model builder (sklearn MLPRegressor, same 128->64->32->1 architecture) ----
    def build_mlp(alpha, lr, max_iter, seed=RANDOM_SEED):
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=seed,
            batch_size=min(32, max(1, len(X_train))),
        )

    # ---- Hyperparameter search (random search over grid) ----
    _set_progress(job, 10, f"Hyperparameter tuning ({TUNER_TRIALS} trials)")

    hp_grid = [
        {'alpha': a, 'lr': lr}
        for a in [1e-4, 1e-3, 1e-2]
        for lr in [1e-4, 5e-4, 1e-3, 5e-3]
    ]
    random.shuffle(hp_grid)
    trials_to_run = hp_grid[:TUNER_TRIALS]

    best_val_loss = float('inf')
    best_hp_values = trials_to_run[0]

    for trial_idx, hp in enumerate(trials_to_run):
        pct = 10 + int(((trial_idx + 1) / len(trials_to_run)) * 20)
        _set_progress(job, pct, f"Tuning trial {trial_idx + 1}/{len(trials_to_run)}")

        try:
            m = build_mlp(alpha=hp['alpha'], lr=hp['lr'], max_iter=TUNER_EPOCHS)
            m.fit(X_train, y_train.ravel())
            y_val_pred_trial = m.predict(X_val).reshape(-1, 1)
            val_loss = float(np.mean((y_val - y_val_pred_trial) ** 2))
            _log(job, f"Trial {trial_idx + 1}: alpha={hp['alpha']}, lr={hp['lr']}, val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_hp_values = hp.copy()
        except Exception as e:
            _log(job, f"Trial {trial_idx + 1} failed: {e}")

    hp_values = best_hp_values
    _log(job, f"Best hyperparameters: {hp_values}")
    job["metrics"]["best_hyperparameters"] = {k: str(v) for k, v in hp_values.items()}

    _set_progress(job, 30, "Hyperparameter tuning complete")

    # ---- K-Fold CV ----
    _set_progress(job, 32, f"Running {K_FOLDS}-Fold CV")

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    r2_scores, rmse_scores, mae_scores = [], [], []
    y_oof_pred_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)
    y_oof_true_inv = np.full(shape=(len(y_train_orig),), fill_value=np.nan, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_orig), start=1):
        pct = 32 + int((fold / K_FOLDS) * 20)
        _set_progress(job, pct, f"CV Fold {fold}/{K_FOLDS}")

        X_tr_orig, X_va_orig = X_train_orig[tr_idx], X_train_orig[va_idx]
        y_tr_orig, y_va_orig = y_train_orig[tr_idx], y_train_orig[va_idx]

        fold_scaler_X = StandardScaler().fit(X_tr_orig)
        fold_scaler_y = StandardScaler().fit(y_tr_orig)

        X_tr = fold_scaler_X.transform(X_tr_orig)
        X_va = fold_scaler_X.transform(X_va_orig)
        y_tr = fold_scaler_y.transform(y_tr_orig)

        model_fold = build_mlp(
            alpha=hp_values['alpha'],
            lr=hp_values['lr'],
            max_iter=CV_EPOCHS,
            seed=RANDOM_SEED + fold,
        )
        model_fold.fit(X_tr, y_tr.ravel())

        y_va_pred_scaled = model_fold.predict(X_va).reshape(-1, 1)
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

        _log(job, f"Fold {fold}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    cv_metrics = {
        "R2_mean": float(np.mean(r2_scores)),
        "R2_std": float(np.std(r2_scores)),
        "RMSE_mean": float(np.mean(rmse_scores)),
        "RMSE_std": float(np.std(rmse_scores)),
        "MAE_mean": float(np.mean(mae_scores)),
        "MAE_std": float(np.std(mae_scores)),
    }
    job["metrics"]["cv"] = cv_metrics
    _log(job, f"CV: R2={cv_metrics['R2_mean']:.4f}+/-{cv_metrics['R2_std']:.4f}")

    ss_res_press = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
    ss_tot_train = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
    pred_R2_train = 1.0 - ss_res_press / ss_tot_train if ss_tot_train != 0 else float('nan')
    job["metrics"]["predicted_R2_Q2"] = float(pred_R2_train)

    # ---- Final training ----
    _set_progress(job, 55, "Final training on TRAIN, validate on VAL")

    np.random.seed(RANDOM_SEED)

    model = build_mlp(
        alpha=hp_values['alpha'],
        lr=hp_values['lr'],
        max_iter=FINAL_EPOCHS,
    )
    model.fit(X_train, y_train.ravel())

    best_epoch = len(model.loss_curve_)
    if hasattr(model, 'validation_scores_') and model.validation_scores_:
        best_epoch = int(np.argmax(model.validation_scores_) + 1)
    _log(job, f"Best epoch (by training convergence): {best_epoch}")
    job["metrics"]["best_epoch"] = best_epoch

    # Evaluate on TEST
    y_test_pred_eval = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).reshape(-1)
    y_test_inv_eval = scaler_y.inverse_transform(y_test).reshape(-1)
    test_r2 = r2_score(y_test_inv_eval, y_test_pred_eval)
    test_rmse = np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval))
    test_mae = np.mean(np.abs(y_test_inv_eval - y_test_pred_eval))
    _log(job, f"TEST: R2={test_r2:.4f}, RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    # Predictions BEFORE retrain (for PI calibration)
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train).reshape(-1, 1))
    y_val_pred = scaler_y.inverse_transform(model.predict(X_val).reshape(-1, 1))
    y_test_pred = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1))

    y_train_inv = scaler_y.inverse_transform(y_train)
    y_val_inv = scaler_y.inverse_transform(y_val)
    y_test_inv = scaler_y.inverse_transform(y_test)

    cal_residuals_val_preretrain = y_val_inv.reshape(-1) - y_val_pred.reshape(-1)

    _set_progress(job, 65, "Final training complete")

    # ---- Optional retrain on TRAIN+VAL ----
    if DO_OPTIONAL_RETRAIN:
        _set_progress(job, 67, "Retraining on TRAIN+VAL")

        X_trainval_orig = np.vstack([X_train_orig, X_val_orig])
        y_trainval_orig = np.vstack([y_train_orig, y_val_orig])

        scaler_X_tv = StandardScaler().fit(X_trainval_orig)
        scaler_y_tv = StandardScaler().fit(y_trainval_orig)

        X_trainval_tv = scaler_X_tv.transform(X_trainval_orig)
        y_trainval_tv = scaler_y_tv.transform(y_trainval_orig)
        X_test_tv = scaler_X_tv.transform(X_test_orig)

        np.random.seed(RANDOM_SEED)

        model_retrain = build_mlp(
            alpha=hp_values['alpha'],
            lr=hp_values['lr'],
            max_iter=best_epoch,
        )
        model_retrain.fit(X_trainval_tv, y_trainval_tv.ravel())

        y_test_pred_rt = scaler_y_tv.inverse_transform(model_retrain.predict(X_test_tv).reshape(-1, 1)).reshape(-1)
        y_test_inv_rt = y_test_orig.reshape(-1)

        rt_r2 = r2_score(y_test_inv_rt, y_test_pred_rt)
        rt_rmse = np.sqrt(mean_squared_error(y_test_inv_rt, y_test_pred_rt))
        rt_mae = np.mean(np.abs(y_test_inv_rt - y_test_pred_rt))
        _log(job, f"TEST (retrained): R2={rt_r2:.4f}, RMSE={rt_rmse:.4f}, MAE={rt_mae:.4f}")

        model = model_retrain
        scaler_X = scaler_X_tv
        scaler_y = scaler_y_tv
        X_train = scaler_X.transform(X_train_orig)
        X_val = scaler_X.transform(X_val_orig)
        X_test = scaler_X.transform(X_test_orig)
        y_train = scaler_y.transform(y_train_orig)
        y_val = scaler_y.transform(y_val_orig)
        y_test = scaler_y.transform(y_test_orig)

        y_train_pred = scaler_y.inverse_transform(model.predict(X_train).reshape(-1, 1))
        y_val_pred = scaler_y.inverse_transform(model.predict(X_val).reshape(-1, 1))
        y_test_pred = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1))
        y_train_inv = scaler_y.inverse_transform(y_train)
        y_val_inv = scaler_y.inverse_transform(y_val)
        y_test_inv = scaler_y.inverse_transform(y_test)

    _set_progress(job, 72, "Computing metrics")

    # ---- Metrics ----
    def compute_basic_metrics(y_true, y_pred):
        actual = np.array(y_true).reshape(-1)
        pred = np.array(y_pred).reshape(-1)
        n = len(actual)
        residuals = actual - pred
        SSE = float(np.sum(residuals ** 2))
        MSE = SSE / n if n > 0 else float('nan')
        RMSE = float(np.sqrt(MSE)) if not np.isnan(MSE) else float('nan')
        MAE = float(np.mean(np.abs(residuals))) if n > 0 else float('nan')
        R2 = float(r2_score(actual, pred)) if n > 0 else float('nan')
        return {"n": n, "SSE": SSE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE, "R2": R2}

    metrics_train = compute_basic_metrics(y_train_inv, y_train_pred)
    metrics_val = compute_basic_metrics(y_val_inv, y_val_pred)
    metrics_test = compute_basic_metrics(y_test_inv, y_test_pred)

    p = X_train.shape[1]
    for name, m_dict in [("train", metrics_train), ("val", metrics_val), ("test", metrics_test)]:
        n_m = m_dict["n"]
        r2_m = m_dict["R2"]
        r2_adj = 1 - (1 - r2_m) * (n_m - 1) / (n_m - p - 1) if (n_m - p - 1) > 0 else float('nan')
        m_dict["R2_adj"] = float(r2_adj)

    metrics_train["Predicted_R2_Q2"] = float(pred_R2_train)

    job["metrics"]["train"] = metrics_train
    job["metrics"]["val"] = metrics_val
    job["metrics"]["test"] = metrics_test

    # ---- Save model & scalers ----
    _set_progress(job, 74, "Saving model and scalers")

    joblib.dump(model, os.path.join(EXPORT_DIR, "final_model.pkl"))
    joblib.dump(scaler_X, os.path.join(EXPORT_DIR, "scaler_X.pkl"))
    joblib.dump(scaler_y, os.path.join(EXPORT_DIR, "scaler_y.pkl"))

    # ---- Plots ----
    _set_progress(job, 76, "Generating diagnostic plots")

    def _save_plot_to_job(fig, name):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        job["plots"][name] = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

    # MSE evolution
    try:
        loss_curve = model.loss_curve_
        if loss_curve:
            epochs_range = range(1, len(loss_curve) + 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs_range, loss_curve, label='Train MSE', marker='o', markersize=3)
            if hasattr(model, 'validation_scores_') and model.validation_scores_:
                val_epochs = range(1, len(model.validation_scores_) + 1)
                ax2 = ax.twinx()
                ax2.plot(val_epochs, model.validation_scores_, label='Val R2', marker='s', markersize=3, color='orange')
                ax2.set_ylabel('Validation R2')
                ax2.legend(loc='center right')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE (loss)')
            ax.set_title('MSE Evolution')
            ax.grid(True)
            ax.legend(loc='upper right')
            fig.tight_layout()
            _save_plot_to_job(fig, "mse_evolution")
            fig.savefig(os.path.join(EXPORT_DIR, "mse_evolution.tiff"), dpi=600, format="tiff")
            plt.close(fig)
    except Exception as e:
        _log(job, f"Error plotting MSE: {e}")

    # Predicted vs Actual
    def plot_pred_vs_actual(y_true, y_pred, title, name):
        y_true = np.array(y_true).reshape(-1)
        y_pred = np.array(y_pred).reshape(-1)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        mn = min(np.min(y_true), np.min(y_pred))
        mx = max(np.max(y_true), np.max(y_pred))
        ax.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
        try:
            sns.regplot(x=y_true, y=y_pred, scatter=False, color='blue', ci=None, ax=ax)
        except Exception:
            pass
        ax.set_title(title)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        _save_plot_to_job(fig, name)
        plt.close(fig)

    plot_pred_vs_actual(y_train_inv, y_train_pred, "Pred vs Actual (Train)", "pred_vs_actual_train")
    plot_pred_vs_actual(y_val_inv, y_val_pred, "Pred vs Actual (Val)", "pred_vs_actual_val")
    plot_pred_vs_actual(y_test_inv, y_test_pred, "Pred vs Actual (Test)", "pred_vs_actual_test")

    # Residual plots
    def plot_residuals(y_true, y_pred, title_prefix, name):
        residuals = np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1)
        std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(std_res, bins=25, color='gray', edgecolor='black', density=True)
        ax1.set_title(f"{title_prefix} - Std Residuals")
        ax1.set_xlabel("Standardized residuals")
        ax1.grid(True)
        ax2.plot(std_res, marker='o', linestyle='-', markersize=3)
        ax2.set_title(f"{title_prefix} - Residuals (time series)")
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Std Residual")
        ax2.grid(True)
        fig.tight_layout()
        _save_plot_to_job(fig, name)
        plt.close(fig)

    plot_residuals(y_train_inv, y_train_pred, "Train", "residuals_train")
    plot_residuals(y_val_inv, y_val_pred, "Val", "residuals_val")
    plot_residuals(y_test_inv, y_test_pred, "Test", "residuals_test")

    _set_progress(job, 80, "Generating 3D surface plots")

    # ---- 3D Surfaces ----
    def predict_from_origX(X_orig_2d):
        X_scaled = scaler_X.transform(X_orig_2d)
        y_pred_scaled = model.predict(X_scaled)
        return scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    n_features = X_train_orig.shape[1]
    Xref = np.median(X_train_orig, axis=0)

    def grid_vals(col_idx, grid_n=30):
        v = X_train_orig[:, col_idx]
        lo = float(np.quantile(v, 0.02))
        hi = float(np.quantile(v, 0.98))
        if np.isclose(lo, hi):
            lo, hi = lo - 1.0, hi + 1.0
        return np.linspace(lo, hi, grid_n)

    pairs_3d = list(combinations(range(n_features), 2))
    max_pairs = min(len(pairs_3d), 15)
    pairs_3d = pairs_3d[:max_pairs]

    plots_dir_3d = os.path.join(EXPORT_DIR, "plots_3d")
    os.makedirs(plots_dir_3d, exist_ok=True)

    for idx, (i, j) in enumerate(pairs_3d):
        pct = 80 + int((idx / max(len(pairs_3d), 1)) * 10)
        _set_progress(job, pct, f"3D surface {idx+1}/{len(pairs_3d)}")

        xi = grid_vals(i)
        xj = grid_vals(j)
        XI, XJ = np.meshgrid(xi, xj)

        Xgrid = np.tile(Xref.reshape(1, -1), (XI.size, 1))
        Xgrid[:, i] = XI.reshape(-1)
        Xgrid[:, j] = XJ.reshape(-1)

        Z = predict_from_origX(Xgrid).reshape(XI.shape)

        fig = plt.figure(figsize=(6, 5))
        ax3d = fig.add_subplot(111, projection="3d")
        surf = ax3d.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.9)
        fig.colorbar(surf, ax=ax3d, shrink=0.6, pad=0.1, label="Predicted")

        col_i = inputs_columns[i] if i < len(inputs_columns) else str(i)
        col_j = inputs_columns[j] if j < len(inputs_columns) else str(j)
        ax3d.set_title(f"{col_i} vs {col_j}")
        ax3d.set_xlabel(str(col_i))
        ax3d.set_ylabel(str(col_j))
        ax3d.set_zlabel("Predicted")
        ax3d.view_init(elev=25, azim=-135)
        fig.tight_layout()

        plot_name = f"surface_{col_i}_vs_{col_j}"
        _save_plot_to_job(fig, plot_name)

        safe_i = str(col_i).replace(" ", "_").replace("/", "_")
        safe_j = str(col_j).replace(" ", "_").replace("/", "_")
        fig.savefig(os.path.join(plots_dir_3d, f"surface_{safe_i}_vs_{safe_j}.tiff"), dpi=600, format="tiff")
        plt.close(fig)

    _set_progress(job, 90, "Exporting results")

    # ---- Export DataFrames ----
    def make_export_df(labels_part, inputs_part, y_true, y_pred):
        actual = np.array(y_true).reshape(-1)
        pred = np.array(y_pred).reshape(-1)
        residual = actual - pred
        abs_err = np.abs(residual)
        df_exp = pd.DataFrame(index=range(len(actual)))
        if labels_part is not None and labels_part.shape[0] == len(actual):
            df_exp = pd.concat([df_exp, labels_part.reset_index(drop=True)], axis=1)
        df_exp = pd.concat([df_exp, inputs_part.reset_index(drop=True)], axis=1)
        df_exp["Actual_Value"] = actual
        df_exp["Predicted_Value"] = pred
        df_exp["Residual"] = residual
        df_exp["Abs_Error"] = abs_err
        return df_exp

    results_train = make_export_df(labels_train.reset_index(drop=True), X_train_df, y_train_inv, y_train_pred)
    results_val = make_export_df(labels_val.reset_index(drop=True), X_val_df, y_val_inv, y_val_pred)
    results_test = make_export_df(labels_test.reset_index(drop=True), X_test_df, y_test_inv, y_test_pred)

    results_train.to_excel(os.path.join(EXPORT_DIR, "train_predictions.xlsx"), index=False, engine='openpyxl')
    results_val.to_excel(os.path.join(EXPORT_DIR, "val_predictions.xlsx"), index=False, engine='openpyxl')
    results_test.to_excel(os.path.join(EXPORT_DIR, "test_predictions.xlsx"), index=False, engine='openpyxl')
    results_train.to_csv(os.path.join(EXPORT_DIR, "train_predictions.csv"), index=False)
    results_val.to_csv(os.path.join(EXPORT_DIR, "val_predictions.csv"), index=False)
    results_test.to_csv(os.path.join(EXPORT_DIR, "test_predictions.csv"), index=False)

    # ---- Save stats file ----
    stats_path = os.path.join(EXPORT_DIR, "model_statistics.txt")
    with open(stats_path, "w") as f:
        f.write("Model statistics summary\n")
        f.write("========================\n\n")
        f.write(f"Date: {_dt.date.today().isoformat()}\n")
        f.write(f"Split: {TRAIN_PERCENT}% / {VAL_PERCENT}% / {TEST_PERCENT}%\n\n")
        f.write("Best Hyperparameters:\n")
        for k, v in hp_values.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTrain: {metrics_train}\n")
        f.write(f"Val: {metrics_val}\n")
        f.write(f"Test: {metrics_test}\n")
        f.write(f"\nQ2: {pred_R2_train}\n")
        f.write(f"Best epoch: {best_epoch}\n")

    # ---- Create ZIP ----
    _set_progress(job, 93, "Creating results ZIP")

    zip_path = os.path.join(EXPORT_DIR, "training_results.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(EXPORT_DIR):
            for fname in files:
                fpath = os.path.join(root, fname)
                if fpath == zip_path:
                    continue
                if "tuner_results" in fpath:
                    continue
                arcname = os.path.relpath(fpath, EXPORT_DIR)
                zipf.write(fpath, arcname)

    job["results_zip"] = zip_path

    # ---- New data predictions ----
    if new_data_path and os.path.exists(new_data_path):
        _set_progress(job, 95, "Processing new data predictions")
        try:
            new_df = load_table(new_data_path, sep=SEP)
            _log(job, f"New data loaded. Shape: {new_df.shape}")

            n_cols_new = new_df.shape[1]
            new_labels_df = new_df.iloc[:, :N_LABELS] if N_LABELS > 0 else pd.DataFrame()

            if n_cols_new >= N_LABELS + N_INPUTS:
                new_inputs_df = new_df.iloc[:, N_LABELS:N_LABELS + N_INPUTS]
            else:
                raise ValueError(f"Insufficient columns: {n_cols_new} < {N_LABELS + N_INPUTS}")

            new_X = new_inputs_df.values
            new_X_scaled = scaler_X.transform(new_X)
            new_y_pred = model.predict(new_X_scaled)
            new_y_pred_inv = scaler_y.inverse_transform(new_y_pred.reshape(-1, 1)).reshape(-1)

            # PI
            cal_residuals = cal_residuals_val_preretrain if PI_CALIBRATION == "val" else (y_oof_true_inv - y_oof_pred_inv)
            q_low = np.quantile(cal_residuals, PI_ALPHA / 2.0)
            q_high = np.quantile(cal_residuals, 1.0 - PI_ALPHA / 2.0)
            pi_lower = new_y_pred_inv + q_low
            pi_upper = new_y_pred_inv + q_high

            new_results = pd.DataFrame(index=range(len(new_X)))
            if new_labels_df.shape[0] == len(new_X):
                new_results = pd.concat([new_results, new_labels_df.reset_index(drop=True)], axis=1)
            new_results = pd.concat([new_results, new_inputs_df.reset_index(drop=True)], axis=1)
            new_results["Predicted_Value"] = new_y_pred_inv
            new_results["PI_Lower_95%"] = pi_lower
            new_results["PI_Upper_95%"] = pi_upper

            new_results.to_excel(os.path.join(EXPORT_DIR, "new_data_predictions.xlsx"), index=False, engine='openpyxl')
            new_results.to_csv(os.path.join(EXPORT_DIR, "new_data_predictions.csv"), index=False)

            pred_zip_path = os.path.join(EXPORT_DIR, "new_data_predictions.zip")
            with zipfile.ZipFile(pred_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(os.path.join(EXPORT_DIR, "new_data_predictions.xlsx"), "new_data_predictions.xlsx")
                zipf.write(os.path.join(EXPORT_DIR, "new_data_predictions.csv"), "new_data_predictions.csv")
            job["predictions_zip"] = pred_zip_path

        except Exception as e:
            _log(job, f"Error processing new data: {e}")

    _set_progress(job, 100, "Pipeline complete!")
    job["status"] = "completed"
    _log(job, "All operations completed successfully.")
