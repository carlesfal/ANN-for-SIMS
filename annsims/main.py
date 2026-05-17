"""Main orchestration pipeline for the ANN regression workflow.

Run this module directly or import ``run_pipeline`` to execute
the full training / evaluation / export / prediction pipeline.

Usage (Colab or CLI)::

    from annsims.main import run_pipeline
    from annsims.config import PipelineConfig

    cfg = PipelineConfig(train_percent=60, val_percent=20, test_percent=20)
    run_pipeline(cfg)
"""

from __future__ import annotations

import os
import traceback

import numpy as np
import tensorflow as tf

from .config import PipelineConfig, SurfacePlotConfig
from .data import (
    fit_scalers,
    load_table,
    select_columns,
    split_data,
    upload_or_find_file,
)
from .export import (
    create_zip,
    download_zip,
    make_export_df,
    make_full_row_export,
    save_model_and_scalers,
    save_model_scheme,
    save_results,
)
from .metrics import (
    adjusted_r2,
    compute_basic_metrics,
    print_metrics,
    save_statistics_file,
)
from .model import (
    make_model_builder,
    retrain_on_trainval,
    run_cv,
    run_tuner,
    train_final_model,
    try_weight_transfer,
)
from .plotting import (
    generate_3d_surfaces,
    generate_diagnostic_plots,
    plot_mse_evolution,
)
from .predict import get_calibration_residuals, predict_new_data
from .utils import detect_notebook, show


def run_pipeline(
    cfg: PipelineConfig,
    surface_cfg: SurfacePlotConfig | None = None,
    warmup_model=None,
) -> None:
    """Execute the full ANN regression pipeline.

    Parameters
    ----------
    cfg:
        Pipeline configuration (splits, epochs, etc.).
    surface_cfg:
        3D surface plot configuration (optional; uses defaults if ``None``).
    warmup_model:
        An optional pre-trained Keras model for weight transfer.
    """
    if surface_cfg is None:
        surface_cfg = SurfacePlotConfig()

    in_notebook = detect_notebook()

    print(f"Data split: TRAIN={cfg.train_percent}%  VAL={cfg.val_percent}%  TEST={cfg.test_percent}%")

    # Reproducibility
    np.random.seed(cfg.random_seed)
    tf.random.set_seed(cfg.random_seed)

    if cfg.disable_gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("GPU disabled. Using CPU.")
        except Exception as exc:
            print(f"Could not change GPU visibility: {exc}")

    # ---- Load data ----
    print("Upload your dataset.")
    file_name = upload_or_find_file()
    df = load_table(file_name, sep=cfg.sep)
    print(f"Data loaded. Shape: {df.shape}")
    show(df.head())

    cols = select_columns(df, cfg)
    X_full = cols.inputs_df.values
    y_full = cols.y_full

    # ---- Split ----
    split = split_data(X_full, y_full, cols.labels_df, cols.inputs_df, cfg)
    scaled = fit_scalers(split)

    # ---- Model builder (no global state) ----
    n_features = split.X_train_orig.shape[1]
    build_fn = make_model_builder(n_features, use_batch_norm=cfg.use_batch_norm)

    # ---- Hyperparameter search ----
    tuner, best_hp = run_tuner(
        build_fn,
        scaled.X_train, scaled.y_train,
        scaled.X_val, scaled.y_val,
        cfg,
    )

    # ---- Weight transfer ----
    pretrained = try_weight_transfer(build_fn, best_hp, warmup_model)

    # ---- Cross-validation ----
    cv = run_cv(
        build_fn, best_hp,
        split.X_train_orig, split.y_train_orig,
        cfg,
        pretrained_model=pretrained,
    )

    # ---- Final training ----
    tr = train_final_model(
        build_fn, best_hp,
        scaled.X_train, scaled.y_train,
        scaled.X_val, scaled.y_val,
        cfg,
    )
    model = tr.model
    scaler_X = scaled.scaler_X
    scaler_y = scaled.scaler_y

    # Evaluate once on TEST
    from sklearn.metrics import r2_score, mean_squared_error

    print("\nEvaluating once on TEST.")
    y_test_pred_eval = scaler_y.inverse_transform(model.predict(scaled.X_test, verbose=0)).reshape(-1)
    y_test_inv_eval = scaler_y.inverse_transform(scaled.y_test).reshape(-1)
    print(
        f"TEST: R²={r2_score(y_test_inv_eval, y_test_pred_eval):.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(y_test_inv_eval, y_test_pred_eval)):.4f}  "
        f"MAE={np.mean(np.abs(y_test_inv_eval - y_test_pred_eval)):.4f}"
    )

    # ---- CV fold ensemble evaluation (if enabled) ----
    if cfg.use_cv_ensemble and cv.fold_models:
        print("\nCV Ensemble evaluation on TEST:")
        ensemble_preds = np.zeros(len(split.X_test_orig))
        for fm, fsx, fsy in zip(cv.fold_models, cv.fold_scalers_X, cv.fold_scalers_y):
            pred_i = fsy.inverse_transform(
                fm.predict(fsx.transform(split.X_test_orig), verbose=0)
            ).reshape(-1)
            ensemble_preds += pred_i
        ensemble_preds /= len(cv.fold_models)
        y_test_true = split.y_test_orig.reshape(-1)
        print(
            f"TEST (ensemble of {len(cv.fold_models)} folds): "
            f"R²={r2_score(y_test_true, ensemble_preds):.4f}  "
            f"RMSE={np.sqrt(mean_squared_error(y_test_true, ensemble_preds)):.4f}  "
            f"MAE={np.mean(np.abs(y_test_true - ensemble_preds)):.4f}"
        )

    # ---- Optional retrain on train+val ----
    if cfg.do_optional_retrain:
        rt = retrain_on_trainval(
            build_fn, best_hp,
            split.X_train_orig, split.X_val_orig,
            split.y_train_orig, split.y_val_orig,
            split.X_test_orig, split.y_test_orig,
            tr.best_epoch,
        )
        model = rt.model
        scaler_X = rt.scaler_X
        scaler_y = rt.scaler_y
        # Re-scale all splits with the new scalers
        scaled_X_train = scaler_X.transform(split.X_train_orig)
        scaled_X_val = scaler_X.transform(split.X_val_orig)
        scaled_X_test = scaler_X.transform(split.X_test_orig)
        scaled_y_train = scaler_y.transform(split.y_train_orig)
        scaled_y_val = scaler_y.transform(split.y_val_orig)
        scaled_y_test = scaler_y.transform(split.y_test_orig)
        print("Using retrained model + train+val-fitted scalers.")
    else:
        scaled_X_train = scaled.X_train
        scaled_X_val = scaled.X_val
        scaled_X_test = scaled.X_test
        scaled_y_train = scaled.y_train
        scaled_y_val = scaled.y_val
        scaled_y_test = scaled.y_test

    # ---- Model scheme ----
    try:
        save_model_scheme(model, cfg.export_dir)
    except Exception as exc:
        print(f"Error saving model scheme: {exc}")

    # ---- Save model + scalers ----
    model_files = save_model_and_scalers(model, scaler_X, scaler_y, cfg.export_dir)

    # ---- Predictions (inverse-scaled) ----
    y_train_pred = scaler_y.inverse_transform(model.predict(scaled_X_train, verbose=0))
    y_val_pred = scaler_y.inverse_transform(model.predict(scaled_X_val, verbose=0))
    y_test_pred = scaler_y.inverse_transform(model.predict(scaled_X_test, verbose=0))
    y_train_inv = scaler_y.inverse_transform(scaled_y_train)
    y_val_inv = scaler_y.inverse_transform(scaled_y_val)
    y_test_inv = scaler_y.inverse_transform(scaled_y_test)

    # ---- Metrics ----
    m_train = compute_basic_metrics(y_train_inv, y_train_pred)
    m_val = compute_basic_metrics(y_val_inv, y_val_pred)
    m_test = compute_basic_metrics(y_test_inv, y_test_pred)

    p = n_features
    m_train["R2_adj"] = adjusted_r2(m_train["R2"], m_train["n"], p)
    m_val["R2_adj"] = adjusted_r2(m_val["R2"], m_val["n"], p)
    m_test["R2_adj"] = adjusted_r2(m_test["R2"], m_test["n"], p)
    m_train["Predicted_R2_Q2"] = cv.pred_r2
    m_val["Predicted_R2_Q2"] = float("nan")
    m_test["Predicted_R2_Q2"] = float("nan")

    print_metrics(m_train, m_val, m_test, cfg.train_percent, cfg.val_percent, cfg.test_percent)

    # ---- Statistics file ----
    try:
        save_statistics_file(
            os.path.join(cfg.export_dir, "model_statistics.txt"),
            p=p,
            train_pct=cfg.train_percent,
            val_pct=cfg.val_percent,
            test_pct=cfg.test_percent,
            best_hp_values=best_hp.values,
            metrics_train=m_train,
            metrics_val=m_val,
            metrics_test=m_test,
            pred_r2_train=cv.pred_r2,
            best_epoch=tr.best_epoch,
            do_optional_retrain=cfg.do_optional_retrain,
            pi_calibration=cfg.pi_calibration,
            pi_alpha=cfg.pi_alpha,
        )
    except Exception as exc:
        print(f"Could not save statistics: {exc}")

    # ---- MSE evolution plot ----
    try:
        plot_mse_evolution(
            tr.history,
            os.path.join(cfg.export_dir, "mse_evolution.png"),
            in_notebook=in_notebook,
        )
    except Exception as exc:
        print(f"Error plotting MSE evolution: {exc}")

    # ---- Diagnostic plots ----
    try:
        generate_diagnostic_plots(
            y_train_inv, y_train_pred,
            y_val_inv, y_val_pred,
            y_test_inv, y_test_pred,
            os.path.join(cfg.export_dir, "plots"),
            in_notebook=in_notebook,
        )
    except Exception as exc:
        print(f"Error creating diagnostic plots: {exc}")
        traceback.print_exc()

    # ---- Export DataFrames ----
    results_train = make_export_df(
        split.labels_train.reset_index(drop=True), split.X_train_df, y_train_inv, y_train_pred,
    )
    results_val = make_export_df(
        split.labels_val.reset_index(drop=True), split.X_val_df, y_val_inv, y_val_pred,
    )
    results_test = make_export_df(
        split.labels_test.reset_index(drop=True), split.X_test_df, y_test_inv, y_test_pred,
    )

    try:
        full_train = make_full_row_export(df, split.idx_train, y_train_inv, y_train_pred)
        full_val = make_full_row_export(df, split.idx_val, y_val_inv, y_val_pred)
        full_test = make_full_row_export(df, split.idx_test, y_test_inv, y_test_pred)
    except Exception as exc:
        print(f"Error creating full-row exports: {exc}")
        traceback.print_exc()
        full_train = full_val = full_test = None

    result_files = save_results(
        cfg.export_dir,
        results_train, results_val, results_test,
        full_train, full_val, full_test,
    )

    show(results_train.head())
    show(results_val.head())
    show(results_test.head())

    # ---- 3D surfaces ----
    try:
        generate_3d_surfaces(
            split.X_train_orig,
            split.input_columns,
            model, scaler_X, scaler_y,
            os.path.join(cfg.export_dir, "plots_3d"),
            surface_cfg,
            in_notebook=in_notebook,
        )
    except Exception as exc:
        print(f"Error generating 3D surfaces: {exc}")
        traceback.print_exc()

    # ---- Download #1: training results ZIP ----
    print("\n" + "=" * 80)
    print("DOWNLOAD #1: TRAINING RESULTS")
    print("=" * 80)

    all_training_files = list(result_files) + list(model_files)
    all_training_files.append(os.path.join(cfg.export_dir, "model_statistics.txt"))
    all_training_files.append(os.path.join(cfg.export_dir, "model_scheme.txt"))
    all_training_files.append(os.path.join(cfg.export_dir, "mse_evolution.png"))
    for subdir in ("plots", "plots_3d"):
        d = os.path.join(cfg.export_dir, subdir)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                all_training_files.append(os.path.join(d, fn))

    zip_path = create_zip("training_results.zip", all_training_files)
    if zip_path:
        download_zip(zip_path)

    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETED!")
    print("=" * 80)

    # ---- New-data prediction section ----
    print("\n" + "=" * 80)
    print("NEW DATA PREDICTION SECTION")
    print("=" * 80)

    cal_residuals = get_calibration_residuals(
        cfg.pi_calibration,
        y_val_inv.reshape(-1), y_val_pred.reshape(-1),
        cv.y_oof_true, cv.y_oof_pred,
    )
    q_lo = float(np.quantile(cal_residuals, cfg.pi_alpha / 2.0))
    q_hi = float(np.quantile(cal_residuals, 1.0 - cfg.pi_alpha / 2.0))
    print(f"Empirical PI ({cfg.pi_calibration}): q_low={q_lo:.4f}  q_high={q_hi:.4f}  alpha={cfg.pi_alpha}")

    try:
        pred_files = predict_new_data(
            model, scaler_X, scaler_y, cfg,
            split.input_columns,
            split.X_train_orig, split.y_train_orig,
            q_lo, q_hi,
            cfg.export_dir,
        )
    except Exception as exc:
        print(f"Error processing new data: {exc}")
        traceback.print_exc()
        pred_files = []

    # ---- Download #2: new-data predictions ZIP ----
    print("\n" + "=" * 80)
    print("DOWNLOAD #2: NEW DATA PREDICTIONS")
    print("=" * 80)

    if pred_files:
        zip2 = create_zip("new_data_predictions.zip", pred_files)
        if zip2:
            download_zip(zip2)
    else:
        print("No new-data predictions available.")

    print("\n" + "=" * 80)
    print("ALL OPERATIONS COMPLETED!")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(PipelineConfig())
