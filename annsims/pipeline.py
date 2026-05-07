"""Core training pipeline logic for ANNSIMS."""

import os
import traceback
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

try:
    import keras_tuner as kt
except ImportError:
    import kerastuner as kt

from annsims.config import PipelineConfig


def transfer_weights_smart(source_model, target_model, mode="smart", verbose=True, log_fn=None):
    """
    Transfer weights from source_model to target_model layer-by-layer.

    Modes:
      - "strict": only transfer when shapes match exactly.
      - "smart":  transfer overlapping slices when shapes partially match.
    """
    if log_fn is None:
        log_fn = print

    dense_src = [l for l in source_model.layers if isinstance(l, layers.Dense)]
    dense_tgt = [l for l in target_model.layers if isinstance(l, layers.Dense)]

    stats = {"full": 0, "partial": 0, "skipped": 0, "details": []}
    n_pairs = min(len(dense_src), len(dense_tgt))

    for i, (ls, lt) in enumerate(zip(dense_src, dense_tgt)):
        ws_list = ls.get_weights()
        wt_list = lt.get_weights()

        if len(ws_list) != len(wt_list):
            info = f"  Layer {i} ({ls.name} -> {lt.name}): skipped (different # weight arrays)"
            stats["skipped"] += 1
            stats["details"].append(info)
            if verbose:
                log_fn(info)
            continue

        all_match = all(ws.shape == wt.shape for ws, wt in zip(ws_list, wt_list))

        if all_match:
            lt.set_weights(ws_list)
            info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                    f"FULL transfer {[w.shape for w in ws_list]}")
            stats["full"] += 1
            stats["details"].append(info)
            if verbose:
                log_fn(info)
        elif mode == "smart":
            new_weights = []
            transferred_something = False

            for ws, wt in zip(ws_list, wt_list):
                wt_current = np.array(wt)

                if ws.ndim == 2 and wt.ndim == 2:
                    min_in = min(ws.shape[0], wt.shape[0])
                    min_out = min(ws.shape[1], wt.shape[1])
                    if min_in > 0 and min_out > 0:
                        wt_new = wt_current.copy()
                        wt_new[:min_in, :min_out] = ws[:min_in, :min_out]
                        new_weights.append(wt_new)
                        transferred_something = True
                    else:
                        new_weights.append(wt_current)
                elif ws.ndim == 1 and wt.ndim == 1:
                    min_dim = min(ws.shape[0], wt.shape[0])
                    if min_dim > 0:
                        wt_new = wt_current.copy()
                        wt_new[:min_dim] = ws[:min_dim]
                        new_weights.append(wt_new)
                        transferred_something = True
                    else:
                        new_weights.append(wt_current)
                else:
                    new_weights.append(wt_current)

            if transferred_something:
                lt.set_weights(new_weights)
                src_shapes = [w.shape for w in ws_list]
                tgt_shapes = [w.shape for w in wt_list]
                info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                        f"PARTIAL transfer (src={src_shapes}, tgt={tgt_shapes})")
                stats["partial"] += 1
            else:
                info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                        f"skipped (no overlap)")
                stats["skipped"] += 1
            stats["details"].append(info)
            if verbose:
                log_fn(info)
        else:
            info = (f"  Layer {i} ({ls.name} -> {lt.name}): "
                    f"skipped (shape mismatch, strict mode)")
            stats["skipped"] += 1
            stats["details"].append(info)
            if verbose:
                log_fn(info)

    if len(dense_src) > n_pairs:
        info = f"  {len(dense_src) - n_pairs} extra source layer(s) not transferred"
        stats["details"].append(info)
        if verbose:
            log_fn(info)
    if len(dense_tgt) > n_pairs:
        info = f"  {len(dense_tgt) - n_pairs} extra target layer(s) kept random init"
        stats["details"].append(info)
        if verbose:
            log_fn(info)

    return stats


def transfer_weights_to_fold(source_model, fold_model, mode="smart", verbose=False, log_fn=None):
    """Transfer weights into a CV fold model. Returns True if at least one layer transferred."""
    stats = transfer_weights_smart(source_model, fold_model, mode=mode, verbose=verbose, log_fn=log_fn)
    return (stats["full"] + stats["partial"]) > 0


def load_table(path, sep="\t"):
    """Load a data table from CSV, TSV, or Excel file."""
    _, ext = os.path.splitext(path.lower())
    if ext in [".xlsx", ".xls", ".xlsm"]:
        return pd.read_excel(path, engine="openpyxl")
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except Exception as e:
            last_err = e
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read '{path}'. Last error: {last_err}")


def compute_basic_metrics(y_true, y_pred):
    """Compute regression metrics."""
    actual = np.array(y_true).reshape(-1)
    pred = np.array(y_pred).reshape(-1)
    n = len(actual)
    residuals = actual - pred
    SSE = np.sum(residuals ** 2)
    MSE = SSE / n if n > 0 else np.nan
    RMSE = np.sqrt(MSE) if not np.isnan(MSE) else np.nan
    MAE = np.mean(np.abs(residuals)) if n > 0 else np.nan
    SEP_val = np.sqrt(SSE / (n - 1)) if n > 1 else np.nan
    mean_abs = np.mean(np.abs(actual))
    MRPD = (100.0 * np.sum(np.abs(residuals)) / (n * mean_abs)
            if n > 0 and not np.isclose(mean_abs, 0.0) else np.nan)
    R2 = r2_score(actual, pred) if n > 0 else np.nan
    return {"n": n, "SSE": SSE, "MSE": MSE, "RMSE": RMSE, "MAE": MAE,
            "SEP": SEP_val, "MRPD_percent": MRPD, "R2": R2}


class PipelineRunner:
    """Runs the full ANN training pipeline with callback-based logging."""

    def __init__(self, config: PipelineConfig, data_path: str, log_fn=None,
                 progress_fn=None, cancelled_fn=None):
        self.cfg = config
        self.data_path = data_path
        self.log = log_fn or print
        self.progress = progress_fn or (lambda x: None)
        self.cancelled = cancelled_fn or (lambda: False)

        # Pipeline state
        self.warmup_model = None
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.best_hp = None
        self.tuner = None
        self.metrics_train = None
        self.metrics_val = None
        self.metrics_test = None
        self.cv_summary = None
        self.X_train_orig = None
        self.inputs_columns = None
        self.plot_paths = []

    def run(self):
        """Execute the full pipeline."""
        try:
            self.cfg.validate()
            self._setup_seeds()
            self._phase1_hill_pretrain()
            if self.cancelled():
                return
            self._phase2_data_loading()
            if self.cancelled():
                return
            self._phase3_tuning()
            if self.cancelled():
                return
            self._phase4_weight_transfer()
            if self.cancelled():
                return
            self._phase5_cv()
            if self.cancelled():
                return
            self._phase6_final_training()
            if self.cancelled():
                return
            if self.cfg.do_optional_retrain:
                self._phase7_retrain()
                if self.cancelled():
                    return
            self._phase8_metrics_export()
            if self.cancelled():
                return
            self._phase9_plots()
            self.log("\n" + "=" * 70)
            self.log("ALL PHASES COMPLETE")
            self.log("=" * 70)
        except Exception as e:
            self.log(f"\nERROR: {e}")
            self.log(traceback.format_exc())
            raise

    def _setup_seeds(self):
        np.random.seed(self.cfg.random_seed)
        tf.random.set_seed(self.cfg.random_seed)
        if self.cfg.disable_gpu:
            try:
                tf.config.set_visible_devices([], 'GPU')
                self.log("GPU disabled. Using CPU.")
            except Exception as e:
                self.log(f"Could not change GPU visibility: {e}")

    def _phase1_hill_pretrain(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 1: HILL PRE-TRAINING")
        self.log("=" * 70)
        self.progress(5)

        cfg = self.cfg
        self.log("[1/4] Generating synthetic Hill data...")
        X_h = np.random.uniform(cfg.hill_x_min, cfg.hill_x_max,
                                (cfg.n_synthetic, cfg.n_inputs)).astype(np.float32)
        x1 = X_h[:, 0]
        y_h = cfg.hill_v_max * (x1 ** cfg.hill_n) / (cfg.hill_k ** cfg.hill_n + x1 ** cfg.hill_n)
        y_h += np.random.normal(0, 0.05 * cfg.hill_v_max, cfg.n_synthetic)
        for i in range(1, min(cfg.n_inputs, 3)):
            y_h += 0.05 * cfg.hill_v_max * (X_h[:, i] - cfg.hill_x_min) / (cfg.hill_x_max - cfg.hill_x_min)
        y_h = y_h.reshape(-1, 1)
        self.log(f"  X {X_h.shape} | y {y_h.shape} | y range [{y_h.min():.2f}, {y_h.max():.2f}]")

        self.log("\n[2/4] Scaling Hill data...")
        scaler_Xh = StandardScaler().fit(X_h)
        scaler_yh = StandardScaler().fit(y_h)
        X_h_sc = scaler_Xh.transform(X_h)
        y_h_sc = scaler_yh.transform(y_h)
        X_h_train, X_h_val, y_h_train, y_h_val = train_test_split(
            X_h_sc, y_h_sc, test_size=0.2, random_state=cfg.random_seed
        )
        self.log(f"  Hill train {X_h_train.shape} | Hill val {X_h_val.shape}")

        self.log("\n[3/4] Building warmup model (128 -> 64 -> 32 -> 1)...")
        tf.keras.backend.clear_session()
        self.warmup_model = keras.Sequential([
            layers.Input(shape=(cfg.n_inputs,)),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
            layers.Dropout(0.1),
            layers.Dense(1, activation='linear'),
        ])
        self.warmup_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
        self.log(f"  {self.warmup_model.count_params():,} parameters")

        self.log(f"\n[4/4] Pre-training ({cfg.pretrain_epochs} epochs, early-stop patience=10)...")
        es_warmup = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        hist_warmup = self.warmup_model.fit(
            X_h_train, y_h_train,
            validation_data=(X_h_val, y_h_val),
            epochs=cfg.pretrain_epochs,
            batch_size=32,
            callbacks=[es_warmup],
            verbose=0,
        )
        self.log(f"\n  Pre-training done | "
                 f"train loss {hist_warmup.history['loss'][-1]:.6f} | "
                 f"val loss   {hist_warmup.history['val_loss'][-1]:.6f}")
        self.progress(15)

    def _phase2_data_loading(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 2: DATA LOADING & SPLITTING")
        self.log("=" * 70)

        cfg = self.cfg
        df = load_table(self.data_path, sep=cfg.sep)
        self.log(f"Data loaded. Shape: {df.shape}")

        n_cols = df.shape[1]
        target_col = cfg.target_col if cfg.target_col >= 0 else None

        if target_col is not None:
            if not (0 <= target_col < n_cols):
                raise IndexError("TARGET_COL out of range.")
            labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
            input_cols = [c for c in range(cfg.n_labels, n_cols) if c != target_col]
            inputs_df = df.iloc[:, input_cols]
            y_full = df.iloc[:, target_col].values.reshape(-1, 1)
        else:
            if cfg.n_labels + cfg.n_inputs >= n_cols:
                self.log("N_LABELS + N_INPUTS >= total columns -> using last column as target.")
                labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
                inputs_df = df.iloc[:, cfg.n_labels:-1]
                y_full = df.iloc[:, -1].values.reshape(-1, 1)
            else:
                labels_df = df.iloc[:, :cfg.n_labels] if cfg.n_labels > 0 else pd.DataFrame()
                inputs_df = df.iloc[:, cfg.n_labels:cfg.n_labels + cfg.n_inputs]
                y_full = df.iloc[:, cfg.n_labels + cfg.n_inputs].values.reshape(-1, 1)

        self.log(f"Labels {labels_df.shape} | Inputs {inputs_df.shape} | y {y_full.shape}")

        X_full = inputs_df.values
        labels_full = labels_df
        self.inputs_columns = list(inputs_df.columns)

        test_frac = cfg.test_percent / 100.0
        val_frac = cfg.val_percent / 100.0
        train_frac = cfg.train_percent / 100.0
        val_split_ratio = val_frac / (train_frac + val_frac)

        (X_temp, X_test_orig,
         y_temp, y_test_orig,
         labels_temp, labels_test,
         _, idx_test) = train_test_split(
            X_full, y_full, labels_full, np.arange(len(df)),
            test_size=test_frac, random_state=cfg.random_seed,
        )
        (X_train_orig, X_val_orig,
         y_train_orig, y_val_orig,
         labels_train, labels_val,
         _, idx_val) = train_test_split(
            X_temp, y_temp, labels_temp, np.arange(len(X_temp)),
            test_size=val_split_ratio, random_state=cfg.random_seed,
        )

        self.log(f"\nSplit (rows): train={len(X_train_orig)} | val={len(X_val_orig)} | test={len(X_test_orig)}")

        # Store originals
        self.X_train_orig = X_train_orig
        self.X_val_orig = X_val_orig
        self.X_test_orig = X_test_orig
        self.y_train_orig = y_train_orig
        self.y_val_orig = y_val_orig
        self.y_test_orig = y_test_orig
        self.labels_train = labels_train
        self.labels_val = labels_val
        self.labels_test = labels_test

        # DataFrames for export
        self.X_train_df = pd.DataFrame(X_train_orig, columns=self.inputs_columns).reset_index(drop=True)
        self.X_val_df = pd.DataFrame(X_val_orig, columns=self.inputs_columns).reset_index(drop=True)
        self.X_test_df = pd.DataFrame(X_test_orig, columns=self.inputs_columns).reset_index(drop=True)

        # Scalers fit on TRAIN only
        self.scaler_X = StandardScaler().fit(X_train_orig)
        self.scaler_y = StandardScaler().fit(y_train_orig)

        self.X_train = self.scaler_X.transform(X_train_orig)
        self.X_val = self.scaler_X.transform(X_val_orig)
        self.X_test = self.scaler_X.transform(X_test_orig)
        self.y_train = self.scaler_y.transform(y_train_orig)
        self.y_val = self.scaler_y.transform(y_val_orig)
        self.y_test = self.scaler_y.transform(y_test_orig)

        self.progress(20)

    def _phase3_tuning(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 3: HYPERPARAMETER TUNING")
        self.log("=" * 70)

        cfg = self.cfg
        X_train = self.X_train
        X_val = self.X_val
        y_train = self.y_train
        y_val = self.y_val

        def build_model(hp):
            model = keras.Sequential()
            model.add(layers.Input(shape=(X_train.shape[1],)))
            n_layers = hp.Int('num_layers', 2, 6, step=1)
            l2_val = hp.Choice('l2_reg', [1e-4, 1e-3, 1e-2])
            for i in range(n_layers):
                model.add(layers.Dense(
                    hp.Int(f'units_{i}', 64, 512, step=64),
                    activation='relu',
                    kernel_regularizer=regularizers.l2(l2_val),
                ))
                model.add(layers.Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)))
            model.add(layers.Dense(1, activation='linear'))
            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=hp.Choice('lr', [1e-4, 5e-4, 1e-3, 5e-3])
                ),
                loss='mse',
                metrics=['mae'],
            )
            return model

        self.build_model_fn = build_model

        self.tuner = kt.RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=cfg.tuner_trials,
            executions_per_trial=1,
            overwrite=True,
            directory='tuner_results',
            project_name=f'ann_{cfg.train_percent}_{cfg.val_percent}_{cfg.test_percent}',
        )

        self.log(f"Starting search ({cfg.tuner_trials} trials)...")
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.tuner_epochs,
            batch_size=32,
            verbose=0,
        )

        self.best_hp = self.tuner.get_best_hyperparameters(1)[0]
        self.log("\nBest hyperparameters:")
        for k, v in self.best_hp.values.items():
            self.log(f"  {k}: {v}")
        self.progress(40)

    def _phase4_weight_transfer(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 4: WEIGHT TRANSFER FROM WARMUP MODEL")
        self.log("=" * 70)

        cfg = self.cfg
        self._model_with_pretrain = None
        model_for_transfer = None

        try:
            self.log("[1/3] Building model from best HPs...")
            model_for_transfer = self.tuner.hypermodel.build(self.best_hp)
            self.log(f"  {model_for_transfer.count_params():,} parameters")

            self.log(f"\n[2/3] Smart weight transfer (mode='{cfg.transfer_mode}')...")
            stats = transfer_weights_smart(
                self.warmup_model, model_for_transfer,
                mode=cfg.transfer_mode, verbose=True, log_fn=self.log
            )

            self.log(f"\n[3/3] Transfer summary: "
                     f"{stats['full']} full, {stats['partial']} partial, "
                     f"{stats['skipped']} skipped")

            if (stats["full"] + stats["partial"]) > 0:
                self._model_with_pretrain = model_for_transfer
                self.log("-> _model_with_pretrain is ready")
            else:
                self.log("WARNING: No weights transferred (architecture mismatch)")
        except Exception as e:
            self.log(f"Weight transfer failed: {e}")
            self._model_with_pretrain = None

        if self._model_with_pretrain is None and model_for_transfer is not None:
            self._model_with_pretrain = model_for_transfer
            self.log("  Falling back to tuned model without transferred weights")

        self.progress(45)

    def _phase5_cv(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 5: K-FOLD CV ON TRAIN SPLIT")
        self.log("=" * 70)

        cfg = self.cfg
        kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        y_oof_pred_inv = np.full(len(self.y_train_orig), np.nan)
        y_oof_true_inv = np.full(len(self.y_train_orig), np.nan)

        best_fold_r2 = -np.inf
        self.best_fold_weights = None

        use_pretrain = self._model_with_pretrain is not None
        if use_pretrain:
            self.log(f"  Will seed each fold from pre-trained weights (mode='{cfg.transfer_mode}')")
        else:
            self.log("  No pre-trained model - folds start from random init")

        self.log(f"\nRunning {cfg.k_folds}-fold CV...\n")

        for fold, (tr_idx, va_idx) in enumerate(kf.split(self.X_train_orig), start=1):
            if self.cancelled():
                return

            X_tr_orig, X_va_orig = self.X_train_orig[tr_idx], self.X_train_orig[va_idx]
            y_tr_orig, y_va_orig = self.y_train_orig[tr_idx], self.y_train_orig[va_idx]

            fold_scaler_X = StandardScaler().fit(X_tr_orig)
            fold_scaler_y = StandardScaler().fit(y_tr_orig)

            X_tr = fold_scaler_X.transform(X_tr_orig)
            X_va = fold_scaler_X.transform(X_va_orig)
            y_tr = fold_scaler_y.transform(y_tr_orig)
            y_va = fold_scaler_y.transform(y_va_orig)

            model_fold = self.tuner.hypermodel.build(self.best_hp)

            if use_pretrain:
                transfer_weights_to_fold(
                    self._model_with_pretrain, model_fold,
                    mode=cfg.transfer_mode, verbose=False, log_fn=self.log
                )

            es_fold = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            model_fold.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=cfg.cv_epochs,
                batch_size=32,
                callbacks=[es_fold],
                verbose=0,
            )

            y_va_pred_inv = fold_scaler_y.inverse_transform(
                model_fold.predict(X_va, verbose=0)
            ).reshape(-1)
            y_va_true_inv = y_va_orig.reshape(-1)

            y_oof_pred_inv[va_idx] = y_va_pred_inv
            y_oof_true_inv[va_idx] = y_va_true_inv

            r2 = r2_score(y_va_true_inv, y_va_pred_inv)
            rmse = np.sqrt(mean_squared_error(y_va_true_inv, y_va_pred_inv))
            mae = np.mean(np.abs(y_va_true_inv - y_va_pred_inv))
            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            self.log(f"  Fold {fold:2d}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

            if r2 > best_fold_r2:
                best_fold_r2 = r2
                self.best_fold_weights = model_fold.get_weights()

            self.progress(45 + int(20 * fold / cfg.k_folds))

        self.log(f"\nCV summary:")
        self.log(f"  R2   {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
        self.log(f"  RMSE {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")
        self.log(f"  MAE  {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")
        self.log(f"\n  Best fold R2: {best_fold_r2:.4f}")

        # Predicted R2 (Q2)
        ss_res = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
        ss_tot = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
        self.pred_R2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
        self.log(f"  Predicted R2 (Q2/PRESS): {self.pred_R2:.4f}")

        self.y_oof_pred_inv = y_oof_pred_inv
        self.y_oof_true_inv = y_oof_true_inv
        self.cv_summary = {
            "r2_mean": np.mean(r2_scores), "r2_std": np.std(r2_scores),
            "rmse_mean": np.mean(rmse_scores), "rmse_std": np.std(rmse_scores),
            "mae_mean": np.mean(mae_scores), "mae_std": np.std(mae_scores),
            "best_fold_r2": best_fold_r2, "pred_R2": self.pred_R2,
        }
        self.progress(65)

    def _phase6_final_training(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 6: FINAL TRAINING")
        self.log("=" * 70)

        cfg = self.cfg
        os.makedirs(cfg.export_dir, exist_ok=True)
        checkpoint_path = os.path.join(cfg.export_dir, "best_model.keras")

        tf.keras.backend.clear_session()
        self.model = self.tuner.hypermodel.build(self.best_hp)

        if cfg.transfer_best_fold and self.best_fold_weights is not None:
            self.log("[Weight Transfer] Seeding final model from best CV fold weights...")
            try:
                self.model.set_weights(self.best_fold_weights)
                self.log("  Full weight transfer from best fold successful")
            except ValueError:
                self.log("  Direct transfer failed, using smart per-layer transfer...")
                temp_source = self.tuner.hypermodel.build(self.best_hp)
                temp_source.set_weights(self.best_fold_weights)
                transfer_weights_smart(temp_source, self.model, mode=cfg.transfer_mode,
                                       verbose=True, log_fn=self.log)
                del temp_source
        elif self._model_with_pretrain is not None:
            self.log("[Weight Transfer] Seeding final model from warmup pre-trained weights...")
            transfer_weights_smart(self._model_with_pretrain, self.model,
                                   mode=cfg.transfer_mode, verbose=True, log_fn=self.log)
        else:
            self.log("[Weight Transfer] No weight seeding (random init)")

        mc = callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0
        )
        es_final = callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=cfg.final_epochs,
            batch_size=32,
            callbacks=[es_final, mc],
            verbose=0,
        )

        if os.path.exists(checkpoint_path):
            self.model = keras.models.load_model(checkpoint_path, compile=False)

        self.best_epoch = int(np.argmin(history.history['val_loss']) + 1)
        self.log(f"\n  Best epoch (by val loss): {self.best_epoch}")

        # Quick test eval
        y_test_pred = self.scaler_y.inverse_transform(
            self.model.predict(self.X_test, verbose=0)).reshape(-1)
        y_test_true = self.scaler_y.inverse_transform(self.y_test).reshape(-1)
        self.log(f"\nEvaluating on TEST:")
        self.log(f"  R2={r2_score(y_test_true, y_test_pred):.4f}  "
                 f"RMSE={np.sqrt(mean_squared_error(y_test_true, y_test_pred)):.4f}  "
                 f"MAE={np.mean(np.abs(y_test_true - y_test_pred)):.4f}")
        self.progress(75)

    def _phase7_retrain(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 7: OPTIONAL RETRAIN ON TRAIN+VAL")
        self.log("=" * 70)

        cfg = self.cfg
        X_trainval_orig = np.vstack([self.X_train_orig, self.X_val_orig])
        y_trainval_orig = np.vstack([self.y_train_orig, self.y_val_orig])

        self.scaler_X = StandardScaler().fit(X_trainval_orig)
        self.scaler_y = StandardScaler().fit(y_trainval_orig)

        X_trainval = self.scaler_X.transform(X_trainval_orig)
        y_trainval = self.scaler_y.transform(y_trainval_orig)

        tf.keras.backend.clear_session()
        model_retrain = self.tuner.hypermodel.build(self.best_hp)

        if cfg.transfer_best_fold and self.best_fold_weights is not None:
            self.log("[Weight Transfer] Seeding retrain model from best CV fold...")
            try:
                model_retrain.set_weights(self.best_fold_weights)
                self.log("  Full weight transfer successful")
            except ValueError:
                self.log("  Direct transfer failed, using smart per-layer transfer...")
                temp_source = self.tuner.hypermodel.build(self.best_hp)
                temp_source.set_weights(self.best_fold_weights)
                transfer_weights_smart(temp_source, model_retrain, mode=cfg.transfer_mode,
                                       verbose=False, log_fn=self.log)
                del temp_source

        model_retrain.fit(
            X_trainval, y_trainval,
            epochs=self.best_epoch,
            batch_size=32,
            verbose=0,
        )

        # Evaluate retrained model on test
        X_test_rt = self.scaler_X.transform(self.X_test_orig)
        y_test_pred_rt = self.scaler_y.inverse_transform(
            model_retrain.predict(X_test_rt, verbose=0)).reshape(-1)
        y_test_inv_rt = self.y_test_orig.reshape(-1)

        self.log(f"\nTEST (retrained model, TRAIN+VAL scalers):")
        self.log(f"  R2={r2_score(y_test_inv_rt, y_test_pred_rt):.4f}  "
                 f"RMSE={np.sqrt(mean_squared_error(y_test_inv_rt, y_test_pred_rt)):.4f}  "
                 f"MAE={np.mean(np.abs(y_test_inv_rt - y_test_pred_rt)):.4f}")

        # Promote retrained model
        self.model = model_retrain
        self.X_train = self.scaler_X.transform(self.X_train_orig)
        self.X_val = self.scaler_X.transform(self.X_val_orig)
        self.X_test = self.scaler_X.transform(self.X_test_orig)
        self.y_train = self.scaler_y.transform(self.y_train_orig)
        self.y_val = self.scaler_y.transform(self.y_val_orig)
        self.y_test = self.scaler_y.transform(self.y_test_orig)
        self.log("\nRetrained model and TRAIN+VAL scalers are now active.")
        self.progress(82)

    def _phase8_metrics_export(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 8: METRICS, PREDICTION INTERVALS & EXPORT")
        self.log("=" * 70)

        cfg = self.cfg
        y_train_pred = self.scaler_y.inverse_transform(self.model.predict(self.X_train, verbose=0))
        y_val_pred = self.scaler_y.inverse_transform(self.model.predict(self.X_val, verbose=0))
        y_test_pred = self.scaler_y.inverse_transform(self.model.predict(self.X_test, verbose=0))

        y_train_inv = self.scaler_y.inverse_transform(self.y_train)
        y_val_inv = self.scaler_y.inverse_transform(self.y_val)
        y_test_inv = self.scaler_y.inverse_transform(self.y_test)

        self.metrics_train = compute_basic_metrics(y_train_inv, y_train_pred)
        self.metrics_val = compute_basic_metrics(y_val_inv, y_val_pred)
        self.metrics_test = compute_basic_metrics(y_test_inv, y_test_pred)

        # Adjusted R2
        p = self.X_train.shape[1]
        for m in (self.metrics_train, self.metrics_val, self.metrics_test):
            n, r2 = m["n"], m["R2"]
            m["R2_adj"] = (1 - (1 - r2) * (n - 1) / (n - p - 1)
                           if (n - p - 1) > 0 else np.nan)

        self.metrics_train["Predicted_R2_Q2"] = self.pred_R2
        self.metrics_val["Predicted_R2_Q2"] = np.nan
        self.metrics_test["Predicted_R2_Q2"] = np.nan

        self.log("\n=== Final Metrics ===")
        for label, m in [(f"Train ({cfg.train_percent}%)", self.metrics_train),
                         (f"Val   ({cfg.val_percent}%)", self.metrics_val),
                         (f"Test  ({cfg.test_percent}%)", self.metrics_test)]:
            self.log(f"\n{label}:")
            for k, v in m.items():
                self.log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        # Prediction Intervals
        self.log(f"\n=== Empirical Prediction Intervals ({int((1 - cfg.pi_alpha) * 100)}% PI) ===")
        if cfg.pi_calibration == "oof":
            residuals_cal = self.y_oof_true_inv - self.y_oof_pred_inv
            cal_label = "OOF residuals (train CV)"
        else:
            residuals_cal = y_val_inv.reshape(-1) - y_val_pred.reshape(-1)
            cal_label = "val residuals"

        pi_lo_q = np.quantile(residuals_cal, cfg.pi_alpha / 2)
        pi_hi_q = np.quantile(residuals_cal, 1 - cfg.pi_alpha / 2)
        self.log(f"Calibrated on {cal_label}: offset = [{pi_lo_q:+.4f}, {pi_hi_q:+.4f}]")

        def add_pi(pred_arr):
            p_arr = pred_arr.reshape(-1)
            return p_arr + pi_lo_q, p_arr + pi_hi_q

        y_train_pi_lo, y_train_pi_hi = add_pi(y_train_pred)
        y_val_pi_lo, y_val_pi_hi = add_pi(y_val_pred)
        y_test_pi_lo, y_test_pi_hi = add_pi(y_test_pred)

        for label, y_true, lo, hi in [
            ("Train", y_train_inv.reshape(-1), y_train_pi_lo, y_train_pi_hi),
            ("Val", y_val_inv.reshape(-1), y_val_pi_lo, y_val_pi_hi),
            ("Test", y_test_inv.reshape(-1), y_test_pi_lo, y_test_pi_hi),
        ]:
            cov = np.mean((y_true >= lo) & (y_true <= hi))
            self.log(f"  {label} empirical coverage: {100 * cov:.1f}%")

        # Save model & scalers
        os.makedirs(cfg.export_dir, exist_ok=True)
        self.model.save(os.path.join(cfg.export_dir, "final_model.keras"))
        try:
            self.model.save(os.path.join(cfg.export_dir, "final_model.h5"))
        except Exception:
            pass
        joblib.dump(self.scaler_X, os.path.join(cfg.export_dir, "scaler_X.pkl"))
        joblib.dump(self.scaler_y, os.path.join(cfg.export_dir, "scaler_y.pkl"))
        self.log(f"\nModel & scalers saved to '{cfg.export_dir}/'")

        # Export result CSVs
        pi_col = int((1 - cfg.pi_alpha) * 100)

        def make_result_df(lbl_df, inp_df, y_true, y_pred, lo, hi):
            out = pd.concat(
                [lbl_df.reset_index(drop=True), inp_df.reset_index(drop=True)], axis=1
            )
            out["y_true"] = np.array(y_true).reshape(-1)
            out["y_pred"] = np.array(y_pred).reshape(-1)
            out[f"PI_{pi_col}_lo"] = lo
            out[f"PI_{pi_col}_hi"] = hi
            return out

        df_train_res = make_result_df(self.labels_train.reset_index(drop=True), self.X_train_df,
                                      y_train_inv, y_train_pred, y_train_pi_lo, y_train_pi_hi)
        df_val_res = make_result_df(self.labels_val.reset_index(drop=True), self.X_val_df,
                                    y_val_inv, y_val_pred, y_val_pi_lo, y_val_pi_hi)
        df_test_res = make_result_df(self.labels_test.reset_index(drop=True), self.X_test_df,
                                     y_test_inv, y_test_pred, y_test_pi_lo, y_test_pi_hi)

        for name, frame in [("results_train.csv", df_train_res),
                            ("results_val.csv", df_val_res),
                            ("results_test.csv", df_test_res)]:
            path = os.path.join(cfg.export_dir, name)
            frame.to_csv(path, index=False)
            self.log(f"Saved: {path}")

        self.progress(88)

    def _phase9_plots(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 9: 3D SURFACE PLOTS + EXPORT")
        self.log("=" * 70)

        cfg = self.cfg
        fig_width = cfg.fig_width_cm / 2.54
        self.plot_paths = []

        def _grid_vals(col_idx, grid_n, range_mode, q_low, q_high):
            v = self.X_train_orig[:, col_idx]
            if range_mode == "minmax":
                lo, hi = float(np.min(v)), float(np.max(v))
            elif range_mode in ("quantile", "train_quantile"):
                lo, hi = float(np.quantile(v, q_low)), float(np.quantile(v, q_high))
            else:
                lo, hi = float(np.min(v)), float(np.max(v))
            if np.isclose(lo, hi):
                lo, hi = lo - 1.0, hi + 1.0
            return np.linspace(lo, hi, grid_n)

        def predict_from_origX(X_orig_2d):
            X_scaled = self.scaler_X.transform(X_orig_2d)
            y_scaled = self.model.predict(X_scaled, verbose=0)
            return self.scaler_y.inverse_transform(y_scaled).reshape(-1)

        def _build_xref(hold_mode, row_index):
            if hold_mode == "median_train":
                return np.median(self.X_train_orig, axis=0)
            elif hold_mode == "mean_train":
                return np.mean(self.X_train_orig, axis=0)
            elif hold_mode == "row":
                return self.X_train_orig[row_index].copy()
            else:
                return np.median(self.X_train_orig, axis=0)

        def _safe_name(s):
            s = str(s)
            for ch in [" ", "/", "\\", ":", ";", "|", "(", ")", "[", "]", "{", "}", "%"]:
                s = s.replace(ch, "_")
            return s

        # Export ALL 3D surfaces to TIFF
        try:
            plots_dir_3d = os.path.join(cfg.export_dir, "plots_3d")
            os.makedirs(plots_dir_3d, exist_ok=True)

            p3d = self.X_train_orig.shape[1]
            Xref_3d = _build_xref(cfg.hold_mode_export, cfg.row_index_export)
            pairs_3d = list(combinations(range(p3d), 2))
            self.log(f"\nGenerating {len(pairs_3d)} 3D surfaces into: {plots_dir_3d}")

            for idx, (i, j) in enumerate(pairs_3d):
                if self.cancelled():
                    return

                xi = _grid_vals(i, cfg.grid_n_export, cfg.range_mode_export,
                                cfg.q_low_export, cfg.q_high_export)
                xj = _grid_vals(j, cfg.grid_n_export, cfg.range_mode_export,
                                cfg.q_low_export, cfg.q_high_export)
                XI, XJ = np.meshgrid(xi, xj)

                Xgrid = np.tile(Xref_3d.reshape(1, -1), (XI.size, 1))
                Xgrid[:, i] = XI.reshape(-1)
                Xgrid[:, j] = XJ.reshape(-1)

                Z = predict_from_origX(Xgrid).reshape(XI.shape)

                fig = plt.figure(figsize=(fig_width, fig_width * 0.85))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0,
                                antialiased=True, alpha=0.95)

                col_i = self.inputs_columns[i]
                col_j = self.inputs_columns[j]
                ax.set_title(f"Predicted: {col_i} vs {col_j}\n(held: {cfg.hold_mode_export})",
                             fontsize=7)
                ax.set_xlabel(str(col_i), fontsize=7)
                ax.set_ylabel(str(col_j), fontsize=7)
                ax.set_zlabel("Predicted", fontsize=7)
                ax.view_init(elev=25, azim=-135)
                plt.tight_layout()

                out_path = os.path.join(
                    plots_dir_3d,
                    f"surface_{_safe_name(col_i)}_vs_{_safe_name(col_j)}.tiff"
                )
                plt.savefig(out_path, dpi=cfg.dpi, format='tiff')
                plt.close(fig)
                self.plot_paths.append(out_path)

            self.log(f"3D surfaces saved. Count: {len(self.plot_paths)}")
        except Exception as e:
            self.log(f"Error generating 3D surfaces: {e}")
            self.log(traceback.format_exc())

        self.progress(100)
