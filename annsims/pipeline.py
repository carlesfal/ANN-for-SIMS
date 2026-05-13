"""Core training pipeline logic for ANNSIMS.

Faithfully implements the user's production ANN training pipeline:
  Phase 1: Hill pre-training (warmup model, loss=mae)
  Phase 2: Data loading & splitting
  Phase 3: Hyperparameter tuning (fixed 128->64->32->1 architecture)
  Phase 4: Weight transfer from warmup model
  Phase 5: K-fold CV on train split (no leakage)
  Phase 6: Final training (fit on train, validate on val)
  Phase 7: Optional retrain on train+val
  Phase 8: Metrics, model statistics, diagnostic plots, exports
  Phase 9: 3D surface plots (all feature pairs)
"""

import os
import shutil
import traceback
import datetime as _dt
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Compute regression metrics: n, SSE, MSE, RMSE, MAE, SEP, MRPD%, R2."""
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


def plot_predicted_vs_actual(y_true, y_pred, title, save_path=None):
    """Scatter plot of predicted vs actual values with ideal line and regression."""
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--', label='Ideal')
    try:
        sns.regplot(x=y_true, y=y_pred, scatter=False, color='blue', ci=None)
    except Exception:
        pass
    plt.title(title)
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_residuals_std_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    """Standardized residuals histogram + time series plot."""
    residuals = np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1)
    std_res = (residuals - np.mean(residuals)) / (np.std(residuals) + 1e-12)
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(std_res, bins=25, kde=True, color='gray', edgecolor='black')
    plt.title(f"{title_prefix} - Standardized residuals distribution")
    plt.xlabel("Standardized residuals")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(std_res, marker='o', linestyle='-')
    plt.title(f"{title_prefix} - Standardized residuals (time series)")
    plt.xlabel("Observation index")
    plt.ylabel("Standardized residual")
    plt.grid(True)
    plt.tight_layout()
    if save_prefix:
        path = save_prefix + "_std_residuals_hist_ts.png"
        plt.savefig(path, dpi=150)
    plt.close()


def plot_residuals_raw_and_timeseries(y_true, y_pred, title_prefix, save_prefix=None):
    """Raw residuals histogram + time series plot."""
    residuals = np.array(y_true).reshape(-1) - np.array(y_pred).reshape(-1)
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, bins=25, kde=True, color='salmon', edgecolor='black')
    plt.title(f"{title_prefix} - Raw residuals distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(residuals, marker='o', linestyle='-')
    plt.title(f"{title_prefix} - Raw residuals (time series)")
    plt.xlabel("Observation index")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    if save_prefix:
        path = save_prefix + "_raw_residuals_hist_ts.png"
        plt.savefig(path, dpi=150)
    plt.close()


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
        self.history = None
        self.best_epoch = None
        self.metrics_train = None
        self.metrics_val = None
        self.metrics_test = None
        self.cv_summary = None
        self.X_train_orig = None
        self.X_val_orig = None
        self.X_test_orig = None
        self.y_train_orig = None
        self.y_val_orig = None
        self.y_test_orig = None
        self.inputs_columns = None
        self.plot_paths = []
        self._model_with_pretrain = None
        self.pred_R2 = None
        self.y_oof_pred_inv = None
        self.y_oof_true_inv = None
        self.df = None
        self.idx_train = None
        self.idx_val = None
        self.idx_test = None

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

    # ------------------------------------------------------------------
    # SEEDS & GPU
    # ------------------------------------------------------------------
    def _setup_seeds(self):
        np.random.seed(self.cfg.random_seed)
        tf.random.set_seed(self.cfg.random_seed)
        if self.cfg.disable_gpu:
            try:
                tf.config.set_visible_devices([], 'GPU')
                self.log("GPU disabled. Using CPU.")
            except Exception as e:
                self.log(f"Could not change GPU visibility: {e}")

    # ------------------------------------------------------------------
    # PHASE 1: HILL PRE-TRAINING
    # ------------------------------------------------------------------
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
        self.warmup_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mae'
        )
        self.log(f"  {self.warmup_model.count_params():,} parameters")

        self.log(f"\n[4/4] Pre-training ({cfg.pretrain_epochs} epochs, early-stop patience=5)...")
        es_warmup = callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
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

    # ------------------------------------------------------------------
    # PHASE 2: DATA LOADING & SPLITTING
    # ------------------------------------------------------------------
    def _phase2_data_loading(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 2: DATA LOADING & SPLITTING")
        self.log("=" * 70)

        cfg = self.cfg
        self.df = load_table(self.data_path, sep=cfg.sep)
        df = self.df
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
        y_full_arr = y_full
        labels_full = labels_df
        self.inputs_columns = list(inputs_df.columns)
        row_pos = np.arange(len(df))

        test_frac = cfg.test_percent / 100.0
        val_frac = cfg.val_percent / 100.0
        train_frac = cfg.train_percent / 100.0

        (X_temp, X_test_orig,
         y_temp, y_test_orig,
         labels_temp, labels_test,
         idx_temp, idx_test) = train_test_split(
            X_full, y_full_arr, labels_full, row_pos,
            test_size=test_frac, random_state=cfg.random_seed,
        )

        val_split_ratio = val_frac / (train_frac + val_frac)
        (X_train_orig, X_val_orig,
         y_train_orig, y_val_orig,
         labels_train, labels_val,
         idx_train, idx_val) = train_test_split(
            X_temp, y_temp, labels_temp, idx_temp,
            test_size=val_split_ratio, random_state=cfg.random_seed,
        )

        self.log(f"\nSplit (rows): train={len(X_train_orig)} | val={len(X_val_orig)} | test={len(X_test_orig)}")
        self.log(f"Split percentages: "
                 f"train={100*len(X_train_orig)/len(df):.1f}% "
                 f"val={100*len(X_val_orig)/len(df):.1f}% "
                 f"test={100*len(X_test_orig)/len(df):.1f}%")

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
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

        # DataFrames for export
        self.X_train_df = pd.DataFrame(X_train_orig, columns=self.inputs_columns).reset_index(drop=True)
        self.X_val_df = pd.DataFrame(X_val_orig, columns=self.inputs_columns).reset_index(drop=True)
        self.X_test_df = pd.DataFrame(X_test_orig, columns=self.inputs_columns).reset_index(drop=True)

        # Scalers fit on TRAIN only (no leakage)
        self.scaler_X = StandardScaler().fit(X_train_orig)
        self.scaler_y = StandardScaler().fit(y_train_orig)

        self.X_train = self.scaler_X.transform(X_train_orig)
        self.X_val = self.scaler_X.transform(X_val_orig)
        self.X_test = self.scaler_X.transform(X_test_orig)
        self.y_train = self.scaler_y.transform(y_train_orig)
        self.y_val = self.scaler_y.transform(y_val_orig)
        self.y_test = self.scaler_y.transform(y_test_orig)

        self.progress(20)

    # ------------------------------------------------------------------
    # PHASE 3: HYPERPARAMETER TUNING (FIXED ARCHITECTURE)
    # ------------------------------------------------------------------
    def _phase3_tuning(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 3: HYPERPARAMETER TUNING (FIXED ARCHITECTURE)")
        self.log("=" * 70)

        cfg = self.cfg
        X_train = self.X_train
        X_val = self.X_val
        y_train = self.y_train
        y_val = self.y_val

        def build_model_fixed(hp):
            """Fixed architecture: 128 -> 64 -> 32 -> 1.
            Only tunes: L2 regularization, dropout rates, learning rate.
            """
            model = keras.Sequential()
            model.add(layers.Input(shape=(X_train.shape[1],)))

            l2_val = hp.Choice('l2_reg', [1e-4, 1e-3, 1e-2])

            model.add(layers.Dense(128, activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_val)))
            model.add(layers.Dropout(hp.Float('dropout_1', 0.0, 0.3, step=0.05)))

            model.add(layers.Dense(64, activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_val)))
            model.add(layers.Dropout(hp.Float('dropout_2', 0.0, 0.3, step=0.05)))

            model.add(layers.Dense(32, activation='relu',
                                   kernel_regularizer=regularizers.l2(l2_val)))
            model.add(layers.Dropout(hp.Float('dropout_3', 0.0, 0.2, step=0.05)))

            model.add(layers.Dense(1, activation='linear'))

            model.compile(
                optimizer=keras.optimizers.Adam(
                    learning_rate=hp.Choice('lr', [1e-4, 5e-4, 1e-3, 5e-3])
                ),
                loss='mae',
                metrics=['mse'],
            )
            return model

        self.build_model_fn = build_model_fixed

        self.tuner = kt.RandomSearch(
            build_model_fixed,
            objective='val_loss',
            max_trials=cfg.tuner_trials,
            executions_per_trial=1,
            overwrite=True,
            directory='tuner_results',
            project_name=f'ann_{cfg.train_percent}_{cfg.val_percent}_{cfg.test_percent}',
        )

        self.log(f"Starting hyperparameter search ({cfg.tuner_trials} trials)...")
        self.tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.tuner_epochs,
            batch_size=32,
            verbose=0,
        )

        self.best_hp = self.tuner.get_best_hyperparameters(1)[0]
        self.log("\nBest hyperparameters found:")
        for k, v in self.best_hp.values.items():
            self.log(f"  {k}: {v}")
        self.progress(40)

    # ------------------------------------------------------------------
    # PHASE 4: WEIGHT TRANSFER FROM WARMUP MODEL
    # ------------------------------------------------------------------
    def _phase4_weight_transfer(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 4: WEIGHT TRANSFER FROM WARMUP MODEL")
        self.log("=" * 70)

        self._model_with_pretrain = None

        if self.warmup_model is not None:
            self.log("Found pre-trained warmup_model!")
            try:
                model_for_transfer = self.tuner.hypermodel.build(self.best_hp)
                self.log(f"  Model built: {model_for_transfer.count_params():,} parameters")

                n_transferred = 0
                for layer_final, layer_warmup in zip(
                    model_for_transfer.layers, self.warmup_model.layers
                ):
                    if isinstance(layer_final, layers.InputLayer) or \
                       isinstance(layer_warmup, layers.InputLayer):
                        continue
                    if not layer_final.weights or not layer_warmup.weights:
                        continue
                    try:
                        if len(layer_final.weights) == len(layer_warmup.weights):
                            all_match = all(
                                w_f.shape == w_w.shape
                                for w_f, w_w in zip(layer_final.weights, layer_warmup.weights)
                            )
                            if all_match:
                                layer_final.set_weights(layer_warmup.get_weights())
                                n_transferred += 1
                                self.log(f"  {layer_warmup.name} -> {layer_final.name}")
                            else:
                                self.log(f"  Skipping {layer_warmup.name}: shape mismatch")
                        else:
                            self.log(f"  Skipping {layer_warmup.name}: different # weight arrays")
                    except Exception as e:
                        self.log(f"  Skipping {layer_warmup.name}: {e}")

                if n_transferred > 0:
                    self.log(f"\nTransferred {n_transferred} layers!")
                    self._model_with_pretrain = model_for_transfer
                else:
                    self.log("\nNo compatible weights transferred (architecture mismatch)")
            except Exception as e:
                self.log(f"Weight transfer failed: {e}")
        else:
            self.log("No pre-trained model found")

        self.progress(45)

    # ------------------------------------------------------------------
    # PHASE 5: K-FOLD CV ON TRAIN SPLIT (NO LEAKAGE)
    # ------------------------------------------------------------------
    def _phase5_cv(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 5: K-FOLD CV ON TRAIN SPLIT")
        self.log("=" * 70)

        cfg = self.cfg
        kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_seed)
        r2_scores, rmse_scores, mae_scores = [], [], []
        y_oof_pred_inv = np.full(len(self.y_train_orig), np.nan, dtype=float)
        y_oof_true_inv = np.full(len(self.y_train_orig), np.nan, dtype=float)

        use_pretrain = self._model_with_pretrain is not None
        if use_pretrain:
            self.log("Using pre-trained model for CV folds")
            base_model = self._model_with_pretrain
        else:
            self.log("No pre-trained model, building fresh models for each fold")

        self.log(f"\nRunning {cfg.k_folds}-Fold CV on TRAIN split ONLY "
                 f"with fold-fitted scalers (no leakage)...\n")

        for fold, (tr_idx, va_idx) in enumerate(kf.split(self.X_train_orig), start=1):
            if self.cancelled():
                return

            X_tr_orig = self.X_train_orig[tr_idx]
            X_va_orig = self.X_train_orig[va_idx]
            y_tr_orig = self.y_train_orig[tr_idx]
            y_va_orig = self.y_train_orig[va_idx]

            fold_scaler_X = StandardScaler().fit(X_tr_orig)
            fold_scaler_y = StandardScaler().fit(y_tr_orig)

            X_tr = fold_scaler_X.transform(X_tr_orig)
            X_va = fold_scaler_X.transform(X_va_orig)
            y_tr = fold_scaler_y.transform(y_tr_orig)
            y_va = fold_scaler_y.transform(y_va_orig)

            model_fold = self.build_model_fn(self.best_hp)

            if use_pretrain:
                model_fold.set_weights(base_model.get_weights())

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

            self.progress(45 + int(20 * fold / cfg.k_folds))

        self.log(f"\nCV Metrics (Mean +/- Std) on TRAIN split (no leakage):")
        self.log(f"  R2:   {np.mean(r2_scores):.4f} +/- {np.std(r2_scores):.4f}")
        self.log(f"  RMSE: {np.mean(rmse_scores):.4f} +/- {np.std(rmse_scores):.4f}")
        self.log(f"  MAE:  {np.mean(mae_scores):.4f} +/- {np.std(mae_scores):.4f}")

        if np.isnan(y_oof_pred_inv).any():
            self.log("WARNING: OOF predictions contain NaNs. Check fold logic.")

        ss_res = np.sum((y_oof_true_inv - y_oof_pred_inv) ** 2)
        ss_tot = np.sum((y_oof_true_inv - np.mean(y_oof_true_inv)) ** 2)
        self.pred_R2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
        self.log(f"\nPredicted R2 (Q2) on TRAIN split (OOF/PRESS, no leakage): {self.pred_R2:.4f}")

        self.y_oof_pred_inv = y_oof_pred_inv
        self.y_oof_true_inv = y_oof_true_inv
        self.cv_summary = {
            "r2_mean": np.mean(r2_scores), "r2_std": np.std(r2_scores),
            "rmse_mean": np.mean(rmse_scores), "rmse_std": np.std(rmse_scores),
            "mae_mean": np.mean(mae_scores), "mae_std": np.std(mae_scores),
            "pred_R2": self.pred_R2,
        }
        self.progress(65)

    # ------------------------------------------------------------------
    # PHASE 6: FINAL TRAINING
    # ------------------------------------------------------------------
    def _phase6_final_training(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 6: FINAL TRAINING (fit on TRAIN, validate on VAL)")
        self.log("=" * 70)

        cfg = self.cfg
        os.makedirs(cfg.export_dir, exist_ok=True)
        checkpoint_path = os.path.join(cfg.export_dir, "best_model.keras")

        mc = callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0
        )
        es_final = callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )

        tf.keras.backend.clear_session()
        self.model = self.tuner.hypermodel.build(self.best_hp)

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=cfg.final_epochs,
            batch_size=32,
            callbacks=[es_final, mc],
            verbose=0,
        )

        if os.path.exists(checkpoint_path):
            self.model = keras.models.load_model(checkpoint_path, compile=False)

        self.best_epoch = int(np.argmin(self.history.history['val_loss']) + 1)
        self.log(f"\nBest epoch selected by VAL loss: {self.best_epoch}")

        # Evaluate ONCE on TEST (no tuning on test)
        self.log("\nEvaluating once on TEST (no selection/tuning on test).")
        y_test_pred = self.scaler_y.inverse_transform(
            self.model.predict(self.X_test, verbose=0)).reshape(-1)
        y_test_true = self.y_test_orig.reshape(-1)
        self.log(f"TEST: R2={r2_score(y_test_true, y_test_pred):.4f}, "
                 f"RMSE={np.sqrt(mean_squared_error(y_test_true, y_test_pred)):.4f}, "
                 f"MAE={np.mean(np.abs(y_test_true - y_test_pred)):.4f}")
        self.progress(75)

    # ------------------------------------------------------------------
    # PHASE 7: OPTIONAL RETRAIN ON TRAIN+VAL
    # ------------------------------------------------------------------
    def _phase7_retrain(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 7: OPTIONAL RETRAIN ON TRAIN+VAL")
        self.log("=" * 70)
        self.log("Refit scalers on TRAIN+VAL, retrain for best_epoch, evaluate once on TEST.")

        X_trainval_orig = np.vstack([self.X_train_orig, self.X_val_orig])
        y_trainval_orig = np.vstack([self.y_train_orig, self.y_val_orig])

        scaler_X_tv = StandardScaler().fit(X_trainval_orig)
        scaler_y_tv = StandardScaler().fit(y_trainval_orig)

        X_trainval_tv = scaler_X_tv.transform(X_trainval_orig)
        y_trainval_tv = scaler_y_tv.transform(y_trainval_orig)

        X_test_tv = scaler_X_tv.transform(self.X_test_orig)

        tf.keras.backend.clear_session()
        model_retrain = self.tuner.hypermodel.build(self.best_hp)
        model_retrain.fit(
            X_trainval_tv, y_trainval_tv,
            epochs=self.best_epoch,
            batch_size=32,
            verbose=0,
        )

        y_test_pred_rt = scaler_y_tv.inverse_transform(
            model_retrain.predict(X_test_tv, verbose=0)).reshape(-1)
        y_test_inv_rt = self.y_test_orig.reshape(-1)

        self.log(f"\nTEST (retrained): R2={r2_score(y_test_inv_rt, y_test_pred_rt):.4f}, "
                 f"RMSE={np.sqrt(mean_squared_error(y_test_inv_rt, y_test_pred_rt)):.4f}, "
                 f"MAE={np.mean(np.abs(y_test_inv_rt - y_test_pred_rt)):.4f}")

        # Promote retrained model and scalers
        self.model = model_retrain
        self.scaler_X = scaler_X_tv
        self.scaler_y = scaler_y_tv
        self.X_train = self.scaler_X.transform(self.X_train_orig)
        self.X_val = self.scaler_X.transform(self.X_val_orig)
        self.X_test = self.scaler_X.transform(self.X_test_orig)
        self.y_train = self.scaler_y.transform(self.y_train_orig)
        self.y_val = self.scaler_y.transform(self.y_val_orig)
        self.y_test = self.scaler_y.transform(self.y_test_orig)
        self.log("Using retrained model + train+val-fitted scalers for exports/predictions.")
        self.progress(82)

    # ------------------------------------------------------------------
    # PHASE 8: METRICS, MODEL STATS, DIAGNOSTIC PLOTS & EXPORT
    # ------------------------------------------------------------------
    def _phase8_metrics_export(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 8: METRICS, DIAGNOSTICS & EXPORT")
        self.log("=" * 70)

        cfg = self.cfg
        os.makedirs(cfg.export_dir, exist_ok=True)

        # --- Save model & scalers ---
        self.model.save(os.path.join(cfg.export_dir, "final_model.keras"))
        try:
            self.model.save(os.path.join(cfg.export_dir, "final_model.h5"))
        except Exception:
            pass
        joblib.dump(self.scaler_X, os.path.join(cfg.export_dir, "scaler_X.pkl"))
        joblib.dump(self.scaler_y, os.path.join(cfg.export_dir, "scaler_y.pkl"))
        self.log(f"Model and scalers saved to {cfg.export_dir}/")

        # --- Predictions (inverse scaled) ---
        y_train_pred = self.scaler_y.inverse_transform(
            self.model.predict(self.X_train, verbose=0))
        y_val_pred = self.scaler_y.inverse_transform(
            self.model.predict(self.X_val, verbose=0))
        y_test_pred = self.scaler_y.inverse_transform(
            self.model.predict(self.X_test, verbose=0))

        y_train_inv = self.scaler_y.inverse_transform(self.y_train)
        y_val_inv = self.scaler_y.inverse_transform(self.y_val)
        y_test_inv = self.scaler_y.inverse_transform(self.y_test)

        # --- Compute metrics ---
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
        for label, m in [(f"Training set ({cfg.train_percent}%)", self.metrics_train),
                         (f"Validation set ({cfg.val_percent}%)", self.metrics_val),
                         (f"Test set ({cfg.test_percent}%)", self.metrics_test)]:
            self.log(f"\n{label}:")
            for k, v in m.items():
                self.log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        # --- Save model_statistics.txt ---
        try:
            stats_path = os.path.join(cfg.export_dir, "model_statistics.txt")
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write("Model statistics summary\n")
                f.write("========================\n\n")
                f.write(f"Current date: {_dt.date.today().isoformat()}\n")
                f.write(f"Number of predictors (p): {p}\n")
                f.write(f"Data split: {cfg.train_percent}% train / "
                        f"{cfg.val_percent}% validation / {cfg.test_percent}% test\n\n")
                f.write("Hyperparameters (best):\n")
                for k, v in self.best_hp.values.items():
                    f.write(f"  {k}: {v}\n")
                for label, m in [(f"Training set metrics ({cfg.train_percent}%)", self.metrics_train),
                                 (f"Validation set metrics ({cfg.val_percent}%)", self.metrics_val),
                                 (f"Test set metrics ({cfg.test_percent}%)", self.metrics_test)]:
                    f.write(f"\n{label}:\n")
                    for k, v in m.items():
                        f.write(f"  {k}: {v}\n")
                f.write(f"\nPredicted R2 (Q2) on training (OOF aggregated, no leakage): {self.pred_R2}\n")
                f.write(f"Best epoch selected on VAL loss: {self.best_epoch}\n")
                f.write(f"Optional retrain on train+val: {cfg.do_optional_retrain}\n")
                f.write(f"PI calibration source: {cfg.pi_calibration}, alpha={cfg.pi_alpha}\n")
            self.log(f"Model statistics saved to: {stats_path}")
        except Exception as e:
            self.log(f"Could not save model statistics: {e}")

        # --- Save model_scheme.txt ---
        try:
            from tensorflow.keras.layers import Dense
            dense_layers = [layer for layer in self.model.layers if isinstance(layer, Dense)]
            if dense_layers:
                self.log("\n=== Final Dense Layers (in model order) ===")
                scheme_lines = []
                for i, layer in enumerate(dense_layers, start=1):
                    units = getattr(layer, "units", None)
                    activation = getattr(layer, "activation", None)
                    act_name = activation.__name__ if activation is not None else "N/A"
                    reg = getattr(layer, "kernel_regularizer", None)
                    reg_str = None
                    try:
                        if reg is not None and hasattr(reg, "l2"):
                            reg_str = f"L2={getattr(reg, 'l2')}"
                        elif reg is not None:
                            reg_str = str(reg)
                    except Exception:
                        reg_str = str(reg) if reg else None
                    line = f"Layer {i}: name='{layer.name}', units={units}, activation={act_name}"
                    if reg_str:
                        line += f", regularizer={reg_str}"
                    self.log(line)
                    scheme_lines.append(line)

                last = dense_layers[-1]
                final_line = (f"Final Dense (output) -> name='{last.name}', "
                              f"units={getattr(last, 'units', None)}, "
                              f"activation={getattr(last, 'activation').__name__ if getattr(last, 'activation', None) else 'N/A'}")
                self.log("\n" + final_line)
                scheme_lines.append("")
                scheme_lines.append(final_line)

                scheme_path = os.path.join(cfg.export_dir, "model_scheme.txt")
                with open(scheme_path, "w", encoding="utf-8") as f:
                    f.write("Final Dense Layers (in model order)\n")
                    f.write("===============================\n")
                    for ln in scheme_lines:
                        f.write(ln + "\n")
                self.log(f"Model scheme saved to: {scheme_path}")
        except Exception as e:
            self.log(f"Error saving model scheme: {e}")

        # --- MSE evolution plot ---
        try:
            mse_img_path = os.path.join(cfg.export_dir, "mse_evolution.png")
            if self.history is not None and hasattr(self.history, "history"):
                hist = self.history.history
                loss = hist.get('loss', None)
                val_loss = hist.get('val_loss', None)
                if loss is not None:
                    epochs = range(1, len(loss) + 1)
                    plt.figure(figsize=(8, 5))
                    plt.plot(epochs, loss, label='Train loss', marker='o', markersize=2)
                    if val_loss is not None:
                        plt.plot(epochs, val_loss, label='Validation loss', marker='o', markersize=2)
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss (MAE)')
                    plt.title('Evolution of loss during final training')
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(mse_img_path, dpi=150)
                    plt.close()
                    self.log(f"Loss evolution plot saved: {mse_img_path}")
        except Exception as e:
            self.log(f"Error plotting loss evolution: {e}")

        # --- Diagnostic plots ---
        try:
            plots_dir = os.path.join(cfg.export_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            plot_predicted_vs_actual(
                y_train_inv, y_train_pred, "Predicted vs Actual (Train)",
                save_path=os.path.join(plots_dir, "pred_vs_actual_train.png"))
            plot_residuals_std_and_timeseries(
                y_train_inv, y_train_pred, "Train",
                save_prefix=os.path.join(plots_dir, "train"))
            plot_residuals_raw_and_timeseries(
                y_train_inv, y_train_pred, "Train",
                save_prefix=os.path.join(plots_dir, "train"))

            plot_predicted_vs_actual(
                y_val_inv, y_val_pred, "Predicted vs Actual (Validation)",
                save_path=os.path.join(plots_dir, "pred_vs_actual_val.png"))
            plot_residuals_std_and_timeseries(
                y_val_inv, y_val_pred, "Validation",
                save_prefix=os.path.join(plots_dir, "val"))
            plot_residuals_raw_and_timeseries(
                y_val_inv, y_val_pred, "Validation",
                save_prefix=os.path.join(plots_dir, "val"))

            plot_predicted_vs_actual(
                y_test_inv, y_test_pred, "Predicted vs Actual (Test)",
                save_path=os.path.join(plots_dir, "pred_vs_actual_test.png"))
            plot_residuals_std_and_timeseries(
                y_test_inv, y_test_pred, "Test",
                save_prefix=os.path.join(plots_dir, "test"))
            plot_residuals_raw_and_timeseries(
                y_test_inv, y_test_pred, "Test",
                save_prefix=os.path.join(plots_dir, "test"))

            self.log(f"Diagnostic plots saved to: {plots_dir}")
        except Exception as e:
            self.log(f"Error creating diagnostic plots: {e}")
            self.log(traceback.format_exc())

        # --- Prediction Intervals ---
        self.log(f"\n=== Empirical Prediction Intervals "
                 f"({int((1 - cfg.pi_alpha) * 100)}% PI) ===")
        if cfg.pi_calibration == "oof":
            residuals_cal = self.y_oof_true_inv - self.y_oof_pred_inv
            cal_label = "OOF residuals (train CV)"
        else:
            residuals_cal = y_val_inv.reshape(-1) - y_val_pred.reshape(-1)
            cal_label = "val residuals"

        pi_lo_q = np.quantile(residuals_cal, cfg.pi_alpha / 2)
        pi_hi_q = np.quantile(residuals_cal, 1 - cfg.pi_alpha / 2)
        self.log(f"Calibrated on {cal_label}: q_low={pi_lo_q:.4f}, q_high={pi_hi_q:.4f}")

        # --- Export result DataFrames ---
        def make_export_df(labels_df_part, inputs_df_part, y_true, y_pred):
            actual = np.array(y_true).reshape(-1)
            pred = np.array(y_pred).reshape(-1)
            residual = actual - pred
            abs_err = np.abs(residual)
            with np.errstate(divide='ignore', invalid='ignore'):
                abs_pct = np.where(np.abs(actual) > 1e-12,
                                   100.0 * abs_err / np.abs(actual), np.nan)
            df_export = pd.DataFrame(index=range(len(actual)))
            if labels_df_part is not None and labels_df_part.shape[0] == len(actual):
                df_export = pd.concat(
                    [df_export, labels_df_part.reset_index(drop=True)], axis=1)
            df_export = pd.concat(
                [df_export, inputs_df_part.reset_index(drop=True)], axis=1)
            df_export["Actual_Value"] = actual
            df_export["Predicted_Value"] = pred
            df_export["Residual"] = residual
            df_export["Abs_Error"] = abs_err
            df_export["Abs_Percent_Error"] = abs_pct
            return df_export

        results_train = make_export_df(
            self.labels_train.reset_index(drop=True), self.X_train_df,
            y_train_inv, y_train_pred)
        results_val = make_export_df(
            self.labels_val.reset_index(drop=True), self.X_val_df,
            y_val_inv, y_val_pred)
        results_test = make_export_df(
            self.labels_test.reset_index(drop=True), self.X_test_df,
            y_test_inv, y_test_pred)

        # Save standard exports (xlsx + csv)
        for name, frame in [("train_predictions", results_train),
                            ("val_predictions", results_val),
                            ("test_predictions", results_test)]:
            frame.to_excel(os.path.join(cfg.export_dir, f"{name}.xlsx"),
                           index=False, engine='openpyxl')
            frame.to_csv(os.path.join(cfg.export_dir, f"{name}.csv"), index=False)

        self.log(f"Results exported (xlsx + csv) to {cfg.export_dir}/")

        # --- Full-row exports ---
        try:
            def _add_pred_cols(df_full, y_true_flat, y_pred_flat):
                df_full = df_full.copy()
                residual = y_true_flat - y_pred_flat
                abs_err = np.abs(residual)
                with np.errstate(divide='ignore', invalid='ignore'):
                    abs_pct = np.where(np.abs(y_true_flat) > 1e-12,
                                       100.0 * abs_err / np.abs(y_true_flat), np.nan)
                df_full["Actual_Value"] = y_true_flat
                df_full["Predicted_Value"] = y_pred_flat
                df_full["Residual"] = residual
                df_full["Abs_Error"] = abs_err
                df_full["Abs_Percent_Error"] = abs_pct
                pred_cols = ["Actual_Value", "Predicted_Value", "Residual",
                             "Abs_Error", "Abs_Percent_Error"]
                cols = [c for c in df_full.columns if c not in pred_cols] + pred_cols
                return df_full[cols]

            train_full = _add_pred_cols(
                self.df.iloc[self.idx_train].reset_index(drop=True),
                np.array(y_train_inv).reshape(-1),
                np.array(y_train_pred).reshape(-1))
            val_full = _add_pred_cols(
                self.df.iloc[self.idx_val].reset_index(drop=True),
                np.array(y_val_inv).reshape(-1),
                np.array(y_val_pred).reshape(-1))
            test_full = _add_pred_cols(
                self.df.iloc[self.idx_test].reset_index(drop=True),
                np.array(y_test_inv).reshape(-1),
                np.array(y_test_pred).reshape(-1))

            for name, frame in [("train_full_with_all_columns", train_full),
                                ("val_full_with_all_columns", val_full),
                                ("test_full_with_all_columns", test_full)]:
                frame.to_excel(os.path.join(cfg.export_dir, f"{name}.xlsx"),
                               index=False, engine='openpyxl')
                frame.to_csv(os.path.join(cfg.export_dir, f"{name}.csv"), index=False)
            self.log("Full-row exports saved (xlsx + csv)")
        except Exception as e:
            self.log(f"Error creating full-row exports: {e}")
            self.log(traceback.format_exc())

        self.progress(88)

    # ------------------------------------------------------------------
    # PHASE 9: 3D SURFACE PLOTS (ALL FEATURE PAIRS)
    # ------------------------------------------------------------------
    def _phase9_plots(self):
        self.log("\n" + "=" * 70)
        self.log("PHASE 9: 3D SURFACE PLOTS")
        self.log("=" * 70)

        cfg = self.cfg
        self.plot_paths = []

        def _safe_name(s):
            s = str(s)
            for ch in [" ", "/", "\\", ":", ";", "|", "(", ")", "[", "]", "{", "}", "%"]:
                s = s.replace(ch, "_")
            return s

        def _grid_vals(col_idx):
            v = self.X_train_orig[:, col_idx]
            if cfg.range_mode == "minmax":
                lo, hi = float(np.min(v)), float(np.max(v))
            elif cfg.range_mode in ("quantile", "train_quantile"):
                lo, hi = float(np.quantile(v, cfg.q_low)), float(np.quantile(v, cfg.q_high))
            else:
                lo, hi = float(np.min(v)), float(np.max(v))
            if np.isclose(lo, hi):
                lo, hi = lo - 1.0, hi + 1.0
            return np.linspace(lo, hi, cfg.grid_n_export)

        def predict_from_origX(X_orig_2d):
            X_scaled = self.scaler_X.transform(X_orig_2d)
            y_scaled = self.model.predict(X_scaled, verbose=0)
            return self.scaler_y.inverse_transform(y_scaled).reshape(-1)

        # Build reference vector
        if cfg.hold_mode == "median_train":
            Xref = np.median(self.X_train_orig, axis=0)
        elif cfg.hold_mode == "mean_train":
            Xref = np.mean(self.X_train_orig, axis=0)
        elif cfg.hold_mode == "row":
            Xref = self.X_train_orig[cfg.row_index].copy()
        else:
            Xref = np.median(self.X_train_orig, axis=0)

        try:
            plots_dir_3d = os.path.join(cfg.export_dir, "plots_3d")
            os.makedirs(plots_dir_3d, exist_ok=True)

            p3d = self.X_train_orig.shape[1]
            pairs_3d = list(combinations(range(p3d), 2))
            self.log(f"\nGenerating {len(pairs_3d)} 3D surfaces into: {plots_dir_3d}")

            for idx, (i, j) in enumerate(pairs_3d):
                if self.cancelled():
                    return

                xi = _grid_vals(i)
                xj = _grid_vals(j)
                XI, XJ = np.meshgrid(xi, xj)

                Xgrid = np.tile(Xref.reshape(1, -1), (XI.size, 1))
                Xgrid[:, i] = XI.reshape(-1)
                Xgrid[:, j] = XJ.reshape(-1)

                Z = predict_from_origX(Xgrid).reshape(XI.shape)

                fig = plt.figure(figsize=(9, 7))
                ax = fig.add_subplot(111, projection="3d")
                surf = ax.plot_surface(XI, XJ, Z, cmap="viridis", linewidth=0,
                                       antialiased=True, alpha=0.95)
                fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Predicted output")

                col_i = self.inputs_columns[i]
                col_j = self.inputs_columns[j]
                ax.set_title(f"Predicted surface: {col_i} vs {col_j}\n"
                             f"(others held constant: {cfg.hold_mode})")
                ax.set_xlabel(str(col_i))
                ax.set_ylabel(str(col_j))
                ax.set_zlabel("Predicted output")
                ax.view_init(elev=25, azim=-135)
                plt.tight_layout()

                out_path = os.path.join(
                    plots_dir_3d,
                    f"surface_{_safe_name(col_i)}_vs_{_safe_name(col_j)}_hold_{cfg.hold_mode}.png"
                )
                plt.savefig(out_path, dpi=cfg.dpi)
                plt.close(fig)
                self.plot_paths.append(out_path)

            self.log(f"3D surfaces saved. Count: {len(self.plot_paths)}")
        except Exception as e:
            self.log(f"Error generating 3D surfaces: {e}")
            self.log(traceback.format_exc())

        self.progress(100)
