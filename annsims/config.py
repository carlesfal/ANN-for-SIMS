"""Configuration dataclass for ANNSIMS pipeline parameters."""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """All user-configurable parameters for the ANN training pipeline."""

    # --- Hill pre-training ---
    n_inputs: int = 10
    n_synthetic: int = 500
    hill_v_max: float = 200.0
    hill_k: float = 0.10
    hill_n: float = 1.20
    hill_x_min: float = 0.1
    hill_x_max: float = 100.0
    pretrain_epochs: int = 200

    # --- Data split ---
    train_percent: int = 60
    val_percent: int = 20
    test_percent: int = 20

    n_labels: int = 7
    target_col: int = -1  # -1 means None (auto-detect)
    sep: str = "\t"

    # --- Training ---
    disable_gpu: bool = True
    tuner_trials: int = 15
    k_folds: int = 15
    random_seed: int = 42
    tuner_epochs: int = 200
    cv_epochs: int = 200
    final_epochs: int = 200
    do_optional_retrain: bool = True

    # --- Weight Transfer ---
    transfer_mode: str = "smart"  # "smart" or "strict"
    transfer_best_fold: bool = True

    # --- Prediction intervals ---
    pi_calibration: str = "val"  # "val" or "oof"
    pi_alpha: float = 0.05

    # --- Output ---
    export_dir: str = "optimized_model"

    # --- Plot settings ---
    feature_x: str = "MnCO3"
    feature_y: str = "FeCO3"
    grid_n_single: int = 50
    range_mode: str = "quantile"
    q_low: float = 0.02
    q_high: float = 0.98
    hold_mode: str = "median_train"
    row_index: int = 0
    overlay_train_scatter: bool = True
    scatter_alpha: float = 0.25
    scatter_size: int = 10

    # Multi-pair settings
    max_pairs: int = 12
    grid_n_multi: int = 35
    grid_range_mode: str = "train_quantile"
    q_low_multi: float = 0.02
    q_high_multi: float = 0.98
    hold_mode_multi: str = "median_train"
    scatter_alpha_multi: float = 0.25
    scatter_size_multi: int = 8

    # Export settings
    grid_n_export: int = 30
    range_mode_export: str = "quantile"
    q_low_export: float = 0.02
    q_high_export: float = 0.98
    hold_mode_export: str = "median_train"
    row_index_export: int = 0
    dpi: int = 600
    fig_width_cm: float = 8.3

    def validate(self):
        """Validate that split percentages sum to 100."""
        total = self.train_percent + self.val_percent + self.test_percent
        if abs(total - 100) > 0.01:
            raise ValueError(
                f"Split percentages must sum to 100. "
                f"Got: {self.train_percent}+{self.val_percent}+{self.test_percent}={total}"
            )
        if self.transfer_mode not in ("smart", "strict"):
            raise ValueError("transfer_mode must be 'smart' or 'strict'")
        if self.pi_calibration not in ("val", "oof"):
            raise ValueError("pi_calibration must be 'val' or 'oof'")
        return True
