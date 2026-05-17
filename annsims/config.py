"""User-facing configuration for the ANN regression pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    """Top-level configuration for the ANN regression pipeline.

    Adjust these values before running.  The three split percentages
    must sum to 100 and are validated automatically.
    """

    # --- Data split (must sum to 100) ---
    train_percent: int = 60
    val_percent: int = 20
    test_percent: int = 20

    # --- Column layout ---
    n_labels: int = 7
    n_inputs: int = 9
    target_col: Optional[int] = None
    sep: str = "\t"

    # --- Hardware ---
    disable_gpu: bool = True

    # --- Tuning / training ---
    tuner_trials: int = 15
    k_folds: int = 15
    random_seed: int = 42
    tuner_epochs: int = 200
    cv_epochs: int = 200
    final_epochs: int = 200

    # --- Optional retrain on train+val ---
    do_optional_retrain: bool = True

    # --- Prediction interval ---
    pi_calibration: str = "val"  # "val" or "oof"
    pi_alpha: float = 0.05      # 95 % PI ⇒ alpha = 0.05

    # --- Export ---
    export_dir: str = "optimized_model"

    def __post_init__(self) -> None:
        total = self.train_percent + self.val_percent + self.test_percent
        if abs(total - 100) > 0.01:
            raise ValueError(
                f"Split percentages must sum to 100. "
                f"Got {self.train_percent}+{self.val_percent}+{self.test_percent}={total}"
            )
        if self.pi_calibration not in ("val", "oof"):
            raise ValueError(
                f"pi_calibration must be 'val' or 'oof', got '{self.pi_calibration}'"
            )
        if not 0 < self.pi_alpha < 1:
            raise ValueError(f"pi_alpha must be in (0, 1), got {self.pi_alpha}")

    @property
    def test_frac(self) -> float:
        return self.test_percent / 100.0

    @property
    def val_frac(self) -> float:
        return self.val_percent / 100.0

    @property
    def train_frac(self) -> float:
        return self.train_percent / 100.0


@dataclass
class SurfacePlotConfig:
    """Configuration for 3D predicted-surface plots."""

    grid_n: int = 35
    range_mode: str = "quantile"  # "quantile" or "minmax"
    q_low: float = 0.02
    q_high: float = 0.98
    hold_mode: str = "median_train"  # "median_train", "mean_train", "row"
    row_index: int = 0
    overlay_train_scatter: bool = True
    scatter_alpha: float = 0.25
    scatter_size: float = 8
    max_pairs: int = 12
    dpi: int = 160
    features_to_use: Optional[list[str]] = None
