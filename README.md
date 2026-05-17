# ANN-for-SIMS

Artificial Neural Network for SIMS Prediction — a modular Python pipeline for
ANN-based regression with configurable train/val/test splitting, hyperparameter
tuning, cross-validation, and calibrated prediction intervals.

## Project Structure

```
annsims/
├── __init__.py      # Package metadata
├── __main__.py      # python -m annsims entry point
├── config.py        # Dataclass-based configuration with validation
├── data.py          # Data loading, column selection, splitting, scaling
├── model.py         # Model building, tuning, CV, training, weight transfer
├── metrics.py       # Metric computation and statistics export
├── plotting.py      # Diagnostic plots, MSE evolution, 3D surfaces
├── export.py        # DataFrame assembly, Excel/CSV/ZIP export
├── predict.py       # New-data prediction with prediction intervals
├── utils.py         # Environment detection and display helpers
└── main.py          # Pipeline orchestration
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with default config

Place your data file as `data.tsv` in the working directory and run:

```bash
python -m annsims
```

### 3. Customise via Python

```python
from annsims.config import PipelineConfig, SurfacePlotConfig
from annsims.main import run_pipeline

cfg = PipelineConfig(
    train_percent=70,
    val_percent=15,
    test_percent=15,
    n_labels=7,
    n_inputs=9,
    tuner_trials=20,
    k_folds=10,
)

run_pipeline(cfg)
```

### 4. Google Colab

The pipeline auto-detects Colab. Run the snippet above in a cell and a file
upload dialog will appear. Results are saved to the `optimized_model/` directory
and packaged into downloadable ZIP archives.

## Configuration

All options live in `PipelineConfig` (see `annsims/config.py`):

| Parameter            | Default | Description                                  |
|----------------------|---------|----------------------------------------------|
| `train_percent`      | 60      | Training split percentage                    |
| `val_percent`        | 20      | Validation split percentage                  |
| `test_percent`       | 20      | Test split percentage                        |
| `n_labels`           | 7       | Number of label (non-input) columns          |
| `n_inputs`           | 9       | Number of input feature columns              |
| `target_col`         | None    | Absolute column index of target (auto if None)|
| `tuner_trials`       | 15      | Number of Keras-Tuner random search trials   |
| `k_folds`            | 15      | Number of CV folds                           |
| `do_optional_retrain`| True    | Retrain on train+val after best epoch found  |
| `pi_calibration`     | "val"   | PI calibration source: "val" or "oof"        |
| `pi_alpha`           | 0.05    | Significance level for 95% PI                |

3D surface plots are configured separately via `SurfacePlotConfig`.

## Outputs

After a successful run the `optimized_model/` directory contains:

- `final_model.keras` / `final_model.h5` — trained model
- `scaler_X.pkl` / `scaler_y.pkl` — fitted scalers
- `model_statistics.txt` — full metrics summary
- `model_scheme.txt` — Dense layer architecture
- `train/val/test_predictions.xlsx/.csv` — per-split predictions
- `plots/` — diagnostic scatter and residual plots
- `plots_3d/` — 3D predicted-surface plots
- `training_results.zip` — all of the above bundled for download
