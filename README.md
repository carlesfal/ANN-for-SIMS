# ANNSIMS - ANN for SIMS Prediction

Desktop application for training Artificial Neural Networks to predict SIMS (Secondary Ion Mass Spectrometry) Instrumental Mass Fractionation.

## Features

- **Hill Pre-training**: Warm-start the network using synthetic Hill-equation data
- **Hyperparameter Tuning**: Automated search via Keras Tuner (RandomSearch)
- **Smart Weight Transfer**: Layer-by-layer weight transfer with partial/slice matching
- **K-Fold Cross-Validation**: Strict no-leakage CV with per-fold scaling
- **Final Training**: Best-fold seeding with optional retrain on Train+Val
- **Prediction Intervals**: Empirical calibration (val or OOF residuals)
- **3D Surface Plots**: Exported as high-DPI PNG for all feature pairs
- **Desktop GUI**: Configure all parameters, load data, run training, and view results

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Requirements

- **Python 3.9–3.12** (TensorFlow does not yet support Python 3.13+)
- TensorFlow 2.10+
- Tkinter (included with most Python installations)

> **Tip:** If your system Python is 3.13+, create a virtual environment with a compatible version:
> ```bash
> # Using conda:
> conda create -n annsims python=3.11 -y && conda activate annsims
> # Or using pyenv:
> pyenv install 3.11.9 && pyenv local 3.11.9
> ```

## Usage

### Launch the Desktop App

```bash
python run_app.py
```

### Workflow

1. **Configuration Tab**: Adjust all pipeline parameters (Hill pre-training, data split, training epochs, weight transfer mode, plot settings, etc.)
2. **Data Tab**: Browse and load your data file (TSV, CSV, or Excel). Preview the data before training.
3. **Training Tab**: Click "Run Pipeline" to execute all phases. Monitor progress via the log and progress bar.
4. **Results Tab**: View final metrics (R2, RMSE, MAE, prediction intervals) and open the export directory.

### Supported Data Formats

- Tab-separated values (`.tsv`)
- Comma-separated values (`.csv`)
- Excel files (`.xlsx`, `.xls`)

### Output

All results are saved to the configured export directory (default: `optimized_model/`):

- `final_model.keras` / `final_model.h5` - Trained model
- `scaler_X.pkl`, `scaler_y.pkl` - Fitted scalers
- `model_statistics.txt` - Hyperparameters and metrics summary
- `model_scheme.txt` - Network architecture documentation
- `mse_evolution.png` - Training loss evolution plot
- `{train,val,test}_predictions.xlsx` / `.csv` - Predictions with prediction intervals
- `{train,val,test}_full_with_all_columns.xlsx` / `.csv` - Full-row exports
- `plots/` - Diagnostic plots (pred vs actual, residuals)
- `plots_3d/` - 3D surface plots as PNG files

## Pipeline Phases

1. **Phase 1**: Hill pre-training (synthetic data warm-up)
2. **Phase 2**: Data loading and train/val/test splitting
3. **Phase 3**: Hyperparameter tuning (Keras Tuner)
4. **Phase 4**: Weight transfer from warmup model
5. **Phase 5**: K-Fold cross-validation
6. **Phase 6**: Final training with best-fold seeding
7. **Phase 7**: Optional retrain on Train+Val combined
8. **Phase 8**: Metrics computation, prediction intervals, export
9. **Phase 9**: 3D surface plot generation
