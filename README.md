# ANN-for-SIMS
Artificial Neural Network for SIMS Prediction - Web Application

## Pipeline Overview

This project implements a two-stage ANN regression pipeline for SIMS
(Secondary Ion Mass Spectrometry) Instrumental Mass Fractionation prediction:

### Stage 1: Hill Pre-Training (`hill_pretrain_pipeline.py`)

Generates synthetic data from a Hill equation and pre-trains a neural network
to learn sigmoidal priors. The trained model (`warmup_model`) is exposed to
the downstream training cell for transfer learning.

**Phases:**
1. **Hyperparameter search** (Keras Tuner) with a variable architecture
   (2-6 layers, 64-512 units).
2. **Pre-train** the best architecture on synthetic Hill-function data.
3. **Expose** `warmup_model` in the caller's globals for the training cell.

### Stage 2: Training Pipeline (`ann_training_pipeline.py`)

Strict no-leakage ANN regression pipeline with frozen transfer learning
support.

**Transfer Learning (frozen warm-up):**
When a `warmup_model` is available (from Stage 1 or loaded from disk), its
layers are **frozen** and used as a feature extractor. A configurable number
of new trainable layers are stacked on top and trained on the real SIMS data.

Configure the top layers in `PipelineConfig`:
```python
cfg = PipelineConfig(
    transfer_top_units=[128, 64],   # 2 layers: 128 then 64 units
    transfer_top_dropout=0.2,
    transfer_top_l2=1e-3,
    transfer_top_lr=1e-3,
)
```
The length of `transfer_top_units` controls the number of layers.

**Pipeline steps:**
1. Data loading and column selection
2. Train / Validation / Test split (configurable percentages)
3. Scaling (fit on TRAIN only - no leakage)
4. Keras Tuner hyperparameter search
5. Frozen transfer learning (when warm-up model available)
6. K-fold cross-validation with fold-fitted scalers
7. Final training with early stopping
8. Optional retrain on TRAIN+VAL
9. Diagnostic plots (pred vs actual, residuals, 3D surfaces)
10. New data predictions with prediction intervals

## Usage

### Colab / Jupyter

```python
# Cell 1: Run Hill pre-training
from hill_pretrain_pipeline import main as hill_main
hill_main()

# Cell 2: Run training pipeline (picks up warmup_model from globals)
%run ann_training_pipeline.py
```

### CLI

```bash
# Stage 1
python hill_pretrain_pipeline.py --trials 20 --pretrain-epochs 80

# Stage 2 (load warm-up from disk)
# Set warmup_model_path in PipelineConfig or place data.tsv in working dir
python ann_training_pipeline.py
```

## Requirements

- Python 3.9+
- TensorFlow / Keras
- keras-tuner
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- openpyxl, joblib
