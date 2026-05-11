# ANN-for-SIMS
Artificial Neural Network for SIMS Prediction - Web Application

## Files

| File | Description |
|---|---|
| `ann_regression.py` | Main Keras training script (Colab-compatible). Hidden layers use **LeakyReLU** (α=0.1), output uses **Sigmoid**. Includes hyperparameter tuning, K-fold CV, final training, optional retrain, diagnostics, and new-data prediction with calibrated prediction intervals. |
| `backprop_nn.py` | Pure NumPy backpropagation neural network module. Implements Dense layers (ReLU/linear), inverted Dropout, L2 regularisation, and the Adam optimiser from scratch. Can be used as an alternative to the Keras training loop. |

## Quick start

```bash
pip install keras-tuner seaborn openpyxl joblib tensorflow scikit-learn
# Place your data file as data.tsv, then:
python ann_regression.py
```

Or paste `ann_regression.py` into a Google Colab cell and run it directly.
