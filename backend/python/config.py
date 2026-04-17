"""
ML configuration and hyperparameter defaults for the ANN-SIMS pipeline.
"""

import os

# Default training hyperparameters
DEFAULT_EPOCHS = 100
DEFAULT_FOLDS = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_UNITS = 64
DEFAULT_LAYERS = 3
DEFAULT_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 20
DEFAULT_VALIDATION_SPLIT = 0.15

# Keras Tuner search space
TUNER_MAX_TRIALS = 20
TUNER_EXECUTIONS_PER_TRIAL = 1
TUNER_MAX_UNITS = 256
TUNER_MIN_UNITS = 16
TUNER_MAX_LAYERS = 5
TUNER_MIN_LAYERS = 1
TUNER_LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3]
TUNER_DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3]

# Pre-training (Hill function)
HILL_N_SAMPLES = 2000
HILL_N_VALUES = [1.0, 2.0, 3.0, 4.0]
HILL_K_RANGE = (0.1, 10.0)
HILL_X_RANGE = (0.0, 20.0)
HILL_NOISE_STD = 0.02
HILL_PRETRAIN_EPOCHS = 50
HILL_PRETRAIN_UNITS = 64
HILL_PRETRAIN_LAYERS = 3

# Prediction intervals
PI_CONFIDENCE = 0.95
PI_Z_SCORE = 1.96
BOOTSTRAP_N_ITERATIONS = 100

# Random seed for reproducibility
RANDOM_SEED = 42

# Validation split thresholds
MIN_VALIDATION_SPLIT = 0.1          # Minimum validation fraction for final model training
MIN_VALIDATION_SAMPLES = 50         # Below this, use MIN_VALIDATION_SPLIT unconditionally
LARGE_DATASET_THRESHOLD = 100       # Samples threshold for adaptive validation split
FALLBACK_UNCERTAINTY_FACTOR = 0.05  # Fraction of prediction range used as PI when no dropout

# Maximum number of 3D surface plots to generate (limits pairs of features)
MAX_3D_PLOTS = 6

# Log history limit (max recent log entries returned per status call)
MAX_LOGS_RETURNED = 50

# File names
MODEL_FILENAME = "best_model.h5"
SAVEDMODEL_DIR = "savedmodel"          # TF SavedModel sub-directory (version 1)
SCALER_X_FILENAME = "scaler_X.pkl"
SCALER_Y_FILENAME = "scaler_y.pkl"
RESULTS_FILENAME = "results.json"
PREDICTIONS_FILENAME = "predictions.json"
PRETRAIN_WEIGHTS_FILENAME = "pretrain_weights.h5"

# TensorFlow Serving
TF_SERVING_URL = os.environ.get("TF_SERVING_URL", "http://localhost:8501")
TF_SERVING_TIMEOUT = 10  # seconds
