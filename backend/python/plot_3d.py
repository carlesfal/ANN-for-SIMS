"""
3D surface plot generation for the ANN-SIMS system.

Generates matplotlib 3D surface plots for pairs of input features,
holding other features at their mean values.
"""

import os
import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(__file__))


def generate_3d_plots(model, scaler_X, scaler_y, X_data, y_data, feature_columns, target_column, output_dir, n_grid=30):
    """
    Generate 3D surface plots for all pairs of features.
    Other features are held at their mean values.

    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_features = len(feature_columns)
    plot_paths = []

    # Limit pairs to avoid too many plots
    feature_pairs = list(itertools.combinations(range(n_features), 2))
    max_plots = 6
    feature_pairs = feature_pairs[:max_plots]

    X_mean = np.mean(X_data, axis=0)
    X_min = np.min(X_data, axis=0)
    X_max = np.max(X_data, axis=0)

    for feat_i, feat_j in feature_pairs:
        fname_i = feature_columns[feat_i]
        fname_j = feature_columns[feat_j]

        xi = np.linspace(X_min[feat_i], X_max[feat_i], n_grid)
        xj = np.linspace(X_min[feat_j], X_max[feat_j], n_grid)
        Xi, Xj = np.meshgrid(xi, xj)

        # Build grid input
        n_pts = n_grid * n_grid
        X_grid = np.tile(X_mean, (n_pts, 1)).astype(np.float32)
        X_grid[:, feat_i] = Xi.flatten()
        X_grid[:, feat_j] = Xj.flatten()

        X_grid_scaled = scaler_X.transform(X_grid)
        y_pred_scaled = model.predict(X_grid_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        Z = y_pred.reshape(n_grid, n_grid)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(Xi, Xj, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=target_column)

        ax.set_xlabel(fname_i, fontsize=12, labelpad=10)
        ax.set_ylabel(fname_j, fontsize=12, labelpad=10)
        ax.set_zlabel(target_column, fontsize=12, labelpad=10)
        ax.set_title(f'ANN Prediction Surface\n{fname_i} vs {fname_j}', fontsize=13, fontweight='bold')

        ax.view_init(elev=25, azim=225)
        plt.tight_layout()

        safe_i = fname_i.replace(' ', '_').replace('/', '_')
        safe_j = fname_j.replace(' ', '_').replace('/', '_')
        filename = f'surface_{safe_i}_vs_{safe_j}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_paths.append(filepath)

    # Also generate a scatter plot: predicted vs actual
    if len(X_data) > 0 and y_data is not None:
        X_scaled = scaler_X.transform(X_data)
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = y_data.flatten()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue', label='Predictions')
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Fit')
        ax.set_xlabel(f'Actual {target_column}', fontsize=12)
        ax.set_ylabel(f'Predicted {target_column}', fontsize=12)
        ax.set_title('Predicted vs. Actual Values', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        scatter_path = os.path.join(output_dir, 'predicted_vs_actual.png')
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_paths.append(scatter_path)

    # Residuals plot
    if len(X_data) > 0 and y_data is not None:
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color='steelblue')
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs. Predicted', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residual', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        residuals_path = os.path.join(output_dir, 'residuals.png')
        plt.savefig(residuals_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        plot_paths.append(residuals_path)

    return plot_paths


if __name__ == '__main__':
    import argparse
    import json
    import tensorflow as tf
    import joblib
    import config as cfg

    parser = argparse.ArgumentParser(description='Generate 3D surface plots')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--data-file', help='Optional data file for scatter plots')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-column', default='target')
    args = parser.parse_args()

    model = tf.keras.models.load_model(os.path.join(args.model_dir, cfg.MODEL_FILENAME))
    scaler_X = joblib.load(os.path.join(args.model_dir, cfg.SCALER_X_FILENAME))
    scaler_y = joblib.load(os.path.join(args.model_dir, cfg.SCALER_Y_FILENAME))

    with open(os.path.join(args.model_dir, 'meta.json')) as f:
        meta = json.load(f)

    X_dummy = np.random.uniform(0, 1, (100, len(meta['feature_columns']))).astype(np.float32)
    y_dummy = None

    if args.data_file:
        from utils import load_data
        X_dummy, y_dummy, _ = load_data(args.data_file, args.target_column, meta.get('feature_columns'))

    paths = generate_3d_plots(model, scaler_X, scaler_y, X_dummy, y_dummy,
                               meta['feature_columns'], args.target_column, args.output_dir)
    print(json.dumps({'plots': [os.path.basename(p) for p in paths]}))
