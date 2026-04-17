"""
Utility to export an existing Keras .h5 model to TF SavedModel format
and generate / update a TensorFlow Serving model configuration file.

Usage:
    # Export a single model
    python export_savedmodel.py --model-dir backend/models/<jobId>

    # Regenerate the TF Serving config for all models in a directory
    python export_savedmodel.py --models-root backend/models --generate-config
"""

import argparse
import json
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg


def export_savedmodel(model_dir):
    """Convert <model_dir>/best_model.h5 → <model_dir>/savedmodel/1/ (SavedModel)."""
    h5_path = os.path.join(model_dir, cfg.MODEL_FILENAME)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"No .h5 model found at {h5_path}")

    model = tf.keras.models.load_model(h5_path)

    savedmodel_path = os.path.join(model_dir, cfg.SAVEDMODEL_DIR, "1")
    os.makedirs(savedmodel_path, exist_ok=True)
    model.save(savedmodel_path, save_format="tf")
    print(f"Exported SavedModel to {savedmodel_path}")
    return savedmodel_path


def generate_serving_config(models_root, config_path=None):
    """
    Scan *models_root* for sub-directories that contain a SavedModel export
    and write a TF Serving ``model_config_list`` protobuf text file.

    The generated config assumes TF Serving mounts *models_root* at
    ``/models`` inside the container.

    Returns the path to the written config file.
    """
    if config_path is None:
        config_path = os.path.join(models_root, "models.config")

    entries = []
    for name in sorted(os.listdir(models_root)):
        model_dir = os.path.join(models_root, name)
        savedmodel_dir = os.path.join(model_dir, cfg.SAVEDMODEL_DIR)
        if os.path.isdir(savedmodel_dir):
            # Use the directory name (jobId) as the model name
            entries.append(
                f'  config {{\n'
                f'    name: "{name}"\n'
                f'    base_path: "/models/{name}/{cfg.SAVEDMODEL_DIR}"\n'
                f'    model_platform: "tensorflow"\n'
                f'  }}'
            )

    config_text = "model_config_list {\n" + "\n".join(entries) + "\n}\n"

    with open(config_path, "w") as f:
        f.write(config_text)

    print(f"Wrote TF Serving config with {len(entries)} model(s) to {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Export models for TF Serving")
    parser.add_argument("--model-dir", help="Path to a single model directory to export")
    parser.add_argument("--models-root", help="Root directory containing all model directories")
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate TF Serving model_config_list for all exported models",
    )
    args = parser.parse_args()

    if args.model_dir:
        export_savedmodel(args.model_dir)

    if args.models_root and args.generate_config:
        generate_serving_config(args.models_root)

    if not args.model_dir and not (args.models_root and args.generate_config):
        parser.error("Provide --model-dir and/or --models-root with --generate-config")


if __name__ == "__main__":
    main()
