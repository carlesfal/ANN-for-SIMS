from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from app.models.trainer import Trainer, TrainingConfig
from app.utils.file_io import save_model_checkpoint


class ModelController:
    def __init__(self) -> None:
        self.trainer = Trainer(device="cpu")
        self.trained_model = None
        self.training_history: dict[str, list[float]] | None = None
        self.model_config: dict[str, object] | None = None

    def parse_hidden_layers(self, text: str) -> list[int]:
        values = [part.strip() for part in text.split(",") if part.strip()]
        if not values:
            raise ValueError("At least one hidden layer size is required")
        return [int(v) for v in values]

    def train_from_splits(self, splits: dict[str, pd.DataFrame | pd.Series], config: TrainingConfig):
        x_train = np.asarray(splits["x_train"], dtype=float)
        y_train = np.asarray(splits["y_train"], dtype=float)
        x_val = np.asarray(splits["x_val"], dtype=float)
        y_val = np.asarray(splits["y_val"], dtype=float)

        model, history = self.trainer.train(x_train, y_train, x_val, y_val, config)
        self.trained_model = model
        self.training_history = history
        self.model_config = {
            "input_dim": int(x_train.shape[1]),
            "output_dim": 1 if y_train.ndim == 1 else int(y_train.shape[1]),
            "hidden_layers": config.hidden_layers,
            "activation": config.activation,
            "training": asdict(config),
        }
        return model, history

    def save_model(self, path: str, best_val_loss: float | None = None) -> None:
        if self.trained_model is None or self.model_config is None:
            raise ValueError("No trained model available")
        save_model_checkpoint(
            path=path,
            model_state_dict=self.trained_model.state_dict(),
            model_config=self.model_config,
            metrics={"best_val_loss": best_val_loss or 0.0},
        )
