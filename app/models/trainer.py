from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.models.ann_model import ANN


@dataclass
class TrainingConfig:
    hidden_layers: list[int]
    activation: str
    optimizer: str
    learning_rate: float
    epochs: int
    batch_size: int


class Trainer:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

    def _build_optimizer(self, optimizer_name: str, params, learning_rate: float):
        optimizer_name = optimizer_name.lower()
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=learning_rate)
        if optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=learning_rate)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        config: TrainingConfig,
    ) -> tuple[ANN, dict[str, list[float]]]:
        input_dim = x_train.shape[1]
        output_dim = 1 if y_train.ndim == 1 else y_train.shape[1]

        model = ANN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=config.hidden_layers,
            activation=config.activation,  # type: ignore[arg-type]
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = self._build_optimizer(config.optimizer, model.parameters(), config.learning_rate)

        train_ds = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train.reshape(-1, output_dim), dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}
        best_state = model.state_dict()
        best_val_loss = float("inf")

        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.reshape(-1, output_dim), dtype=torch.float32).to(self.device)

        for _epoch in range(config.epochs):
            model.train()
            batch_losses: list[float] = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_tensor)
                val_loss = float(criterion(val_pred, y_val_tensor).item())

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        return model, history
