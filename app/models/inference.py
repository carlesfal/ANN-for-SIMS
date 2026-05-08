from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

from app.models.ann_model import ANN
from app.utils.file_io import load_model_checkpoint


def load_ann_model(path: str, device: str = "cpu") -> ANN:
    checkpoint = load_model_checkpoint(path, device=device)
    model_config = checkpoint["model_config"]

    model = ANN(
        input_dim=int(model_config["input_dim"]),
        output_dim=int(model_config["output_dim"]),
        hidden_layers=list(model_config["hidden_layers"]),
        activation=model_config["activation"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def predict(model: ANN, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor_x = torch.tensor(x, dtype=torch.float32)
        preds = model(tensor_x).cpu().numpy()
    return preds.squeeze()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
