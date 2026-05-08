from __future__ import annotations

from typing import Literal

import torch
from torch import nn

ActivationName = Literal["relu", "sigmoid", "tanh"]


def _activation(name: ActivationName) -> nn.Module:
    mapping: dict[ActivationName, nn.Module] = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }
    return mapping[name]


class ANN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(_activation(activation))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
