from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

SUPPORTED_TABULAR_FORMATS = {".csv", ".txt", ".tsv", ".xlsx", ".xls"}


def load_tabular_data(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_TABULAR_FORMATS:
        raise ValueError(f"Unsupported file format: {suffix}")

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    return pd.read_excel(file_path)


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_model_checkpoint(
    path: str | Path,
    model_state_dict: dict[str, Any],
    model_config: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model_state_dict,
        "model_config": model_config,
        "metrics": metrics or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)
