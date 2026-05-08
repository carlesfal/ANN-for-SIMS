from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    recent_files: list[str] = field(default_factory=list)
    last_model_path: str = ""
    normalization_method: str = "z-score"

    @staticmethod
    def default_path() -> Path:
        return Path.home() / ".annsims" / "config.json"


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or AppConfig.default_path()
    if not config_path.exists():
        return AppConfig()
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return AppConfig(**data)


def save_config(config: AppConfig, path: Path | None = None) -> None:
    config_path = path or AppConfig.default_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
