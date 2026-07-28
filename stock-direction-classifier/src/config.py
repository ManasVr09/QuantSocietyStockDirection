from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "configs" / "project.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config

