from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .features import FEATURE_COLUMNS


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def build_manifest(config: dict, champion: str, threshold: float, data_metadata: dict) -> dict:
    config_payload = {key: value for key, value in config.items() if not key.startswith("_")}
    fingerprint = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "schema_version": 1,
        "run_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{fingerprint}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_status": config["research_status"],
        "confirmatory": False,
        "code_commit": _git_commit(Path(config["_config_path"]).resolve().parents[2]),
        "config_fingerprint": fingerprint,
        "universe": config["tickers"],
        "market_ticker": config["market_ticker"],
        "feature_schema": FEATURE_COLUMNS,
        "execution_contract": config["execution"],
        "splits": config["splits"],
        "validation_declared_champion": champion,
        "validation_selected_threshold": threshold,
        "data_metadata": data_metadata,
        "warning": "Exploratory post-hoc universe. Do not use for portfolio action until remediated and re-audited.",
    }


def fold_assignments(splits) -> pd.DataFrame:
    frames = []
    for name in ("train", "validation", "test"):
        frame = getattr(splits, name)[["ticker", "feature_time", "target_date"]].copy()
        frame["fold"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["feature_time", "ticker"])
