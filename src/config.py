"""Utilities for loading and resolving pipeline configuration."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    """Container for pipeline configuration with resolved paths."""

    raw: Dict[str, Any]
    path: Path

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def _resolve_path(base: Path, value: Any, substitutions: Dict[str, Any]) -> Path:
    value_str = str(value)
    substituted = value_str.format(**substitutions)
    candidate = Path(substituted)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load a YAML config file and resolve any formatted paths."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    base_dir = config_file.parent
    paths = data.setdefault("paths", {})

    data_dir = paths.get("data_dir", "data")
    data_dir_path = _resolve_path(base_dir, data_dir, {"data_dir": data_dir})

    substitutions = {"data_dir": str(data_dir_path)}

    cache_dir = paths.get("cache_dir", "{data_dir}/cached_tensors")
    model_checkpoint = paths.get("model_checkpoint", "data/saved_model/model_v1.pth")
    results_dir = paths.get("results_dir", "results_pipeline")

    paths["data_dir"] = data_dir_path
    paths["cache_dir"] = _resolve_path(base_dir, cache_dir, substitutions)
    paths["model_checkpoint"] = _resolve_path(base_dir, model_checkpoint, substitutions)
    paths["results_dir"] = _resolve_path(base_dir, results_dir, substitutions)

    datasets = data.setdefault("datasets", {})
    for key, rel_path in list(datasets.items()):
        datasets[key] = _resolve_path(paths["data_dir"], rel_path, substitutions)

    logging.debug("Loaded configuration: %s", data)
    return PipelineConfig(raw=data, path=config_file)
