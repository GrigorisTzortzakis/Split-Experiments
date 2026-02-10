"""Experiment configuration loading/saving.

What this file is for:
- Load experiment settings from YAML/JSON into a `Config` object.
- Keep the config API stable (`yaml_config.load`, `json_config.load`).

The rest of the framework treats the returned config like a dict (`cfg["key"]`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


def _apply_mapping_to_config(config: "Config", mapping: Dict[str, Any]) -> None:
    for key, value in mapping.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.extra[key] = value


@dataclass
class Config:
    dataset: Any = ""
    model: str = ""
    dataDir: str = ""
    partition_method: str = ""
    partition_alpha: float = 0.5
    split_layer: int = 1
    server_rank: int = 0
    download: bool = True

    client_number: int = 2
    batch_size: int = 15
    lr: float = 0.001
    wd: float = 0.001
    epochs: int = 15
    comm_round: int = 10
    log_step: int = 20
    frequency_of_the_test: int = 1
    gpu_server_num: int = 1
    gpu_num_per_server: int = 1
    seed: int = -1
    partition_method_attributes: int = 9

    log_save_path: str = "./results/run.log"
    model_save_path: str = "./saved_progress/{}_{}_{}.pkl"
    model_tmp_path: str = "./saved_progress/client_tmp.pkl"

    save_acts_step: int = 0
    save_attack_acts_step: int = 0
    save_model_epoch: int = 0

    comm: Any = None
    process_id: Optional[int] = None
    rank: Optional[int] = None
    worker_number: Optional[int] = None
    max_rank: Optional[int] = None
    device: Any = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            return getattr(self, item)
        return self.extra.get(item)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def as_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__.keys() if k != "extra"}
        d.update(self.extra)
        return d


def load_yaml(path: str) -> Config:
    with open(path, "r") as f:
        mapping = yaml.load(f, Loader=yaml.FullLoader) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(mapping)}")

    cfg = Config()
    _apply_mapping_to_config(cfg, mapping)

    base_dir = Path(__file__).resolve().parents[1]

    # Normalize relative paths to the Split-Framework/ root.
    if cfg.log_save_path and not os.path.isabs(str(cfg.log_save_path)):
        cfg.log_save_path = str(base_dir / str(cfg.log_save_path))
    if cfg.model_save_path and not os.path.isabs(str(cfg.model_save_path)):
        cfg.model_save_path = str(base_dir / str(cfg.model_save_path))
    if cfg.model_tmp_path and not os.path.isabs(str(cfg.model_tmp_path)):
        cfg.model_tmp_path = str(base_dir / str(cfg.model_tmp_path))

    log_dir = os.path.dirname(str(cfg.log_save_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Create save dirs up-front so training doesn't crash when saving checkpoints.
    os.makedirs(base_dir / "saved_progress", exist_ok=True)
    os.makedirs(base_dir / "results", exist_ok=True)

    return cfg


def save_yaml(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.as_dict(), f)


def load_json(path: str) -> Config:
    with open(path, "r") as f:
        mapping = json.load(f) or {}
    if not isinstance(mapping, dict):
        raise ValueError(f"JSON config must be an object, got {type(mapping)}")

    cfg = Config()
    _apply_mapping_to_config(cfg, mapping)

    base_dir = Path(__file__).resolve().parents[1]

    # Normalize relative paths to the Split-Framework/ root.
    if cfg.log_save_path and not os.path.isabs(str(cfg.log_save_path)):
        cfg.log_save_path = str(base_dir / str(cfg.log_save_path))
    if cfg.model_save_path and not os.path.isabs(str(cfg.model_save_path)):
        cfg.model_save_path = str(base_dir / str(cfg.model_save_path))
    if cfg.model_tmp_path and not os.path.isabs(str(cfg.model_tmp_path)):
        cfg.model_tmp_path = str(base_dir / str(cfg.model_tmp_path))

    log_dir = os.path.dirname(str(cfg.log_save_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Create save dirs up-front so training doesn't crash when saving checkpoints.
    os.makedirs(base_dir / "saved_progress", exist_ok=True)
    os.makedirs(base_dir / "results", exist_ok=True)

    return cfg


def save_json(cfg: Config, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cfg.as_dict(), f, indent=2)


# Backwards-compatible module-like shims.
class yaml_config:  # noqa: N801
    load = staticmethod(load_yaml)
    save = staticmethod(save_yaml)


class json_config:  # noqa: N801
    load = staticmethod(load_json)
    save = staticmethod(save_json)
