"""Shared helpers: config loading, seeding, device selection, repo paths."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(*names: str) -> dict:
    """Load configs/base.yaml, then merge each named config (e.g. "client_a") on top."""
    cfg: dict = {}
    for name in ("base", *names):
        path = REPO_ROOT / "configs" / f"{name}.yaml"
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    return cfg


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(pref: str = "auto", debug: bool = False):
    """--debug always runs on CPU; otherwise cuda > mps > cpu."""
    import torch

    if debug or pref == "cpu":
        return torch.device("cpu")
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
