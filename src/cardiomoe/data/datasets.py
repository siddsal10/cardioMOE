"""PyTorch dataset over the prepared PTB-XL signals.

A sample is a dict with "key" (ecg_id), "y" (5 labels) and "signal" [12, 1000]
(per-lead z-scored; Section 5.4 augmentations when train=True).
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from cardiomoe.data import ptbxl
from cardiomoe.data.labels import SUPERCLASSES
from cardiomoe.data.ptbxl import processed_dir
from cardiomoe.data.signal import augment as augment_signal
from cardiomoe.data.signal import normalize

YCOLS = [f"y_{c}" for c in SUPERCLASSES]


@lru_cache(maxsize=None)
def _meta(debug: bool, cfg_key: str) -> pd.DataFrame:
    return ptbxl.load_meta(json.loads(cfg_key), debug)


def load_meta(cfg: dict, debug: bool = False) -> pd.DataFrame:
    return _meta(debug, json.dumps({"paths": cfg["paths"]}))


def keys_for(cfg: dict, folds: list[int], debug: bool = False) -> list:
    """ecg_ids of the given PTB-XL folds."""
    m = load_meta(cfg, debug)
    return m.index[m.strat_fold.isin(folds)].tolist()


class SignalDataset(Dataset):
    def __init__(self, cfg: dict, keys: list, *, train: bool = False, debug: bool = False):
        self.keys, self.train = list(keys), train
        meta = load_meta(cfg, debug).loc[self.keys]
        self.y = np.array(meta[YCOLS].to_numpy(np.float32))
        self.rows = meta["row"].to_numpy()
        self.signals_path = processed_dir(cfg, debug) / "ptbxl_signals.npy"
        self._signals = None
        self._rng, self._rng_pid = None, None

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def signals(self):
        if self._signals is None:  # opened lazily so each DataLoader worker maps its own copy
            self._signals = np.load(self.signals_path, mmap_mode="r")
        return self._signals

    @property
    def rng(self) -> np.random.Generator:
        """Augmentation RNG, created per process from torch's (seeded) per-worker seed."""
        import os

        if self._rng_pid != os.getpid():
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._rng_pid = os.getpid()
        return self._rng

    def __getitem__(self, i: int) -> dict:
        sig = normalize(np.asarray(self.signals[self.rows[i]], dtype=np.float32))
        if self.train:
            rng = self.rng
            sig = augment_signal(sig, rng)
            rng.random()  # draw kept so augmentation streams match runs made before the cleanup
        return {"key": self.keys[i], "y": torch.from_numpy(self.y[i]),
                "signal": torch.from_numpy(np.ascontiguousarray(sig))}


def collate(batch: list[dict]) -> dict:
    return {"key": [b["key"] for b in batch], "y": torch.stack([b["y"] for b in batch]),
            "signal": torch.stack([b["signal"] for b in batch])}


def make_loader(ds: SignalDataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate,
                      persistent_workers=num_workers > 0, drop_last=shuffle and len(ds) > batch_size)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
