"""Evaluation metrics (Section 8.1). Labels may contain -1 (unknown); those entries are ignored.

All functions take y [N, C] in {0, 1, -1} and probabilities p [N, C].
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from cardiomoe.data.labels import SUPERCLASSES


def _cols(y: np.ndarray, p: np.ndarray, c: int):
    k = y[:, c] >= 0
    return y[k, c], p[k, c]


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        yc, pc = _cols(y, p, c)
        if len(np.unique(yc)) == 2:
            out[c] = roc_auc_score(yc, pc)
    return out


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    a = per_class_auroc(y, p)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def macro_auprc(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for c in range(y.shape[1]):
        yc, pc = _cols(y, p, c)
        if len(np.unique(yc)) == 2:
            vals.append(average_precision_score(yc, pc))
    return float(np.mean(vals)) if vals else float("nan")


def tune_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-class threshold maximising F1 (tuned on fold 9, then frozen)."""
    th = np.full(y.shape[1], 0.5)
    grid = np.linspace(0.02, 0.98, 49)
    for c in range(y.shape[1]):
        yc, pc = _cols(y, p, c)
        if yc.sum() == 0:
            continue
        f1s = [f1_score(yc, pc >= t, zero_division=0) for t in grid]
        th[c] = grid[int(np.argmax(f1s))]
    return th


def macro_f1(y: np.ndarray, p: np.ndarray, th: np.ndarray) -> float:
    vals = []
    for c in range(y.shape[1]):
        yc, pc = _cols(y, p, c)
        if len(yc) and yc.sum() > 0:
            vals.append(f1_score(yc, pc >= th[c], zero_division=0))
    return float(np.mean(vals)) if vals else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    k = y >= 0
    return float(np.mean((p[k] - y[k]) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error over all known (sample, class) probabilities."""
    k = y >= 0
    yy, pp = y[k].ravel(), p[k].ravel()
    bins = np.minimum((pp * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        s = bins == b
        if s.any():
            e += s.mean() * abs(pp[s].mean() - yy[s].mean())
    return float(e)


def summary(y: np.ndarray, p: np.ndarray, th: np.ndarray | None = None) -> dict:
    a = per_class_auroc(y, p)
    out = {"macro_auroc": float(np.nanmean(a)), "macro_auprc": macro_auprc(y, p),
           "brier": brier(y, p), "ece": ece(y, p)}
    out.update({f"auroc_{c}": float(v) for c, v in zip(SUPERCLASSES, a)})
    if th is not None:
        out["macro_f1"] = macro_f1(y, p, th)
    return out


def bootstrap_ci(y: np.ndarray, p: np.ndarray, fn=macro_auroc, n: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        v = fn(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --------------------------------------------------------------------------- thresholded
def exact_match(y: np.ndarray, p: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Per-sample correctness: every known class decision right."""
    pred = p >= th
    ok = (pred == (y == 1)) | (y < 0)
    return ok.all(axis=1)
