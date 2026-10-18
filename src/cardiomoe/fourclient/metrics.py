"""Thresholded metrics for the multi-label task, on top of eval/metrics.py.

Each superclass is a separate yes/no decision at a threshold tuned on fold 9, so every class
gets its own 2x2 confusion matrix. "Exact-match accuracy" counts a patient as correct only if
all 5 decisions are right.
"""
from __future__ import annotations

import numpy as np

from cardiomoe.data.labels import SUPERCLASSES
from cardiomoe.eval.metrics import exact_match, macro_auprc, per_class_auroc


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"tp": int(((pred == 1) & (y == 1)).sum()), "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()), "tn": int(((pred == 0) & (y == 0)).sum())}


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def class_metrics(cm: dict) -> dict:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"precision": prec, "recall": rec, "specificity": _div(tn, tn + fp),
            "f1": _div(2 * prec * rec, prec + rec) if tp else 0.0, "accuracy": _div(tp + tn, tp + fp + fn + tn)}


def full_report(y: np.ndarray, p: np.ndarray, th: np.ndarray) -> dict:
    """Macro metrics, per-class metrics and confusion matrices."""
    pred = (p >= th).astype(int)
    auc = per_class_auroc(y, p)
    per_class, cms = {}, {}
    for c, name in enumerate(SUPERCLASSES):
        cms[name] = confusion(y[:, c], pred[:, c])
        per_class[name] = {"auroc": float(auc[c]), **class_metrics(cms[name]), "threshold": float(th[c])}
    macro = {k: float(np.mean([per_class[n][k] for n in SUPERCLASSES]))
             for k in ("auroc", "precision", "recall", "specificity", "f1", "accuracy")}
    out = {f"macro_{k}": v for k, v in macro.items()}
    out["macro_auprc"] = macro_auprc(y, p)
    out["exact_match_accuracy"] = float(exact_match(y, p, th).mean())
    out["labelwise_accuracy"] = float((pred == y).mean())
    return {"summary": out, "per_class": per_class, "confusion": cms}


def paired_bootstrap(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, fn, n: int = 1000, seed: int = 42) -> dict:
    """Bootstrap the difference fn(p1) - fn(p2) over test patients (resampling records)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], p1[i]) - fn(y[i], p2[i])
        if np.isfinite(v):
            d.append(v)
    d = np.array(d)
    return {"diff": float(fn(y, p1) - fn(y, p2)), "ci_low": float(np.quantile(d, 0.025)),
            "ci_high": float(np.quantile(d, 0.975)), "p_le_0": float((d <= 0).mean())}
