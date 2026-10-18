"""Step 2: learn the aggregation weights on fold 9 (validation only, never fold 10).

    python -m cardiomoe.fourclient.aggregate [--seeds 42] [--debug]

1. Stack the four clients' calibrated fold-9 probabilities -> P [N, 4, 5].
2. Patient-grouped 5-fold CV inside fold 9 gives out-of-fold predictions for each combiner
   (equal, static, dynamic). These pick the combiner and tune its per-class thresholds.
3. Refit every combiner on all of fold 9 and save it.

Selection prefers the simplest combiner: candidates are tried in order equal -> static ->
dynamic, and a more complex one replaces the current choice only if its CV macro AUROC beats
it by at least `gate.min_gain`.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from cardiomoe.data.datasets import load_meta
from cardiomoe.eval.metrics import macro_auroc, tune_thresholds
from cardiomoe.fourclient.gate import DynamicGate, EqualWeights, StaticWeights, bce_prob
from cardiomoe.fourclient.train import CLIENTS, fc_config, run_dir
from cardiomoe.utils import set_seed

log = logging.getLogger(__name__)
COMBINERS = ["equal", "static", "dynamic"]


def make(name: str, cfg: dict, seed: int):
    g = cfg["fourclient"]["gate"]
    if name == "equal":
        return EqualWeights()
    if name == "static":
        return StaticWeights(**g["static"])
    return DynamicGate(**g["dynamic"], seed=seed)


def load_val(cfg: dict, seed: int, debug: bool):
    d = run_dir(cfg, seed, debug)
    preds = [torch.load(d / c / "val_preds.pt", weights_only=False) for c in CLIENTS]
    keys = preds[0]["keys"]
    assert all(p["keys"] == keys for p in preds), "clients disagree on fold-9 record order"
    P = torch.stack([p["probs"] for p in preds], dim=1).float()
    y = preds[0]["y"].float()
    groups = load_meta(cfg, debug).loc[keys, "patient_id"].to_numpy()
    return keys, P, y, groups


def run(cfg: dict, seed: int, debug: bool) -> dict:
    set_seed(seed)
    keys, P, y, groups = load_val(cfg, seed, debug)
    yn = y.numpy()
    n_splits = min(cfg["fourclient"]["gate"]["cv_folds"], len(np.unique(groups)))
    oof = {n: torch.zeros_like(y) for n in COMBINERS}
    for tr, va in GroupKFold(n_splits).split(P, yn, groups):
        for n in COMBINERS:
            m = make(n, cfg, seed).fit(P[tr], y[tr], groups[tr])
            oof[n][va] = m.predict(P[va])
    cv = {n: {"cv_macro_auroc": macro_auroc(yn, oof[n].numpy()), "cv_bce": float(bce_prob(oof[n], y))}
          for n in COMBINERS}
    gain = cv["dynamic"]["cv_macro_auroc"] - cv["static"]["cv_macro_auroc"]
    selected = COMBINERS[0]
    for n in COMBINERS[1:]:  # simplest first; complexity must earn min_gain
        if cv[n]["cv_macro_auroc"] >= cv[selected]["cv_macro_auroc"] + cfg["fourclient"]["gate"]["min_gain"]:
            selected = n
    log.info("fold-9 CV: %s; selected %s (dynamic - static = %+.4f)", json.dumps(cv), selected, gain)

    # Per-class thresholds: combiners from their CV predictions, clients from their own fold-9 output.
    th = {n: tune_thresholds(yn, oof[n].numpy()).tolist() for n in COMBINERS}
    th.update({c: tune_thresholds(yn, P[:, i].numpy()).tolist() for i, c in enumerate(CLIENTS)})

    fitted = {n: make(n, cfg, seed).fit(P, y, groups) for n in COMBINERS}
    out = run_dir(cfg, seed, debug) / "gate"
    out.mkdir(exist_ok=True)
    with open(out / "combiners.pkl", "wb") as f:
        pickle.dump(fitted, f)
    res = {"selected": selected, "dynamic_minus_static_cv_auroc": gain, "cv": cv, "thresholds": th,
           "static_weights": dict(zip(CLIENTS, fitted["static"].w.tolist())),
           "dynamic_mean_weights_fold9": dict(zip(CLIENTS, fitted["dynamic"].weights(P).mean(0).tolist())),
           "client_fold9_auroc": {c: macro_auroc(yn, P[:, i].numpy()) for i, c in enumerate(CLIENTS)},
           "n_val": len(keys)}
    (out / "aggregate.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = fc_config()
    for seed in args.seeds:
        print(json.dumps(run(cfg, seed, args.debug), indent=2))


if __name__ == "__main__":
    main()
