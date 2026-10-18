"""Reference comparisons on fold 10 (run after evaluate.py; nothing is tuned here).

    python -m cardiomoe.fourclient.baseline [--seeds 42 43 44] [--debug]

1. All-data baseline: one model trained on folds 1-8 (config `baselines.ALL`) vs the selected
   4-client ensemble, with a paired bootstrap (thresholds for both tuned on fold 9).
2. Leave-one-client-out: equal-weight ensembles of every subset of A-D, on fold 9 and fold 10,
   from the clients' saved probabilities.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import pandas as pd
import torch

from cardiomoe.data.datasets import SignalDataset, keys_for
from cardiomoe.eval.metrics import bootstrap_ci, macro_auroc, tune_thresholds
from cardiomoe.fourclient.aggregate import load_val
from cardiomoe.fourclient.evaluate import _paired_f1
from cardiomoe.fourclient.metrics import full_report, paired_bootstrap
from cardiomoe.fourclient.train import CLIENTS, calibrated_preds, fc_config, load_client, run_dir
from cardiomoe.utils import get_device, repo_path

log = logging.getLogger(__name__)


def baseline_seed(cfg: dict, seed: int, debug: bool, name: str = "ALL") -> dict:
    fc = cfg["fourclient"]
    d = run_dir(cfg, seed, debug)
    agg = json.loads((d / "gate" / "aggregate.json").read_text())
    saved = torch.load(d / "test_preds.pt", weights_only=False)
    sel = agg["selected"]
    p_ens, y = saved["probs"][sel], saved["y"].numpy()

    device = get_device(cfg["device"], debug)
    ds = SignalDataset(cfg, keys_for(cfg, [fc["test_fold"]], debug), debug=debug)
    out = calibrated_preds(load_client(cfg, name, seed, debug, device), ds, device, fc["train"]["eval_batch_size"])
    assert out["keys"] == saved["keys"], "test record order differs from the saved ensemble predictions"
    p_base = out["probs"].numpy()

    val = torch.load(d / name / "val_preds.pt", weights_only=False)
    th_base = tune_thresholds(val["y"].numpy(), val["probs"].numpy())
    th_ens = np.array(agg["thresholds"][sel])
    nb = fc["bootstrap"]
    rep = {}
    for key, p, th in (("baseline", p_base, th_base), ("ensemble", p_ens, th_ens)):
        r = full_report(y, p, th)["summary"]
        r["auroc_ci_low"], r["auroc_ci_high"] = bootstrap_ci(y, p, n=nb, seed=seed)
        rep[key] = r
    meta = json.loads((d / name / "meta.json").read_text())
    return {"seed": seed, "selected": sel, "baseline_fold9_auroc": meta["best_val_macro_auroc"], "reports": rep,
            "baseline_minus_ensemble": {
                "auroc": paired_bootstrap(y, p_base, p_ens, macro_auroc, nb, seed),
                "f1": _paired_f1(y, p_base, th_base, p_ens, th_ens, nb, seed)}}


def subsets_seed(cfg: dict, seed: int, debug: bool) -> dict:
    _, P9, y9, _ = load_val(cfg, seed, debug)
    saved = torch.load(run_dir(cfg, seed, debug) / "test_preds.pt", weights_only=False)
    P10, y10 = saved["P"], saved["y"].numpy()
    out = {}
    for k in (4, 3, 2, 1):
        for sub in itertools.combinations(range(len(CLIENTS)), k):
            name = "".join(CLIENTS[i] for i in sub)
            out[name] = {"fold9": macro_auroc(y9.numpy(), P9[:, list(sub)].mean(1).numpy()),
                         "fold10": macro_auroc(y10, P10[:, list(sub)].mean(1).numpy())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = fc_config()
    base = [baseline_seed(cfg, s, args.debug) for s in args.seeds]
    subs = [subsets_seed(cfg, s, args.debug) for s in args.seeds]
    tables = repo_path(cfg["fourclient"]["out_dir"]) / ("debug" if args.debug else "") / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "baseline_comparison.json").write_text(json.dumps(base, indent=2))

    keys = ("macro_auroc", "macro_f1", "macro_precision", "macro_recall", "macro_specificity",
            "exact_match_accuracy", "labelwise_accuracy")
    rows = []
    for k in ("ensemble", "baseline"):
        row = {"model": k}
        for m in keys:
            v = np.array([b["reports"][k][m] for b in base])
            row[m], row[m + "_sd"] = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables / "baseline_comparison.csv", index=False)
    sub_df = pd.DataFrame([{"subset": n, "split": sp, "seed": s, "macro_auroc": v[sp]}
                           for s, sub in zip(args.seeds, subs) for n, v in sub.items() for sp in ("fold9", "fold10")])
    summ = sub_df.groupby(["subset", "split"]).macro_auroc.agg(["mean", "std"]).unstack("split")
    summ = summ.sort_values(("mean", "fold9"), ascending=False)
    summ.to_csv(tables / "client_subsets.csv")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(json.dumps([{"seed": b["seed"], **b["baseline_minus_ensemble"]} for b in base], indent=2))
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
