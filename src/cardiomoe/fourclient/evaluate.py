"""Step 3: final evaluation on the untouched test fold (fold 10) plus tables and figures.

    python -m cardiomoe.fourclient.evaluate [--seeds 42] [--debug] [--rerun]

Thresholds, aggregation weights and the "best individual model" are all fixed on fold 9
beforehand; nothing here is tuned on fold 10. A marker file records the first full run, and
a second one needs --rerun so re-evaluating the test fold is always deliberate.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cardiomoe.data.datasets import SignalDataset, keys_for
from cardiomoe.data.labels import SUPERCLASSES
from cardiomoe.eval.metrics import bootstrap_ci, macro_auroc, macro_f1
from cardiomoe.fourclient.aggregate import COMBINERS
from cardiomoe.fourclient.metrics import full_report, paired_bootstrap
from cardiomoe.fourclient.models import ARCH_NAMES
from cardiomoe.fourclient.train import CLIENTS, calibrated_preds, fc_config, load_client, run_dir
from cardiomoe.utils import get_device, repo_path

log = logging.getLogger(__name__)
COMB_LABEL = {"equal": "Equal-weight mean", "static": "Static learned weights", "dynamic": "Dynamic gate"}


def test_preds(cfg: dict, seed: int, debug: bool):
    fc = cfg["fourclient"]
    device = get_device(cfg["device"], debug)
    ds = SignalDataset(cfg, keys_for(cfg, [fc["test_fold"]], debug), debug=debug)
    outs = [calibrated_preds(load_client(cfg, c, seed, debug, device), ds, device, fc["train"]["eval_batch_size"])
            for c in CLIENTS]
    P = torch.stack([o["probs"] for o in outs], dim=1).float()
    return outs[0]["keys"], P, outs[0]["y"].float()


def evaluate_seed(cfg: dict, seed: int, debug: bool) -> dict:
    d = run_dir(cfg, seed, debug)
    agg = json.loads((d / "gate" / "aggregate.json").read_text())
    with open(d / "gate" / "combiners.pkl", "rb") as f:
        comb = pickle.load(f)
    keys, P, y = test_preds(cfg, seed, debug)
    yn = y.numpy()
    probs = {c: P[:, i].numpy() for i, c in enumerate(CLIENTS)}
    probs.update({n: comb[n].predict(P).numpy() for n in COMBINERS})
    dyn_w = comb["dynamic"].weights(P).numpy()

    reports = {}
    nb = cfg["fourclient"]["bootstrap"]
    for name, p in probs.items():
        th = np.array(agg["thresholds"][name])
        r = full_report(yn, p, th)
        lo, hi = bootstrap_ci(yn, p, n=nb, seed=seed)
        r["summary"].update({"auroc_ci_low": lo, "auroc_ci_high": hi})
        reports[name] = r

    best = max(CLIENTS, key=lambda c: agg["client_fold9_auroc"][c])  # chosen on fold 9, not on test
    sel = agg["selected"]
    th_sel, th_best = np.array(agg["thresholds"][sel]), np.array(agg["thresholds"][best])
    sig = {"best_individual": best, "selected": sel,
           "auroc": paired_bootstrap(yn, probs[sel], probs[best], macro_auroc, nb, seed),
           # each model keeps its own fold-9 thresholds
           "f1": _paired_f1(yn, probs[sel], th_sel, probs[best], th_best, nb, seed)}

    torch.save({"keys": keys, "P": P, "y": y, "probs": probs, "dynamic_weights": dyn_w}, d / "test_preds.pt")
    res = {"seed": seed, "reports": reports, "significance": sig, "aggregate": agg,
           "dynamic_mean_weights_test": dict(zip(CLIENTS, dyn_w.mean(0).tolist())), "n_test": len(keys)}
    (d / "test_results.json").write_text(json.dumps(res, indent=2))
    return res


def _paired_f1(y, p1, th1, p2, th2, n, seed) -> dict:
    rng = np.random.default_rng(seed)
    f = lambda i: macro_f1(y[i], p1[i], th1) - macro_f1(y[i], p2[i], th2)
    d = np.array([f(rng.integers(0, len(y), len(y))) for _ in range(n)])
    return {"diff": float(f(np.arange(len(y)))), "ci_low": float(np.quantile(d, 0.025)),
            "ci_high": float(np.quantile(d, 0.975)), "p_le_0": float((d <= 0).mean())}


# --------------------------------------------------------------------------- tables
def model_rows(cfg: dict, results: list[dict]) -> pd.DataFrame:
    fc = cfg["fourclient"]
    r0 = results[0]
    sel = r0["aggregate"]["selected"]
    # each seed's own best client, chosen on fold 9
    best_auc = np.array([r["reports"][r["significance"]["best_individual"]]["summary"]["macro_auroc"]
                         for r in results])
    rows = []
    for name in CLIENTS + COMBINERS:
        s = {k: np.array([r["reports"][name]["summary"][k] for r in results]) for k in r0["reports"][name]["summary"]}
        is_client = name in CLIENTS
        row = {"model": f"Client {name}" if is_client else COMB_LABEL[name] + (" (selected)" if name == sel else ""),
               "architecture": ARCH_NAMES[fc["clients"][name]["arch"]] if is_client else "aggregation of A-D",
               "train_folds": ",".join(map(str, fc["partition"][name])) if is_client else "gate: fold 9",
               "static_weight": np.mean([r["aggregate"]["static_weights"][name] for r in results]) if is_client else np.nan,
               "dynamic_mean_weight": (np.mean([r["dynamic_mean_weights_test"][name] for r in results])
                                       if is_client else np.nan),
               "fold9_cv_auroc": (np.mean([r["aggregate"]["client_fold9_auroc"][name] for r in results]) if is_client
                                  else np.mean([r["aggregate"]["cv"][name]["cv_macro_auroc"] for r in results])),
               "delta_auroc_vs_best": float(np.mean(s["macro_auroc"] - best_auc))}
        for k in ("macro_auroc", "auroc_ci_low", "auroc_ci_high", "macro_auprc", "macro_f1", "macro_precision",
                  "macro_recall", "macro_specificity", "exact_match_accuracy", "labelwise_accuracy"):
            row[k] = float(s[k].mean())
            if len(results) > 1:
                row[k + "_sd"] = float(s[k].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_rows(results: list[dict]) -> pd.DataFrame:
    rows = []
    for name, rep in results[0]["reports"].items():
        for c in SUPERCLASSES:
            rows.append({"model": name, "class": c, **rep["per_class"][c], **rep["confusion"][c]})
    return pd.DataFrame(rows)


def fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def markdown(df: pd.DataFrame, results: list[dict]) -> str:
    multi = len(results) > 1
    pm = lambda r, k: fmt(r[k]) + (f" ± {fmt(r[k + '_sd'])}" if multi else "")
    lines = ["| Model | Architecture | Train data | Static w | Mean dynamic w (test) | Test macro AUROC [95% CI] "
             "| Macro F1 | Precision | Recall | Specificity | Exact-match acc | Label-wise acc | Δ AUROC vs best client |",
             "|" + "---|" * 13]
    for _, r in df.iterrows():
        lines.append(f"| {r.model} | {r.architecture} | {r.train_folds} | {fmt(r.static_weight)} | "
                     f"{fmt(r.dynamic_mean_weight)} | {pm(r, 'macro_auroc')} [{fmt(r.auroc_ci_low)}, "
                     f"{fmt(r.auroc_ci_high)}] | {pm(r, 'macro_f1')} | {fmt(r.macro_precision)} | "
                     f"{fmt(r.macro_recall)} | {fmt(r.macro_specificity)} | {pm(r, 'exact_match_accuracy')} | "
                     f"{fmt(r.labelwise_accuracy)} | {r.delta_auroc_vs_best:+.3f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- figures
def plot_training(cfg: dict, seed: int, debug: bool, out) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in CLIENTS:
        path = run_dir(cfg, seed, debug) / c / "history.jsonl"
        h = [json.loads(line) for line in path.read_text().splitlines()]
        lab = f"{c}: {ARCH_NAMES[cfg['fourclient']['clients'][c]['arch']]}"
        ax[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], label=lab)
        ax[1].plot([r["epoch"] for r in h], [r["val_macro_auroc"] for r in h], label=lab)
    ax[0].set(xlabel="epoch", ylabel="train loss (weighted BCE)", title="Training loss")
    ax[1].set(xlabel="epoch", ylabel="fold-9 macro AUROC", title="Validation macro AUROC")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, out) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(df))
    colors = ["#4c78a8"] * 4 + ["#9d9d9d", "#f58518", "#54a24b"]
    err = np.vstack([df.macro_auroc - df.auroc_ci_low, df.auroc_ci_high - df.macro_auroc])
    ax[0].bar(x, df.macro_auroc, yerr=err, color=colors, capsize=3)
    lo = max(0.0, float(df.auroc_ci_low.min()) - 0.03)
    ax[0].set(ylim=(lo, 1.0), ylabel="macro AUROC", title="Test (fold 10) macro AUROC, 95% CI")
    ax[1].bar(x, df.macro_f1, color=colors)
    ax[1].set(ylim=(0, 1), ylabel="macro F1", title="Test macro F1 (fold-9 thresholds)")
    for a in ax:
        a.set_xticks(x, [m.replace(" (selected)", "*") for m in df.model], rotation=30, ha="right", fontsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion(res: dict, names: list[str], out) -> None:
    fig, axes = plt.subplots(len(names), 5, figsize=(12, 2.3 * len(names)))
    for i, n in enumerate(names):
        for j, c in enumerate(SUPERCLASSES):
            cm = res["reports"][n]["confusion"][c]
            m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            ax = axes[i, j]
            ax.imshow(m / m.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
            for (a, b), v in np.ndenumerate(m):
                ax.text(b, a, str(v), ha="center", va="center", fontsize=8)
            ax.set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7)
            ax.set_yticks([0, 1], ["true 0", "true 1"], fontsize=7)
            if i == 0:
                ax.set_title(c, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Client {n}" if n in CLIENTS else COMB_LABEL[n], fontsize=8)
    fig.suptitle("Per-class confusion matrices on fold 10 (row-normalised colour, counts shown)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_weights(res: dict, cfg: dict, seed: int, debug: bool, out) -> None:
    w = torch.load(run_dir(cfg, seed, debug) / "test_preds.pt", weights_only=False)["dynamic_weights"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    sw = [res["aggregate"]["static_weights"][c] for c in CLIENTS]
    ax[0].bar(CLIENTS, sw, color="#f58518")
    ax[0].axhline(0.25, ls="--", c="gray", lw=1, label="equal (0.25)")
    ax[0].set(ylim=(0, 1), ylabel="weight", title="Static learned weights (fit on fold 9)")
    ax[0].legend(fontsize=8)
    ax[1].boxplot([w[:, i] for i in range(len(CLIENTS))], tick_labels=CLIENTS, showfliers=False)
    ax[1].axhline(0.25, ls="--", c="gray", lw=1)
    ax[1].set(ylim=(0, 1), ylabel="weight", title="Dynamic gate: per-patient weights on fold 10")
    fig.tight_layout()
    fig.savefig(out / "aggregation_weights.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--rerun", action="store_true", help="allow re-evaluating fold 10 after a previous full run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = fc_config()
    base = repo_path(cfg["fourclient"]["out_dir"]) / ("debug" if args.debug else "")
    marker = base / "test_evaluated.json"
    if marker.exists() and not args.debug and not args.rerun:
        raise SystemExit(f"fold 10 was already evaluated ({marker.read_text().strip()}); pass --rerun to repeat")

    results = [evaluate_seed(cfg, s, args.debug) for s in args.seeds]
    tables, figs = base / "tables", base / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = model_rows(cfg, results)
    df.to_csv(tables / "comparison.csv", index=False)
    per_class_rows(results).to_csv(tables / "per_class.csv", index=False)
    md = markdown(df, results)
    (tables / "comparison.md").write_text(md + "\n")
    (tables / "significance.json").write_text(json.dumps([r["significance"] for r in results], indent=2))

    r0, s0 = results[0], args.seeds[0]
    plot_training(cfg, s0, args.debug, figs)
    plot_comparison(df, figs)
    plot_confusion(r0, CLIENTS + [r0["aggregate"]["selected"]], figs)
    plot_weights(r0, cfg, s0, args.debug, figs)
    if not args.debug:
        marker.write_text(json.dumps({"evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                      "seeds": args.seeds}) + "\n")
    print(md)
    print(json.dumps([r["significance"] for r in results], indent=2))


if __name__ == "__main__":
    main()
