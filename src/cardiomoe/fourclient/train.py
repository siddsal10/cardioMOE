"""Step 1: train the four clients independently.

    python -m cardiomoe.fourclient.train [--clients A B C D] [--seeds 42] [--debug]
    python -m cardiomoe.fourclient.train --clients ALL      # baseline: one model on all folds 1-8

Each client gets only its own PTB-XL folds (config `partition`), early-stops on fold 9,
then has a temperature fitted on fold 9. Saved per client under outputs/fourclient/seed{s}/{X}/:
model.pt, history.jsonl and val_preds.pt (calibrated fold-9 probabilities, used by the gate).
Fold 10 is not touched here.
"""
from __future__ import annotations

import argparse
import json
import logging

import torch

from cardiomoe.data.datasets import SignalDataset, keys_for
from cardiomoe.fourclient.models import SignalClient
from cardiomoe.fusion.calibration import fit_temperature
from cardiomoe.train.common import predict, save_checkpoint, train_model
from cardiomoe.utils import get_device, load_config, repo_path, set_seed

log = logging.getLogger(__name__)
CLIENTS = ["A", "B", "C", "D"]


def fc_config() -> dict:
    return load_config("fourclient")


def run_dir(cfg: dict, seed: int, debug: bool):
    d = repo_path(cfg["fourclient"]["out_dir"]) / ("debug" if debug else "") / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def spec(cfg: dict, client: str) -> tuple[str, list[int]]:
    """(architecture, training folds) for a client A-D or a baseline from `baselines`."""
    fc = cfg["fourclient"]
    if client in fc["clients"]:
        return fc["clients"][client]["arch"], fc["partition"][client]
    b = fc["baselines"][client]
    return b["arch"], b["folds"]


def build(cfg: dict, client: str) -> SignalClient:
    fc = cfg["fourclient"]
    return SignalClient(spec(cfg, client)[0], fc["embed_dim"], fc["dropout"])


def load_client(cfg: dict, client: str, seed: int, debug: bool, device="cpu") -> SignalClient:
    ck = torch.load(run_dir(cfg, seed, debug) / client / "model.pt", map_location="cpu", weights_only=False)
    model = build(cfg, client)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def calibrated_preds(model: SignalClient, ds: SignalDataset, device, bs: int) -> dict:
    out = predict(model, ds, device, bs)
    out["probs"] = torch.sigmoid(out["logits"] / model.temperature.cpu())
    return out


def train_one(cfg: dict, client: str, seed: int, debug: bool) -> dict:
    fc = cfg["fourclient"]
    tcfg = fc["train"]
    set_seed(seed)
    device = get_device(cfg["device"], debug)
    folds = spec(cfg, client)[1]
    train_ds = SignalDataset(cfg, keys_for(cfg, folds, debug), train=True, debug=debug)
    val_ds = SignalDataset(cfg, keys_for(cfg, [fc["val_fold"]], debug), debug=debug)
    model = build(cfg, client)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("client %s (%s): folds %s, train %d, val %d, %d params, device %s", client, model.arch, folds,
             len(train_ds), len(val_ds), n_params, device)

    out = run_dir(cfg, seed, debug) / client
    hist_path = out / "history.jsonl"
    hist_path.unlink(missing_ok=True)
    model, info = train_model(model, train_ds, val_ds, tcfg, device, debug, hist_path)

    val = predict(model, val_ds, device, tcfg["eval_batch_size"])
    T = fit_temperature(val["logits"], val["y"])
    model.temperature.fill_(T)
    model.cpu()
    save_checkpoint(model, out / "model.pt", client=client, arch=model.arch, folds=folds, seed=seed,
                    temperature=T, n_params=n_params, best_val_macro_auroc=info["best_val_macro_auroc"])
    probs = torch.sigmoid(val["logits"] / T)
    torch.save({"keys": val["keys"], "probs": probs, "y": val["y"]}, out / "val_preds.pt")
    meta = {"client": client, "arch": model.arch, "folds": folds, "n_train": len(train_ds), "n_params": n_params,
            "temperature": T, "best_val_macro_auroc": info["best_val_macro_auroc"],
            "epochs_run": len(info["history"])}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("client %s done: fold-9 macro AUROC %.4f, T=%.3f", client, info["best_val_macro_auroc"], T)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", nargs="+", default=CLIENTS, choices=CLIENTS + list(fc_config()["fourclient"]["baselines"]))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="retrain clients that already finished")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = fc_config()
    for seed in args.seeds:
        for c in args.clients:
            done = run_dir(cfg, seed, args.debug) / c / "meta.json"
            if done.exists() and not args.retrain:
                log.info("client %s seed %d already trained, skipping (%s)", c, seed, done)
                continue
            print(json.dumps(train_one(cfg, c, seed, args.debug)))


if __name__ == "__main__":
    main()
