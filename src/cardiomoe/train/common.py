"""Shared training / inference machinery for the four clients."""
from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from cardiomoe.clients.base import BaseClient, masked_bce
from cardiomoe.data.datasets import SignalDataset, make_loader, to_device
from cardiomoe.eval.metrics import macro_auroc

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(model: BaseClient, path: Path, **meta) -> None:
    """Stores trainable parameters + buffers (BatchNorm statistics, temperature)."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if k in trainable or k in dict(model.named_buffers())}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, **meta}, path)


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model: BaseClient, ds: SignalDataset, device, batch_size: int = 256, num_workers: int = 0,
            desc: str = "predict") -> dict:
    model.eval()
    keys, emb, logits, ys = [], [], [], []
    for batch in tqdm(make_loader(ds, batch_size, False, num_workers), desc=desc, leave=False):
        out = model(to_device(batch, device))
        keys += batch["key"]
        emb.append(out.embedding.float().cpu())
        logits.append(out.logits.float().cpu())
        ys.append(batch["y"])
    return {"keys": keys, "embedding": torch.cat(emb), "logits": torch.cat(logits), "y": torch.cat(ys)}


# --------------------------------------------------------------------------- training
def pos_weight_from(y: np.ndarray) -> torch.Tensor:
    pos = ((y == 1).sum(0)).astype(np.float64)
    neg = ((y == 0).sum(0)).astype(np.float64)
    return torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), dtype=torch.float32)


def make_optimizer(model: BaseClient, tcfg: dict):
    groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": tcfg["lr"]}]
    return torch.optim.AdamW(groups, weight_decay=tcfg["weight_decay"])


def make_scheduler(opt, tcfg: dict, total_steps: int):
    kind = tcfg["schedule"]
    if kind == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                                                   total_steps=total_steps)
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    if kind == "linear_warmup":
        warm = max(1, int(tcfg["warmup_frac"] * total_steps))
        f = lambda s: s / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm))
        return torch.optim.lr_scheduler.LambdaLR(opt, f)
    raise ValueError(kind)


def train_model(model: BaseClient, train_ds: SignalDataset, val_ds: SignalDataset, tcfg: dict,
                device, debug: bool = False, log_path: Path | None = None) -> tuple[BaseClient, dict]:
    """Train with early stopping on validation macro AUROC; returns the best model + history."""
    epochs = 2 if debug else tcfg["epochs"]
    bs = min(tcfg["batch_size"], 32) if debug else tcfg["batch_size"]
    nw = 0 if debug else tcfg.get("num_workers", 0)
    train_dl = make_loader(train_ds, bs, True, nw)
    model.to(device)
    opt = make_optimizer(model, tcfg)
    sched = make_scheduler(opt, tcfg, epochs * len(train_dl))
    pw = pos_weight_from(train_ds.y).to(device)
    # Mixed precision on CUDA only (MPS fp16 gave no speed-up).
    amp = device.type == "cuda" and tcfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, bad, hist = -math.inf, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in tqdm(train_dl, desc=f"epoch {ep}/{epochs}", leave=False):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(to_device(batch, device))
                loss = masked_bce(out.logits.float(), batch["y"].to(device), pw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(batch["y"])
            n += len(batch["y"])
        pred = predict(model, val_ds, device, tcfg.get("eval_batch_size", 256), nw, desc="val")
        auc = macro_auroc(pred["y"].numpy(), torch.sigmoid(pred["logits"]).numpy())
        rec = {"epoch": ep, "train_loss": tot / max(n, 1), "val_macro_auroc": auc, "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        log.info("%s", json.dumps(rec))
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if auc > best:
            best, bad = auc, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                log.info("early stop at epoch %d (best val macro AUROC %.4f)", ep, best)
                break
    model.load_state_dict(best_state)
    if device.type == "mps":
        torch.mps.empty_cache()
    return model.eval(), {"best_val_macro_auroc": best, "history": hist}

