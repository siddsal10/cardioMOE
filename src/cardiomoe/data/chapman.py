"""Chapman-Shaoxing (ecg-arrhythmia 1.0.0) loading and SNOMED -> superclass mapping (Section 5.7).

    python -m cardiomoe.data.chapman --prepare [--debug]

Writes to data/processed/ (or .../debug/):
    chapman_meta.parquet     one row per record: split, binary label, superclass labels (-1 = unknown),
                             Client A features (PTB-XL train statistics, height/weight missing)
    chapman_signals.npy      float16 [N, 12, 1000] filtered mV signal, resampled 500 -> 100 Hz
    chapman_label_map.csv    the full code -> superclass mapping

The dataset publishes one record per patient and no patient identifier, so a random
record-level split is patient-level.
"""
from __future__ import annotations

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

from cardiomoe.data import labels as L
from cardiomoe.data.ptbxl import TABULAR_SOURCE_COLUMNS, FEATURE_COLUMNS, processed_dir, tabular_features
from cardiomoe.data.signal import LEADS, preprocess
from cardiomoe.utils import load_config, repo_path, set_seed

log = logging.getLogger(__name__)
ADAPT_POOL_FRAC = 0.30


def parse_header(path: Path) -> dict:
    lines = path.read_text().splitlines()
    rec, nsig, fs, nsamp = lines[0].split()[:4]
    leads = [ln.split()[-1] for ln in lines[1 : 1 + int(nsig)]]
    com = {}
    for ln in lines[1 + int(nsig) :]:
        if ln.startswith("#") and ":" in ln:
            k, v = ln[1:].split(":", 1)
            com[k.strip()] = v.strip()
    age = pd.to_numeric(com.get("Age"), errors="coerce")
    sex = {"male": 0.0, "female": 1.0}.get(com.get("Sex", "").lower(), np.nan)
    return {"record": rec, "path": str(path.with_suffix("")), "fs": int(float(fs)), "n_samples": int(nsamp),
            "leads": leads, "dx": [c.strip() for c in com.get("Dx", "").split(",") if c.strip()],
            "age": age, "sex": sex}


def _read_one(path: str) -> tuple[np.ndarray, int, bool]:
    sig, info = wfdb.rdsamp(path)
    x, n_nan = preprocess(sig, int(info["fs"]), info["sig_name"])
    return x, n_nan, bool(np.allclose(sig, 0))


def build_meta(headers: list[dict], tab_stats: dict, seed: int) -> pd.DataFrame:
    meta = pd.DataFrame(headers).set_index("record").sort_index()
    lab = meta.dx.apply(L.chapman_labels)
    meta["normal"] = lab.str[0].astype(int)
    y = np.stack(lab.str[1].values)
    for i, c in enumerate(L.SUPERCLASSES):
        meta[f"y_{c}"] = y[:, i]
    meta["dx"] = meta.dx.str.join(",")

    demo = meta[["age", "sex"]].assign(height=np.nan, weight=np.nan)[TABULAR_SOURCE_COLUMNS]
    feats, _ = tabular_features(demo, tab_stats)
    for c in FEATURE_COLUMNS:
        meta[f"x_{c}"] = feats[c].values

    rng = np.random.default_rng(seed)
    pool = rng.random(len(meta)) < ADAPT_POOL_FRAC
    meta["split"] = np.where(pool, "adapt_pool", "adapt_test")
    return meta.drop(columns=["leads"])


def label_summary(meta: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for split, g in meta.groupby("split"):
        r = {"n": len(g), "abnormal": 1 - g.normal.mean()}
        for c in L.SUPERCLASSES:
            col = g[f"y_{c}"]
            r[f"{c}_pos"] = (col == 1).mean()
            r[f"{c}_unknown"] = (col == L.UNKNOWN).mean()
        rows[split] = r
    return pd.DataFrame(rows).T


def load_meta(cfg: dict, debug: bool = False) -> pd.DataFrame:
    return pd.read_parquet(processed_dir(cfg, debug) / "chapman_meta.parquet")


def load_signal_array(cfg: dict, debug: bool = False) -> np.ndarray:
    return np.load(processed_dir(cfg, debug) / "chapman_signals.npy", mmap_mode="r")


def prepare(cfg: dict, debug: bool) -> pd.DataFrame:
    raw = repo_path(cfg["paths"]["chapman"])
    out = processed_dir(cfg, debug)
    stats_path = out / "tabular_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} missing: run `python -m cardiomoe.data.ptbxl --prepare` first")
    tab_stats = json.loads(stats_path.read_text())

    names = L.chapman_condition_names(raw)
    L.chapman_label_table(names).to_csv(out / "chapman_label_map.csv", index=False)

    heas = sorted((raw / "WFDBRecords").rglob("*.hea"))
    if debug:
        heas = heas[: cfg["debug_n_samples"]]
    headers, bad = [], []
    for p in tqdm(heas, desc="chapman headers"):
        try:
            h = parse_header(p)
            ok = [l.upper() for l in h["leads"]] == [l.upper() for l in LEADS] and h["n_samples"] > 0
        except (ValueError, IndexError):
            ok = False
        (headers.append(h) if ok else bad.append(p.stem))
    log.info("excluded %d record(s) with malformed headers: %s", len(bad), bad)
    unnamed = {c for h in headers for c in h["dx"]} - set(names.Snomed_CT)
    if unnamed:
        raise ValueError(f"Dx codes with no name and no mapping decision: {sorted(unnamed)}")
    meta = build_meta(headers, tab_stats, cfg["seed"])

    with Pool(1 if debug else 8) as pool:
        res = list(tqdm(pool.imap(_read_one, meta.path.tolist(), chunksize=64), total=len(meta), desc="chapman signals"))
    log.info("replaced %d NaN/inf samples with 0", sum(r[1] for r in res))
    meta["all_zero"] = [r[2] for r in res]
    log.info("%d records are entirely zero (kept; normalise to zeros)", meta.all_zero.sum())
    np.save(out / "chapman_signals.npy", np.stack([r[0] for r in res]).astype(np.float16))
    meta["row"] = np.arange(len(meta))
    meta.to_parquet(out / "chapman_meta.parquet")

    summary = label_summary(meta)
    tables = repo_path(cfg["paths"]["tables"])
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / f"chapman_prevalence{'_debug' if debug else ''}.csv")
    print(f"{len(meta)} records -> {out}")
    print(summary.round(3).to_string())
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    set_seed(cfg["seed"])
    if args.prepare:
        prepare(cfg, args.debug)


if __name__ == "__main__":
    main()
