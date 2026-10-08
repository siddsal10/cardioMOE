"""Mendeley "ECG Images dataset of Cardiac Patients" (gwbz3fsgp8 v2) loader (Sections 5.6, 5.8).

    python -m cardiomoe.data.pk_images --prepare [--debug]

Writes data/processed/pk_images_meta.parquet: path, class, binary label, superclass
labels (-1 = unknown) and a stratified 30% adapt / 70% test split (seed 42).
The dataset has no patient IDs, so the split cannot be guaranteed patient-level.
Images are letterboxed to 256 x 512 at load time (`load_image`), not stored.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from cardiomoe.data import labels as L
from cardiomoe.data.ptbxl import processed_dir
from cardiomoe.utils import load_config, repo_path

IMG_H, IMG_W = 256, 512
EXTS = {".jpg", ".jpeg", ".png"}
ADAPT_FRAC = 0.30


def letterbox(img: Image.Image, h: int = IMG_H, w: int = IMG_W, fill=(255, 255, 255)) -> Image.Image:
    """Resize preserving aspect ratio and pad to exactly (h, w)."""
    img = img.convert("RGB")
    s = min(w / img.width, h / img.height)
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (w, h), fill)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
    return canvas


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return letterbox(im)


def build_meta(raw: Path, seed: int) -> pd.DataFrame:
    rows = []
    for folder in sorted(p for p in raw.iterdir() if p.is_dir()):
        name, normal, y = L.pk_folder_label(folder.name)
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in EXTS:
                rows.append({"image_id": f"{name}/{f.stem}", "path": str(f.relative_to(raw)), "cls": name,
                             "normal": normal, **{f"y_{c}": v for c, v in zip(L.SUPERCLASSES, y)}})
    meta = pd.DataFrame(rows).set_index("image_id")
    pos = np.arange(len(meta))
    adapt, _ = train_test_split(pos, train_size=ADAPT_FRAC, stratify=meta.cls.to_numpy(), random_state=seed)
    meta["split"] = np.where(np.isin(pos, adapt), "adapt", "test")
    return meta


def load_meta(cfg: dict, debug: bool = False) -> pd.DataFrame:
    return pd.read_parquet(processed_dir(cfg, debug) / "pk_images_meta.parquet")


def prepare(cfg: dict, debug: bool) -> pd.DataFrame:
    raw = repo_path(cfg["paths"]["pk_images"])
    meta = build_meta(raw, cfg["seed"])
    if debug:
        meta = meta.groupby("cls", group_keys=False).apply(lambda g: g.head(cfg["debug_n_samples"] // 4))
    for p in meta.path.head(3):  # make sure images decode and letterbox correctly
        assert load_image(raw / p).size == (IMG_W, IMG_H)
    meta.to_parquet(processed_dir(cfg, debug) / "pk_images_meta.parquet")
    print(pd.crosstab(meta.cls, meta.split, margins=True).to_string())
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare(load_config(), args.debug)


if __name__ == "__main__":
    main()
