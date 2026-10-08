"""Render 12-lead signals as paper-style ECG printouts, plus photo-like augmentations (Section 5.6).

    python -m cardiomoe.data.render_ecg --dataset ptbxl [--debug]     # -> data/processed/ptbxl_images/{ecg_id}.png
    python -m cardiomoe.data.render_ecg --dataset chapman [--debug]   # -> data/processed/chapman_images/{record}.png

Layout: standard 3 x 4 grid (I/II/III | aVR/aVL/aVF | V1-V3 | V4-V6, 2.5 s each) plus a
10 s lead II rhythm strip; 25 mm/s, 10 mm/mV, 1 mm pink grid with 5 mm red major lines.
Input is the filtered, un-normalised signal in mV. Drawn at 2x and downsampled for
anti-aliasing. Output 256 x 512 (H x W) RGB PNG.
"""
from __future__ import annotations

import argparse
import io
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from cardiomoe.data.signal import FS, LEADS
from cardiomoe.utils import load_config, repo_path

IMG_H, IMG_W = 256, 512
SS = 2                                   # supersampling factor
PAPER_MM_W = 250.0                       # 10 s at 25 mm/s
PX_PER_MM = IMG_W * SS / PAPER_MM_W      # ~4.1 px/mm while drawing
MM_PER_S, MM_PER_MV = 25.0, 10.0
N_ROWS = 4
ROW_MM = IMG_H * SS / PX_PER_MM / N_ROWS
LAYOUT = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
RHYTHM_LEAD = "II"

PAPER = (255, 246, 246)
MINOR = (250, 205, 205)
MAJOR = (235, 140, 140)
INK = (20, 20, 20)


@lru_cache(maxsize=1)
def _paper() -> Image.Image:
    w, h = IMG_W * SS, IMG_H * SS
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    n_x, n_y = int(w / PX_PER_MM) + 1, int(h / PX_PER_MM) + 1
    for major in (False, True):  # minor first so major lines sit on top
        for i in range(n_x):
            if (i % 5 == 0) == major:
                x = round(i * PX_PER_MM)
                d.line([(x, 0), (x, h)], fill=MAJOR if major else MINOR, width=2 if major else 1)
        for j in range(n_y):
            if (j % 5 == 0) == major:
                y = round(j * PX_PER_MM)
                d.line([(0, y), (w, y)], fill=MAJOR if major else MINOR, width=2 if major else 1)
    return img


def _trace(sig: np.ndarray, t0_s: float, x0_mm: float, y_mm: float) -> list[tuple[float, float]]:
    t = t0_s + np.arange(len(sig)) / FS
    x = (x0_mm + (t - t0_s) * MM_PER_S) * PX_PER_MM
    y = (y_mm - sig * MM_PER_MV) * PX_PER_MM
    y = np.clip(y, 0, IMG_H * SS - 1)
    return list(zip(x.tolist(), y.tolist()))


def render(sig_mv: np.ndarray) -> Image.Image:
    """(12, 1000) filtered mV signal in LEADS order -> 256 x 512 RGB paper-style image."""
    sig = np.nan_to_num(np.asarray(sig_mv, dtype=np.float32))
    img = _paper().copy()
    d = ImageDraw.Draw(img)
    seg = sig.shape[1] // 4                       # 2.5 s per column
    seg_mm = seg / FS * MM_PER_S
    for r, row in enumerate(LAYOUT):
        y = (r + 0.5) * ROW_MM
        for c, lead in enumerate(row):
            s = sig[LEADS.index(lead), c * seg : (c + 1) * seg]
            d.line(_trace(s, c * seg / FS, c * seg_mm, y), fill=INK, width=SS)
            d.text((c * seg_mm * PX_PER_MM + 4, (y - ROW_MM * 0.42) * PX_PER_MM), lead, fill=INK)
            if c:  # lead-change tick
                x = c * seg_mm * PX_PER_MM
                d.line([(x, (y - 2.5) * PX_PER_MM), (x, (y + 2.5) * PX_PER_MM)], fill=INK, width=SS)
    y = 3.5 * ROW_MM
    d.line(_trace(sig[LEADS.index(RHYTHM_LEAD)], 0.0, 0.0, y), fill=INK, width=SS)
    d.text((4, (y - ROW_MM * 0.42) * PX_PER_MM), RHYTHM_LEAD, fill=INK)
    return img.resize((IMG_W, IMG_H), Image.LANCZOS)


# --------------------------------------------------------------------------- photo-like augmentation
class PhotoAugment:
    """Train-time augmentations that mimic a photographed printout (Section 5.6).

    Operates on PIL images and returns a PIL image of the same size.
    """

    def __init__(self, seed: int | None = None):
        import torchvision.transforms.v2 as T

        self.rng = np.random.default_rng(seed)
        fill = (255, 255, 255)
        self.geom = T.Compose([
            T.RandomRotation(3, fill=fill),
            T.RandomPerspective(distortion_scale=0.08, p=0.5, fill=fill),
            T.RandomResizedCrop((IMG_H, IMG_W), scale=(0.9, 1.0), ratio=(IMG_W / IMG_H * 0.95, IMG_W / IMG_H * 1.05)),
            T.ColorJitter(brightness=0.25, contrast=0.25),
        ])

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self.geom(img)
        if self.rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(self.rng.uniform(0.2, 1.0))))
        if self.rng.random() < 0.5:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(self.rng.integers(30, 90)))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


# --------------------------------------------------------------------------- batch rendering
def _render_job(job: tuple[np.ndarray, str]) -> None:
    sig, path = job
    render(sig.astype(np.float32)).save(path, optimize=True)


def render_dataset(name: str, cfg: dict, debug: bool, overwrite: bool = False) -> Path:
    if name == "ptbxl":
        from cardiomoe.data import ptbxl as ds
    elif name == "chapman":
        from cardiomoe.data import chapman as ds
    else:
        raise ValueError(name)
    meta, sig = ds.load_meta(cfg, debug), ds.load_signal_array(cfg, debug)
    out = ds.processed_dir(cfg, debug) / f"{name}_images"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, out / f"{key}.png") for key, i in zip(meta.index, meta.row)]
    jobs = [(i, str(p)) for i, p in jobs if overwrite or not p.exists()]
    with Pool(1 if debug else 8) as pool:
        it = pool.imap_unordered(_render_job, ((np.asarray(sig[i]), p) for i, p in jobs), chunksize=32)
        for _ in tqdm(it, total=len(jobs), desc=f"render {name}"):
            pass
    print(f"{len(meta)} images in {out} ({len(jobs)} newly rendered)")
    spot_check(meta, out, repo_path(cfg["paths"]["figures"]) / f"render_samples{'_debug' if debug else ''}.png",
               cfg["seed"], name)
    return out


def spot_check(meta, img_dir: Path, dest: Path, seed: int, name: str, n: int = 20) -> None:
    """Contact sheet of n random renders with their labels (Phase 1 acceptance)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cardiomoe.data.labels import SUPERCLASSES

    pick = meta.sample(n=min(n, len(meta)), random_state=seed)
    fig, axes = plt.subplots(5, 4, figsize=(20, 13))
    for ax, (key, r) in zip(axes.flat, pick.iterrows()):
        ax.imshow(Image.open(img_dir / f"{key}.png"))
        lab = [c for c in SUPERCLASSES if r.get(f"y_{c}", 0) == 1]
        ax.set_title(f"{name} {key}: {'+'.join(lab) or '-'}", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    print(f"spot-check sheet -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl", choices=["ptbxl", "chapman"])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    render_dataset(args.dataset, load_config(), args.debug, args.overwrite)


if __name__ == "__main__":
    main()
