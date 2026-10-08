"""ECG signal preprocessing shared by Clients A and C and by Chapman (Section 5.4).

Signals are handled as float arrays of shape (12, T) in the standard lead order.
Stored caches hold the *filtered, un-normalised* signal in mV (Client C renders from it);
`normalize` is applied at load time for Client A.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

log = logging.getLogger(__name__)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 100
N_SAMPLES = 1000  # 10 s at 100 Hz


def reorder_leads(sig: np.ndarray, names: list[str]) -> np.ndarray:
    """Reorder (12, T) to LEADS. Matching is case-insensitive (PTB-XL writes AVR/AVL/AVF)."""
    idx = {n.upper(): i for i, n in enumerate(names)}
    missing = [l for l in LEADS if l.upper() not in idx]
    if missing:
        raise ValueError(f"missing leads {missing} in {names}")
    return sig[[idx[l.upper()] for l in LEADS]]


def resample(sig: np.ndarray, fs_in: int, fs_out: int = FS) -> np.ndarray:
    if fs_in == fs_out:
        return sig
    g = np.gcd(int(fs_in), int(fs_out))
    return resample_poly(sig, fs_out // g, int(fs_in) // g, axis=-1)


def fix_length(sig: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Crop or zero-pad the time axis to exactly n samples."""
    if sig.shape[-1] >= n:
        return sig[..., :n]
    return np.pad(sig, [(0, 0)] * (sig.ndim - 1) + [(0, n - sig.shape[-1])])


_SOS = {}


def bandpass(sig: np.ndarray, fs: int = FS, lo: float = 0.5, hi: float = 40.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    key = (fs, lo, hi, order)
    if key not in _SOS:
        _SOS[key] = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(_SOS[key], sig, axis=-1)


def clean(sig: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace NaN/inf with 0; return the count so callers can log it."""
    bad = ~np.isfinite(sig)
    n = int(bad.sum())
    if n:
        sig = np.where(bad, 0.0, sig)
    return sig, n


def preprocess(sig: np.ndarray, fs_in: int, lead_names: list[str]) -> tuple[np.ndarray, int]:
    """Raw (T, 12) WFDB array -> filtered (12, 1000) float32 in mV, plus NaN count."""
    x = reorder_leads(np.asarray(sig, dtype=np.float64).T, lead_names)
    x, n_nan = clean(x)
    x = fix_length(resample(x, fs_in), N_SAMPLES)
    x = bandpass(x)
    return x.astype(np.float32), n_nan


def normalize(sig: np.ndarray, clip: float = 10.0, eps: float = 1e-6) -> np.ndarray:
    """Per-lead z-score per recording, clipped to ±clip. Flat leads become all-zero."""
    mu = sig.mean(axis=-1, keepdims=True)
    sd = sig.std(axis=-1, keepdims=True)
    out = np.where(sd > eps, (sig - mu) / np.maximum(sd, eps), 0.0)
    return np.clip(out, -clip, clip).astype(np.float32)


# --------------------------------------------------------------------------- noise
def add_noise_snr(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise per lead at the given SNR (relative to each lead's power)."""
    power = np.mean(sig**2, axis=-1, keepdims=True)
    noise_sd = np.sqrt(power / 10 ** (snr_db / 10))
    return sig + rng.standard_normal(sig.shape) * noise_sd


def baseline_wander(sig: np.ndarray, rng: np.random.Generator, amp: float, fs: int = FS,
                    f_range: tuple[float, float] = (0.1, 0.5)) -> np.ndarray:
    """Add a low-frequency sinusoid (random frequency and phase per lead) of amplitude amp."""
    t = np.arange(sig.shape[-1]) / fs
    f = rng.uniform(*f_range, size=(sig.shape[0], 1))
    phase = rng.uniform(0, 2 * np.pi, size=(sig.shape[0], 1))
    return sig + amp * np.sin(2 * np.pi * f * t + phase)


def augment(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train-only augmentation on a normalised (12, T) signal (Section 5.4).

    Wander amplitude is in normalised (per-lead SD) units, drawn from U(0.05, 0.3).
    """
    x = sig * rng.uniform(0.9, 1.1)
    x = add_noise_snr(x, rng.uniform(20, 30), rng)
    x = baseline_wander(x, rng, amp=rng.uniform(0.05, 0.3))
    drop = rng.random(x.shape[0]) < 0.05
    x[drop] = 0.0
    return x.astype(np.float32)


def corrupt(sig: np.ndarray, snr_db: float, rng: np.random.Generator, wander: bool = True) -> np.ndarray:
    """Test-time corruption for the noise-robustness experiments (Sections 7.2, 8.4).

    Applied to the filtered mV signal *before* normalisation, as a real noisy
    recording would be. Wander amplitude is scaled to the same SNR so both
    components degrade together.
    """
    if not np.isfinite(snr_db):
        return sig
    x = add_noise_snr(sig, snr_db, rng)
    if wander:
        rms = np.sqrt(np.mean(sig**2, axis=-1, keepdims=True))
        x = baseline_wander(x, rng, amp=rms * np.sqrt(2) / 10 ** (snr_db / 20))
    return x.astype(np.float32)
