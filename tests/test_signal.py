import numpy as np

from cardiomoe.data.signal import LEADS, augment, corrupt, normalize, preprocess, reorder_leads


def test_preprocess_500hz_to_100hz_shape():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((5000, 12))
    raw[10, 3] = np.nan
    x, n_nan = preprocess(raw, 500, LEADS)
    assert x.shape == (12, 1000) and x.dtype == np.float32 and n_nan == 1
    assert np.isfinite(x).all()


def test_reorder_case_insensitive():
    names = ["I", "II", "III", "AVR", "AVL", "AVF"] + [f"V{i}" for i in range(1, 7)]
    sig = np.arange(12)[:, None] * np.ones((12, 5))
    shuffled = sig[::-1]
    out = reorder_leads(shuffled, names[::-1])
    assert (out[:, 0] == np.arange(12)).all()


def test_normalize_flat_lead_and_clip():
    x = np.zeros((12, 1000), np.float32)
    x[0, 500] = 1e6
    n = normalize(x)
    assert np.abs(n).max() <= 10 and (n[1:] == 0).all()


def test_augment_and_corrupt_shapes():
    rng = np.random.default_rng(0)
    x = normalize(np.sin(np.linspace(0, 60, 1000))[None].repeat(12, 0))
    assert augment(x, rng).shape == (12, 1000)
    noisy = corrupt(x, 10, rng, wander=False)
    snr = 10 * np.log10(np.mean(x**2) / np.mean((noisy - x) ** 2))
    assert 9 < snr < 11
    assert corrupt(x, float("inf"), rng) is x
