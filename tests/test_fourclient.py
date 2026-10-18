"""Four-client system: data partition, model shapes, aggregation constraints, metrics."""
import numpy as np
import pytest
import torch

from cardiomoe.fourclient.gate import DynamicGate, EqualWeights, StaticWeights
from cardiomoe.fourclient.metrics import class_metrics, confusion, full_report
from cardiomoe.fourclient.models import ARCHS, SignalClient
from cardiomoe.utils import load_config


@pytest.fixture(scope="module")
def fc():
    return load_config("fourclient")["fourclient"]


def test_partition_disjoint_and_patient_separated(fc, ptbxl_meta):
    folds = {c: set(f) for c, f in fc["partition"].items()}
    used = [f for fs in folds.values() for f in fs]
    assert len(used) == len(set(used)) == 8
    assert set(used) == set(range(1, 9))
    assert fc["val_fold"] == 9 and fc["test_fold"] == 10
    groups = {c: set(ptbxl_meta[ptbxl_meta.strat_fold.isin(f)].patient_id) for c, f in folds.items()}
    groups["val"] = set(ptbxl_meta[ptbxl_meta.strat_fold == 9].patient_id)
    groups["test"] = set(ptbxl_meta[ptbxl_meta.strat_fold == 10].patient_id)
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not groups[a] & groups[b], f"patients shared between {a} and {b}"


@pytest.mark.parametrize("arch", list(ARCHS))
def test_models_output_shape(arch):
    m = SignalClient(arch).eval()
    out = m({"signal": torch.randn(3, 12, 1000)})
    assert out.logits.shape == (3, 5) and out.embedding.shape == (3, 256)
    assert torch.isfinite(out.logits).all()


def _toy(n=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, 5, generator=g) < 0.3).float()
    good = (0.7 * y + 0.3 * torch.rand(n, 5, generator=g)).clamp(0.01, 0.99)
    noise = torch.rand(n, 4, 5, generator=g)
    P = torch.cat([good.unsqueeze(1), noise[:, 1:]], dim=1)  # client 0 informative, 1-3 random
    return P, y


@pytest.mark.parametrize("cls", [EqualWeights, StaticWeights, DynamicGate])
def test_weights_on_simplex(cls):
    P, y = _toy()
    m = cls().fit(P, y, np.arange(len(P)))
    w = m.weights(P)
    assert w.shape == (len(P), 4)
    assert (w >= 0).all()
    assert torch.allclose(w.sum(1), torch.ones(len(P)), atol=1e-5)
    p = m.predict(P)
    assert p.shape == (len(P), 5) and ((p >= 0) & (p <= 1)).all()


def test_static_weights_find_informative_client():
    P, y = _toy()
    w = StaticWeights().fit(P, y).w
    assert int(w.argmax()) == 0 and w[0] > 0.7


def test_confusion_and_metrics():
    y = np.array([1, 1, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 0])
    cm = confusion(y, pred)
    assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}
    m = class_metrics(cm)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["specificity"] == pytest.approx(2 / 3)
    assert m["accuracy"] == 0.6


def test_full_report_perfect():
    y = np.array([[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1], [1, 0, 0, 0, 0]], float)
    r = full_report(y, y * 0.9 + 0.05, np.full(5, 0.5))
    assert r["summary"]["exact_match_accuracy"] == 1.0
    assert r["summary"]["macro_f1"] == 1.0
