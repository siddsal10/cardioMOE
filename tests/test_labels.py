import numpy as np
import pandas as pd

from cardiomoe.data import labels as L


def test_ptbxl_superclass_mapping():
    scp = pd.DataFrame({"diagnostic": [1, 1, 1, np.nan], "diagnostic_class": ["MI", "NORM", "CD", np.nan]},
                       index=["IMI", "NORM", "CLBBB", "SR"])
    cmap = L.diagnostic_map(scp)
    assert L.ptbxl_superclasses("{'IMI': 100.0, 'SR': 0.0, 'CLBBB': 50.0}", cmap) == ["MI", "CD"]
    assert L.ptbxl_superclasses("{'SR': 0.0}", cmap) == []
    assert L.multi_hot(["MI", "CD"]).tolist() == [0, 1, 0, 1, 0]


def test_chapman_normal_is_sinus_only():
    n, y = L.chapman_labels(["426783006"])
    assert n == 1 and y.tolist() == [1, 0, 0, 0, 0]
    n, y = L.chapman_labels(["426783006", "59118001"])  # SR + RBBB
    assert n == 0 and y.tolist() == [0, 0, 0, 1, 0]
    n, y = L.chapman_labels(["426177001"])  # sinus bradycardia alone: abnormal, no superclass
    assert n == 0 and y.tolist() == [0, 0, 0, 0, 0]


def test_chapman_ambiguous_codes_are_unknown():
    _, y = L.chapman_labels(["164917005"])  # abnormal Q wave
    assert y[L.SUPERCLASSES.index("MI")] == L.UNKNOWN
    _, y = L.chapman_labels(["164917005", "164865005"])  # ...but a positive MI code wins
    assert y[L.SUPERCLASSES.index("MI")] == 1


def test_chapman_positive_and_ambiguous_disjoint():
    assert not set(L.CHAPMAN_POSITIVE) & set(L.CHAPMAN_AMBIGUOUS)
    assert not set(L.CHAPMAN_POSITIVE) & L.NORMAL_CODES


def test_pk_folder_labels():
    name, n, y = L.pk_folder_label("Normal Person ECG Images (284x12=3408)")
    assert (name, n) == ("normal", 1) and y.tolist() == [1, 0, 0, 0, 0]
    _, n, y = L.pk_folder_label("ECG Images of Patient that have History of MI (172x12=2064)")
    assert n == 0 and y[1] == 1
    _, n, y = L.pk_folder_label("ECG Images of Patient that have abnormal heartbeat (233x12=2796)")
    assert n == 0 and y[0] == 0 and (y[1:] == L.UNKNOWN).all()
