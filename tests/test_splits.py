"""No patient may appear in more than one split (Section 2.2, 5.2)."""
from cardiomoe.data.ptbxl import TEST_FOLD, TRAIN_FOLDS, VAL_FOLD


def test_ptbxl_no_patient_overlap(ptbxl_meta):
    pats = {s: set(g.patient_id) for s, g in ptbxl_meta.groupby("split")}
    assert set(pats) <= {"train", "val", "test"}
    for a in pats:
        for b in pats:
            if a < b:
                assert not pats[a] & pats[b], f"patients shared between {a} and {b}"


def test_ptbxl_official_folds(ptbxl_meta):
    m = ptbxl_meta
    assert set(m[m.split == "train"].strat_fold) <= set(TRAIN_FOLDS)
    assert set(m[m.split == "val"].strat_fold) <= {VAL_FOLD}
    assert set(m[m.split == "test"].strat_fold) <= {TEST_FOLD}


def test_ptbxl_every_record_has_a_label(ptbxl_meta):
    y = ptbxl_meta[[c for c in ptbxl_meta if c.startswith("y_")]]
    assert (y.sum(axis=1) >= 1).all()
