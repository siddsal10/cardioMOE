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


def test_chapman_split_disjoint(chapman_meta):
    assert set(chapman_meta.split) <= {"adapt_pool", "adapt_test"}
    assert chapman_meta.index.is_unique  # one record per patient in this dataset
    if len(chapman_meta) > 1000:
        frac = (chapman_meta.split == "adapt_pool").mean()
        assert 0.28 < frac < 0.32


def test_pk_split_disjoint_and_stratified(pk_meta):
    assert pk_meta.index.is_unique
    assert set(pk_meta.split) == {"adapt", "test"}
    if len(pk_meta) > 500:
        frac = pk_meta.groupby("cls").split.apply(lambda s: (s == "adapt").mean())
        assert frac.between(0.27, 0.33).all()
