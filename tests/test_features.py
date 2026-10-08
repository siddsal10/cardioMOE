"""Client A tabular features use only the allow-listed metadata (Section 5.3)."""
import numpy as np
import pandas as pd

from cardiomoe.data.ptbxl import (FEATURE_COLUMNS, FORBIDDEN_COLUMNS, TABULAR_SOURCE_COLUMNS,
                                  tabular_features)


def test_allow_list_is_exact():
    assert TABULAR_SOURCE_COLUMNS == ["age", "sex", "height", "weight"]
    assert FEATURE_COLUMNS == ["age", "sex", "height", "weight", "bmi", "height_missing", "weight_missing"]
    assert not set(TABULAR_SOURCE_COLUMNS) & set(FORBIDDEN_COLUMNS)


def test_meta_features_equal_allow_list(ptbxl_meta, chapman_meta):
    for meta in (ptbxl_meta, chapman_meta):
        feats = [c[2:] for c in meta.columns if c.startswith("x_")]
        assert feats == FEATURE_COLUMNS
        assert not set(meta.columns) & set(FORBIDDEN_COLUMNS)


def test_tabular_cleaning():
    demo = pd.DataFrame({"age": [300, 50, 60, 70], "sex": [0, 1, 0, 1],
                         "height": [180, np.nan, 160, 170], "weight": [80, 70, np.nan, 60]})
    # Extra columns (even leaky ones) must be ignored.
    demo["report"] = "sinus rhythm"
    x, stats = tabular_features(demo)
    assert list(x.columns) == FEATURE_COLUMNS
    assert x.notna().all().all()
    assert x.height_missing.tolist() == [0, 1, 0, 0]
    assert x.weight_missing.tolist() == [0, 0, 1, 0]
    raw_age = x.age * stats["std"]["age"] + stats["mean"]["age"]
    assert np.isclose(raw_age.max(), 90)  # 300 -> capped at 90


def test_chapman_tabular_has_missing_flags(chapman_meta):
    assert (chapman_meta.x_height_missing == 1).all()
    assert (chapman_meta.x_weight_missing == 1).all()
