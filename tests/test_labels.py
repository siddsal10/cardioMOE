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
