"""PTB-XL superclass definitions and SCP-code -> superclass mapping (Section 5.1).

Label convention: float vector over SUPERCLASSES with 1 = present, 0 = absent
(-1 = unknown is still honoured by the loss and metrics).
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
UNKNOWN = -1.0


# --------------------------------------------------------------------------- PTB-XL
def diagnostic_map(scp_statements: pd.DataFrame) -> dict[str, str]:
    """SCP code -> diagnostic superclass, using rows with diagnostic == 1."""
    diag = scp_statements[scp_statements.diagnostic == 1]
    return diag.diagnostic_class.to_dict()


def ptbxl_superclasses(scp_codes: str | dict, code_map: dict[str, str]) -> list[str]:
    codes = ast.literal_eval(scp_codes) if isinstance(scp_codes, str) else scp_codes
    return sorted({code_map[c] for c in codes if c in code_map}, key=SUPERCLASSES.index)


def multi_hot(classes: list[str]) -> np.ndarray:
    y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for c in classes:
        y[SUPERCLASSES.index(c)] = 1.0
    return y
