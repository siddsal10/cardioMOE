"""Superclass definitions and label mapping tables (Sections 5.1, 5.7, 5.8).

Label convention used everywhere: float vector over SUPERCLASSES with
1 = present, 0 = absent, -1 = unknown (masked out of the loss and of per-class metrics).
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


# --------------------------------------------------------------------------- Chapman
SINUS_RHYTHM = "426783006"
# The ecg-arrhythmia condition list contains no separate "normal ECG" code, so a record
# is normal iff its only code is sinus rhythm.
NORMAL_CODES = {SINUS_RHYTHM}

# Codes that map unambiguously to a superclass (Section 5.7).
CHAPMAN_POSITIVE = {
    # MI: myocardial infarction (wall-specific variants share this code), anterior MI, acute MI
    "164865005": "MI",
    "54329005": "MI",
    "57054005": "MI",
    # CD: LBBB (incl. left anterior/posterior fascicular variants, same code), complete LBBB,
    # RBBB, complete and incomplete RBBB, unspecified bundle branch block, LAFB, LPFB,
    # 1st/2nd (incl. Mobitz II)/3rd-degree and unspecified AV block, intraventricular block / IVCD
    "164909002": "CD",
    "733534002": "CD",
    "59118001": "CD",
    "713427006": "CD",
    "713426002": "CD",
    "6374002": "CD",
    "445118002": "CD",
    "445211001": "CD",
    "426183003": "CD",
    "270492004": "CD",
    "195042002": "CD",
    "54016002": "CD",
    "28189009": "CD",
    "27885002": "CD",
    "233917008": "CD",
    "698252002": "CD",
    # HYP: LVH, RVH
    "164873001": "HYP",
    "89792004": "HYP",
    # STTC: ST depression, ST elevation, T-wave abnormal, T-wave inversion, ST-T change
    "429622005": "STTC",
    "164931005": "STTC",
    "164934002": "STTC",
    "59931005": "STTC",
    "428750005": "STTC",
    "55930002": "STTC",           # "S T changes"
}

# Codes that *might* imply a superclass but not cleanly. A record carrying one of these
# (and no positive code for that class) gets label UNKNOWN for that class, not 0.
CHAPMAN_AMBIGUOUS = {
    "164917005": ["MI"],          # abnormal Q wave
    "164942001": ["MI", "CD"],    # fragmented QRS
    "164930006": ["STTC"],        # "ST extension" (ST interval abnormal)
    "111975006": ["STTC"],        # QT prolongation (STTC in PTB-XL)
    "428417006": ["STTC"],        # early repolarisation
    "164937009": ["STTC"],        # U wave
    "164947007": ["CD"],          # PR prolongation (1st-degree AVB equivalent, not coded as such)
    "74390002": ["CD"],           # WPW (CD in PTB-XL, not in the plan's CD list)
    "195060002": ["CD"],          # ventricular pre-excitation
    "446358003": ["HYP"],         # right atrial hypertrophy (HYP in PTB-XL, not LVH/RVH)
    "164912004": ["HYP"],         # P wave change (possible atrial enlargement)
    # Codes found in the headers but not in the dataset's own condition list (names from
    # chapman_snomed_supplement.csv):
    "55827005": ["HYP"],          # left ventricular high voltage: a voltage criterion, not an LVH
                                  # diagnosis (VCLVH is a form statement in PTB-XL)
    "446813000": ["HYP"],         # left atrial hypertrophy
    "67741000119109": ["HYP"],    # left atrial enlargement
    "67751000119106": ["HYP"],    # right atrial high voltage
    "251223006": ["HYP"],         # tall P wave
    "251205003": ["HYP"],         # prolonged P wave
    "365413008": ["MI"],          # poor R-wave progression
    "251120003": ["CD"],          # incomplete LBBB (not in the plan's CD list)
    "50799005": ["CD"],           # AV dissociation
    "65778007": ["CD"],           # sinoatrial block
    "49578007": ["CD"],           # short PR interval (possible pre-excitation)
    "418818005": ["CD", "STTC"],  # Brugada pattern
    "77867006": ["STTC"],         # short QT
    "10370003": ["MI", "STTC", "CD", "HYP"],  # paced rhythm masks the native morphology
}
# Every other code (rhythms, ectopy, axis, rotation, low voltage) is a non-diagnostic
# finding: it rules out NORM but says nothing positive about MI/STTC/CD/HYP.


def chapman_labels(dx: list[str]) -> tuple[int, np.ndarray]:
    """Return (binary_normal, superclass vector with UNKNOWN entries)."""
    codes = set(dx)
    normal = int(bool(codes) and codes <= NORMAL_CODES)
    y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    y[0] = normal
    for c in codes:
        for cls in CHAPMAN_AMBIGUOUS.get(c, []):
            y[SUPERCLASSES.index(cls)] = UNKNOWN
    for c in codes:
        if c in CHAPMAN_POSITIVE:
            y[SUPERCLASSES.index(CHAPMAN_POSITIVE[c])] = 1.0
    return normal, y


def chapman_condition_names(raw_dir) -> pd.DataFrame:
    """The dataset's ConditionNames_SNOMED-CT.csv plus the supplement for codes it omits."""
    from pathlib import Path

    own = pd.read_csv(Path(raw_dir) / "ConditionNames_SNOMED-CT.csv", encoding="utf-8-sig", dtype={"Snomed_CT": str})
    own["source"] = "ConditionNames_SNOMED-CT.csv"
    sup = pd.read_csv(Path(__file__).with_name("chapman_snomed_supplement.csv"), comment="#", dtype={"Snomed_CT": str})
    sup["source"] = "Challenge 2021 dx_mapping"
    return pd.concat([own, sup], ignore_index=True)


def chapman_label_table(condition_names: pd.DataFrame) -> pd.DataFrame:
    """Full mapping, one row per (acronym, code), for data/processed/chapman_label_map.csv."""
    rows = []
    for _, r in condition_names.iterrows():
        code = str(r["Snomed_CT"])
        if code in NORMAL_CODES:
            role = "NORM (if only code)"
        elif code in CHAPMAN_POSITIVE:
            role = CHAPMAN_POSITIVE[code]
        elif code in CHAPMAN_AMBIGUOUS:
            role = "unknown for " + "/".join(CHAPMAN_AMBIGUOUS[code])
        else:
            role = "non-diagnostic (rules out NORM only)"
        rows.append({"acronym": r["Acronym Name"].strip(), "name": r["Full Name"].strip(),
                     "snomed_ct": code, "maps_to": role, "name_source": r.get("source", "")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Mendeley
# Folder-name prefix -> (binary_normal, superclass vector). Abnormal heartbeat rules out
# NORM but its superclass is unknown (Section 5.8).
_U = UNKNOWN
PK_CLASSES = {
    "Normal Person": ("normal", 1, [1, 0, 0, 0, 0]),
    "ECG Images of Myocardial Infarction": ("mi", 0, [0, 1, _U, _U, _U]),
    "ECG Images of Patient that have History of MI": ("history_mi", 0, [0, 1, _U, _U, _U]),
    "ECG Images of Patient that have abnormal heartbeat": ("abnormal_heartbeat", 0, [0, _U, _U, _U, _U]),
}


def pk_folder_label(folder_name: str) -> tuple[str, int, np.ndarray]:
    for prefix, (name, normal, y) in PK_CLASSES.items():
        if folder_name.startswith(prefix):
            return name, normal, np.asarray(y, dtype=np.float32)
    raise ValueError(f"unrecognised Mendeley folder: {folder_name}")
