"""Phase 0 acceptance: verify the raw downloads and write data/raw/MANIFEST.md.

Checks (Section 3.1 of the plan):
  * file counts match the published numbers (±1%)
  * ptbxl_database.csv and scp_statements.csv load
  * a random PTB-XL WFDB record reads with wfdb.rdsamp and has shape (1000, 12) at 100 Hz

    python scripts/verify_data.py           # full check, exits non-zero on failure
    python scripts/verify_data.py --debug   # partial (debug) download: report, don't enforce counts
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from pathlib import Path

import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def same_leads(names) -> bool:
    """Lead order check; datasets differ only in case (PTB-XL writes AVR/AVL/AVF)."""
    return [n.upper() for n in names] == [n.upper() for n in STD_LEADS]

# Published sizes from the PTB-XL PhysioNet page.
EXPECTED = {
    "ptbxl_records": 21799,
    "ptbxl_patients": 18869,
}


class Report:
    def __init__(self, enforce: bool):
        self.enforce, self.rows, self.failed = enforce, [], False

    def check(self, name: str, ok: bool, detail: str, hard: bool = False) -> None:
        status = "PASS" if ok else ("FAIL" if (self.enforce or hard) else "WARN")
        self.failed |= status == "FAIL"
        self.rows.append((name, status, detail))
        print(f"[{status}] {name}: {detail}", flush=True)

    def count(self, name: str, got: int, key: str) -> None:
        want = EXPECTED[key]
        self.check(name, abs(got - want) <= 0.01 * want, f"{got} (expected {want} ±1%)")


def verify_ptbxl(r: Report, rng: random.Random) -> dict:
    d = RAW / "ptbxl"
    db = pd.read_csv(d / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(d / "scp_statements.csv", index_col=0)
    r.check("ptbxl_database.csv loads", True, f"{len(db)} rows, {db.shape[1]} columns")
    r.check("scp_statements.csv loads", True,
            f"{len(scp)} statements, {int(scp.diagnostic.fillna(0).sum())} diagnostic", hard=True)
    r.count("PTB-XL records in database", len(db), "ptbxl_records")
    r.count("PTB-XL unique patients", db.patient_id.nunique(), "ptbxl_patients")
    r.check("PTB-XL has report text", db.report.notna().all(),
            f"{db.report.notna().sum()} non-empty reports")
    r.check("PTB-XL strat_fold 1..10", set(db.strat_fold) == set(range(1, 11)),
            ", ".join(map(str, sorted(db.strat_fold.unique()))))

    heas = sorted((d / "records100").rglob("*_lr.hea"))
    dats = list((d / "records100").rglob("*_lr.dat"))
    r.count("PTB-XL records100 .hea files", len(heas), "ptbxl_records")
    r.count("PTB-XL records100 .dat files", len(dats), "ptbxl_records")

    have = {int(p.stem.split("_")[0]) for p in heas}
    present = db[db.index.isin(have)]
    ecg_id = rng.choice(list(present.index))
    sig, meta = wfdb.rdsamp(str(d / present.loc[ecg_id, "filename_lr"]))
    r.check("random PTB-XL record reads", sig.shape == (1000, 12) and meta["fs"] == 100,
            f"ecg_id {ecg_id}: shape {sig.shape}, fs {meta['fs']} Hz, units {meta['units'][0]}", hard=True)
    r.check("PTB-XL lead order", same_leads(meta["sig_name"]), ", ".join(meta["sig_name"]), hard=True)
    return {"records": len(db), "patients": db.patient_id.nunique(), "hea": len(heas),
            "folds": db.strat_fold.value_counts().sort_index().to_dict()}


def write_manifest(r: Report, info: dict, debug: bool) -> None:
    du = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9
    lines = [
        "# Raw data manifest",
        "",
        f"Verified: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `scripts/verify_data.py`"
        + (" (**--debug partial download**)" if debug else ""),
        "",
        "| Dataset | Version | Local path | Source | Size on disk |",
        "|---|---|---|---|---|",
        f"| PTB-XL | 1.0.3 | `data/raw/ptbxl` (records100 only) | physionet.org/content/ptb-xl/1.0.3 via s3://physionet-open | {du(RAW/'ptbxl'):.2f} GB |",
        "",
        "## Counts",
        "",
        f"- PTB-XL: {info['ptbxl']['records']} records, {info['ptbxl']['patients']} patients, "
        f"{info['ptbxl']['hea']} records100 headers. Records per strat_fold: {info['ptbxl']['folds']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
        *[f"| {n} | {s} | {d.replace('|', '/')} |" for n, s, d in r.rows],
        "",
    ]
    (RAW / "MANIFEST.md").write_text("\n".join(lines))
    print(f"wrote {RAW / 'MANIFEST.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    r = Report(enforce=not args.debug)
    info = {"ptbxl": verify_ptbxl(r, rng)}
    write_manifest(r, info, args.debug)
    if r.failed:
        sys.exit("Phase 0 verification FAILED")
    print("Phase 0 verification passed")


if __name__ == "__main__":
    main()
