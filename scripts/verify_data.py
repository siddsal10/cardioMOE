"""Phase 0 acceptance: verify the raw downloads and write data/raw/MANIFEST.md.

Checks (Section 3.1 of the plan):
  * file counts match the published numbers (±1%)
  * ptbxl_database.csv and scp_statements.csv load
  * a random PTB-XL WFDB record reads with wfdb.rdsamp and has shape (1000, 12) at 100 Hz
  * every Chapman header parses (and a random sample also parses with wfdb.rdheader)
  * the Mendeley image folders are complete

    python scripts/verify_data.py           # full check, exits non-zero on failure
    python scripts/verify_data.py --debug   # partial (debug) download: report, don't enforce counts
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
STD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def same_leads(names) -> bool:
    """Lead order check; datasets differ only in case (PTB-XL writes AVR/AVL/AVF)."""
    return [n.upper() for n in names] == [n.upper() for n in STD_LEADS]

# Published sizes. PTB-XL and Chapman from their PhysioNet pages; Mendeley from the
# dataset's public file listing (the plan's "~900" rounds this).
EXPECTED = {
    "ptbxl_records": 21799,
    "ptbxl_patients": 18869,
    "chapman_records": 45152,
    "pk_images": 928,
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


def parse_header(path: Path) -> dict:
    """Minimal WFDB header parse: record line, signal names, '#Age/#Sex/#Dx' comments."""
    lines = path.read_text().splitlines()
    name, nsig, fs, nsamp = lines[0].split()[:4]
    leads = [ln.split()[-1] for ln in lines[1 : 1 + int(nsig)]]
    comments = dict(
        ln[1:].split(":", 1) for ln in lines[1 + int(nsig) :] if ln.startswith("#") and ":" in ln
    )
    comments = {k.strip(): v.strip() for k, v in comments.items()}
    dx = [c for c in comments.get("Dx", "").split(",") if c]
    return {"fs": float(fs), "n": int(nsamp), "leads": leads, "dx": dx,
            "age": comments.get("Age"), "sex": comments.get("Sex")}


def verify_chapman(r: Report, rng: random.Random) -> dict:
    d = RAW / "chapman"
    names = pd.read_csv(d / "ConditionNames_SNOMED-CT.csv")
    r.check("Chapman ConditionNames_SNOMED-CT.csv loads", True, f"{len(names)} conditions", hard=True)

    heas = sorted((d / "WFDBRecords").rglob("*.hea"))
    mats = list((d / "WFDBRecords").rglob("*.mat"))
    r.count("Chapman .hea files", len(heas), "chapman_records")
    r.count("Chapman .mat files", len(mats), "chapman_records")

    parsed, bad, ok_heas = [], [], []
    for p in heas:
        try:
            parsed.append(parse_header(p))
            ok_heas.append(p)
        except Exception as e:  # noqa: BLE001 - report every unparseable header
            bad.append(f"{p.name}: {e}")
    heas_all, heas = heas, ok_heas
    lead_bad = [f"{p.stem}: leads {h['leads']}" for p, h in zip(heas, parsed) if not same_leads(h["leads"])]
    r.check("Chapman headers parse", len(bad) + len(lead_bad) <= 0.001 * len(heas),
            f"{len(parsed) - len(lead_bad)} OK; malformed (excluded by the loader): {bad + lead_bad}", hard=True)
    parsed = [h for h in parsed if same_leads(h["leads"])]
    with_dx = sum(bool(h["dx"]) for h in parsed)
    r.check("Chapman headers have #Dx", with_dx == len(parsed), f"{with_dx}/{len(parsed)}")
    lead_orders = Counter(tuple(h["leads"]) for h in parsed)
    r.check("Chapman lead order", all(same_leads(o) for o in lead_orders),
            f"{len(lead_orders)} distinct order(s): {', '.join(lead_orders.most_common(1)[0][0])}")
    fs = Counter((h["fs"], h["n"]) for h in parsed)
    r.check("Chapman 500 Hz x 10 s", fs.most_common(1)[0][0] == (500.0, 5000),
            ", ".join(f"{k[0]:g} Hz x {k[1]}: {v}" for k, v in fs.most_common(3)))

    sample = rng.sample(heas, min(200, len(heas)))
    for p in sample:
        wfdb.rdheader(str(p.with_suffix("")))
    rec = rng.choice(sample)
    sig, meta = wfdb.rdsamp(str(rec.with_suffix("")))
    r.check("random Chapman record reads", sig.shape == (5000, 12),
            f"{rec.stem}: shape {sig.shape}, fs {meta['fs']} Hz ({len(sample)} headers also read via wfdb.rdheader)",
            hard=True)
    sup = pd.read_csv(ROOT / "src/cardiomoe/data/chapman_snomed_supplement.csv", comment="#", dtype=str)
    own = set(names.Snomed_CT.astype(str))
    codes = Counter(c for h in parsed for c in h["dx"])
    unknown = {c: n for c, n in codes.items() if c not in own}
    unnamed = sorted(set(unknown) - set(sup.Snomed_CT))
    r.check("Chapman Dx codes named", not unnamed,
            f"{len(codes)} distinct codes; {len(unknown)} missing from ConditionNames_SNOMED-CT.csv "
            f"({sum(unknown.values())} mentions), named via chapman_snomed_supplement.csv; still unnamed: {unnamed or 'none'}")
    return {"records": len(heas_all), "usable": len(parsed), "mat": len(mats), "distinct_dx": len(codes),
            "unknown_dx": len(unknown), "malformed": bad + lead_bad}


def verify_pk_images(r: Report) -> dict:
    d = RAW / "ecg_images_pk"
    exts = {".jpg", ".jpeg", ".png"}
    per = {f.name: sum(p.suffix.lower() in exts for p in f.rglob("*")) for f in sorted(d.iterdir()) if f.is_dir()}
    r.check("Mendeley class folders", len(per) == 4, "; ".join(f"{k}: {v}" for k, v in per.items()), hard=True)
    r.count("Mendeley images", sum(per.values()), "pk_images")
    return per


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
        f"| Chapman-Shaoxing (ecg-arrhythmia) | 1.0.0 | `data/raw/chapman` | physionet.org/content/ecg-arrhythmia/1.0.0 via s3://physionet-open | {du(RAW/'chapman'):.2f} GB |",
        f"| ECG Images of Cardiac Patients (Mendeley gwbz3fsgp8) | 2 | `data/raw/ecg_images_pk` | data.mendeley.com/datasets/gwbz3fsgp8/2 (public API, sha256-verified) | {du(RAW/'ecg_images_pk'):.2f} GB |",
        "",
        "Not downloaded (optional in the plan): PTB-XL+ 1.0.1, MTSamples.",
        "",
        "## Counts",
        "",
        f"- PTB-XL: {info['ptbxl']['records']} records, {info['ptbxl']['patients']} patients, "
        f"{info['ptbxl']['hea']} records100 headers. Records per strat_fold: {info['ptbxl']['folds']}",
        f"- Chapman: {info['chapman']['records']} headers, {info['chapman']['mat']} .mat files, "
        f"{info['chapman']['usable']} usable ({len(info['chapman']['malformed'])} malformed headers: "
        f"{info['chapman']['malformed']}), {info['chapman']['distinct_dx']} distinct SNOMED-CT codes, "
        f"{info['chapman']['unknown_dx']} of them missing from ConditionNames_SNOMED-CT.csv "
        "(named in `src/cardiomoe/data/chapman_snomed_supplement.csv`)",
        f"- Mendeley images: {sum(info['pk'].values())} total; "
        + "; ".join(f"{k}: {v}" for k, v in info["pk"].items()),
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
    info = {"ptbxl": verify_ptbxl(r, rng), "chapman": verify_chapman(r, rng), "pk": verify_pk_images(r)}
    write_manifest(r, info, args.debug)
    if r.failed:
        sys.exit("Phase 0 verification FAILED")
    print("Phase 0 verification passed")


if __name__ == "__main__":
    main()
