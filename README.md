# CardioMoE

Multimodal cardiac diagnosis from three independent modality clients (EHR + ECG signal,
clinical text, ECG image) combined by a learned per-patient gate. When a new dataset
arrives, the clients stay frozen and only the gate is re-fit. The full design is in
[Claude.MD](Claude.MD); per-phase results are in [RESULTS.md](RESULTS.md).

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## Data (Phase 0)

```bash
bash scripts/download.sh            # PTB-XL 1.0.3 (100 Hz), Chapman-Shaoxing 1.0.0, Mendeley ECG images v2
bash scripts/download.sh --debug    # metadata + ~200 records per dataset, for a quick end-to-end check
.venv/bin/python scripts/verify_data.py   # re-run the checks; writes data/raw/MANIFEST.md
```

All datasets are open access (CC BY 4.0) and need no login. PhysioNet data is pulled from
PhysioNet's AWS open-data mirror (`s3://physionet-open`), and downloads resume if interrupted.
`--wget` uses the original `wget -r` recipe against physionet.org instead.
Disk: about 0.6 GB for PTB-XL (records100 only) and a few GB for Chapman.

## Status

| Phase | Status |
|---|---|
| 0 Environment, download, manifest | done (see RESULTS.md) |
| 1 Data prep | code written + debug-tested; full run pending review |
| 2–10 | not started |
