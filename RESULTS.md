# CardioMoE results log

One section per phase, appended at the end of each phase.

---

## Phase 0: environment, download, manifest (2026-10-07)

**Built**
- Python 3.11.17 venv (`.venv`, installed with Homebrew; the machine only had 3.10), PyTorch 2.14.1 with MPS on an Apple M1 (8 GB), no CUDA. Package `cardiomoe` installed in editable mode via `pyproject.toml`.
- Repository skeleton per Section 4, `requirements.txt`, `configs/base.yaml`, and `src/cardiomoe/utils.py` (config merging, seeding, device selection cuda → mps → cpu, with `--debug` forcing CPU).
- `scripts/download.sh` (`--debug`, `--optional`, `--wget`), `scripts/fetch_physionet.py`, `scripts/fetch_mendeley.py`, and `scripts/verify_data.py` (acceptance checks; writes `data/raw/MANIFEST.md`).

**Acceptance: passed** (`python scripts/verify_data.py`)

| Dataset | Expected | Found | Size |
|---|---|---|---|
| PTB-XL 1.0.3 records | 21,799 | 21,799 (`records100/`: 21,799 `.hea` + 21,799 `.dat`) | 0.55 GB |
| PTB-XL patients | 18,869 | 18,869 | |
| Chapman-Shaoxing 1.0.0 records | ~45k | 45,152 headers / 45,152 `.mat`; **45,150 usable** | 5.46 GB |
| Mendeley ECG images v2 | ~900 | 928 (normal 284, MI 239, history of MI 172, abnormal heartbeat 233) | 0.62 GB |

- `ptbxl_database.csv` (27 columns) and `scp_statements.csv` (71 statements, 44 diagnostic) load. Every record has report text and folds 1–10 are present (fold 9: 2,183 records, fold 10: 2,198).
- A random PTB-XL record (ecg_id 20991) reads with `wfdb.rdsamp` as `(1000, 12)` at 100 Hz in mV, with lead order I, II, III, AVR, AVL, AVF, V1–V6. PTB-XL writes the augmented leads in capitals; the order is the standard one.
- All 45,150 usable Chapman headers parse, have `#Dx`, use the standard lead order, and are 500 Hz × 5,000 samples. 200 random headers also parse with `wfdb.rdheader`, and a random record reads as `(5000, 12)`.

**Deviations and findings for review**
1. **Download method.** PhysioNet's zip endpoint ran at about 21 KB/s here (more than a day for PTB-XL), and wget wasn't installed. PhysioNet data is pulled from PhysioNet's official AWS open-data mirror (`s3://physionet-open`, anonymous HTTPS; same files), with parallel resumable downloads. The plan's `wget -r` recipe is still available as `download.sh --wget`. `records500/` is skipped as the plan allows.
2. **Mendeley images** are fetched through Mendeley Data's public API (no login, SHA-256 checked) instead of a manual "Download All". The resulting folder layout is the same. The MI folder is named "(240x12=2880)" but the published version has 239 images.
3. **Two corrupt Chapman headers** are excluded and logged by the loader:
   - `JS01052`: the record line and lead I's signal line are merged, so the sample count and lead I are missing.
   - `JS23074`: lead I is missing, and there is a stray channel named "84".
4. **Chapman's `ConditionNames_SNOMED-CT.csv` is incomplete.** The headers use 94 distinct SNOMED-CT codes, and 43 of them (16,942 diagnosis mentions) are not in the CSV. Examples are complete/incomplete RBBB, complete LBBB, LAFB, anterior and acute MI, "ST changes", and "left ventricular high voltage". Their names come from the PhysioNet/CinC Challenge 2021 mapping tables and are committed with the source cited in `src/cardiomoe/data/chapman_snomed_supplement.csv`. This affects the Section 5.7 mapping; see the Phase 1 notes.
5. Optional datasets (PTB-XL+, MTSamples) were not downloaded.
6. Disk: about 12 GB free after the downloads. Phase 1 adds about 0.5 GB of PTB-XL signal cache, about 1.1 GB of Chapman signal cache and about 1.7 GB of PTB-XL renders.

### Design decision (2026-10-08)
- **Selective prediction added** (plan Section 8.6b): the target is ≥ 95% exact-match accuracy on the patients the system answers. The rest are referred, and coverage is always reported. The threshold is set on fold 9 only. Reason: 95% accuracy on all PTB-XL cases is not achievable without leakage (the best published models reach about 0.93 macro AUROC), and an abstain option is the honest, clinically meaningful way to reach a 95% operating point.
