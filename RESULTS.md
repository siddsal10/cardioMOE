# CardioMoE results log

One section per phase, appended at the end of each phase.

> **Note (2026-10-17):** the project was redesigned into the four-client system (last section).
> The Phase 0–3 sections below describe the earlier three-modality design. Its code, data
> (Chapman, Mendeley, rendered images, translations) and checkpoints were removed, and they
> are kept here only as history.

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

---

## Phase 2: Client A, EHR + ECG signal (2026-10-14)

**Built:** `clients/base.py` (`ClientOutput`, `BaseClient` with a fold-9 temperature buffer, masked BCE), `clients/client_a.py`, `data/datasets.py` (one dataset class for every client × dataset, with seeded per-worker augmentation), `train/common.py` (training loop, early stopping, compact checkpoints), `train/train_client.py`, `fusion/calibration.py`, `eval/metrics.py`.

**Model:** 1D ResNet. Stem conv k = 7 with 64 channels, then 4 stages with [64, 128, 256, 256] channels and [3, 4, 6, 3] residual blocks (xresnet50-style depth, basic blocks, k = 5). Average and max pooling are concatenated to 512 → 256. The tabular MLP is 7 → 64 → 64, and local fusion is 320 → 256 (embedding) → 5. 6.75M trainable parameters.

**Training:** AdamW (lr 1e-3, wd 1e-2), OneCycle, batch 128, `pos_weight` from train prevalence, height/weight dropout p = 0.5, and the Section 5.4 signal augmentations. Trained on an Apple M1 GPU (MPS) at about 100 s per epoch. Early stopping (patience 8) stopped at epoch 20; the best epoch was 12.

**Acceptance: passed.** Fold-9 macro AUROC **0.906** (target ≥ 0.88).

| Fold 9 | macro AUROC | NORM | MI | STTC | CD | HYP | macro AUPRC | macro F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| Client A | **0.906** | 0.935 | 0.917 | 0.929 | 0.921 | 0.828 | 0.761 | 0.707 | 0.138 |

- **Calibration:** T = 1.22. ECE was 0.143 before scaling and 0.149 after, so one temperature does not help here. Training with `pos_weight` shifts probabilities upwards per class, which a single scalar cannot correct. The gate and the selective-prediction thresholds are fitted on these calibrated outputs, so this is reported rather than worked around.
- **Not yet done:** 500 Hz input and ensembling, which could raise this toward the 0.92–0.93 the literature reports. They're outside the plan as written.

---

## Phase 1 completion note (2026-10-14)

- Translation cache: **100%** of 21,799 reports (German 10,882 via opus-mt-de-en, English 9,295 kept, Swedish 1,622 via opus-mt-sv-en). Sources are lowercased before translation; the cache is saved every 1,024 reports so runs can resume.
- Leakage-term list v1 (the plan's list, scope "nonrhythm") has 136 terms, saved to `data/processed/mask_terms_v1.txt`. 78.4% of reports contain at least one `[MASK]`.
- All 21,388 PTB-XL and 45,150 Chapman images are rendered. Spot-check sheets: `outputs/figures/render_samples.png` (PTB-XL) and `render_samples_chapman.png`.
- Mendeley: the split has no patient IDs, so it cannot be guaranteed patient-level (plan Section 10).

---

## Phase 3: Client B, clinical text, plus the leakage audit (2026-10-14)

**Model:** Bio_ClinicalBERT with max length 128, mean pooling over non-padding tokens, then 768 → 256 (embedding) → 5. Embeddings and the first 8 layers are frozen (28.5M trainable parameters). AdamW with lr 2e-5 for the encoder and 1e-3 for the head, linear warmup 10%, batch 32, early stopping with patience 3. The model uses eager attention because PyTorch's fused attention on MPS does not support dropout. About 9 minutes per epoch on MPS.

| Fold 9 | macro AUROC | NORM | MI | STTC | CD | HYP | macro AUPRC | macro F1 | Brier | epochs (best) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Client B, masked (default)** | **0.929** | 0.947 | 0.932 | 0.917 | 0.950 | 0.899 | 0.814 | 0.753 | 0.099 | 8 (5) |
| Client B, raw (**leaky upper bound**) | 0.940 | 0.952 | 0.938 | 0.920 | 0.959 | 0.931 | 0.847 | 0.785 | 0.091 | 8 (5) |

**Leakage audit:**

| Iteration | Mask list | Client B masked, fold-9 macro AUROC | TF-IDF + LR proxy (masked / raw) | Action |
|---|---|---|---|---|
| 1 | v1, the plan's list (136 terms) | **0.929**, below the 0.95 threshold | 0.929 / 0.938 | Plan rule satisfied; v1 stays the default |
| (preview) | v2, audit extension (`AUDIT_TERMS[2]`) | not trained | 0.912 / – | Ready (`--mask_version 2`); not applied |

- **The plan's criterion is met** (masked < 0.95), so `report_masked` v1 is used for all fusion experiments.
- **Masking removes only part of the leakage.** The masked text model still beats the ECG-signal model (0.929 vs 0.906). A bag-of-words model on the masked text gets 0.929, against 0.938 on raw text. The remaining signal is structural: untranslated German compounds ("rechtsschenkelblock", "linkshypertrophie", "st-senkung"), translator variants ("leg block", "st-lift", "forecourt" for atrium), qualifiers ("old", "chronic", "non-specific"), and the presence and position of `[MASK]` itself ("right [MASK]", "[MASK] wave"). Because PTB-XL's labels were written from these reports, no word list can remove this completely (plan Section 10). Client B's results should be read as **partly leaky**, and the gate's per-client weights should be interpreted with that in mind.
- **Calibration:** T = 1.38 (masked) and 1.27 (raw). ECE 0.075 → 0.083 (masked) and 0.071 → 0.078 (raw), so text calibration was already good.

---

## Four-client system with weighted gated aggregation (2026-10-15)

**Redesign** (approved by the project owner, CLAUDE.md Section 0): four ECG-signal clients with different architectures, each trained on its own quarter of the PTB-XL training folds, combined by `P_final = Σ wᵢPᵢ` (wᵢ ≥ 0, Σ wᵢ = 1). Task unchanged: multi-label, 5 superclasses. Code: `src/cardiomoe/fourclient/`; outputs: `outputs/fourclient/`.

**Setup.** Folds 1–8 split by pairs: A = {1,2}, B = {3,4}, C = {5,6}, D = {7,8} (about 4,270 records each, patient-separated; tested). Fold 9 (2,146) is used for early stopping, per-client temperature, per-class thresholds and the aggregation weights. Fold 10 (2,158) was evaluated **once**. Every client used the same recipe: 100 Hz signal only, Section 5.4 preprocessing and augmentation, weighted BCE, AdamW lr 1e-3 / wd 1e-2, OneCycle, batch 128, early stopping (patience 8). Seed 42 only.

**Aggregator selection** (patient-grouped 5-fold CV inside fold 9; the dynamic gate needed ≥ 0.002 AUROC over static to be chosen):

| Combiner | CV macro AUROC | CV BCE |
|---|---|---|
| Equal weights | 0.908 | 0.377 |
| Static learned weights | 0.907 | 0.345 |
| Dynamic per-patient gate | 0.904 | 0.338 |

The dynamic gate scored 0.003 below static, so **static weights were selected**. Learned weights: A 0.001, B 0.012, **C 0.703**, D 0.284.

**Test results (fold 10, n = 2,158, seed 42):**

| Model | Architecture | Train folds | Params | Fold-9 AUROC | Static w | Test macro AUROC [95% CI] | Macro F1 | Precision | Recall | Specificity | Exact-match acc | Label-wise acc | Δ AUROC vs best client |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Client A | 1D CNN | 1,2 | 0.44M | 0.894 | 0.001 | 0.888 [0.879, 0.896] | 0.677 | 0.638 | 0.724 | 0.860 | 0.498 | 0.846 | −0.005 |
| Client B | 1D ResNet | 3,4 | 6.73M | 0.884 | 0.012 | 0.884 [0.875, 0.893] | 0.674 | 0.648 | 0.709 | 0.874 | 0.539 | 0.849 | −0.009 |
| Client C | BiLSTM | 5,6 | 0.67M | **0.901** | 0.703 | 0.893 [0.885, 0.902] | 0.690 | 0.652 | 0.736 | 0.873 | 0.533 | 0.854 | (reference) |
| Client D | CNN-LSTM | 7,8 | 0.47M | 0.898 | 0.284 | 0.895 [0.886, 0.902] | 0.689 | 0.640 | 0.754 | 0.861 | 0.513 | 0.850 | +0.001 |
| Equal-weight mean | – | – | – | – | – | **0.905** [0.898, 0.913] | **0.709** | 0.696 | 0.731 | 0.891 | **0.578** | **0.868** | **+0.012** |
| **Static learned weights (selected)** | – | – | 4 | – | – | 0.902 [0.894, 0.910] | 0.700 | 0.672 | 0.735 | 0.877 | 0.550 | 0.862 | +0.009 |
| Dynamic gate | – | – | ~1k | – | – | 0.902 [0.894, 0.910] | 0.702 | 0.694 | 0.713 | 0.900 | 0.573 | 0.867 | +0.009 |

"Best client" is C, chosen on fold 9 (0.901). On fold 10, D is marginally higher (0.895), and the selected ensemble beats D by +0.007.

**Per class, selected ensemble vs best client C (test):**

| Class | AUROC C → ens | F1 C → ens | Recall C → ens | Specificity C → ens |
|---|---|---|---|---|
| NORM | 0.930 → 0.937 | 0.841 → 0.841 | 0.892 → 0.942 | 0.814 → 0.760 |
| MI | 0.900 → 0.911 | 0.697 → 0.716 | 0.775 → 0.753 | 0.846 → 0.881 |
| STTC | 0.927 → 0.931 | 0.739 → 0.753 | 0.810 → 0.821 | 0.878 → 0.885 |
| CD | 0.901 → 0.913 | 0.724 → 0.739 | 0.716 → 0.722 | 0.922 → 0.931 |
| HYP | 0.809 → 0.820 | 0.450 → 0.451 | 0.489 → 0.439 | 0.906 → 0.930 |

Confusion matrices for every class and model are in `outputs/fourclient/tables/per_class.csv` and `figures/confusion_matrices.png`.

**Findings**
- **Aggregation helps, and the gain is statistically clear.** Selected ensemble vs best client C, paired bootstrap on test (1,000 resamples): macro AUROC **+0.009 [+0.007, +0.011]**, macro F1 **+0.010 [+0.003, +0.017]**. All 5 classes improve in AUROC.
- **Learning the weights did not beat equal weights.** Equal weights scored best on test (0.905). It also scored best on fold-9 CV AUROC, but the selection rule only compared static against dynamic. The learned weights almost ignore A and B (w < 0.02) even though those clients are only about 0.01 AUROC behind. Equal weights keep the diversity benefit those clients add. The learned weights optimise BCE (calibration), which they do improve (CV BCE 0.345 vs 0.377), but not ranking.
- **The dynamic gate gave no improvement** (CV −0.003 vs static; test equal to static). Its per-patient weights barely vary from the static ones (mean A 0.005, B 0.033, C 0.662, D 0.300). With identical inputs across clients, the gate has no per-patient signal to exploit.
- **Architectures:** recurrent models came out ahead on a quarter of the data: BiLSTM 0.901 and CNN-LSTM 0.898 on fold 9, vs 0.894 for the 1D CNN and 0.884 for the 6.7M-parameter ResNet, which overfits early (best epoch 11). Every client is below the 0.906 of the old Client A trained on all 8 folds plus demographics, as expected with 4× less data each.
- **Exact-match accuracy** (all 5 decisions right) is 0.50–0.58. Label-wise accuracy is 0.85–0.87. HYP is the weak class everywhere (F1 ≈ 0.45).

**Limitations / deviations**
- One seed (42). CLAUDE.md asks for 3 seeds; run `bash scripts/run_fourclient.sh 43 44` to add them.
- Fold 9 serves for early stopping, temperature, thresholds and the weights, so fold-9 numbers are optimistic. Fold 10 is clean.
- Client C's first run crashed with an MPS out-of-memory error during validation (eval batch 512). It was restarted with eval batch 128, which does not change any outputs. A and B were not retrained.

---

## Four-client system: final results over 3 seeds (2026-10-18)

**What changed since the seed-42 section above**
- Seeds 43 and 44 were trained with the identical recipe.
- **Selection rule** (approved by the owner on 2026-10-18, *before* the 3-seed test run): candidates are tried from simplest to most complex (equal → static → dynamic). A more complex combiner is chosen only if its fold-9 CV macro AUROC beats the current choice by ≥ 0.002. Under the old rule (static vs dynamic only), static weights had been selected for seed 42.
- Fold 10 was evaluated a second time, for all 3 seeds, with `--rerun`, after the rule was fixed. Nothing was tuned on it. Seed 42's models and weights are unchanged, and its client numbers match the first evaluation.
- Bug fix: "Δ vs best client" now uses each seed's own best client, chosen on fold 9 (C for seeds 42/43, D for 44). Before, seed 42's best client was used for every seed.
- Operational: seed 44's Client C was interrupted when the MacBook Air (8 GB) froze under memory pressure and was restarted by the hardware watchdog. It was retrained from scratch with an MPS memory cap (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.25`, `LOW=1.0`). The cap only limits memory; it does not change the computation.

**Fold-9 validation (selection data)**

| Seed | A | B | C | D | CV: equal | CV: static | CV: dynamic | Selected | Static w (A/B/C/D) |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.894 | 0.883 | 0.901 | 0.898 | **0.908** | 0.907 | 0.904 | equal | 0.00 / 0.01 / 0.70 / 0.28 |
| 43 | 0.893 | 0.889 | 0.901 | 0.897 | **0.909** | 0.907 | 0.903 | equal | 0.00 / 0.05 / 0.50 / 0.45 |
| 44 | 0.892 | 0.886 | 0.899 | 0.900 | **0.909** | 0.908 | 0.905 | equal | 0.00 / 0.07 / 0.44 / 0.49 |

**Test results (fold 10, n = 2,158; mean ± SD over seeds 42, 43, 44; CI = mean of per-seed bootstrap 95% CIs)**

| Model | Architecture | Train folds | Test macro AUROC [95% CI] | Macro F1 | Precision | Recall | Specificity | Exact-match acc | Label-wise acc | Δ AUROC vs best client |
|---|---|---|---|---|---|---|---|---|---|---|
| Client A | 1D CNN | 1,2 | 0.888 ± 0.002 [0.879, 0.897] | 0.680 ± 0.004 | 0.646 | 0.726 | 0.867 | 0.509 ± 0.009 | 0.848 | −0.006 |
| Client B | 1D ResNet | 3,4 | 0.882 ± 0.003 [0.873, 0.891] | 0.672 ± 0.005 | 0.630 | 0.734 | 0.857 | 0.506 ± 0.029 | 0.840 | −0.013 |
| Client C | BiLSTM | 5,6 | 0.894 ± 0.000 [0.885, 0.902] | 0.692 ± 0.003 | 0.662 | 0.733 | 0.877 | 0.542 ± 0.013 | 0.855 | −0.001 |
| Client D | CNN-LSTM | 7,8 | 0.896 ± 0.001 [0.887, 0.904] | 0.690 ± 0.001 | 0.670 | 0.722 | 0.882 | 0.547 ± 0.036 | 0.857 | +0.001 |
| **Equal-weight mean (selected)** | aggregation | fold 9 | **0.905 ± 0.000** [0.897, 0.913] | **0.707 ± 0.007** | **0.698** | 0.725 | **0.894** | **0.581 ± 0.008** | **0.869** | **+0.011** |
| Static learned weights | aggregation | fold 9 | 0.903 ± 0.001 [0.895, 0.911] | 0.703 ± 0.003 | 0.685 | 0.728 | 0.887 | 0.570 ± 0.018 | 0.866 | +0.009 |
| Dynamic gate | aggregation | fold 9 | 0.901 ± 0.002 [0.892, 0.909] | 0.702 ± 0.002 | 0.681 | 0.729 | 0.888 | 0.568 ± 0.008 | 0.864 | +0.006 |

**Selected ensemble vs best client** (paired bootstrap on fold 10, 1,000 resamples):

| Seed | Best client (fold 9) | Δ macro AUROC [95% CI] | Δ macro F1 [95% CI] |
|---|---|---|---|
| 42 | C | +0.012 [+0.009, +0.016] | +0.018 [+0.009, +0.028] |
| 43 | C | +0.011 [+0.007, +0.014] | +0.010 [−0.001, +0.021] |
| 44 | D | +0.009 [+0.005, +0.013] | +0.024 [+0.014, +0.034] |

**Per-class test AUROC / F1** (mean of 3 seeds):

| Class | Client C | Client D | Equal-weight ensemble |
|---|---|---|---|
| NORM | 0.931 / 0.842 | 0.933 / 0.853 | **0.938 / 0.850** |
| MI | 0.901 / 0.702 | 0.906 / 0.715 | **0.912 / 0.725** |
| STTC | 0.926 / 0.741 | 0.923 / 0.727 | **0.933 / 0.768** |
| CD | 0.902 / 0.720 | 0.909 / 0.716 | **0.914 / 0.735** |
| HYP | 0.809 / 0.454 | 0.808 / 0.438 | **0.828 / 0.461** |

**Verdict**
1. **Aggregating the four clients reliably beats the best single client.** Macro AUROC improves by +0.009 to +0.012 in every seed, with every bootstrap CI above 0. Macro F1 improves by +0.010 to +0.024, and the CI excludes 0 in 2 of 3 seeds. Every class improves in AUROC, with the largest gain on HYP (+0.019).
2. **The best aggregator is the simplest.** Equal weights beat learned static weights (−0.002) and the dynamic per-patient gate (−0.004) on both fold-9 CV and the test set. Learned weights put almost no weight on Client A (≈ 0.00) and concentrate on C and D. This improves calibration (lower BCE) but loses the diversity benefit of A and B for ranking.
3. **The dynamic gate never earned its complexity.** All clients see the same ECG, so there is no per-patient signal for deciding whom to trust. A dynamic gate is more promising when clients differ in modality or data source.
4. **Seed stability is high.** Client SDs are ≤ 0.003 AUROC, and the ensemble SD is < 0.001. The ranking of clients (D ≈ C > A > B) is consistent across seeds.

**Limitations:** fold 9 is used for early stopping, temperature, thresholds and weights, so its numbers are optimistic; fold 10 is clean. Each client trains on a quarter of the data. The bootstrap CIs are per seed and resample records, not patients (fold 10 has 2,158 records from 1,877 patients).

**Files:** `outputs/fourclient/tables/{comparison.md, comparison.csv, per_class.csv (seed 42), significance.json}` and `outputs/fourclient/figures/{training_curves, model_comparison, confusion_matrices, aggregation_weights}.png` (per-model and per-class figures use seed 42).

---

## Reference experiments: all-data baseline and leave-one-client-out (2026-10-18)

Approved by the owner. Fold 10 was used again for evaluation only; nothing was tuned on it. Code: `python -m cardiomoe.fourclient.baseline --seeds 42 43 44`. Outputs: `outputs/fourclient/tables/{baseline_comparison.csv, baseline_comparison.json, client_subsets.csv}`.

### 1. All-data baseline: what does splitting the data cost?

One **CNN-LSTM** (the architecture of Client D, the best client on test) is trained on **all of folds 1–8** (17,084 records), with the identical recipe and seeds, early stopping, temperature and fold-9 thresholds. This is the model you would train if the data could be pooled.

| Model | Training data | Fold-9 AUROC | Test macro AUROC [95% CI] | Macro F1 | Precision | Recall | Specificity | Exact-match acc |
|---|---|---|---|---|---|---|---|---|
| Best single client (D, CNN-LSTM) | folds 7–8 (25%) | 0.898 | 0.896 ± 0.001 | 0.690 | 0.670 | 0.722 | 0.882 | 0.547 |
| **4-client equal-weight ensemble** | 4 × 2 folds, never pooled | 0.909 | **0.905 ± 0.000** | **0.708 ± 0.007** | 0.698 | 0.725 | 0.894 | **0.581 ± 0.009** |
| **All-data CNN-LSTM** | folds 1–8 pooled (100%) | **0.914** (0.916 / 0.914 / 0.912) | **0.909 ± 0.001** | 0.707 ± 0.009 | 0.685 | 0.740 | 0.887 | 0.566 ± 0.026 |

**Baseline − ensemble** (paired bootstrap on fold 10, 1,000 resamples):

| Seed | Δ macro AUROC [95% CI] | Δ macro F1 [95% CI] |
|---|---|---|
| 42 | +0.003 [−0.001, +0.008] | −0.012 [−0.024, +0.001] |
| 43 | +0.005 [+0.000, +0.009] | +0.012 [+0.001, +0.024] |
| 44 | +0.003 [−0.001, +0.006] | −0.001 [−0.012, +0.009] |

- **Pooling the data still wins on ranking, by a small margin.** The all-data model is +0.0035 test AUROC better on average, and the CI excludes 0 in 1 of 3 seeds. On fold 9 the gap is larger (+0.005).
- **On thresholded decisions it is a tie.** Macro F1 is the same on average (−0.0005, mixed sign). The ensemble has higher exact-match accuracy (0.581 vs 0.566) and specificity; the baseline has higher recall.
- **The ensemble closes about 70% of the gap.** On test AUROC, the best single client is 0.896, the ensemble 0.905 and the pooled model 0.909, so aggregation recovers 0.009 of the 0.013 lost by giving each client only a quarter of the data.
- **Interpretation:** when data cannot be pooled (separate sites), the 4-client ensemble nearly matches a pooled model. When data can be pooled, one model on everything is slightly better on AUROC and equivalent on F1.

### 2. Leave-one-client-out (equal-weight ensembles of client subsets)

Mean of 3 seeds; SDs ≤ 0.003 (see `client_subsets.csv`).

| Ensemble | Fold-9 AUROC | Test AUROC | Test Δ vs all four |
|---|---|---|---|
| **ABCD (all four)** | **0.909** | **0.905** | – |
| ACD (drop B) | 0.908 | 0.905 | −0.000 |
| BCD (drop A) | 0.908 | 0.904 | −0.001 |
| ABD (drop C) | 0.905 | 0.902 | −0.003 |
| ABC (drop D) | 0.906 | 0.901 | −0.004 |
| best pair (CD) | 0.907 | 0.903 | −0.002 |
| best single (D on test, C on fold 9) | 0.900 | 0.896 | −0.009 |

- All four together is best on both splits. Every client contributes, but A and B add only about 0.0005 each. That is why learned weights could set A to about 0 at little cost, yet equal weights still came out slightly ahead.
- C and D carry the most signal: dropping either costs 0.003–0.004.
- Every 2-client ensemble beats every single client on test. Most of the gain comes from combining models at all, not from which ones are combined.
