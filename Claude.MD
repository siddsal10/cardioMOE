# CardioMoE: Four-Client ECG Classification with Weighted Gated Aggregation

> **Instructions for Claude Code:** Read this file before writing code. After each piece of work,
> run its checks, append a short summary (what was built, the metrics, any deviations) to
> `RESULTS.md`, and stop for review. Do **not** add datasets beyond PTB-XL, do **not** change the
> test split, and **ask before changing any design decision** in this file. Every script must
> support `--debug` (about 200 samples, CPU, a few minutes).

The project was redesigned on 2026-10-15 at the owner's request. The earlier three-modality
design (ECG + text + image clients, Chapman/Mendeley adaptation) was replaced, and its code and
data were removed on 2026-10-17. Its results remain in `RESULTS.md` as history, and the code is
in the initial git commit.

---

## 1. Goal

Four independent clients each train a **different architecture** on their **own share** of the
PTB-XL training data. A central gate combines their predictions:

```
P_final(y|x) = Σ_i w_i P_i(y|x),   w_i ≥ 0,   Σ_i w_i = 1,   i ∈ {A, B, C, D}
```

The question is whether the weighted gate beats the best individual client. Don't assume it does.

## 2. Non-negotiable constraints

1. Only **PTB-XL v1.0.3** (open access, CC BY 4.0, no credentialing).
2. **Patient-level splits.** No patient appears in more than one of: any client's training data, validation, test.
3. **Fold 10 is the test set.** Use it only for the final evaluation; nothing is tuned on it.
   `evaluate.py` refuses a second full run without `--rerun`. Ask the owner before using it.
4. All randomness is seeded (`seed=42`). Final results are reported over seeds 42, 43, 44.
5. Clients never share data or parameters. Architectures differ, so parameters are never averaged.
   The gate sees only client probabilities.

## 3. Data

- `bash scripts/download.sh` downloads PTB-XL `records100/` (100 Hz) and writes `data/raw/MANIFEST.md`.
- `python -m cardiomoe.data.ptbxl --prepare` does the following:
  - **Labels:** multi-hot over `[NORM, MI, STTC, CD, HYP]`, from `scp_codes` and the
    `diagnostic_class` column of `scp_statements.csv` (rows with `diagnostic == 1`). Records
    with no diagnostic superclass are dropped (21,799 → 21,388).
  - **Signals:** shape `(12, 1000)`, lead order I, II, III, aVR, aVL, aVF, V1–V6, band-pass
    0.5–40 Hz (Butterworth order 3, zero-phase). Stored un-normalised in
    `data/processed/ptbxl_signals.npy`.
- **At load time:** per-lead z-score per recording, clipped to ±10, NaN → 0.
- **Train-only augmentation:** amplitude scaling 0.9–1.1, Gaussian noise (SNR 20–30 dB),
  baseline wander 0.1–0.5 Hz, lead dropout p = 0.05.

## 4. Splits

| Use | PTB-XL folds |
|---|---|
| Client A training | 1, 2 |
| Client B training | 3, 4 |
| Client C training | 5, 6 |
| Client D training | 7, 8 |
| Validation: early stopping, temperature, per-class thresholds, aggregation weights | 9 |
| Test: final evaluation only | 10 |

## 5. Clients (`src/cardiomoe/fourclient/models.py`)

All four clients have the same input (signal only) and the same training recipe, so only the
architecture and the data differ:

| Client | Architecture |
|---|---|
| A | 1D CNN: 4 × [Conv k7 → BN → ReLU → MaxPool], 32→256 ch, avg+max pool |
| B | 1D ResNet: stem k7/64, stages [64,128,256,256] × [3,4,6,3] blocks, avg+max pool |
| C | BiLSTM: 2 layers, hidden 128, over the 1,000 time steps, avg+max pool |
| D | CNN-LSTM: 3 conv blocks (1000 → 125 steps), then BiLSTM hidden 128, avg+max pool |

- **Head (all):** Linear 256 → ReLU → Dropout 0.3 → Linear 5.
- **Training:** BCEWithLogits with train-prevalence `pos_weight`, AdamW (lr 1e-3, wd 1e-2),
  OneCycle, batch 128, up to 50 epochs, early stopping on fold-9 macro AUROC (patience 8),
  gradient clipping 1.0.
- **Calibration:** after training, one temperature per client is fitted on fold 9.

## 6. Aggregation (`src/cardiomoe/fourclient/gate.py`, `aggregate.py`)

Weights are kept on the simplex with a softmax. Each combiner is fitted with BCE on the mixed
probabilities.

- **Equal weights:** `w_i = 1/4`, nothing learned.
- **Static learned weights:** one global `w_i` per client.
- **Dynamic gate:** an MLP (20 client probabilities + 4 entropies → 32 → 4) produces per-patient weights.

**Selection:**
- Patient-grouped 5-fold CV inside fold 9 gives out-of-fold predictions for each combiner.
- Candidates are tried from simplest to most complex (equal → static → dynamic). A more complex
  combiner is selected only if its CV macro AUROC beats the current choice by ≥ 0.002. This
  rule was approved by the owner on 2026-10-18, before any 3-seed test evaluation. Until then
  only static and dynamic were compared.
- Per-class decision thresholds are tuned to maximise F1 on fold 9: on the CV predictions for
  combiners, and on each client's own fold-9 output for clients.

## 7. Evaluation (`src/cardiomoe/fourclient/evaluate.py`)

Run on fold 10 for each client and each combiner:
- macro and per-class AUROC with a 95% bootstrap CI
- macro AUPRC
- precision, recall, specificity, F1 and accuracy per class, plus macro averages
- a per-class 2×2 confusion matrix
- exact-match accuracy (all 5 decisions right) and label-wise accuracy

Compare against the best individual client, chosen on fold 9, with a paired bootstrap for
Δ AUROC and Δ macro F1.

**Outputs:** `outputs/fourclient/tables/{comparison.md, comparison.csv, per_class.csv,
significance.json}` and `outputs/fourclient/figures/{training_curves, model_comparison,
confusion_matrices, aggregation_weights}.png`.

## 8. Repository layout

```
configs/base.yaml, configs/fourclient.yaml
scripts/download.sh, fetch_physionet.py, verify_data.py, run_fourclient.sh, launch_detached.py
src/cardiomoe/
  data/       ptbxl.py (prepare), labels.py, signal.py, datasets.py (SignalDataset)
  clients/    base.py (BaseClient, masked BCE)
  train/      common.py (training loop, early stopping, prediction)
  fusion/     calibration.py (temperature scaling)
  eval/       metrics.py (AUROC, AUPRC, F1, thresholds, ECE, bootstrap)
  fourclient/ models.py, train.py, gate.py, aggregate.py, metrics.py, evaluate.py,
              baseline.py (all-data baseline + leave-one-client-out, run after evaluate)
tests/        splits / patient leakage, labels, features, signal, four-client models + gate + metrics
data/, outputs/   git-ignored
```

## 9. How to run

```bash
pip install -r requirements.txt && pip install -e .
bash scripts/download.sh
python -m cardiomoe.data.ptbxl --prepare
python -m cardiomoe.fourclient.train --seeds 42 43 44
python -m cardiomoe.fourclient.aggregate --seeds 42 43 44
python -m cardiomoe.fourclient.evaluate --seeds 42 43 44     # fold 10: once
python -m cardiomoe.fourclient.train --clients ALL --seeds 42 43 44   # reference: one model on folds 1-8
python -m cardiomoe.fourclient.baseline --seeds 42 43 44     # baseline vs ensemble, client subsets
```

## 10. Known limitations

- Fold 9 is used for early stopping, temperature, thresholds and the weights, so fold-9
  numbers are optimistic. Fold 10 is clean.
- Each client sees only a quarter of the training data. One CNN-LSTM trained on all folds 1–8
  (`baselines.ALL`) reaches 0.909 test AUROC vs 0.905 for the ensemble; F1 is tied.
- All clients see the same input modality. A per-patient gate therefore has little signal for
  deciding whom to trust. Across seeds 42–44 the dynamic gate never beat equal or static
  weights, and equal weights were selected.
