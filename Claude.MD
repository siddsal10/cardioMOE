# CardioMoE — Multimodal Cardiac Diagnosis with Modality Clients and a Learned Gate

> **Instructions for Claude Code:** Read this entire file before writing any code. Implement it **phase by phase**, in order. At the end of each phase: run that phase's acceptance checks, append a short summary (what was built, the metrics, any deviations) to `RESULTS.md`, and stop for review before starting the next phase. Do **not** add datasets that are not listed here, do **not** change the test split, and **ask before changing any design decision** in this file. Every script must support a `--debug` flag that runs end-to-end on ~200 samples on CPU in a few minutes.

---

## 1. Goal

Build a cardiac diagnosis system made of **three independent "client" models**, each specialised in one modality, plus a **gating network** that decides, *per patient*, how much to trust each client before combining their predictions.

| Client | Modality | Input |
|---|---|---|
| **A** | EHR + ECG signal | Demographics/body measurements + raw 12-lead ECG waveform |
| **B** | Clinical text | Free-text cardiologist ECG report (translated to English) |
| **C** | ECG image | 12-lead ECG rendered as a printed paper-style image (and real ECG photos) |

**Key property:** when a **new dataset** arrives (possibly missing some modalities), we **freeze the clients and re-fit only the small gate** on a few labelled samples. The gate learns new trust weights for that population. This is the main experiment of the project.

**Prediction task (main):** multi-label classification of the 5 PTB-XL diagnostic superclasses:
`NORM` (normal), `MI` (myocardial infarction), `STTC` (ST/T change), `CD` (conduction disturbance), `HYP` (hypertrophy).

**Prediction task (adaptation):** binary normal vs. abnormal (`abnormal score = 1 − p(NORM)`), plus any superclasses that map cleanly in the new dataset.

---

## 2. Non-negotiable constraints

1. **Only fully public datasets** that can be downloaded with **no credentialing, no ethics/CITI training, and no data-use application**.
2. **Patient-level splits** only. No patient may appear in more than one of train / validation / test.
3. **The PTB-XL test fold (fold 10) is touched exactly once**, at the end, for final numbers.
4. All randomness is seeded (`seed=42` default; final results reported over 3 seeds: 42, 43, 44).
5. Every client must expose the same output interface (Section 6.1) so the gate never needs to know client internals. Clients never see each other's raw data. This keeps the design compatible with a federated setting later.

---

## 3. Datasets (all public, no credentialing)

| Name | Role | Access | Size | License |
|---|---|---|---|---|
| **PTB-XL v1.0.3** | Paired core: Clients A, B, C trained here; gate trained here | Open access, PhysioNet, no login | 21,799 12-lead ECGs, 18,869 patients, 10 s, 100/500 Hz, with `report` text + metadata | CC BY 4.0 |
| **Chapman-Shaoxing 12-lead ECG ("A large scale 12-lead electrocardiogram database for arrhythmia study") v1.0.0** | "New dataset" #1 for gate adaptation (ECG + age/sex, **no text**) | Open access, PhysioNet, no login | ~45k 12-lead ECGs, 10 s, 500 Hz, SNOMED-CT labels | CC BY 4.0 |
| **ECG Images dataset of Cardiac Patients (Mendeley gwbz3fsgp8 v2)** | "New dataset" #2 (**image-only**): tests Client C on real ECG printouts | Public, Mendeley Data, no login | ~900 ECG images, 4 classes | CC BY 4.0 |
| *(optional)* **PTB-XL+ v1.0.1** | Extra engineered ECG features for Client A ablation | Open access, PhysioNet | Features for all PTB-XL records | CC BY 4.0 |
| *(optional)* **MTSamples (Kaggle: tboyle10/medicaltranscriptions)** | Domain-adaptive pretraining of Client B's text encoder on cardiology notes | Free Kaggle account + API key (no training) | ~5k transcribed notes | See Kaggle page |

### 3.1 Download (Phase 0)

```bash
# PTB-XL (open access)
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/ptbxl https://physionet.org/files/ptb-xl/1.0.3/

# Chapman-Shaoxing (open access)
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/chapman https://physionet.org/files/ecg-arrhythmia/1.0.0/

# PTB-XL+ (optional)
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/ptbxl_plus https://physionet.org/files/ptb-xl-plus/1.0.1/

# Mendeley ECG images: download manually ("Download All" zip) from
#   https://data.mendeley.com/datasets/gwbz3fsgp8/2
# and unzip into data/raw/ecg_images_pk/

# MTSamples (optional): kaggle datasets download -d tboyle10/medicaltranscriptions -p data/raw/mtsamples --unzip
```

Only `records100/` (100 Hz) is needed from PTB-XL for training. If bandwidth is limited, skip `records500/`.

**Phase 0 must verify:** file counts match the numbers above (±1%), `ptbxl_database.csv` and `scp_statements.csv` load, a random WFDB record reads with `wfdb.rdsamp` and has shape `(1000, 12)` at 100 Hz, and the Chapman headers parse. Write `data/raw/MANIFEST.md` with the download date, dataset versions, and the counts.

---

## 4. Repository layout

```
cardio-moe/
├── PROJECT_PLAN.md          # this file
├── RESULTS.md               # appended after every phase
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml            # paths, seed, device, debug
│   ├── client_a.yaml
│   ├── client_b.yaml
│   ├── client_c.yaml
│   ├── gate.yaml
│   └── adapt.yaml
├── scripts/
│   └── download.sh
├── src/cardiomoe/
│   ├── data/
│   │   ├── ptbxl.py         # loading, labels, metadata cleaning, splits
│   │   ├── chapman.py       # loading, SNOMED→superclass mapping
│   │   ├── pk_images.py     # Mendeley image dataset loader
│   │   ├── labels.py        # superclass definitions + mapping tables
│   │   ├── signal.py        # filtering, resampling, normalisation, noise augmentation
│   │   ├── render_ecg.py    # signal → paper-style 12-lead image
│   │   └── text_prep.py     # language detection, translation, diagnostic-term masking
│   ├── clients/
│   │   ├── base.py          # ClientOutput dataclass + BaseClient interface
│   │   ├── client_a.py      # EHR + ECG signal
│   │   ├── client_b.py      # text
│   │   └── client_c.py      # image
│   ├── fusion/
│   │   ├── gate.py          # gating network (ours)
│   │   ├── baselines.py     # mean, static weights, stacking LR, early-fusion MLP
│   │   └── calibration.py   # per-client temperature scaling
│   ├── train/
│   │   ├── train_client.py
│   │   ├── crossfit.py      # out-of-fold client outputs for gate training
│   │   ├── cache_outputs.py
│   │   ├── train_gate.py
│   │   └── adapt.py         # new-dataset gate re-fitting experiments
│   └── eval/
│       ├── metrics.py       # AUROC, AUPRC, F1, Brier, ECE, bootstrap CIs
│       ├── robustness.py    # missing-modality + noise tests
│       └── plots.py
├── tests/                   # pytest: shapes, masking, label mapping, no split leakage
├── data/{raw,processed}/    # git-ignored
└── outputs/{checkpoints,cache,figures,tables}/  # git-ignored
```

**Environment:** Python 3.11, PyTorch ≥ 2.2, `wfdb`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `transformers`, `sentencepiece`, `timm`, `matplotlib`, `langdetect`, `pyyaml`, `tqdm`, `pytest`. A single 8–16 GB GPU (e.g. Colab T4) is enough. All scripts must fall back to CPU.

---

## 5. Data preparation (Phase 1)

### 5.1 PTB-XL labels
- Parse `scp_codes` (a dict string) for every record.
- Map each code to its `diagnostic_class` using `scp_statements.csv` rows where `diagnostic == 1`.
- Label vector = multi-hot over `[NORM, MI, STTC, CD, HYP]`.
- Drop records with no diagnostic superclass. Log how many were dropped.

### 5.2 PTB-XL splits (official, patient-separated)
- Folds 1–8 → train
- Fold 9 → validation (early stopping, temperature scaling, gate early stopping)
- Fold 10 → test (final evaluation only)

Add a pytest that asserts there is no `patient_id` overlap between the three sets.

### 5.3 PTB-XL metadata → Client A tabular features
Use **only** these columns: `age`, `sex`, `height`, `weight`.
- `age`: values ≥ 90 (PTB-XL stores 300 for ages above 89) → set to 90. Then standardise using train statistics.
- `sex`: 0/1.
- `height`, `weight`: very often missing. Impute with the train median and add binary `height_missing` and `weight_missing` flags. Then standardise.
- Optionally add BMI when both are present.

**Never use these columns as features (they leak labels or are annotations):** `scp_codes`, `report`, `infarction_stadium1`, `infarction_stadium2`, `heart_axis`, `extra_beats`, `pacemaker`, `validated_by`, `second_opinion`, `initial_autogenerated_report`, `validated_by_human`, and the noise/drift columns. Add a test that asserts the feature list equals the allow-list.

### 5.4 ECG signal preprocessing (shared by Clients A and C and by Chapman)
- Use 100 Hz, 10 s → shape `(12, 1000)`. Resample Chapman from 500 Hz to 100 Hz with `scipy.signal.resample_poly`.
- Standard lead order: I, II, III, aVR, aVL, aVF, V1–V6. Verify the order in each dataset's header and reorder if needed.
- Band-pass filter 0.5–40 Hz (Butterworth, order 3, zero-phase).
- Per-lead z-score normalisation per recording. Clip to ±10.
- Replace NaNs with 0 and log how many occurred.
- **Augmentations (train only):** random amplitude scaling (0.9–1.1), Gaussian noise (SNR 20–30 dB), baseline wander (sinusoid 0.1–0.5 Hz), random lead dropout (each lead with p = 0.05).

### 5.5 Text (Client B input)
1. Take the PTB-XL `report` column.
2. Detect the language with `langdetect` (German, English, or Swedish).
3. Translate German → English with `Helsinki-NLP/opus-mt-de-en` and Swedish → English with `Helsinki-NLP/opus-mt-sv-en`, offline. Cache the results to `data/processed/ptbxl_reports_en.parquet`. Run this once.
4. Lowercase and normalise whitespace.
5. **Leakage control.** PTB-XL's diagnostic labels were *derived from these reports*, so raw text nearly states the answer. Produce **two text variants**:
   - `report_raw`: the translated report. Treated as an *oracle / upper bound*.
   - `report_masked`: replace diagnostic terms with `[MASK]`. Build the term list from (a) every `description` in `scp_statements.csv`, (b) every SCP code string itself (e.g. `IMI`, `LVH`, `LBBB`), (c) English and German stems: infarct*, ischem*/ischäm*, hypertroph*, block*, bundle, myocard*, injury, "st elevation", "st depression", "t wave", "t-wave", "q wave", "normal ecg", "abnormal", "pathologic*", "lvh", "rvh", "mi". Keep descriptive content such as rhythm, rate, axis, intervals, and recording quality.
6. **Leakage audit (required).** Train Client B on both variants. Report validation macro AUROC for each in `RESULTS.md`. If `report_masked` still reaches macro AUROC > 0.95, extend the mask list and repeat, recording each iteration. **The masked variant is the default for all fusion experiments.** The raw variant appears only as a clearly labelled upper-bound row.

### 5.6 ECG images (Client C input)
- Render every PTB-XL record (from the **filtered, un-normalised** signal in mV) into a paper-style image:
  - Standard 3×4 layout plus a lead II rhythm strip.
  - 25 mm/s, 10 mm/mV, pink/red millimetre grid, lead labels.
- Save as PNG, 256 × 512 (H × W), to `data/processed/ptbxl_images/{ecg_id}.png`.
- **Train-time image augmentations** that mimic photographed printouts: small rotation (±3°), perspective warp, brightness/contrast jitter, JPEG compression, Gaussian blur, random crop/resize (0.9–1.0).
- Mendeley images: load the originals, convert to RGB, resize to 256 × 512 (letterbox to preserve aspect ratio).

### 5.7 Chapman-Shaoxing labels
- Parse the `#Dx:` SNOMED-CT codes from each `.hea` header. Use the dataset's included condition-name CSV to get readable names.
- **Binary label:** `normal = 1` if the only codes are sinus rhythm (426783006) or normal-ECG codes; otherwise `abnormal`.
- **Superclass mapping** (only where it is clinically unambiguous; leave other classes as *unknown*, which is masked out of the loss):
  - `MI`: myocardial infarction codes
  - `CD`: LBBB, RBBB, incomplete RBBB, 1st/2nd/3rd-degree AV block, IVCD, LAFB/LPFB
  - `HYP`: LVH, RVH
  - `STTC`: ST depression, ST elevation, T-wave abnormal/inversion, ST-T change
  - `NORM`: as in the binary label
- Write the full mapping to `data/processed/chapman_label_map.csv` and include it in `RESULTS.md`.
- Chapman has `age` and `sex` only. Set the height/weight missing flags to 1.
- **Splits:** a patient-level random split with seed 42 into `adapt_pool` (30%) and `adapt_test` (70%).

### 5.8 Mendeley ECG images labels
- Map the 4 folders: Normal → `NORM`; Myocardial Infarction → `MI`; History of MI → `MI`; Abnormal heartbeat → abnormal (binary only, superclass unknown).
- Use a stratified split with seed 42: 30% adapt / 70% test. Patient IDs are not available, so note this limitation in `RESULTS.md`.

**Phase 1 acceptance:** label prevalences printed per split; no patient overlap; translation cache exists for 100% of reports; leakage-term list saved; 20 rendered images visually spot-checked and saved to `outputs/figures/render_samples.png`; all tests pass.

---

## 6. Models

### 6.1 Common client interface (`clients/base.py`)
```python
@dataclass
class ClientOutput:
    embedding: Tensor    # [B, d_i]  penultimate representation
    logits: Tensor       # [B, 5]    one logit per superclass
    uncertainty: Tensor  # [B]       mean binary entropy of sigmoid(logits / T_i)

class BaseClient(nn.Module):
    modality: str        # "ehr_ecg" | "text" | "image"
    embed_dim: int
    def forward(self, batch: dict) -> ClientOutput: ...
```
Each client has its own temperature `T_i`, fitted on fold 9 after training (Section 7.3).

### 6.2 Client A: EHR + ECG signal
- **ECG encoder:** 1D ResNet (xresnet1d50-style). Input `(12, 1000)`; stem conv k = 7 with 64 channels; 4 residual stages with [64, 128, 256, 256] channels; global average + max pooling concatenated → 512 → linear → **256**.
- **Tabular encoder:** MLP with 7 inputs (age, sex, height, weight, BMI, 2 missing flags) → 64 → 64, ReLU, dropout 0.2.
- **Local fusion:** concatenate (256 + 64) → linear 256 → ReLU → dropout 0.3 → this is the **embedding (d_A = 256)** → linear → 5 logits.
- **Training:** AdamW, lr 1e-3, weight decay 1e-2, OneCycle schedule, 50 epochs, batch 128, BCEWithLogits with `pos_weight` from train prevalence, early stopping on fold-9 macro AUROC (patience 8).
- **Tabular dropout:** during training, set height/weight to missing with p = 0.5 so the model works on Chapman, which lacks them.
- **Acceptance:** fold-9 macro AUROC ≥ 0.88. The literature reports ~0.92–0.93 for strong 1D CNNs on this task.

### 6.3 Client B: clinical text
- **Encoder:** `emilyalsentzer/Bio_ClinicalBERT`. Max length 128. Use mean pooling over tokens, excluding padding, → 768 → linear → **embedding (d_B = 256)** → linear → 5 logits.
- **Fine-tuning:** freeze embeddings and the first 8 layers; fine-tune the last 4 layers and the head. AdamW, lr 2e-5 for the encoder and 1e-3 for the head, 10 epochs, batch 32, linear warmup 10%, early stopping on fold-9 macro AUROC.
- Train two checkpoints: `client_b_masked` (**default**) and `client_b_raw` (upper bound only).
- *(Optional, Phase 9)* domain-adaptive MLM pretraining on MTSamples "Cardiovascular / Pulmonary" notes for 2 epochs before fine-tuning.

### 6.4 Client C: ECG image
- **Backbone:** `timm` `efficientnet_b0` (ImageNet-pretrained), input 256 × 512 → global pool 1280 → linear → **embedding (d_C = 256)** → linear → 5 logits.
- **Training:** AdamW, lr 3e-4, cosine schedule, 25 epochs, batch 64, with the augmentations from 5.6, early stopping on fold-9 macro AUROC.
- **Acceptance:** fold-9 macro AUROC ≥ 0.80.

### 6.5 The gate (ours): `fusion/gate.py`
Inputs for each sample, after the clients are frozen:
- `h_i`: client embedding (256-d), passed through a per-client projector `P_i: 256 → 128` (LayerNorm + Linear)
- `ℓ_i`: temperature-scaled client logits (5)
- `u_i`: client uncertainty (scalar)
- `m_i ∈ {0, 1}`: availability mask (1 if modality i is present)

**Computation:**
```
x      = concat( m_A·P_A(h_A), m_B·P_B(h_B), m_C·P_C(h_C), m_A·u_A, m_B·u_B, m_C·u_C, m_A, m_B, m_C )   # 3·128 + 3 + 3 = 390
s      = MLP(x): 390 → 128 → ReLU → Dropout(0.2) → 15  → reshape to [3 clients, 5 classes]
α_i,c  = m_i · exp(s_i,c) / Σ_j m_j · exp(s_j,c)          # masked softmax over clients, separately per class
ŷ_c    = Σ_i α_i,c · ℓ_i,c + b_c                           # fused logit for class c
```
- **Per-class weights** let the gate learn, for example, "trust ECG image for MI, trust text for CD". A config flag `per_class: false` collapses this to one weight per client.
- The output is a **weighted average of calibrated client logits**, so α is directly interpretable as "how much each client contributed".
- **Missing modality:** α is exactly 0 when `m_i = 0`. At least one modality must be present.

**Gate training:**
- **Training data:** out-of-fold client outputs on folds 1–8 (Section 7.2). Early stopping on fold 9.
- **Loss:** BCEWithLogits(ŷ, y) + λ_bal · L_balance, where `L_balance = Σ_i (mean_batch(α_i) − 1/n_available)²` and λ_bal = 0.01. This prevents collapse onto a single client.
- **Modality dropout:** for each training sample, drop each client independently with p = 0.3 (resample if all three are dropped).
- **Corruption augmentation:** for 20% of training samples, swap Client A's cached outputs for the cached outputs from a noisy version of the signal (Section 7.2). The gate must learn to down-weight A when its signal is bad, which it can detect from A's uncertainty.
- AdamW, lr 1e-3, 100 epochs maximum, batch 256, patience 10.

### 6.6 Baselines (`fusion/baselines.py`), all on the same frozen client outputs
1. Each client alone (A, B-masked, C)
2. **Mean fusion:** equal weights over available clients
3. **Static learned weights:** one learnable α per client per class, independent of the input. *This is the key ablation: it shows whether per-patient gating helps.*
4. **Stacking:** logistic regression on the concatenated calibrated logits
5. **Early fusion:** MLP on concatenated embeddings [256·3 → 256 → 5] with modality dropout. This mirrors the concatenation approach of Zhang & Wang, Front. Cardiovasc. Med. 2026.
6. **Upper bound row:** gate with `client_b_raw` instead of `client_b_masked`, clearly labelled as leaky.

### 6.7 Gate ablations
- No uncertainty inputs
- No modality dropout
- No balance loss
- Per-class vs. per-sample (single weight)
- No corruption augmentation

---

## 7. Training pipeline

### 7.1 Phase 2–4: train the clients
Train A, B (masked and raw), and C on folds 1–8 with early stopping on fold 9. Save checkpoints to `outputs/checkpoints/{client}/final.pt`.

### 7.2 Phase 5: cross-fitting and caching
The gate must be trained on client outputs for data the clients **did not train on**; otherwise the clients look over-confident and the gate learns the wrong trust levels.
- **Cross-fitting:** split folds 1–8 into 4 groups ({1,2}, {3,4}, {5,6}, {7,8}). For each group, train each client on the other 6 folds (same hyperparameters, early stopping on fold 9) and predict the held-out group. This gives **out-of-fold (OOF)** outputs for all of folds 1–8.
- The final clients from 7.1 produce outputs for fold 9, fold 10, Chapman, and the Mendeley images.
- **Noisy variants for Client A:** also cache A's outputs on signals corrupted at SNR 10 dB and 5 dB (plus baseline wander) for OOF and for fold 9.
- Cache everything to `outputs/cache/{dataset}_{split}_{client}.pt` as dicts of tensors keyed by `ecg_id`.
- **Quick mode** (`--gate_data fold9`): skip cross-fitting and train the gate on fold 9 with 5-fold CV. Use this for development only; final results use OOF.

### 7.3 Calibration
Fit a per-client temperature `T_i` on fold 9 (minimise BCE). Report ECE before and after.

### 7.4 Phase 6: train the gate and baselines
Run the gate, all baselines, and all ablations for 3 seeds. Evaluate on fold 9 during development and on **fold 10 once** at the end.

---

## 8. Experiments and evaluation

### 8.1 Metrics (`eval/metrics.py`)
- Macro and per-class AUROC (primary metric: macro AUROC)
- Macro AUPRC
- Macro F1 at thresholds tuned per class on fold 9
- Brier score and ECE (15 bins)
- 95% CIs by bootstrap (1,000 resamples of the test set)
- Mean ± SD over 3 seeds

### 8.2 Experiment 1: main results on PTB-XL fold 10
One table: all clients alone, all baselines, the gate, ablations, and the leaky upper bound.

### 8.3 Experiment 2: missing-modality robustness
Evaluate the gate and the baselines on fold 10 under all 7 non-empty availability combinations: {A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C}.

### 8.4 Experiment 3: noise robustness
Corrupt fold-10 signals at SNR {∞, 20, 10, 5} dB and re-run Client A (Client C keeps clean images). Plot macro AUROC against SNR for the gate, mean fusion, static weights, and Client A alone. Also plot mean α_A against SNR; the gate should shift weight away from A as noise rises.

### 8.5 Experiment 4: adaptation to a new dataset (the core claim)
**Chapman-Shaoxing** (modalities A and C available; B missing; Client C uses images rendered from Chapman signals):
- Adaptation budgets: N ∈ {0, 50, 200, 1000, all of adapt_pool} labelled samples, 5 random draws per N.
- Strategies:
  1. **Zero-shot:** the PTB-XL gate applied as-is
  2. **Gate-only re-fit (ours):** clients frozen; re-fit the gate (and the bias `b`), initialised from the PTB-XL gate
  3. **Full fine-tune:** fine-tune Clients A and C plus the gate on the same N samples
  4. **Client A only**, fine-tuned on N
- The loss uses only the classes mapped for Chapman (masked BCE).
- **Report:** binary AUROC (normal vs. abnormal) and per-class AUROC for the mapped superclasses on `adapt_test`, against N. Also plot the **gate weights before vs. after adaptation**, which visualises "the model deciding how much to trust each client" on a new population. Record wall-clock time and trainable parameter count for strategies 2 and 3.

**Mendeley ECG images** (only modality C available):
- Zero-shot Client C vs. fine-tuning Client C on the adapt split.
- Report binary AUROC and MI AUROC. This measures the synthetic-render → real-photo domain gap and confirms that the gate routes 100% of the weight to C when only images exist.

### 8.6 Interpretability
- Distribution of α per class per client on fold 10 (violin plots)
- α broken down by ground-truth class and by Client A uncertainty quartile
- Five case studies: patients where the gate's weighting changed the decision compared with mean fusion. Show the ECG, the rendered image, the masked report, and the α bar chart for each.
- Grad-CAM for Client C and integrated-gradients saliency for Client A on those cases.

### 8.6b Selective prediction (added 2026-10-08, approved by the project owner)
Target: **≥ 95% accuracy on the cases the system answers**; the remaining cases are referred to a cardiologist.
- **Confidence score** per patient from the fused, calibrated output: `conf = mean_c max(p_c, 1 − p_c)` over the 5 classes (per-class decisions at the fold-9-tuned thresholds). Compare against the gate's own uncertainty signal (α-weighted client uncertainty) and keep whichever gives better risk–coverage on fold 9.
- **Accuracy definition:** a patient counts as correct only if all 5 per-class decisions are right (exact match). Also report label-wise accuracy and the binary normal/abnormal accuracy.
- **Threshold:** pick the confidence threshold on **fold 9** as the lowest threshold whose fold-9 accuracy is ≥ 95%. Apply it unchanged to fold 10, Chapman and Mendeley. Never tune it on test data.
- **Report:** accuracy on the answered cases, **coverage** (% answered), per-class AUROC on all cases, and the full risk–coverage curve. Compare gate vs mean fusion vs Client A alone at equal coverage. 95% accuracy at very low coverage is not a success, so always state the coverage.
- `eval/selective.py`; figure `outputs/figures/risk_coverage.png`.

### 8.7 Outputs
- `outputs/tables/*.csv` and Markdown versions in `RESULTS.md`
- `outputs/figures/*.png`: architecture diagram, ROC curves, robustness curves, adaptation curves, α plots, case studies

---

## 9. Phases and acceptance checklist

| Phase | Deliverable | Acceptance |
|---|---|---|
| 0 | Env, `download.sh`, `MANIFEST.md` | Counts verified, a sample record reads |
| 1 | Data prep (5.1–5.8), tests | No split leakage, translation cache complete, renders spot-checked |
| 2 | Client A | Fold-9 macro AUROC ≥ 0.88 |
| 3 | Client B (masked + raw), leakage audit | Both AUROCs reported; masked variant documented |
| 4 | Client C | Fold-9 macro AUROC ≥ 0.80 |
| 5 | Cross-fitting, caching, calibration | OOF outputs for 100% of folds 1–8; ECE reported |
| 6 | Gate + baselines + ablations (fold 9) | Gate ≥ mean fusion on fold 9; no NaNs; α sums to 1 over available clients (unit test) |
| 7 | Robustness experiments (8.3, 8.4) | Figures saved |
| 8 | Adaptation experiments (8.5) | Adaptation curves + α before/after figures saved |
| 9 | Final fold-10 evaluation, interpretability, README | All tables and figures in `RESULTS.md`; optional MTSamples DAPT |
| 10 *(optional)* | Federated simulation with Flower: each client trains in its own process and sends only `ClientOutput` to a server that trains the gate | Same results as the centralised gate within noise |

---

## 10. Known limitations to state honestly in the write-up
- PTB-XL "notes" are ECG reports, not full physician notes, and the labels are derived from them. Masking reduces but may not remove leakage, which is why the raw-text result is reported only as an upper bound.
- Client C images are rendered from the same signals Client A uses, so A and C share information. The Mendeley experiment tests real-world images separately.
- PTB-XL "EHR" is limited to demographics and body measurements.
- The Mendeley image dataset has no patient IDs, so its split cannot be guaranteed patient-level.
- Chapman labels were mapped to PTB-XL superclasses by us. The mapping is published in the repo.

---

## 11. How to run (target end state)
```bash
pip install -r requirements.txt
bash scripts/download.sh
python -m cardiomoe.data.ptbxl --prepare             # + chapman, pk_images, text_prep, render_ecg
python -m cardiomoe.train.train_client --client a
python -m cardiomoe.train.train_client --client b --text masked
python -m cardiomoe.train.train_client --client b --text raw
python -m cardiomoe.train.train_client --client c
python -m cardiomoe.train.crossfit --clients a b c
python -m cardiomoe.train.cache_outputs
python -m cardiomoe.train.train_gate --seeds 42 43 44
python -m cardiomoe.eval.robustness
python -m cardiomoe.train.adapt --dataset chapman
python -m cardiomoe.train.adapt --dataset pk_images
python -m cardiomoe.eval.plots
```