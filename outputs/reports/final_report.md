# Multimodal ASD classification with MMASD, Move4AS and Q-CHAT-10 — research report

**Status: academic research prototype. Not a diagnostic system. No clinical claims.**
Two runs, identical data, folds and seed (42, 5 participant-level folds):

| Run | Config | Training | Outputs | Tables |
|---|---|---|---|---|
| `SMOKE_TEST` (2026-10-25) | `configs/default.yaml` | 3 epochs, batch 8, lr 1e-4, no early stopping | `outputs/` | [`results_tables.md`](results_tables.md) |
| **`FULL_TRAINING`** (2026-10-26) | `configs/full_training.yaml` | ≤ 100 epochs, batch 32, lr 5e-4, early stopping (patience 10) on inner-validation participants | `outputs/full/` | [`../full/reports/results_tables.md`](../full/reports/results_tables.md) |

**Section 5 (full training) holds the primary results.** Sections 4.1–4.6 document the smoke test. All numbers
come from saved metric files and are reproduced in the auto-generated tables (`scripts/make_report.py`).

---

## 1. Executive summary

1. **Only Move4AS supports an ASD-vs-non-ASD experiment.** MMASD contains only autistic children (no controls), and
   the Kaggle questionnaire's label is a deterministic screening rule (`sum(A1..A10) > 3`, 100 % agreement), not a
   diagnosis. No two datasets share participants, so no cross-dataset or questionnaire fusion was possible.
2. **Move4AS motion separates the groups almost perfectly** (participant-level pooled ROC-AUC 0.985 for
   kinematic-feature logistic regression, 0.992 for CTR-GCN; 33 participants, 14 ASD).
   **EEG is weak** (band-power LR AUC 0.60 [0.39, 0.79]; EEG CNN at chance, balanced accuracy 0.49).
3. **Fusion adds nothing measurable over motion alone.** Early 0.966, intermediate 0.985, late 0.985 (feature) /
   0.992 (deep) pooled AUC. Late fusion learned motion weights of 0.8–1.0 in every fold, i.e. it discards EEG.
   With 33 participants the CIs of all motion-containing models overlap completely.
4. **The motion result must be treated as possibly confounded by acquisition, not as evidence of an ASD motor
   signature.** The audit found that the authors' visual QC rejected **34 % of control vs 6 % of ASD** motion trials,
   4 controls were recorded at 40 fps vs 60 fps, and task mix differs by group. The signal survived every
   sensitivity analysis we could run (per task, excluding 40-fps participants, 6 Hz low-pass, no acceleration/jerk
   features: AUC 0.95–1.00), and age + sex alone do not explain it (AUC 0.53). But per-session recording conditions
   cannot be separated from group membership with the released data.
5. **MMASD (movement analysis):** 10-class activity recognition, participant-grouped. PoseFormerV2 adapter on 3D
   ROMP reached macro-F1 0.66 (chance 0.10) vs 0.53 on 2D OpenPose. CTR-GCN is clearly under-trained after 3 epochs
   (macro-F1 ≈ 0.21–0.26). ADOS-2 regression from kinematics (n = 20) is no better than predicting the mean.
6. **Questionnaire:** with Q-CHAT items the models reproduce the scoring rule (LogReg AUC 1.000). This is target
   leakage by construction, not a finding. The leakage-controlled model (demographics/history only) reaches AUC
   ≈ 0.60.
7. **Full training (Section 5) leaves the conclusions unchanged.**
   - **Move4AS:** CTR-GCN reaches participant AUC 0.992 and trial AUC 0.972 (smoke test: 0.932). Intermediate
     fusion reaches participant balanced accuracy 0.974 (14/14 ASD, 18/19 controls), but its AUC equals
     motion-only CTR-GCN. Deep late fusion put weight 1.0 on motion in every fold, and the EEG CNN stays at
     chance (AUC 0.571).
   - **MMASD:** the smoke test's CTR-GCN weakness was under-training (macro-F1 0.21 → 0.69). The best model is
     the MAE-pretrained encoder on 3D ROMP (macro-F1 0.775).
   - **Questionnaire:** the leakage-free MLP stays at AUC ≈ 0.59.

## 2. Data, labels and audit (Steps 1–4)

Audit files: `outputs/audit/{move4as,mmasd,questionnaire}_audit.json`, `dataset_inventory.csv`,
`dataset_comparison.md`, extraction manifests.

**Retrieval/integrity.** All six archives in `E:\ASD Project\Dataset` opened and passed CRC tests. No download was
needed, and no fetching error was reproducible: the earlier failures are no longer present, and every archive is
complete. All 11 Move4AS files match the official figshare record in both size and **MD5** (CC BY 4.0;
`move4as_download_manifest.json`, produced by `scripts/download_move4as.py --verify-only`). The nested
`Dataset.zip` (5.3 GB stored, 10.3 GB of `.mat` files) was also CRC-tested. The script downloads only missing or
corrupt files, resumes partial downloads and prints manual recovery steps. Raw archives were never
modified; extraction went to new folders `Dataset/{Move4AS,MMASD,Questionnaire}`. `optical_flow-001.zip`
(6.5 GB) was audited but not extracted, because optical flow is outside this study.

| | Move4AS | MMASD | Q-CHAT-10 toddlers |
|---|---|---|---|
| Participants | 34 adults: 14 ASD (P1–P14), 20 controls (S1–S20) | 32 children, **all ASD** | 1,054 screens (row IDs) |
| Label | clinical group (folder prefix) | none | screening rule (Q-CHAT > 3) |
| Modalities used | 3D mocap (21 rigid bodies, 40/60 fps), EEG (15/16 ch, 250 Hz) | OpenPose BODY_25 2D, ROMP SMPL-24 3D, ADOS-2 | 10 items + demographics |
| Observed vs README | matches (20 NT + 14 ASD) | 1,693 clips vs "1,315" in README; OpenPose lacks frog pose | matches (1,054) |

**Key validated facts and decisions**
* *Move4AS EEG:* `eegDataT` = EEG rows + stimulus row + cue row. 203 blocks have 15 channels and 40 have 16. We used
  the 14 channels valid for everyone (dropping PO3, which is noise for some participants per the authors' script,
  and Oz). Execution windows are 4 s (stim code 4). 6,097/7,253 trials pass a ±500 µV peak-to-peak check after a
  1–45 Hz band-pass and 50 Hz notch.
* *Move4AS motion:* 192 `mdata` blocks. The authors' blacklist (22 blocks) and their per-trial "Selected Trials"
  QC were applied, plus our own checks (≥ 4 s, gaps ≤ 0.25 s, ≤ 20 % missing). That leaves 3,388 trials from 33
  participants; S2 has no motion data, and P3, P4 and P13 have no walk motion. Timestamps are irregular
  (≈ 50–60 Hz), so trials were resampled on a time grid to 128 frames.
* *Motion ↔ EEG alignment:* both modalities carry the same stimulus and cue registers. The 30-trial pseudo-random
  cue sequence was used as a fingerprint to find each block's trial offset (0 in 109 blocks, −1 in 38). 147 of 155
  blocks aligned with ≥ 95 % agreement. Eight short blocks (< 20 overlapping trials) were excluded from fusion.
  Every one of the **2,709 paired trials** (33 participants) has identical cues in both modalities.
* *MMASD:* the ADOS-2 classification column contains only 1/2, i.e. ASD/autism, so there is **no control
  group**. ADOS scores exist for 20 of the 32 participants with clips. 2D and 3D come from the same recordings:
  identical clip names and frame counts, ROMP-2D equals the x,y of ROMP-3D, and the 24-joint file equals the first
  24 of the 71-joint file. Frames contain 1–7 people and **the child is not labelled**. We tracked people and
  used the most persistent track, matching OpenPose to the same person (median centroid distance 21 px), so in
  some clips the selected person may be a therapist.
* *Questionnaire:* the Toddler file is used. `Autism_Screening_Data_Combined.csv` was **excluded**: every toddler
  row in it is labelled NO, even with Q-CHAT = 10, and it contains 847 duplicate rows. 79 Toddler rows are
  identical apart from Case_No, so they were kept together in CV groups.

**Confounds found in Move4AS** (all computed, see `move4as_audit.json`):
- authors' visual QC rejects 34 % of control vs 6 % of ASD motion trials;
- S6, S8, S9 and S10 (controls) were recorded at 40 fps, everyone else at 60 fps;
- dance share of usable motion trials is 76 % for controls vs 59 % for ASD;
- sex ratio differs (7/20 vs 2/14 female); mean age 25.9 vs 27.0 years;
- AQ exists only for controls and ADOS only for ASD, so neither may be used as a feature.

## 3. Methods (Steps 5–10)

See `docs/architecture.md` and `docs/workflow.md`. In summary:
* **CV:** StratifiedGroupKFold by participant, 5 folds, seed 42. The same fold assignment is used by every
  Move4AS model (`outputs/metrics/A_move4as/fold_assignments.csv`, 6–7 participants per fold).
* **No leakage by construction:**
  - scalers, encoders and class weights are fit on training folds only;
  - per-trial operations use only that trial;
  - thresholds (Youden J on participant-mean probability) and late-fusion weights come from inner CV or
    inner-validation participants;
  - MAE pre-training uses training-fold participants only;
  - the authors' bone-length normalisation, which pools all participants, was deliberately not used.
* **Unit of evaluation:** trials are scored, then probabilities are averaged per participant. Participant-level
  metrics are primary; trial-level metrics are secondary. CIs are from a participant-level bootstrap.
* **Models:**
  - kinematic features (256) + L2-LR;
  - EEG band power (140) + L2-LR;
  - CTR-GCN (official CTRGC blocks, depth 4, width 32, Move4AS 21-body graph);
  - PoseFormerV2 adapter (spatial transformer + DCT low-frequency tokens; not an exact reproduction, since the
    official checkpoints are for 17-joint H36M lifting);
  - Skeleton MAE (temporal-patch tokens, mask 0.5, loss on masked & valid coordinates; STH-MAE not used);
  - EEGNet-lite (1.3 k parameters).
* **Fusion (exactly three):**
  - early = per-modality standardised feature concatenation + LR;
  - intermediate = CTR-GCN and EEGNet encoders → 32-d projections → concatenated head, trained end-to-end;
  - late = weighted average of modality probabilities, weight minimising inner-validation log-loss.
* **Smoke-test training:** 3 epochs, batch 8, AdamW lr 1e-4, AMP on an RTX 4060 Laptop GPU. Validation loss is
  recorded every epoch; no early stopping.

## 4. Results (Steps 11–12)

### 4.1 Experiment A — Move4AS ASD vs non-ASD (paired set, participant level, n = 33)

| Model | Fold AUC (mean ± SD) | Pooled AUC [95 % CI] | Pooled bal. acc. | Pooled MCC (fold mean) | TP/FP/TN/FN |
|---|---|---|---|---|---|
| Motion features + LR | 1.000 ± 0.000 | 0.985 [0.944, 1.000] | 0.867 | 0.771 | 11/1/18/3 |
| EEG band power + LR | 0.628 ± 0.256 | 0.602 [0.385, 0.789] | 0.654 | 0.373 | 8/5/14/6 |
| **Early fusion** (LR) | 0.967 ± 0.075 | 0.966 [0.901, 1.000] | 0.867 | 0.756 | 11/1/18/3 |
| **Late fusion** (LR probs) | 1.000 ± 0.000 | 0.985 [0.944, 1.000] | 0.902 | 0.823 | 12/1/18/2 |
| Motion CTR-GCN | 1.000 ± 0.000 | 0.992 [0.966, 1.000] | 0.938 | 0.891 | 13/1/18/1 |
| EEG CNN | 0.544 ± 0.336 | 0.609 [0.419, 0.796] | 0.494 | −0.025 | 5/7/12/9 |
| **Intermediate fusion** | 1.000 ± 0.000 | 0.985 [0.942, 1.000] | 0.929 | 0.894 | 12/0/19/2 |
| **Late fusion** (deep) | 1.000 ± 0.000 | 0.992 [0.966, 1.000] | 0.938 | 0.891 | 13/1/18/1 |
| Motion PoseFormerV2 | 0.983 ± 0.037 | 0.966 [0.885, 1.000] | 0.912 | 0.837 | 13/2/17/1 |
| Motion Skeleton-MAE | 0.967 ± 0.075 | 0.977 [0.919, 1.000] | 0.929 | 0.894 | 12/0/19/2 |

Supplementary, all QC-ok trials: motion LR AUC 0.992 (33 participants); EEG LR AUC 0.621 (34 participants).
Trial-level AUCs are lower (motion LR 0.895, CTR-GCN 0.932, intermediate fusion 0.949), as expected for single
4-s trials. The full metric set (sensitivity, specificity, PPV, NPV, FPR, FNR, F1, macro-F1, κ, log-loss, Brier)
is in the per-model `*_metrics.json` files and `results_tables.md`. Plots: `outputs/plots/{roc,precision_recall,
confusion_matrices,calibration,distributions,thresholds,training_curves,comparisons}/A_move4as__*`.

**Interpretation.**
- Fold AUCs of 1.000 for 6–7 participants per fold carry very little information. The pooled participant-level
  CIs are the honest summary.
- EEG carries at most weak group information in these 4-s execution windows, and the tiny CNN learned nothing in
  3 epochs.
- Every fusion variant is statistically indistinguishable from motion alone. Late fusion weighted motion
  0.8–1.0 in every fold, and early fusion was slightly *worse* (EEG features add noise to a 33-subject problem).

### 4.2 Sensitivity analyses (feature models; `A_move4as_sensitivity`)

| Analysis | n | Pooled AUC [95 % CI] |
|---|---|---|
| Reference motion LR | 33 | 0.985 [0.944, 1.000] |
| S1 task-balanced participant score | 33 | 0.981 [0.935, 1.000] |
| S2 walk only / dance only | 24 / 30 | 1.000 / 0.981 |
| S3 excluding 40-fps controls | 29 | 0.990 [0.957, 1.000] |
| S4 6 Hz low-pass before features | 33 | 0.959 [0.874, 1.000] |
| S5 no acceleration/jerk features | 33 | 0.955 [0.871, 1.000] |
| S3+S4+S5 combined | 29 | 0.967 [0.895, 1.000] |
| S6 age + sex only (no movement) | 34 | 0.534 [0.323, 0.715] |
| EEG LR excluding 40-fps controls | 29 | 0.771 [0.571, 0.926] |

The motion signal is not explained by task mix, frame rate, high-frequency jitter or demographics. It is still
compatible with session-level acquisition differences: marker placement, tracking quality (cf. the 34 % vs 6 %
QC rejection), smoothing in the capture software, or instructions. Those differences cannot be tested without
per-session metadata from the dataset authors.

### 4.3 Experiment B — MMASD movement analysis (no ASD-vs-non-ASD task)

Activity recognition: 10 classes, 1,367 clips with both OpenPose and ROMP, 32 participants, GroupKFold by
participant. Chance accuracy is ≈ 0.10.

| Model | 2D OpenPose macro-F1 / acc | 3D ROMP macro-F1 / acc |
|---|---|---|
| CTR-GCN | 0.256 / 0.269 | 0.213 / 0.210 |
| PoseFormerV2 adapter | 0.533 / 0.517 | **0.663 / 0.652** |
| Skeleton-MAE (pre-train + fine-tune) | 0.458 / 0.480 | 0.518 / 0.525 |

* 3D ROMP beats 2D OpenPose for the transformer models. CTR-GCN at lr 1e-4 for 3 epochs is under-trained: its
  training loss was still falling, and this says nothing about its capacity.
* MAE reconstruction on held-out validation participants: mean absolute coordinate error 0.171 torso-lengths for
  both 2D and 3D, with validation loss decreasing over all 3 epochs. Reconstruction quality is not a
  classification result.
* ADOS-2 regression, leave-one-participant-out, n = 20: ridge on kinematics gives comparison-score MAE 1.30
  (mean baseline 1.28) and overall-total MAE 9.87 (baseline 7.99), with R² < 0 for both. **No predictive value.**
  The strongly negative correlations are an artefact of leave-one-out mean shrinkage; the mean baseline itself
  has r = −1.

### 4.4 Experiment C — questionnaire (independent)

| Feature set | LogReg AUC | LightGBM AUC | MLP AUC (3 ep.) |
|---|---|---|---|
| `items` (A1–A10 + demographics) — rule reproduction | 1.000 ± 0.000 | 0.997 ± 0.001 | 0.964 ± 0.041 |
| `no_items` (age, sex, ethnicity, jaundice, family history, respondent) | 0.601 ± 0.029 | 0.611 ± 0.060 | 0.543 ± 0.067 |

`items` measures how well a model rediscovers a sum-and-threshold rule; it must not be reported as ASD detection.
`no_items` is the only leakage-free result, and it predicts the *screening outcome* only weakly.

### 4.5 Experiment D — cross-dataset generalisation: **not performed**

| Requirement | Status |
|---|---|
| Compatible target | ✗ — MMASD has no negatives; questionnaire target is a toddler screening rule |
| Harmonisable inputs | ✗ — OptiTrack 21 rigid bodies (adults, lab) vs ROMP/OpenPose from video (children, therapy) |
| Independent but comparable populations | ✗ — adults vs children vs toddlers |

A defensible next step would be representation transfer only, for example MAE pre-training on MMASD with a
SMPL→OptiTrack joint mapping and evaluation on Move4AS folds, reported as such.

### 4.6 Computational efficiency (RTX 4060 Laptop, batch 8, AMP)

| Model | Params | s/epoch | Latency ms/batch | Throughput /s | Peak GPU MB |
|---|---|---|---|---|---|
| EEGNet-lite | 1,345 | 0.7 | 0.3 | 26,742 | 33 |
| CTR-GCN (Move4AS) | 59,439 | 7.6 | 7.3 | 1,097 | 119 |
| Intermediate fusion | 63,601 | 12.4 | 12.5 | 641 | 121 |
| PoseFormerV2 adapter | 371,937 | 4.1 | 2.7 | 2,923 | 155 |
| Skeleton-MAE classifier | 660,477 | 2.3 | 1.0 | 8,047 | 35 |

## 5. Full-training results (`FULL_TRAINING`, primary)

**Protocol.**
- Up to 100 epochs, AdamW lr 5e-4, weight decay 5e-4, batch 32, AMP.
- Early stopping (patience 10) on the inner-validation participants of each training fold; the best-validation
  weights are restored. The outer test fold is never used.
- MAE pre-training ran all 100 epochs, still on training-fold participants only.
- Folds, thresholds and fusion weights follow exactly the smoke-test procedure.
- Feature-based models (logistic regression, LightGBM) are unaffected by epochs, so their results are
  bit-identical to the smoke test, which confirms determinism. This includes all sensitivity analyses
  (Section 4.2) and the ADOS regression.

**Epochs actually run per fold** (best epoch = run length − 10), from `outputs/full/metrics/*/*_history.csv`:

| Experiment | Model | Epochs run per fold |
|---|---|---|
| Move4AS | CTR-GCN | 11–15 |
| Move4AS | EEG CNN | 11–21 |
| Move4AS | Intermediate fusion | 15–24 |
| Move4AS | PoseFormerV2 | 11–23 |
| Move4AS | MAE fine-tune | 11–13 |
| MMASD | CTR-GCN | 26–52 |
| MMASD | PoseFormerV2 | 13–24 |
| MMASD | MAE fine-tune | 12–25 |
| Questionnaire | MLP | 21–85 |

On Move4AS the inner-validation loss bottoms out within the first few epochs. With 5–6 validation participants
this criterion is noisy, so longer training does not mean better generalisation here.

### 5.1 Move4AS ASD vs non-ASD (paired set, participant level, n = 33)

| Model | Fold AUC (mean ± SD) | Pooled AUC [95 % CI] | Pooled bal. acc. | TP/FP/TN/FN | Trial-level pooled AUC | Δ vs smoke (participant AUC) |
|---|---|---|---|---|---|---|
| Motion features + LR | 1.000 ± 0.000 | 0.985 [0.944, 1.000] | 0.867 | 11/1/18/3 | 0.895 | 0 (not epoch-based) |
| EEG band power + LR | 0.628 ± 0.256 | 0.602 [0.385, 0.789] | 0.654 | 8/5/14/6 | 0.590 | 0 |
| **Early fusion** (LR) | 0.967 ± 0.075 | 0.966 [0.901, 1.000] | 0.867 | 11/1/18/3 | 0.869 | 0 |
| **Late fusion** (LR probs) | 1.000 ± 0.000 | 0.985 [0.944, 1.000] | 0.902 | 12/1/18/2 | 0.897 | 0 |
| Motion CTR-GCN | 1.000 ± 0.000 | 0.992 [0.967, 1.000] | 0.938 | 13/1/18/1 | **0.972** | 0.000 (trial +0.040) |
| EEG CNN | 0.650 ± 0.246 | 0.571 [0.374, 0.778] | 0.549 | 8/9/10/6 | 0.542 | −0.038 |
| **Intermediate fusion** | 1.000 ± 0.000 | 0.992 [0.963, 1.000] | **0.974** | 14/1/18/0 | 0.962 | +0.007 |
| **Late fusion** (deep) | 1.000 ± 0.000 | 0.992 [0.967, 1.000] | 0.938 | 13/1/18/1 | 0.972 | 0.000 |
| Motion PoseFormerV2 | 0.917 ± 0.118 | 0.959 [0.881, 1.000] | 0.885 | 13/3/16/1 | 0.955 | −0.007 |
| Motion Skeleton-MAE | 0.983 ± 0.037 | 0.989 [0.956, 1.000] | 0.902 | 12/1/18/2 | 0.961 | +0.012 |

**Interpretation.**
- *Late fusion (deep):* the inner-validation log-loss chose motion weight **1.0 in all five folds**, so the
  model is identical to motion-only CTR-GCN. The feature-level late fusion chose 0.9.
- *Intermediate fusion:* its balanced accuracy (0.974 vs 0.938) differs from CTR-GCN by one participant, well
  inside the CIs. EEG adds no demonstrable value.
- *EEG CNN:* still at chance after full training. The EEG representation, not the epoch budget, is the
  bottleneck.
- *Motion:* full training improved trial-level discrimination (CTR-GCN trial AUC 0.932 → 0.972).
  Participant-level separation was already at ceiling.
- *Overall:* the acquisition-confound caveat (Section 4.2) applies unchanged. Better models mainly fit the
  group difference more tightly; they do not clarify where it comes from.

### 5.2 MMASD activity recognition (10 classes, 1,367 clips, 32 participants, chance ≈ 0.10)

| Model | 2D OpenPose macro-F1 / acc | 3D ROMP macro-F1 / acc | 3D macro-AUC (OvR) | Smoke → full (3D macro-F1) |
|---|---|---|---|---|
| CTR-GCN | 0.694 / 0.702 | 0.688 / 0.693 | 0.933 | 0.213 → 0.688 |
| PoseFormerV2 adapter | 0.646 / 0.657 | 0.745 / 0.735 | 0.944 | 0.663 → 0.745 |
| Skeleton-MAE (pre-train + fine-tune) | 0.685 / 0.691 | **0.775 / 0.775** | 0.946 | 0.518 → 0.775 |

- **Non-deep baselines** (`scripts/run_mmasd_feature_baselines.py`, same clips and folds):
  - majority class: balanced accuracy 0.100, macro-F1 0.040;
  - 3D kinematic features (per-joint speed, acceleration, range of motion) + logistic regression: macro-F1
    **0.720**;
  - same features + LightGBM: 0.661.

  So simple kinematics come within about 0.05 of the best deep model. The deep models' advantage is real but
  modest, and their bootstrap CIs overlap.
- **Training budget:** the smoke-test ranking was an artefact of training length. Fully trained, the three
  architectures land within about 0.09 macro-F1 of each other.
- **2D vs 3D:**
  - 3D beats 2D for the transformer models: PoseFormerV2 +0.10 and MAE +0.09 macro-F1.
  - CTR-GCN is equal on both (0.69).
  - Fold SDs are 0.02–0.10, so only the transformer 3D gains are reasonably clear.
- **Self-supervised pre-training:** MAE pre-training on training-fold participants is the best 3D option.
- **Reconstruction:** after 100 pre-training epochs, the validation masked-coordinate MAE fell to 0.090 (2D) /
  0.093 (3D) torso-lengths, from 0.171 / 0.171 in the smoke test.
- **ADOS-2 regression:** unchanged, because it is feature-based. It has no predictive value (R² < 0).

### 5.3 Questionnaire (Q-CHAT-10 toddlers)

| Feature set | LogReg AUC | LightGBM AUC | MLP AUC (full) | MLP AUC (smoke) |
|---|---|---|---|---|
| `items` (rule reproduction / leakage) | 1.000 ± 0.000 | 0.997 ± 0.001 | 0.999 ± 0.001 | 0.964 ± 0.041 |
| `no_items` (leakage-controlled) | 0.601 ± 0.029 | 0.611 ± 0.060 | 0.588 ± 0.049 | 0.543 ± 0.067 |

The fully trained MLP now also rediscovers the scoring rule, which is expected. The leakage-free signal remains
weak (AUC ≈ 0.59–0.61) whatever the model.

### 5.4 Efficiency (full training, inference batch = 32, RTX 4060 Laptop)

| Model | Params | s/epoch | Train time, 5 folds | Peak GPU MB |
|---|---|---|---|---|
| Move4AS CTR-GCN | 59 k | 1.8 | 122 s | 327 |
| Move4AS intermediate fusion | 64 k | 2.0 | 196 s | 328 |
| MMASD CTR-GCN (3D) | 62 k | 1.0 | 221 s | 384 |
| MMASD PoseFormerV2 (3D) | 385 k | 0.7 | 67 s | 480 |
| MMASD MAE (3D, fine-tune only) | 669 k | 0.3 | 31 s | 45 |

The whole full-training chain took **36 minutes**, from 23:33 to 00:09. Full efficiency tables, including
latency and throughput, are in `outputs/full/reports/results_tables.md`.

## 6. Limitations

1. **Tiny cohort:** 33 participants (14 ASD) and 6–7 participants per test fold. A single misclassified
   participant moves balanced accuracy by about 7 points, and CIs are wide. Fold-wise AUC = 1.000 must not be
   quoted alone.
2. **Possible acquisition confound in Move4AS motion** (Section 4.2). Near-perfect separation should be treated as
   a red flag until it is replicated with acquisition-matched recordings or confirmed with the dataset authors.
3. **Training-budget sensitivity:** the smoke test (3 epochs) ranked architectures wrongly on MMASD. The
   full-training ranking rests on a single 5-fold split and has not been tuned per architecture: one
   learning rate, no augmentation, small model widths. In full training, the same inner-validation participants
   are used both for early stopping and for threshold/fusion-weight selection. That makes threshold choice
   slightly optimistic within the training fold, but it never touches the test fold.
4. **No calibration step.** Class-weighted models shift probabilities. Brier scores and calibration plots are
   reported, but Platt or isotonic calibration on inner folds is left for full training.
5. **MMASD subject identity is a heuristic.** The tracked person may be a therapist in some clips. Pixel-unit
   ROMP "3D" depth is not metric.
6. **Questionnaire target is a screening rule**, and the leakage-free model uses only demographics. Population
   (toddlers, app users) differs entirely from the other datasets.
7. **Unused data:** EEG preprocessing is automated (filter + amplitude rejection) rather than the authors' manual
   ICA pipeline. IMU and the 71-joint extras were not used.
8. **Alignment:** 8 of 155 blocks were excluded from trial-level fusion.

## 7. Recommendations (after full training)

1. Full training is done. Next: add inner-fold probability calibration and repeated CV (e.g. 10 × 5-fold, or
   leave-one-participant-out for Move4AS) to stabilise estimates. Use a separate inner split for early stopping
   vs threshold selection. Run a small per-architecture learning-rate sweep in nested CV before claiming any
   MMASD architecture ranking.
2. Before scaling models, contact the Move4AS authors about per-session acquisition settings (capture software
   filtering, calibration, dates, marker sets) and the asymmetric QC rejection. Add session metadata as a
   covariate or matching variable.
3. Add EEG ICA artefact removal (or use the authors' epoched EEGLAB files) and event-related features
   (μ/β desynchronisation relative to baseline). The current EEG representation may simply be too crude.
4. For MMASD, add a child-identification step (e.g. smallest skeleton or clinician-annotated seed frames) and
   evaluate ADOS prediction only with a pre-registered, participant-level design.
5. Any three-modality fusion requires a questionnaire (e.g. SRS-2/AQ-10) collected from all Move4AS participants
   under the same protocol.

## 8. Reproducibility

* Code: `scripts/` and `src/`. Environment: `requirements.txt` (Python 3.12.10, torch 2.14.1+cu126). Hardware:
  Windows 11, i7-14650HX, 14 GB RAM, RTX 4060 Laptop 8 GB.
* Order of execution: `README.md`; `notebooks/ASD_multimodal_workflow.ipynb`; `scripts/run_pipeline_chain.sh`.
* Full training: `bash scripts/run_full_training_chain.sh` runs questionnaire → Move4AS → sensitivity → MMASD →
  tables sequentially with `configs/full_training.yaml`. Its artefacts are under `outputs/full/`, with the same
  experiment IDs.
* Model-comparison figures: `scripts/make_comparison_figures.py --config configs/full_training.yaml` writes
  `outputs/full/plots/model_comparison/*.jpg`, each with a CSV of the plotted values. They cover:
  - MMASD balanced accuracy, macro-F1 and per-activity F1;
  - Move4AS fusion and motion-encoder balanced accuracy / ROC-AUC;
  - confound checks;
  - questionnaire models.

  Bars are pooled out-of-fold values; whiskers are 95 % participant-bootstrap CIs.
* Artefacts are linked by experiment ID (`A_move4as`, `A_move4as_sensitivity`, `B_mmasd`, `C_questionnaire`):
  - fold assignments: `outputs/metrics/<exp>/fold_assignments.csv`
  - out-of-fold predictions: `outputs/predictions/`
  - checkpoints (per fold): `outputs/checkpoints/<exp>/`
  - training histories: `outputs/metrics/<exp>/*_history.csv`
  - logs: `outputs/logs/`
* Note on execution: the run was interrupted once by an out-of-memory condition while three jobs shared 14 GB of
  RAM. The affected models (Move4AS MAE, late-fusion-deep, supplementary all-trial models, sensitivity analyses)
  were rerun sequentially with identical seeds and folds. CTR-GCN and the EEG CNN were retrained in that rerun
  and reproduced their earlier participant-level metrics exactly.
