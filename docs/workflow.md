# Workflow

All commands run from `E:\ASD Project` with the project venv (`.venv\Scripts\python`). Every step
reads `configs/default.yaml` (SMOKE_TEST) unless `--config configs/full_training.yaml` is given.

| # | Step | Command / module | Inputs | Outputs | Depends on | Fails when |
|---|---|---|---|---|---|---|
| 1 | Dataset discovery & download recovery | `scripts/download_move4as.py --verify-only` (Move4AS); manual Google-Drive / Kaggle files for MMASD / questionnaire | `Dataset/*.zip`, figshare API | `outputs/audit/move4as_download_manifest.json` | network for API check | API unreachable (script prints manual recovery steps), MD5 mismatch |
| 2 | File integrity validation | `scripts/prepare_datasets.py` | raw zips | extracted dirs, `extract_manifest_*.json` | disk space (≈25 GB free needed) | CRC failure, unsafe member path |
| 3 | Dataset audit | `scripts/audit_datasets.py` | extracted data + build QC tables | `*_audit.json`, `dataset_inventory.csv`, `dataset_comparison.md` | steps 2, 6a | missing files |
| 4 | Label & participant validation | in audit + loaders | `Participants.xlsx`, `ADOS_rating.xlsx`, CSV | label maps, cohort counts | 2 | label column missing/ambiguous |
| 5 | Modality alignment | `scripts/build_move4as.py` (`align_by_cue_sequence`) | EEG + motion stim/cue registers | `alignment.csv`, `motion_index.eeg_trial_idx` | 2 | <20 overlapping trials or <95 % cue agreement → block excluded from fusion |
| 6a | Preprocessing | `build_move4as.py`, `build_mmasd.py` | raw `.mat`, JSON, npz | `data/processed/{move4as,mmasd}/` | 2 | malformed files (logged per block/clip) |
| 6b | Participant-level splitting | `src/training/cv.py` | participant IDs, labels | `outputs/metrics/<exp>/fold_assignments.csv` | 6a | fewer participants than folds per class |
| 7 | Baselines | `run_questionnaire.py`, feature-LR part of `run_move4as.py` | features | OOF predictions, metrics | 6 | — |
| 8 | Skeleton models | CTR-GCN, PoseFormerV2 adapter, MAE in `run_move4as.py` / `run_mmasd.py` | tensors | checkpoints, histories | 6, GPU optional | CUDA OOM → lower batch size |
| 9 | Three-epoch smoke tests | default config (`run_label: SMOKE_TEST`) | — | all of `outputs/` | 7-8 | — |
| 10 | Fusion experiments | `run_move4as.py` (early / intermediate / late) | paired trials | fusion metrics | 5, 6 | no paired trials |
| 11 | Cross-validation evaluation | `src/evaluation/report.py` | OOF predictions | fold CSVs, pooled JSON, participant-bootstrap CIs | 9-10 | metric undefined → `null` + reason |
| 12 | Metric & plot generation | `src/evaluation/plots.py` | OOF predictions | `outputs/plots/**` | 11 | — |
| 13 | Sensitivity analyses | `scripts/run_move4as_sensitivity.py` | paired features | `A_move4as_sensitivity/*` | 10 | — |
| 14 | Final reporting | `scripts/make_report.py` | all metrics | `outputs/reports/final_report.md` | 11-13 | — |

## Leakage controls (applied everywhere)
1. Folds are formed on participants (StratifiedGroupKFold); an assertion checks train/test participant disjointness.
2. Normalisation (StandardScaler), imputation, one-hot vocabularies, class weights and resampling are fit on
   training-fold data only. Per-trial operations (filtering, root-centring, body-size scaling, z-scoring) use no
   other participant.
3. Thresholds (Youden J on participant-mean probabilities) and late-fusion weights use training-fold inner
   predictions (3-fold inner CV for sklearn models; 20 % inner-validation participants for deep models).
4. MAE pre-training uses only the outer-training participants of each fold.
5. The authors' bone-length normalisation (`03_normalize.py`) averages across *all* participants and was
   deliberately **not** used.
