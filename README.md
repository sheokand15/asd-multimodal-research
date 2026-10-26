# Multimodal ASD research prototype (MMASD · Move4AS · Q-CHAT-10)

> **Academic research prototype. Not a clinically validated diagnostic system. Do not use for diagnosis.**

Participant-level, leakage-controlled comparison of skeleton, EEG and questionnaire models and of exactly
three fusion strategies (early, intermediate, late). Final report: [`outputs/reports/final_report.md`](outputs/reports/final_report.md).

## Layout
```
E:/ASD Project/
├── configs/        default.yaml (SMOKE_TEST, 3 epochs) · full_training.yaml (100 epochs + early stopping)
├── scripts/        prepare_datasets · download_move4as · build_move4as · build_mmasd · audit_datasets
│                   run_questionnaire · run_move4as · run_move4as_sensitivity · run_mmasd · make_report
├── src/            data/ models/ fusion/ training/ evaluation/ utils/
├── notebooks/      ASD_multimodal_workflow.ipynb
├── docs/           architecture.md · architecture.mmd · workflow.md
├── Dataset/        raw archives (untouched) + extracted Move4AS/ MMASD/ Questionnaire/
├── data/processed/ trial / clip tensors and features
└── outputs/        audit/ checkpoints/ predictions/ metrics/ plots/ logs/ reports/
```

## Windows setup
```powershell
cd "E:\ASD Project"
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```
Paths contain a space — always quote them. All code uses `pathlib` and reads paths from `configs/*.yaml`.

## Data
Put the original archives in `E:\ASD Project\Dataset\` (already present on this machine):

| file | source |
|---|---|
| `28296518.zip` | Move4AS, figshare "Download all" — https://springernature.figshare.com/articles/dataset/Move4AS_A_Multimodal_Dataset_Addressing_Motor_Function_in_Autism/28296518 |
| `drive-download-*-1-00{2,3,4}.zip` | MMASD Google Drive (https://tinyurl.com/mr3rd2dp): 2D_openpose, ROMP 2D/3D(24/71), ADOS_rating.xlsx |
| `optical_flow-001.zip` | MMASD optical flow (not used) |
| `autism-screening-for-toddlers-data.zip` | Kaggle Q-CHAT-10 toddler screening |

Verify Move4AS against the official MD5s (downloads only what is missing/corrupt; never deletes):
```powershell
.venv\Scripts\python scripts\download_move4as.py --verify-only
```

## Run everything (SMOKE_TEST)
```powershell
.venv\Scripts\python scripts\prepare_datasets.py          # CRC check + safe extraction (~10 min)
.venv\Scripts\python scripts\build_move4as.py --workers 4  # trials, EEG/motion alignment (~1 min)
.venv\Scripts\python scripts\build_mmasd.py --workers 4    # person tracking, 2D/3D tensors (~30 min)
.venv\Scripts\python scripts\audit_datasets.py
.venv\Scripts\python scripts\run_questionnaire.py
.venv\Scripts\python scripts\run_move4as.py
.venv\Scripts\python scripts\run_move4as_sensitivity.py
.venv\Scripts\python scripts\run_mmasd.py
.venv\Scripts\python scripts\make_report.py
```
Full training: append `--config configs\full_training.yaml` to the `run_*` and `make_report` commands
(outputs go to `outputs/full/`). Or open `notebooks/ASD_multimodal_workflow.ipynb`.

## Conventions
* Positive class = ASD (label 1). Probabilities are P(ASD).
* Participant-level CV everywhere (StratifiedGroupKFold, seed 42); fold assignments saved per experiment.
* Metrics that cannot be computed are stored as `null` with a reason, never as 0.
* Experiment IDs: `A_move4as`, `A_move4as_sensitivity`, `B_mmasd`, `C_questionnaire` link configs, checkpoints
  (`outputs/checkpoints/<exp>/`), predictions, metrics and plots (`outputs/plots/*/<exp>__<model>_*.png`).
