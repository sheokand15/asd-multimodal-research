"""Dataset audit -> outputs/audit/{mmasd,move4as,questionnaire}_audit.json, dataset_inventory.csv,
dataset_comparison.md.  Read-only with respect to raw data.

Run after scripts/prepare_datasets.py (extraction) and the build_* scripts (which produce the
QC/alignment tables that the audit summarises).
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import mmasd as MD  # noqa: E402
from src.data import move4as as M  # noqa: E402
from src.data import questionnaire as Q  # noqa: E402
from src.utils.config import load_config, save_json  # noqa: E402


def inventory(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_file():
            rec = {"dataset_hint": "", "path": str(p), "kind": "archive" if p.suffix == ".zip" else "file",
                   "size_bytes": p.stat().st_size}
            if p.suffix == ".zip":
                try:
                    with zipfile.ZipFile(p) as z:
                        infos = z.infolist()
                        rec.update(n_members=len(infos), uncompressed_bytes=sum(i.file_size for i in infos),
                                   zip_open="ok")
                except zipfile.BadZipFile as e:
                    rec["zip_open"] = f"BAD: {e}"
            rows.append(rec)
        else:
            files = [f for f in p.rglob("*") if f.is_file()]
            rows.append({"path": str(p), "kind": "extracted_dir", "n_files": len(files),
                         "size_bytes": sum(f.stat().st_size for f in files)})
    df = pd.DataFrame(rows)
    hint = {"28296518": "Move4AS (figshare bundle)", "drive-download": "MMASD (Google Drive part)",
            "optical_flow": "MMASD optical flow (not used: no skeleton/EEG/questionnaire)",
            "autism-screening": "Questionnaire (Kaggle Q-CHAT-10 toddlers)", "Move4AS": "Move4AS extracted",
            "MMASD": "MMASD extracted", "Questionnaire": "Questionnaire extracted"}
    df["dataset_hint"] = [next((v for k, v in hint.items() if k in Path(p).name), "") for p in df["path"]]
    return df


def audit_move4as(cfg) -> dict:
    root = cfg["paths"]["move4as"]
    proc = Path(cfg["paths"]["processed"]) / "move4as"
    blocks = M.list_blocks(root / "Dataset")
    parts = pd.read_excel(root / "Participants.xlsx", header=None, dtype=str)
    al = pd.read_csv(proc / "alignment.csv")
    trials = pd.read_csv(proc / "trials_all.csv")
    eI, mI = pd.read_csv(proc / "eeg_index.csv"), pd.read_csv(proc / "motion_index.csv")
    pids = sorted(blocks.participant.unique(), key=lambda s: (s[0], int(s[1:])))
    per = blocks.pivot_table(index="participant", columns="modality", values="block", aggfunc="count").fillna(0).astype(int)
    demo = parts.iloc[1:, :6]
    demo.columns = ["id", "age", "gender", "laterality", "AQ", "FSIQ"]
    demo = demo[demo.id.fillna("").str.match(r"^[SP]\d+$")].copy()
    demo["group"] = np.where(demo.id.str.startswith("P"), "ASD", "control")
    demo["age"] = pd.to_numeric(demo.age, errors="coerce")
    fps = al.assign(participant=al.block_id.str.extract(r"^([SP]\d+)")[0])
    tm = trials[(trials.modality == "motion") & trials.qc.isin(["ok", "not_selected_by_authors_qc", "short_or_gappy"])]
    rej = (tm.qc == "not_selected_by_authors_qc").groupby(tm.participant.str.startswith("P")).mean()
    qc_rej = (float(rej.get(False, np.nan)), float(rej.get(True, np.nan)))
    with open(root / "README.txt", encoding="utf-8", errors="replace") as fh:
        readme = fh.read()
    return {
        "dataset": "Move4AS (Scientific Data 2025, doi:10.1038/s41597-025-05313-0; figshare 28296518 v1)",
        "license": "CC BY 4.0 (figshare record)",
        "source_integrity": "28296518.zip and nested Dataset.zip CRC-tested OK; member sizes/MD5 checked by "
                            "scripts/download_move4as.py --verify-only against the figshare API",
        "label_definition": "folder prefix P* = autistic (clinical group), S* = neurotypical control (README)",
        "positive_class": "ASD (P*) = 1",
        "participants": {"total": len(pids), "ASD": sum(p.startswith("P") for p in pids),
                         "control": sum(p.startswith("S") for p in pids), "ids": pids},
        "readme_reported": "20 neurotypical + 14 autistic participants" if "20 Neurotypical" in readme else readme[:200],
        "files_per_participant": per.reset_index().to_dict(orient="records"),
        "file_counts": blocks.groupby("modality").size().to_dict(),
        "file_formats": {"eeg": "MATLAB .mat, eegDataT (n_ch+2, samples): n_ch EEG rows, stim row, cue row",
                         "mdata": "MATLAB .mat: per-frame rigidbodyData (7x21: xyz+quat), markerData (3x37), "
                                  "timestamp, stimRegister, cueRegister",
                         "imu": "MATLAB .mat imuDataT (5, samples) — not used (not requested; undocumented channels)"},
        "eeg": {"sampling_rate_hz": 250, "n_channels_by_block": al.eeg_n_ch.value_counts().to_dict(),
                "channels_used": M.COMMON_CH,
                "channel_note": "15-ch layout = 16-ch minus Oz; PO3 is noise for 14-usable-channel participants "
                                "(authors' script) -> 14 common channels used",
                "events": "stim codes 1 baseline, 2 instruction, 3 high beep, 4 execution (4 s), 5 low beep, 6 return",
                "trials_total": int((trials.modality == "eeg").sum()),
                "trials_qc": trials[trials.modality == "eeg"].qc.value_counts().to_dict(),
                "per_participant_observations": eI.participant.value_counts().sort_index().to_dict(),
                "observations_file": "EEG_Participants_info.xlsx (impedance/artifact notes)"},
        "motion": {"rigid_bodies": M.JOINTS, "edges": M.EDGES,
                   "fps_by_block": fps.motion_fps.value_counts().to_dict(),
                   "blocks_40fps": fps.loc[fps.motion_fps < 59, "block_id"].tolist(),
                   "participants_with_40fps_blocks": sorted(fps.loc[fps.motion_fps < 59, "participant"].unique().tolist()),
                   "mdata_blocks": int((blocks.modality == "mdata").sum()),
                   "participants_without_motion": sorted(set(pids) - set(blocks[blocks.modality == "mdata"].participant)),
                   "trial_qc": trials[trials.modality == "motion"].qc.value_counts().to_dict(),
                   "qc_sources": ["authors' blacklist (01_parse_and_extract_parts.py)",
                                  "authors' Selected Trials (3DMotion_Participants_info.xlsx)",
                                  "own: >=4 s execution, max frame gap <=0.25 s, <=20 % missing"],
                   "trials_ok": int(len(mI)), "participants_ok": int(mI.participant.nunique()),
                   "trials_ok_by_group_task": mI.groupby(["label", "task"]).size().rename(
                       index={0: "control", 1: "ASD"}, level=0).reset_index().to_dict(orient="records")},
        "alignment": {"method": "per block: execution onsets (stim==4) in both modalities; offset s chosen so "
                                "EEG trial k+s <-> motion trial k maximises cue agreement; accepted if >=95 % "
                                "agreement over >=20 trials",
                      "blocks_with_both": int(al.motion_onsets.notna().sum()),
                      "blocks_aligned": int(al.trial_level_aligned.fillna(False).astype(bool).sum()),
                      "shift_counts": al.align_shift.value_counts().to_dict(),
                      "unaligned_blocks": al.loc[al.motion_onsets.notna() & ~al.trial_level_aligned.fillna(False).astype(bool),
                                                 "block_id"].tolist(),
                      "timestamp_note": "motion timestamps irregular (effective 50-60 Hz) -> resampled on time grid"},
        "demographics": {g: {"n": int(len(d)), "age_mean": float(d.age.mean()), "age_sd": float(d.age.std()),
                             "female": int((d.gender.str.strip() == "Female").sum())}
                         for g, d in demo.groupby("group")},
        "clinical_scores": "AQ for controls only; ADOS-2 total/Com/SI for some ASD (NA or 'Positive' for others) "
                           "-> not used as features (group-specific availability would leak the label)",
        "confounds_found": [
            "task mix differs: controls 76 % dance trials vs ASD 59 % (motion QC-ok trials)",
            "4 control participants (S6, S8, S9, S10) recorded at 40 fps vs 60 fps for everyone else",
            "acceleration/jerk features differ strongly between groups even within task and at 60 fps -> "
            "possible acquisition/smoothing difference; see sensitivity analyses",
            "sex ratio differs (controls 7/20 female, ASD 2/14) and ASD group older on average",
            "authors' visual-inspection QC ('Selected Trials') rejects {:.0%} of control vs {:.0%} of ASD motion "
            "trials from usable blocks -> tracking quality differs systematically by group".format(*qc_rej),
        ],
        "duplicates": "none found (unique block ids)",
        "missing": {"S2": "no motion data (all sessions 'Data not available')",
                    "walk motion missing": ["P3", "P4", "P13"]},
    }


def audit_mmasd(cfg) -> dict:
    root = cfg["paths"]["mmasd"]
    proc = Path(cfg["paths"]["processed"]) / "mmasd"
    clips = pd.read_csv(proc / "clips_all.csv")
    ados = pd.read_excel(root / "ADOS_rating.xlsx")
    ados = ados[ados["ID#"].notna() & ados["ID#"].astype(str).str.match(r"^\d+")]
    ados_ids = set(ados["ID#"].astype(str).str.extract(r"^(\d+)")[0])
    pids = set(clips.participant.astype(str))
    modal = {}
    for name, sub in [("openpose_2d", "2D_openpose/output"), ("romp_2d", "ROMP_2D_Coordinates"),
                      ("romp_3d_24", "ROMP_3D_Coordinates"), ("romp_3d_71", "ROMP_3D_Coordinates_71-joints")]:
        d = root / sub
        acts = sorted(a.name for a in d.iterdir() if a.is_dir())
        modal[name] = {"activities": acts, "n_clips": sum(len(list((d / a).iterdir())) for a in acts)}
    dx = ados.iloc[:, 7]
    return {
        "dataset": "MMASD (ICMI 2023, doi:10.1145/3577190.3614117)",
        "license": "See https://github.com/Li-Jicheng/MMASD-A-Multimodal-Dataset-for-Autism-Intervention-Analysis "
                   "(research use; check Google Drive terms)",
        "source_integrity": "3 Google-Drive part zips + inner zips CRC-tested OK; optical_flow-001.zip not extracted "
                            "(optical flow not part of this study)",
        "readme_reported": "32 children with ASD, 1,315 samples, 11 activities, 25 OpenPose / 71 ROMP keypoints",
        "observed": {"participants": len(pids), "clips_romp": int(len(clips)),
                     "clips_by_activity": clips.activity.value_counts().to_dict(),
                     "clips_per_participant": clips.participant.astype(str).value_counts().to_dict(),
                     "modalities": modal,
                     "note": "1,693 clips observed vs 1,315 in README (README count may predate the release); "
                             "OpenPose lacks frog_pose (1,374 clips)"},
        "control_group": "NONE. All 32 participants are children with ASD; ADOS-2 classification column contains "
                         f"only {sorted(dx.dropna().astype(str).unique().tolist())} (1 = ASD, 2 = autism). "
                         "=> No ASD-vs-non-ASD classification is performed on MMASD.",
        "clinical_scores": {"columns": [str(c).strip() for c in ados.columns],
                            "n_rows": int(len(ados)), "ids_with_scores": sorted(ados_ids),
                            "participants_with_clips_and_scores": sorted(pids & ados_ids),
                            "n_overlap": len(pids & ados_ids),
                            "duplicate_note": "40023 appears twice (40023 and 40023_R, identical scores)"},
        "modality_correspondence": "Verified: identical clip names and frame counts across OpenPose and ROMP; "
                                   "ROMP_2D == xy of ROMP_3D_24; ROMP_3D_24 == first 24 joints of ROMP_3D_71",
        "coordinates": "pixel units (ROMP 3D third axis is depth in pixel-like units)",
        "frames_per_clip": clips.n_frames.describe().to_dict(),
        "persons_per_frame": {"mean_of_clip_means": float(clips.persons_mean.mean()),
                              "max": int(clips.persons_max.max())},
        "subject_identity": "NOT annotated; primary tracked person heuristic (see src/data/mmasd.py)",
        "tracking_qc": {"track_coverage_3d_median": float(clips.track_coverage_3d.median()),
                        "track_coverage_2d_median": float(clips.track_coverage_2d.median()),
                        "openpose_match_dist_px_median": float(clips.openpose_match_dist_px.median())},
        "clip_name_fields": "activity_participant_day_clipNo_suffix; suffix in {i, n, y} undocumented",
        "errors": clips.filter(like="error").dropna(how="all").shape[0] if "error" in clips else 0,
    }


def comparison_md(m4, mm, q) -> str:
    return f"""# Dataset comparison

| | Move4AS | MMASD | Q-CHAT-10 toddlers (Kaggle) |
|---|---|---|---|
| Population | adults (ASD {m4['participants']['ASD']}, control {m4['participants']['control']}) | children with ASD only ({mm['observed']['participants']}) | toddlers 12-36 months ({q['n_rows']} rows) |
| ASD label | clinical group (folder P*/S*) | none (all ASD) | screening outcome = Q-CHAT sum > 3 |
| Valid binary ASD vs non-ASD task | **yes** | **no** | only as rule reproduction / demographics-only |
| Modalities | 3D mocap (21 rigid bodies), EEG (15/16 ch, 250 Hz), IMU | OpenPose 2D (25), ROMP 3D (24/71), optical flow, ADOS-2 | 10 binary items + demographics |
| Participant IDs | P1-P14, S1-S20 | 5-digit IDs | Case_No (row id) |
| Clinical scores | AQ (controls), ADOS (some ASD) | ADOS-2 for {mm['clinical_scores']['n_overlap']} participants with clips | none (screen score only) |
| Licence | CC BY 4.0 | research dataset (see repo) | Kaggle / original author terms |
| Key issues | small n; task-mix and 40-fps acquisition confounds | no controls; subject identity unlabelled in multi-person frames | target leakage by construction; Combined file mislabelled |

**Cross-dataset pairing:** none of the three datasets share participants, age group or label definition.
No multimodal fusion across datasets is performed; questionnaire fusion with Move4AS/MMASD is impossible
(would require questionnaires collected from the same Move4AS participants — e.g. the AQ, which exists only for
controls and therefore cannot be used).
"""


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    out = Path(cfg["paths"]["audit"])
    out.mkdir(parents=True, exist_ok=True)
    inv = inventory(cfg["paths"]["dataset_root"])
    inv.to_csv(out / "dataset_inventory.csv", index=False)
    m4 = audit_move4as(cfg)
    save_json(m4, out / "move4as_audit.json")
    mm = audit_mmasd(cfg)
    save_json(mm, out / "mmasd_audit.json")
    q = Q.audit(cfg["paths"]["questionnaire"])
    save_json(q, out / "questionnaire_audit.json")
    (out / "dataset_comparison.md").write_text(comparison_md(m4, mm, q), encoding="utf-8")
    print("audit written to", out)


if __name__ == "__main__":
    main()
