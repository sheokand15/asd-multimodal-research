"""Build processed Move4AS trial tensors/features + alignment and QC reports.

Outputs (data/processed/move4as/):
  eeg_X.npy        (N_e, 14, 500)  EEG execution windows, 1-45 Hz, downsampled to 125 Hz, per-trial z-scored
  eeg_F.npy        (N_e, 140)      log/relative band powers
  eeg_index.csv    trial metadata (participant, label, task, block, trial_idx, cue, qc)
  motion_X.npy     (N_m, 128, 21, 3) normalised 3D rigid-body skeleton over the 4-s execution window
  motion_F.npy     (N_m, 256)      kinematic features
  motion_index.csv
  alignment.csv    per block: onset counts and cue-sequence agreement between EEG and motion
  trials_all.csv   every trial incl. rejected ones with reason
Usage:  .venv\\Scripts\\python scripts\\build_move4as.py [--workers 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import move4as as M  # noqa: E402
from src.utils.config import load_config, save_json  # noqa: E402


def process_block(block_id: str, eeg_path: str | None, mdata_path: str | None, selected, n_frames: int,
                  sfreq: float, band):
    res = {"block_id": block_id, "eeg": [], "motion": [], "align": {"block_id": block_id}}
    eeg_cues = mot_cues = None
    if eeg_path:
        try:
            eeg, stim, cue, n_ch = M.load_eeg_block(Path(eeg_path), sfreq, band)
            trials = M.eeg_trials(eeg, stim, cue, sfreq)
            res["align"].update(eeg_n_ch=n_ch, eeg_duration_s=len(stim) / sfreq, eeg_onsets=len(trials))
            eeg_cues = [t["cue"] for t in trials]
            for t in trials:
                if t["data"] is not None:
                    seg = t["data"]
                    t["feat"] = M.eeg_features(seg, sfreq)
                    ds = signal.resample_poly(seg, 1, 2, axis=1)  # 250 -> 125 Hz
                    ds = (ds - ds.mean(1, keepdims=True)) / (ds.std(1, keepdims=True) + 1e-6)
                    t["data"] = ds.astype(np.float32)
                res["eeg"].append(t)
        except Exception as e:
            res["align"]["eeg_error"] = f"{type(e).__name__}: {e}"
    if mdata_path:
        try:
            ts, pos, stim, cue = M.load_motion_block(Path(mdata_path))
            trials = M.motion_trials(ts, pos, stim, cue, n_frames)
            res["align"].update(motion_fps=float(1 / np.median(np.diff(ts))), motion_duration_s=float(ts[-1] - ts[0]),
                                motion_onsets=len(trials))
            mot_cues = [t["cue"] for t in trials]
            rank = 0
            for t in trials:
                # index used by the authors' "Selected Trials" column: rank among executions > 4 s
                t["authors_idx"] = rank if t["exec_len_s"] > M.TRIAL_S else None
                rank += t["exec_len_s"] > M.TRIAL_S
                if t["qc"] == "ok" and selected is not None and t["authors_idx"] not in selected:
                    t["qc"], t["data"] = "not_selected_by_authors_qc", None
                if t["data"] is not None:
                    xn = M.normalise_skeleton(t["data"])
                    t["feat"] = M.motion_features(xn, M.TRIAL_S / (n_frames - 1))
                    t["data"] = xn
                res["motion"].append(t)
        except Exception as e:
            res["align"]["motion_error"] = f"{type(e).__name__}: {e}"
    if eeg_cues is not None and mot_cues is not None:
        res["align"]["same_onset_count"] = len(eeg_cues) == len(mot_cues)
        res["align"]["cue_sequence_match_unshifted"] = eeg_cues == mot_cues
        res["align"].update(align_by_cue_sequence(eeg_cues, mot_cues))
    return res


def align_by_cue_sequence(eeg_cues, mot_cues, max_shift=3, min_overlap=20, min_agree=0.95):
    """Find the trial offset s such that EEG trial (k + s) <-> motion trial k.

    The pseudo-random 30-trial cue sequence (3 classes) acts as a fingerprint: chance agreement of a
    wrong offset is ~1/3, so >=95 % agreement over >=20 trials identifies the offset unambiguously.
    """
    best = {"shift": None, "agree": -1.0, "overlap": 0}
    for s in range(-max_shift, max_shift + 1):
        pairs = [(eeg_cues[k + s], mot_cues[k]) for k in range(len(mot_cues)) if 0 <= k + s < len(eeg_cues)]
        if len(pairs) < min_overlap:
            continue
        ag = float(np.mean([a == b for a, b in pairs]))
        if ag > best["agree"]:
            best = {"shift": s, "agree": ag, "overlap": len(pairs)}
    ok = best["shift"] is not None and best["agree"] >= min_agree
    return {"align_shift": best["shift"], "align_cue_agreement": best["agree"], "align_overlap": best["overlap"],
            "trial_level_aligned": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    m4 = cfg["paths"]["move4as"]
    out = Path(cfg["paths"]["processed"]) / "move4as"
    out.mkdir(parents=True, exist_ok=True)
    n_frames = cfg["experiment"]["max_sequence_length"]
    sfreq = cfg["move4as"]["eeg_sfreq"]
    band = tuple(cfg["move4as"]["eeg_band_hz"])

    blocks = M.list_blocks(m4 / "Dataset")
    qc = M.parse_motion_qc_sheet(m4 / "3DMotion_Participants_info.xlsx")
    qc_sel = {r.block_id: r.selected for r in qc.itertuples()}
    qc.to_csv(out / "motion_qc_sheet_parsed.csv", index=False)

    jobs = []
    for bid, g in blocks.groupby("block_id"):
        e = g[g.modality == "eeg"].path.tolist()
        md = g[g.modality == "mdata"].path.tolist()
        mpath = md[0] if md else None
        mrej = None
        if mpath and Path(mpath).name in M.MOTION_BLACKLIST:
            mrej, mpath = "authors_blacklist", None
        elif mpath and bid not in qc_sel:
            mrej, mpath = "block_absent_from_authors_qc_sheet", None
        elif mpath and qc_sel.get(bid) is None:
            mrej, mpath = "no_trials_selected_in_authors_qc_sheet", None
        jobs.append((bid, e[0] if e else None, mpath, qc_sel.get(bid), mrej, bool(md)))

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_block, bid, e, mp, sel, n_frames, sfreq, band): (bid, mrej, has_md)
                for bid, e, mp, sel, mrej, has_md in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            bid, mrej, has_md = futs[f]
            r = f.result()
            r["align"]["motion_file_present"] = has_md
            r["align"]["motion_block_rejected"] = mrej
            results.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} blocks ({time.time() - t0:.0f}s)", flush=True)

    rows_all, eX, eF, eI, mX, mF, mI = [], [], [], [], [], [], []
    for r in sorted(results, key=lambda r: r["block_id"]):
        b = blocks[blocks.block_id == r["block_id"]].iloc[0]
        meta = {"participant": b.participant, "label": M.pid_label(b.participant), "task": b.task,
                "block": int(b.block), "block_id": r["block_id"]}
        al = r["align"]
        for mod, lst, X, F, I in (("eeg", r["eeg"], eX, eF, eI), ("motion", r["motion"], mX, mF, mI)):
            for t in lst:
                row = {**meta, "modality": mod, **{k: v for k, v in t.items() if k not in ("data", "feat")}}
                if mod == "motion":  # index of the matching EEG trial in the same block (if verified)
                    row["eeg_trial_idx"] = (t["trial_idx"] + al["align_shift"]) if al.get("trial_level_aligned") else None
                rows_all.append(row)
                if t.get("data") is not None:
                    X.append(t["data"]); F.append(t["feat"]); I.append(row)
        if r["align"].get("motion_block_rejected"):
            rows_all.append({**meta, "modality": "motion", "qc": r["align"]["motion_block_rejected"]})

    np.save(out / "eeg_X.npy", np.stack(eX)); np.save(out / "eeg_F.npy", np.stack(eF))
    np.save(out / "motion_X.npy", np.stack(mX)); np.save(out / "motion_F.npy", np.stack(mF))
    pd.DataFrame(eI).to_csv(out / "eeg_index.csv", index=False)
    pd.DataFrame(mI).to_csv(out / "motion_index.csv", index=False)
    pd.DataFrame(rows_all).to_csv(out / "trials_all.csv", index=False)
    al = pd.DataFrame([r["align"] for r in results]).sort_values("block_id")
    al.to_csv(out / "alignment.csv", index=False)
    save_json({"eeg_feature_names": M.eeg_feature_names(), "motion_feature_names": M.motion_feature_names(),
               "joints": M.JOINTS, "edges": M.EDGES, "eeg_channels": M.COMMON_CH, "eeg_sfreq_tensor": sfreq / 2,
               "motion_frames": n_frames}, out / "meta.json")
    print(f"done in {time.time() - t0:.0f}s: EEG trials {len(eX)}, motion trials {len(mX)}")


if __name__ == "__main__":
    main()
