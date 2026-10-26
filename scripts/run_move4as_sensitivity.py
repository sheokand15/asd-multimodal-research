"""Experiment A — sensitivity analyses for the Move4AS feature models (confound checks).

Why: the primary motion-feature model separates groups almost perfectly. Two acquisition-related
confounds were found in the audit:
  * task mix differs by group (controls: 76 % dance trials; autistic: 59 %),
  * 21 control blocks (S6, S8, S9, S10) were recorded at 40 fps instead of 60 fps,
and the most discriminative features are acceleration/jerk, which are sensitive to tracking jitter
and to motion-capture smoothing settings.  These analyses test how much of the signal survives.

Analyses (same participant folds as the primary run; paired trial set):
  S1 task-balanced participant score (mean of per-task mean probabilities)
  S2 walk-only / dance-only models
  S3 excluding the four 40-fps control participants
  S4 motion features recomputed after a 6 Hz low-pass filter (removes jitter > 6 Hz)
  S5 motion features without acceleration/jerk (speed, range of motion, posture variability only)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json  # noqa: E402

from scripts.run_move4as import participant_fold_map, sk_cv, trial_folds  # noqa: E402
from src.data import move4as as M  # noqa: E402
from src.evaluation import plots  # noqa: E402
from src.evaluation.report import evaluate_binary, summary_table  # noqa: E402
from src.fusion.fusion import EarlyFusion, unimodal_classifier  # noqa: E402
from src.utils.config import ensure_dirs, get_logger, load_config, set_seed  # noqa: E402

EXP = "A_move4as_sensitivity"


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    ensure_dirs(cfg)
    log = get_logger("move4as_sens", cfg["paths"]["logs"]).info
    seed, K = cfg["experiment"]["random_seed"], cfg["experiment"]["n_splits"]
    set_seed(seed)
    d = Path(cfg["paths"]["processed"]) / "move4as"
    meta = json.loads((d / "meta.json").read_text())
    eF, mF, mX = np.load(d / "eeg_F.npy"), np.load(d / "motion_F.npy"), np.load(d / "motion_X.npy")
    eI, mI = pd.read_csv(d / "eeg_index.csv"), pd.read_csv(d / "motion_index.csv")
    al = pd.read_csv(d / "alignment.csv")
    parts = participant_fold_map([eI, mI], K, seed)
    fmap = dict(zip(parts.participant, parts.fold))
    eok = (eI.qc == "ok").to_numpy()
    mI["_im"], eI["_ie"] = np.arange(len(mI)), np.arange(len(eI))
    pair = mI.dropna(subset=["eeg_trial_idx"]).astype({"eeg_trial_idx": int}).merge(
        eI.loc[eok, ["block_id", "trial_idx", "_ie"]].rename(columns={"trial_idx": "eeg_trial_idx"}),
        on=["block_id", "eeg_trial_idx"]).merge(al[["block_id", "motion_fps"]], on="block_id")
    im, ie = pair._im.to_numpy(), pair._ie.to_numpy()
    PFm, PFe = mF[im], eF[ie]
    # S4: low-pass motion features. Trials are resampled to 128 frames / 4 s = 32 Hz.
    fs = (meta["motion_frames"] - 1) / M.TRIAL_S
    sos = signal.butter(4, 6.0, fs=fs, output="sos")
    PFm_lp = np.stack([M.motion_features(signal.sosfiltfilt(sos, x, axis=0).astype(np.float32), 1 / fs) for x in mX[im]])
    names = meta["motion_feature_names"]
    nojerk = np.array([not (n.startswith("mot_acc") or n.startswith("mot_logjerk")) for n in names])

    results = []

    def run(name, F, mask=None, extra=None, F2=None):
        sel = np.ones(len(pair), bool) if mask is None else mask
        y = pair.label.to_numpy()[sel].astype(int)
        g = pair.participant.to_numpy()[sel]
        task = pair.task.to_numpy()[sel]
        Fs = F[sel]
        F2s = None if F2 is None else F2[sel]

        def fp(tr, te):
            if F2s is None:
                return unimodal_classifier(seed=seed).fit(Fs[tr], y[tr]).predict_proba(Fs[te])[:, 1]
            return EarlyFusion(seed=seed).fit([Fs[tr], F2s[tr]], y[tr]).predict_proba([Fs[te], F2s[te]])

        oof, _ = sk_cv(name, fp, y, g, trial_folds(g, fmap, K), seed, pd.DataFrame({"task": task}))
        r = evaluate_binary(name, oof, cfg, EXP, extra={"analysis": extra, "n_participants": int(len(np.unique(g)))})
        log(f"{name}: AUC={r['pooled_participant']['roc_auc']} BA={r['pooled_participant']['balanced_accuracy']}")
        results.append(r)
        return oof

    oof = run("motion_feat_lr_reference", PFm, extra="reference = primary model")
    # S1 task-balanced participant score from the reference predictions
    tb = (oof.groupby(["participant", "task"]).y_prob.mean().groupby("participant").mean())
    first = oof.groupby("participant").first()
    tb_df = pd.DataFrame({"participant": tb.index, "fold": first.loc[tb.index, "fold"].values,
                          "y_true": first.loc[tb.index, "y_true"].values, "y_prob": tb.values,
                          "threshold": first.loc[tb.index, "threshold"].values})
    r = evaluate_binary("motion_feat_lr_task_balanced", tb_df, cfg, EXP, trial_level=False,
                        extra={"analysis": "S1 participant score = mean of per-task means"})
    results.append(r)
    for task in ("walk", "dance"):
        m = (pair.task == task).to_numpy()
        run(f"motion_feat_lr_{task}_only", PFm, m, f"S2 {task} trials only")
        run(f"eeg_feat_lr_{task}_only", PFe, m, f"S2 {task} trials only")
        run(f"early_fusion_lr_{task}_only", PFm, m, f"S2 {task} trials only", F2=PFe)
    m60 = ~pair.participant.isin(["S6", "S8", "S9", "S10"]).to_numpy()
    run("motion_feat_lr_excl_40fps", PFm, m60, "S3 exclude 40-fps control participants")
    run("eeg_feat_lr_excl_40fps", PFe, m60, "S3 exclude 40-fps control participants")
    run("motion_feat_lr_lowpass6hz", PFm_lp, None, "S4 6 Hz low-pass before feature extraction")
    run("motion_feat_lr_no_acc_jerk", PFm[:, nojerk], None, "S5 drop acceleration/jerk features")
    run("motion_feat_lr_lowpass6hz_no_acc_jerk_excl_40fps", PFm_lp[:, nojerk], m60, "S3+S4+S5 combined")

    # S6 demographic confound: can age + sex alone separate the groups? (participant-level LR, same folds)
    demo = pd.read_excel(cfg["paths"]["move4as"] / "Participants.xlsx", header=None, dtype=str).iloc[:, :3]
    demo.columns = ["participant", "age", "gender"]
    demo = demo[demo.participant.fillna("").str.match(r"^[SP]\d+$")].copy()
    demo["age"] = pd.to_numeric(demo.age, errors="coerce")
    demo["female"] = (demo.gender.str.strip() == "Female").astype(float)
    demo["label"] = demo.participant.str.startswith("P").astype(int)
    demo = demo[demo.participant.isin(fmap)].reset_index(drop=True)
    Fd, yd, gd = demo[["age", "female"]].to_numpy(), demo.label.to_numpy(), demo.participant.to_numpy()

    def fp_demo(tr, te):
        return unimodal_classifier(C=1.0, seed=seed).fit(Fd[tr], yd[tr]).predict_proba(Fd[te])[:, 1]

    oof, _ = sk_cv("demographics_age_sex_lr", fp_demo, yd, gd, trial_folds(gd, fmap, K), seed)
    r = evaluate_binary("demographics_age_sex_lr", oof, cfg, EXP, trial_level=False,
                        extra={"analysis": "S6 confound check: age + sex only (no movement data)"})
    log(f"demographics_age_sex_lr: AUC={r['pooled_participant']['roc_auc']}")
    results.append(r)

    summ = summary_table(results)
    mdir = Path(cfg["paths"]["metrics"]) / EXP
    summ.to_csv(mdir / "summary_participant_level.csv", index=False)
    plots.plot_model_comparison(summ.set_index("model"), ["roc_auc", "balanced_accuracy", "mcc"],
                                "Move4AS sensitivity analyses (participant level)",
                                Path(cfg["paths"]["plots"]) / "comparisons" / f"{EXP}.png")
    log("\n" + summ[["model", "n", "pooled_roc_auc", "roc_auc_mean", "roc_auc_std", "balanced_accuracy_mean"]].to_string())


if __name__ == "__main__":
    main()
