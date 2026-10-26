"""Experiment A — Move4AS ASD vs non-ASD (participant-level 5-fold CV), SMOKE_TEST by default.

Primary comparison on the PAIRED trial set (motion trial + its verified aligned EEG trial, both QC-ok):
  feature family : motion_feat_lr | eeg_feat_lr | early_fusion_lr | late_fusion_lr
  deep family    : motion_ctrgcn  | eeg_cnn     | intermediate_fusion | late_fusion_deep
  extra skeleton : motion_poseformerv2 | motion_mae (MAE pre-trained on training participants only)
Supplementary unimodal on all QC-ok trials: motion_feat_lr_all (33 participants), eeg_feat_lr_all (34).

Every model uses the SAME participant fold assignment (outputs/metrics/A_move4as/fold_assignments.csv).
Thresholds and late-fusion weights are chosen on training-fold data only (inner CV / inner validation).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.evaluation import plots  # noqa: E402
from src.evaluation.report import evaluate_binary, summary_table  # noqa: E402
from src.fusion.fusion import EarlyFusion, IntermediateFusion, LateFusion, unimodal_classifier  # noqa: E402
from src.models.ctrgcn import CTRGCN  # noqa: E402
from src.models.eeg_cnn import EEGNetLite  # noqa: E402
from src.models.mae import MAEClassifier, SkeletonMAE  # noqa: E402
from src.models.poseformer import PoseFormerV2Classifier  # noqa: E402
from src.training.cv import youden_threshold  # noqa: E402
from src.training.deep_cv import participant_mean, run_deep_binary_cv, threshold_from_val  # noqa: E402
from src.training.mae_pretrain import pretrain_mae  # noqa: E402
from src.utils.config import ensure_dirs, get_logger, load_config, save_json, set_seed  # noqa: E402

EXP = "A_move4as"


def participant_fold_map(index_frames, n_splits, seed):
    parts = pd.concat([f[["participant", "label"]] for f in index_frames]).drop_duplicates("participant")
    parts = parts.sort_values("participant").reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fmap = {}
    for k, (_, te) in enumerate(cv.split(parts, parts["label"], parts["participant"])):
        for p in parts.loc[te, "participant"]:
            fmap[p] = k
    parts["fold"] = parts["participant"].map(fmap)
    return parts


def trial_folds(groups, fmap, n_splits):
    f = np.array([fmap[g] for g in groups])
    return [(k, np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def inner_folds(groups, y, seed, n=3):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(groups, y, groups))


def sk_cv(name, fit_predict, y, groups, folds, seed, meta=None):
    """fit_predict(train_idx, test_idx) -> probs. Returns OOF frame and inner-OOF store (for late fusion)."""
    rows, inner_store = [], {}
    for k, tr, te in folds:
        p_in = np.zeros(len(tr))
        for a, b in inner_folds(groups[tr], y[tr], seed + k):
            p_in[b] = fit_predict(tr[a], tr[b])
        th = threshold_from_val(groups[tr], y[tr], p_in)
        p = fit_predict(tr, te)
        inner_store[k] = (tr, p_in)
        df = pd.DataFrame({"participant": groups[te], "fold": k, "y_true": y[te], "y_prob": p, "threshold": th})
        if meta is not None:
            df = pd.concat([df, meta.iloc[te].reset_index(drop=True)], axis=1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True), inner_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="subset of model names")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = get_logger("move4as", cfg["paths"]["logs"]).info
    E = cfg["experiment"]
    seed, K = E["random_seed"], E["n_splits"]
    set_seed(seed)
    d = Path(cfg["paths"]["processed"]) / "move4as"
    meta = json.loads((d / "meta.json").read_text())
    eX, eF = np.load(d / "eeg_X.npy"), np.load(d / "eeg_F.npy")
    mX, mF = np.load(d / "motion_X.npy"), np.load(d / "motion_F.npy")
    eI, mI = pd.read_csv(d / "eeg_index.csv"), pd.read_csv(d / "motion_index.csv")
    eok = (eI["qc"] == "ok").to_numpy()

    parts = participant_fold_map([eI, mI], K, seed)
    mdir = Path(cfg["paths"]["metrics"]) / EXP
    mdir.mkdir(parents=True, exist_ok=True)
    parts.to_csv(mdir / "fold_assignments.csv", index=False)
    fmap = dict(zip(parts.participant, parts.fold))
    log(f"participants={len(parts)} ASD={int(parts.label.sum())} folds:\n"
        + parts.groupby("fold").label.agg(["count", "sum"]).to_string())

    # ---------------- paired set (verified trial-level alignment + both modalities QC ok)
    mI["_im"] = np.arange(len(mI))
    eI["_ie"] = np.arange(len(eI))
    pair = mI.dropna(subset=["eeg_trial_idx"]).astype({"eeg_trial_idx": int}).merge(
        eI.loc[eok, ["block_id", "trial_idx", "cue", "_ie"]].rename(columns={"trial_idx": "eeg_trial_idx", "cue": "eeg_cue"}),
        on=["block_id", "eeg_trial_idx"])
    assert (pair["cue"] == pair["eeg_cue"]).all(), "aligned trials must share the cue"
    im, ie = pair["_im"].to_numpy(), pair["_ie"].to_numpy()
    y, g = pair["label"].to_numpy().astype(int), pair["participant"].to_numpy()
    PFm, PFe, PXm, PXe = mF[im], eF[ie], mX[im], eX[ie]
    pmeta = pair[["block_id", "task", "trial_idx", "cue"]].reset_index(drop=True)
    folds = trial_folds(g, fmap, K)
    log(f"paired trials={len(pair)} participants={len(np.unique(g))} (ASD {len(np.unique(g[y == 1]))})")
    save_json({"n_paired_trials": len(pair), "participants": int(len(np.unique(g))),
               "asd_participants": int(len(np.unique(g[y == 1]))),
               "trials_per_participant": pair.groupby("participant").size().describe().to_dict(),
               "by_task": pair.groupby("task").size().to_dict()}, mdir / "paired_set_summary.json")

    want = lambda n: args.only is None or n in args.only
    results = []

    def done(name, oof, extra=None, hist=None):
        if hist is not None and len(hist):
            hist.to_csv(mdir / f"{name}_history.csv", index=False)
            plots.plot_training_curves(hist, f"{name} — {cfg['run_label']}",
                                       Path(cfg["paths"]["plots"]) / "training_curves" / f"{EXP}__{name}.png")
        r = evaluate_binary(name, oof, cfg, EXP, extra={"run_label": cfg["run_label"], **(extra or {})})
        pp = r["pooled_participant"]
        log(f"==> {name}: participant AUC={pp['roc_auc']} BA={pp['balanced_accuracy']} MCC={pp['mcc']}")
        results.append(r)
        return r

    # ---------------- feature family
    def lr_on(F):
        def fp(tr, te):
            return unimodal_classifier(seed=seed).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        return fp

    oof_m, in_m = sk_cv("motion_feat_lr", lr_on(PFm), y, g, folds, seed, pmeta)
    oof_e, in_e = sk_cv("eeg_feat_lr", lr_on(PFe), y, g, folds, seed, pmeta)
    if want("motion_feat_lr"):
        done("motion_feat_lr", oof_m, {"family": "features", "modality": "motion", "set": "paired"})
    if want("eeg_feat_lr"):
        done("eeg_feat_lr", oof_e, {"family": "features", "modality": "eeg", "set": "paired"})
    if want("early_fusion_lr"):
        def fp_early(tr, te):
            return EarlyFusion(seed=seed).fit([PFm[tr], PFe[tr]], y[tr]).predict_proba([PFm[te], PFe[te]])
        oof, _ = sk_cv("early_fusion_lr", fp_early, y, g, folds, seed, pmeta)
        done("early_fusion_lr", oof, {"family": "features", "fusion": "early", "set": "paired"})
    if want("late_fusion_lr"):
        rows, weights = [], {}
        for k, tr, te in folds:
            tr_m, p_in_m = in_m[k]
            _, p_in_e = in_e[k]
            lf = LateFusion().fit([p_in_m, p_in_e], y[tr_m])
            th = threshold_from_val(g[tr_m], y[tr_m], lf.predict_proba([p_in_m, p_in_e]))
            pm = oof_m.loc[oof_m.fold == k, "y_prob"].to_numpy()
            pe = oof_e.loc[oof_e.fold == k, "y_prob"].to_numpy()
            weights[k] = lf.w
            rows.append(oof_m.loc[oof_m.fold == k].assign(y_prob=lf.predict_proba([pm, pe]), threshold=th))
        done("late_fusion_lr", pd.concat(rows, ignore_index=True),
             {"family": "features", "fusion": "late", "set": "paired", "late_weight_motion_by_fold": weights})

    # ---------------- deep family
    V, edges = len(meta["joints"]), [tuple(e) for e in meta["edges"]]
    n_ch, n_s = PXe.shape[1], PXe.shape[2]
    mk_ctr = lambda nc=1: CTRGCN(nc, V, edges, in_channels=3)
    mk_eeg = lambda nc=1: EEGNetLite(n_ch, nc)
    deep = {}
    for name, build, inputs, kw in [
        ("motion_ctrgcn", mk_ctr, [PXm], {}),
        ("eeg_cnn", mk_eeg, [PXe], {}),
        ("intermediate_fusion", lambda: IntermediateFusion({"motion": mk_ctr(), "eeg": mk_eeg()}), [PXm, PXe], {}),
        ("motion_poseformerv2", lambda: PoseFormerV2Classifier(1, V, 3), [PXm], {}),
    ]:
        if not want(name) and not (name in ("motion_ctrgcn", "eeg_cnn") and want("late_fusion_deep")):
            continue
        log(f"--- {name}")
        oof, hist, info, vs = run_deep_binary_cv(name, build, inputs, y, g, folds, cfg, EXP, log=log, meta=pmeta)
        deep[name] = (oof, vs)
        extra = {"family": "deep", "set": "paired", **info}
        if name == "intermediate_fusion":
            extra["fusion"] = "intermediate"
        done(name, oof, extra, hist)

    if want("motion_mae"):
        mc = cfg["mae"]
        recon_rows, mae_hist = [], []

        def pretrain(model, tr_inputs, va_inputs, k):
            mae, h, rm = pretrain_mae(model.mae, tr_inputs[0], va_inputs[0], epochs=E["epochs"],
                                      batch_size=max(E["batch_size"], 32), lr=mc["lr"],
                                      weight_decay=cfg["training"]["weight_decay"], seed=seed + k,
                                      amp=cfg["training"]["mixed_precision"],
                                      ckpt=Path(cfg["paths"]["checkpoints"]) / EXP / f"motion_mae_pretrain_fold{k}.pt", log=log)
            mae_hist.append(h.assign(fold=k))
            recon_rows.append({"fold": k, **{kk: vv for kk, vv in rm.items() if kk != "per_joint_mae"},
                               **{f"joint_mae_{j}": v for j, v in zip(meta["joints"], rm.get("per_joint_mae", []))}})
            model.mae = mae
            return model

        build = lambda: MAEClassifier(SkeletonMAE(V, 3, PXm.shape[1], patch_len=4, embed_dim=mc["embed_dim"],
                                                  depth=mc["depth"], heads=mc["heads"], mask_ratio=mc["mask_ratio"]), 1)
        log("--- motion_mae")
        oof, hist, info, _ = run_deep_binary_cv("motion_mae", build, [PXm], y, g, folds, cfg, EXP, log=log,
                                                pretrain=pretrain, meta=pmeta)
        mh = pd.concat(mae_hist)
        mh.to_csv(mdir / "motion_mae_pretrain_history.csv", index=False)
        plots.plot_training_curves(mh, "Move4AS skeleton MAE reconstruction loss",
                                   Path(cfg["paths"]["plots"]) / "training_curves" / f"{EXP}__motion_mae_pretrain.png")
        pd.DataFrame(recon_rows).to_csv(mdir / "motion_mae_reconstruction_metrics.csv", index=False)
        done("motion_mae", oof, {"family": "deep", "set": "paired", **info}, hist)

    if want("late_fusion_deep") and "motion_ctrgcn" in deep and "eeg_cnn" in deep:
        (om, vm), (oe, ve) = deep["motion_ctrgcn"], deep["eeg_cnn"]
        rows, weights = [], {}
        for k in range(K):
            gv, yv, pvm = vm[k]
            gv2, _, pve = ve[k]
            assert (gv == gv2).all(), "inner validation sets must coincide"
            lf = LateFusion().fit([pvm, pve], yv)
            th = threshold_from_val(gv, yv, lf.predict_proba([pvm, pve]))
            a, b = om[om.fold == k], oe[oe.fold == k]
            weights[k] = lf.w
            rows.append(a.assign(y_prob=lf.predict_proba([a.y_prob.to_numpy(), b.y_prob.to_numpy()]), threshold=th))
        done("late_fusion_deep", pd.concat(rows, ignore_index=True),
             {"family": "deep", "fusion": "late", "set": "paired", "late_weight_motion_by_fold": weights})

    # ---------------- supplementary: unimodal on all QC-ok trials
    for name, F, I in (("motion_feat_lr_all", mF, mI), ("eeg_feat_lr_all", eF[eok], eI[eok].reset_index(drop=True))):
        if not want(name):
            continue
        yy, gg = I["label"].to_numpy().astype(int), I["participant"].to_numpy()

        def fp(tr, te, F=F, yy=yy):
            return unimodal_classifier(seed=seed).fit(F[tr], yy[tr]).predict_proba(F[te])[:, 1]

        oof, _ = sk_cv(name, fp, yy, gg, trial_folds(gg, fmap, K), seed)
        done(name, oof, {"family": "features", "set": "all_qc_ok_trials"})

    # ---------------- comparison tables & plots (all models saved in this experiment, incl. earlier partial runs)
    results = [json.loads(f.read_text()) for f in sorted(mdir.glob("*_metrics.json"))]
    summ = summary_table(results)
    summ.to_csv(mdir / "summary_participant_level.csv", index=False)
    summary_table(results, "trial").to_csv(mdir / "summary_trial_level.csv", index=False)
    fold_df = pd.concat([pd.read_csv(mdir / f"{r['model']}_folds.csv") for r in results])
    fold_df.to_csv(mdir / "all_folds.csv", index=False)
    fp_ = fold_df[fold_df.level == "participant"]
    P = Path(cfg["paths"]["plots"]) / "comparisons"
    s = summ.set_index("model")
    plots.plot_model_comparison(s, ["roc_auc", "balanced_accuracy", "f1", "mcc"],
                                f"Move4AS ASD vs non-ASD — participant level ({cfg['run_label']})", P / f"{EXP}_models.png")
    for fam, names in {"features": ["motion_feat_lr", "eeg_feat_lr", "early_fusion_lr", "late_fusion_lr"],
                       "deep": ["motion_ctrgcn", "eeg_cnn", "intermediate_fusion", "late_fusion_deep"]}.items():
        sub = [n for n in names if n in s.index]
        if sub:
            plots.plot_model_comparison(s.loc[sub], ["roc_auc", "balanced_accuracy", "mcc"],
                                        f"Fusion comparison ({fam}) — {cfg['run_label']}", P / f"{EXP}_fusion_{fam}.png")
            plots.plot_foldwise(fp_[fp_.model.isin(sub)], "roc_auc", f"Fold-wise AUC ({fam})",
                                P / f"{EXP}_foldwise_auc_{fam}.png")
    log("\n" + summ[["model", "roc_auc_mean", "roc_auc_std", "pooled_roc_auc", "balanced_accuracy_mean",
                     "mcc_mean"]].to_string())


if __name__ == "__main__":
    main()
