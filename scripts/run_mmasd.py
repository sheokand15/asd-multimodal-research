"""Experiment B — MMASD movement analysis (no ASD-vs-non-ASD task: MMASD has no control group).

B1  Activity recognition (supported label), 2D (OpenPose BODY_25) vs 3D (ROMP SMPL-24) on the SAME clips
    and the SAME participant-grouped 5-fold split; models: CTR-GCN, PoseFormerV2 adapter, Skeleton-MAE
    (pre-trained on training-fold participants only, then fine-tuned).
B2  Skeleton-MAE reconstruction metrics (train/val loss, coordinate MAE/MSE, per-joint error).
B3  Clinician-score regression (ADOS-2 comparison score, overall total) from participant-level
    kinematic features, ridge regression with leave-one-participant-out CV + mean-predictor baseline.
B4  Movement-pattern summary across activities (descriptive).
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.evaluation import plots  # noqa: E402
from src.evaluation.metrics import count_params, measure_inference, multiclass_metrics, regression_metrics  # noqa: E402
from src.models.ctrgcn import CTRGCN  # noqa: E402
from src.models.mae import MAEClassifier, SkeletonMAE  # noqa: E402
from src.models.poseformer import PoseFormerV2Classifier  # noqa: E402
from src.training.cv import participant_folds, save_fold_assignments  # noqa: E402
from src.training.mae_pretrain import pretrain_mae  # noqa: E402
from src.training.trainer import device, predict, train_model  # noqa: E402
from src.utils.config import (ensure_dirs, early_stopping_patience, get_logger, load_config, save_json,  # noqa: E402
                              set_seed)

EXP = "B_mmasd"


def plot_cm(cm, classes, title, path):
    cm = np.asarray(cm, float)
    n = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if n[i, j] > .5 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = get_logger("mmasd", cfg["paths"]["logs"]).info
    E, Tr = cfg["experiment"], cfg["training"]
    seed, K = E["random_seed"], E["n_splits"]
    set_seed(seed)
    d = Path(cfg["paths"]["processed"]) / "mmasd"
    meta = json.loads((d / "meta.json").read_text())
    idx = pd.read_csv(d / "index.csv", dtype={"participant": str})
    X3, V3, X2, V2, F3 = (np.load(d / f) for f in ("X3d.npy", "V3d.npy", "X2d.npy", "V2d.npy", "F3d.npy"))
    mdir = Path(cfg["paths"]["metrics"]) / EXP
    mdir.mkdir(parents=True, exist_ok=True)

    # ---------- B1 common clip set: valid 2D and 3D, activity with >= 30 clips
    counts = idx[idx.ok2d].activity.value_counts()
    acts = sorted(counts[counts >= 30].index)
    sel = (idx.ok2d & idx.activity.isin(acts)).to_numpy()
    I = idx[sel].reset_index(drop=True)
    y = I.activity.map({a: i for i, a in enumerate(acts)}).to_numpy()
    g = I.participant.to_numpy()
    reps = {"2d_openpose": (X2[sel], V2[sel], [tuple(e) for e in meta["edges_2d"]]),
            "3d_romp": (X3[sel], V3[sel], [tuple(e) for e in meta["edges_3d"]])}
    folds = list(participant_folds(g, None, K, seed))
    save_fold_assignments(g, None, folds, mdir / "fold_assignments.csv")
    log(f"B1: {len(I)} clips, {len(acts)} activities {acts}, {len(np.unique(g))} participants "
        f"(excluded: {sorted(set(idx.activity) - set(acts))} — <30 clips or no OpenPose)")
    save_json({"activities": acts, "n_clips": len(I), "n_participants": int(len(np.unique(g))),
               "clips_by_activity": I.activity.value_counts().to_dict()}, mdir / "b1_dataset.json")

    rows, eff_rows, mae_recon = [], [], []
    for rep, (X, Vm, edges) in reps.items():
        Vn, C = X.shape[2], X.shape[3]
        builders = {
            "ctrgcn": lambda: CTRGCN(len(acts), Vn, edges, in_channels=C),
            "poseformerv2": lambda: PoseFormerV2Classifier(len(acts), Vn, C),
            "mae": lambda: MAEClassifier(SkeletonMAE(Vn, C, X.shape[1], 4, cfg["mae"]["embed_dim"], cfg["mae"]["depth"],
                                                     cfg["mae"]["heads"], mask_ratio=cfg["mae"]["mask_ratio"]), len(acts)),
        }
        for mname, build in builders.items():
            name = f"{mname}__{rep}"
            log(f"--- {name}")
            oof = np.zeros((len(y), len(acts)))
            fold_rows, hist_rows, infos, mae_hist = [], [], [], []
            for k, trn, te in folds:
                a, b = _group_val(g[trn], Tr["inner_val_fraction"], seed + k)
                itr, iva = trn[a], trn[b]
                model = build()
                if mname == "mae":
                    mae, h, rm = pretrain_mae(model.mae, X[itr], X[iva], epochs=E["epochs"], batch_size=32,
                                              lr=cfg["mae"]["lr"], weight_decay=Tr["weight_decay"], seed=seed + k,
                                              amp=Tr["mixed_precision"], valid_train=Vm[itr], valid_val=Vm[iva],
                                              ckpt=Path(cfg["paths"]["checkpoints"]) / EXP / f"mae_pretrain__{rep}_fold{k}.pt",
                                              log=log)
                    model.mae = mae
                    mae_hist.append(h.assign(fold=k))
                    mae_recon.append({"representation": rep, "fold": k,
                                      **{kk: v for kk, v in rm.items() if kk != "per_joint_mae"},
                                      "per_joint_mae": json.dumps(rm.get("per_joint_mae"))})
                model, hist, info = train_model(model, [X[itr]], y[itr], [X[iva]], y[iva], task="multiclass",
                                                n_class=len(acts), epochs=E["epochs"], batch_size=E["batch_size"],
                                                lr=Tr["learning_rate"], weight_decay=Tr["weight_decay"],
                                                amp=Tr["mixed_precision"], seed=seed + k,
                                                ckpt_path=Path(cfg["paths"]["checkpoints"]) / EXP / f"{name}_fold{k}.pt",
                                                early_stopping_patience=early_stopping_patience(cfg), log=log)
                oof[te] = predict(model, [X[te]], task="multiclass")
                m = multiclass_metrics(y[te], oof[te], len(acts))
                fold_rows.append({"model": name, "fold": k, **{kk: v for kk, v in m.items() if kk not in ("confusion_matrix", "unavailable")}})
                hist_rows += [{**hh, "fold": k} for hh in hist]
                infos.append(info)
            ex = torch.as_tensor(X[: E["batch_size"]], dtype=torch.float32, device=device())
            eff = {"model": name, **count_params(model), **measure_inference(model, (ex,)),
                   "train_time_s": sum(i["train_time_s"] for i in infos),
                   "epoch_time_s": float(np.mean([i["epoch_time_s"] for i in infos])),
                   "peak_gpu_mb": max((i["peak_gpu_mb"] or 0) for i in infos), "checkpoint_mb": infos[0]["checkpoint_mb"]}
            eff_rows.append(eff)
            pooled = multiclass_metrics(y, oof, len(acts))
            fd = pd.DataFrame(fold_rows)
            fd.to_csv(mdir / f"{name}_folds.csv", index=False)
            pd.DataFrame(hist_rows).to_csv(mdir / f"{name}_history.csv", index=False)
            plots.plot_training_curves(pd.DataFrame(hist_rows), f"{name} — {cfg['run_label']}",
                                       Path(cfg["paths"]["plots"]) / "training_curves" / f"{EXP}__{name}.png")
            if mae_hist:
                mh = pd.concat(mae_hist)
                mh.to_csv(mdir / f"mae_pretrain__{rep}_history.csv", index=False)
                plots.plot_training_curves(mh, f"MAE reconstruction loss ({rep})",
                                           Path(cfg["paths"]["plots"]) / "training_curves" / f"{EXP}__mae_pretrain__{rep}.png")
            plot_cm(pooled["confusion_matrix"], acts, f"{name} (pooled OOF)",
                    Path(cfg["paths"]["plots"]) / "confusion_matrices" / f"{EXP}__{name}.png")
            pd.DataFrame({"clip": I["clip"].values, "participant": g, "y_true": y, "y_pred": oof.argmax(1),
                          **{f"p_{a}": oof[:, i] for i, a in enumerate(acts)}}).to_csv(
                Path(cfg["paths"]["predictions"]) / f"{EXP}__{name}_oof.csv", index=False)
            summ = {"model": name, "representation": rep, "architecture": mname,
                    **{f"pooled_{kk}": v for kk, v in pooled.items() if kk not in ("confusion_matrix", "unavailable")},
                    **{f"{c}_mean": fd[c].mean() for c in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")},
                    **{f"{c}_std": fd[c].std() for c in ("accuracy", "balanced_accuracy", "macro_f1", "mcc")}}
            save_json({**summ, "confusion_matrix": pooled["confusion_matrix"], "classes": acts, "efficiency": eff},
                      mdir / f"{name}_metrics.json")
            rows.append(summ)
            log(f"==> {name}: acc={pooled['accuracy']:.3f} bal_acc={pooled['balanced_accuracy']:.3f} macroF1={pooled['macro_f1']:.3f}")

    summ = pd.DataFrame(rows)
    summ.to_csv(mdir / "b1_activity_summary.csv", index=False)
    pd.DataFrame(eff_rows).to_csv(mdir / "b1_efficiency.csv", index=False)
    pd.DataFrame(mae_recon).to_csv(mdir / "b2_mae_reconstruction.csv", index=False)
    s = summ.set_index("model").rename(columns={"accuracy_mean": "accuracy_mean"})
    plots.plot_model_comparison(s, ["accuracy", "balanced_accuracy", "macro_f1"],
                                f"MMASD activity recognition, 2D vs 3D ({len(acts)} classes, chance≈{1/len(acts):.2f})",
                                Path(cfg["paths"]["plots"]) / "comparisons" / f"{EXP}_activity.png")

    # ---------- B3 ADOS regression (participant level)
    ados = pd.read_excel(cfg["paths"]["mmasd"] / "ADOS_rating.xlsx")
    ados = ados[ados["ID#"].astype(str).str.fullmatch(r"\d+")].copy()
    ados["participant"] = ados["ID#"].astype(str)
    tcols = {"ados_comparison_score": [c for c in ados.columns if str(c).startswith("ADOS Comparison")][0],
             "ados_overall_total": [c for c in ados.columns if str(c).strip() == "Overall Total"][0]}
    ok3 = idx.ok3d.to_numpy()
    Fp = pd.DataFrame(F3[ok3]).assign(participant=idx.participant[ok3].values).groupby("participant").mean()
    reg_rows = []
    for tname, col in tcols.items():
        t = pd.to_numeric(ados.set_index("participant")[col], errors="coerce").dropna()
        common = sorted(set(t.index) & set(Fp.index))
        Xp, yp = Fp.loc[common].to_numpy(), t.loc[common].to_numpy(float)
        pred, base = np.zeros(len(yp)), np.zeros(len(yp))
        for i in range(len(yp)):  # leave-one-participant-out; scaler/ridge alpha fit on training participants only
            tr = np.arange(len(yp)) != i
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 20))).fit(Xp[tr], yp[tr])
            pred[i] = mdl.predict(Xp[i:i + 1])[0]
            base[i] = yp[tr].mean()
        for nm, pr in (("ridge_kinematics", pred), ("mean_baseline", base)):
            reg_rows.append({"target": tname, "model": nm, "n_participants": len(yp), **regression_metrics(yp, pr)})
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(yp, pred)
        ax.plot([yp.min(), yp.max()], [yp.min(), yp.max()], "--", color="grey")
        ax.set(xlabel=f"true {tname}", ylabel="LOPO prediction", title=f"MMASD {tname} (n={len(yp)})")
        fig.tight_layout()
        fig.savefig(Path(cfg["paths"]["plots"]) / "comparisons" / f"{EXP}_ados_{tname}.png", dpi=130)
        plt.close(fig)
    pd.DataFrame(reg_rows).to_csv(mdir / "b3_ados_regression.csv", index=False)
    log("\n" + pd.DataFrame(reg_rows)[["target", "model", "n_participants", "mae", "rmse", "r2", "spearman_rho"]].to_string())

    # ---------- B4 movement-pattern summary across activities
    V = X3.shape[2]
    desc = pd.DataFrame({"activity": idx.activity[ok3].values,
                         "mean_speed": F3[ok3][:, :V].mean(1), "speed_variability": F3[ok3][:, V:2 * V].mean(1),
                         "mean_accel": F3[ok3][:, 2 * V:3 * V].mean(1), "range_of_motion": F3[ok3][:, 3 * V:].mean(1)})
    agg = desc.groupby("activity").agg(["mean", "std", "count"])
    agg.to_csv(mdir / "b4_movement_by_activity.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    order = desc.groupby("activity").mean_speed.median().sort_values().index
    ax.boxplot([desc.loc[desc.activity == a, "mean_speed"] for a in order], tick_labels=order, showfliers=False)
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=7)
    ax.set(ylabel="mean joint speed (torso-lengths / frame)", title="MMASD movement intensity by activity (3D, tracked person)")
    fig.tight_layout()
    fig.savefig(Path(cfg["paths"]["plots"]) / "comparisons" / f"{EXP}_movement_by_activity.png", dpi=130)
    plt.close(fig)
    log("\n" + summ[["model", "pooled_accuracy", "pooled_balanced_accuracy", "pooled_macro_f1", "macro_f1_std"]].to_string())


def _group_val(groups, frac, seed):
    """Participant-level train/validation split for multi-class tasks."""
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    val = set(rng.choice(u, size=max(1, int(round(frac * len(u)))), replace=False))
    m = np.array([x in val for x in groups])
    return np.where(~m)[0], np.where(m)[0]


if __name__ == "__main__":
    main()
