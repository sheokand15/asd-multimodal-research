"""Publication-style model-comparison figures (JPG) from saved out-of-fold predictions.

Every bar = pooled out-of-fold value on held-out participants; whisker = 95 % participant-level bootstrap CI
(participants resampled with replacement, 1,000 resamples, seed from config).  Nothing is typed by hand.
Each figure is written with a CSV of its plotted values (table view).

Usage:  .venv\\Scripts\\python scripts\\make_comparison_figures.py [--config configs\\full_training.yaml]
Output: <plots>/model_comparison/*.jpg + *.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.utils.config import load_config  # noqa: E402

# reference palette (validated light-mode slots 1-3) + text/surface tokens
SURFACE, TEXT, TEXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
FOOT = "bar: pooled out-of-fold value · whisker: 95% participant-bootstrap CI"
N_BOOT = 1000


def style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
                         "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT,
                         "axes.edgecolor": GRID, "axes.linewidth": 1.0})


# ------------------------------------------------------------------ metrics + participant bootstrap

def bal_acc(y, p, k):
    rec = [np.mean(p[y == c] == c) for c in range(k) if np.any(y == c)]
    return float(np.mean(rec))


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def boot(fn, groups, seed, *arrays):
    """Value on all data + 95 % CI from resampling participants."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rows = {g_: np.where(groups == g_)[0] for g_ in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rows[g_] for g_ in rng.choice(uniq, len(uniq))])
        v = fn(*(a[idx] for a in arrays))
        if np.isfinite(v):
            vals.append(v)
    return fn(*arrays), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------ figures

def heading(fig, ax, title, subtitle):
    """Short left-aligned title + muted subtitle, anchored to the axes' left edge (never clipped)."""
    x0 = ax.get_position().x0
    fig.text(x0, 0.985, title, ha="left", va="top", fontsize=14, color=TEXT)
    if subtitle:
        fig.text(x0, 0.985 - 0.36 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=TEXT2)


def hbar(df, title, subtitle, chance, chance_label, path, xmax=1.0):
    """df: label, value, lo, hi  (top = weakest, bottom = best, as in the reference figures)."""
    df = df.sort_values("value", ascending=True).reset_index(drop=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(9.75, 1.35 + 0.72 * n))
    yy = np.arange(n)
    ax.barh(yy, df.value, height=0.55, color=SERIES[0], zorder=2)
    err = np.vstack([df.value - df.lo, df.hi - df.value]).clip(min=0)
    has = (df.hi - df.lo) > 1e-9
    ax.errorbar(df.value[has], yy[has], xerr=err[:, has.values], fmt="none", ecolor="#3a3936", elinewidth=1.2,
                capsize=4, capthick=1.2, zorder=3)
    for i, r in df.iterrows():
        ax.text(r.value + 0.008, i - 0.38, f"{r.value:.3f}", fontsize=10, color=TEXT, va="bottom", ha="left")
    if chance is not None:
        ax.axvline(chance, color="#3a3936", ls=(0, (4, 3)), lw=1.2, zorder=4)
        ax.text(chance + 0.01, n - 0.55, chance_label, fontsize=9.5, color=TEXT2, va="top")
    ax.set_yticks(yy, df.label, fontsize=10.5)
    ax.set_ylim(n - 0.5 + (0.35 if chance is not None else 0.1), -0.7)
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.linspace(0, xmax, 6))
    ax.grid(axis="x", color=GRID, lw=1.0, zorder=0)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(FOOT, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    heading(fig, ax, title, subtitle)
    save(fig, df, path)


def grouped(per_class: pd.DataFrame, series: list[str], title, subtitle, path):
    """per_class: index = class names, columns = series labels (fixed order = slot order)."""
    n, m = len(per_class), len(series)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    w = 0.8 / m
    x = np.arange(n)
    for j, s in enumerate(series):
        ax.bar(x + (j - (m - 1) / 2) * w, per_class[s], width=w, color=SERIES[j], label=s, zorder=2,
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x, per_class.index, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.16)  # headroom so the legend never sits on a bar
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color=GRID, lw=1.0, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper left", ncol=m, frameon=True, facecolor=SURFACE, edgecolor=SURFACE, fontsize=10,
              labelcolor=TEXT2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    heading(fig, ax, title, subtitle)
    save(fig, per_class.reset_index(), path)


def save(fig, table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, pil_kwargs={"quality": 95})
    plt.close(fig)
    table.to_csv(path.with_suffix(".csv"), index=False)
    print("wrote", path)


# ------------------------------------------------------------------ data loaders

def mmasd(cfg, out, seed):
    P = Path(cfg["paths"]["predictions"])
    names = {"majority_class": "Majority class",
             "kinematic3d_lightgbm": "3D kinematic features + gradient boosting",
             "kinematic3d_logreg": "3D kinematic features + logistic regression",
             "ctrgcn__2d_openpose": "CTR-GCN · 2D OpenPose", "ctrgcn__3d_romp": "CTR-GCN · 3D ROMP",
             "poseformerv2__2d_openpose": "PoseFormerV2 · 2D OpenPose",
             "poseformerv2__3d_romp": "PoseFormerV2 · 3D ROMP",
             "mae__2d_openpose": "MAE + fine-tune · 2D OpenPose", "mae__3d_romp": "MAE + fine-tune · 3D ROMP"}
    rows, preds = [], {}
    for key, label in names.items():
        f = P / f"B_mmasd__{key}_oof.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={"participant": str})
        classes = [c[2:] for c in d.columns if c.startswith("p_")]
        k = len(classes)
        y, p, g = d.y_true.to_numpy(), d.y_pred.to_numpy(), d.participant.to_numpy()
        preds[label] = (y, p, classes)
        ba = boot(lambda a, b: bal_acc(a, b, k), g, seed, y, p)
        f1 = boot(lambda a, b: macro_f1(a, b, k), g, seed, y, p)
        rows.append({"label": label, "ba": ba, "f1": f1})
    k = len(classes)
    sub = f"MMASD activity recognition · {k} classes · {len(np.unique(g))} children · participant-grouped 5-fold CV"
    for met, title, fname in (("ba", "Balanced accuracy, held-out participants", "mmasd_balanced_accuracy"),
                              ("f1", "F1 macro, held-out participants", "mmasd_f1_macro")):
        df = pd.DataFrame([{"label": r["label"], "value": r[met][0], "lo": r[met][1], "hi": r[met][2]} for r in rows])
        hbar(df, title, sub, 1 / k if met == "ba" else None, f"chance {1 / k:.3f}", out / f"{fname}.jpg")
    # per-activity F1 of the three best models by macro F1
    best = sorted(rows, key=lambda r: -r["f1"][0])[:3]
    per = {}
    for r in best:
        y, p, classes = preds[r["label"]]
        per[r["label"]] = [(2 * np.sum((p == c) & (y == c)) / max(np.sum(p == c) + np.sum(y == c), 1))
                           for c in range(k)]
    tab = pd.DataFrame(per, index=[c.rsplit("_", 1)[0] for c in classes])
    tab.index.name = "activity"
    grouped(tab, [r["label"] for r in best], "Per-activity F1 on held-out participants",
            "MMASD · three best models by macro F1 · pooled out-of-fold predictions", out / "mmasd_per_activity_f1.jpg")


def participant_level(d):
    g = d.groupby("participant")
    return pd.DataFrame({"participant": list(g.groups), "y": g.y_true.first().values, "p": g.y_prob.mean().values,
                         "th": g.threshold.first().values, "fold": g.fold.first().values})


def binary_rows(files: dict, seed, fold_file: Path | None = None):
    rows = []
    for label, f in files.items():
        if f == "MAJORITY":
            ref = participant_level(pd.read_csv(next(iter(v for v in files.values() if v != "MAJORITY"))))
            y, fold = ref.y.to_numpy(), ref.fold.to_numpy()
            pred = np.zeros_like(y)
            for k in np.unique(fold):  # majority label of each fold's training participants
                pred[fold == k] = np.bincount(y[fold != k], minlength=2).argmax()
            s = pred.astype(float)
            g = ref.participant.to_numpy()
        else:
            d = participant_level(pd.read_csv(f))
            y, s, g = d.y.to_numpy(), d.p.to_numpy(), d.participant.to_numpy()
            pred = (s >= d.th.to_numpy()).astype(int)
        ba = boot(lambda a, b: bal_acc(a, b, 2), g, seed, y, pred)
        au = boot(auc, g, seed, y, s) if len(np.unique(s)) > 1 else (0.5, 0.5, 0.5)
        rows.append({"label": label, "ba": ba, "auc": au, "n": len(y)})
    return rows


def binary_figs(rows, prefix, subtitle, out):
    for met, title in (("ba", "Balanced accuracy"), ("auc", "ROC-AUC")):
        df = pd.DataFrame([{"label": r["label"], "value": r[met][0], "lo": r[met][1], "hi": r[met][2], "n": r["n"]}
                           for r in rows])
        hbar(df, f"{title}, held-out participants", subtitle, 0.5, "chance 0.5",
             out / f"{prefix}_{'balanced_accuracy' if met == 'ba' else 'roc_auc'}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["experiment"]["random_seed"]
    style()
    out = Path(cfg["paths"]["plots"]) / "model_comparison"
    P = Path(cfg["paths"]["predictions"])

    mmasd(cfg, out, seed)

    A = P / "A_move4as"
    fusion = {"Majority class": "MAJORITY",
              "EEG only · band power + LR": A / "eeg_feat_lr_oof.csv",
              "EEG only · EEGNet-lite CNN": A / "eeg_cnn_oof.csv",
              "Motion only · kinematic features + LR": A / "motion_feat_lr_oof.csv",
              "Motion only · CTR-GCN": A / "motion_ctrgcn_oof.csv",
              "Early fusion · features + LR": A / "early_fusion_lr_oof.csv",
              "Intermediate fusion · CTR-GCN + EEG CNN": A / "intermediate_fusion_oof.csv",
              "Late fusion · feature models": A / "late_fusion_lr_oof.csv",
              "Late fusion · deep models": A / "late_fusion_deep_oof.csv"}
    m4 = "Move4AS ASD vs non-ASD · 33 adults (14 ASD) · paired motion–EEG trials · participant level"
    binary_figs(binary_rows(fusion, seed), "move4as_fusion", m4, out)

    skel = {"Motion · kinematic features + LR": A / "motion_feat_lr_oof.csv",
            "Motion · CTR-GCN": A / "motion_ctrgcn_oof.csv",
            "Motion · PoseFormerV2 adapter": A / "motion_poseformerv2_oof.csv",
            "Motion · MAE + fine-tune": A / "motion_mae_oof.csv"}
    binary_figs(binary_rows(skel, seed), "move4as_motion_encoders",
                "Move4AS ASD vs non-ASD · motion-only encoders · participant level", out)

    S = P / "A_move4as_sensitivity"
    sens = {"Reference (all trials)": S / "motion_feat_lr_reference_oof.csv",
            "Walk trials only": S / "motion_feat_lr_walk_only_oof.csv",
            "Dance trials only": S / "motion_feat_lr_dance_only_oof.csv",
            "Excluding 40-fps controls": S / "motion_feat_lr_excl_40fps_oof.csv",
            "6 Hz low-pass features": S / "motion_feat_lr_lowpass6hz_oof.csv",
            "No acceleration / jerk": S / "motion_feat_lr_no_acc_jerk_oof.csv",
            "Low-pass + no acc/jerk + excl. 40 fps": S / "motion_feat_lr_lowpass6hz_no_acc_jerk_excl_40fps_oof.csv",
            "Age + sex only (no movement)": S / "demographics_age_sex_lr_oof.csv"}
    rows = binary_rows(sens, seed)
    df = pd.DataFrame([{"label": r["label"], "value": r["auc"][0], "lo": r["auc"][1], "hi": r["auc"][2], "n": r["n"]}
                       for r in rows])
    hbar(df, "ROC-AUC, held-out participants",
         "Move4AS confound checks · kinematic features + LR · participant level", 0.5, "chance 0.5",
         out / "move4as_sensitivity_roc_auc.jpg")

    C = P / "C_questionnaire"
    q = {"Majority class": "MAJORITY",
         "Items + LR (reproduces rule)": C / "logreg__items_oof.csv",
         "Items + gradient boosting (reproduces rule)": C / "lightgbm__items_oof.csv",
         "Items + MLP (reproduces rule)": C / "mlp__items_oof.csv",
         "Demographics only + LR": C / "logreg__no_items_oof.csv",
         "Demographics only + gradient boosting": C / "lightgbm__no_items_oof.csv",
         "Demographics only + MLP": C / "mlp__no_items_oof.csv"}
    binary_figs(binary_rows(q, seed), "questionnaire",
                "Q-CHAT-10 toddlers (n = 1,054) · target = screening rule, so item models leak the label", out)


if __name__ == "__main__":
    main()
