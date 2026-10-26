"""Plotting helpers. Every plot is drawn from real out-of-fold predictions saved by the runners."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn import metrics as skm  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563", "#ca8a04"]


def _save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _two_classes(y):
    return len(np.unique(y)) == 2


def plot_roc(y, p, folds, title, path):
    if not _two_classes(y):
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, f in enumerate(sorted(np.unique(folds))):
        m = folds == f
        if _two_classes(y[m]):
            fpr, tpr, _ = skm.roc_curve(y[m], p[m])
            ax.plot(fpr, tpr, lw=0.8, alpha=0.5, color=PALETTE[i % len(PALETTE)], label=f"fold {f} AUC={skm.auc(fpr, tpr):.2f}")
    fpr, tpr, _ = skm.roc_curve(y, p)
    ax.plot(fpr, tpr, lw=2.2, color="black", label=f"pooled OOF AUC={skm.roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title=title)
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path)


def plot_pr(y, p, folds, title, path):
    if not _two_classes(y):
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, f in enumerate(sorted(np.unique(folds))):
        m = folds == f
        if _two_classes(y[m]):
            pr, rc, _ = skm.precision_recall_curve(y[m], p[m])
            ax.plot(rc, pr, lw=0.8, alpha=0.5, color=PALETTE[i % len(PALETTE)],
                    label=f"fold {f} AP={skm.average_precision_score(y[m], p[m]):.2f}")
    pr, rc, _ = skm.precision_recall_curve(y, p)
    ax.plot(rc, pr, lw=2.2, color="black", label=f"pooled OOF AP={skm.average_precision_score(y, p):.3f}")
    ax.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"prevalence={y.mean():.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="lower left")
    _save(fig, path)


def plot_confusion(cm, title, path, normalize=False, labels=("non-ASD", "ASD")):
    cm = np.asarray(cm, float)
    disp = cm / cm.sum(1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else max(1, disp.max()))
    for i in range(2):
        for j in range(2):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > (0.5 if normalize else disp.max() / 2) else "black")
    ax.set_xticks([0, 1], [f"pred {labels[0]}", f"pred {labels[1]}"])
    ax.set_yticks([0, 1], [f"true {labels[0]}", f"true {labels[1]}"])
    ax.set_title(title + (" (row-normalised)" if normalize else ""), fontsize=9)
    _save(fig, path)


def plot_calibration(y, p, title, path, n_bins=5):
    if not _two_classes(y):
        return
    frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.plot(mean, frac, "o-", color=PALETTE[0], label=f"Brier={skm.brier_score_loss(y, p):.3f}")
    ax.set(xlabel="Mean predicted P(ASD)", ylabel="Observed fraction ASD", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    _save(fig, path)


def plot_prob_dist(y, p, title, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, label="non-ASD", color=PALETTE[0])
    ax.hist(p[y == 1], bins=bins, alpha=0.6, label="ASD", color=PALETTE[1])
    ax.set(xlabel="Predicted P(ASD)", ylabel="count", title=title)
    ax.legend()
    _save(fig, path)


def plot_threshold(y, p, title, path):
    if not _two_classes(y):
        return
    ths = np.linspace(0.02, 0.98, 49)
    rows = []
    for t in ths:
        pred = p >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        fn = (~pred & (y == 1)).sum(); tn = (~pred & (y == 0)).sum()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (tp + fp) and (prec + sens) else np.nan
        rows.append((sens, spec, prec, f1))
    r = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for k, name in enumerate(["sensitivity", "specificity", "precision", "F1"]):
        ax.plot(ths, r[:, k], label=name, color=PALETTE[k])
    ax.set(xlabel="Decision threshold on P(ASD)", ylabel="value", title=title, ylim=(0, 1.02))
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_training_curves(history: pd.DataFrame, title, path):
    """history columns: fold, epoch, train_loss, val_loss"""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for i, (f, h) in enumerate(history.groupby("fold")):
        c = PALETTE[i % len(PALETTE)]
        ax.plot(h["epoch"], h["train_loss"], "-o", color=c, ms=3, label=f"fold {f} train")
        if "val_loss" in h and h["val_loss"].notna().any():
            ax.plot(h["epoch"], h["val_loss"], "--s", color=c, ms=3, label=f"fold {f} val")
    ax.set(xlabel="epoch", ylabel="loss", title=title)
    ax.legend(fontsize=6, ncol=2)
    _save(fig, path)


def plot_foldwise(fold_df: pd.DataFrame, metric: str, title, path):
    """fold_df columns: model, fold, <metric>"""
    piv = fold_df.pivot(index="fold", columns="model", values=metric)
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * piv.shape[1] + 2), 3.8))
    w = 0.8 / max(piv.shape[1], 1)
    for i, m in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w, piv[m].astype(float), width=w, label=m, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(np.arange(len(piv)) + w * (piv.shape[1] - 1) / 2, [f"fold {f}" for f in piv.index])
    ax.set(ylabel=metric, title=title, ylim=(0, 1.05))
    ax.axhline(0.5, ls=":", color="grey")
    ax.legend(fontsize=7)
    _save(fig, path)


def plot_model_comparison(summary: pd.DataFrame, metrics, title, path):
    """summary index=model; columns <metric>_mean, <metric>_std"""
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(summary) + 3), 4))
    w = 0.8 / len(metrics)
    x = np.arange(len(summary))
    for i, m in enumerate(metrics):
        mu = summary[f"{m}_mean"].astype(float)
        sd = summary[f"{m}_std"].astype(float).fillna(0)
        ax.bar(x + i * w, mu, yerr=sd, width=w, capsize=2, label=m, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x + w * (len(metrics) - 1) / 2, summary.index, rotation=25, ha="right", fontsize=8)
    ax.axhline(0.5, ls=":", color="grey")
    ax.set(ylim=(0, 1.05), title=title, ylabel="fold mean ± SD")
    ax.legend(fontsize=7)
    _save(fig, path)


def full_classifier_plot_set(name: str, oof: pd.DataFrame, plots_dir: Path, level: str = "participant"):
    """oof columns: y_true, y_prob, fold, (threshold). Produces the standard plot set for one model."""
    y = oof["y_true"].to_numpy().astype(int)
    p = oof["y_prob"].to_numpy().astype(float)
    folds = oof["fold"].to_numpy()
    th = float(oof["threshold"].iloc[0]) if "threshold" in oof else 0.5
    pred = (p >= oof["threshold"].to_numpy() if "threshold" in oof else p >= 0.5).astype(int)
    cm = skm.confusion_matrix(y, pred, labels=[0, 1])
    tag = f"{name} [{level}]"
    plot_roc(y, p, folds, f"ROC — {tag}", plots_dir / "roc" / f"{name}_{level}.png")
    plot_pr(y, p, folds, f"PR — {tag}", plots_dir / "precision_recall" / f"{name}_{level}.png")
    plot_confusion(cm, f"{tag}", plots_dir / "confusion_matrices" / f"{name}_{level}.png")
    plot_confusion(cm, f"{tag}", plots_dir / "confusion_matrices" / f"{name}_{level}_norm.png", normalize=True)
    plot_calibration(y, p, f"Calibration — {tag}", plots_dir / "calibration" / f"{name}_{level}.png")
    plot_prob_dist(y, p, f"P(ASD) by class — {tag}", plots_dir / "distributions" / f"{name}_{level}.png")
    plot_threshold(y, p, f"Threshold analysis — {tag}", plots_dir / "thresholds" / f"{name}_{level}.png")
    return th
