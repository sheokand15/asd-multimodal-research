"""Reusable metrics.

Positive-class convention: label 1 = ASD (Move4AS ``P*`` participants / questionnaire "Yes"),
label 0 = non-ASD.  Metrics that cannot be computed (e.g. a class absent from a fold) are
returned as ``None`` together with a reason in ``unavailable`` — never silently replaced by 0.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import stats
from sklearn import metrics as skm

POSITIVE_LABEL = 1
POSITIVE_NAME = "ASD"


def _safe_div(a: float, b: float):
    return float(a) / float(b) if b else None


def binary_metrics(y_true, y_prob, threshold: float = 0.5, y_pred=None) -> dict:
    """All binary classification metrics from probabilities of the positive class.

    ``y_pred`` overrides thresholding (used for pooled OOF predictions where every fold has its own
    training-selected threshold); threshold-free metrics always use ``y_prob``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)
        threshold = float("nan")
    out: dict = {"n": int(len(y_true)), "n_pos": int(y_true.sum()), "n_neg": int((1 - y_true).sum()),
                 "threshold": float(threshold)}
    unavailable: dict = {}
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))
    out["confusion_matrix"] = [[int(tn), int(fp)], [int(fn), int(tp)]]
    out["accuracy"] = _safe_div(tp + tn, tp + tn + fp + fn)
    out["recall_sensitivity"] = _safe_div(tp, tp + fn)
    out["specificity"] = _safe_div(tn, tn + fp)
    out["precision"] = _safe_div(tp, tp + fp)
    out["npv"] = _safe_div(tn, tn + fn)
    out["fpr"] = _safe_div(fp, fp + tn)
    out["fnr"] = _safe_div(fn, fn + tp)
    p, r = out["precision"], out["recall_sensitivity"]
    out["f1"] = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else (
        0.0 if (p is not None and r is not None) else None)
    sens, spec = out["recall_sensitivity"], out["specificity"]
    out["balanced_accuracy"] = (sens + spec) / 2 if sens is not None and spec is not None else None
    both = len(np.unique(y_true)) == 2
    if both:
        out["macro_f1"] = float(skm.f1_score(y_true, y_pred, average="macro", zero_division=0))
        out["mcc"] = float(skm.matthews_corrcoef(y_true, y_pred))
        out["cohen_kappa"] = float(skm.cohen_kappa_score(y_true, y_pred))
        out["roc_auc"] = float(skm.roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(skm.average_precision_score(y_true, y_prob))
        out["log_loss"] = float(skm.log_loss(y_true, y_prob, labels=[0, 1]))
    else:
        for k in ("macro_f1", "mcc", "cohen_kappa", "roc_auc", "pr_auc", "log_loss"):
            out[k] = None
            unavailable[k] = "only one class present in y_true"
    out["brier"] = float(skm.brier_score_loss(y_true, y_prob, pos_label=1))
    for k in ("recall_sensitivity", "specificity", "precision", "npv", "fpr", "fnr", "f1", "balanced_accuracy"):
        if out[k] is None:
            unavailable[k] = "zero denominator (class absent or no predictions of that class)"
    out["unavailable"] = unavailable
    return out


SCALAR_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall_sensitivity", "specificity", "f1", "macro_f1",
               "mcc", "cohen_kappa", "roc_auc", "pr_auc", "npv", "fpr", "fnr", "log_loss", "brier"]


def bootstrap_ci(y_true, y_prob, groups, metric: str = "roc_auc", n_boot: int = 1000, seed: int = 42,
                 threshold: float = 0.5, alpha: float = 0.05, y_pred=None):
    """Participant-level (cluster) bootstrap CI: resample participants, not windows."""
    rng = np.random.default_rng(seed)
    y_true, y_prob, groups = map(np.asarray, (y_true, y_prob, groups))
    y_pred = None if y_pred is None else np.asarray(y_pred)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y_true[idx])) < 2:
            continue
        yt = y_true[idx]
        if metric == "roc_auc":
            v = skm.roc_auc_score(yt, y_prob[idx])
        elif metric == "balanced_accuracy":
            yp = (y_prob[idx] >= threshold).astype(int) if y_pred is None else y_pred[idx]
            v = 0.5 * (np.mean(yp[yt == 1] == 1) + np.mean(yp[yt == 0] == 0))
        else:
            v = binary_metrics(yt, y_prob[idx], threshold, None if y_pred is None else y_pred[idx])[metric]
        if v is not None:
            vals.append(v)
    if len(vals) < 50:
        return None, None
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def aggregate_to_participant(groups, y_true, y_prob):
    """Average window/trial probabilities per participant -> participant-level predictions."""
    import pandas as pd

    df = pd.DataFrame({"g": groups, "y": y_true, "p": y_prob}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
    return df.index.to_numpy(), df["y"].to_numpy(), df["p"].to_numpy()


def multiclass_metrics(y_true, prob, n_class: int) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, float)
    prob = prob / prob.sum(1, keepdims=True)  # re-normalise float rounding from mixed-precision softmax
    pred = prob.argmax(1)
    labels = list(range(n_class))
    out = {"n": int(len(y_true)), "accuracy": float(skm.accuracy_score(y_true, pred)),
           "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, pred)),
           "macro_f1": float(skm.f1_score(y_true, pred, average="macro", labels=labels, zero_division=0)),
           "weighted_f1": float(skm.f1_score(y_true, pred, average="weighted", labels=labels, zero_division=0)),
           "cohen_kappa": float(skm.cohen_kappa_score(y_true, pred)),
           "mcc": float(skm.matthews_corrcoef(y_true, pred)),
           "log_loss": float(skm.log_loss(y_true, np.clip(prob, 1e-7, 1), labels=labels)),
           "confusion_matrix": skm.confusion_matrix(y_true, pred, labels=labels).tolist()}
    present = np.unique(y_true)
    if len(present) == n_class:
        out["macro_roc_auc_ovr"] = float(skm.roc_auc_score(y_true, prob, multi_class="ovr", labels=labels))
    else:
        out["macro_roc_auc_ovr"] = None
        out["unavailable"] = {"macro_roc_auc_ovr": f"only {len(present)}/{n_class} classes present"}
    return out


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    out = {"n": int(len(y_true)), "mae": float(np.mean(np.abs(y_true - y_pred))), "mse": mse, "rmse": mse ** 0.5}
    out["r2"] = float(skm.r2_score(y_true, y_pred)) if len(y_true) > 1 else None
    if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        out["pearson_r"], out["pearson_p"] = map(float, stats.pearsonr(y_true, y_pred))
        out["spearman_rho"], out["spearman_p"] = map(float, stats.spearmanr(y_true, y_pred))
    else:
        out["pearson_r"] = out["spearman_rho"] = None
        out["unavailable"] = {"pearson_r": "constant predictions or targets", "spearman_rho": "constant"}
    return out


def reconstruction_metrics(target, recon, mask) -> dict:
    """mask: bool array, True where a coordinate was masked AND valid (scored)."""
    target, recon, mask = np.asarray(target), np.asarray(recon), np.asarray(mask, bool)
    err = (recon - target)
    m = mask
    out = {"n_scored": int(m.sum()),
           "mean_abs_coord_error": float(np.abs(err[m]).mean()) if m.any() else None,
           "mean_sq_coord_error": float((err[m] ** 2).mean()) if m.any() else None}
    if target.ndim == 4:  # (N, T, J, C) -> per-joint
        pj = []
        for j in range(target.shape[2]):
            mj = m[:, :, j]
            pj.append(float(np.abs(err[:, :, j][mj]).mean()) if mj.any() else None)
        out["per_joint_mae"] = pj
    return out


# ---------------- computational efficiency ----------------

def count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


def measure_inference(model, example_inputs: tuple, n_warmup: int = 5, n_runs: int = 20) -> dict:
    """Latency per batch (ms) and throughput (samples/s) on the inputs' device, eval mode, no grad."""
    import torch

    model.eval()
    dev = example_inputs[0].device
    bs = example_inputs[0].shape[0]
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*example_inputs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n_runs):
            model(*example_inputs)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        el = (time.perf_counter() - t) / n_runs
    return {"latency_ms_per_batch": el * 1000, "batch_size": int(bs), "throughput_samples_per_s": bs / el,
            "device": str(dev)}
