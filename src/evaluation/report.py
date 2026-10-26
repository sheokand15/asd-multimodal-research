"""Turn saved out-of-fold predictions into fold-wise / pooled metrics, CIs and plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import plots
from src.evaluation.metrics import SCALAR_KEYS, binary_metrics, bootstrap_ci
from src.utils.config import save_json


def _participant_level(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean P(ASD) over a participant's trials; threshold/fold are constant within participant."""
    g = preds.groupby("participant")
    return pd.DataFrame({"participant": list(g.groups), "y_true": g["y_true"].first().values,
                         "y_prob": g["y_prob"].mean().values, "fold": g["fold"].first().values,
                         "threshold": g["threshold"].first().values})


def evaluate_binary(name: str, preds: pd.DataFrame, cfg: dict, experiment: str, extra: dict | None = None,
                    trial_level: bool = True, make_plots: bool = True) -> dict:
    """preds columns: participant, fold, y_true, y_prob, threshold (+ any metadata).

    Primary unit = participant (trial probabilities averaged).  Trial-level metrics are also kept
    when observations are trials.  Fold metrics use each fold's own training-selected threshold.
    """
    P = cfg["paths"]
    preds = preds.copy()
    preds["experiment"] = experiment
    preds["model"] = name
    pred_path = Path(P["predictions"]) / experiment / f"{name}_oof.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)

    levels = {"participant": _participant_level(preds)}
    if trial_level and len(preds) > preds["participant"].nunique():
        levels["trial"] = preds
    result = {"model": name, "experiment": experiment, "run_label": cfg.get("run_label"), **(extra or {})}
    fold_rows = []
    for lvl, df in levels.items():
        y = df["y_true"].to_numpy().astype(int)
        p = df["y_prob"].to_numpy().astype(float)
        th = df["threshold"].to_numpy().astype(float)
        # pooled OOF: threshold applied per row (each row carries its own fold's threshold)
        pooled = binary_metrics(y, p, 0.5)
        pred_bin = (p >= th).astype(int)
        pooled_t = binary_metrics(y, p, y_pred=pred_bin)
        pooled_t["threshold"] = "per-fold (training-selected)"
        groups = df["participant"].to_numpy()
        seed = cfg["experiment"]["random_seed"]
        lo, hi = bootstrap_ci(y, p, groups, "roc_auc", seed=seed)
        blo, bhi = bootstrap_ci(y, p, groups, "balanced_accuracy", seed=seed, y_pred=pred_bin)
        pooled_t["roc_auc_ci95"] = [lo, hi]
        pooled_t["balanced_accuracy_ci95"] = [blo, bhi]
        result[f"pooled_{lvl}"] = pooled_t
        result[f"pooled_{lvl}_fixed0.5"] = pooled
        for f, d in df.groupby("fold"):
            m = binary_metrics(d["y_true"].to_numpy().astype(int), d["y_prob"].to_numpy(),
                               float(d["threshold"].iloc[0]))
            fold_rows.append({"model": name, "experiment": experiment, "level": lvl, "fold": int(f),
                              "n_participants": int(d["participant"].nunique()),
                              **{k: m[k] for k in SCALAR_KEYS}, "threshold": m["threshold"],
                              "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                              "unavailable": "; ".join(f"{k}: {v}" for k, v in m["unavailable"].items())})
        if make_plots:
            plots.full_classifier_plot_set(f"{experiment}__{name}", df, Path(P["plots"]), level=lvl)
    fold_df = pd.DataFrame(fold_rows)
    mdir = Path(P["metrics"]) / experiment
    mdir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(mdir / f"{name}_folds.csv", index=False)
    summ = {}
    for lvl, d in fold_df.groupby("level"):
        for k in SCALAR_KEYS:
            v = pd.to_numeric(d[k], errors="coerce")
            summ[f"{lvl}_{k}_mean"] = float(v.mean()) if v.notna().any() else None
            summ[f"{lvl}_{k}_std"] = float(v.std(ddof=1)) if v.notna().sum() > 1 else None
            summ[f"{lvl}_{k}_n_folds_available"] = int(v.notna().sum())
    result["fold_summary"] = summ
    save_json(result, mdir / f"{name}_metrics.json")
    return result


def summary_table(results: list[dict], level: str = "participant") -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"experiment": r["experiment"], "model": r["model"]}
        pooled = r.get(f"pooled_{level}", {})
        for k in SCALAR_KEYS:
            row[f"{k}_mean"] = r["fold_summary"].get(f"{level}_{k}_mean")
            row[f"{k}_std"] = r["fold_summary"].get(f"{level}_{k}_std")
            row[f"pooled_{k}"] = pooled.get(k)
        row["pooled_roc_auc_ci95"] = pooled.get("roc_auc_ci95")
        row["n"] = pooled.get("n")
        for k in ("total_params", "trainable_params", "checkpoint_mb", "train_time_s", "epoch_time_s",
                  "latency_ms_per_batch", "throughput_samples_per_s", "peak_gpu_mb"):
            if k in r:
                row[k] = r[k]
        rows.append(row)
    return pd.DataFrame(rows)
