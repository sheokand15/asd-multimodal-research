"""Collect actual metric files into markdown tables: outputs/reports/results_tables.md

The narrative report (outputs/reports/final_report.md) embeds these tables; numbers are never typed by hand.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config  # noqa: E402


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x, nd) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def ms(row, k):
    return f"{fmt(row.get(f'{k}_mean'))} ± {fmt(row.get(f'{k}_std'))}"


def binary_table(mdir: Path, names: list[str] | None = None, level="participant") -> str:
    rows = []
    for f in sorted(mdir.glob("*_metrics.json")):
        r = json.loads(f.read_text())
        if names and r["model"] not in names:
            continue
        fs = r["fold_summary"]
        get = lambda k: (fs.get(f"{level}_{k}_mean"), fs.get(f"{level}_{k}_std"))
        p = r.get(f"pooled_{level}", {})
        rows.append({"model": r["model"], "n": p.get("n"),
                     **{k: f"{fmt(get(k)[0])} ± {fmt(get(k)[1])}" for k in
                        ("roc_auc", "balanced_accuracy", "recall_sensitivity", "specificity", "f1", "mcc")},
                     "pooled AUC [95% CI]": f"{fmt(p.get('roc_auc'))} {fmt(p.get('roc_auc_ci95'))}",
                     "pooled BA": fmt(p.get("balanced_accuracy")), "pooled Brier": fmt(p.get("brier")),
                     "TP/FP/TN/FN": f"{p.get('tp')}/{p.get('fp')}/{p.get('tn')}/{p.get('fn')}"})
    if not rows:
        return "_no results_\n"
    df = pd.DataFrame(rows)
    if names:
        df["_o"] = df.model.map({n: i for i, n in enumerate(names)})
        df = df.sort_values("_o").drop(columns="_o")
    return df.to_markdown(index=False) + "\n"


def efficiency_table(mdir: Path) -> str:
    rows = []
    for f in sorted(mdir.glob("*_metrics.json")):
        r = json.loads(f.read_text())
        if "total_params" in r:
            rows.append({"model": r["model"], "params": r["total_params"], "trainable": r["trainable_params"],
                         "ckpt MB": fmt(r.get("checkpoint_mb"), 2), "s/epoch": fmt(r.get("epoch_time_s"), 2),
                         "total train s (5 folds)": fmt(r.get("train_time_s"), 1),
                         "latency ms/batch": fmt(r.get("latency_ms_per_batch"), 2),
                         "throughput /s": fmt(r.get("throughput_samples_per_s"), 0),
                         "peak GPU MB": fmt(r.get("peak_gpu_mb"), 0)})
    return pd.DataFrame(rows).to_markdown(index=False) + "\n" if rows else "_none_\n"


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    M = Path(cfg["paths"]["metrics"])
    out = []
    out.append(f"# Results tables ({cfg['run_label']})\n\nGenerated from `{M}`. Mean ± SD over 5 participant-level "
               "folds; pooled = out-of-fold predictions of all participants (each fold's own training-selected "
               "threshold); 95 % CI = participant-level bootstrap (1,000 resamples).\n")
    out.append("## A. Move4AS — primary comparison (paired trials, participant level)\n")
    core = ["motion_feat_lr", "eeg_feat_lr", "early_fusion_lr", "late_fusion_lr",
            "motion_ctrgcn", "eeg_cnn", "intermediate_fusion", "late_fusion_deep", "motion_poseformerv2", "motion_mae",
            "motion_feat_lr_all", "eeg_feat_lr_all"]
    out.append(binary_table(M / "A_move4as", core))
    out.append("\n### A. Trial level (secondary)\n")
    out.append(binary_table(M / "A_move4as", core, level="trial"))
    out.append(f"\n### A. Computational efficiency (deep models; inference batch = {cfg['experiment']['batch_size']})\n")
    out.append(efficiency_table(M / "A_move4as"))
    if (M / "A_move4as_sensitivity").exists():
        out.append("\n## A. Sensitivity analyses (participant level)\n")
        out.append(binary_table(M / "A_move4as_sensitivity"))
    out.append("\n## B. MMASD\n")
    if (M / "B_mmasd" / "b1_activity_summary.csv").exists():
        s = pd.read_csv(M / "B_mmasd" / "b1_activity_summary.csv")
        b1 = json.loads((M / "B_mmasd" / "b1_dataset.json").read_text())
        out.append(f"### B1 Activity recognition ({len(b1['activities'])} classes, {b1['n_clips']} clips, "
                   f"{b1['n_participants']} participants; chance accuracy ≈ {1 / len(b1['activities']):.3f})\n")
        cols = ["model", "pooled_accuracy", "pooled_balanced_accuracy", "pooled_macro_f1", "pooled_cohen_kappa",
                "pooled_mcc", "pooled_macro_roc_auc_ovr", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
        out.append(s[[c for c in cols if c in s]].round(3).to_markdown(index=False) + "\n")
        e = pd.read_csv(M / "B_mmasd" / "b1_efficiency.csv").round(2)
        out.append("\n### B1 Efficiency\n" + e.to_markdown(index=False) + "\n")
    if (M / "B_mmasd" / "b2_mae_reconstruction.csv").exists():
        r = pd.read_csv(M / "B_mmasd" / "b2_mae_reconstruction.csv")
        out.append("\n### B2 Skeleton-MAE reconstruction (validation participants, masked & valid coordinates)\n")
        out.append(r.groupby("representation")[["mean_abs_coord_error", "mean_sq_coord_error", "n_scored"]]
                   .agg(["mean", "std"]).round(4).to_markdown() + "\n")
    if (M / "B_mmasd" / "b3_ados_regression.csv").exists():
        r = pd.read_csv(M / "B_mmasd" / "b3_ados_regression.csv")
        out.append("\n### B3 ADOS-2 regression (leave-one-participant-out)\n")
        out.append(r[["target", "model", "n_participants", "mae", "rmse", "r2", "pearson_r", "spearman_rho",
                      "spearman_p"]].round(3).to_markdown(index=False) + "\n")
    out.append("\n## C. Questionnaire (Q-CHAT-10 toddlers; target = screening rule)\n")
    out.append(binary_table(M / "C_questionnaire"))
    out.append("\n### C. Efficiency\n" + efficiency_table(M / "C_questionnaire"))
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        tv = torch.__version__
    except Exception:
        gpu, tv = "unknown", "n/a"
    out.append(f"\n## Hardware / software\n- {platform.platform()}, Python {platform.python_version()}, torch {tv}\n"
               f"- GPU: {gpu}; mixed precision: {cfg['training']['mixed_precision']}\n"
               f"- Seed {cfg['experiment']['random_seed']}, {cfg['experiment']['n_splits']} folds, "
               f"{cfg['experiment']['epochs']} epochs, batch {cfg['experiment']['batch_size']}, "
               f"lr {cfg['training']['learning_rate']}\n")
    p = Path(cfg["paths"]["reports"]) / "results_tables.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
