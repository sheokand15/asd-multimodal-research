"""Experiment C — independent questionnaire (Q-CHAT-10 toddler) classification.

Models: logistic regression, gradient boosting (LightGBM), MLP (PyTorch, SMOKE_TEST epochs).
Feature sets: 'items' (target-defining items -> rule reproduction) and 'no_items' (leakage-controlled).
CV: 5-fold StratifiedGroupKFold, groups = identical-row clusters (possible repeat submissions).
All preprocessing (imputation, one-hot, scaling) is fit on training folds only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.data import questionnaire as Q  # noqa: E402
from src.evaluation import plots  # noqa: E402
from src.evaluation.report import evaluate_binary, summary_table  # noqa: E402
from src.models.eeg_cnn import TabularMLP  # noqa: E402
from src.training.cv import participant_folds, save_fold_assignments, youden_threshold  # noqa: E402
from src.training.trainer import predict, train_model  # noqa: E402
from src.utils.config import (ensure_dirs, early_stopping_patience, get_logger, load_config, save_json,  # noqa: E402
                              set_seed)

EXP = "C_questionnaire"


def preprocessor(fs):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), fs["numeric"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), fs["categorical"]),
    ], sparse_threshold=0)


def sk_models(seed):
    return {
        "logreg": LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed),
        "lightgbm": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, class_weight="balanced",
                                   random_state=seed, verbose=-1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    log = get_logger("questionnaire", cfg["paths"]["logs"])
    seed, E = cfg["experiment"]["random_seed"], cfg["experiment"]
    set_seed(seed)
    df = Q.load_toddler(cfg["paths"]["questionnaire"])
    save_json(Q.audit(cfg["paths"]["questionnaire"]), Path(cfg["paths"]["audit"]) / "questionnaire_audit.json")
    y, groups = df["y"].to_numpy(), df["dup_group"].to_numpy()
    folds = list(participant_folds(groups, y, E["n_splits"], seed))
    save_fold_assignments(groups, y, folds, Path(cfg["paths"]["metrics"]) / EXP / "fold_assignments.csv")
    results = []
    for fs_name, fs in Q.feature_sets().items():
        cols = fs["numeric"] + fs["categorical"]
        X = df[cols]
        # --- classical models
        for mname, est in sk_models(seed).items():
            rows, t0 = [], time.perf_counter()
            for k, tr, te in folds:
                pipe = Pipeline([("pre", preprocessor(fs)), ("clf", est)])
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                p_in = cross_val_predict(pipe, X.iloc[tr], y[tr], groups=groups[tr], cv=inner,
                                         method="predict_proba")[:, 1]
                th = youden_threshold(y[tr], p_in)
                pipe.fit(X.iloc[tr], y[tr])
                p = pipe.predict_proba(X.iloc[te])[:, 1]
                rows.append(pd.DataFrame({"participant": df["Case_No"].iloc[te].astype(str).values, "fold": k,
                                          "y_true": y[te], "y_prob": p, "threshold": th}))
            name = f"{mname}__{fs_name}"
            r = evaluate_binary(name, pd.concat(rows), cfg, EXP,
                                extra={"feature_set": fs_name, "feature_note": fs["note"],
                                       "train_time_s": time.perf_counter() - t0})
            log.info(f"{name}: AUC={r['pooled_participant']['roc_auc']:.3f} "
                     f"BA={r['pooled_participant']['balanced_accuracy']:.3f}")
            results.append(r)
        # --- MLP
        rows, hist_rows, info_all = [], [], []
        for k, tr, te in folds:
            pre = preprocessor(fs).fit(X.iloc[tr])
            Xtr, Xte = pre.transform(X.iloc[tr]).astype(np.float32), pre.transform(X.iloc[te]).astype(np.float32)
            tri = np.arange(len(tr))
            vsplit = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xtr, y[tr], groups[tr]))
            a, b = vsplit
            model = TabularMLP(Xtr.shape[1])
            model, hist, info = train_model(model, [Xtr[a]], y[tr][a], [Xtr[b]], y[tr][b], epochs=E["epochs"],
                                            batch_size=E["batch_size"], lr=cfg["training"]["learning_rate"],
                                            weight_decay=cfg["training"]["weight_decay"],
                                            amp=cfg["training"]["mixed_precision"], seed=seed,
                                            ckpt_path=Path(cfg["paths"]["checkpoints"]) / EXP / f"mlp__{fs_name}_fold{k}.pt",
                                            early_stopping_patience=early_stopping_patience(cfg), log=log.info)
            th = youden_threshold(y[tr][b], predict(model, [Xtr[b]]))
            p = predict(model, [Xte])
            rows.append(pd.DataFrame({"participant": df["Case_No"].iloc[te].astype(str).values, "fold": k,
                                      "y_true": y[te], "y_prob": p, "threshold": th}))
            hist_rows += [{**h, "fold": k} for h in hist]
            info_all.append(info)
            del tri
        name = f"mlp__{fs_name}"
        hist_df = pd.DataFrame(hist_rows)
        hist_df.to_csv(Path(cfg["paths"]["metrics"]) / EXP / f"{name}_history.csv", index=False)
        plots.plot_training_curves(hist_df, f"MLP ({fs_name}) — {cfg['run_label']}",
                                   Path(cfg["paths"]["plots"]) / "training_curves" / f"{EXP}__{name}.png")
        from src.evaluation.metrics import count_params
        extra = {"feature_set": fs_name, "feature_note": fs["note"], **count_params(model),
                 "train_time_s": sum(i["train_time_s"] for i in info_all),
                 "epoch_time_s": float(np.mean([i["epoch_time_s"] for i in info_all])),
                 "checkpoint_mb": info_all[0].get("checkpoint_mb"), "peak_gpu_mb": info_all[0].get("peak_gpu_mb")}
        r = evaluate_binary(name, pd.concat(rows), cfg, EXP, extra=extra)
        log.info(f"{name}: AUC={r['pooled_participant']['roc_auc']:.3f} BA={r['pooled_participant']['balanced_accuracy']:.3f}")
        results.append(r)

    summ = summary_table(results)
    summ.to_csv(Path(cfg["paths"]["metrics"]) / EXP / "summary.csv", index=False)
    fold_df = pd.concat([pd.read_csv(Path(cfg["paths"]["metrics"]) / EXP / f"{r['model']}_folds.csv") for r in results])
    fold_df = fold_df[fold_df.level == "participant"]
    fold_df.to_csv(Path(cfg["paths"]["metrics"]) / EXP / "all_folds.csv", index=False)
    plots.plot_foldwise(fold_df, "roc_auc", "Questionnaire — fold-wise ROC-AUC",
                        Path(cfg["paths"]["plots"]) / "comparisons" / f"{EXP}_foldwise_auc.png")
    s = summ.set_index("model")
    plots.plot_model_comparison(s, ["roc_auc", "balanced_accuracy", "f1", "mcc"], "Questionnaire models (5-fold)",
                                Path(cfg["paths"]["plots"]) / "comparisons" / f"{EXP}_models.png")
    log.info("\n" + summ[["model", "roc_auc_mean", "roc_auc_std", "balanced_accuracy_mean", "f1_mean"]].to_string())


if __name__ == "__main__":
    main()
