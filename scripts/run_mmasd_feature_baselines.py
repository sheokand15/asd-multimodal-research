"""MMASD activity recognition — non-deep reference baselines on the SAME clips and participant folds as B1.

* majority class      : most frequent activity among each fold's training clips
* 3D kinematic + LR   : per-joint speed / speed SD / acceleration / range of motion (F3d.npy), L2 logistic regression
* 3D kinematic + GBM  : same features, LightGBM
Scaling and model fitting use training-fold clips only.  Also repairs the `clip` column of earlier B1 prediction
files (a bug wrote pandas' DataFrame.clip method instead of clip names) after verifying row alignment.
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
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.evaluation.metrics import multiclass_metrics  # noqa: E402
from src.utils.config import load_config, save_json  # noqa: E402

EXP = "B_mmasd"


def b1_clip_set(cfg):
    """Re-create run_mmasd.py's B1 selection (same order)."""
    d = Path(cfg["paths"]["processed"]) / "mmasd"
    idx = pd.read_csv(d / "index.csv", dtype={"participant": str})
    counts = idx[idx.ok2d].activity.value_counts()
    acts = sorted(counts[counts >= 30].index)
    sel = (idx.ok2d & idx.activity.isin(acts)).to_numpy()
    return idx, sel, acts, np.load(d / "F3d.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["experiment"]["random_seed"]
    idx, sel, acts, F = b1_clip_set(cfg)
    I = idx[sel].reset_index(drop=True)
    X = F[sel]
    y = I.activity.map({a: i for i, a in enumerate(acts)}).to_numpy()
    g = I.participant.to_numpy()
    mdir = Path(cfg["paths"]["metrics"]) / EXP
    pdir = Path(cfg["paths"]["predictions"])
    fa = pd.read_csv(mdir / "fold_assignments.csv", dtype={"participant": str})
    fold = I.participant.map(dict(zip(fa.participant, fa.fold))).to_numpy()
    assert not np.isnan(fold.astype(float)).any()

    # repair clip column of existing deep-model prediction files
    for f in sorted(pdir.glob(f"{EXP}__*_oof.csv")):
        p = pd.read_csv(f, dtype={"participant": str})
        if len(p) == len(I) and (p.participant.values == g).all() and (p.y_true.values == y).all():
            if not p["clip"].astype(str).str.fullmatch(r"[a-z]+_\d+_D\d+_\d+_[a-z]").all():
                p["clip"] = I["clip"].values
                p.to_csv(f, index=False)
                print("repaired clip column:", f.name)
        else:
            raise SystemExit(f"row alignment check failed for {f}")

    models = {
        "majority_class": None,
        "kinematic3d_logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)),
        "kinematic3d_lightgbm": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                                       class_weight="balanced", random_state=seed, verbose=-1),
    }
    rows = []
    for name, build in models.items():
        oof = np.zeros((len(y), len(acts)))
        for k in sorted(np.unique(fold)):
            tr, te = fold != k, fold == k
            if build is None:
                oof[te, np.bincount(y[tr], minlength=len(acts)).argmax()] = 1.0
            else:
                m = build().fit(X[tr], y[tr])
                pr = np.zeros((te.sum(), len(acts)))
                pr[:, m.classes_] = m.predict_proba(X[te])
                oof[te] = pr
        oof = np.clip(oof, 1e-6, 1)
        met = multiclass_metrics(y, oof, len(acts))
        pd.DataFrame({"clip": I["clip"].values, "participant": g, "y_true": y, "y_pred": oof.argmax(1),
                      **{f"p_{a}": oof[:, i] for i, a in enumerate(acts)}}).to_csv(
            pdir / f"{EXP}__{name}_oof.csv", index=False)
        save_json({"model": name, "representation": "3d_romp_kinematic_features" if build else "none",
                   **{f"pooled_{k}": v for k, v in met.items() if k not in ("confusion_matrix", "unavailable")},
                   "confusion_matrix": met["confusion_matrix"], "classes": acts}, mdir / f"{name}_metrics.json")
        rows.append({"model": name, "accuracy": met["accuracy"], "balanced_accuracy": met["balanced_accuracy"],
                     "macro_f1": met["macro_f1"]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
