"""Participant-level cross-validation utilities shared by all experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def participant_folds(groups, y=None, n_splits: int = 5, seed: int = 42):
    """Yield (fold, train_idx, test_idx). Every observation of a participant lands in one fold.

    StratifiedGroupKFold when a binary label is available (labels are constant within a
    participant), otherwise GroupKFold.
    """
    groups = np.asarray(groups)
    if y is not None and len(np.unique(y)) > 1:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = cv.split(np.zeros(len(groups)), y, groups)
    else:
        it = GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)
    for k, (tr, te) in enumerate(it):
        assert not set(groups[tr]) & set(groups[te]), "participant leakage between train and test"
        yield k, tr, te


def inner_split(groups, y, val_fraction: float = 0.2, seed: int = 42):
    """Single participant-level train/validation split inside a training fold (for deep models)."""
    groups = np.asarray(groups)
    y = np.asarray(y)
    n_splits = max(2, int(round(1 / val_fraction)))
    tr, va = next(iter(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                       .split(np.zeros(len(groups)), y, groups)))
    return tr, va


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity − 1 on *training/validation* predictions only."""
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(np.concatenate([p, [0.5]]))
    best, best_t = -1.0, 0.5
    for t in cand:
        pred = p >= t
        sens = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
        spec = (~pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        j = sens + spec - 1
        if j > best + 1e-12 or (abs(j - best) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = j, float(t)
    return best_t


def save_fold_assignments(groups, y, folds_iter, path: Path, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    groups = np.asarray(groups)
    y = np.asarray(y) if y is not None else None
    for k, tr, te in folds_iter:
        for g in np.unique(groups[te]):
            rows.append({"participant": g, "fold": k,
                         "label": int(y[groups == g][0]) if y is not None else None})
    df = pd.DataFrame(rows).sort_values(["fold", "participant"])
    if extra:
        for k_, v in extra.items():
            df[k_] = v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
