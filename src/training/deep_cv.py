"""Participant-level CV loop for deep binary classifiers (used by the Move4AS experiments)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import count_params, measure_inference
from src.training.cv import inner_split, youden_threshold
from src.training.trainer import device, predict, train_model
from src.utils.config import early_stopping_patience


def participant_mean(groups, p):
    s = pd.Series(p).groupby(np.asarray(groups)).mean()
    return s


def threshold_from_val(groups_val, y_val, p_val) -> float:
    pm = participant_mean(groups_val, p_val)
    ym = pd.Series(y_val).groupby(np.asarray(groups_val)).first().loc[pm.index]
    return youden_threshold(ym.to_numpy(), pm.to_numpy())


def run_deep_binary_cv(name: str, build: Callable[[], torch.nn.Module], inputs: list[np.ndarray], y: np.ndarray,
                       groups: np.ndarray, folds, cfg: dict, experiment: str, log=print,
                       pretrain: Callable | None = None, meta: pd.DataFrame | None = None):
    """Train ``build()`` in each outer fold on inner-train participants, select threshold on the
    inner-validation participants, predict the held-out fold.

    Returns (oof DataFrame, history DataFrame, info dict, val_preds dict fold->(groups,y,p)).
    """
    E, T = cfg["experiment"], cfg["training"]
    rows, hist_rows, infos, val_store = [], [], [], {}
    for k, tr, te in folds:
        a, b = inner_split(groups[tr], y[tr], T.get("inner_val_fraction", 0.2), E["random_seed"] + k)
        itr, iva = tr[a], tr[b]
        assert not set(groups[iva]) & set(groups[te]) and not set(groups[itr]) & set(groups[te])
        model = build()
        if pretrain is not None:  # e.g. MAE pre-training on training-fold participants only
            model = pretrain(model, [x[itr] for x in inputs], [x[iva] for x in inputs], k)
        model, hist, info = train_model(
            model, [x[itr] for x in inputs], y[itr], [x[iva] for x in inputs], y[iva],
            epochs=E["epochs"], batch_size=E["batch_size"], lr=T["learning_rate"], weight_decay=T["weight_decay"],
            amp=T["mixed_precision"], seed=E["random_seed"] + k,
            ckpt_path=Path(cfg["paths"]["checkpoints"]) / experiment / f"{name}_fold{k}.pt" if T["save_checkpoints"] else None,
            early_stopping_patience=early_stopping_patience(cfg), log=log)
        p_val = predict(model, [x[iva] for x in inputs])
        th = threshold_from_val(groups[iva], y[iva], p_val)
        p = predict(model, [x[te] for x in inputs])
        val_store[k] = (groups[iva], y[iva], p_val)
        df = pd.DataFrame({"participant": groups[te], "fold": k, "y_true": y[te], "y_prob": p, "threshold": th})
        if meta is not None:
            df = pd.concat([df.reset_index(drop=True), meta.iloc[te].reset_index(drop=True)], axis=1)
        rows.append(df)
        hist_rows += [{**h, "fold": k} for h in hist]
        infos.append(info)
        log(f"  [{name}] fold {k}: n_test_participants={len(np.unique(groups[te]))} threshold={th:.3f}")
    # efficiency (last fold model, batch = config batch size)
    ex = tuple(torch.as_tensor(x[: E["batch_size"]], dtype=torch.float32, device=device()) for x in inputs)
    eff = {**count_params(model), **measure_inference(model, ex)}
    info = {**eff, "train_time_s": float(sum(i["train_time_s"] for i in infos)),
            "epoch_time_s": float(np.mean([i["epoch_time_s"] for i in infos])),
            "peak_gpu_mb": max((i["peak_gpu_mb"] or 0) for i in infos) or None,
            "checkpoint_mb": infos[0].get("checkpoint_mb"), "epochs": E["epochs"]}
    return pd.concat(rows, ignore_index=True), pd.DataFrame(hist_rows), info, val_store
