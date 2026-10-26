"""Skeleton-MAE pre-training restricted to training-fold participants (strict participant generalisation)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import reconstruction_metrics
from src.models.mae import SkeletonMAE
from src.training.trainer import device


def pretrain_mae(mae: SkeletonMAE, X_train: np.ndarray, X_val: np.ndarray, *, epochs: int, batch_size: int, lr: float,
                 weight_decay: float, seed: int, amp: bool, ckpt: Path | None, valid_train=None, valid_val=None,
                 log=print):
    dev = device()
    torch.manual_seed(seed)
    mae = mae.to(dev)
    vt = valid_train if valid_train is not None else np.ones_like(X_train, dtype=bool)
    vv = valid_val if valid_val is not None else np.ones_like(X_val, dtype=bool)
    tl = DataLoader(TensorDataset(torch.as_tensor(X_train), torch.as_tensor(vt)), batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed), drop_last=len(X_train) > batch_size)
    vl = DataLoader(TensorDataset(torch.as_tensor(X_val), torch.as_tensor(vv)), batch_size=128)
    opt = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = []
    for ep in range(1, epochs + 1):
        mae.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for xb, mb in tl:
            xb, mb = xb.to(dev).float(), mb.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss, _, _ = mae(xb, mb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(xb)
            n += len(xb)
        mae.eval()
        vtot, vn, recs, tg, ms = 0.0, 0, [], [], []
        torch.manual_seed(seed + 999)  # same random masks every epoch -> comparable validation loss
        with torch.no_grad():
            for xb, mb in vl:
                xb, mb = xb.to(dev).float(), mb.to(dev)
                loss, rec, m = mae(xb, mb)
                vtot += loss.item() * len(xb)
                vn += len(xb)
                recs.append(rec.float().cpu().numpy()); tg.append(xb.cpu().numpy()); ms.append(m.cpu().numpy())
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": vtot / max(vn, 1),
                     "epoch_time_s": time.perf_counter() - t0})
        log(f"    [MAE] epoch {ep}/{epochs} recon train={hist[-1]['train_loss']:.4f} val={hist[-1]['val_loss']:.4f}")
    rm = reconstruction_metrics(np.concatenate(tg), np.concatenate(recs), np.concatenate(ms))
    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(mae.state_dict(), ckpt)
    return mae, pd.DataFrame(hist), rm
