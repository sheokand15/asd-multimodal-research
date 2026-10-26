"""Generic PyTorch trainer used by every deep model (binary or multi-class, single or multi-input)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(inputs, y, bs, shuffle, seed, nw=0):
    ts = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in inputs]
    yt = torch.as_tensor(np.asarray(y))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*ts, yt), batch_size=bs, shuffle=shuffle, generator=g, num_workers=nw,
                      drop_last=shuffle and len(yt) > bs)


def _loss_fn(task, y_train, n_class):
    if task == "binary":
        pos = float(np.sum(y_train == 1))
        neg = float(np.sum(y_train == 0))
        pw = torch.tensor([neg / max(pos, 1.0)], device=device())  # class weighting from TRAINING labels only
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    counts = np.bincount(y_train.astype(int), minlength=n_class).astype(float)
    w = counts.sum() / np.maximum(counts, 1) / n_class
    return nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device()))


def _batch_loss(model, batch, loss_fn, task):
    *xs, yb = batch
    xs = [x.to(device(), non_blocking=True) for x in xs]
    yb = yb.to(device())
    out = model(*xs)
    if task == "binary":
        return loss_fn(out.squeeze(-1), yb.float()), out
    return loss_fn(out, yb.long()), out


def train_model(model: nn.Module, train_inputs, y_train, val_inputs=None, y_val=None, *, task="binary", n_class=1,
                epochs=3, batch_size=8, lr=1e-4, weight_decay=1e-4, amp=True, seed=42, ckpt_path: Path | None = None,
                early_stopping_patience: int | None = None, log=print):
    dev = device()
    torch.manual_seed(seed)
    model = model.to(dev)
    y_train = np.asarray(y_train)
    tl = _loader(train_inputs, y_train, batch_size, True, seed)
    vl = _loader(val_inputs, y_val, max(batch_size, 64), False, seed) if val_inputs is not None else None
    loss_fn = _loss_fn(task, y_train, n_class)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hist, best, bad, best_state = [], np.inf, 0, None
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for batch in tl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=use_amp):
                loss, _ = _batch_loss(model, batch, loss_fn, task)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(batch[-1])
            n += len(batch[-1])
        rec = {"epoch": ep, "train_loss": tot / max(n, 1), "epoch_time_s": time.perf_counter() - t0}
        if vl is not None:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad(), torch.autocast(dev.type, enabled=use_amp):
                for batch in vl:
                    l, _ = _batch_loss(model, batch, loss_fn, task)
                    vt += l.item() * len(batch[-1])
                    vn += len(batch[-1])
            rec["val_loss"] = vt / max(vn, 1)
        hist.append(rec)
        log(f"    epoch {ep}/{epochs} train_loss={rec['train_loss']:.4f}"
            + (f" val_loss={rec['val_loss']:.4f}" if "val_loss" in rec else "") + f" ({rec['epoch_time_s']:.1f}s)")
        if early_stopping_patience and "val_loss" in rec:
            if rec["val_loss"] < best - 1e-4:
                best, bad = rec["val_loss"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= early_stopping_patience:
                    log(f"    early stop at epoch {ep}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"train_time_s": time.perf_counter() - t_start,
            "epoch_time_s": float(np.mean([h["epoch_time_s"] for h in hist])),
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else None,
            "epochs_run": len(hist)}
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        info["checkpoint"] = str(ckpt_path)
        info["checkpoint_mb"] = ckpt_path.stat().st_size / 2 ** 20
    return model, hist, info


@torch.no_grad()
def predict(model, inputs, task="binary", batch_size=128):
    model.eval()
    dl = _loader(inputs, np.zeros(len(inputs[0])), batch_size, False, 0)
    outs = []
    for batch in dl:
        xs = [x.to(device()) for x in batch[:-1]]
        o = model(*xs).float()
        outs.append(torch.sigmoid(o.squeeze(-1)) if task == "binary" else torch.softmax(o, -1))
    return torch.cat(outs).cpu().numpy()
