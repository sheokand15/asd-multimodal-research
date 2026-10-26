"""Exactly three fusion strategies: early (input-level), intermediate (feature-level), late (decision-level)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------ 1. Early fusion

class EarlyFusion:
    """Standardise each modality's compact feature block with training-fold statistics, concatenate,
    and fit one classifier on the joint representation."""

    def __init__(self, C: float = 0.05, seed: int = 42):
        self.scalers: list[StandardScaler] = []
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)

    def _join(self, blocks, fit: bool):
        if fit:
            self.scalers = [StandardScaler().fit(b) for b in blocks]
        return np.concatenate([s.transform(b) for s, b in zip(self.scalers, blocks)], axis=1)

    def fit(self, blocks: list[np.ndarray], y):
        self.clf.fit(self._join(blocks, True), y)
        return self

    def predict_proba(self, blocks):
        return self.clf.predict_proba(self._join(blocks, False))[:, 1]


def unimodal_classifier(C: float = 0.05, seed: int = 42) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed))])


# ------------------------------------------------------------------ 2. Intermediate fusion

class IntermediateFusion(nn.Module):
    """Separate modality encoders -> linear projection to a shared size -> concatenation -> head."""

    def __init__(self, encoders: dict[str, nn.Module], proj_dim: int = 32, num_class: int = 1, drop: float = 0.3):
        super().__init__()
        self.names = list(encoders)
        self.encoders = nn.ModuleDict(encoders)
        self.proj = nn.ModuleDict({k: nn.Sequential(nn.Linear(e.out_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
                                   for k, e in encoders.items()})
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(proj_dim * len(encoders), num_class))

    def forward(self, *inputs):
        z = [self.proj[k](self.encoders[k].forward_features(x)) for k, x in zip(self.names, inputs)]
        return self.head(torch.cat(z, 1))


# ------------------------------------------------------------------ 3. Late fusion

class LateFusion:
    """Weighted average of modality-specific calibrated probabilities.
    The weight is chosen on *training-fold validation* predictions (minimum log loss); it is never
    tuned on held-out participants."""

    def __init__(self, grid=np.linspace(0, 1, 11)):
        self.grid = grid
        self.w = None
        self.val_losses = None

    def fit(self, probs: list[np.ndarray], y):
        assert len(probs) == 2, "two-modality late fusion"
        y = np.asarray(y)
        self.val_losses = [log_loss(y, np.clip(w * probs[0] + (1 - w) * probs[1], 1e-6, 1 - 1e-6), labels=[0, 1])
                           for w in self.grid]
        self.w = float(self.grid[int(np.argmin(self.val_losses))])
        return self

    def predict_proba(self, probs):
        return self.w * probs[0] + (1 - self.w) * probs[1]
