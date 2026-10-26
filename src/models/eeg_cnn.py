"""Lightweight EEG 1D/2D CNN (EEGNet-style: temporal conv -> depthwise spatial conv -> separable conv).

Input: (N, n_channels, n_samples) per-trial z-scored EEG at 125 Hz (4-s execution window = 500 samples).
~3k parameters with defaults — deliberately tiny for a 34-participant cohort.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNetLite(nn.Module):
    def __init__(self, n_channels: int, num_class: int = 1, F1: int = 8, D: int = 2, F2: int = 16,
                 kern: int = 64, drop: float = 0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        self.out_dim = F2
        self.fc = nn.Linear(F2, num_class)

    def forward_features(self, x):
        return self.block(x.unsqueeze(1)).mean(dim=(2, 3))

    def forward(self, x):
        return self.fc(self.forward_features(x))


class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, num_class: int = 1, hidden=(64, 32), drop: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)]
            d = h
        self.body = nn.Sequential(*layers)
        self.out_dim = d
        self.fc = nn.Linear(d, num_class)

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.fc(self.forward_features(x))
