"""PoseFormerV2-style classifier (adapter).

Reference: Zhao et al., "PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human
Pose Estimation", CVPR 2023, official code https://github.com/QitaoZhao/PoseFormerV2.

The official model is a 2D->3D *lifting* network for 17 Human3.6M joints that predicts the 3D pose of
the centre frame; its checkpoints are trained on Human3.6M and are NOT compatible with the
Move4AS (21 rigid bodies) / OpenPose BODY_25 / SMPL-24 layouts or with sequence classification.
We therefore re-implement its two defining ideas and use them as a sequence encoder:
  1. Spatial transformer over joint tokens within each frame (as in PoseFormer/PoseFormerV2).
  2. Frequency-domain temporal tokens: DCT along time of the per-frame embeddings, keep the
     first ``n_coeff`` low-frequency coefficients, plus ``n_time`` time-domain tokens (uniformly
     sampled frames, the V2 "few-frame" branch); a joint temporal transformer fuses both token sets
     ("time-frequency feature fusion").
Changes vs. official: no lifting head / centre-frame regression, mean-pooled classification
head, smaller width/depth. This is an adaptation, not an exact reproduction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def dct_matrix(n: int, device=None) -> torch.Tensor:
    k = torch.arange(n, device=device).float()
    m = torch.cos(math.pi / n * (k[None, :] + 0.5) * k[:, None])
    m[0] *= 1 / math.sqrt(2)
    return m * math.sqrt(2 / n)


class PoseFormerV2Classifier(nn.Module):
    def __init__(self, num_class: int, num_joints: int, in_channels: int = 3, dim_joint: int = 32,
                 dim_frame: int = 128, depth_s: int = 2, depth_t: int = 2, heads: int = 4,
                 n_coeff: int = 9, n_time: int = 9, drop: float = 0.1, max_frames: int = 512):
        super().__init__()
        self.n_coeff, self.n_time = n_coeff, n_time
        self.joint_embed = nn.Linear(in_channels, dim_joint)
        self.joint_pos = nn.Parameter(torch.zeros(1, num_joints, dim_joint))
        enc = lambda d: nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(enc(dim_joint), depth_s)
        self.frame_proj = nn.Linear(num_joints * dim_joint, dim_frame)
        self.freq_pos = nn.Parameter(torch.zeros(1, n_coeff, dim_frame))
        self.time_pos = nn.Parameter(torch.zeros(1, n_time, dim_frame))
        self.type_emb = nn.Parameter(torch.zeros(2, dim_frame))
        self.temporal = nn.TransformerEncoder(enc(dim_frame), depth_t)
        self.norm = nn.LayerNorm(dim_frame)
        self.out_dim = dim_frame
        self.fc = nn.Linear(dim_frame, num_class)
        nn.init.trunc_normal_(self.joint_pos, std=0.02)
        nn.init.trunc_normal_(self.freq_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)

    def forward_features(self, x):  # x (N, T, V, C)
        N, T, V, C = x.shape
        z = self.joint_embed(x.reshape(N * T, V, C)) + self.joint_pos
        z = self.spatial(z).reshape(N, T, -1)
        f = self.frame_proj(z)                              # (N, T, D)
        D = dct_matrix(T, x.device)[: self.n_coeff]         # (K, T)
        freq = torch.einsum("kt,ntd->nkd", D, f) + self.freq_pos + self.type_emb[0]
        idx = torch.linspace(0, T - 1, self.n_time, device=x.device).round().long()
        tim = f[:, idx] + self.time_pos + self.type_emb[1]
        h = self.temporal(torch.cat([freq, tim], 1))
        return self.norm(h.mean(1))

    def forward(self, x):
        return self.fc(self.forward_features(x))
