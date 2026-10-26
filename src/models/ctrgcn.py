"""CTR-GCN (Chen et al., ICCV 2021) adapted from the official implementation
https://github.com/Uason-Chen/CTR-GCN (model/ctrgcn.py).

Kept from the official code:
  * CTRGC: channel-wise topology refinement  A_c = alpha * tanh(phi(x)_u - psi(x)_v) * conv4 + A
  * unit_gcn with 3 adjacency subsets (self, inward, outward), learnable PA, residual + BN
  * MultiScale_TemporalConv (2 dilated branches + max-pool branch + 1x1 branch), TCN_GCN_unit
  * input layout (N, C, T, V, M)
Adaptations (documented in docs/architecture.md):
  * graph = verified skeleton of the dataset (Move4AS 21 rigid bodies / OpenPose BODY_25 / SMPL-24)
  * depth reduced from 10 to ``n_blocks`` (default 4) and base width 64 -> ``base`` (default 32),
    because cohorts are tiny (33 Move4AS participants) and smoke tests run 3 epochs
  * M (persons) = 1; head replaced by a task head; ``forward_features`` exposes the pooled embedding
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def build_adjacency(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Official 'spatial' strategy: [I, normalised inward, normalised outward]."""
    def norm(A):
        d = A.sum(0)
        dn = np.zeros_like(A)
        for i in range(A.shape[0]):
            if d[i] > 0:
                dn[i, i] = d[i] ** -1
        return A @ dn

    I = np.eye(num_nodes)
    inward = np.zeros((num_nodes, num_nodes))
    for c, p in edges:
        inward[p, c] = 1  # official: A[j, i] = 1 for (i, j) in edges
    outward = inward.T
    return np.stack([I, norm(inward), norm(outward)]).astype(np.float32)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class TemporalConv(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(cin, cout, (kernel_size, 1), (stride, 1), (pad, 0), (dilation, 1))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=5, stride=1, dilations=(1, 2), residual=True):
        super().__init__()
        nb = len(dilations) + 2
        bc = cout // nb
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
                                                     TemporalConv(bc, bc, kernel_size, stride, d)) for d in dilations])
        self.branches.append(nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
                                           nn.MaxPool2d((3, 1), (stride, 1), (1, 0)), nn.BatchNorm2d(bc)))
        self.branches.append(nn.Sequential(nn.Conv2d(cin, cout - bc * (nb - 1), 1, stride=(stride, 1)),
                                           nn.BatchNorm2d(cout - bc * (nb - 1))))
        if not residual:
            self.residual = lambda x: 0
        elif cin == cout and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(cin, cout, 1, stride)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1) + self.residual(x)


class CTRGC(nn.Module):
    def __init__(self, cin, cout, rel_reduction=8):
        super().__init__()
        self.rel = 8 if cin in (2, 3, 9) else max(cin // rel_reduction, 1)
        self.conv1 = nn.Conv2d(cin, self.rel, 1)
        self.conv2 = nn.Conv2d(cin, self.rel, 1)
        self.conv3 = nn.Conv2d(cin, cout, 1)
        self.conv4 = nn.Conv2d(self.rel, cout, 1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)

    def forward(self, x, A, alpha):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + A.unsqueeze(0).unsqueeze(0)
        return torch.einsum("ncuv,nctv->nctu", x1, x3)


class UnitGCN(nn.Module):
    def __init__(self, cin, cout, A, residual=True):
        super().__init__()
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList([CTRGC(cin, cout) for _ in range(self.num_subset)])
        self.down = (nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout)) if cin != cout else (lambda x: x)) \
            if residual else (lambda x: 0)
        self.PA = nn.Parameter(torch.from_numpy(A))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = sum(self.convs[i](x, self.PA[i], self.alpha) for i in range(self.num_subset))
        return self.relu(self.bn(y) + self.down(x))


class TCNGCNUnit(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True):
        super().__init__()
        self.gcn = UnitGCN(cin, cout, A)
        self.tcn = MultiScaleTemporalConv(cout, cout, 5, stride, (1, 2), residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif cin == cout and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(cin, cout, 1, stride)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class CTRGCN(nn.Module):
    """Input x: (N, T, V, C) or (N, C, T, V, M). Output logits (N, num_class)."""

    def __init__(self, num_class: int, num_point: int, edges, in_channels: int = 3, base: int = 32,
                 n_blocks: int = 4, drop_out: float = 0.2, num_person: int = 1):
        super().__init__()
        A = build_adjacency(num_point, edges)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        widths = [base] * (n_blocks // 2) + [base * 2] * (n_blocks - n_blocks // 2)
        strides = [1] * (n_blocks // 2) + [2] + [1] * (n_blocks - n_blocks // 2 - 1)
        layers, cin = [], in_channels
        for i, (w, s) in enumerate(zip(widths, strides)):
            layers.append(TCNGCNUnit(cin, w, A, stride=s, residual=i > 0))
            cin = w
        self.layers = nn.ModuleList(layers)
        self.out_dim = cin
        self.fc = nn.Linear(cin, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out)

    def _to_nctvm(self, x):
        if x.dim() == 4:  # (N,T,V,C)
            x = x.permute(0, 3, 1, 2).unsqueeze(-1)
        return x

    def forward_features(self, x):
        x = self._to_nctvm(x)
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for layer in self.layers:
            x = layer(x)
        c = x.size(1)
        x = x.view(N, M, c, -1).mean(3).mean(1)
        return self.drop_out(x)

    def forward(self, x):
        return self.fc(self.forward_features(x))
