"""Skeleton Masked Autoencoder (MAE-style, He et al. 2022, applied to skeleton sequences).

Tokenisation: temporal patches of ``patch_len`` frames x all joints x C coords -> one token.
A random ``mask_ratio`` of tokens is hidden; the encoder sees only visible tokens; a light decoder
reconstructs the coordinates of the masked tokens.  The loss is computed only on masked tokens and
only on *valid* coordinates (``valid`` mask excludes missing keypoints and padded frames).

STH-MAE was considered but not used: no verified official implementation compatible with these
joint layouts was available offline, so this generic MAE is used instead (documented).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SkeletonMAE(nn.Module):
    def __init__(self, num_joints: int, in_channels: int, seq_len: int, patch_len: int = 4, embed_dim: int = 128,
                 depth: int = 4, heads: int = 4, dec_dim: int = 64, dec_depth: int = 2, mask_ratio: float = 0.5,
                 drop: float = 0.1):
        super().__init__()
        assert seq_len % patch_len == 0
        self.V, self.C, self.P = num_joints, in_channels, patch_len
        self.n_tok = seq_len // patch_len
        self.mask_ratio = mask_ratio
        tok_dim = patch_len * num_joints * in_channels
        self.embed = nn.Linear(tok_dim, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tok, embed_dim))
        layer = lambda d: nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(embed_dim), depth)
        self.enc_norm = nn.LayerNorm(embed_dim)
        self.dec_embed = nn.Linear(embed_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.n_tok, dec_dim))
        self.decoder = nn.TransformerEncoder(layer(dec_dim), dec_depth)
        self.dec_head = nn.Linear(dec_dim, tok_dim)
        self.out_dim = embed_dim
        for p in (self.pos, self.mask_token, self.dec_pos):
            nn.init.trunc_normal_(p, std=0.02)

    def patchify(self, x):  # (N,T,V,C) -> (N, L, P*V*C)
        N, T, V, C = x.shape
        return x.reshape(N, T // self.P, self.P * V * C)

    def forward_features(self, x):
        """Encoder on all tokens (no masking), mean-pooled -> (N, embed_dim)."""
        h = self.embed(self.patchify(x)) + self.pos
        return self.enc_norm(self.encoder(h)).mean(1)

    def forward(self, x, valid=None):
        """Returns (loss, reconstruction (N,T,V,C), scored mask (N,T,V,C) bool)."""
        N = x.shape[0]
        tok = self.patchify(x)
        L = tok.shape[1]
        n_keep = max(1, int(round(L * (1 - self.mask_ratio))))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = noise.argsort(1)
        ids_restore = ids_shuffle.argsort(1)
        keep = ids_shuffle[:, :n_keep]
        h = self.embed(tok) + self.pos
        h = torch.gather(h, 1, keep.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        h = self.enc_norm(self.encoder(h))
        d = self.dec_embed(h)
        d = torch.cat([d, self.mask_token.expand(N, L - n_keep, -1)], 1)
        d = torch.gather(d, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d.shape[-1])) + self.dec_pos
        rec = self.dec_head(self.decoder(d))                       # (N, L, tok_dim)
        tok_mask = torch.ones(N, L, device=x.device)
        tok_mask[:, :n_keep] = 0
        tok_mask = torch.gather(tok_mask, 1, ids_restore)          # 1 = masked
        m = tok_mask.unsqueeze(-1).expand_as(rec).bool()
        if valid is not None:
            m = m & self.patchify(valid.float()).bool()
        loss = ((rec - tok) ** 2)[m].mean() if m.any() else rec.sum() * 0
        return loss, rec.reshape(x.shape), m.reshape(x.shape)


class MAEClassifier(nn.Module):
    """Pretrained MAE encoder + linear head (fine-tuned end-to-end)."""

    def __init__(self, mae: SkeletonMAE, num_class: int):
        super().__init__()
        self.mae = mae
        self.out_dim = mae.out_dim
        self.fc = nn.Linear(mae.out_dim, num_class)

    def forward_features(self, x):
        return self.mae.forward_features(x)

    def forward(self, x):
        return self.fc(self.forward_features(x))
