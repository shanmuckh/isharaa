"""Sequence classifier over per-frame landmark features."""

import torch
from torch import nn


class SignClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, dim=192, layers=3, heads=4, seq_len=64, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU())
        # local motion patterns first, then global context
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2, groups=4), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=2, groups=4), nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, seq_len, dim))
        block = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Dropout(dropout), nn.Linear(dim * 2, num_classes))

    def forward(self, x):                      # x: (B, T, F)
        h = self.embed(x)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        h = self.encoder(h + self.pos[:, : h.shape[1]])
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=-1))
