"""Этап 3: графовая модель (edge-aware GINE-подобная) на фиксированной топологии."""
from __future__ import annotations

import torch
import torch.nn as nn


class GINELayer(nn.Module):
    def __init__(self, h, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, src, dst, e):
        m = torch.relu(h[:, src, :] + e.unsqueeze(0))       # [B,E,H]
        agg = torch.zeros_like(h)
        agg.index_add_(1, dst, m)                           # [B,N,H]
        out = self.mlp(h + agg)
        return self.norm(h + self.drop(out))


class GridGNN(nn.Module):
    def __init__(self, f_node, f_edge, hidden=64, layers=3, dropout=0.0, n_lines=80):
        super().__init__()
        self.n_lines = n_lines
        self.node_in = nn.Linear(f_node, hidden)
        self.edge_in = nn.Linear(f_edge, hidden)
        self.layers = nn.ModuleList([GINELayer(hidden, dropout) for _ in range(layers)])
        self.node_head = nn.Linear(hidden, 2)               # V, Delta
        self.edge_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, src, dst, eattr):
        # x: [B,N,Fn], eattr: [E,Fe] (forward E=n_lines, then reverse)
        h = self.node_in(x)
        e = self.edge_in(eattr)
        for layer in self.layers:
            h = layer(h, src, dst, e)
        vd = self.node_head(h)                              # [B,N,2]
        E = self.n_lines
        ef = h[:, src[:E], :] + h[:, dst[:E], :] + e[:E].unsqueeze(0)  # [B,E,H]
        iz = self.edge_head(ef).squeeze(-1)                # [B,E]
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=-1)
        logit = self.cls_head(pooled).squeeze(-1)
        return vd, iz, logit
