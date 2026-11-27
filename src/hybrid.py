"""Гибридные признаки: DC-приближение углов по Y-матрице (без GPU/итераций NR)."""
from __future__ import annotations

import numpy as np
import pandas as pd

S_BASE = 100.0


def build_dc_features(frame, lines, umap):
    node_numbers = [int(c) for c in frame["Pn"].columns]
    nidx = {n: i for i, n in enumerate(node_numbers)}
    N = len(node_numbers)
    B = np.zeros((N, N), float)
    geo = lines.drop_duplicates("Номер")
    for r in geo.itertuples(index=False):
        i, j = nidx[int(r.N_s)], nidx[int(r.N_e)]
        zb = (umap[int(r.N_s)] ** 2) / S_BASE
        z = complex(r.R / zb, r.X / zb)
        b = (1.0 / z).imag          # series susceptance (pu), приближение
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b
    slack = nidx.get(999)
    nons = [i for i in range(N) if i != slack]
    Bred = B[np.ix_(nons, nons)]
    inv = np.linalg.pinv(Bred)

    Pn = frame["Pn"].reindex(columns=node_numbers).fillna(0.0).to_numpy(float)
    Pg = frame["Pg"].reindex(columns=node_numbers).fillna(0.0).to_numpy(float)
    gen_mask = np.zeros(N)
    for g in (29, 33):
        if g in nidx:
            gen_mask[nidx[g]] = 1.0
    Pnet = (Pg * gen_mask - Pn) / S_BASE
    theta_red = Pnet[:, nons] @ inv.T
    theta = np.zeros((len(Pnet), N))
    theta[:, nons] = theta_red
    deg = np.degrees(theta)
    return pd.DataFrame(deg, index=frame["exp"], columns=[f"dc_{n}" for n in node_numbers])
