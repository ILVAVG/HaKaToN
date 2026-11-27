"""Оценка шунтов на транзитных PQ-узлах: Q_calc ≈ q0 + q_sh·Vpu²."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import phys_b_check as b  # noqa: E402

TRANSIT = [5, 6, 9, 10, 11, 12, 15, 18, 19]


def main():
    ns, ls, sample = b.load_sample(n=1500)
    df = b.build_long(ns, ls, sample)
    out = {}
    for n in TRANSIT:
        g = df[df.node == n]
        if len(g) < 10:
            continue
        V = g.Vpu.to_numpy(); y = g.Qc.to_numpy()
        A = np.column_stack([np.ones_like(V), V * V])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        rms = float(np.sqrt(np.mean((y - A @ c) ** 2)))
        out[int(n)] = {"q0": float(c[0]), "q_sh": float(c[1]), "rmsQ": rms,
                       "Qc_std": float(y.std())}
    (ROOT / "reports" / "phys_shunts.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for n, d in out.items():
        print(f"node {n}: q0={d['q0']:.3f} q_sh={d['q_sh']:.3f} rmsQ={d['rmsQ']:.3f} Qc_std={d['Qc_std']:.3f}")


if __name__ == "__main__":
    main()
