"""Этап B4-probe: проверка гипотез о роли узлов 13, 29, 33."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import phys_b_check as b  # noqa: E402


def main():
    ns, ls, sample = b.load_sample(n=800)
    df = b.build_long(ns, ls, sample)
    f = lambda V: 0.8234770208 - 0.2854556639 * V + 0.4620430672 * V * V

    for n in (13, 29, 33):
        g = df[df.node == n]
        z13 = g.Pc + g.Pn                    # гипотеза: const генерация (13)
        z = g.Pc + g.Pn * f(g.Vpu)           # гипотеза: Pc = C - Pn*f(V)
        # Q модель как в B2
        Qn = g.Qn.to_numpy(); V = g.Vpu.to_numpy(); yq = g.Qc.to_numpy()
        A = np.column_stack([-Qn, -Qn * V, -Qn * V * V, V * V])
        cq, *_ = np.linalg.lstsq(A, yq, rcond=None)
        rq = np.sqrt(np.mean((yq - A @ cq) ** 2))
        print(f"node {n}: n={len(g)}")
        print(f"   Pc+Pn:      median={z13.median():8.2f} std={z13.std():7.3f}")
        print(f"   Pc+Pn*f(V): median={z.median():8.2f} std={z.std():7.3f}")
        print(f"   Q model c={np.round(cq,3).tolist()} rmsQ={rq:.3f}")


if __name__ == "__main__":
    main()
