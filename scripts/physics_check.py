"""Этап 2: проверка, воспроизводятся ли V/Delta моделью сети.

Идея: для валидного опыта берём ИЗМЕРЕННЫЕ V и Delta, строим Y-матрицу из
R/X/B/Kt и считаем комплексы мощностей S_i = V_i * conj(sum_j Y_ij V_j).
Если модель сети верна, для PQ-узлов S_i должны совпасть с заданными
(-P_н + P_г) + j(-Q_н). Меряем невязку. Так проверяем конвенции без NR.

Запуск: python physics_check.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grid_lib as gl  # noqa: E402

S_BASE = 100.0  # МВА (в итоге сокращается, но держим для pu)


def build_ybus(nodes_exp, lines_geo, U, k_b=0.0, tap_mode="from", kt_mode="ratio"):
    """nodes_exp: dict num->(V_kV, Delta_deg). lines_geo: DataFrame геометрии.
    k_b: b_pu = B * k_b (для type=0 и type=1 одинаково). tap_mode: from/to.
    kt_mode: 'ratio' (tau=Kt/(Ue/Us)) или 'one' (tau=1)."""
    nums = sorted(U.index)
    idx = {n: i for i, n in enumerate(nums)}
    nb = len(nums)
    Y = np.zeros((nb, nb), complex)
    for _, l in lines_geo.iterrows():
        i, j = int(l["N_s"]), int(l["N_e"])
        if i not in idx or j not in idx:
            continue
        Us, Ue = float(U[i]), float(U[j])
        zb = Us ** 2 / S_BASE
        r = float(l["R"]) / zb
        x = float(l["X"]) / zb
        y = 1.0 / complex(r, x)
        if int(l["type"]) == 0:
            Y[idx[i], idx[i]] += y
            Y[idx[j], idx[j]] += y
            Y[idx[i], idx[j]] -= y
            Y[idx[j], idx[i]] -= y
            bpu = float(l["B"]) * k_b
            Y[idx[i], idx[i]] += 1j * bpu / 2
            Y[idx[j], idx[j]] += 1j * bpu / 2
        else:
            tau = (float(l["Kt"]) / (Ue / Us)) if kt_mode == "ratio" else 1.0
            if tap_mode == "from":
                Y[idx[i], idx[i]] += y / tau ** 2
                Y[idx[i], idx[j]] += -y / tau
                Y[idx[j], idx[i]] += -y / tau
                Y[idx[j], idx[j]] += y
            else:
                Y[idx[i], idx[i]] += y
                Y[idx[i], idx[j]] += -y / tau
                Y[idx[j], idx[i]] += -y / tau
                Y[idx[j], idx[j]] += y / tau ** 2
            bpu = float(l["B"]) * k_b
            Y[idx[i], idx[i]] += 1j * bpu  # на стороне from
    return Y, nums, idx


def residual(nodes_exp, lines_geo, U, k_b, tap_mode, kt_mode, pq_nodes):
    Y, nums, idx = build_ybus(nodes_exp, lines_geo, U, k_b, tap_mode, kt_mode)
    v = np.array([nodes_exp[n][0] / float(U[n]) * np.exp(1j * np.deg2rad(nodes_exp[n][1]))
                  for n in nums])
    S = v * np.conj(Y @ v) * S_BASE  # МВА
    res_p, res_q = [], []
    for n in pq_nodes:
        pe = nodes_exp[n]
        Pspec = pe[2] - pe[3]   # P_г - P_н
        Qspec = -pe[4]          # -Q_н
        res_p.append(S[idx[n]].real - Pspec)
        res_q.append(S[idx[n]].imag - Qspec)
    res_p = np.array(res_p); res_q = np.array(res_q)
    return {"rmse_p": float(np.sqrt(np.mean(res_p ** 2))),
            "rmse_q": float(np.sqrt(np.mean(res_q ** 2))),
            "max_p": float(np.max(np.abs(res_p))),
            "max_q": float(np.max(np.abs(res_q))),
            "rmse_all": float(np.sqrt(np.mean(np.concatenate([res_p, res_q]) ** 2)))}


def main():
    nodes, lines = gl.load_data()
    lines_geo = lines.drop_duplicates("Номер").copy()
    U = nodes.drop_duplicates("Номер").set_index("Номер")["U_ном"].astype(float)
    valid = nodes[nodes["regime_valid"] == 1]
    # геометрия линии в порядке номеров
    lines_geo = lines_geo.sort_values("Номер")

    test_exps = [1, 2, 4, 5, 6]
    pq_nodes = [n for n in U.index if n not in (999, 29, 33)]

    # подбор k_b и конвенций
    k_grid = [0.0] + list(np.logspace(-9, -1, 17))
    results = []
    for exp in test_exps:
        ne = valid[valid["ID_опыта"] == exp]
        nodes_exp = {int(r["Номер"]): (float(r["V"]), float(r["Delta"]),
                                       float(r["P_г"]), float(r["P_н"]), float(r["Q_н"]))
                     for _, r in ne.iterrows()}
        for tap_mode, kt_mode in itertools.product(["from", "to"], ["ratio", "one"]):
            for kb in k_grid:
                r = residual(nodes_exp, lines_geo, U, kb, tap_mode, kt_mode, pq_nodes)
                results.append({"exp": exp, "tap": tap_mode, "kt": kt_mode, "k_b": kb, **r})
    df = pd.DataFrame(results)
    # агрегируем по (tap,kt,kb)
    agg = df.groupby(["tap", "kt", "k_b"]).agg(rmse_all=("rmse_all", "mean"),
                                               rmse_p=("rmse_p", "mean"),
                                               rmse_q=("rmse_q", "mean")).reset_index()
    agg = agg.sort_values("rmse_all")
    print("TOP-10 conventions by mean rmse_all (MVA):")
    print(agg.head(10).to_string(index=False))
    print("\nB=0 baseline:")
    print(agg[agg["k_b"] == 0.0].to_string(index=False))

    best = agg.iloc[0].to_dict()
    out = {"best": best, "top10": agg.head(10).to_dict(orient="records"),
           "b0": agg[agg["k_b"] == 0.0].to_dict(orient="records")}
    with open(gl.REPORTS_DIR / "physics_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nSaved physics_check.json")


if __name__ == "__main__":
    main()
