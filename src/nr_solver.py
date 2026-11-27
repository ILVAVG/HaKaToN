"""Решатель установившегося режима (Ньютон-Рафсон через scipy.root).

Модель (по результатам этапов A/B):
- слэк 999: V=510 кВ, Delta=0;
- PV-узлы 1,2,3,4,7: V=510; узел 13: V=230 и фикс. генерация 600 МВт;
- узлы 29,33: фикс. генерация 300 МВт, PQ;
- PQ-нагрузки: СХН P = Pн * f(Vpu), Q по поузловой модели B2;
- Q-лимиты PV: 1,3 -> 500 МВар (переход PV->PQ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import root

import physics as ph

# СХН по P (пул, этап B1)
F_C = (0.8234770208, -0.2854556639, 0.4620430672)
# PV-узлы и уставки напряжения (кВ)
PV_V = {1: 510.0, 2: 510.0, 3: 510.0, 4: 510.0, 7: 510.0, 13: 230.0}
PV_PGEN = {13: 600.0}
# Q-лимиты PV (МВар)
QMAX = {1: 500.0, 3: 500.0}
QMIN = {}
# фиксированная генерация на PQ-узлах
SPECIAL_PGEN = {29: 300.0, 33: 300.0}
# Q-модель для 29/33 (фит на 800 опытах, этап B4-probe): [d0,d1,d2,q_sh]
QCOEFF_EXTRA = {29: (3.879, -12.038, 9.758, 299.171),
                33: (4.510, -12.928, 10.041, 309.254)}


def _f(vpu):
    return F_C[0] + F_C[1] * vpu + F_C[2] * vpu * vpu


def solve_pf(nodes_e, lines_e, qcoeff=None, max_outer=6, flat_start=True, init_vpu=1.0):
    ids, ix = ph.index_nodes(nodes_e)
    if 999 not in ix:
        return {"ok": False, "reason": "no_slack"}
    N = len(ids)
    U = nodes_e.set_index("Номер")["U_ном"].reindex(ids).to_numpy(float)
    Pn = nodes_e.set_index("Номер")["P_н"].reindex(ids).to_numpy(float)
    Qn = nodes_e.set_index("Номер")["Q_н"].reindex(ids).to_numpy(float)
    qcoeff = qcoeff or {}

    slack_i = ix[999]
    order = [n for n in ids if n != 999]
    # pq set: не PV
    pq = set(n for n in order if n not in PV_V)
    switched = {}

    def vset_of(n):
        return PV_V[n]

    # начальное приближение
    vpu = np.full(N, float(init_vpu)); ang = np.zeros(N)
    vpu[slack_i] = 510.0 / U[slack_i]
    for n in order:
        if n in PV_V:
            vpu[ix[n]] = PV_V[n] / U[ix[n]]

    def pack(vpu, ang):
        x = [ang[ix[n]] for n in order] + [vpu[ix[n]] for n in order if n in pq]
        return np.array(x, float)

    def unpack(x):
        vp = vpu.copy(); an = ang.copy(); an[slack_i] = 0.0; vp[slack_i] = 510.0 / U[slack_i]
        k = 0
        for n in order:
            an[ix[n]] = x[k]; k += 1
        for n in order:
            if n in pq:
                vp[ix[n]] = x[k]; k += 1
        return vp, an

    def pspec(n, vpi):
        i = ix[n]
        if n in SPECIAL_PGEN:
            return SPECIAL_PGEN[n] - Pn[i] * _f(vpi)
        if n in PV_V:
            vfixed = PV_V[n] / U[i]
            gen = PV_PGEN.get(n, 0.0)
            return gen - Pn[i] * _f(vfixed)
        return -Pn[i] * _f(vpi)

    def qspec(n, vpi):
        if n in switched:
            return switched[n]
        if n in qcoeff:
            d0, d1, d2, qsh = qcoeff[n]
            return -Qn[ix[n]] * (d0 + d1 * vpi + d2 * vpi * vpi) + qsh * vpi * vpi
        return 0.0

    x = pack(vpu, ang)
    for _ in range(max_outer):
        def fun(xx):
            vp, an = unpack(xx)
            V = U * vp * np.exp(1j * an)
            S, _ = ph.branch_calc(V, lines_e, ix)
            res = []
            for n in order:
                i = ix[n]
                res.append(S[i].real - pspec(n, vp[i]))
                if n in pq:
                    res.append(S[i].imag - qspec(n, vp[i]))
            return np.array(res, float)

        sol = root(fun, x, method="hybr", options={"maxfev": 4000})
        x = sol.x
        vp, an = unpack(x)
        V = U * vp * np.exp(1j * an)
        S, _ = ph.branch_calc(V, lines_e, ix)
        new_switch = {}
        for n in order:
            if n in PV_V and n not in switched:
                q = float(S[ix[n]].imag)
                qmax = QMAX.get(n, np.inf); qmin = QMIN.get(n, -np.inf)
                if q > qmax + 1e-6:
                    new_switch[n] = qmax
                elif q < qmin - 1e-6:
                    new_switch[n] = qmin
        if not new_switch:
            break
        for n, qlim in new_switch.items():
            switched[n] = qlim
            pq.add(n)
            vp[ix[n]] = PV_V[n] / U[ix[n]]
        x = pack(vp, an)

    vp, an = unpack(x)
    V = U * vp * np.exp(1j * an)
    S, bdf = ph.branch_calc(V, lines_e, ix)
    ok = bool(sol.success) and np.all(np.isfinite(vp)) and np.all(vp > 0.5)
    # I_max предсказанный из решённых токов
    if len(bdf):
        bdf = bdf.copy()
        bdf["Imax_pred"] = np.where(bdf["type"] == 0,
                                    np.maximum(bdf["I_s"], bdf["I_t"]), bdf["I_s"])
    return {"ok": ok, "success": bool(sol.success), "nfev": int(sol.nfev),
            "V_kV": U * vp, "Vpu": vp, "Delta_deg": np.degrees(an),
            "I_df": bdf, "ids": ids, "ix": ix,
            "switched": switched, "message": str(sol.message)}
