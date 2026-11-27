"""Этап A: проверка физического модуля на измеренных V/Delta."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import physics as ph  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
EXCL = {999, 29, 33, 13, 1, 2, 3, 4, 7}


def load_sample(seed=42, n=1000):
    nodes = pd.read_csv(DATA / "train_nodes_34_100000.csv",
                        dtype={"ID_опыта": "int32", "Номер": "int16"})
    valid = nodes.groupby("ID_опыта")["regime_valid"].first()
    has999 = set(nodes.loc[nodes["Номер"] == 999, "ID_опыта"].unique())
    cands = sorted(e for e in has999 if int(valid.loc[e]) == 1)
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(cands, size=min(n, len(cands)), replace=False).tolist())
    ns = nodes[nodes["ID_опыта"].isin(sample)].copy()
    parts = []
    for ch in pd.read_csv(DATA / "train_lines_34_100000.csv", chunksize=1_000_000,
                          dtype={"ID_опыта": "int32", "Номер": "int16",
                                 "N_s": "int16", "N_e": "int16", "type": "int8"}):
        parts.append(ch[ch["ID_опыта"].isin(sample)])
    ls = pd.concat(parts, ignore_index=True)
    return ns, ls, sample


def q(a):
    a = np.asarray(a, float)
    return {"median": float(np.median(a)), "q05": float(np.quantile(a, 0.05)),
            "q95": float(np.quantile(a, 0.95)), "n": int(len(a))}


def main():
    ns, ls, sample = load_sample()
    gN = {e: g for e, g in ns.groupby("ID_опыта")}
    gL = {e: g for e, g in ls.groupby("ID_опыта")}

    rat0, rat1, slack = [], [], []
    pres, qres = [], []
    node_p_res = {}
    node_q_res = {}
    pq_nodes = [n for n in range(1, 35) if n not in EXCL]

    for e in sample:
        ne = gN[e].copy(); le = gL.get(e)
        if le is None:
            continue
        ids, ix = ph.index_nodes(ne)
        if 999 not in ix:
            continue
        V = ph.voltages(ne)
        S, bdf = ph.branch_calc(V, le, ix)
        t0 = bdf[bdf["type"] == 0]
        t1 = bdf[bdf["type"] == 1]
        rat0 += (t0["I_max"].to_numpy() / np.maximum(t0["I_s"], t0["I_t"]).to_numpy()).tolist()
        rat1 += (t1["I_max"].to_numpy() / t1["I_s"].to_numpy()).tolist()
        pg999 = float(ne.loc[ne["Номер"] == 999, "P_г"].iloc[0])
        slack.append(abs(float(S[ix[999]].real) - pg999))
        for n in pq_nodes:
            if n not in ix:
                continue
            r = ne.loc[ne["Номер"] == n].iloc[0]
            Pc, Qc = float(S[ix[n]].real), float(S[ix[n]].imag)
            pr = Pc + float(r["P_н"])
            qr = Qc + float(r["Q_н"])
            pres.append(pr); qres.append(qr)
            node_p_res.setdefault(n, []).append(pr)
            node_q_res.setdefault(n, []).append(qr)

    rmsP = float(np.sqrt(np.mean(np.square(pres))))
    rmsQ = float(np.sqrt(np.mean(np.square(qres))))
    ranking = []
    for n in sorted(node_p_res):
        sp = float(np.sqrt(np.mean(np.square(node_p_res[n]))))
        sq = float(np.sqrt(np.mean(np.square(node_q_res.get(n, [0])))))
        ranking.append((n, sp, sq, sp + sq))
    ranking.sort(key=lambda t: -t[3])
    top10 = [{"node": n, "rmsP": sp, "rmsQ": sq} for n, sp, sq, _ in ranking[:10]]

    out = {"n_exp": len(sample),
           "line_ratio_Imax_over_maxIsIt": q(rat0),
           "trafo_ratio_Imax_over_Is": q(rat1),
           "slack_abs_Pcalc_minus_Pg": q(slack),
           "pq_rms_P_MW": rmsP, "pq_rms_Q_MVAr": rmsQ,
           "top10_pq_residual": top10}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phys_a.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print("STAGE A")
    print(f"experiences={len(sample)}")
    for k in ("line_ratio_Imax_over_maxIsIt", "trafo_ratio_Imax_over_Is",
              "slack_abs_Pcalc_minus_Pg"):
        d = out[k]
        print(f"{k}: median={d['median']:.5f} q05={d['q05']:.5f} q95={d['q95']:.5f} n={d['n']}")
    print(f"PQ RMS: P={rmsP:.3f} MW, Q={rmsQ:.3f} MVAr")
    print("top-10 PQ nodes by RMS(P)+RMS(Q):")
    for t in top10:
        print(f"  node {t['node']:2d}: rmsP={t['rmsP']:8.2f} rmsQ={t['rmsQ']:8.2f}")


if __name__ == "__main__":
    main()
