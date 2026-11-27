"""Этап B: идентификация параметров (СХН, шунты, регулируемые узлы, 13/29/33)."""
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
REG_NODES = [1, 2, 3, 4, 7]
SPECIAL = [13, 29, 33]
SEED, N = 42, 2000


def load_sample(seed=SEED, n=N):
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
    return ns, pd.concat(parts, ignore_index=True), sample


def build_long(ns, ls, sample):
    gN = {e: g for e, g in ns.groupby("ID_опыта")}
    gL = {e: g for e, g in ls.groupby("ID_опыта")}
    rows = []
    for e in sample:
        ne, le = gN[e], gL.get(e)
        if le is None:
            continue
        ids, ix = ph.index_nodes(ne)
        if 999 not in ix:
            continue
        V = ph.voltages(ne)
        S, _ = ph.branch_calc(V, le, ix)
        arr = ne.to_numpy()
        for n, i in ix.items():
            r = ne.loc[ne["Номер"] == n].iloc[0]
            rows.append((e, n, float(S[i].real), float(S[i].imag), float(r["V"]),
                         float(r["U_ном"]), float(r["P_н"]), float(r["Q_н"]), float(r["P_г"])))
    df = pd.DataFrame(rows, columns=["exp", "node", "Pc", "Qc", "V", "U", "Pn", "Qn", "Pg"])
    df["Vpu"] = df["V"] / df["U"]
    return df


def fit_k(x, y):
    return float(np.sum(x * y) / np.sum(x * x))


def fit_quad(V, r):
    A = np.column_stack([np.ones_like(V), V, V * V])
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    res = r - A @ coef
    return coef, float(np.sqrt(np.mean(res ** 2))), float(np.std(res))


def main():
    ns, ls, sample = load_sample()
    df = build_long(ns, ls, sample)
    loads = df[(df.Pn > 0) & (~df.node.isin(SPECIAL + [999]))].copy()
    loads["r"] = -loads.Pc / loads.Pn
    out = {"n_exp": len(sample), "n_load_rows": int(len(loads))}

    # ---------- B1: СХН по P ----------
    x = np.log(loads.Vpu.to_numpy()); y = np.log(loads.r.to_numpy())
    k_pool = fit_k(x, y)
    pred = loads.Vpu.to_numpy() ** k_pool
    rmsP_lin = float(np.sqrt(np.mean((loads.Pc.to_numpy() + loads.Pn.to_numpy() * pred) ** 2)))
    coef, rms_quad_in_r, std_quad_in_r = fit_quad(loads.Vpu.to_numpy(), loads.r.to_numpy())
    predq = coef[0] + coef[1] * loads.Vpu.to_numpy() + coef[2] * loads.Vpu.to_numpy() ** 2
    rmsP_quad = float(np.sqrt(np.mean((loads.Pc.to_numpy() + loads.Pn.to_numpy() * predq) ** 2)))

    by_class = {}
    for u, g in loads.groupby("U"):
        xg = np.log(g.Vpu.to_numpy()); yg = np.log(g.r.to_numpy())
        kg = fit_k(xg, yg)
        cg, _, _ = fit_quad(g.Vpu.to_numpy(), g.r.to_numpy())
        predg = g.Vpu.to_numpy() ** kg
        rmsg = float(np.sqrt(np.mean((g.Pc.to_numpy() + g.Pn.to_numpy() * predg) ** 2)))
        by_class[str(int(u))] = {"k": kg, "rmsP_k": rmsg,
                                 "quad": [float(c) for c in cg]}

    per_node_k, per_node_rms = {}, {}
    for n, g in loads.groupby("node"):
        if g.Vpu.std() < 1e-3:
            continue
        xg = np.log(g.Vpu.to_numpy()); yg = np.log(g.r.to_numpy())
        kg = fit_k(xg, yg)
        cg, _, _ = fit_quad(g.Vpu.to_numpy(), g.r.to_numpy())
        predg = cg[0] + cg[1] * g.Vpu.to_numpy() + cg[2] * g.Vpu.to_numpy() ** 2
        per_node_k[int(n)] = kg
        per_node_rms[int(n)] = float(np.sqrt(np.mean((g.Pc.to_numpy() + g.Pn.to_numpy() * predg) ** 2)))

    ks = np.array(list(per_node_k.values()))
    out["B1"] = {"k_pooled": k_pool, "rmsP_k_MW": rmsP_lin,
                 "quad_c": [float(c) for c in coef], "rmsP_quad_MW": rmsP_quad,
                 "quad_std_in_r": std_quad_in_r, "by_class": by_class,
                 "k_per_node": per_node_k,
                 "k_per_node_std": float(ks.std()), "k_per_node_min": float(ks.min()),
                 "k_per_node_max": float(ks.max()),
                 "rmsP_quad_per_node_max": float(max(per_node_rms.values()))}

    # ---------- B2: СХН по Q + шунты ----------
    qrows = []
    for n, g in loads.groupby("node"):
        Qn = g.Qn.to_numpy(); V = g.Vpu.to_numpy(); yq = g.Qc.to_numpy()
        A = np.column_stack([-Qn, -Qn * V, -Qn * V * V, V * V])
        coefq, *_ = np.linalg.lstsq(A, yq, rcond=None)
        res = yq - A @ coefq
        qrows.append({"node": int(n), "d0": float(coefq[0]), "d1": float(coefq[1]),
                      "d2": float(coefq[2]), "q_sh": float(coefq[3]),
                      "rmsQ": float(np.sqrt(np.mean(res ** 2)))})
    qrows.sort(key=lambda r: -r["rmsQ"])
    out["B2"] = {"per_node": qrows,
                 "rmsQ_max": qrows[0]["rmsQ"] if qrows else None,
                 "rmsQ_median": float(np.median([r["rmsQ"] for r in qrows])) if qrows else None}

    # ---------- B3: регулируемые узлы ----------
    b3 = {}
    for n in REG_NODES:
        g = df[df.node == n]
        if len(g) == 0:
            continue
        dev = (g.V - 510.0).abs()
        below = g[g.V < 509.5]["Qc"]
        above = g[g.V > 510.5]["Qc"]
        b3[int(n)] = {"frac_off_510": float((dev > 0.5).mean()),
                      "Qc_min": float(g.Qc.min()), "Qc_q05": float(g.Qc.quantile(.05)),
                      "Qc_median": float(g.Qc.median()), "Qc_q95": float(g.Qc.quantile(.95)),
                      "Qc_max": float(g.Qc.max()),
                      "Qmax_when_below": float(below.max()) if len(below) else None,
                      "Qmin_when_above": float(above.min()) if len(above) else None}
    g999 = df[df.node == 999]
    b3["999"] = {"Pc_min": float(g999.Pc.min()), "Pc_q05": float(g999.Pc.quantile(.05)),
                 "Pc_median": float(g999.Pc.median()), "Pc_q95": float(g999.Pc.quantile(.95)),
                 "Pc_max": float(g999.Pc.max()),
                 "Qc_min": float(g999.Qc.min()), "Qc_q95": float(g999.Qc.quantile(.95)),
                 "Qc_max": float(g999.Qc.max())}
    out["B3"] = b3

    # ---------- B4: узлы 13, 29, 33 ----------
    b4 = {}
    for n in SPECIAL:
        g = df[df.node == n]
        b4[int(n)] = {"n": int(len(g)),
                      "Pc": [float(g.Pc.min()), float(g.Pc.median()), float(g.Pc.max())],
                      "Qc": [float(g.Qc.min()), float(g.Qc.median()), float(g.Qc.max())],
                      "V": [float(g.V.min()), float(g.V.median()), float(g.V.max())],
                      "Pn": [float(g.Pn.min()), float(g.Pn.median()), float(g.Pn.max())],
                      "Pg": [float(g.Pg.min()), float(g.Pg.median()), float(g.Pg.max())],
                      "corr_Pc_Pn": float(np.corrcoef(g.Pc, g.Pn)[0, 1]) if g.Pn.std() > 1e-9 else None,
                      "corr_Pc_Pg": float(np.corrcoef(g.Pc, g.Pg)[0, 1]) if g.Pg.std() > 1e-9 else None,
                      "corr_Qc_Qn": float(np.corrcoef(g.Qc, g.Qn)[0, 1]) if g.Qn.std() > 1e-9 else None,
                      "Pc_plus_Pn_median": float((g.Pc + g.Pn).median()),
                      "Pc_minus_Pg_median": float((g.Pc - g.Pg).median())}
    out["B4"] = b4

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phys_b.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    # ---------- компактный вывод ----------
    print("STAGE B (n_exp=%d, load_rows=%d)" % (out["n_exp"], out["n_load_rows"]))
    print(f"B1: k_pooled={k_pool:.4f} rmsP(k)={rmsP_lin:.3f} MW | quad c={np.round(coef,4).tolist()} rmsP(quad)={rmsP_quad:.3f} MW")
    print(f"    k per node: std={ks.std():.4f} min={ks.min():.4f} max={ks.max():.4f}; max per-node rmsP(quad)={max(per_node_rms.values()):.3f} MW")
    for u, d in by_class.items():
        print(f"    class {u}: k={d['k']:.4f} rmsP={d['rmsP_k']:.3f} MW")
    print("B2: per-node Q rms (top5):", [(r['node'], round(r['rmsQ'],2)) for r in qrows[:5]])
    print(f"    rmsQ median={out['B2']['rmsQ_median']:.3f} MVAr")
    print("B3 regulated:")
    for n in REG_NODES + [999]:
        d = b3.get(n) or b3.get(str(n))
        if d:
            print(f"    node {n}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.items() if isinstance(v, (int, float))))
    print("B4 special:")
    for n, d in b4.items():
        print(f"    node {n}: Pc={np.round(d['Pc'],1).tolist()} Qc={np.round(d['Qc'],1).tolist()} V={np.round(d['V'],2).tolist()} corr(Pc,Pn)={d['corr_Pc_Pn']}")


if __name__ == "__main__":
    main()
