"""Этап 1: быстрые победы на sklearn + честная диагностика. Пишет stage1_results.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grid_lib as gl  # noqa: E402
import grid_mlp  # noqa: E402


def eval_regression(pn, pl, nodes, lines, exp_set, label=""):
    tn = nodes[nodes["ID_опыта"].isin(exp_set)]
    tl = lines[lines["ID_опыта"].isin(exp_set)]
    mn = tn[["ID_опыта", "Номер", "U_ном", "P_н", "V", "Delta", "regime_valid"]].merge(
        pn, on=["ID_опыта", "Номер"], suffixes=("_true", "_pred"))
    ml = tl[["ID_опыта", "Номер", "I_max", "I_zarg", "I_доп"]].merge(
        pl, on=["ID_опыта", "Номер"], suffixes=("_true", "_pred"))
    val = mn[mn["regime_valid"] == 1]
    vl = ml[ml["ID_опыта"].isin(set(val["ID_опыта"]))]
    out = {}
    # V all nodes / varying only
    out["V_all_nodes"] = gl.reg_metrics(val["V_true"], val["V_pred"])
    varying = val.groupby("Номер")["V_true"].std()
    varying = varying[varying > 1e-6].index
    vv = val[val["Номер"].isin(varying)]
    out["V_varying_nodes"] = gl.reg_metrics(vv["V_true"], vv["V_pred"])
    out["V_by_class"] = {}
    for u, g in val.groupby("U_ном"):
        out["V_by_class"][str(int(u))] = {
            **gl.reg_metrics(g["V_true"], g["V_pred"]),
            "rel_MAE_pct": float(mean_absolute_error(g["V_true"], g["V_pred"]) / g["V_true"].mean() * 100)}
    out["Delta"] = gl.reg_metrics(val["Delta_true"], val["Delta_pred"])
    out["Delta_by_class"] = {}
    for u, g in val.groupby("U_ном"):
        out["Delta_by_class"][str(int(u))] = gl.reg_metrics(g["Delta_true"], g["Delta_pred"])
    out["Izarg"] = gl.reg_metrics(vl["I_zarg"], vl["Izarg, %"])
    out["Imax_nominal"] = gl.reg_metrics(vl["I_max"], vl["Imax, A"])
    if vl["I_доп"].notna().all():
        imax_given = vl["Izarg, %"] * vl["I_доп"] / 100.0
        out["Imax_given_idop"] = gl.reg_metrics(vl["I_max"], imax_given)

    # per node V R2 (varying)
    per = []
    for nd, g in val.groupby("Номер"):
        if g["V_true"].std() > 1e-6:
            per.append((int(nd), float(r2_score(g["V_true"], g["V_pred"])),
                        float(mean_absolute_error(g["V_true"], g["V_pred"])),
                        float(g["V_true"].std())))
    per.sort(key=lambda t: t[1])
    out["V_worst_nodes"] = [{"node": n, "R2": r, "MAE": m, "std": s} for n, r, m, s in per[:6]]
    r2s = np.array([p[1] for p in per])
    out["V_uniform_R2_nodes"] = float(r2s.mean())
    out["V_median_R2_nodes"] = float(np.median(r2s))
    out["V_nodes_R2_below_0.99"] = int((r2s < 0.99).sum())

    # load quintile V MAE
    agg = val.assign(ae=np.abs(val["V_true"] - val["V_pred"])).groupby("ID_опыта")["ae"].mean()
    load = val.groupby("ID_опыта")["P_н"].sum() if "P_н" in val else None
    if load is not None:
        j = pd.concat([load.rename("sumPn"), agg.rename("mVerr")], axis=1).dropna()
        q = pd.qcut(j["sumPn"], 5, labels=False, duplicates="drop")
        out["V_MAE_by_load_quintile"] = [
            {"q": int(k), "sumPn": float(j.loc[q == k, "sumPn"].mean()),
             "MAE": float(j.loc[q == k, "mVerr"].mean())} for k in sorted(q.unique())]

    # parallel spread
    groups = gl.parallel_groups(lines)
    inv = {}
    for ln, g in groups.items():
        inv.setdefault(g, []).append(ln)
    dup = {g: m for g, m in inv.items() if len(m) > 1}
    mp = tl[["ID_опыта", "Номер"]].merge(pl, on=["ID_опыта", "Номер"])
    spreads = []
    for g, members in dup.items():
        sub = mp[mp["Номер"].isin(members)]
        sp = sub.groupby("ID_опыта")["Izarg, %"].agg(lambda s: s.max() - s.min())
        spreads.append(sp)
    if spreads:
        sp = pd.concat(spreads)
        out["parallel_spread"] = {"groups": len(dup), "mean": float(sp.mean()),
                                  "median": float(sp.median()), "max": float(sp.max()),
                                  "share_gt_0.05pct": float((sp > 0.05).mean())}
    return out


def main():
    t0 = time.time()
    nodes, lines = gl.load_data()
    frame = gl.build_frame(nodes, lines)
    exp, regime = frame["exp"], frame["regime"]
    tr, va, te = gl.split_experiences(exp, seed=42, scheme="random")
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    print("Обучение AFTER bundle...", flush=True)
    bundle = grid_mlp.fit_bundle(frame, tr, config=grid_mlp.AFTER, seed=42, verbose=True)

    # ---- threshold на val ----
    X = frame["X"].reindex(columns=bundle["features"], fill_value=0.0)
    clf = bundle["clf"]
    def proba(ids):
        return clf.predict_proba(X.loc[ids])[:, list(clf.classes_).index(1)]
    pva = proba(va)
    yva = regime.loc[va].to_numpy()
    t_best, f_best = gl.best_threshold(yva, pva, "f1macro")
    bundle["threshold"] = t_best
    print(f"threshold tuned on val: {t_best:.2f} (val F1={f_best:.4f})", flush=True)

    # ---- метрики флага на test ----
    pte = proba(te)
    yte = regime.loc[te].to_numpy()
    flag05 = gl.clf_metrics(yte, pte, 0.5)
    flagT = gl.clf_metrics(yte, pte, t_best)

    # ---- регрессия/пост-обработка on test ----
    tn = nodes[nodes["ID_опыта"].isin(set(te))]
    tl = lines[lines["ID_опыта"].isin(set(te))]
    res = {"split": {"train": len(tr), "val": len(va), "test": len(te)},
           "flag_test_at_0.5": flag05, "flag_test_at_tuned": flagT,
           "threshold_val": t_best}

    for parallel in (False, True):
        b = dict(bundle); b["parallel_avg"] = parallel
        pn, pl, reg, _ = grid_mlp.predict_bundle(tn, tl, b, zero_invalid=False)
        key = "parallel_avg" if parallel else "no_parallel_avg"
        res[key] = eval_regression(pn, pl, nodes, lines, set(te))

    res["elapsed_sec"] = round(time.time() - t0, 1)
    with open(gl.REPORTS_DIR / "stage1_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: res[k] for k in ["flag_test_at_0.5", "flag_test_at_tuned"]},
                     ensure_ascii=False, indent=2))
    for key in ("no_parallel_avg", "parallel_avg"):
        r = res[key]
        print(f"\n[{key}] V varying R2={r['V_varying_nodes']['R2']:.4f} MAE={r['V_varying_nodes']['MAE']:.4f} "
              f"| V_uniform_R2_nodes={r['V_uniform_R2_nodes']:.4f} | Delta R2={r['Delta']['R2']:.4f} "
              f"| Izarg MAE={r['Izarg']['MAE']:.4f} R2={r['Izarg']['R2']:.4f} "
              f"| Imax MAE={r['Imax_nominal']['MAE']:.4f}")
        if "parallel_spread" in r:
            ps = r["parallel_spread"]
            print(f"   parallel spread: mean={ps['mean']:.4f} median={ps['median']:.4f} max={ps['max']:.4f} "
                  f"share>0.05%={ps['share_gt_0.05pct']*100:.2f}%")
    print(f"\nГотово за {res['elapsed_sec']} c -> stage1_results.json")


if __name__ == "__main__":
    main()
