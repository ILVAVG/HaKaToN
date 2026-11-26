"""Этап 4: корректная валидация.

- 3 случайных GroupShuffleSplit по ID_опыта (seed 42/7/2024)
- блочный сплит по ID (тест — верхний хвост ID)
- экстраполяция: обучение на лёгких 70% по sumPn, тест на тяжёлых 15%
Пишет stage4_results.json инкрементально.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_lib as gl  # noqa: E402
import grid_mlp  # noqa: E402
import exp_stage1 as e1  # noqa: E402


def flag_probs(bundle, X, ids):
    clf = bundle["clf"]
    Xf = X.reindex(columns=bundle["features"], fill_value=0.0).loc[ids]
    return clf.predict_proba(Xf)[:, list(clf.classes_).index(1)]


def run_one(frame, nodes, lines, tr, va, te, label, seed, results):
    t0 = time.time()
    print(f"  fit {label} seed={seed} ...", flush=True)
    b = grid_mlp.fit_bundle(frame, tr, config=grid_mlp.MEDIUM, seed=seed,
                            parallel_avg=True, snap=True, verbose=False)
    X = frame["X"]
    pv = flag_probs(b, X, va)
    yv = frame["regime"].loc[va].to_numpy()
    if len(np.unique(yv)) < 2:
        tbest = 0.5
    else:
        tbest, _ = gl.best_threshold(yv, pv, "f1macro")
    b["threshold"] = tbest
    pt = flag_probs(b, X, te)
    yt = frame["regime"].loc[te].to_numpy()
    pred = (pt >= tbest).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    classes = np.unique(yt)
    fm = {"accuracy": float(accuracy_score(yt, pred)), "threshold": tbest,
          "confusion": [[int(((yt == 0) & (pred == 0)).sum()), int(((yt == 0) & (pred == 1)).sum())],
                        [int(((yt == 1) & (pred == 0)).sum()), int(((yt == 1) & (pred == 1)).sum())]],
          "positive_rate": float(yt.mean())}
    if len(classes) >= 2:
        fm["F1_macro"] = float(f1_score(yt, pred, average="macro"))
        fm["AUC"] = float(roc_auc_score(yt, pt))
    else:
        fm["F1_macro"] = None; fm["AUC"] = None
        fm["note"] = f"single-class test (class={int(classes[0])})"

    tn = nodes[nodes["ID_опыта"].isin(set(te))]
    tl = lines[lines["ID_опыта"].isin(set(te))]
    pn, pl, _, _ = grid_mlp.predict_bundle(tn, tl, b, zero_invalid=False)
    n_valid_te = int((frame["regime"].loc[te].to_numpy() == 1).sum())
    if n_valid_te > 0:
        reg = e1.eval_regression(pn, pl, nodes, lines, set(te))
    else:
        reg = {"V_varying_nodes": {"MAE": None, "R2": None},
               "V_all_nodes": {"MAE": None, "R2": None},
               "V_uniform_R2_nodes": None, "V_median_R2_nodes": None,
               "V_worst_nodes": [], "Delta": {"MAE": None, "R2": None},
               "Izarg": {"MAE": None, "R2": None}, "Imax_nominal": {"MAE": None, "R2": None},
               "parallel_spread": None, "V_MAE_by_load_quintile": [],
               "note": "no valid experiences in test -> regression undefined"}

    rec = {
        "label": label, "seed": seed, "n_valid_test": n_valid_te,
        "flag": fm,
        "V_varying": reg["V_varying_nodes"],
        "V_all": reg["V_all_nodes"],
        "V_uniform_R2_nodes": reg["V_uniform_R2_nodes"],
        "V_median_R2_nodes": reg["V_median_R2_nodes"],
        "V_worst": reg["V_worst_nodes"][:3],
        "Delta": reg["Delta"],
        "Izarg": reg["Izarg"],
        "Imax_nominal": reg["Imax_nominal"],
        "parallel_spread": reg.get("parallel_spread"),
        "V_MAE_by_load_quintile": reg.get("V_MAE_by_load_quintile"),
        "note": reg.get("note"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    results.append(rec)
    with open(gl.REPORTS_DIR / "stage4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    f1s = "n/a" if fm["F1_macro"] is None else f"{fm['F1_macro']:.4f}"
    vmae = "n/a" if reg["V_varying_nodes"]["MAE"] is None else f"{reg['V_varying_nodes']['MAE']:.4f}"
    print(f"    F1={f1s} acc={fm['accuracy']:.4f} V_MAE={vmae} "
          f"valid_test={n_valid_te} ({rec['elapsed_sec']}s)", flush=True)


def main():
    nodes, lines = gl.load_data()
    frame = gl.build_frame(nodes, lines)
    exp = frame["exp"]
    results = []

    for seed in (42, 7, 2024):
        tr, va, te = gl.split_experiences(exp, seed=seed, scheme="random")
        run_one(frame, nodes, lines, tr, va, te, "random", seed, results)

    # блочный по ID
    tr, va, te = gl.split_experiences(exp, seed=42, scheme="blocks")
    run_one(frame, nodes, lines, tr, va, te, "blocks_id", 42, results)

    # экстраполяция по нагрузке
    sumPn = frame["Pn"].sum(axis=1).reindex(exp).to_numpy()
    tr, va, te = gl.split_experiences(exp, seed=42, scheme="loadrank", sort_key=sumPn)
    run_one(frame, nodes, lines, tr, va, te, "extrapolate_load", 42, results)

    # сводка по 3 random seed
    rnd = [r for r in results if r["label"] == "random"]
    summary = {}
    for key, getter in {
        "flag_F1": lambda r: r["flag"]["F1_macro"],
        "V_MAE": lambda r: r["V_varying"]["MAE"],
        "V_R2_nodes": lambda r: r["V_uniform_R2_nodes"],
        "Delta_MAE": lambda r: r["Delta"]["MAE"],
        "Izarg_MAE": lambda r: r["Izarg"]["MAE"],
        "Imax_MAE": lambda r: r["Imax_nominal"]["MAE"],
    }.items():
        vals = np.array([getter(r) for r in rnd])
        summary[key] = {"mean": float(vals.mean()), "std": float(vals.std()), "values": vals.tolist()}
    with open(gl.REPORTS_DIR / "stage4_results.json", "w", encoding="utf-8") as f:
        json.dump({"runs": results, "random_summary": summary}, f, ensure_ascii=False, indent=2)
    print("\n3-seed summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
