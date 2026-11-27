"""Этап D1: гибрид (DC-признаки + MLP) против v2 на тех же сплитах."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import grid_lib as gl  # noqa: E402
import grid_mlp  # noqa: E402
import hybrid  # noqa: E402

REPORTS = ROOT / "reports"


def predict_metrics(bundle, frame_h, frame, lines, te):
    te = np.asarray(te)
    Xte = frame_h["X"].loc[te]
    clf = bundle["clf"]
    prob = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
    yt = frame["regime"].loc[te].to_numpy()
    pred = (prob >= bundle["threshold"]).astype(int)
    f1 = float(f1_score(yt, pred, average="macro"))

    val = te[yt == 1]
    Xv = frame_h["X"].loc[val]
    Vp = grid_mlp._pred(bundle["v_model"], bundle["v_xs"], bundle["v_ys"], Xv)
    Dp = grid_mlp._pred(bundle["d_model"], bundle["d_xs"], bundle["d_ys"], Xv)
    Izp = grid_mlp._pred(bundle["iz_model"], bundle["iz_xs"], bundle["iz_ys"], Xv)
    Vt = frame["V"].loc[val, bundle["v_cols"]].to_numpy(float)
    Dt = frame["Delta"].loc[val, bundle["d_cols"]].to_numpy(float)
    Izdf = pd.DataFrame(Izp, index=val, columns=bundle["iz_cols"])
    Izdf = gl.average_parallel_izarg(Izdf, gl.parallel_groups(lines))
    Izt = frame["Izarg"].loc[val, bundle["iz_cols"]].to_numpy(float)
    idop = bundle["idop_nominal"].reindex(bundle["iz_cols"]).to_numpy()
    Imax_pred = Izdf.to_numpy() * idop / 100.0
    Imax_t = frame["Imax"].loc[val, bundle["iz_cols"]].to_numpy(float)

    r2s = [r2_score(Vt[:, k], Vp[:, k]) for k in range(Vt.shape[1]) if Vt[:, k].std() > 1e-6]
    return {"F1_macro": f1, "V_MAE": float(mean_absolute_error(Vt, Vp)),
            "V_R2_nodes": float(np.mean(r2s)), "Delta_MAE": float(mean_absolute_error(Dt, Dp)),
            "Izarg_MAE": float(mean_absolute_error(Izt, Izdf.to_numpy())),
            "Imax_MAE": float(mean_absolute_error(Imax_t, Imax_pred))}


def v2_baseline():
    d = json.loads((REPORTS / "stage4_results.json").read_text(encoding="utf-8"))
    out = {}
    for r in d["runs"]:
        if r["label"] == "random" and r["seed"] == 42:
            out["random42"] = {"F1_macro": r["flag"]["F1_macro"], "V_MAE": r["V_varying"]["MAE"],
                               "V_R2_nodes": r["V_uniform_R2_nodes"], "Delta_MAE": r["Delta"]["MAE"],
                               "Izarg_MAE": r["Izarg"]["MAE"], "Imax_MAE": r["Imax_nominal"]["MAE"]}
        if r["label"] == "blocks_id":
            out["blocks"] = {"F1_macro": r["flag"]["F1_macro"], "V_MAE": r["V_varying"]["MAE"],
                             "V_R2_nodes": r["V_uniform_R2_nodes"], "Delta_MAE": r["Delta"]["MAE"],
                             "Izarg_MAE": r["Izarg"]["MAE"], "Imax_MAE": r["Imax_nominal"]["MAE"]}
    return out


def run(frame, frame_h, lines, tr, va, te, seed=42):
    t = time.time()
    b = grid_mlp.fit_bundle(frame_h, tr, config=grid_mlp.MEDIUM, seed=seed,
                            parallel_avg=True, snap=True, verbose=False)
    Xv = frame_h["X"].loc[va]
    pv = b["clf"].predict_proba(Xv)[:, list(b["clf"].classes_).index(1)]
    tb, _ = gl.best_threshold(frame["regime"].loc[va].to_numpy(), pv, "f1macro")
    b["threshold"] = tb
    m = predict_metrics(b, frame_h, frame, lines, te)
    m["elapsed_sec"] = round(time.time() - t, 1)
    m["threshold"] = tb
    return m


def main():
    nodes, lines = gl.load_data()
    frame = gl.build_frame(nodes, lines)
    umap = nodes.drop_duplicates("Номер").set_index("Номер")["U_ном"].astype(float).to_dict()
    dc = hybrid.build_dc_features(frame, lines, umap)
    frame_h = dict(frame)
    frame_h["X"] = pd.concat([frame["X"], dc], axis=1)
    exp = frame["exp"]

    out = {"v2": v2_baseline(), "hybrid": {}}
    for name, scheme, seed in (("random42", "random", 42), ("blocks", "blocks", 42)):
        tr, va, te = gl.split_experiences(exp, seed=seed, scheme=scheme)
        print(f"fit hybrid {name} ...", flush=True)
        out["hybrid"][name] = run(frame, frame_h, lines, tr, va, te, seed=seed)
        print(name, out["hybrid"][name], flush=True)

    (REPORTS / "phys_d1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    keys = ["F1_macro", "V_MAE", "V_R2_nodes", "Delta_MAE", "Izarg_MAE", "Imax_MAE"]
    print("\nMETRIC      | v2 random42 | hyb random42 | v2 blocks | hyb blocks")
    for k in keys:
        a = out["v2"]["random42"].get(k); c = out["hybrid"]["random42"].get(k)
        b_ = out["v2"]["blocks"].get(k); d = out["hybrid"]["blocks"].get(k)
        print(f"{k:11s} | {a:11.4f} | {c:12.4f} | {b_:9.4f} | {d:10.4f}")


if __name__ == "__main__":
    main()
