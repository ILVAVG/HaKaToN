"""Этап C: проверка решателя режима (NR) на валидных/невалидных опытах."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import physics as ph  # noqa: E402
import nr_solver as nr  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def load_pool(seed=7):
    nodes = pd.read_csv(DATA / "train_nodes_34_100000.csv",
                        dtype={"ID_опыта": "int32", "Номер": "int16"})
    valid = nodes.groupby("ID_опыта")["regime_valid"].first()
    has999 = set(nodes.loc[nodes["Номер"] == 999, "ID_опыта"].unique())
    v_ids = sorted(e for e in has999 if int(valid.loc[e]) == 1)
    i_ids = sorted(e for e in has999 if int(valid.loc[e]) == 0)
    rng = np.random.default_rng(seed)
    return nodes, valid, v_ids, i_ids, rng


def load_subset(nodes, ids):
    ns = nodes[nodes["ID_опыта"].isin(set(ids))].copy()
    parts = []
    for ch in pd.read_csv(DATA / "train_lines_34_100000.csv", chunksize=1_000_000,
                          dtype={"ID_опыта": "int32", "Номер": "int16",
                                 "N_s": "int16", "N_e": "int16", "type": "int8"}):
        parts.append(ch[ch["ID_опыта"].isin(set(ids))])
    return ns, pd.concat(parts, ignore_index=True)


def load_qcoeff():
    d = json.loads((REPORTS / "phys_b.json").read_text(encoding="utf-8"))
    q = {int(r["node"]): (r["d0"], r["d1"], r["d2"], r["q_sh"])
         for r in d["B2"]["per_node"]}
    q.update(nr.QCOEFF_EXTRA)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvalid", type=int, default=300)
    ap.add_argument("--ninvalid", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    nodes, valid, v_ids, i_ids, rng = load_pool(args.seed)
    vs = sorted(rng.choice(v_ids, size=min(args.nvalid, len(v_ids)), replace=False).tolist())
    is_ = sorted(rng.choice(i_ids, size=min(args.ninvalid, len(i_ids)), replace=False).tolist())
    all_ids = vs + is_
    ns, ls = load_subset(nodes, all_ids)
    gN = {e: g for e, g in ns.groupby("ID_опыта")}
    gL = {e: g for e, g in ls.groupby("ID_опыта")}
    q = load_qcoeff()

    t0 = time.time()
    # ---- валидные ----
    v_abs, d_abs, im_abs = [], [], []
    node_v_abs = {}
    node_d_abs = {}
    conv_valid = 0; nvalid_done = 0
    pred_valid = []
    fails = []
    for e in vs:
        ne, le = gN.get(e), gL.get(e)
        if ne is None or le is None:
            continue
        r = nr.solve_pf(ne, le, q)
        if not r["ok"]:
            for iv in (0.9, 1.05):
                r2 = nr.solve_pf(ne, le, q, init_vpu=iv)
                if r2["ok"]:
                    r = r2; break
        nvalid_done += 1
        pred_valid.append(1 if r["ok"] else 0)
        if r["ok"]:
            conv_valid += 1
        else:
            fails.append(e)
        data = ne.set_index("Номер")
        ids = r["ids"]
        Vt = data["V"].reindex(ids).to_numpy(float)
        Dt = data["Delta"].reindex(ids).to_numpy(float)
        if r["ok"] and np.all(np.isfinite(r["V_kV"])):
            av = np.abs(r["V_kV"] - Vt); ad = np.abs(r["Delta_deg"] - Dt)
            v_abs.append(av); d_abs.append(ad)
            for n, x, y in zip(ids, av, ad):
                node_v_abs.setdefault(int(n), []).append(float(x))
                node_d_abs.setdefault(int(n), []).append(float(y))
        if r["ok"] and len(r.get("I_df", [])):
            bdf = r["I_df"].drop(columns=["I_max"], errors="ignore")
            m = bdf.merge(le[["Номер", "I_max"]], left_on="line", right_on="Номер", how="left")
            ok = m["I_max"].notna()
            im_abs.append(np.abs(m.loc[ok, "Imax_pred"].to_numpy() - m.loc[ok, "I_max"].to_numpy()))

    # ---- невалидные ----
    conv_invalid = 0; ninv_done = 0
    pred_invalid = []
    for e in is_:
        ne, le = gN.get(e), gL.get(e)
        if ne is None or le is None:
            continue
        r = nr.solve_pf(ne, le, q)
        if not r["ok"]:
            for iv in (0.9, 1.05):
                r2 = nr.solve_pf(ne, le, q, init_vpu=iv)
                if r2["ok"]:
                    r = r2; break
        ninv_done += 1
        pred_invalid.append(1 if r["ok"] else 0)
        if r["ok"]:
            conv_invalid += 1
    elapsed = time.time() - t0

    Vmae = float(np.mean(np.concatenate(v_abs))) if v_abs else None
    Dmae = float(np.mean(np.concatenate(d_abs))) if d_abs else None
    Imae = float(np.mean(np.concatenate(im_abs))) if im_abs else None
    y_true = np.array([1] * nvalid_done + [0] * ninv_done)
    y_pred = np.array(pred_valid + pred_invalid)
    f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else None
    cm = [[int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
          [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())]]

    out = {"n_valid": nvalid_done, "n_invalid": ninv_done,
           "valid_converged": conv_valid, "valid_converged_frac": conv_valid / max(1, nvalid_done),
           "invalid_converged": conv_invalid, "invalid_converged_frac": conv_invalid / max(1, ninv_done),
           "MAE_V_kV_converged": Vmae, "MAE_Delta_deg_converged": Dmae, "MAE_Imax_A_converged": Imae,
           "F1_macro": f1, "confusion_rows_true0_1": cm,
           "time_per_100_s": elapsed / max(1, nvalid_done + ninv_done) * 100,
           "top_v_nodes": [{"node": int(n), "maeV": float(np.mean(v))}
                           for n, v in sorted(node_v_abs.items(), key=lambda kv: -np.mean(kv[1]))[:6]],
           "fail_valid_examples": fails[:10]}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "phys_c.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print("STAGE C")
    print(f"valid: {conv_valid}/{nvalid_done} converged ({conv_valid/max(1,nvalid_done)*100:.1f}%)")
    print(f"invalid: {conv_invalid}/{ninv_done} converged ({conv_invalid/max(1,ninv_done)*100:.1f}%)")
    print(f"MAE V={Vmae:.4f} kV | Delta={Dmae:.4f} deg | Imax={Imae:.3f} A (converged only)")
    print(f"F1_macro={f1:.4f} confusion(true0/1 x pred0/1)={cm}")
    print(f"time/100={out['time_per_100_s']:.2f} s")
    print("top V-error nodes:", [(d["node"], round(d["maeV"], 3)) for d in out["top_v_nodes"]])
    print("fail valid:", out["fail_valid_examples"])


if __name__ == "__main__":
    main()
