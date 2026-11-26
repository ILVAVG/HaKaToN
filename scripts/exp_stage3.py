"""Этап 3: обучение графовой модели (GINE-подобной) + честная оценка.

Запуск: python exp_stage3.py [--epochs 150] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grid_lib as gl  # noqa: E402
from gnn_model import GridGNN  # noqa: E402

torch.set_num_threads(4)


def build_tensors(frame):
    nodes = frame["exp"]
    node_numbers = sorted(frame["Pn"].columns)
    line_numbers = list(frame["Izarg"].columns)
    N, E = len(node_numbers), len(line_numbers)

    Pn = frame["Pn"].reindex(columns=node_numbers).fillna(0).to_numpy(np.float32)
    Pg = frame["Pg"].reindex(columns=node_numbers).fillna(0).to_numpy(np.float32)
    Qn = frame["Qn"].reindex(columns=node_numbers).fillna(0).to_numpy(np.float32)
    gen_mask = np.zeros(N, np.float32)
    for g in gl.GEN_INPUT_NODES:
        gen_mask[node_numbers.index(g)] = 1.0
    Pg_in = Pg * gen_mask  # 999 и прочие обнулены
    net = Pg_in - Pn

    slack = np.array([1.0 if n == 999 else 0.0 for n in node_numbers], np.float32)
    pv = np.array([1.0 if n in gl.GEN_INPUT_NODES else 0.0 for n in node_numbers], np.float32)
    pq = 1.0 - slack - pv
    Umap = frame.get("Umap")
    U = np.array([Umap[n] for n in node_numbers], np.float32) / 500.0

    onehot = np.stack([slack, pv, pq], axis=1)  # [N,3]
    static = np.concatenate([onehot, U[:, None]], axis=1)  # [N,4]
    static = np.broadcast_to(static, (len(nodes), N, static.shape[1]))

    X = np.concatenate([Pn[:, :, None], Pg_in[:, :, None], Qn[:, :, None],
                        net[:, :, None], static], axis=2).astype(np.float32)

    Vt = frame["V"].reindex(columns=node_numbers).to_numpy(np.float32)
    Dt = frame["Delta"].reindex(columns=node_numbers).to_numpy(np.float32)
    Iz = frame["Izarg"].reindex(columns=line_numbers).fillna(0.0).to_numpy(np.float32)
    return X, Vt, Dt, Iz, node_numbers, line_numbers


def build_graph(lines, node_numbers):
    geo = lines.drop_duplicates("Номер").sort_values("Номер").copy()
    nidx = {n: i for i, n in enumerate(node_numbers)}
    src = np.array([nidx[int(x)] for x in geo["N_s"]], np.int64)
    dst = np.array([nidx[int(x)] for x in geo["N_e"]], np.int64)
    eattr = geo[["R", "X", "B", "Kt", "type"]].to_numpy(np.float32)
    # forward (0:E) соответствует строкам geo (номер линии по порядку)
    src2 = np.concatenate([src, dst])
    dst2 = np.concatenate([dst, src])
    eattr2 = np.concatenate([eattr, eattr], axis=0)
    return src2, dst2, eattr2, len(geo)


def masked_mse(pred, targ, mask):
    se = ((pred - targ) ** 2).mean(dim=1)
    return (se * mask).sum() / mask.sum().clamp(min=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    t0 = time.time()
    nodes, lines = gl.load_data()
    frame = gl.build_frame(nodes, lines)
    frame["Umap"] = nodes.drop_duplicates("Номер").set_index("Номер")["U_ном"].astype(float).to_dict()

    exp, regime = frame["exp"], frame["regime"]
    tr, va, te = gl.split_experiences(exp, seed=args.seed, scheme="random")
    exp_pos = {int(e): i for i, e in enumerate(exp)}
    tr_i = np.array([exp_pos[int(e)] for e in tr])
    va_i = np.array([exp_pos[int(e)] for e in va])
    te_i = np.array([exp_pos[int(e)] for e in te])

    X, Vt, Dt, Iz, node_numbers, line_numbers = build_tensors(frame)
    src, dst, eattr, E = build_graph(lines, node_numbers)
    N = len(node_numbers)

    # стандартизация входов
    Xtr = X[tr_i]
    xmu = Xtr.reshape(-1, Xtr.shape[2]).mean(0); xsd = Xtr.reshape(-1, Xtr.shape[2]).std(0)
    xsd[xsd < 1e-6] = 1.0
    emu = eattr.mean(0); esd = eattr.std(0); esd[esd < 1e-6] = 1.0

    reg = regime.to_numpy()
    trv = tr_i[reg[tr_i] == 1]
    # стандартизация целей по столбцам (по валидным train)
    vmu = Vt[trv].mean(0); vsd = Vt[trv].std(0); vsd[vsd < 1e-6] = 1.0
    dmu = Dt[trv].mean(0); dsd = Dt[trv].std(0); dsd[dsd < 1e-6] = 1.0
    imu = Iz[trv].mean(0); isd = Iz[trv].std(0); isd[isd < 1e-6] = 1.0

    Xn = torch.tensor((X - xmu) / xsd)
    Vz = torch.tensor((Vt - vmu) / vsd)
    Dz = torch.tensor((Dt - dmu) / dsd)
    IzZ = torch.tensor((Iz - imu) / isd)
    src_t = torch.tensor(src); dst_t = torch.tensor(dst)
    eattr_t = torch.tensor((eattr - emu) / esd)
    ytr = torch.tensor(reg.astype(np.float32))

    model = GridGNN(X.shape[2], eattr.shape[1], hidden=args.hidden, layers=args.layers,
                    n_lines=E)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    bce = nn.BCEWithLogitsLoss()
    va_mon = va_i[:min(len(va_i), 4000)]
    ckpt_path = gl.MODELS_DIR / "gnn_best.pt"

    def run_batches(idx, train=True):
        model.train(train)
        order = np.random.permutation(len(idx)) if train else np.arange(len(idx))
        tot = {"loss": 0.0, "cls": 0.0, "v": 0.0, "d": 0.0, "iz": 0.0}; nb = 0
        for s in range(0, len(idx), args.batch):
            bidx = idx[order[s:s + args.batch]]
            bt = torch.tensor(bidx)
            xb = Xn[bt]
            vd, izp, logit = model(xb, src_t, dst_t, eattr_t)
            yb = ytr[bt]
            mask = yb
            loss_cls = bce(logit, yb)
            loss_v = masked_mse(vd[:, :, 0], Vz[bt], mask)
            loss_d = masked_mse(vd[:, :, 1], Dz[bt], mask)
            loss_iz = masked_mse(izp[:, :E], IzZ[bt], mask)
            loss = loss_cls + loss_v + loss_d + loss_iz
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot["loss"] += float(loss.detach()); tot["cls"] += float(loss_cls.detach())
            tot["v"] += float(loss_v.detach()); tot["d"] += float(loss_d.detach())
            tot["iz"] += float(loss_iz.detach())
            nb += 1
        return {k: v / nb for k, v in tot.items()}

    best = {"val_loss": 1e9, "epoch": -1, "state": None}
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        trs = run_batches(tr_i, True)
        if ep % 5 == 0 or ep == 1:
            with torch.no_grad():
                vas = run_batches(va_mon, False)
            sched.step(vas["loss"])
            if vas["loss"] < best["val_loss"]:
                best = {"val_loss": vas["loss"], "epoch": ep,
                        "state": {k: v.clone() for k, v in model.state_dict().items()}}
                torch.save({"state": best["state"], "epoch": ep, "args": vars(args)},
                           ckpt_path)
            print(f"ep {ep:3d} train {trs['loss']:.4f} (cls {trs['cls']:.3f} v {trs['v']:.4f} "
                  f"d {trs['d']:.4f} iz {trs['iz']:.4f}) | val {vas['loss']:.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} | {time.time()-t_start:.0f}s", flush=True)
    model.load_state_dict(best["state"])

    # ---- предсказания на test ----
    model.eval()
    with torch.no_grad():
        bt = torch.tensor(te_i)
        vd, izp, logit = model(Xn[bt], src_t, dst_t, eattr_t)
    Vp = vd[:, :, 0].numpy() * vsd + vmu
    Dp = vd[:, :, 1].numpy() * dsd + dmu
    Izp = izp[:, :E].numpy() * isd + imu
    prob = torch.sigmoid(logit).numpy()

    # val proba для порога
    with torch.no_grad():
        bv = torch.tensor(va_i)
        _, _, lv = model(Xn[bv], src_t, dst_t, eattr_t)
    pval = torch.sigmoid(lv).numpy()
    yval = reg[va_i]
    from grid_lib import best_threshold, clf_metrics
    tbest, fbest = best_threshold(yval, pval, "f1macro")

    res = {"seed": args.seed, "epochs": args.epochs, "best_epoch": best["epoch"],
           "best_val_loss": best["val_loss"], "threshold_val": tbest,
           "flag_test_0.5": clf_metrics(reg[te_i], prob, 0.5),
           "flag_test_tuned": clf_metrics(reg[te_i], prob, tbest)}

    # ---- метрики регрессии ----
    te_valid = te_i[reg[te_i] == 1]
    pos_te = {int(e): i for i, e in enumerate(te_i)}
    sel = np.array([pos_te[int(e)] for e in te_valid])
    vtrue = Vt[te_valid]; dtrue = Dt[te_valid]; itrue = Iz[te_valid]
    vpred = Vp[sel]; dpred = Dp[sel]; ipred = Izp[sel]

    varying = np.where(Vt[trv].std(0) > 1e-6)[0]
    res["V_varying"] = gl.reg_metrics(vtrue[:, varying], vpred[:, varying])
    res["V_all_nodes"] = gl.reg_metrics(vtrue, vpred)
    res["Delta"] = gl.reg_metrics(dtrue, dpred)
    res["Izarg"] = gl.reg_metrics(itrue, ipred)
    Idop_nom = frame["Idop"].loc[exp[trv]].mean(0).reindex(line_numbers).to_numpy()
    Imax_true = frame["Imax"].loc[exp[te_valid], line_numbers].to_numpy(np.float64)
    res["Imax_nominal"] = gl.reg_metrics(Imax_true, ipred * Idop_nom / 100.0)

    # per-node V R2
    per = []
    for k in varying:
        r2 = r2_score(vtrue[:, k], vpred[:, k]) if vtrue[:, k].std() > 0 else np.nan
        per.append((int(node_numbers[k]), float(r2), float(mean_absolute_error(vtrue[:, k], vpred[:, k])),
                    float(vtrue[:, k].std())))
    per.sort(key=lambda t: t[1])
    res["V_worst_nodes"] = [{"node": a, "R2": b, "MAE": c, "std": d} for a, b, c, d in per[:6]]
    r2s = np.array([p[1] for p in per])
    res["V_uniform_R2_nodes"] = float(r2s.mean())
    res["V_uniform_R2_excl_node1"] = float(np.delete(r2s, np.where(np.array([p[0] for p in per]) == 1)[0]).mean())
    res["V_median_R2_nodes"] = float(np.median(r2s))
    res["V_by_class"] = {}
    Umap = frame["Umap"]
    classes = np.array([Umap[n] for n in node_numbers])
    for u in (500.0, 220.0, 110.0):
        cols = np.where(classes == u)[0]
        res["V_by_class"][str(int(u))] = {
            **gl.reg_metrics(vtrue[:, cols], vpred[:, cols]),
            "rel_MAE_pct": float(mean_absolute_error(vtrue[:, cols], vpred[:, cols]) /
                                 vtrue[:, cols].mean() * 100)}
    # load quintiles
    sumPn = frame["Pn"].sum(axis=1).to_numpy()[te_valid]
    ae = np.abs(vtrue[:, varying] - vpred[:, varying]).mean(1)
    q = pd.qcut(sumPn, 5, labels=False, duplicates="drop")
    res["V_MAE_by_load_quintile"] = [
        {"q": int(k), "sumPn": float(sumPn[q == k].mean()), "MAE": float(ae[q == k].mean())}
        for k in sorted(pd.unique(q))]

    res["elapsed_sec"] = round(time.time() - t0, 1)
    torch.save({"state": best["state"], "meta": {"xmu": xmu, "xsd": xsd,
                "vmu": vmu, "vsd": vsd, "dmu": dmu, "dsd": dsd, "imu": imu, "isd": isd,
                "node_numbers": node_numbers, "line_numbers": line_numbers, "args": vars(args)}},
               gl.MODELS_DIR / "gnn_v2.pt")
    with open(gl.REPORTS_DIR / "gnn_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: res[k] for k in ["V_varying", "V_uniform_R2_nodes",
           "V_uniform_R2_excl_node1", "Delta", "Izarg", "Imax_nominal",
           "flag_test_tuned"]}, ensure_ascii=False, indent=2))
    print(f"Готово за {res['elapsed_sec']} c -> gnn_results.json, gnn_v2.pt")


if __name__ == "__main__":
    main()
