"""Общие утилиты для v2-пайплайна (Python 3.10, sklearn/torch)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent  # корень проекта (src/..)
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
PREDICTIONS_DIR = ROOT / "predictions"
MY_TEST_DIR = ROOT / "my_test"
GEN_INPUT_NODES = [29, 33]
NODE_PRED_COLS = ["ID_опыта", "Номер", "V", "Delta", "Regime_valid"]
LINE_PRED_COLS = ["ID_опыта", "Номер", "Imax, A", "Izarg, %"]


def load_data(nodes_path=None, lines_path=None):
    nodes_path = Path(nodes_path) if nodes_path else DATA_DIR / "train_nodes_34_100000.csv"
    lines_path = Path(lines_path) if lines_path else DATA_DIR / "train_lines_34_100000.csv"
    nodes = pd.read_csv(nodes_path, dtype={"ID_опыта": "int32", "Номер": "int16"})
    lines = pd.read_csv(lines_path, dtype={"ID_опыта": "int32", "Номер": "int16",
                                           "N_s": "int16", "N_e": "int16", "type": "int8"})
    for c in ["U_ном", "P_н", "Q_н", "P_г", "V", "Delta"]:
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce").astype("float32")
    for c in ["R", "X", "B", "Kt", "I_max", "I_доп", "I_zarg"]:
        if c in lines.columns:
            lines[c] = pd.to_numeric(lines[c], errors="coerce").astype("float32")
    return nodes, lines


def build_frame(nodes, lines):
    exp = np.sort(nodes["ID_опыта"].unique())
    node_numbers = [int(c) for c in sorted(nodes["Номер"].unique())]
    line_numbers = [int(c) for c in sorted(lines["Номер"].unique())]

    def npiv(col):
        if col in nodes.columns:
            return nodes.pivot(index="ID_опыта", columns="Номер", values=col).reindex(
                index=exp, columns=node_numbers)
        return pd.DataFrame(np.nan, index=exp, columns=node_numbers)

    def lpiv(col):
        if col in lines.columns:
            return lines.pivot(index="ID_опыта", columns="Номер", values=col).reindex(
                index=exp, columns=line_numbers)
        return pd.DataFrame(np.nan, index=exp, columns=line_numbers)

    pn = npiv("P_н"); qn = npiv("Q_н"); pg = npiv("P_г")
    v = npiv("V"); d = npiv("Delta")
    iz = lpiv("I_zarg"); imax = lpiv("I_max"); idop = lpiv("I_доп")
    if "regime_valid" in nodes.columns:
        regime = nodes.groupby("ID_опыта")["regime_valid"].first().reindex(exp).astype(int)
    else:
        regime = pd.Series(1, index=exp)

    feat_nodes = [int(c) for c in node_numbers if int(c) != 999]
    X = pd.DataFrame({f"Pn_{c}": pn[c].to_numpy() for c in feat_nodes}, index=exp)
    for g in GEN_INPUT_NODES:
        X[f"Pg_{g}"] = pg[g].to_numpy() if g in pg.columns else 0.0
    X["sumPn"] = pn[feat_nodes].sum(axis=1).to_numpy()
    X["maxPn"] = pn[feat_nodes].max(axis=1).to_numpy()
    X["sumPg_in"] = X[[f"Pg_{g}" for g in GEN_INPUT_NODES]].sum(axis=1).to_numpy()
    X["balance"] = X["sumPg_in"] - X["sumPn"]
    X = X.fillna(0.0)

    return {"exp": exp, "X": X, "Pn": pn, "Qn": qn, "Pg": pg, "V": v, "Delta": d,
            "Izarg": iz, "Imax": imax, "Idop": idop, "regime": regime,
            "node_numbers": node_numbers, "line_numbers": line_numbers}


def split_experiences(exp, seed=42, train_frac=0.70, val_frac=0.15, test_frac=0.15,
                      scheme="random", sort_key=None):
    """Возвращает (tr, va, te) массивы ID опытов. scheme='random' или 'blocks'."""
    exp = np.asarray(exp)
    n = len(exp)
    if scheme == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        tr = np.sort(exp[perm[:n_tr]])
        va = np.sort(exp[perm[n_tr:n_tr + n_va]])
        te = np.sort(exp[perm[n_tr + n_va:]])
    elif scheme == "blocks":
        # блоки по ID: тест — верхний хвост по ID (проверка сдвига по времени/ID)
        order = exp[np.argsort(exp)]
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        tr, va, te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    elif scheme == "loadrank":
        key = sort_key if sort_key is not None else exp
        order = exp[np.argsort(key)]
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        tr, va, te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    else:
        raise ValueError(scheme)
    return np.sort(tr), np.sort(va), np.sort(te)


def reg_metrics(true, pred):
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    return {"MAE": float(mean_absolute_error(true, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
            "R2": float(r2_score(true, pred, multioutput="uniform_average"))}


def clf_metrics(y_true, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    cm = np.zeros((2, 2), int)
    for t, p in zip(np.asarray(y_true).astype(int), pred):
        cm[t, p] += 1
    return {"F1_macro": float(f1_score(y_true, pred, average="macro")),
            "accuracy": float(accuracy_score(y_true, pred)),
            "AUC": float(roc_auc_score(y_true, prob)),
            "confusion": cm.tolist(),
            "threshold": float(threshold)}


def best_threshold(y_true, prob, metric="f1macro"):
    grid = np.linspace(0.05, 0.95, 91)
    best_t, best_v = 0.5, -1.0
    for t in grid:
        pred = (prob >= t).astype(int)
        if metric == "f1macro":
            v = f1_score(y_true, pred, average="macro")
        else:
            v = accuracy_score(y_true, pred)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


def parallel_groups(lines):
    """Отображение номер_линии -> id группы идентичных параллельных линий."""
    geo = lines.drop_duplicates("Номер")[["Номер", "N_s", "N_e", "type", "R", "X", "B", "Kt"]].copy()
    key = geo[["N_s", "N_e", "type", "R", "X", "B", "Kt"]].round(5).astype(str).agg("|".join, axis=1)
    ids = pd.factorize(key)[0]
    return dict(zip(geo["Номер"].astype(int), ids))


def average_parallel_izarg(iz_pred, groups):
    """iz_pred: DataFrame(index=exp, columns=line_no). Усредняет внутри групп."""
    out = iz_pred.copy()
    inv = {}
    for ln, gid in groups.items():
        inv.setdefault(gid, []).append(ln)
    for gid, members in inv.items():
        if len(members) > 1:
            members = [m for m in members if m in out.columns]
            m = out[members].mean(axis=1)
            for c in members:
                out[c] = m
    return out


def quasi_constant_nodes(v_train, mode_frac=0.99, tol=1e-3):
    """Возвращает {node: mode} для узлов, у которых >mode_frac значений равны моде."""
    res = {}
    for c in v_train.columns:
        s = v_train[c].dropna()
        if len(s) == 0:
            continue
        mode = s.round(3).mode()
        if len(mode) == 0:
            continue
        mode = float(mode.iloc[0])
        frac = float((np.abs(s - mode) <= tol).mean())
        if frac >= mode_frac:
            res[int(c)] = mode
    return res


def predict_frames(nodes, lines, bundle):
    """Инференс bundle sklearn-пайплайна; возвращает pn, pl (в порядке входа)."""
    from grid_mlp import predict_bundle
    return predict_bundle(nodes, lines, bundle)
