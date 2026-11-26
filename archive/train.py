"""
Проверочный train для хакатона: прогноз установившегося режима сети.

Ключевые решения (по итогам анализа данных и конспекту статьи):
  * НЕТ утечки: не используются I_доп, P_г узла 999, Q_н (избыточен).
  * Входы: P_н по всем узлам + P_г генераторных узлов 29/33 + агрегаты.
  * Флаг regime_valid ищется лёгкой моделью (HGB) по этим же признакам;
    дополнительно печатается простой порог по суммарной нагрузке для сравнения.
  * Регрессия учится только на валидных режимах (маскирование как в статье).
  * Константные узлы V (2,4,7,13,999) не предсказываются, а берутся из данных.
  * I_max восстанавливается из I_zarg и номинального I_доп по линии.

Разбиение: GroupShuffleSplit по ID_опыта, 70% train / 30% проверка.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

NODE_PRED_COLS = ["ID_опыта", "Номер", "V", "Delta", "Regime_valid"]
LINE_PRED_COLS = ["ID_опыта", "Номер", "Imax, A", "Izarg, %"]

# Генераторные узлы, которые задаются на входе (999 исключён - это утечка).
GEN_INPUT_NODES = [29, 33]


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_data(nodes_path: Path, lines_path: Path):
    nodes = pd.read_csv(
        nodes_path,
        dtype={"ID_опыта": "int32", "Номер": "int16"},
    )
    lines = pd.read_csv(
        lines_path,
        dtype={"ID_опыта": "int32", "Номер": "int16", "N_s": "int16", "N_e": "int16"},
    )
    for c in ["U_ном", "P_н", "Q_н", "P_г", "V", "Delta"]:
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce").astype("float32")
    for c in ["R", "X", "B", "Kt", "I_max", "I_доп", "I_zarg"]:
        if c in lines.columns:
            lines[c] = pd.to_numeric(lines[c], errors="coerce").astype("float32")
    return nodes, lines


def build_experience_frame(nodes: pd.DataFrame, lines: pd.DataFrame):
    """Собирает матрицы по опытам: признаки X, цели V/Delta/Izarg, номинальные I_доп."""
    exp_index = np.sort(nodes["ID_опыта"].unique())

    pn = nodes.pivot(index="ID_опыта", columns="Номер", values="P_н").reindex(exp_index)
    pg = nodes.pivot(index="ID_опыта", columns="Номер", values="P_г").reindex(exp_index)
    v = nodes.pivot(index="ID_опыта", columns="Номер", values="V").reindex(exp_index)
    dlt = nodes.pivot(index="ID_опыта", columns="Номер", values="Delta").reindex(exp_index)
    regime = nodes.groupby("ID_опыта")["regime_valid"].first().reindex(exp_index).astype(int)

    # Признаки: P_н по всем обычным узлам (999 всегда 0), P_г только 29/33.
    feat_nodes = [int(c) for c in pn.columns if int(c) != 999]
    X = pd.DataFrame(
        {f"Pn_{c}": pn[c].to_numpy() for c in feat_nodes}, index=exp_index
    )
    for g in GEN_INPUT_NODES:
        X[f"Pg_{g}"] = pg[g].to_numpy()

    X["sumPn"] = pn[feat_nodes].sum(axis=1).to_numpy()
    X["maxPn"] = pn[feat_nodes].max(axis=1).to_numpy()
    X["sumPg_in"] = X[[f"Pg_{g}" for g in GEN_INPUT_NODES]].sum(axis=1).to_numpy()
    X["balance"] = X["sumPg_in"] - X["sumPn"]

    izarg = lines.pivot(index="ID_опыта", columns="Номер", values="I_zarg").reindex(exp_index)
    idop = lines.pivot(index="ID_опыта", columns="Номер", values="I_доп").reindex(exp_index)
    imax = lines.pivot(index="ID_опыта", columns="Номер", values="I_max").reindex(exp_index)

    return {
        "exp": exp_index,
        "X": X,
        "V": v,
        "Delta": dlt,
        "Izarg": izarg,
        "Imax": imax,
        "Idop": idop,
        "regime": regime,
    }


def _const_columns(df: pd.DataFrame, tol: float = 1e-6):
    """Возвращает (const_cols, const_values, varying_cols) по тренировочной части."""
    const_cols, const_values, varying = [], {}, []
    for c in df.columns:
        col = df[c]
        if col.nunique(dropna=True) <= 1 or float(col.std(ddof=0)) < tol:
            const_cols.append(c)
            const_values[c] = float(col.iloc[0])
        else:
            varying.append(c)
    return const_cols, const_values, varying


def _reg_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "R2": float(r2_score(true, pred, multioutput="uniform_average")),
    }


def _fit_mlp(Xtr, Ytr, hidden, seed=42, alpha=1e-4, max_iter=300) -> tuple:
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
    )
    model.fit(xs.transform(Xtr), ys.transform(Ytr))
    return model, xs, ys


def _pred_mlp(model, xs, ys, X) -> np.ndarray:
    return ys.inverse_transform(model.predict(xs.transform(X)))


def train(args: argparse.Namespace) -> dict:
    t0 = time.time()
    _log(f"[1/6] Загрузка данных: {args.nodes.name}, {args.lines.name}")
    nodes, lines = load_data(args.nodes, args.lines)
    frame = build_experience_frame(nodes, lines)
    X, regime = frame["X"], frame["regime"]
    _log(f"      опытов={len(X)}, признаков={X.shape[1]}, валидных={int(regime.sum())}")

    exp = frame["exp"]
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(gss.split(X, regime, groups=exp))
    tr_exp, te_exp = exp[tr_idx], exp[te_idx]
    _log(f"[2/6] Split по ID_опыта: train={len(tr_idx)}, test={len(te_idx)}")

    # ---------------- Классификация флага ----------------
    Xtr_c, Xte_c = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = regime.iloc[tr_idx].to_numpy(), regime.iloc[te_idx].to_numpy()

    # простая проверка: порог по суммарной нагрузке (обучается только на train)
    best_t, best_f = None, -1.0
    for t in np.quantile(Xtr_c["sumPn"], np.linspace(0.01, 0.99, 99)):
        f = f1_score(ytr, (Xtr_c["sumPn"] < t).astype(int), average="macro")
        if f > best_f:
            best_t, best_f = float(t), float(f)
    rule_pred = (Xte_c["sumPn"] < best_t).astype(int)
    rule_f1 = f1_score(yte, rule_pred, average="macro")

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_depth=None,
        early_stopping=True, validation_fraction=0.15, random_state=args.seed,
    )
    clf.fit(Xtr_c, ytr)
    clf_prob = clf.predict_proba(Xte_c)[:, list(clf.classes_).index(1)]
    clf_pred = (clf_prob >= 0.5).astype(int)
    clf_f1 = f1_score(yte, clf_pred, average="macro")
    clf_acc = accuracy_score(yte, clf_pred)
    clf_auc = roc_auc_score(yte, clf_prob)
    _log(
        f"[3/6] Классификация holdout: F1macro={clf_f1:.4f} acc={clf_acc:.4f} "
        f"AUC={clf_auc:.4f} | правило(sumPn) F1={rule_f1:.4f} (порог {best_t:.1f})"
    )

    # ---------------- Регрессия (только валидные) ----------------
    # Константы определяем только по train-валидным опытам.
    tr_valid_mask = regime.iloc[tr_idx].to_numpy() == 1
    tr_valid_exp = tr_exp[tr_valid_mask]
    te_valid_mask = regime.iloc[te_idx].to_numpy() == 1
    te_valid_exp = te_exp[te_valid_mask]

    Xtr_v = X.loc[tr_valid_exp]
    Xte_v = X.loc[te_valid_exp]

    v_train = frame["V"].loc[tr_valid_exp]
    d_train = frame["Delta"].loc[tr_valid_exp]
    v_const, v_const_val, v_cols = _const_columns(v_train)
    d_const, d_const_val, d_cols = _const_columns(d_train)
    _log(f"      константные V-узлы={v_const}, константные Delta-узлы={d_const}")

    # V
    Vtr = v_train[v_cols].to_numpy(np.float64)
    Vte = frame["V"].loc[te_valid_exp, v_cols].to_numpy(np.float64)
    v_model, v_xs, v_ys = _fit_mlp(Xtr_v, Vtr, hidden=(256, 128, 64))
    Vpred = _pred_mlp(v_model, v_xs, v_ys, Xte_v)
    v_metrics = _reg_metrics(Vte, Vpred)

    # Delta
    Dtr = d_train[d_cols].to_numpy(np.float64)
    Dte = frame["Delta"].loc[te_valid_exp, d_cols].to_numpy(np.float64)
    d_model, d_xs, d_ys = _fit_mlp(Xtr_v, Dtr, hidden=(256, 128, 64))
    Dpred = _pred_mlp(d_model, d_xs, d_ys, Xte_v)
    d_metrics = _reg_metrics(Dte, Dpred)

    # Izarg
    iz_train = frame["Izarg"].loc[tr_valid_exp]
    iz_cols = list(iz_train.columns)
    Itr = iz_train.to_numpy(np.float64)
    Ite = frame["Izarg"].loc[te_valid_exp, iz_cols].to_numpy(np.float64)
    iz_model, iz_xs, iz_ys = _fit_mlp(Xtr_v, Itr, hidden=(256, 128), max_iter=400)
    Ipred = _pred_mlp(iz_model, iz_xs, iz_ys, Xte_v)
    iz_metrics = _reg_metrics(Ite, Ipred)

    # I_max = I_zarg/100 * I_доп. Номинальный I_доп по линии - из train-валидных.
    idop_nominal = frame["Idop"].loc[tr_valid_exp].mean(axis=0)
    idop_nom_vec = idop_nominal.reindex(iz_cols).to_numpy(np.float64)
    Imax_true = frame["Imax"].loc[te_valid_exp, iz_cols].to_numpy(np.float64)
    Imax_from_nom = Ipred * idop_nom_vec / 100.0
    imax_metrics = _reg_metrics(Imax_true, Imax_from_nom)
    # верхняя граница: если истинный I_доп доступен (обычно только для валидных)
    idop_true = frame["Idop"].loc[te_valid_exp, iz_cols].to_numpy(np.float64)
    Imax_from_true = Ipred * idop_true / 100.0
    imax_metrics_true_idop = _reg_metrics(Imax_true, Imax_from_true)

    _log(f"[4/6] V      : {v_metrics}")
    _log(f"      Delta  : {d_metrics}")
    _log(f"      Izarg  : {iz_metrics}")
    _log(f"      Imax(nom I_доп): {imax_metrics}")
    _log(f"      Imax(true I_доп): {imax_metrics_true_idop}")

    bundle = {
        "features": list(X.columns),
        "gen_input_nodes": GEN_INPUT_NODES,
        "clf": clf,
        "threshold": 0.5,
        "v_model": v_model, "v_xs": v_xs, "v_ys": v_ys,
        "v_cols": v_cols, "v_const_val": v_const_val,
        "d_model": d_model, "d_xs": d_xs, "d_ys": d_ys,
        "d_cols": d_cols, "d_const_val": d_const_val,
        "iz_model": iz_model, "iz_xs": iz_xs, "iz_ys": iz_ys,
        "iz_cols": iz_cols, "idop_nominal": idop_nominal,
        "meta": {
            "train_experiences": int(len(tr_idx)),
            "test_experiences": int(len(te_idx)),
            "test_size": float(args.test_size),
            "seed": int(args.seed),
        },
    }
    bundle_path = Path(args.out)
    if not bundle_path.is_absolute():
        bundle_path = ROOT / bundle_path
    joblib.dump(bundle, bundle_path)
    _log(f"[5/6] Веса сохранены: {bundle_path}")

    report = {
        "n_experiences": int(len(X)),
        "n_features": int(X.shape[1]),
        "split": {"train": int(len(tr_idx)), "test": int(len(te_idx)),
                  "test_size": float(args.test_size)},
        "classification": {
            "model": {"F1_macro": clf_f1, "accuracy": clf_acc, "AUC": clf_auc},
            "rule_sumPn": {"F1_macro": rule_f1, "threshold": best_t},
        },
        "regression_valid_only": {
            "V": v_metrics,
            "Delta": d_metrics,
            "Izarg": iz_metrics,
            "Imax_nominal_idop": imax_metrics,
            "Imax_true_idop": imax_metrics_true_idop,
        },
        "const_v_nodes": [int(c) for c in v_const],
        "const_delta_nodes": [int(c) for c in d_const],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report_path = ROOT / args.report
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"      Отчёт: {report_path}  (время обучения {report['elapsed_sec']} c)")

    # ---------------- Проверка на my_test (если есть разметка) ----------------
    test_nodes_path = Path(args.test_nodes)
    test_lines_path = Path(args.test_lines)
    if not test_nodes_path.is_absolute():
        test_nodes_path = ROOT / test_nodes_path
    if not test_lines_path.is_absolute():
        test_lines_path = ROOT / test_lines_path
    if test_nodes_path.exists() and test_lines_path.exists() and not args.no_test:
        _log("[6/6] Проверка на my_test...")
        tn, tl = load_data(test_nodes_path, test_lines_path)
        report["my_test"] = evaluate_on(nodes=tn, lines=tl, bundle=bundle)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"      my_test: {json.dumps(report['my_test'], ensure_ascii=False)}")

    return report


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------
def build_features_for(nodes: pd.DataFrame, lines: pd.DataFrame, feature_cols) -> pd.DataFrame:
    frame = build_experience_frame(nodes, lines)
    X = frame["X"]
    X = X.reindex(columns=feature_cols, fill_value=0.0)
    return frame, X


def predict(nodes: pd.DataFrame, lines: pd.DataFrame, bundle: dict,
            zero_invalid: bool = True):
    frame, X = build_features_for(nodes, lines, bundle["features"])
    exp = frame["exp"]

    prob = bundle["clf"].predict_proba(X)[:, list(bundle["clf"].classes_).index(1)]
    reg = pd.Series((prob >= bundle["threshold"]).astype(int), index=exp)
    # внутри опыта флаг одинаковый
    reg_nodes = nodes["ID_опыта"].map(reg).astype(int).to_numpy()
    reg_lines = lines["ID_опыта"].map(reg).astype(int).to_numpy()

    # V
    v_pred = _pred_mlp(bundle["v_model"], bundle["v_xs"], bundle["v_ys"], X)
    Vdf = pd.DataFrame(v_pred, index=exp, columns=bundle["v_cols"])
    for c, val in bundle["v_const_val"].items():
        Vdf[c] = val
    # Delta
    d_pred = _pred_mlp(bundle["d_model"], bundle["d_xs"], bundle["d_ys"], X)
    Ddf = pd.DataFrame(d_pred, index=exp, columns=bundle["d_cols"])
    for c, val in bundle["d_const_val"].items():
        Ddf[c] = val
    # токи
    iz_pred = _pred_mlp(bundle["iz_model"], bundle["iz_xs"], bundle["iz_ys"], X)
    Izdf = pd.DataFrame(iz_pred, index=exp, columns=bundle["iz_cols"])
    nom = bundle["idop_nominal"].reindex(bundle["iz_cols"]).to_numpy()

    def gather(df, exp_series, num_series):
        return df.reindex(exp_series.values).to_numpy()[
            np.arange(len(exp_series)), [df.columns.get_loc(int(n)) for n in num_series]
        ]

    pn = pd.DataFrame({"ID_опыта": nodes["ID_опыта"].to_numpy(), "Номер": nodes["Номер"].to_numpy()})
    pn["V"] = gather(Vdf, nodes["ID_опыта"], nodes["Номер"])
    pn["Delta"] = gather(Ddf, nodes["ID_опыта"], nodes["Номер"])
    pn["Regime_valid"] = reg_nodes

    pl = pd.DataFrame({"ID_опыта": lines["ID_опыта"].to_numpy(), "Номер": lines["Номер"].to_numpy()})
    pl["Izarg, %"] = gather(Izdf, lines["ID_опыта"], lines["Номер"])
    pl["Imax, A"] = pl["Izarg, %"].to_numpy() * nom[
        [Izdf.columns.get_loc(int(n)) for n in lines["Номер"]]
    ] / 100.0

    if zero_invalid:
        m_n = pn["Regime_valid"].to_numpy() == 0
        pn.loc[m_n, ["V", "Delta"]] = 0.0
        m_l = (pl["ID_опыта"].map(reg).to_numpy() == 0)
        pl.loc[m_l, ["Imax, A", "Izarg, %"]] = 0.0

    for df in (pn, pl):
        num = df.select_dtypes("number").columns
        df[num] = np.nan_to_num(df[num].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

    return pn[NODE_PRED_COLS], pl[LINE_PRED_COLS], reg


def evaluate_on(nodes: pd.DataFrame, lines: pd.DataFrame, bundle: dict) -> dict:
    pn, pl, reg = predict(nodes, lines, bundle, zero_invalid=False)
    out: dict = {}
    if "regime_valid" in nodes.columns:
        true_reg = nodes.groupby("ID_опыта")["regime_valid"].first().astype(int)
        pred_reg = reg.reindex(true_reg.index)
        out["flag"] = {
            "F1_macro": float(f1_score(true_reg, pred_reg, average="macro")),
            "accuracy": float(accuracy_score(true_reg, pred_reg)),
        }
    # регрессию сравниваем только на валидных опытах с разметкой
    valid_exp = set(nodes.loc[nodes["regime_valid"] == 1, "ID_опыта"].unique()) if "regime_valid" in nodes.columns else set()
    if valid_exp and "V" in nodes.columns:
        nm = nodes["ID_опыта"].isin(valid_exp)
        out["V"] = _reg_metrics(nodes.loc[nm, "V"].to_numpy(np.float64), pn.loc[nm, "V"].to_numpy(np.float64))
        out["Delta"] = _reg_metrics(nodes.loc[nm, "Delta"].to_numpy(np.float64), pn.loc[nm, "Delta"].to_numpy(np.float64))
    if valid_exp and "I_max" in lines.columns:
        lm = lines["ID_опыта"].isin(valid_exp)
        out["Imax"] = _reg_metrics(lines.loc[lm, "I_max"].to_numpy(np.float64), pl.loc[lm, "Imax, A"].to_numpy(np.float64))
        out["Izarg"] = _reg_metrics(lines.loc[lm, "I_zarg"].to_numpy(np.float64), pl.loc[lm, "Izarg, %"].to_numpy(np.float64))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Проверочный train (70/30 по ID_опыта)")
    p.add_argument("--nodes", type=Path, default=ROOT / "data" / "train_nodes_34_100000.csv")
    p.add_argument("--lines", type=Path, default=ROOT / "data" / "train_lines_34_100000.csv")
    p.add_argument("--test-nodes", type=Path, default=ROOT / "my_test" / "nodes.csv")
    p.add_argument("--test-lines", type=Path, default=ROOT / "my_test" / "lines.csv")
    p.add_argument("--out", type=str, default="weights_trial.joblib")
    p.add_argument("--report", type=str, default="train_report.json")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
