"""
Решение для хакатона: прогноз установившегося режима электрической сети.

Запуск автопроверкой:
    python baseline_solution.py --nodes test_nodes.csv --lines test_lines.csv --out-dir predictions

Если рядом лежит файл weights.joblib (его создаёт train.py) - используется обученная модель.
Если весов нет - работает тривиальный бейзлайн (V = Uном, нули, Regime_valid = 1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Названия выходных столбцов. ВАЖНО: сверьте с шаблоном организаторов!
# --------------------------------------------------------------------------
NODE_PRED_COLS = ["ID_опыта", "Номер", "V", "Delta", "Regime_valid"]
LINE_PRED_COLS = ["ID_опыта", "Номер", "Imax, A", "Izarg, %"]

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights.joblib"

# --------------------------------------------------------------------------
# Нормализация названий столбцов: в разных файлах/таблицах задания они
# пишутся по-разному (U_ном / Uном, I_max / Imax, regime_valid / Regime_valid).
# Внутри кода используем короткие внутренние имена.
# --------------------------------------------------------------------------
NODE_ALIASES = {
    "idопыта": "exp", "номер": "num",
    "uном": "U", "pн": "Pn", "qн": "Qn", "pг": "Pg",
    "v": "V", "delta": "Delta", "regimevalid": "Regime",
}
LINE_ALIASES = {
    "idопыта": "exp", "номер": "num",
    "ns": "Ns", "ne": "Ne", "type": "type",
    "r": "R", "x": "X", "b": "B", "kt": "Kt",
    "iдоп": "Idop", "iдопa": "Idop", "iдопа": "Idop",
    "imax": "Imax", "imaxa": "Imax",
    "izarg": "Izarg", "izagr": "Izarg",
}


def _key(name: str) -> str:
    return str(name).replace("_", "").replace(" ", "").replace(",", "").replace("%", "").lower()


def normalize(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    rename = {c: aliases[_key(c)] for c in df.columns if _key(c) in aliases}
    return df.rename(columns=rename)


# --------------------------------------------------------------------------
# Признаки
# --------------------------------------------------------------------------
def node_features(nodes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Признаки по узлам. Порядок строк сохраняется (как во входном nodes)."""
    n = nodes[["exp", "num", "U", "Pn", "Qn", "Pg"]].copy()
    n["inj"] = n["Pg"] - n["Pn"]  # чистая инжекция мощности
    n["is_load"] = (n["Pn"] > 0).astype(int)
    n["is_gen"] = (n["Pg"] > 0).astype(int)

    # агрегаты по опыту (по узлам)
    agg = n.groupby("exp").agg(
        sum_Pn=("Pn", "sum"), sum_Qn=("Qn", "sum"), sum_Pg=("Pg", "sum"),
        max_Pn=("Pn", "max"), max_Pg=("Pg", "max"),
        n_load=("is_load", "sum"), n_gen=("is_gen", "sum"),
    ).reset_index()
    agg["balance"] = agg["sum_Pg"] - agg["sum_Pn"]

    # агрегаты по опыту (по линиям)
    l = lines.copy()
    l["z"] = np.hypot(l["R"], l["X"]).clip(lower=1e-6)
    l["y"] = 1.0 / l["z"]
    l["absB"] = l["B"].abs()
    lagg = l.groupby("exp").agg(
        sum_R=("R", "sum"), sum_X=("X", "sum"), sum_absB=("absB", "sum"),
        n_lines=("y", "size"), mean_z=("z", "mean"),
    ).reset_index()

    # признаки окрестности узла: степень, суммарная проводимость, инжекция соседей
    inj_other = n[["exp", "num", "inj"]].rename(columns={"num": "other", "inj": "inj_other"})

    def side(a: str, b: str) -> pd.DataFrame:
        t = l[["exp", a, b, "y", "absB"]].rename(columns={a: "node", b: "other"})
        return t.merge(inj_other, on=["exp", "other"], how="left")

    ends = pd.concat([side("Ns", "Ne"), side("Ne", "Ns")], ignore_index=True)
    nb = ends.groupby(["exp", "node"]).agg(
        deg=("y", "size"), sum_y=("y", "sum"),
        sum_absB_nb=("absB", "sum"), sum_inj_nb=("inj_other", "sum"),
    ).reset_index().rename(columns={"node": "num"})

    out = (
        n.merge(agg, on="exp", how="left")
         .merge(lagg, on="exp", how="left")
         .merge(nb, on=["exp", "num"], how="left")
    )
    for c in ["deg", "sum_y", "sum_absB_nb", "sum_inj_nb"]:
        out[c] = out[c].fillna(0.0)
    out["Pn_rel"] = out["Pn"] / (out["sum_Pn"] + 1e-6)
    out["Pg_rel"] = out["Pg"] / (out["sum_Pg"] + 1e-6)
    return out


NODE_SIDE_COLS = ["U", "Pn", "Qn", "Pg", "inj", "deg", "sum_y", "sum_absB_nb", "sum_inj_nb"]
GLOBAL_COLS = ["sum_Pn", "sum_Qn", "sum_Pg", "balance", "sum_R", "sum_X", "sum_absB", "n_lines"]


def line_features(nodes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Признаки по ветвям. Порядок строк сохраняется (как во входном lines)."""
    nf = node_features(nodes, lines)
    l = lines.copy()
    l["z"] = np.hypot(l["R"], l["X"])
    if "Idop" not in l.columns:
        l["Idop"] = np.nan

    keep = ["exp", "num", "Ns", "Ne", "type", "R", "X", "B", "Kt", "Idop", "z"]
    l = l[keep]
    for side in ("Ns", "Ne"):
        t = nf[["exp", "num"] + NODE_SIDE_COLS].rename(
            columns={"num": side, **{c: f"{side}_{c}" for c in NODE_SIDE_COLS}}
        )
        l = l.merge(t, on=["exp", side], how="left")  # узел 999 и т.п. -> NaN
    g = nf.drop_duplicates("exp")[["exp"] + GLOBAL_COLS]
    return l.merge(g, on="exp", how="left")


def regime_features(nodes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Признаки на уровне опыта (одна строка = один ID_опыта), индекс = exp."""
    nf = node_features(nodes, lines)
    g = nf.groupby("exp")[GLOBAL_COLS + ["max_Pn", "max_Pg", "n_load", "n_gen", "mean_z"]].first()

    pn = nodes.pivot_table(index="exp", columns="num", values=["Pn", "Qn", "Pg"], aggfunc="first")
    pn.columns = [f"{a}_{b}" for a, b in pn.columns]

    lcols = [c for c in ["R", "X", "B", "Kt", "Idop"] if c in lines.columns]
    pl = lines.pivot_table(index="exp", columns="num", values=lcols, aggfunc="first")
    pl.columns = [f"L{a}_{b}" for a, b in pl.columns]
    return g.join(pn).join(pl)


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------
class PowerGridBaseline:
    """Загружает обученные модели (weights.joblib) и делает предсказания."""

    def __init__(self, weights_path: Path | None = None):
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self.bundle = None

    def load(self) -> None:
        if self.weights_path.exists():
            self.bundle = joblib.load(self.weights_path)
        else:
            print(f"[warn] веса {self.weights_path} не найдены - используется тривиальный бейзлайн")

    # ---- тривиальный вариант без обучения
    def _predict_trivial(self, nodes: pd.DataFrame, lines: pd.DataFrame):
        pn = pd.DataFrame({"ID_опыта": nodes["exp"].values, "Номер": nodes["num"].values})
        pn["V"] = nodes["U"].astype(float).values
        pn["Delta"] = 0.0
        pn["Regime_valid"] = 1
        pl = pd.DataFrame({"ID_опыта": lines["exp"].values, "Номер": lines["num"].values})
        pl["Imax, A"] = 0.0
        pl["Izarg, %"] = 0.0
        return pn, pl

    # ---- обученная модель
    def predict(self, nodes_df: pd.DataFrame, lines_df: pd.DataFrame):
        nodes = normalize(nodes_df, NODE_ALIASES)
        lines = normalize(lines_df, LINE_ALIASES)
        if self.bundle is None:
            return self._predict_trivial(nodes, lines)
        b = self.bundle

        # 1) сходимость режима (на уровне ID_опыта)
        Xr = regime_features(nodes, lines).reindex(columns=b["regime_cols"])
        clf = b["clf"]
        if 1 in list(clf.classes_):
            p = clf.predict_proba(Xr)[:, list(clf.classes_).index(1)]
        else:
            p = np.zeros(len(Xr))
        regime = pd.Series((p >= b.get("threshold", 0.5)).astype(int), index=Xr.index)

        # 2) V и Delta по узлам
        nf = node_features(nodes, lines)
        Xn = nf[b["node_cols"]]
        V = nodes["U"].values * b["m_Vrel"].predict(Xn)
        Delta = b["m_Delta"].predict(Xn)

        pn = pd.DataFrame({"ID_опыта": nodes["exp"].values, "Номер": nodes["num"].values})
        pn["V"] = V
        pn["Delta"] = Delta
        pn["Regime_valid"] = nodes["exp"].map(regime).fillna(1).astype(int).values

        # 3) токи по ветвям
        lf = line_features(nodes, lines)
        Xl = lf[b["line_cols"]]
        izarg = np.clip(b["m_Izarg"].predict(Xl), 0, None)
        imax_model = np.clip(b["m_Imax"].predict(Xl), 0, None)
        idop = lf["Idop"].values
        # I_zarg = I_max / I_доп * 100 -> если I_доп известен, берём согласованное значение
        imax = np.where(np.isfinite(idop), izarg * idop / 100.0, imax_model)

        pl = pd.DataFrame({"ID_опыта": lines["exp"].values, "Номер": lines["num"].values})
        pl["Imax, A"] = imax
        pl["Izarg, %"] = izarg

        for df in (pn, pl):  # страховка от NaN/inf
            num_cols = df.select_dtypes("number").columns
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pn[NODE_PRED_COLS], pl[LINE_PRED_COLS]


# --------------------------------------------------------------------------
# Валидация формата
# --------------------------------------------------------------------------
def validate_predictions(nodes_n, lines_n, pred_nodes, pred_lines) -> None:
    assert list(pred_nodes.columns) == NODE_PRED_COLS, f"Столбцы pred_nodes: {list(pred_nodes.columns)}"
    assert list(pred_lines.columns) == LINE_PRED_COLS, f"Столбцы pred_lines: {list(pred_lines.columns)}"
    assert len(pred_nodes) == len(nodes_n), "Число строк pred_nodes не совпадает с входом"
    assert len(pred_lines) == len(lines_n), "Число строк pred_lines не совпадает с входом"
    assert np.isfinite(pred_nodes.drop(columns=NODE_PRED_COLS[:2]).to_numpy(float)).all(), "NaN/inf в pred_nodes"
    assert np.isfinite(pred_lines.drop(columns=LINE_PRED_COLS[:2]).to_numpy(float)).all(), "NaN/inf в pred_lines"
    assert set(pred_nodes["Regime_valid"].unique()).issubset({0, 1}), "Regime_valid должен быть 0/1"
    assert (pred_nodes.groupby("ID_опыта")["Regime_valid"].nunique() == 1).all(), \
        "Regime_valid должен быть одинаков внутри ID_опыта"
    assert np.array_equal(pred_nodes[["ID_опыта", "Номер"]].to_numpy(), nodes_n[["exp", "num"]].to_numpy()), \
        "Ключи pred_nodes не совпадают с test_nodes"
    assert np.array_equal(pred_lines[["ID_опыта", "Номер"]].to_numpy(), lines_n[["exp", "num"]].to_numpy()), \
        "Ключи pred_lines не совпадают с test_lines"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Прогноз установившегося режима сети")
    p.add_argument("--nodes", type=Path, required=True, help="Путь к test_nodes.csv")
    p.add_argument("--lines", type=Path, required=True, help="Путь к test_lines.csv")
    p.add_argument("--out-dir", type=Path, required=True, help="Куда сохранить pred_*.csv")
    p.add_argument("--weights", type=Path, default=None, help="Файл весов (по умолчанию weights.joblib рядом)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nodes_df = pd.read_csv(args.nodes)
    lines_df = pd.read_csv(args.lines)

    model = PowerGridBaseline(weights_path=args.weights)
    model.load()
    pred_nodes, pred_lines = model.predict(nodes_df, lines_df)

    validate_predictions(
        normalize(nodes_df, NODE_ALIASES), normalize(lines_df, LINE_ALIASES), pred_nodes, pred_lines
    )
    pred_nodes.to_csv(args.out_dir / "pred_nodes.csv", index=False)
    pred_lines.to_csv(args.out_dir / "pred_lines.csv", index=False)
    print(f"Готово: {args.out_dir / 'pred_nodes.csv'}, {args.out_dir / 'pred_lines.csv'}")


if __name__ == "__main__":
    main()
