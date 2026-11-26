"""Проверки границ инференса; не импортирует sklearn и не читает целевые поля."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["ID_опыта", "Номер"]
GEOMETRY = ["N_s", "N_e", "type", "R", "X", "B", "Kt"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _numeric(df, columns, table, integers=()):
    for col in columns:
        try:
            values = pd.to_numeric(df[col], errors="raise")
            _require(np.isfinite(values.to_numpy(dtype=float)).all(),
                     f"{table}: NaN/inf в {col}")
            if col in integers:
                _require(((values % 1) == 0).all(), f"{table}: {col} должен быть целым")
                values = values.astype("int64")
            df[col] = values
        except (TypeError, OverflowError) as exc:
            raise ValueError(f"{table}: некорректный столбец {col}") from exc


def validate_inputs(nodes, lines, bundle):
    """Возвращает числовые копии входов. Узел 999 необязателен: модель его не использует."""
    nodes, lines = nodes.copy(), lines.copy()
    for df, name, cols in ((nodes, "nodes", KEYS + ["U_ном", "P_н", "Q_н", "P_г"]),
                           (lines, "lines", KEYS + GEOMETRY)):
        _require(df.columns.is_unique, f"{name}: дублирующиеся имена столбцов")
        _require(not df.empty, f"{name}: пустая таблица")
        _require(set(cols) <= set(df.columns), f"{name}: отсутствуют столбцы {sorted(set(cols) - set(df.columns))}")
        _numeric(df, KEYS, name, KEYS)
        _require(not df.duplicated(KEYS).any(), f"{name}: повтор ключа ID_опыта/Номер")
    _numeric(nodes, ["U_ном", "P_н", "Q_н"], "nodes")
    # Генерация балансирующего узла — результат расчёта, не входной признак.
    nodes.loc[nodes["Номер"] == 999, "P_г"] = 0.0
    _numeric(nodes, ["P_г"], "nodes")
    _numeric(lines, GEOMETRY, "lines", ["N_s", "N_e", "type"])
    _require((nodes["U_ном"] > 0).all(), "nodes: U_ном должен быть положительным")
    _require(set(nodes["ID_опыта"]) == set(lines["ID_опыта"]), "Не совпадают ID опытов nodes/lines")

    expected_nodes = set(bundle["v_cols"]) | set(bundle["v_const_val"])
    delta_nodes = set(bundle["d_cols"]) | set(bundle["d_const_val"])
    _require(expected_nodes == delta_nodes, "Несогласованные выходы V/Delta в весах")
    feature_nodes = {int(f.split("_", 1)[1]) for f in bundle["features"]
                     if f.startswith(("Pn_", "Pg_"))}
    expected_lines = set(bundle["iz_cols"])
    for _, group in nodes.groupby("ID_опыта", sort=False):
        actual = set(group["Номер"])
        missing = (feature_nodes | (expected_nodes - {999})) - actual
        _require(not missing and actual <= expected_nodes,
                 f"nodes: неполный/неизвестный набор узлов, отсутствуют {sorted(missing)}")
    for _, group in lines.groupby("ID_опыта", sort=False):
        _require(set(group["Номер"]) == expected_lines, "lines: неполный/неизвестный набор линий")
    _require(set(lines["N_s"]) | set(lines["N_e"]) <= expected_nodes,
             "lines: неизвестные узлы на концах линий")
    # parallel_groups использует геометрию первого опыта; запретим скрытые изменения.
    variation = lines.groupby("Номер")[GEOMETRY].nunique()
    _require(not (variation > 1).any().any(), "lines: геометрия линии меняется между опытами")
    return nodes, lines


def validate_numpy_bundle(bundle):
    """Проверяет размерности и конечность параметров до матричных операций."""
    if not isinstance(bundle, dict):
        raise ValueError("Веса должны содержать словарь")
    if bundle.get("kind") != "numpy_v2":
        return  # sklearn-бандл загружается только в совместимом окружении.
    try:
        features = bundle["features"]
        _require(len(features) > 0 and len(features) == len(set(features)), "Некорректные признаки весов")
        _require(np.isfinite(bundle["threshold"]) and 0 <= bundle["threshold"] <= 1,
                 "Некорректный порог весов")
        for name, md, output in [("clf", bundle["clf"], 1)] + [
                (k, bundle["reg"][k], len(bundle[c]))
                for k, c in (("v", "v_cols"), ("d", "d_cols"), ("iz", "iz_cols"))]:
            width = len(features)
            for key in ("x_mean", "x_scale"):
                a = np.asarray(md[key], dtype=float)
                _require(a.shape == (width,) and np.isfinite(a).all(), f"{name}: неверный {key}")
            _require((np.asarray(md["x_scale"]) > 0).all(), f"{name}: нулевой масштаб")
            _require(len(md["coefs"]) > 0, f"{name}: нет слоёв")
            for W, b in md["coefs"]:
                W, b = np.asarray(W), np.asarray(b)
                _require(W.ndim == 2 and W.shape[0] == width and b.shape == (W.shape[1],)
                         and np.isfinite(W).all() and np.isfinite(b).all(), f"{name}: неверный слой")
                width = W.shape[1]
            _require(width == output, f"{name}: неверное число выходов")
            if name != "clf":
                for key in ("y_mean", "y_scale"):
                    a = np.asarray(md[key], dtype=float)
                    _require(a.shape == (output,) and np.isfinite(a).all(), f"{name}: неверный {key}")
        nom = bundle["idop_nominal"].reindex(bundle["iz_cols"]).to_numpy(float)
        _require(np.isfinite(nom).all() and (nom > 0).all(), "Некорректный номинальный I_доп")
        for key in ("v_const_val", "d_const_val", "quasi_v"):
            _require(all(np.isfinite(v) for v in bundle.get(key, {}).values()), f"Некорректный {key}")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"Несовместимый numpy-бандл: {exc}") from exc


def validate_predictions(nodes, lines, pn, pl, node_cols, line_cols):
    for src, pred, cols, values in ((nodes, pn, node_cols, ["V", "Delta"]),
                                    (lines, pl, line_cols, ["Imax, A", "Izarg, %"])):
        _require(list(pred.columns) == cols and len(src) == len(pred), "Неверный формат предсказаний")
        _require(np.array_equal(src[KEYS].to_numpy(), pred[KEYS].to_numpy()), "Нарушен порядок ключей")
        _require(np.isfinite(pred[values].to_numpy(float)).all(), "NaN/inf в предсказаниях")
    _require(pn["Regime_valid"].isin([0, 1]).all(), "Regime_valid должен быть 0/1")
    _require((pn.groupby("ID_опыта")["Regime_valid"].nunique() == 1).all(), "Разные флаги внутри опыта")
    invalid = set(pn.loc[pn["Regime_valid"] == 0, "ID_опыта"])
    _require((pn.loc[pn["ID_опыта"].isin(invalid), ["V", "Delta"]] == 0).all().all(),
             "Невалидные узлы должны иметь нулевые выходы")
    _require((pl.loc[pl["ID_опыта"].isin(invalid), ["Imax, A", "Izarg, %"]] == 0).all().all(),
             "Невалидные линии должны иметь нулевые выходы")
