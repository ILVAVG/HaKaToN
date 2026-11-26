"""Решение хакатона v2: загрузка weights_v2.joblib и выдача в формате организаторов.

Запуск:
    python baseline_solution.py --nodes test_nodes.csv --lines test_lines.csv --out-dir predictions

Если weights_v2.joblib отсутствует — тривиальный фоллбэк.
Формат выхода:
    pred_nodes.csv: ID_опыта, Номер, V, Delta, Regime_valid
    pred_lines.csv: ID_опыта, Номер, Imax, A, Izarg, %
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import grid_lib as gl  # noqa: E402
import grid_mlp  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "models" / "weights_v2_numpy.joblib"
FALLBACK_WEIGHTS = ROOT / "models" / "weights_v2.joblib"
NODE_PRED_COLS = gl.NODE_PRED_COLS
LINE_PRED_COLS = gl.LINE_PRED_COLS

# Канонические имена (как в train). Ключ — нормализованное имя.
NODE_ALIASES = {
    "idопыта": "ID_опыта", "номер": "Номер", "uном": "U_ном", "уном": "U_ном",
    "pн": "P_н", "qн": "Q_н", "pг": "P_г",
    "v": "V", "delta": "Delta", "regimevalid": "regime_valid",
}
LINE_ALIASES = {
    "idопыта": "ID_опыта", "номер": "Номер", "ns": "N_s", "ne": "N_e", "type": "type",
    "r": "R", "x": "X", "b": "B", "kt": "Kt",
    "imax": "I_max", "iдоп": "I_доп", "izarg": "I_zarg", "izagr": "I_zarg",
}


def _key(name: str) -> str:
    return (str(name).replace("_", "").replace(" ", "").replace(",", "")
            .replace("%", "").replace(".", "").lower())


def normalize(df: pd.DataFrame, aliases: dict) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        k = _key(c)
        if k in aliases:
            rename[c] = aliases[k]
    return df.rename(columns=rename)


def trivial(nodes, lines):
    U = nodes["U_ном"] if "U_ном" in nodes.columns else pd.Series(0.0, index=nodes.index)
    pn = pd.DataFrame({"ID_опыта": nodes["ID_опыта"].to_numpy(),
                       "Номер": nodes["Номер"].to_numpy(),
                       "V": np.asarray(U, float),
                       "Delta": 0.0, "Regime_valid": 1})
    pl = pd.DataFrame({"ID_опыта": lines["ID_опыта"].to_numpy(),
                       "Номер": lines["Номер"].to_numpy(),
                       "Imax, A": 0.0, "Izarg, %": 0.0})
    return pn[NODE_PRED_COLS], pl[LINE_PRED_COLS]


def validate(nodes_n, lines_n, pn, pl):
    assert list(pn.columns) == NODE_PRED_COLS, f"pred_nodes columns {list(pn.columns)}"
    assert list(pl.columns) == LINE_PRED_COLS, f"pred_lines columns {list(pl.columns)}"
    assert len(pn) == len(nodes_n), "nodes length mismatch"
    assert len(pl) == len(lines_n), "lines length mismatch"
    assert np.isfinite(pn[["V", "Delta"]].to_numpy(float)).all(), "NaN/inf in V/Delta"
    assert np.isfinite(pl[["Imax, A", "Izarg, %"]].to_numpy(float)).all(), "NaN/inf in currents"
    assert set(pn["Regime_valid"].unique()).issubset({0, 1}), "Regime_valid must be 0/1"
    assert pn.groupby("ID_опыта")["Regime_valid"].nunique().max() == 1, "flag varies within exp"
    assert np.array_equal(pn[["ID_опыта", "Номер"]].to_numpy(),
                          nodes_n[["ID_опыта", "Номер"]].to_numpy()), "node keys mismatch"
    assert np.array_equal(pl[["ID_опыта", "Номер"]].to_numpy(),
                          lines_n[["ID_опыта", "Номер"]].to_numpy()), "line keys mismatch"


def main():
    ap = argparse.ArgumentParser(description="Прогноз режима сети (v2)")
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--lines", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    nodes = normalize(pd.read_csv(args.nodes), NODE_ALIASES)
    lines = normalize(pd.read_csv(args.lines), LINE_ALIASES)

    wpath = Path(args.weights)
    if not wpath.exists():
        wpath = FALLBACK_WEIGHTS if wpath == DEFAULT_WEIGHTS else wpath
    bundle = None
    if wpath.exists():
        try:
            bundle = joblib.load(wpath)
        except Exception as exc:  # несовместимость версий sklearn и т.п.
            print(f"[warn] не удалось загрузить {wpath}: {exc}")
            bundle = None
    if bundle is not None:
        pn, pl, _, _ = grid_mlp.predict_bundle(nodes, lines, bundle, zero_invalid=True)
    else:
        print(f"[warn] веса не найдены/не загружены — тривиальный фоллбэк")
        pn, pl = trivial(nodes, lines)

    validate(nodes, lines, pn, pl)
    pn.to_csv(args.out_dir / "pred_nodes.csv", index=False)
    pl.to_csv(args.out_dir / "pred_lines.csv", index=False)
    print(f"Готово: {args.out_dir/'pred_nodes.csv'}, {args.out_dir/'pred_lines.csv'}")


if __name__ == "__main__":
    main()
