"""Инференс v2 с проверкой входа и numpy-весами в формате организаторов.

Запуск: python baseline_solution.py --nodes test_nodes.csv --lines test_lines.csv --out-dir predictions
Отсутствие или несовместимость весов — ошибка, а не успешная выдача заглушки.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
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


def validate(nodes_n, lines_n, pn, pl):
    from grid_validation import validate_predictions
    validate_predictions(nodes_n, lines_n, pn, pl, NODE_PRED_COLS, LINE_PRED_COLS)


def main():
    ap = argparse.ArgumentParser(description="Прогноз режима сети (v2)")
    ap.add_argument("--nodes", type=Path, required=True)
    ap.add_argument("--lines", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = ap.parse_args()
    nodes = normalize(pd.read_csv(args.nodes), NODE_ALIASES)
    lines = normalize(pd.read_csv(args.lines), LINE_ALIASES)

    wpath = Path(args.weights)
    if not wpath.exists() and wpath == DEFAULT_WEIGHTS:
        wpath = FALLBACK_WEIGHTS
        print("[warn] sklearn-веса требуют совместимой версии sklearn", file=sys.stderr)
    if not wpath.is_file():
        raise ValueError(f"Файл весов не найден: {wpath}")
    try:
        # joblib/pickle допускает исполнение кода: только доверенные файлы весов.
        bundle = joblib.load(wpath)
    except Exception as exc:
        raise ValueError(f"Не удалось загрузить веса {wpath}: {exc}") from exc
    pn, pl, _, _ = grid_mlp.predict_bundle(nodes, lines, bundle, zero_invalid=True)
    validate(nodes, lines, pn, pl)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pn.to_csv(args.out_dir / "pred_nodes.csv", index=False)
    pl.to_csv(args.out_dir / "pred_lines.csv", index=False)
    print(f"Готово: {args.out_dir/'pred_nodes.csv'}, {args.out_dir/'pred_lines.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
