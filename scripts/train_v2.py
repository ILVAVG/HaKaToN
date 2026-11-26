"""Финальное обучение v2: sklearn-MLP + HGB + пост-обработка.

Обучает на ВСЕХ данных (для сабмита) и сохраняет weights_v2.joblib.
Честные holdout-метрики см. в stage1_results.json / exp_stage4.py.

Запуск:
  python train_v2.py                 # AFTER-конфиг, все данные
  python train_v2.py --config medium
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grid_lib as gl  # noqa: E402
import grid_mlp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["after", "medium", "before"], default="after")
    ap.add_argument("--out", default="weights_v2.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    nodes, lines = gl.load_data()
    frame = gl.build_frame(nodes, lines)
    exp = frame["exp"]
    print(f"Опытов: {len(exp)}, валидных: {int(frame['regime'].sum())}", flush=True)

    config = {"after": grid_mlp.AFTER, "medium": grid_mlp.MEDIUM, "before": grid_mlp.BEFORE}[args.config]
    print(f"Обучение на ВСЕХ опытах, конфиг={args.config} ...", flush=True)
    bundle = grid_mlp.fit_bundle(frame, exp, config=config, seed=args.seed,
                                 parallel_avg=True, snap=True, verbose=True)
    bundle["version"] = {"name": "v2_sklearn", "config": args.config, "seed": args.seed,
                         "trained_experiences": int(len(exp))}
    out = Path(args.out) if Path(args.out).is_absolute() else gl.MODELS_DIR / args.out
    joblib.dump(bundle, out)
    print(f"Сохранено: {out} ({out.stat().st_size/1e6:.1f} MB), {time.time()-t0:.0f} c", flush=True)

    # Проверка/демо на my_test (он входит в train -> метрики оптимистичны!)
    tn = gl.MY_TEST_DIR / "nodes.csv"
    tl = gl.MY_TEST_DIR / "lines.csv"
    if tn.exists() and tl.exists():
        nd = pd.read_csv(tn)
        ld = pd.read_csv(tl)
        pn, pl, reg, _ = grid_mlp.predict_bundle(nd, ld, bundle, zero_invalid=False)
        pred_dir = gl.PREDICTIONS_DIR
        pred_dir.mkdir(exist_ok=True)
        pn.to_csv(pred_dir / "pred_nodes.csv", index=False)
        pl.to_csv(pred_dir / "pred_lines.csv", index=False)
        print(f"predictions/ обновлены на my_test: {len(pn)} узлов, {len(pl)} линий")


if __name__ == "__main__":
    main()
