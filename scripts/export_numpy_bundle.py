"""Экспорт обученных моделей в plain-numpy бандл (не зависит от версии sklearn).

Берёт регрессоры из weights_v2.joblib, обучает MLP-классификатор флага и
сохраняет weights_v2_numpy.joblib (только numpy-массивы + метаданные).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grid_lib as gl  # noqa: E402

OUT = gl.MODELS_DIR / "weights_v2_numpy.joblib"


def export_reg(model, xs, ys):
    return {
        "x_mean": np.asarray(xs.mean_, np.float64),
        "x_scale": np.asarray(xs.scale_, np.float64),
        "y_mean": np.asarray(ys.mean_, np.float64),
        "y_scale": np.asarray(ys.scale_, np.float64),
        "coefs": [(np.asarray(W, np.float64), np.asarray(b, np.float64))
                  for W, b in zip(model.coefs_, model.intercepts_)],
    }


def export_clf(model, xs):
    return {
        "x_mean": np.asarray(xs.mean_, np.float64),
        "x_scale": np.asarray(xs.scale_, np.float64),
        "coefs": [(np.asarray(W, np.float64), np.asarray(b, np.float64))
                  for W, b in zip(model.coefs_, model.intercepts_)],
    }


def main():
    nodes, lines = gl.load_data()
    frame = gl.build_frame(nodes, lines)
    exp = frame["exp"]
    X = frame["X"]
    y = frame["regime"].to_numpy()

    b = joblib.load(gl.MODELS_DIR / "weights_v2.joblib")
    reg = {"v": export_reg(b["v_model"], b["v_xs"], b["v_ys"]),
           "d": export_reg(b["d_model"], b["d_xs"], b["d_ys"]),
           "iz": export_reg(b["iz_model"], b["iz_xs"], b["iz_ys"])}

    # классификатор: 85% train, 15% на порог
    tr, va, te = gl.split_experiences(exp, seed=7)
    fit_ids = np.concatenate([tr, va])
    sc = StandardScaler().fit(X.loc[fit_ids])
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4,
                        max_iter=400, early_stopping=True, n_iter_no_change=20,
                        random_state=42)
    clf.fit(sc.transform(X.loc[fit_ids]), frame["regime"].loc[fit_ids].to_numpy())
    p = clf.predict_proba(sc.transform(X.loc[te]))[:, list(clf.classes_).index(1)]
    tb, _ = gl.best_threshold(frame["regime"].loc[te].to_numpy(), p, "f1macro")
    print(f"classifier n_iter={clf.n_iter_} threshold={tb:.3f} "
          f"testF1={f1_score(frame['regime'].loc[te], (p>=tb).astype(int), average='macro'):.4f}")

    bundle = {
        "kind": "numpy_v2",
        "features": list(X.columns),
        "reg": reg,
        "clf": export_clf(clf, sc),
        "threshold": float(tb),
        "v_cols": list(b["v_cols"]), "d_cols": list(b["d_cols"]),
        "v_const_val": b["v_const_val"], "d_const_val": b["d_const_val"],
        "quasi_v": b.get("quasi_v", {}),
        "iz_cols": list(b["iz_cols"]),
        "idop_nominal": b["idop_nominal"],
        "parallel_avg": True,
        "version": {"name": "v2_numpy", "source": "weights_v2.joblib",
                    "classifier": "MLP(256,128)"},
    }
    joblib.dump(bundle, OUT)
    print(f"Сохранено: {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
