"""Экспорт регрессоров и независимо проверенного классификатора в numpy-бандл.

Регрессоры источника обучены на всех данных. Holdout ниже относится ТОЛЬКО
к классификатору, не является независимой оценкой всего бандла.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
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


def fit_evaluated_classifier(X, y, tr, va, te, classifier=None):
    """Scalers/fit: train, threshold: validation, final metrics: test.

    Не дообучает классификатор после оценки: сохранённая модель соответствует
    измеренной. Внутренний early_stopping использует только часть train.
    """
    groups = [set(ids) for ids in (tr, va, te)]
    if any(not ids for ids in groups):
        raise ValueError("Для оценки нужны непустые train/val/test")
    if any(len(ids) != len(group) for ids, group in zip((tr, va, te), groups)):
        raise ValueError("Повторяющиеся ID внутри сплита")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Train/val/test пересекаются")
    if not X.index.is_unique or not y.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X/y должны иметь совпадающие уникальные ID")
    if not all(group <= set(X.index) for group in groups):
        raise ValueError("Сплит содержит неизвестные ID")
    if not y.isin([0, 1]).all() or y.loc[tr].nunique() < 2:
        raise ValueError("Train должен содержать оба класса 0/1")
    sc = StandardScaler().fit(X.loc[tr])
    clf = classifier if classifier is not None else MLPClassifier(
        hidden_layer_sizes=(256, 128), activation="relu", alpha=1e-4,
        max_iter=400, early_stopping=True, n_iter_no_change=20, random_state=42)
    clf.fit(sc.transform(X.loc[tr]), y.loc[tr].to_numpy())

    def probabilities(ids):
        return clf.predict_proba(sc.transform(X.loc[ids]))[:, list(clf.classes_).index(1)]

    pva = probabilities(va)
    if y.loc[va].nunique() >= 2:
        threshold, _ = gl.best_threshold(y.loc[va].to_numpy(), pva, "f1macro")
        source = "validation"
    else:
        threshold, source = 0.5, "fixed_0.5_single_class_validation"
    test = gl.clf_metrics(y.loc[te].to_numpy(), probabilities(te), threshold)
    evaluation = {
        "scope": "classifier_only; source regressors were fitted on all data",
        "split": {"train": len(tr), "validation": len(va), "test": len(te)},
        "threshold_source": source,
        "test": test,
    }
    return clf, sc, threshold, evaluation


def main():
    nodes, lines = gl.load_data()
    frame = gl.build_frame(nodes, lines)
    X = frame["X"]
    b = joblib.load(gl.MODELS_DIR / "weights_v2.joblib")
    if list(X.columns) != list(b["features"]):
        raise ValueError("Признаки регрессоров не совпадают с данными экспорта")
    reg = {"v": export_reg(b["v_model"], b["v_xs"], b["v_ys"]),
           "d": export_reg(b["d_model"], b["d_xs"], b["d_ys"]),
           "iz": export_reg(b["iz_model"], b["iz_xs"], b["iz_ys"])}
    tr, va, te = gl.split_experiences(frame["exp"], seed=7)
    clf, sc, threshold, evaluation = fit_evaluated_classifier(X, frame["regime"], tr, va, te)
    evaluation["split_seed"] = 7
    print(f"classifier n_iter={clf.n_iter_} threshold={threshold:.3f} "
          f"independent_test={evaluation['test']}")
    bundle = {
        "kind": "numpy_v2", "features": list(X.columns), "reg": reg,
        "clf": export_clf(clf, sc), "threshold": float(threshold),
        "v_cols": list(b["v_cols"]), "d_cols": list(b["d_cols"]),
        "v_const_val": b["v_const_val"], "d_const_val": b["d_const_val"],
        "quasi_v": b.get("quasi_v", {}), "iz_cols": list(b["iz_cols"]),
        "idop_nominal": b["idop_nominal"], "parallel_avg": True,
        "version": {"name": "v2_numpy", "source": "weights_v2.joblib",
                    "classifier": "MLP(256,128)"},
        "classifier_evaluation": evaluation,
    }
    from grid_validation import validate_numpy_bundle
    validate_numpy_bundle(bundle)
    joblib.dump(bundle, OUT)
    print(f"Сохранено: {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
