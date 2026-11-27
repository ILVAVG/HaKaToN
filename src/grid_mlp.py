"""sklearn-MLP пайплайн с пост-обработкой (параллельные линии, квази-константы)
и версионно-независимым numpy-инференсом."""
from __future__ import annotations

import numpy as np
import pandas as pd

import grid_lib as gl

BEFORE = {
    "v": dict(hidden=(256, 128, 64), max_iter=300, early_stopping=True, n_iter_no_change=15),
    "d": dict(hidden=(256, 128, 64), max_iter=300, early_stopping=True, n_iter_no_change=15),
    "iz": dict(hidden=(256, 128), max_iter=400, early_stopping=True, n_iter_no_change=15),
}
AFTER = {
    "v": dict(hidden=(256, 128, 64), max_iter=1500, early_stopping=True, validation_fraction=0.15,
              n_iter_no_change=40, learning_rate_init=5e-4, learning_rate="adaptive",
              batch_size=512, alpha=1e-6, tol=1e-7),
    "d": dict(hidden=(256, 128, 64), max_iter=1500, early_stopping=True, validation_fraction=0.15,
              n_iter_no_change=40, learning_rate_init=5e-4, learning_rate="adaptive",
              batch_size=512, alpha=1e-6, tol=1e-7),
    "iz": dict(hidden=(256, 128, 64), max_iter=1500, early_stopping=True, validation_fraction=0.15,
               n_iter_no_change=40, learning_rate_init=5e-4, learning_rate="adaptive",
               batch_size=512, alpha=1e-6, tol=1e-7),
}
MEDIUM = {
    "v": dict(hidden=(256, 128), max_iter=900, early_stopping=True, validation_fraction=0.15,
              n_iter_no_change=30, learning_rate_init=5e-4, learning_rate="adaptive",
              batch_size=512, alpha=1e-6, tol=1e-7),
    "d": dict(hidden=(256, 128), max_iter=900, early_stopping=True, validation_fraction=0.15,
              n_iter_no_change=30, learning_rate_init=5e-4, learning_rate="adaptive",
              batch_size=512, alpha=1e-6, tol=1e-7),
    "iz": dict(hidden=(256, 128), max_iter=900, early_stopping=True, validation_fraction=0.15,
               n_iter_no_change=30, learning_rate_init=5e-4, learning_rate="adaptive",
               batch_size=512, alpha=1e-6, tol=1e-7),
}


def _fit_mlp(Xtr, Ytr, seed=42, **kw):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    kw = dict(kw)
    if "hidden" in kw:
        kw["hidden_layer_sizes"] = kw.pop("hidden")
    model = MLPRegressor(random_state=seed, **kw)
    model.fit(xs.transform(Xtr), ys.transform(Ytr))
    return model, xs, ys


def _pred(model, xs, ys, X):
    return ys.inverse_transform(model.predict(xs.transform(X)))


# ---- версионно-независимый инференс из plain-numpy бандла ----
def np_forward_reg(md, X):
    z = (np.asarray(X, np.float64) - md["x_mean"]) / md["x_scale"]
    coefs = md["coefs"]
    for i, (W, b) in enumerate(coefs):
        z = z @ W + b                 # sklearn хранит coefs_ как (in, out)
        if i < len(coefs) - 1:
            z = np.maximum(z, 0.0)
    return z * md["y_scale"] + md["y_mean"]


def np_forward_prob(cd, X):
    z = (np.asarray(X, np.float64) - cd["x_mean"]) / cd["x_scale"]
    coefs = cd["coefs"]
    for i, (W, b) in enumerate(coefs):
        z = z @ W + b
        if i < len(coefs) - 1:
            z = np.maximum(z, 0.0)
    z = z.ravel()
    # Устойчивая сигмоида без переполнения exp.
    p = np.empty_like(z)
    positive = z >= 0
    p[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    p[~positive] = ez / (1.0 + ez)
    return p


def _const_cols(df, tol=1e-6):
    const, vals, varying = [], {}, []
    for c in df.columns:
        col = df[c]
        if col.nunique(dropna=True) <= 1 or float(col.std(ddof=0)) < tol:
            const.append(c); vals[c] = float(col.iloc[0])
        else:
            varying.append(c)
    return const, vals, varying


def fit_bundle(frame, tr_exp, config=None, seed=42, parallel_avg=True, snap=True,
               clf_max_iter=400, verbose=False):
    config = config or AFTER
    X = frame["X"]
    regime = frame["regime"]
    tr = np.asarray(tr_exp)
    ytr = regime.loc[tr].to_numpy()

    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(
        max_iter=clf_max_iter, learning_rate=0.06, early_stopping=True,
        validation_fraction=0.15, random_state=seed)
    clf.fit(X.loc[tr], ytr)

    tr_valid = tr[regime.loc[tr].to_numpy() == 1]
    Xv = X.loc[tr_valid]

    v_train = frame["V"].loc[tr_valid]
    v_const, v_const_val, v_cols = _const_cols(v_train)
    d_train = frame["Delta"].loc[tr_valid]
    d_const, d_const_val, d_cols = _const_cols(d_train)
    quasi = gl.quasi_constant_nodes(v_train) if snap else {}

    import time
    bundle = {"features": list(X.columns), "clf": clf, "threshold": 0.5,
              "v_const_val": v_const_val, "d_const_val": d_const_val,
              "v_cols": v_cols, "d_cols": d_cols, "quasi_v": quasi,
              "parallel_avg": parallel_avg, "config": config,
              "meta": {"train_exp": int(len(tr)), "train_valid": int(len(tr_valid))}}

    for key, Ydf, cols, tag in [
        ("v", frame["V"], v_cols, "V"),
        ("d", frame["Delta"], d_cols, "Delta"),
    ]:
        t = time.time()
        Ym = Ydf.loc[tr_valid, cols].to_numpy(np.float64)
        m, xs, ys = _fit_mlp(Xv, Ym, seed=seed, **config[key])
        bundle[f"{key}_model"], bundle[f"{key}_xs"], bundle[f"{key}_ys"] = m, xs, ys
        if verbose:
            print(f"  fit {tag}: n_iter={m.n_iter_} loss={m.loss_:.4g} {time.time()-t:.1f}s", flush=True)

    iz_cols = list(frame["Izarg"].columns)
    Izm = frame["Izarg"].loc[tr_valid, iz_cols].to_numpy(np.float64)
    t = time.time()
    m, xs, ys = _fit_mlp(Xv, Izm, seed=seed, **config["iz"])
    bundle["iz_model"], bundle["iz_xs"], bundle["iz_ys"] = m, xs, ys
    bundle["iz_cols"] = iz_cols
    bundle["idop_nominal"] = frame["Idop"].loc[tr_valid].mean(axis=0)
    if verbose:
        print(f"  fit Izarg: n_iter={m.n_iter_} loss={m.loss_:.4g} {time.time()-t:.1f}s", flush=True)
    return bundle


def predict_bundle(nodes, lines, bundle, zero_invalid=True):
    from grid_validation import validate_inputs, validate_numpy_bundle

    validate_numpy_bundle(bundle)
    nodes, lines = validate_inputs(nodes, lines, bundle)
    # Цели не должны влиять на инференс, даже если заполнены NaN или строками.
    frame = gl.build_frame(nodes.drop(columns=["V", "Delta", "regime_valid"], errors="ignore"),
                           lines.drop(columns=["I_max", "I_zarg", "I_доп"], errors="ignore"))
    exp = frame["exp"]
    missing = set(bundle["features"]) - set(frame["X"].columns)
    if missing:
        raise ValueError(f"Отсутствуют признаки модели: {sorted(missing)}")
    X = frame["X"].reindex(columns=bundle["features"])

    if bundle.get("kind") == "numpy_v2":
        Xn = X.to_numpy(np.float64)
        prob = np_forward_prob(bundle["clf"], Xn)
        reg = pd.Series((prob >= bundle["threshold"]).astype(int), index=exp)
        v_pred = np_forward_reg(bundle["reg"]["v"], Xn)
        d_pred = np_forward_reg(bundle["reg"]["d"], Xn)
        iz_pred = np_forward_reg(bundle["reg"]["iz"], Xn)
    else:
        prob = bundle["clf"].predict_proba(X)[:, list(bundle["clf"].classes_).index(1)]
        reg = pd.Series((prob >= bundle["threshold"]).astype(int), index=exp)
        v_pred = _pred(bundle["v_model"], bundle["v_xs"], bundle["v_ys"], X)
        d_pred = _pred(bundle["d_model"], bundle["d_xs"], bundle["d_ys"], X)
        iz_pred = _pred(bundle["iz_model"], bundle["iz_xs"], bundle["iz_ys"], X)

    Vdf = pd.DataFrame(v_pred, index=exp, columns=bundle["v_cols"])
    for c, val in bundle["v_const_val"].items():
        Vdf[c] = val
    for c, mode in bundle.get("quasi_v", {}).items():
        if c in Vdf.columns:
            Vdf[c] = mode

    Ddf = pd.DataFrame(d_pred, index=exp, columns=bundle["d_cols"])
    for c, val in bundle["d_const_val"].items():
        Ddf[c] = val

    Izdf = pd.DataFrame(iz_pred, index=exp, columns=bundle["iz_cols"])
    groups = gl.parallel_groups(lines)
    if bundle.get("parallel_avg", True):
        Izdf = gl.average_parallel_izarg(Izdf, groups)

    nom = bundle["idop_nominal"].reindex(bundle["iz_cols"]).to_numpy()

    def gather(df, exp_series, num_series):
        cols_pos = {int(c): i for i, c in enumerate(df.columns)}
        return df.reindex(exp_series.values).to_numpy()[
            np.arange(len(exp_series)), [cols_pos[int(n)] for n in num_series]]

    pn = pd.DataFrame({"ID_опыта": nodes["ID_опыта"].to_numpy(),
                       "Номер": nodes["Номер"].to_numpy()})
    pn["V"] = gather(Vdf, nodes["ID_опыта"], nodes["Номер"])
    pn["Delta"] = gather(Ddf, nodes["ID_опыта"], nodes["Номер"])
    pn["Regime_valid"] = nodes["ID_опыта"].map(reg).astype(int).to_numpy()

    pl = pd.DataFrame({"ID_опыта": lines["ID_опыта"].to_numpy(),
                       "Номер": lines["Номер"].to_numpy()})
    pl["Izarg, %"] = gather(Izdf, lines["ID_опыта"], lines["Номер"])
    line_pos = {int(c): i for i, c in enumerate(bundle["iz_cols"])}
    pl["Imax, A"] = pl["Izarg, %"].to_numpy() * nom[[line_pos[int(n)] for n in lines["Номер"]]] / 100.0

    if zero_invalid:
        mn = pn["Regime_valid"].to_numpy() == 0
        pn.loc[mn, ["V", "Delta"]] = 0.0
        ml = lines["ID_опыта"].map(reg).astype(int).to_numpy() == 0
        pl.loc[ml, ["Imax, A", "Izarg, %"]] = 0.0

    pn = pn[gl.NODE_PRED_COLS]
    pl = pl[gl.LINE_PRED_COLS]
    return pn, pl, reg, frame
