import os, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from mlfeat import add_feats, FEATS, TARGETS

warnings.filterwarnings("ignore")
SEED = 42
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
FLOORS = {"nmos_off": 1e-21}   # real tiny currents in NMOS OFF
DEFAULT_FLOOR = 1e-15          # simulator noise floor elsewhere

def make_models():
    return {
        "mlp": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(128, 128, 128), activation="tanh",
                learning_rate_init=1e-3, max_iter=600, early_stopping=True,
                n_iter_no_change=30, random_state=SEED)),
            transformer=StandardScaler()),
        "lgb": lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05,
                                 num_leaves=63, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=4),
        "xgb": xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=8,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=SEED, n_jobs=4, verbosity=0),
        "rf": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                    n_jobs=4, random_state=SEED),
    }

def make_clf(ytr_sign):
    if len(set(ytr_sign.tolist())) == 1:
        return DummyClassifier(strategy="constant", constant=int(ytr_sign[0]))
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=SEED, verbose=-1, n_jobs=4)

def metrics(yt_log, yp_log):
    pct = np.abs(10 ** yp_log - 10 ** yt_log) / 10 ** yt_log * 100.0
    return r2_score(yt_log, yp_log), np.median(pct), np.percentile(pct, 95)

rows_val, rows_test = [], []
for case in CASES:
    df = pd.read_csv("data/%s.csv" % case)
    X_all = add_feats(df).values
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED)

    Xtr = X_all[tr]
    rail_path = "data/rail/%s.csv" % case      # optional, TRAIN ONLY
    rail = pd.read_csv(rail_path) if os.path.exists(rail_path) else None
    floor = FLOORS.get(case, DEFAULT_FLOOR)
    bundle = {}
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))

    for k, t in enumerate(TARGETS):
        I = df[t].values
        mag = np.abs(I)
        ylog = np.log10(np.maximum(mag, floor))
        sgn = (I >= 0).astype(int)
        ok = mag >= floor

        tr_i = tr[ok[tr]]
        Xt, yt, st = X_all[tr_i], ylog[tr_i], sgn[tr_i]
        if rail is not None:
            Ir = rail[t].values
            mr = np.abs(Ir) >= floor
            Xt = np.vstack([Xt, add_feats(rail).values[mr]])
            yt = np.concatenate([yt, np.log10(np.abs(Ir[mr]))])
            st = np.concatenate([st, (Ir[mr] >= 0).astype(int)])
        va_i, te_i = va[ok[va]], te[ok[te]]

        best, best_med, best_model = None, np.inf, None
        for name, m in make_models().items():
            m.fit(Xt, yt)
            r2, med, p95 = metrics(ylog[va_i], m.predict(X_all[va_i]))
            rows_val.append((case, t, name, r2, med, p95))
            print("%-9s %-2s %-4s val R2 %.4f  med %6.2f%%  p95 %7.2f%%"
                  % (case, t, name, r2, med, p95), flush=True)
            if med < best_med:
                best, best_med, best_model = name, med, m

        clf = make_clf(st).fit(Xt, st)
        r2, med, p95 = metrics(ylog[te_i], best_model.predict(X_all[te_i]))
        sacc = float(np.mean(clf.predict(X_all[te_i]) == sgn[te_i]))
        rows_test.append((case, t, best, r2, med, p95, sacc, floor, len(te_i)))
        print("  -> chosen %s | TEST R2 %.4f med %.2f%% p95 %.2f%% sign acc %.4f"
              % (best, r2, med, p95, sacc), flush=True)
        bundle[t] = {"reg": best_model, "clf": clf, "feats": FEATS,
                     "floor": floor, "model_name": best}

        ax = axs[k // 2][k % 2]
        yp = best_model.predict(X_all[te_i])
        ax.loglog(10 ** ylog[te_i], 10 ** yp, ".", ms=2, alpha=0.4)
        lo, hi = 10 ** ylog[te_i].min(), 10 ** ylog[te_i].max()
        ax.loglog([lo, hi], [lo, hi], "k-", lw=0.8)
        ax.set_title("%s %s (%s)" % (case, t, best))
        ax.set_xlabel("SPICE |I| (A)"); ax.set_ylabel("ML |I| (A)")

    fig.tight_layout(); fig.savefig("plots/pred_vs_spice_%s.png" % case, dpi=130)
    plt.close(fig)
    joblib.dump(bundle, "saved_models/%s.joblib" % case)

cols = ["case", "target", "model", "val_r2", "val_median_pct", "val_p95_pct"]
pd.DataFrame(rows_val, columns=cols).to_csv("results/part_a_model_comparison.csv", index=False)
cols = ["case", "target", "model", "test_r2", "test_median_pct", "test_p95_pct",
        "sign_acc", "floor_A", "n_test"]
pd.DataFrame(rows_test, columns=cols).to_csv("results/part_a_test.csv", index=False)
print("done")
