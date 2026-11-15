#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
import joblib

DATA_DIR = "data"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
TARGETS = ["id", "ig", "is", "ib"]
FEATURES = ["vdd","temp","w","vd","vs","vg","vb","vth0","toxe","ndep","u0","xj"]
SEED = 42
EPS = 1e-30

def load_case(case):
    df = pd.read_csv("%s/%s.csv" % (DATA_DIR, case))
    rail = "%s/rail/%s.csv" % (DATA_DIR, case)
    extra = pd.read_csv(rail) if os.path.exists(rail) else None
    return df, extra

def make_reg(kind):
    if kind == "lgb":
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=8, num_leaves=63, random_state=SEED, n_jobs=2, verbose=-1)
    if kind == "xgb":
        return xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=SEED, n_jobs=2, verbosity=0)
    return RandomForestRegressor(n_estimators=200, max_depth=16, random_state=SEED, n_jobs=2)

def make_clf(kind):
    if kind == "lgb":
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=SEED, n_jobs=2, verbose=-1)
    if kind == "xgb":
        return xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=SEED, n_jobs=2, verbosity=0)
    return RandomForestClassifier(n_estimators=150, max_depth=14, random_state=SEED, n_jobs=2)

def pct_err(y_true, y_pred):
    denom = np.maximum(np.abs(y_true), 1e-30)
    return np.abs(y_pred - y_true) / denom * 100.0

def predict_current(reg, clf, X):
    log_hat = reg.predict(X)
    sign_hat = clf.predict(X)
    sign_hat = np.where(sign_hat >= 0, 1.0, -1.0)
    return sign_hat * np.power(10.0, log_hat)

class Dummy:
    def fit(self, X, y=None):
        import numpy as np
        if y is not None:
            y = np.asarray(y).ravel()
            if len(y):
                self.sign_ = y[0]
        return self
    def predict(self, X):
        import numpy as np, sys
        if not hasattr(self, "sign_"):
            frame = sys._getframe()
            while frame is not None:
                if "ytr_sign" in frame.f_locals:
                    self.sign_ = np.asarray(frame.f_locals["ytr_sign"]).ravel()[0]
                    break
                frame = frame.f_back
            else:
                raise RuntimeError("Dummy has no sign and ytr_sign is not in scope")
        return np.full(len(X), self.sign_)


def main():
    rows = []
    for case in CASES:
        print("===== %s =====" % case, flush=True)
        df, extra = load_case(case)
        X = df[FEATURES]
        idx = np.arange(len(df))
        itr, itmp = train_test_split(idx, test_size=0.30, random_state=SEED)
        iva, ite = train_test_split(itmp, test_size=0.50, random_state=SEED)
        chosen = {}
        for target in TARGETS:
            print("  %s" % target, flush=True)
            y = df[target].to_numpy()
            y_log = np.log10(np.abs(y) + EPS)
            y_sign = np.where(y >= 0, 1, -1)
            best = None
            for kind in ("lgb", "xgb", "rf"):
                reg = make_reg(kind)
                clf = make_clf(kind)
                Xtr = X.iloc[itr]
                if extra is not None:
                    Xtr = pd.concat([Xtr, extra[FEATURES]], ignore_index=True)
                    ytr_log = np.concatenate([y_log[itr], np.log10(np.abs(extra[target].to_numpy()) + EPS)])
                    ytr_sign = np.concatenate([y_sign[itr], np.where(extra[target].to_numpy() >= 0, 1, -1)])
                else:
                    ytr_log = y_log[itr]
                    ytr_sign = y_sign[itr]
                reg.fit(Xtr, ytr_log)
                # classifier needs both classes
                if len(np.unique(ytr_sign)) < 2:
                    clf = Dummy()
                else:
                    clf.fit(Xtr, ytr_sign)
                pred_va = predict_current(reg, clf, X.iloc[iva])
                e = pct_err(y[iva], pred_va)
                med = float(np.median(e))
                p95 = float(np.percentile(e, 95))
                print("    %-4s  val median=%.3f%%  p95=%.3f%%" % (kind, med, p95), flush=True)
                rec = {"kind": kind, "reg": reg, "clf": clf, "med": med, "p95": p95}
                if best is None or med < best["med"]:
                    best = rec
            pred_te = predict_current(best["reg"], best["clf"], X.iloc[ite])
            et = pct_err(y[ite], pred_te)
            med_te = float(np.median(et))
            p95_te = float(np.percentile(et, 95))
            r2 = float(r2_score(y_log[ite], np.log10(np.abs(pred_te) + EPS)))
            print("    CHOSE %-4s  test median=%.3f%%  p95=%.3f%%  logR2=%.3f" % (best["kind"], med_te, p95_te, r2), flush=True)
            chosen[target] = {"kind": best["kind"], "reg": best["reg"], "clf": best["clf"]}
            rows.append({
                "case": case, "target": target, "model": best["kind"],
                "val_median_pct": best["med"], "val_p95_pct": best["p95"],
                "test_median_pct": med_te, "test_p95_pct": p95_te, "test_log_r2": r2,
            })
        joblib.dump(chosen, "%s/%s.joblib" % (SAVE_DIR, case))
        print("  saved %s" % case, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/part_a_full.csv", index=False)
    print("Done.", flush=True)
    print(out.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
