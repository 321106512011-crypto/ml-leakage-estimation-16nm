#!/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, median_absolute_error
import lightgbm as lgb
import joblib

DATA_DIR = "data"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
TARGETS = ["id", "ig", "is", "ib"]
FEATURES = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
SEED = 42

def main():
    rows = []
    for case in CASES:
        print("===== %s =====" % case, flush=True)
        df = pd.read_csv("%s/%s.csv" % (DATA_DIR, case))
        X = df[FEATURES]
        models = {}
        for target in TARGETS:
            print("  fitting %s ..." % target, flush=True)
            y = np.log10(np.abs(df[target].to_numpy()) + 1e-30)
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=SEED)
            Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED)
            model = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=6,
                num_leaves=31, random_state=SEED, n_jobs=2, verbose=-1
            )
            model.fit(Xtr, ytr)
            pred = model.predict(Xva)
            med = float(median_absolute_error(yva, pred))
            r2 = float(r2_score(yva, pred))
            print("  %-4s  median log-error=%.4f  R2=%.4f" % (target, med, r2), flush=True)
            models[target] = model
            rows.append({"case": case, "target": target, "median_log_error": med, "r2": r2})
        joblib.dump(models, "%s/%s.joblib" % (SAVE_DIR, case))
        print("  saved %s" % case, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/part_a_model_comparison.csv", index=False)
    print("Done.", flush=True)
    print(out.to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
