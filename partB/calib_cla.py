import numpy as np, pandas as pd
d = pd.read_csv("results/cla_ml.csv")
d["ones"] = d.vec.apply(lambda v: bin(int(v)).count("1"))
tr, te = d[d.corner < 100], d[d.corner >= 100].copy()
def rep(name, ml, ref):
    e = (ml - ref).abs() / ref.abs() * 100
    print("%-22s bias %6.2f%%  median %5.2f%%  mean %5.2f%%  p95 %5.2f%%  -> acc %.2f%%"
          % (name, ((ml-ref)/ref.abs()*100).median(), e.median(), e.mean(), e.quantile(.95), 100-e.mean()))
rep("uncorrected (test)", te.ml, te.spice)
# 1) factor per number of ones
f1 = (tr.spice / tr.ml).groupby(tr.ones).median()
te["ml1"] = te.ml * te.ones.map(f1)
rep("per-#ones factor", te.ml1, te.spice)
# 2) factor per vector
f2 = (tr.spice / tr.ml).groupby(tr.vec).median()
te["ml2"] = te.ml * te.vec.map(f2)
rep("per-vector factor", te.ml2, te.spice)
print("\n#ones factors:\n", f1.round(3).to_string())
te.to_csv("results/cla_ml_calibrated_test.csv", index=False)
