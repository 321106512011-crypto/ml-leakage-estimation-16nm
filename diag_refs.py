import os
import numpy as np, pandas as pd
RES = "partB/results"
d = pd.read_csv(os.path.join(RES, "gates_dev_ml.csv"), dtype={"ab": str})
sp = pd.read_csv(os.path.join(RES, "gates_spice.csv"), dtype={"ab": str})[["gate","ab","corner","leak_ptot_over_vdd"]]

P = d[["p_id","p_ig","p_is","p_ib"]].abs()
d["kcl_ml"] = d[["p_id","p_ig","p_is","p_ib"]].sum(axis=1).abs() / P.max(axis=1)
print("ML KCL violation (rel to largest term), median by case:")
print(d.groupby("case").kcl_ml.median().to_string())

out = {}
for ref in ["zero", "vs", "vd", "vb", "vg"]:
    r = 0.0 if ref == "zero" else d[ref]
    d["pw"] = (d.vd-r)*d.p_id + (d.vg-r)*d.p_ig + (d.vs-r)*d.p_is + (d.vb-r)*d.p_ib
    g = d.groupby(["gate","ab","corner"]).agg(pw=("pw","sum"), vdd=("vdd","first")).reset_index()
    g = g.merge(sp, on=["gate","ab","corner"])
    g["err"] = ((g.pw/g.vdd) - g.leak_ptot_over_vdd).abs() / g.leak_ptot_over_vdd.abs() * 100
    out[ref] = g.groupby(["gate","ab"]).err.median()
    print("ref=%-5s overall median err %.2f%%  mean %.2f%%" % (ref, g.err.median(), g.err.mean()))
pd.set_option("display.width", 200)
print(pd.DataFrame(out).round(1).to_string())
