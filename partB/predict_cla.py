import os, sys
import numpy as np, pandas as pd
import cla_def as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"
spice_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "cla_spice.csv")

# gate-level ML leakage lookup: (gate, input bits, corner) -> ml leakage
gm = pd.read_csv(os.path.join(RES, "gates_ml.csv"), dtype={"ab": str})
lut = {(r.gate, r.ab, int(r.corner)): r.ml for r in gm.itertuples()}

sp = pd.read_csv(spice_file)
sp = sp[sp.spice.notna()]

# gate input bits for each of the 128 vectors (independent of corner)
rows = [C.evaluate(C.vector(v))[1] for v in range(128)]   # list of (type, out, "bits")

ml = []
for r in sp.itertuples():
    tot = 0.0
    for (t, out, bits) in rows[int(r.vec)]:
        tot += lut[(t, bits, int(r.corner))]
    ml.append(tot)
sp = sp.assign(ml=ml)
sp["err_pct"] = (sp.ml - sp.spice).abs() / sp.spice.abs() * 100
sp["signed_pct"] = (sp.ml - sp.spice) / sp.spice.abs() * 100
sp.to_csv(os.path.join(RES, "cla_ml.csv"), index=False)

print("corners: %d  runs: %d" % (sp.corner.nunique(), len(sp)))
print("SPICE median %.3e  ML median %.3e" % (sp.spice.median(), sp.ml.median()))
print("bias %.2f%%  median err %.2f%%  mean err %.2f%%  p95 %.2f%%  -> mean accuracy %.2f%%"
      % (sp.signed_pct.median(), sp.err_pct.median(), sp.err_pct.mean(),
         sp.err_pct.quantile(0.95), 100 - sp.err_pct.mean()))
sp[["spice", "ml", "bias"] if False else ["spice"]]  # no-op
pd.DataFrame([dict(spice_median=sp.spice.median(), ml_median=sp.ml.median(),
                   bias_pct=sp.signed_pct.median(), median_err=sp.err_pct.median(),
                   p95_err=sp.err_pct.quantile(0.95))]).to_csv(
    os.path.join(RES, "cla_table.csv"), index=False)

# worst cases, for the error analysis
print("\nmedian error by corner temperature band:")
cor = pd.read_csv(os.path.join(RES, "corners.csv"))[["corner", "temp", "vdd"]]
m = sp.merge(cor, on="corner")
m["Tband"] = pd.cut(m.temp, [-41, 0, 40, 80, 126])
print(m.groupby("Tband").err_pct.median().round(2).to_string())

# parity plot, log axes
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.loglog(sp.spice, sp.ml, ".", ms=2, alpha=0.4)
lo, hi = min(sp.spice.min(), sp.ml.min()), max(sp.spice.max(), sp.ml.max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel("SPICE leakage (A)"); ax.set_ylabel("ML leakage (A)")
ax.set_title("3-bit CLA: ML vs SPICE (128 vectors x %d corners)" % sp.corner.nunique())
fig.tight_layout()
os.makedirs("../plots", exist_ok=True)
fig.savefig("../plots/parity_cla.png", dpi=150)
print("saved ../plots/parity_cla.png")
