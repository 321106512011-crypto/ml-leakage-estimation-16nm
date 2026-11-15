import os
import numpy as np, pandas as pd, joblib
from mlfeat import predict_current, TARGETS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "partB", "results")
MOD = os.path.join(HERE, "saved_models")
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
models = {c: joblib.load(os.path.join(MOD, c + ".joblib")) for c in CASES}

dev = pd.read_csv(os.path.join(RES, "gates_devices.csv"), dtype={"ab": str})
cor = pd.read_csv(os.path.join(RES, "corners.csv"))
sp = pd.read_csv(os.path.join(RES, "gates_spice.csv"), dtype={"ab": str})

# sanity: training width range vs gate widths
tw = pd.read_csv(os.path.join(HERE, "data", "nmos_off.csv"))["w"]
print("training w range: %.3e .. %.3e" % (tw.min(), tw.max()))

dev["wm"] = dev["w"].map(lambda s: float(s.replace("n", "e-9")))
print("gate widths used:", sorted(dev["wm"].unique()))
dev = dev.merge(cor, on="corner").reset_index(drop=True)

# which model case each device falls in
isn = dev["type"] == "nmos"
on = np.where(isn, dev["vg"] > 0.5 * dev["vdd"], dev["vg"] < 0.5 * dev["vdd"])
dev["case"] = dev["type"] + "_" + np.where(on, "on", "off")
for p in ["vth0", "toxe", "ndep", "u0", "xj"]:
    dev[p] = np.where(isn, dev["n_" + p], dev["p_" + p])

for t in TARGETS:
    dev["p_" + t] = np.nan
for case, grp in dev.groupby("case"):
    X = pd.DataFrame({"vdd": grp["vdd"], "temp": grp["temp"], "w": grp["wm"],
                      "vd": grp["vd"], "vs": grp["vs"], "vg": grp["vg"], "vb": grp["vb"],
                      "vth0": grp["vth0"], "toxe": grp["toxe"], "ndep": grp["ndep"],
                      "u0": grp["u0"], "xj": grp["xj"]})
    for t in TARGETS:
        dev.loc[grp.index, "p_" + t] = predict_current(models[case][t], X)
    print("predicted", case, len(grp))

def power(prefix):
    # source-referenced: the source current drops out by KCL (sum of the 4 terminal currents = 0)
    return ((dev["vd"] - dev["vs"]) * dev[prefix + "id"]
            + (dev["vg"] - dev["vs"]) * dev[prefix + "ig"]
            + (dev["vb"] - dev["vs"]) * dev[prefix + "ib"])
dev["pw_spice"] = power("")
dev["pw_ml"] = power("p_")

g = dev.groupby(["gate", "ab", "corner"]).agg(vdd=("vdd", "first"), pw_spice=("pw_spice", "sum"),
                                              pw_ml=("pw_ml", "sum")).reset_index()
g = g.merge(sp[["gate", "ab", "corner", "leak_ptot_over_vdd"]], on=["gate", "ab", "corner"])
g["spice"] = g["leak_ptot_over_vdd"]
g["oracle"] = g["pw_spice"] / g["vdd"]
g["ml"] = g["pw_ml"] / g["vdd"]
print("\nORACLE check (formula on SPICE currents / SPICE target), should be ~1:")
print((g["oracle"] / g["spice"]).describe())

g["err_pct"] = (g["ml"] - g["spice"]).abs() / g["spice"].abs() * 100
g["signed_pct"] = (g["ml"] - g["spice"]) / g["spice"].abs() * 100
g.to_csv(os.path.join(RES, "gates_ml.csv"), index=False)

tab = g.groupby(["gate", "ab"]).agg(
    spice_med=("spice", "median"), ml_med=("ml", "median"), bias_pct=("signed_pct", "median"),
    med_err=("err_pct", "median"), p95_err=("err_pct", lambda s: s.quantile(0.95))).reset_index()
tab.to_csv(os.path.join(RES, "gate_table.csv"), index=False)
pd.set_option("display.width", 200)
print("\n", tab.to_string(index=False))
print("\nOVERALL: median err %.2f%%  mean err %.2f%%  p95 %.2f%%  -> mean accuracy %.2f%%"
      % (g.err_pct.median(), g.err_pct.mean(), g.err_pct.quantile(0.95), 100 - g.err_pct.mean()))
dev.to_csv(os.path.join(RES, "gates_dev_ml.csv"), index=False)
