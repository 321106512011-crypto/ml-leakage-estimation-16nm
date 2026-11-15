import numpy as np

BASE = ["vdd","temp","w","vd","vs","vg","vb","vth0","toxe","ndep","u0","xj"]
FEATS = BASE + ["vds","vgs","vbs","vgd","invT","logw","ov"]
TARGETS = ["id","ig","is","ib"]

def add_feats(df):
    d = df[BASE].copy()
    d["vds"] = d["vd"] - d["vs"]
    d["vgs"] = d["vg"] - d["vs"]
    d["vbs"] = d["vb"] - d["vs"]
    d["vgd"] = d["vg"] - d["vd"]
    d["invT"] = 1.0 / (d["temp"] + 273.15)
    d["logw"] = np.log(d["w"])
    d["ov"] = d["vg"] - d["vs"] - d["vth0"]
    return d[FEATS]

def predict_current(entry, df):
    """entry = saved_models/<case>.joblib[target]; df has the 12 BASE columns."""
    X = add_feats(df)[entry["feats"]].values
    logi = np.asarray(entry["reg"].predict(X), float)
    sgn = np.where(np.asarray(entry["clf"].predict(X)) == 1, 1.0, -1.0)
    return sgn * 10.0 ** logi
