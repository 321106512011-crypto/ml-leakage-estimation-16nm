#!/usr/bin/env python3
"""Part B step 3: SPICE reference for INV/NAND2/NOR2/AND2/OR2 at N random corners.
usage: python3 run_gates.py [ncorners=200] [nproc=4]"""
import os, re, sys, shutil, subprocess, tempfile
import numpy as np, pandas as pd
from multiprocessing import Pool
from make_card import card

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.abspath(os.path.join(HERE, "..", "models", "16nm_LP.pm"))
CARDS = os.path.join(HERE, "cards")
RUNS = os.path.join(HERE, "runs")
RES = os.path.join(HERE, "results")
L = "16n"
WN = "32n"
WP = "64n"
SEED = 42
LADDER = ("1e-18", "1e-17", "1e-16", "1e-15")
PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]
NOM = {"n": dict(vth0=0.68191, toxe=1.2e-9, ndep=7e18, u0=0.028, xj=5e-9),
       "p": dict(vth0=-0.6862, toxe=1.22e-9, ndep=4.4e18, u0=0.0075, xj=7.2e-9)}

def N(n, d, g, s):
    return (n, "nmos", d, g, s, "0")

def P(n, d, g, s):
    return (n, "pmos", d, g, s, "vdd")

def devices(gate):
    if gate == "INV":
        return [N("Mn1", "out", "a", "0"), P("Mp1", "out", "a", "vdd")]
    if gate == "NAND2":
        return [N("Mn1", "out", "a", "x"), N("Mn2", "x", "b", "0"),
                P("Mp1", "out", "a", "vdd"), P("Mp2", "out", "b", "vdd")]
    if gate == "NOR2":
        return [N("Mn1", "out", "a", "0"), N("Mn2", "out", "b", "0"),
                P("Mp1", "out", "a", "y"), P("Mp2", "y", "b", "vdd")]
    if gate == "AND2":
        return [N("Mn1", "n", "a", "x"), N("Mn2", "x", "b", "0"),
                P("Mp1", "n", "a", "vdd"), P("Mp2", "n", "b", "vdd"),
                N("Mn3", "out", "n", "0"), P("Mp3", "out", "n", "vdd")]
    if gate == "OR2":
        return [N("Mn1", "n", "a", "0"), N("Mn2", "n", "b", "0"),
                P("Mp1", "n", "a", "y"), P("Mp2", "y", "b", "vdd"),
                N("Mn3", "out", "n", "0"), P("Mp3", "out", "n", "vdd")]
    raise ValueError(gate)

NODEMAP = {
    "INV":   {"out": ("Mn1", "d")},
    "NAND2": {"out": ("Mn1", "d"), "x": ("Mn1", "s")},
    "NOR2":  {"out": ("Mn1", "d"), "y": ("Mp1", "s")},
    "AND2":  {"nand_out": ("Mn1", "d"), "x": ("Mn1", "s"), "out": ("Mn3", "d")},
    "OR2":   {"nor_out": ("Mp1", "d"), "y": ("Mp1", "s"), "out": ("Mn3", "d")},
}
COMBOS = {"INV": ["0", "1"], "NAND2": ["00", "01", "10", "11"], "NOR2": ["00", "01", "10", "11"],
          "AND2": ["00", "01", "10", "11"], "OR2": ["00", "01", "10", "11"]}

def netlist(gate, ab, vdd, temp, cardpath):
    t = ["simulator lang=spice", '.include "%s"' % cardpath,
         ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9",
         ".temp %.6g" % temp, "Vdd vdd 0 DC %.9g" % vdd]
    for name, bit in zip(["VA", "VB"], ab):
        t.append("%s %s 0 DC %.9g" % (name, name[1].lower(), vdd if bit == "1" else 0.0))
    for (n, mod, d, g, s, b) in devices(gate):
        t.append("%s %s %s %s %s %s W=%s L=%s" % (n, d, g, s, b, mod, WN if mod == "nmos" else WP, L))
    t += [".op", ".end"]
    return "\n".join(t) + "\n"

PFX = {"a": 1e-18, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
def num(s, unit):
    v = float(s)
    if unit and len(unit) >= 2 and unit[0] in PFX and unit[1:] in ("V", "A", "W"):
        v *= PFX[unit[0]]
    return v

TERM = re.compile(r"^\s+([dgsbpn])\s*:\s*(?:V\(\S+\)|val\(0\))\s*=\s*([-+0-9.eE]+)\s*([A-Za-z]*)")
CUR = re.compile(r"^\s*(ide|ige|ise|ibe|i|pwr)\s*=\s*([-+0-9.eE]+)\s*([A-Za-z]*)")

def parse_log(text):
    devs, srcs = {}, {}
    for blk in text.split("\nInstance: ")[1:]:
        lines = blk.split("\n")
        name = lines[0].strip()
        prim = re.search(r"Primitive:\s*(\w+)", blk)
        prim = prim.group(1) if prim else ""
        rec = {}
        for ln in lines[1:]:
            m = TERM.match(ln)
            if m and m.group(1) in "dgsb":
                rec["v" + m.group(1)] = num(m.group(2), m.group(3))
                continue
            m = CUR.match(ln)
            if m and m.group(1) not in rec:
                rec[m.group(1)] = num(m.group(2), m.group(3))
        if prim == "bsim4" and all(k in rec for k in ("vd", "vg", "vs", "vb", "ide", "ige", "ise", "ibe")):
            rv = re.search(r"reversed\s*=\s*(\w+)", blk)
            rec["reversed"] = rv.group(1) if rv else ""
            devs[name] = rec
        elif prim == "vsource" and "i" in rec:
            srcs[name] = rec["i"]
    m = re.search(r"Total Power Dissipation\s*=\s*([-+0-9.eE]+)\s*([A-Za-z]*)", text)
    ptot = num(m.group(1), m.group(2)) if m else float("nan")
    return devs, srcs, ptot

def run_spice(gate, ab, vdd, temp, cardpath):
    os.makedirs(RUNS, exist_ok=True)
    d = tempfile.mkdtemp(dir=RUNS)
    try:
        base = netlist(gate, ab, vdd, temp, cardpath)
        for A in LADDER:
            open(os.path.join(d, "run.scs"), "w").write(base.replace("abstol=1e-18", "abstol=" + A))
            subprocess.run(["spectre", "-64", "+log", "run.log", "run.scs"], cwd=d, timeout=180,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            lp = os.path.join(d, "run.log")
            text = open(lp).read() if os.path.exists(lp) else ""
            devs, srcs, ptot = parse_log(text)
            if len(devs) == len(devices(gate)) and "Vdd" in srcs:
                return devs, srcs, ptot, A
        return devs, srcs, ptot, "none"
    finally:
        shutil.rmtree(d, ignore_errors=True)

def make_corners(n):
    rs = np.random.RandomState(SEED)
    rows = []
    for k in range(n):
        r = {"corner": k, "vdd": rs.uniform(0.9, 1.1), "temp": rs.uniform(-40.0, 125.0)}
        for pre in ("n", "p"):
            for p in PARAMS:
                r["%s_%s" % (pre, p)] = NOM[pre][p] * rs.uniform(0.9, 1.1)
        rows.append(r)
    return pd.DataFrame(rows)

def do_corner(r):
    k = int(r["corner"])
    os.makedirs(CARDS, exist_ok=True)
    cp = os.path.join(CARDS, "c%03d.pm" % k)
    sc = lambda pre: {p: r["%s_%s" % (pre, p)] / NOM[pre][p] for p in PARAMS}
    card(PM, cp, sc("n"), sc("p"))
    node_rows, dev_rows, fails = [], [], 0
    for gate in ["INV", "NAND2", "NOR2", "AND2", "OR2"]:
        for ab in COMBOS[gate]:
            devs, srcs, ptot, used = run_spice(gate, ab, r["vdd"], r["temp"], cp)
            if len(devs) != len(devices(gate)) or "Vdd" not in srcs:
                fails += 1
                continue
            nr = {"gate": gate, "ab": ab, "corner": k, "vdd": r["vdd"], "temp": r["temp"], "abstol_used": used,
                  "i_vdd": srcs["Vdd"], "i_va": srcs.get("VA", 0.0), "i_vb": srcs.get("VB", 0.0),
                  "ptot": ptot, "leak_supply": -srcs["Vdd"], "leak_ptot_over_vdd": ptot / r["vdd"]}
            for nn, (dn, term) in NODEMAP[gate].items():
                nr["v_" + nn] = devs[dn]["v" + term]
            node_rows.append(nr)
            for (n, mod, d, g, s, b) in devices(gate):
                x = devs[n]
                scale = max(abs(x["ide"]), abs(x["ige"]), abs(x["ise"]), abs(x["ibe"]), 1e-30)
                dev_rows.append({"gate": gate, "ab": ab, "corner": k, "dev": n, "type": mod,
                                 "w": WN if mod == "nmos" else WP,
                                 "vd": x["vd"], "vg": x["vg"], "vs": x["vs"], "vb": x["vb"],
                                 "id": x["ide"], "ig": x["ige"], "is": x["ise"], "ib": x["ibe"],
                                 "kcl_rel": abs(x["ide"] + x["ige"] + x["ise"] + x["ibe"]) / scale,
                                 "reversed": x["reversed"]})
    return node_rows, dev_rows, fails

if __name__ == "__main__":
    ncorn = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(RES, exist_ok=True)
    corners = make_corners(ncorn)
    corners.to_csv(os.path.join(RES, "corners.csv"), index=False)
    jobs = [r for _, r in corners.iterrows()]
    nodes, devs_, fails = [], [], 0
    with Pool(nproc) as pool:
        for i, (a, b, f) in enumerate(pool.imap_unordered(do_corner, jobs)):
            nodes += a; devs_ += b; fails += f
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print("corners done %d/%d  failed runs so far %d" % (i + 1, len(jobs), fails), flush=True)
    pd.DataFrame(nodes).sort_values(["gate", "ab", "corner"]).to_csv(os.path.join(RES, "gates_spice.csv"), index=False)
    pd.DataFrame(devs_).sort_values(["gate", "ab", "corner", "dev"]).to_csv(os.path.join(RES, "gates_devices.csv"), index=False)
    d = pd.DataFrame(devs_)
    print("rows: gate runs %d (expected %d), device rows %d" % (len(nodes), ncorn * 18, len(d)))
    print("per-device KCL residual (rel): median %.2e  p99 %.2e  max %.2e"
          % (d.kcl_rel.median(), d.kcl_rel.quantile(0.99), d.kcl_rel.max()))
    print(pd.DataFrame(nodes).abstol_used.value_counts())
    print("done")
