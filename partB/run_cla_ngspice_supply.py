import os, sys, shutil, subprocess, tempfile
import numpy as np, pandas as pd
from multiprocessing import Pool
import run_gates as g
import cla_def as C

def netlist(vec, vdd, temp, cp):
    t = ["simulator lang=spice", '.include "%s"' % cp,
         ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9",
         ".temp %.6g" % temp, "Vdd vdd 0 DC %.9g" % vdd]
    for n, b in vec.items():
        t.append("V%s %s 0 DC %.9g" % (n, n, vdd if b else 0.0))
    for k, (gt, out, ins) in enumerate(C.GATES):
        m = {"out": out, "a": ins[0], "vdd": "vdd", "0": "0",
             "x": "g%d_x" % k, "y": "g%d_y" % k, "n": "g%d_n" % k}
        if len(ins) > 1:
            m["b"] = ins[1]
        for (n, mod, d, gg, s, b) in g.devices(gt):
            t.append("M%d_%s %s %s %s %s %s W=%s L=%s" % (
                k, n[1:], m[d], m[gg], m[s], m[b], mod,
                g.WN if mod == "nmos" else g.WP, g.L))
    t += [".op", ".end"]
    return "\n".join(t) + "\n"

NDEV = sum(len(g.devices(t)) for t, _, _ in C.GATES)

def one(vec, vdd, temp, cp):
    base = netlist(vec, vdd, temp, cp)
    base = base.replace(".op", ".control\nop\nprint i(Vdd)\n.endc")

    d = tempfile.mkdtemp(dir=g.RUNS)
    try:
        for A in g.LADDER:
            fn = os.path.join(d, "run.scs")
            open(fn, "w").write(
                base.replace("abstol=1e-18", "abstol=" + A)
            )

            p = subprocess.run(
                ["ngspice", "-b", "run.scs"],
                cwd=d,
                timeout=30,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            import re
            m = re.search(
                r"i\(vdd\)\s*=\s*([-+0-9.eE]+)",
                p.stdout,
                re.I
            )

            if m:
                ivdd = float(m.group(1))
                return -ivdd, A

        return float("nan"), "none"

    finally:
        shutil.rmtree(d, ignore_errors=True)

def do_corner(r):
    k = int(r["corner"])
    cp = os.path.join(g.CARDS, "c%03d.pm" % k)
    rows = []
    for v in range(128):
        s, A = one(C.vector(v), r["vdd"], r["temp"], cp)
        rows.append(dict(corner=k, vec=v, spice=s, abstol_used=A))
    return rows

if __name__ == "__main__":
    nc = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    npr = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    cs = pd.read_csv(os.path.join(g.RES, "corners.csv")).iloc[:nc]
    print("transistors per run:", NDEV, flush=True)
    out = []
    with Pool(npr) as p:
        for i, rows in enumerate(p.imap_unordered(do_corner, [r for _, r in cs.iterrows()])):
            out += rows
            if (i + 1) % 5 == 0 or i + 1 == len(cs):
                print("corners done %d/%d" % (i + 1, len(cs)), flush=True)
    df = pd.DataFrame(out).sort_values(["corner", "vec"])
    df.to_csv(os.path.join(g.RES, "cla_spice.csv"), index=False)
    print("failed:", df.spice.isna().sum(), "of", len(df))
    print(df.abstol_used.value_counts())
