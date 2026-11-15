import re, sys
def card(src, out, n, p):
    """n, p: dicts with keys vth0,toxe,ndep,u0,xj -> scale factors (1.0 = nominal)."""
    t = open(src).read()
    k = re.search(r"\.model\s+pmos", t, re.I).start()
    parts = [(t[:k], n), (t[k:], p)]
    res = ""
    for text, sc in parts:
        for name, f in sc.items():
            pat = re.compile(r"(\b%s\s*=\s*)([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)" % name, re.I)
            text = pat.sub(lambda m: m.group(1) + "%.6e" % (float(m.group(2)) * f), text, count=1)
        res += text
    open(out, "w").write(res)
if __name__ == "__main__":
    one = dict(vth0=1.0, toxe=1.0, ndep=1.0, u0=1.0, xj=1.0)
    card("../models/16nm_LP.pm", "test.pm", one, one)
