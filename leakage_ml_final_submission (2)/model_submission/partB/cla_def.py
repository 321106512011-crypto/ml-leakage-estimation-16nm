FUNC = {"INV": lambda a: 1 - a, "NAND2": lambda a, b: 1 - (a & b),
        "AND2": lambda a, b: a & b, "OR2": lambda a, b: a | b}

def build():
    G = []
    def add(t, out, *ins):
        G.append((t, out, ins)); return out
    def xor(o, a, b):
        x = add("OR2", o + "_or", a, b); n = add("NAND2", o + "_nd", a, b)
        add("AND2", o, x, n)
    for i in range(3):
        xor("p%d" % i, "a%d" % i, "b%d" % i)
        add("AND2", "g%d" % i, "a%d" % i, "b%d" % i)
    add("AND2", "t10", "p0", "cin"); add("OR2", "c1", "g0", "t10")
    add("AND2", "t21", "p1", "g0"); add("AND2", "t22", "p1", "p0"); add("AND2", "t23", "t22", "cin")
    add("OR2", "u2", "g1", "t21"); add("OR2", "c2", "u2", "t23")
    add("AND2", "t31", "p2", "g1"); add("AND2", "t32", "p2", "p1")
    add("AND2", "t33", "t32", "g0"); add("AND2", "t34", "t32", "p0"); add("AND2", "t35", "t34", "cin")
    add("OR2", "u31", "g2", "t31"); add("OR2", "u32", "u31", "t33"); add("OR2", "c3", "u32", "t35")
    xor("s0", "p0", "cin"); xor("s1", "p1", "c1"); xor("s2", "p2", "c2")
    return G

GATES = build()
INPUTS = ["a0", "a1", "a2", "b0", "b1", "b2", "cin"]

def vector(v):
    return {n: (v >> i) & 1 for i, n in enumerate(INPUTS)}

def evaluate(vec):
    val = dict(vec)
    rows = []
    for t, out, ins in GATES:
        x = [val[i] for i in ins]
        val[out] = FUNC[t](*x)
        rows.append((t, out, "".join(str(b) for b in x)))
    return val, rows          # rows = (gate type, out node, input bits "ab")

if __name__ == "__main__":
    assert len(GATES) == 36
    for v in range(128):
        val, _ = evaluate(vector(v))
        a = val["a0"] + 2*val["a1"] + 4*val["a2"]
        b = val["b0"] + 2*val["b1"] + 4*val["b2"]
        tot = a + b + val["cin"]
        got = val["s0"] + 2*val["s1"] + 4*val["s2"] + 8*val["c3"]
        assert tot == got, (v, tot, got)
    print("logic OK: 36 gates, 128 vectors match a+b+cin")
