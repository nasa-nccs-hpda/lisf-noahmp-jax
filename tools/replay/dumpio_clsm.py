"""Reader of the CLSM F2.5 call dump (jaxdump_clsm.bin + jaxdump_clsm.layout, see lis_patches/make_clsm_hook.py)."""
import re
import numpy as np

SRC = "/panfs/ccds02/nobackup/people/gtamkin/dev/ilab-agentic-ai/ilab-agentic-ai/projects/LISF/lis/surfacemodels/land/clsm.f2.5/clsmf25_model.F90"


def dummy_names():
    s = open(SRC).read()
    i = s.index("SUBROUTINE CLSMF25 (")
    j = s.index(")", i)
    body = re.sub(r"&\s*\n", "", "\n".join(l.split("!")[0] for l in s[i + len("SUBROUTINE CLSMF25 ("): j].split("\n")))
    names = [a.strip().lower() for a in body.split(",") if a.strip()]
    if names[-1] != "fldfrc":
        names += ["rivsto", "fldsto", "fldfrc"]
    return names


def load(dirname, maxstep=None):
    lay = [l.split("|") for l in open(f"{dirname}/jaxdump_clsm.layout").read().split("\n") if l]
    dn = [d for d in dummy_names() if d not in ("nch", "bug")]
    assert len(dn) == len(lay), (len(dn), len(lay))
    raw = open(f"{dirname}/jaxdump_clsm.bin", "rb").read()
    pos = 0

    def rd(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype=dtype, count=n, offset=pos)
        pos += a.nbytes
        return a
    ns = int(rd(">i4", 1)[0]); idx = rd(">i4", ns).astype(int) - 1
    import os
    ip = f"{dirname}/jaxdump_clsm_inter.layout"
    inames = [n.lower() for n in open(ip).read().split()] if os.path.exists(ip) else []
    recs = {0: {}, 1: {}, 2: {}}
    nlev = {"k": None}
    while pos < len(raw):
        tag, step = [int(v) for v in rd(">i4", 2)]
        if maxstep and step > maxstep:
            break
        d = {}
        if tag == 2:
            for n in inames:
                d[n] = rd(">f4", ns).astype(np.float32)
            recs[2][step] = d
            continue
        for (expr, kind), name in zip(lay, dn):
            if kind == "s":
                d[name] = float(rd(">f4", 1)[0])
            elif kind == "v":
                d[name] = rd(">f4", ns).astype(np.float32)
            elif kind == "k":
                k = 6 if name == "ghtcnt" else 3
                d[name] = rd(">f4", k * ns).reshape(ns, k).T.astype(np.float32)           # (k, ns)
            elif kind == "t":
                d[name] = rd(">f4", ns * 3).reshape(3, ns).astype(np.float32)             # (ns,3)-> store (3,ns)
        recs[tag][step] = d
    return idx, recs, [(e, k) for e, k in lay], dn
