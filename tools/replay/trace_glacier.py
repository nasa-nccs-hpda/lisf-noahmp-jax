import sys, datetime as dt
import numpy as np, jax, jax.numpy as jnp
from jax import lax
import dumpio
from lisf_noahmp_jax.driver.params import Setup
from lisf_noahmp_jax.driver.forcing import NLDAS2
from lisf_noahmp_jax.driver import model
S = Setup("run/conus/lis.config", "run/conus"); F = NLDAS2("run/INPUT/NLDAS2.FORCING", S.lat, S.lon)
hdr, pre, post = dumpio.load("run/conus_g/jaxdump_noah39.bin")
m = hdr[:, 5] == 99
hdr = hdr[m]; pre = {k: v[m] for k, v in pre.items()}; post = {k: v[m] for k, v in post.items()}
G = S.glacier_idx; ng = len(G)
assert (hdr[:ng, 6] - 1 == G).all(), (hdr[:ng, 6][:5], G[:5])
nt = len(hdr) // ng
v = model.make_step(S, jit=False)
par = {k: jnp.asarray(x) for k, x in S.par.items()}
start = dt.datetime(2017, 2, 1)
st = {k: jnp.asarray(x) for k, x in S.initial_state(2017, 2, 1).items()}
keys = ("t1", "sneqv", "snowh", "albedo", "z0", "emiss", "ch", "q1")
def body(st, x):
    dyn, raw = x
    st, diag = v(par, dyn, st, raw)
    return st, jnp.stack([st[k][G] for k in keys] + [st["stc"][0][G]])
run = jax.jit(lambda st, dyn, raw: lax.scan(body, st, (dyn, raw)))
K = 24; rows = []
for c0 in range(0, nt, K):
    times = [start + dt.timedelta(seconds=S.dt * (s + 1)) for s in range(c0, c0 + K)]
    raws, dyns = [], []
    for now in times:
        newday = (now.hour == 0 and now.minute == 0) or (now == start + dt.timedelta(seconds=S.dt))
        dyns.append(S.dyn(now.year, now.month, now.day, newday=newday))
        o = F.at(now)
        raws.append({"tair": o["Tair"], "qair": o["Qair"], "swd": o["SWdown"], "lwd": o["LWdown"], "psurf": o["PSurf"],
                     "pcp": o["Rainf"], "u": o["Wind_E"], "v": o["Wind_N"]})
    rawK = {k: jnp.asarray(np.stack([r[k] for r in raws])) for k in raws[0]}
    dynK = {k: jnp.asarray(np.stack([d[k] for d in dyns])) for k in dyns[0]}
    st, outK = run(st, dynK, rawK)
    rows.append(np.asarray(outK))
rows = np.concatenate(rows)            # (nt, 9, ng)
names = list(keys) + ["stc1"]
for j, n in enumerate(names):
    ref = post[{"emiss": "emissi"}.get(n, n) if n not in ("ch",) else "ch"] if True else None
    ref = post[{"emiss": "emissi"}.get(n, n)].reshape(nt, ng)
    e = np.abs(rows[:, j] - ref)
    print(f"{n:7s} max err step0 {e[0].max():.3e}  step 12 {e[12].max():.3e}  step 48 {e[48].max():.3e}  step {nt-1} {e[-1].max():.3e}   ref range [{ref.min():.3f},{ref.max():.3f}]")
print("per-step errors (max over glacier tiles) for first 14 steps: t1 q1 ch stc1")
for s in range(14):
    r = [np.abs(rows[s, names.index(n)] - post[{"emiss": "emissi"}.get(n, n)].reshape(nt, ng)[s]).max() for n in ("t1", "q1", "ch", "stc1")]
    print(s, hdr[s * ng, :6], " ".join("%.3e" % x for x in r))
j = int(np.argmax(np.abs(rows[12, 0] - post["t1"].reshape(nt, ng)[12])))
print("worst tile (glacier idx)", j, "tile", G[j])
print("step  py_t1 lis_t1 | py_q1 lis_q1 | py_ch lis_ch | tair swd")
raw0 = None
for s in range(0, 16):
    print(s, "%.3f %.3f | %.5f %.5f | %.5f %.5f" % (rows[s, 0, j], post["t1"].reshape(nt, ng)[s, j], rows[s, 7, j], post["q1"].reshape(nt, ng)[s, j], rows[s, 6, j], post["ch"].reshape(nt, ng)[s, j]))
