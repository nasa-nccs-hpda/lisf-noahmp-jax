import sys, datetime as dt
import numpy as np, jax, jax.numpy as jnp
from jax import lax
import dumpio
from lisf_noahmp_jax.driver.params import Setup
from lisf_noahmp_jax.driver.forcing import NLDAS2
from lisf_noahmp_jax.driver import model
cfg, dump, tile = sys.argv[1], sys.argv[2], int(sys.argv[3])
t_lo, t_hi = int(sys.argv[4]), int(sys.argv[5])      # step range to print
S = Setup(cfg, "run"); F = NLDAS2("run/INPUT/NLDAS2.FORCING", S.lat, S.lon)
hdr, pre, post = dumpio.load(dump); T = S.ntile
v = model.make_step(S, jit=False)
par = {k: jnp.asarray(v_) for k, v_ in S.par.items()}
start = dt.datetime(2017, 1, 1)
st = {k: jnp.asarray(x) for k, x in S.initial_state(2017, 1, 1).items()}

def body(st, x):
    dyn, raw = x
    st, diag = v(par, dyn, st, raw)
    keys = ("sneqv", "snowh", "t1", "albedo", "sncovr", "z0", "ch", "q1", "xlai")
    return st, jnp.stack([st[k] for k in keys if k in st])

run = jax.jit(lambda st, dyn, raw: lax.scan(body, st, (dyn, raw)))
K = 24
rows = []
for c0 in range(0, t_hi, K):
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
    for j in range(K):
        rows.append(np.asarray(outK[j])[:, tile])
rows = np.array(rows)
bad = np.nonzero(~np.isfinite(rows[:,2]))[0]
print('first non-finite t1 at step', bad[:1], 'time', start+dt.timedelta(seconds=S.dt*(int(bad[0])+1)) if len(bad) else None)
print("step  time  | py sneqv, lis sneqv | py t1, lis t1 | py snowh, lis | py albedo, lis | py sncovr")
for s in range(t_lo, t_hi):
    i = s * T + tile
    print(s, start + dt.timedelta(seconds=S.dt * (s + 1)), "%.6f %.6f | %.4f %.4f | %.5f %.5f | %.4f %.4f | %.3f" % (
        rows[s][0], post["sneqv"][i], rows[s][2], post["t1"][i], rows[s][1], post["snowh"][i], rows[s][3], post["albedo"][i], rows[s][4]))
