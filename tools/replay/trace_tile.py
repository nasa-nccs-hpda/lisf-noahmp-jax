import sys, datetime as dt
import numpy as np, jax, jax.numpy as jnp
import dumpio
from lisf_noahmp_jax.driver.params import Setup
from lisf_noahmp_jax.driver.forcing import NLDAS2
from lisf_noahmp_jax.driver import model
cfg, dump, tile, nstep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
S = Setup(cfg, "run"); F = NLDAS2("run/INPUT/NLDAS2.FORCING", S.lat, S.lon)
hdr, pre, post = dumpio.load(dump)
T = S.ntile
step = model.make_step(S)
par = {k: jnp.asarray(v) for k, v in S.par.items()}
start = dt.datetime(2017, 1, 1)
st = {k: jnp.asarray(v) for k, v in S.initial_state(2017, 1, 1).items()}
dyn_c = {}
out = []
for s in range(nstep):
    now = start + dt.timedelta(seconds=S.dt * (s + 1))
    key = (now.year, now.month, now.day)
    newday = (now.hour == 0 and now.minute == 0) or s == 0
    if (key, newday) not in dyn_c:
        dyn_c[(key, newday)] = {k: jnp.asarray(v) for k, v in S.dyn(*key, newday=newday).items()}
    o = F.at(now)
    raw = {k: jnp.asarray(v) for k, v in {"tair": o["Tair"], "qair": o["Qair"], "swd": o["SWdown"], "lwd": o["LWdown"],
           "psurf": o["PSurf"], "pcp": o["Rainf"], "u": o["Wind_E"], "v": o["Wind_N"]}.items()}
    st, diag = step(par, dyn_c[(key, newday)], st, raw)
    i = s * T + tile
    out.append((now, float(st["sneqv"][tile]), float(post["sneqv"][i]), float(st["t1"][tile]), float(post["t1"][i]),
                float(st["snowh"][tile]), float(post["snowh"][i]), float(st["albedo"][tile]), float(post["albedo"][i]),
                float(st["stc"][0][tile]), float(post["stc1"][i]), float(raw["tair"][tile]), float(raw["swd"][tile])))
a = np.array([[o[1], o[2]] for o in out])
d = np.abs(a[:, 0] - a[:, 1])
first = np.argmax(d > 1e-3)
print("first step with |sneqv diff| > 1e-3 m:", first, out[first][0] if d.max() > 1e-3 else None)
lo, hi = max(0, first - 6), min(len(out), first + 10)
print("time  sneqv(py,lis)  t1(py,lis)  snowh(py,lis)  albedo(py,lis)  stc1(py,lis) tair swd")
for o in out[lo:hi]:
    print(o[0], "%.6f %.6f | %.4f %.4f | %.5f %.5f | %.4f %.4f | %.3f %.3f | %.2f %.0f" % o[1:])
