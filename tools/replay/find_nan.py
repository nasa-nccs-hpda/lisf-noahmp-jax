import os; os.environ.setdefault("LISJAX_FORCING_PROCS", "1")   # no __main__ guard in this script: keep the forcing reader serial (spawned workers would re-run it)
import sys, datetime as dt
import numpy as np, jax, jax.numpy as jnp
from lisf_noahmp_jax.driver.params import Setup
from lisf_noahmp_jax.driver.forcing import NLDAS2
from lisf_noahmp_jax.driver import model
S = Setup("run/lis.config.n39ua", "run"); F = NLDAS2("run/INPUT/NLDAS2.FORCING", S.lat, S.lon)
step = model.make_step(S)
par = {k: jnp.asarray(v) for k, v in S.par.items()}
start = dt.datetime(2017, 1, 1)
st = {k: jnp.asarray(v) for k, v in S.initial_state(2017, 1, 1).items()}
tile = 430
dyn_c = {}
for s in range(1802):
    now = start + dt.timedelta(seconds=S.dt * (s + 1))
    key = (now.year, now.month, now.day)
    newday = (now.hour == 0 and now.minute == 0) or s == 0
    if (key, newday) not in dyn_c:
        dyn_c[(key, newday)] = {k: jnp.asarray(v) for k, v in S.dyn(*key, newday=newday).items()}
    o = F.at(now)
    raw = {k: jnp.asarray(v) for k, v in {"tair": o["Tair"], "qair": o["Qair"], "swd": o["SWdown"], "lwd": o["LWdown"],
           "psurf": o["PSurf"], "pcp": o["Rainf"], "u": o["Wind_E"], "v": o["Wind_N"]}.items()}
    prev = st
    st, diag = step(par, dyn_c[(key, newday)], st, raw)
    if not np.isfinite(float(st["t1"][tile])):
        print("NaN at step", s, now)
        print("PRE-state tile:", {k: np.asarray(v)[..., tile].tolist() for k, v in prev.items()})
        print("forcing tile:", {k: float(v[tile]) for k, v in raw.items()})
        print("dyn tile:", {k: float(v[tile]) for k, v in dyn_c[(key, newday)].items()})
        print("veg", S.vegt[tile], "ztopv", S.par["ztopv"][tile], "zbotv", S.par["zbotv"][tile], "snup", S.par["snup"][tile])
        break
