import os; os.environ.setdefault("LISJAX_FORCING_PROCS", "1")   # no __main__ guard in this script: keep the forcing reader serial (spawned workers would re-run it)
import sys, datetime as dt
import numpy as np, dumpio
from lisf_noahmp_jax.driver.forcing import NLDAS2
hdr, pre, post = dumpio.load("run/d1/jaxdump_noah39.bin")
nt, T = 672, 616
R = lambda a: a.reshape(nt, T)
col = hdr[:T, 7] - 1; row = hdr[:T, 8] - 1
f = NLDAS2("run/INPUT/NLDAS2.FORCING", 34.375 + 0.25 * row, -102.875 + 0.25 * col)
t0 = dt.datetime(2017, 2, 1, 0, 0)
errs = {k: [] for k in ("sfctmp", "q2", "sfcprs", "soldn", "lwdn", "prcp", "sfcspd")}
for s in range(nt):
    now = t0 + dt.timedelta(minutes=15 * (s + 1))
    o = f.at(now)
    emiss = R(pre["emiss"])[s]
    mine = {"sfctmp": o["Tair"], "q2": np.maximum(o["Qair"], np.float32(1e-6)), "sfcprs": o["PSurf"],
            "soldn": o["SWdown"], "lwdn": o["LWdown"] * emiss, "prcp": o["Rainf"],
            "sfcspd": np.maximum(np.sqrt(o["Wind_E"] ** 2 + o["Wind_N"] ** 2), np.float32(0.01))}
    for k in errs:
        errs[k].append(np.abs(mine[k].astype(np.float64) - R(pre[k])[s]))
print(f"{'field':8s} {'p50':>10s} {'p99':>10s} {'max':>10s} {'bitwise%':>9s}")
for k, v in errs.items():
    e = np.stack(v); print(f"{k:8s} {np.percentile(e,50):10.2e} {np.percentile(e,99):10.2e} {e.max():10.2e} {100*(e==0).mean():9.2f}")
