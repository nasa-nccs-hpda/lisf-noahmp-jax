import sys, datetime as dt
import numpy as np, dumpio
from lisf_noahmp_jax.driver.params import Setup
from lisf_noahmp_jax.driver.forcing import MERRA2
cfg, dump, start, nsteps = sys.argv[1], sys.argv[2], dt.datetime.strptime(sys.argv[3], "%Y%m%d"), int(sys.argv[4])
S = Setup(cfg, "run/glob/lis1")
hdr, pre, post = dumpio.load(dump)
nt_tiles = int(hdr[:, 6].max()); nst = len(hdr) // nt_tiles
assert (hdr[:nt_tiles, 6] == np.arange(1, nt_tiles + 1)).all()
F = MERRA2("/css/merra2", S.lat[:nt_tiles], S.lon[:nt_tiles])
R = lambda a: a.reshape(nst, nt_tiles)
errs = {k: [] for k in ("sfctmp", "q2", "sfcprs", "soldn", "lwdn", "prcp", "sfcspd")}
for s in range(min(nst, nsteps)):
    now = start + dt.timedelta(minutes=15 * (s + 1))
    o = F.at(now)
    emiss = R(pre["emiss"])[s]
    mine = {"sfctmp": o["Tair"], "q2": np.maximum(o["Qair"], np.float32(1e-6)), "sfcprs": o["PSurf"], "soldn": o["SWdown"],
            "lwdn": o["LWdown"] * emiss, "prcp": o["Rainf"],
            "sfcspd": np.maximum(np.sqrt(o["Wind_E"] ** 2 + o["Wind_N"] ** 2), np.float32(0.01))}
    for k in errs:
        errs[k].append(np.abs(mine[k].astype(np.float64) - R(pre[k])[s]))
print(f"{'field':8s} {'p50':>10s} {'p99':>10s} {'max':>10s} {'bitwise%':>9s}")
for k, v in errs.items():
    e = np.stack(v); print(f"{k:8s} {np.percentile(e,50):10.2e} {np.percentile(e,99):10.2e} {e.max():10.2e} {100*(e==0).mean():9.2f}")
