"""Open-loop replay of the real Mosaic call (dump from lis_patches/make_mos_hook.py) through the JAX port.
usage: python lis/replay_mos.py DUMP.bin CONFIG BASE first_step n_steps    (records are time-ordered; N tiles per step)"""
import sys, numpy as np, datetime as dt
sys.path.insert(0, "."); sys.path.insert(0, "lisjax")
import jax, jax.numpy as jnp
from lisjax import run_mos as R

path, cfg, base = sys.argv[1:4]
first, nst = int(sys.argv[4]), int(sys.argv[5])
S = R.SetupMos(cfg, base)
N = S.ntile
with open(path, "rb") as f:
    nh, npre, npost = (int(x) for x in np.fromfile(f, ">i4", 3))
dtp = np.dtype([("hdr", ">i4", nh), ("pre", ">f4", npre), ("post", ">f4", npost)])
mm = np.memmap(path, dtype=dtp, mode="r", offset=12)
rec = np.array(mm[first * N:(first + nst) * N])
names = (["vegt"] + [f"vegp{i}" for i in range(1, 25)] + [f"vegip{i}" for i in range(1, 7)] + [f"soilp{i}" for i in range(1, 11)] +
         "tmp2m humid swdn lwdwn wnd sfcprs tpcp cpcp sun0 lat lon ct sot qa sowet1 sowet2 sowet3 ics snow".split() +
         "dedqa dedtc hsturb dhsdqa dhsdtc qm cd sunang pardir pardif swnet psur zlai green z2 sqscat rsoil1 rsoil2 rdc u2fac qsattc dqsdtc alwrad blwrad ra eturb tm".split())
pn = "ct sot qa sowet1 sowet2 sowet3 ics snow evap shflux runoff eint esoi eveg esno smelt hlatn hlwup gdrain runsrf fwsoil ghflux water1 water2 water3 por1 por2 por3 albave snwfrc tempac tempcc".split()
P = {n: rec["pre"][:, i].astype(np.float32) for i, n in enumerate(names)}
Q = {n: rec["post"][:, i].astype(np.float32) for i, n in enumerate(pn)}
print(len(rec), "records, first", rec["hdr"][0][:6], "last", rec["hdr"][-1][:6])
# parameters check against the dump
for k, a in (("vegt", S.vegt.astype(np.float32)),):
    print("vegt equal:", np.array_equal(a, P["vegt"][:N]))
bad = [(i + 1, np.abs(S.vegp[i] - P[f"vegp{i+1}"][:N]).max()) for i in range(24) if not np.array_equal(S.vegp[i], P[f"vegp{i+1}"][:N])]
print("vegp mismatches:", bad)
bad = [(i + 1, float(np.abs(S.soilp[i] - P[f"soilp{i+1}"][:N]).max())) for i in range(10) if not np.allclose(S.soilp[i], P[f"soilp{i+1}"][:N], rtol=1e-6, atol=0)]
print("soilp mismatches:", bad)
step = jax.jit(R.make_step(S, debug=True))
par = {"vegt": jnp.asarray(S.vegt), "vegp": jnp.asarray(S.vegp), "soilp": jnp.asarray(S.soilp)}
res = {}
for s in range(nst):
    sl = slice(s * N, (s + 1) * N)
    g = lambda k: jnp.asarray(P[k][sl])
    f = dict(tair=g("tmp2m"), qair=g("humid"), psurf=g("sfcprs"), swd=g("swdn"), lwd=g("lwdwn"), u=g("wnd"), v=jnp.zeros(N, jnp.float32), pcp=g("tpcp"), cpcp=g("cpcp"))
    dyn = dict(green=g("vegip1"), lai=g("vegip2"), vip3=g("vegip3"), vip4=g("vegip4"), vip5=g("vegip5"), vip6=g("vegip6"), sunang=g("sun0"))
    st = {"ct": g("ct"), "sot": g("sot"), "qa": g("qa"), "ics": g("ics"), "snow": g("snow"), "sowet": jnp.stack([g("sowet1"), g("sowet2"), g("sowet3")])}
    new, diag = step(par, st, f, dyn)
    cmp = {"ct": new["ct"], "sot": new["sot"], "qa": new["qa"], "sowet1": new["sowet"][0], "sowet2": new["sowet"][1], "sowet3": new["sowet"][2], "ics": new["ics"], "snow": new["snow"]}
    for k in ("evap", "shflux", "runoff", "eint", "esoi", "eveg", "esno", "smelt", "hlatn", "hlwup", "gdrain", "runsrf", "fwsoil", "ghflux", "albave", "snwfrc", "tempcc"):
        cmp[k] = diag["x_" + k]
    cmp.update(water1=diag["x_water1"], water2=diag["x_water2"], water3=diag["x_water3"], por1=diag["x_por1"], por2=diag["x_por2"], por3=diag["x_por3"])
    for k, v in cmp.items():
        res.setdefault(k, []).append((np.asarray(v, np.float64), Q[k][sl].astype(np.float64)))
    for k in ("dedqa", "dedtc", "hsturb", "dhsdqa", "dhsdtc", "qm", "cd", "pardir", "swnet", "zlai", "z2", "sqscat", "rsoil1", "rsoil2", "rdc", "u2fac", "qsattc", "dqsdtc", "alwrad", "blwrad", "ra", "eturb"):
        res.setdefault("arg:" + k, []).append((np.asarray(diag["x_" + k], np.float64), P[k][sl].astype(np.float64)))
print(f"{'field':14s} {'range':>10s} {'p50':>9s} {'p99':>9s} {'max':>9s} {'bitwise%':>9s}")
for k, l in res.items():
    a = np.concatenate([x for x, _ in l]); b = np.concatenate([y for _, y in l])
    e = np.abs(a - b); ok = np.isfinite(e)
    rng = max(np.nanmax(b) - np.nanmin(b), 1e-30)
    print(f"{k:14s} {rng:10.3g} {np.percentile(e[ok],50):9.2e} {np.percentile(e[ok],99):9.2e} {e[ok].max():9.2e} {100*(a==b).mean():9.2f}   nonfinite {int((~ok).sum())}")
