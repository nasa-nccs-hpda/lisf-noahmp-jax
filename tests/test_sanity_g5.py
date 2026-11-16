"""Gate G5 (sanity), water balance: per-step closure of the Noah-MP 5.0 column on the recorded one-step replay data (tests/data/noahmp50, 599 records).
Residual = d(storage) - (P - ECAN - ETRAN - ESOIL - RUNSF - RUNSB) * DT, storage = soil water + SWE + canopy water + aquifer WA (mm).
Computed the same way from the Fortran post-state and from the JAX output. Pass rule fixed 2026-11-17 (PORT_ASSURANCE.md): JAX p99 and max no worse
than Fortran's x 1.1 + 1e-5 mm. Measured on first run: both 4.7e-4 mm (p99), 5.0e-4 mm (max), identical. Energy and range checks: driver/sanity.py, run in test_example_sgp.py."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..", "tools", "replay"), os.path.join(HERE, "..", "tools", "lis_patches")):
    if p not in sys.path:
        sys.path.insert(0, p)
OPT = dict(dveg="IDVEG", crs="IOPT_CRS", btr="IOPT_BTR", run="IOPT_RUN", sfc="IOPT_SFC", frz="IOPT_FRZ", inf="IOPT_INF", rad="IOPT_RAD", alb="IOPT_ALB",
           snf="IOPT_SNF", tbot="IOPT_TBOT", stc="IOPT_STC", rsf="IOPT_RSF", tksno="IOPT_TKSNO", infdv="IOPT_INFDV", tdrn="IOPT_TDRN", irr="IOPT_IRR", irrm="IOPT_IRRM")


def _resid(pre, out):
    z = pre["ZSOIL"]
    dz = np.concatenate([-z[:, :1], z[:, :-1] - z[:, 1:]], 1)
    sto = lambda s: (s["SMC"] * dz * 1000).sum(1) + s["SWE"] + s["CANLIQ"] + s["CANICE"] + s["WA"]
    prcp = pre["PRCPCONV"] + pre["PRCPNONC"] + pre["PRCPSHCV"]
    flux = (prcp - out["ECAN"] - out["ETRAN"] - out["ESOIL"] - out["RUNSF"] - out["RUNSB"]) * pre["DT"]
    e = np.abs(sto(out) - sto(pre) - flux)
    return e[np.isfinite(e)]


def test_noahmp50_water_balance():
    import dump50
    from lisf_noahmp_jax.physics import noahmp50_jax as nm
    pre, post = dump50.load50(os.path.join(HERE, "data", "noahmp50"))
    P = {k[2:]: v for k, v in pre.items() if k.startswith("p_")}
    x = {k: v for k, v in pre.items() if not k.startswith("p_")}
    o = {k: (int(pre[v][0]) if v in pre else (0 if k in ("tdrn", "irr", "irrm", "crop") else 1)) for k, v in OPT.items()}
    x["ISNOW"] = x["ISNOW"].astype(np.int32)
    fn = jax.jit(jax.vmap(lambda P_, x_: nm.sflx_mp(P_, x_, o)))
    r = fn({k: jnp.asarray(v) for k, v in P.items()}, {k: jnp.asarray(v) for k, v in x.items()})
    d = {k: (np.stack([np.asarray(e) for e in v], -1) if isinstance(v, list) else np.asarray(v)) for k, v in r.items()}
    ef, ej = _resid(pre, post), _resid(pre, d)
    assert ef.size > 500 and ej.size > 500
    for q in (99, 100):
        a, b = np.percentile(ef, q), np.percentile(ej, q)
        assert b <= 1.1 * a + 1e-5, f"water residual p{q}: JAX {b:.3g} mm above Fortran {a:.3g} mm"
