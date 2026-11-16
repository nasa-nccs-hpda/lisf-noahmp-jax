"""Open-loop replay helpers used by the regression tests: run the JAX column physics on recorded inputs of the real LIS (GNU 14 serial) and
return the error statistics against the recorded Fortran outputs. Mirrors tools/replay/replay_mp.py."""
import os, sys
import numpy as np
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "tools", "replay"),):
    if p not in sys.path:
        sys.path.insert(0, p)


def stats(d, post, names):
    """nonfinite counts JAX values that are not finite where the recorded Fortran value is (option run=2 has water-soil tiles that are NaN in the Fortran itself)."""
    out = {}
    for n in names:
        a, b = np.asarray(post[n], np.float64), np.asarray(d[n], np.float64)
        ok = np.isfinite(a) & np.isfinite(b)
        e = np.abs(a - b)[ok]
        out[n] = dict(p50=float(np.percentile(e, 50)), p99=float(np.percentile(e, 99)), max=float(e.max()), nonfinite=int((np.isfinite(a) & ~np.isfinite(b)).sum()),
                      rng=float(np.ptp(a[ok])))
    return out


def noahmp(version, ddir):
    import dumpio_mp
    if version == "36":
        from lisf_noahmp_jax.physics import noahmp36_jax as nm
    else:
        from lisf_noahmp_jax.physics import noahmp401_jax as nm
    pre, post = dumpio_mp.load(ddir)
    P = {k[2:]: v for k, v in pre.items() if k.startswith("p_")}
    x = {k: v for k, v in pre.items() if not k.startswith("p_")}
    x["ISNOW"] = x["ISNOW"].astype(np.int32)
    OPT = dict(dveg="IDVEG", crs="IOPT_CRS", btr="IOPT_BTR", run="IOPT_RUN", sfc="IOPT_SFC", frz="IOPT_FRZ", inf="IOPT_INF",
               rad="IOPT_RAD", alb="IOPT_ALB", snf="IOPT_SNF", tbot="IOPT_TBOT", stc="IOPT_STC", rsf="IOPT_RSF")
    o = {k: (int(pre[v][0]) if v in pre else 1) for k, v in OPT.items()}
    fn = jax.jit(jax.vmap(lambda P_, x_: nm.sflx_mp(P_, x_, o)))
    r = fn({k: jnp.asarray(v) for k, v in P.items()}, {k: jnp.asarray(v) for k, v in x.items()})
    d = {k: np.stack([np.asarray(e) for e in v], -1) if isinstance(v, list) else np.asarray(v) for k, v in r.items()}
    names = [n for n in d if n in post]
    return stats(d, post, names), int((x["SWE"] > 0).sum() + (x["ISNOW"] < 0).sum())


def noahmp50(ddir, ice=False):
    """Noah-MP 5.0: land points through noahmp50_jax (sflx_mp), land-ice points through noahmp50_glacier_jax (sflx_glacier); dump adapter lis/dump50.py"""
    import dumpio_mp, dump50
    if ice:
        os.environ["DUMP50_KEEP_ICE"] = "1"
    else:
        os.environ.pop("DUMP50_KEEP_ICE", None)
    pre, post = dump50.load50(ddir)
    if ice:
        sel = pre["VEGTYP"] == pre["p_ISICE"]
        pre = {k: v[sel] for k, v in pre.items()}; post = {k: v[sel] for k, v in post.items()}
        from lisf_noahmp_jax.physics import noahmp50_glacier_jax as nm
        P = {k: pre["p_" + k] for k in ("ZBOT", "EICE", "SNOW_EMIS", "Z0SNO", "SWEMX", "ALBICE", "MNSNALB", "MXSNALB", "sndecayexp", "SSI", "SNOW_RET_FAC",
                                         "SNLIQMAXFRAC", "C2_SNOWCOMPACT", "C3_SNOWCOMPACT", "C4_SNOWCOMPACT", "C5_SNOWCOMPACT", "DM_SNOWCOMPACT",
                                         "ETA0_SNOWCOMPACT", "SWEMAXGLA")}
        x = {k: v for k, v in pre.items() if not k.startswith("p_")}
        x["PRCP"] = x["PRCPNONC"]; x["TBOT"] = np.minimum(x["TBOT"], np.float32(263.15))
        o = dict(alb=int(pre["IOPT_ALB"][0]), snf=int(pre["IOPT_SNF"][0]), tbot=int(pre["IOPT_TBOT"][0]), stc=int(pre["IOPT_STC"][0]), gla=1, sndpth=2000.0)
        fn = jax.jit(jax.vmap(lambda P_, x_: nm.sflx_glacier(P_, x_, o)))
    else:
        from lisf_noahmp_jax.physics import noahmp50_jax as nm
        P = {k[2:]: v for k, v in pre.items() if k.startswith("p_")}
        x = {k: v for k, v in pre.items() if not k.startswith("p_")}
        OPT = dict(dveg="IDVEG", crs="IOPT_CRS", btr="IOPT_BTR", run="IOPT_RUN", sfc="IOPT_SFC", frz="IOPT_FRZ", inf="IOPT_INF",
                   rad="IOPT_RAD", alb="IOPT_ALB", snf="IOPT_SNF", tbot="IOPT_TBOT", stc="IOPT_STC", rsf="IOPT_RSF", tksno="IOPT_TKSNO", infdv="IOPT_INFDV", tdrn="IOPT_TDRN", irr="IOPT_IRR", irrm="IOPT_IRRM", crop="IOPT_CROP")
        o = {k: (int(pre[v][0]) if v in pre else (0 if k in ("tdrn", "irr", "irrm", "crop") else 1)) for k, v in OPT.items()}
        fn = jax.jit(jax.vmap(lambda P_, x_: nm.sflx_mp(P_, x_, o)))
    x["ISNOW"] = x["ISNOW"].astype(np.int32)
    r = fn({k: jnp.asarray(v) for k, v in P.items()}, {k: jnp.asarray(v) for k, v in x.items()})
    d = {k: np.stack([np.asarray(e) for e in v], -1) if isinstance(v, list) else np.asarray(v) for k, v in r.items()}
    return stats(d, post, [n for n in d if n in post]), len(x["SWE"])
