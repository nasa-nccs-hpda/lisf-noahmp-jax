"""Replay of the land-ice (VEGTYP == ISICE) calls of the real LIS Noah-MP 5.0 driver (NoahmpMainGlacier) through the JAX land-ice port.
Usage: GLA=401|50 python lis/replay_gla50.py <dumpdir> [max_records]   (GLA=50 uses noahmp50_glacier_jax.py)"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
os.environ["DUMP50_KEEP_ICE"] = "1"
import dumpio_mp, dump50
if os.environ.get("GLA", "50") == "50":
    from lisf_noahmp_jax.physics import noahmp50_glacier_jax as ng
else:
    from lisf_noahmp_jax.physics import noahmp401_glacier_jax as ng


def main():
    ddir = sys.argv[1]
    maxr = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pre, post = dump50.load50(ddir, maxr, 0, None)
    ice = pre["VEGTYP"] == pre["p_ISICE"]
    pre = {k: v[ice] for k, v in pre.items()}; post = {k: v[ice] for k, v in post.items()}
    N = len(pre["DT"])
    o = dict(alb=int(pre["IOPT_ALB"][0]), snf=int(pre["IOPT_SNF"][0]), tbot=int(pre["IOPT_TBOT"][0]), stc=int(pre["IOPT_STC"][0]),
             gla=1, sndpth=2000.0)
    print("options", o, "ice records", N)
    x = {k: v for k, v in pre.items() if not k.startswith("p_")}
    x["PRCP"] = x["PRCPNONC"]; x["ISNOW"] = x["ISNOW"].astype(np.int32)
    x["TBOT"] = np.minimum(x["TBOT"], np.float32(263.15))
    P = {k: pre["p_" + k] for k in ("ZBOT", "EICE", "SNOW_EMIS", "Z0SNO", "SWEMX", "ALBICE", "MNSNALB", "MXSNALB", "sndecayexp", "SSI", "SNOW_RET_FAC", "SNLIQMAXFRAC", "C2_SNOWCOMPACT", "C3_SNOWCOMPACT", "C4_SNOWCOMPACT", "C5_SNOWCOMPACT", "DM_SNOWCOMPACT", "ETA0_SNOWCOMPACT", "SWEMAXGLA")}
    fn = jax.jit(jax.vmap(lambda P, x: ng.sflx_glacier(P, x, o)))
    chunk = 20000; outs = []
    t = time.time()
    for i in range(0, N, chunk):
        sl = slice(i, i + chunk)
        r = fn({k: jnp.asarray(v[sl]) for k, v in P.items()}, {k: jnp.asarray(v[sl]) for k, v in x.items()})
        outs.append({k: np.stack([np.asarray(e) for e in v], -1) if isinstance(v, list) else np.asarray(v) for k, v in r.items()})
    d = {k: np.concatenate([o_[k] for o_ in outs], axis=0) for k in outs[0]}
    print(f"replay {N} calls in {time.time()-t:.1f}s")
    print(f"{'field':10s} {'range':>10s} {'p50':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s} {'bitwise%':>9s}  nan")
    for nme in d:
        if nme not in post:
            continue
        a = post[nme].astype(np.float64); b = np.asarray(d[nme]).astype(np.float64)
        e = np.abs(a - b); good = np.isfinite(a) & np.isfinite(b)
        nan = int((~np.isfinite(b)).sum())
        if good.sum() == 0:
            print(f"{nme:10s} all nonfinite ({nan})"); continue
        e = e[good]; rng = max(np.ptp(a[good]), 1e-30)
        q = np.percentile(e, [50, 99, 99.9, 100])
        print(f"{nme:10s} {rng:10.3g} {q[0]:9.2e} {q[1]:9.2e} {q[2]:9.2e} {q[3]:9.2e} {100*(a == b).mean():9.2f}  {nan}")


if __name__ == "__main__":
    main()
