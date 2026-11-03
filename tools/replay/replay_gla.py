"""Open-loop replay of real LIS Noah-MP 4.0.1 NOAHMP_GLACIER calls through the JAX land-ice port.
Usage: python lis/replay_gla.py <dumpdir> [max_records]"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
import dumpio_mp
from lisf_noahmp_jax.physics import noahmp401_jax as nm
from lisf_noahmp_jax.physics import noahmp401_glacier_jax as ng



def main():
    ddir = sys.argv[1]
    maxr = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pre, post = dumpio_mp.load(ddir, maxr, 0, None, "jaxdump_gla")
    N = len(pre["DT"])
    o = dict(alb=int(pre["IOPT_ALB"][0]), snf=int(pre["IOPT_SNF"][0]), tbot=int(pre["IOPT_TBOT"][0]),
             stc=int(pre["IOPT_STC"][0]), gla=int(pre["IOPT_GLA"][0]), sndpth=float(pre["IOPT_SNDPTH"][0]))
    print("options", o, "records", N)
    x = dict(pre)
    x["ISNOW"] = x["ISNOW"].astype(np.int32)
    x["PRCPNONC"] = x["PRCP"]; x["PRCPCONV"] = np.zeros(N, np.float32); x["PRCPSHCV"] = np.zeros(N, np.float32)
    P = {"ZBOT": np.full(N, -8.0, np.float32)}

    def one(P, x):
        return ng.sflx_glacier(P, x, o)
    fn = jax.jit(jax.vmap(one))
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
