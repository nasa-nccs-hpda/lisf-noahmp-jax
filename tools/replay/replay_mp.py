"""Open-loop replay of real LIS Noah-MP 4.0.1 NOAHMP_SFLX calls through the JAX port.
Usage: python lis/replay_mp.py <dumpdir> [max_records] [start]"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
import dumpio_mp
import os
if os.environ.get("MPVER") == "36":
    from lisf_noahmp_jax.physics import noahmp36_jax as nm
elif os.environ.get("MPVER") == "50":
    from lisf_noahmp_jax.physics import noahmp50_jax as nm
else:
    from lisf_noahmp_jax.physics import noahmp401_jax as nm

OPT = dict(dveg="IDVEG", crs="IOPT_CRS", btr="IOPT_BTR", run="IOPT_RUN", sfc="IOPT_SFC", frz="IOPT_FRZ", inf="IOPT_INF",
           rad="IOPT_RAD", alb="IOPT_ALB", snf="IOPT_SNF", tbot="IOPT_TBOT", stc="IOPT_STC", rsf="IOPT_RSF", tksno="IOPT_TKSNO", infdv="IOPT_INFDV", tdrn="IOPT_TDRN", irr="IOPT_IRR", irrm="IOPT_IRRM", crop="IOPT_CROP")
ARR_STATE = ("ZSOIL", "SMCEQ", "FICEOLD", "STC", "SMH2O", "SMC", "ZSNSO", "SNICE", "SNLIQ")


def split_inputs(pre):
    P = {k[2:]: v for k, v in pre.items() if k.startswith("p_")}
    x = {k: v for k, v in pre.items() if not k.startswith("p_")}
    x["ISNOW"] = x["ISNOW"].astype(np.int32)
    return P, x


def options(pre):
    return {k: (int(pre[v][0]) if v in pre else (0 if k in ("tdrn", "irr", "irrm", "crop") else 1)) for k, v in OPT.items()}


def main():
    ddir = sys.argv[1]
    maxr = int(sys.argv[2]) if len(sys.argv) > 2 else None
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    idx = None
    if len(sys.argv) > 4 and sys.argv[4].startswith("sample"):
        # sample:N -> N random records plus every record with snow on the ground
        n = int(sys.argv[4].split(":")[1]); rng = np.random.default_rng(1)
        p0, q0 = dumpio_mp.load(ddir)
        sn = np.where((p0["SWE"] > 0) | (p0["ISNOW"] < 0))[0]
        idx = np.unique(np.concatenate([rng.choice(len(p0["DT"]), n, replace=False), sn]))
        del p0, q0
    if len(sys.argv) > 4 and sys.argv[4].startswith("rand:"):
        n = int(sys.argv[4].split(":")[1]); p0, q0 = dumpio_mp.load(ddir)
        idx = np.sort(np.random.default_rng(2).choice(len(p0["DT"]), n, replace=False)); del p0, q0
    pre, post = dumpio_mp.load(ddir, maxr, start, idx)
    o = options(pre)
    print("options", o, "records", len(pre["DT"]))
    P, x = split_inputs(pre)

    def one(P, x):
        return nm.sflx_mp(P, x, o)
    fn = jax.jit(jax.vmap(one))
    N = len(x["DT"]); chunk = 20000; outs = []
    t = time.time()
    for i in range(0, N, chunk):
        sl = slice(i, i + chunk)
        Pj = {k: jnp.asarray(v[sl]) for k, v in P.items()}
        xj = {k: jnp.asarray(v[sl]) for k, v in x.items()}
        r = fn(Pj, xj)
        outs.append({k: np.stack([np.asarray(e) for e in v], -1) if isinstance(v, list) else np.asarray(v) for k, v in r.items()})
    d = {k: np.concatenate([o_[k] for o_ in outs], axis=0) for k in outs[0]}
    print(f"replay {N} calls in {time.time()-t:.1f}s")
    print(f"{'field':10s} {'range':>10s} {'p50':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s} {'bitwise%':>9s}  nan")
    for nme in d:
        if nme not in post:
            continue
        a = post[nme].astype(np.float64); b = np.asarray(d[nme]).astype(np.float64)
        if nme == "ISNOW":
            pass
        e = np.abs(a - b)
        good = np.isfinite(a) & np.isfinite(b)
        nan = int((~np.isfinite(b)).sum())
        if good.sum() == 0:
            print(f"{nme:10s} all nonfinite ({nan})"); continue
        e = e[good]
        rng = max(np.ptp(a[good]), 1e-30)
        q = np.percentile(e, [50, 99, 99.9, 100])
        print(f"{nme:10s} {rng:10.3g} {q[0]:9.2e} {q[1]:9.2e} {q[2]:9.2e} {q[3]:9.2e} {100*(a == b).mean():9.2f}  {nan}")
    pass


if __name__ == "__main__":
    main()
