"""Open-loop replay of real LIS Noah-2.7.1 SFLX calls (dump from lis_patches/make_noah271_hook.py) through the JAX port.
usage: python lis/replay271.py DUMP.bin [first_record] [n_records]"""
import sys, numpy as np
import jax, jax.numpy as jnp
from lisf_noahmp_jax.physics import noah271_jax as n2

PRE = ("sbeta fxexp czil cfactr cmcmax rsmax topt refdk refkdt csoil frzk ffrozp ice dt zlvl lwdn soldn sfcprs prcp sfctmp q2 sfcspd th2 q2sat dqsdt2 "
       "slope shdfac shdmin alb snoalb rsmin rgl hs snup z0 emissi xlai nroot psisat bexp dksat smcmax quartz dwsat smcwlt smcref smcdry f1 kdt frzx tbot "
       "cmc t1 stc1 stc2 stc3 stc4 smc1 smc2 smc3 smc4 sh2o1 sh2o2 sh2o3 sh2o4 snowh sneqv ch0 cm0 t1v th2v soiltyp vegt").split()
POST = ("cmc t1 stc1 stc2 stc3 stc4 smc1 smc2 smc3 smc4 sh2o1 sh2o2 sh2o3 sh2o4 snowh sneqv albedo ch cm evp eta sheat eta_kin ec edir et1 et2 et3 et4 ett esnow "
        "drip dew beta etp gflx flx1 flx2 flx3 snomlt sncovr ro1 ro2 ro3 rc pc rcs rct rcq rcsoil q1 soilm etanrg tsoil emissi_out ustar soilw soilt").split()
assert len(PRE) == 73 and len(POST) == 58

def load(path, first=0, n=None):
    with open(path, "rb") as f:
        nh, npre, npost = (int(x) for x in np.fromfile(f, ">i4", 3))
    dt = np.dtype([("hdr", ">i4", nh), ("pre", ">f4", npre), ("post", ">f4", npost)])
    mm = np.memmap(path, dtype=dt, mode="r", offset=12)
    n = len(mm) - first if n is None else n
    rec = np.array(mm[first:first + n])
    pre = {k: rec["pre"][:, i].astype(np.float32) for i, k in enumerate(PRE)}
    post = {k: rec["post"][:, i].astype(np.float32) for i, k in enumerate(POST)}
    return rec["hdr"], pre, post

PAR = "nroot shdfac alb snoalb snup smcwlt smcdry smcref smcmax emissi quartz sbeta rsmin topt rsmax rgl hs xlai cmcmax cfactr slope kdt frzx psisat dksat dwsat tbot zbot fxexp csoil bexp emiss_out0".split()
FORC = "sfctmp q2 soldn lwdn sfcprs prcp ffrozp q2sat dqsdt2 th2 ch".split()

def build(pre, post):
    n = len(pre["dt"])
    par = {k: pre[k] for k in ("snoalb shdfac alb snup smcwlt smcdry smcref smcmax quartz sbeta rsmin topt rsmax rgl hs cmcmax cfactr slope kdt frzx "
                               "psisat dksat dwsat tbot fxexp csoil bexp").split()}
    par["xlai"] = pre["xlai"]; par["emissi"] = pre["emissi"]; par["emiss_out0"] = pre["emissi"]
    par["nroot"] = pre["nroot"].astype(np.int32)
    par["zbot"] = np.full(n, -8.0, np.float32)
    st = {k: pre[k] for k in "cmc t1 snowh sneqv".split()}
    for g in ("stc", "smc", "sh2o"):
        st[g] = np.stack([pre[f"{g}{k}"] for k in range(1, 5)])
    forc = {k: pre[k] for k in FORC if k not in ("ch",)}
    forc["ch"] = post["ch"]
    return par, st, forc

def flatten(new_st, out):
    d = {}
    for k, v in new_st.items():
        v = np.asarray(v)
        if k in ("stc", "smc", "sh2o"):
            for i in range(4): d[f"{k}{i+1}"] = v[i]
        else: d[k] = v
    for k, v in out.items():
        if k == "et":
            for i in range(4): d[f"et{i+1}"] = np.asarray(v[i])
        else: d[k] = np.asarray(v)
    d["ro1"], d["ro2"], d["ro3"] = d.pop("ro1"), d.pop("ro2"), d.pop("ro3")
    d["sheat"] = d["sheat"]; d["albedo"] = d["albedo"]
    return d

def report(d, post, title, names=None):
    print(title)
    print(f"{'field':10s} {'range':>10s} {'p50':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s} {'bitwise%':>9s}")
    for n in (names or POST):
        if n not in d: continue
        a, b = post[n].astype(np.float64), np.asarray(d[n]).astype(np.float64)
        e = np.abs(a - b); rng = max(np.nanmax(a) - np.nanmin(a), 1e-30)
        bad = ~np.isfinite(e)
        q = np.percentile(e[~bad], [50, 99, 99.9, 100]) if (~bad).any() else [np.nan] * 4
        print(f"{n:10s} {rng:10.3g} {q[0]:9.2e} {q[1]:9.2e} {q[2]:9.2e} {q[3]:9.2e} {100*(a == b).mean():9.2f}  nonfinite={bad.sum()}")

if __name__ == "__main__":
    first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n = int(sys.argv[3]) if len(sys.argv) > 3 else None
    hdr, pre, post = load(sys.argv[1], first, n)
    print(len(hdr), "records; first", hdr[0][:6], "last", hdr[-1][:6])
    par, st, forc = build(pre, post)
    sld = [jnp.float32(x) for x in (0.1, 0.3, 0.6, 1.0)]
    dt = jnp.float32(pre["dt"][0])
    one = lambda p, s, f: n2.sflx(f, p, s, dt, sld)
    st_ax = {k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in "cmc t1 snowh sneqv stc smc sh2o".split()}
    fn = jax.jit(jax.vmap(one, in_axes=({k: 0 for k in par}, st_ax, {k: 0 for k in forc}),
                          out_axes=(st_ax, 0)))
    outs = []
    N = len(hdr)
    for i in range(0, N, 100000):
        sl = slice(i, i + 100000)
        outs.append(flatten(*fn({k: jnp.asarray(v[sl]) for k, v in par.items()}, {k: jnp.asarray(v[..., sl]) for k, v in st.items()},
                                {k: jnp.asarray(v[sl]) for k, v in forc.items()})))
    d = {k: np.concatenate([o[k] for o in outs], axis=-1) for k in outs[0]}
    report(d, post, "OPEN LOOP: JAX sflx() (Noah 2.7.1) vs real LIS SFLX")
    sf = jax.jit(jax.vmap(lambda z0, t1v, th2v, spd, czil, cm0, ch0: n2.sfcdif(jnp.float32(pre["zlvl"][0]), z0, t1v, th2v, spd, czil, cm0, ch0)))
    cm, ch, ustar = [np.asarray(x) for x in sf(*[jnp.asarray(pre[k]) for k in ("z0", "t1v", "th2v", "sfcspd", "czil", "cm0", "ch0")])]
    report({"ch": ch, "cm": cm, "ustar": ustar}, post, "SFCDIF: JAX vs real LIS", ["ch", "cm", "ustar"])
    np.savez("/tmp/replay271_last.npz", hdr=hdr, **{f"d_{k}": v for k, v in d.items()}, **{f"p_{k}": v for k, v in post.items()}, **{f"i_{k}": v for k, v in pre.items()})
