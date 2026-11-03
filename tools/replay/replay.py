"""Open-loop replay of real LIS Noah-3.9 SFLX calls through the JAX port.
Usage: python lis/replay.py jaxdump_noah39.bin [max_records]"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
import dumpio
from lisf_noahmp_jax.physics import noah39_jax as nj

PAR_KEYS = ("vegt soiltyp nroot slope shdfac shdmin shdmax alb snoalb tbot z0brd embrd smcwlt smcdry smcref smcmax "
            "hs rgl snup cmcmax rsmax topt bexp dksat dwsat f1 quartz psisat czil sbeta fxexp csoil salp kdt cfactr "
            "zbot refkdt frzx lvcoef rsmin ztopv zbotv").split()
FORC_KEYS = "sfctmp q2 soldn lwdn sfcprs prcp ffrozp q2sat dqsdt2 th2 solnet ch cm ribb".split()


def build(pre):
    par = {k: pre[k] for k in PAR_KEYS}
    par["vegt"] = pre["vegt"].astype(np.int32); par["soiltyp"] = pre["soiltyp"].astype(np.int32)
    par["nroot"] = pre["nroot"].astype(np.int32)
    st = {k: pre[k] for k in "cmc t1 snowh sneqv albedo z0 snotime1".split()}
    st["xlai"] = pre["lai"]; st["emissi"] = pre["emiss"]
    for g in ("stc", "smc", "sh2o"):
        st[g] = np.stack([pre[f"{g}{k}"] for k in range(1, 5)])
    forc = {k: pre[k] for k in FORC_KEYS}
    return par, st, forc


def make_fn(dt, isurban, opt_thcnd, aoasis, ua=False):
    sldpth = [jnp.float32(x) for x in (0.1, 0.3, 0.6, 1.0)]

    def one(par, st, forc):
        return nj.sflx(forc, par, st, jnp.float32(dt), jnp.float32(2.0), sldpth, opt_thcnd=opt_thcnd,
                       isurban=isurban, aoasis=jnp.float32(aoasis), ua=ua)
    st_axes = {k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in
               "cmc t1 snowh sneqv albedo z0 snotime1 xlai emissi stc smc sh2o".split()}
    return jax.jit(jax.vmap(one, in_axes=({k: 0 for k in PAR_KEYS}, st_axes, {k: 0 for k in FORC_KEYS}),
                            out_axes=({k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in
                                       "cmc t1 snowh sneqv albedo sncovr z0 snotime1 q1 emissi xlai stc smc sh2o".split()}, 0)))


def flatten(new_st, out):
    d = {}
    for k, v in new_st.items():
        v = np.asarray(v)
        if k in ("stc", "smc", "sh2o"):
            for i in range(4):
                d[f"{k}{i+1}"] = v[i]
        else:
            d[k] = v
    for k, v in out.items():
        d[k] = np.asarray(v)
    return d


def report(d, post, title):
    print(title)
    print(f"{'field':10s} {'range':>10s} {'p50':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s} {'bitwise%':>9s}")
    for n in dumpio.POST_NAMES:
        if n not in d:
            continue
        a, b = post[n].astype(np.float64), np.asarray(d[n]).astype(np.float64)
        e = np.abs(a - b); rng = max(a.max() - a.min(), 1e-30)
        q = np.percentile(e, [50, 99, 99.9, 100])
        print(f"{n:10s} {rng:10.3g} {q[0]:9.2e} {q[1]:9.2e} {q[2]:9.2e} {q[3]:9.2e} {100*(a == b).mean():9.2f}")


if __name__ == "__main__":
    hdr, pre, post = dumpio.load(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
    par, st, forc = build(pre)
    ua = bool(pre["uaflag"][0]) if "uaflag" in pre else False
    fn = make_fn(float(pre["dt"][0]), int(pre["isurban"][0]), int(pre["opt_thcnd"][0]), float(pre["aoasis"][0]), ua)
    N = len(hdr); chunk = 100000; outs = []
    t = time.time()
    for i in range(0, N, chunk):
        sl = slice(i, i + chunk)
        ps = {k: jnp.asarray(v[sl]) for k, v in par.items()}
        ss = {k: jnp.asarray(v[..., sl]) for k, v in st.items()}
        fs = {k: jnp.asarray(v[sl]) for k, v in forc.items()}
        outs.append(flatten(*fn(ps, ss, fs)))
    d = {k: np.concatenate([o[k] for o in outs], axis=-1) for k in outs[0]}
    print(f"{N} real SFLX calls replayed in {time.time()-t:.1f}s (incl. compile)")
    report(d, post, "OPEN LOOP: JAX sflx() vs real LIS Noah-3.9 SFLX, same recorded inputs")

    # ---- SFCDIF check
    opt = int(pre["sfcdifopt"][0]) if "sfcdifopt" in pre else 0
    if opt == 0:
        sf = jax.jit(jax.vmap(lambda z0, t1v, th2v, spd, czil, cm0, ch0: nj.sfcdif_off(
            jnp.float32(2.0), jnp.float32(10.0), z0, t1v, th2v, spd, czil, cm0, ch0)))
        cm, ch, ustar = [np.asarray(x) for x in sf(*[jnp.asarray(pre[k]) for k in
                                                     ("z0_old", "t1v", "th2v", "sfcspd_b", "czil", "cm0", "ch0")])]
        d2 = {"ch": ch, "cm": cm, "ustar": ustar}
        report(d2, post, "SFCDIF_OFF: JAX vs real LIS (zlvl=2 m, zlvl_wind=10 m from the config)")
    else:
        tab = nj.myj_tables()
        sf = jax.jit(jax.vmap(lambda z0, z0b, prs, t1, tair, q1, q2, spd, czil, cm0, ch0: nj.sfcdif_myj(
            jnp.float32(2.0), jnp.float32(10.0), z0, z0b, prs, t1, tair, q1, q2, spd, czil, cm0, ch0, tab)))
        cm, ch, rib, ustar = [np.asarray(x) for x in sf(*[jnp.asarray(pre[k]) for k in
            ("z0_old", "z0brd_old", "sfcprs", "t1", "sfctmp", "q1_pre", "q2", "sfcspd_b", "czil", "cm0", "ch0")])]
        d2 = {"cm": cm, "ustar": ustar}
        # recorded pre ch/cm are the values AFTER sfcdif but BEFORE the UA modification of CH inside SFLX
        e = {"ch(after sfcdif)": np.abs(ch - pre["ch"]), "cm(after sfcdif)": np.abs(cm - pre["cm"]), "ribb": np.abs(rib - pre["ribb"])}
        print("SFCDIF_MYJ: JAX vs real LIS")
        for k, v in e.items():
            print(f"{k:18s} p50 {np.percentile(v,50):9.2e} p99 {np.percentile(v,99):9.2e} max {v.max():9.2e} bitwise% {100*(v==0).mean():6.2f}")
