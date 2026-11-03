"""Free-running replay: each real tile is run by the JAX port for the whole week using only the recorded
forcing, parameters and the first-step state; compared with the real LIS Noah-3.9 state/flux at every step."""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import dumpio, replay
from lisf_noahmp_jax.physics import noah39_jax as nj

hdr, pre, post = dumpio.load(sys.argv[1])
tile = hdr[:, 6]; ntile = int(tile.max()); nt = len(hdr) // ntile
assert (tile.reshape(nt, ntile) == np.arange(1, ntile + 1)).all()
R = lambda a: a.reshape((nt, ntile) + a.shape[1:])
par, st0, forc = replay.build({k: v[:ntile] for k, v in pre.items()})
DYN = "shdfac shdmin shdmax alb snoalb tbot z0brd embrd lai".split()   # time-varying parameters set by LIS dynsetup
FK = "sfctmp q2 soldn lwdn sfcprs prcp ffrozp q2sat dqsdt2 th2 sfcspd_b".split() + DYN
fseq = {k: jnp.asarray(R(pre[k])) for k in FK}
parj = {k: jnp.asarray(v) for k, v in par.items()}
st0j = {k: jnp.asarray(v) for k, v in st0.items()}
st0j["ch"] = jnp.asarray(pre["ch0"][:ntile]); st0j["cm"] = jnp.asarray(pre["cm0"][:ntile])
sldpth = [jnp.float32(x) for x in (0.1, 0.3, 0.6, 1.0)]
dt = float(pre["dt"][0]); isurban = int(pre["isurban"][0])


def one(par, st, f):
    q2 = f["q2"]
    t1v = st["t1"] * (1.0 + 0.61 * q2); th2v = f["th2"] * (1.0 + 0.61 * q2)
    cm, ch, _ = nj.sfcdif_off(jnp.float32(2.0), jnp.float32(10.0), st["z0"], t1v, th2v, f["sfcspd_b"],
                              par["czil"], st["cm"], st["ch"])
    forc = dict(sfctmp=f["sfctmp"], q2=q2, soldn=f["soldn"], lwdn=f["lwdn"], sfcprs=f["sfcprs"], prcp=f["prcp"],
                ffrozp=f["ffrozp"], q2sat=f["q2sat"], dqsdt2=f["dqsdt2"], th2=f["th2"],
                solnet=f["soldn"] * (1.0 - st["albedo"]), ch=ch, cm=cm)
    s = {k: st[k] for k in "cmc t1 snowh sneqv albedo z0 snotime1 xlai emissi stc smc sh2o".split()}
    s["xlai"] = f["lai"]
    par = dict(par); par.update({k: f[k] for k in DYN if k != "lai"})
    ns, out = nj.sflx(forc, par, s, jnp.float32(dt), jnp.float32(2.0), sldpth, opt_thcnd=1, isurban=isurban,
                      aoasis=jnp.float32(1.0))
    ns = dict(ns); ns["ch"], ns["cm"] = ch, cm
    return ns, out


axes_st = {k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in
           "cmc t1 snowh sneqv albedo z0 snotime1 xlai emissi stc smc sh2o ch cm".split()}
axes_out = {k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in
            "cmc t1 snowh sneqv albedo sncovr z0 snotime1 q1 emissi xlai stc smc sh2o ch cm".split()}
vone = jax.vmap(one, in_axes=({k: 0 for k in parj}, axes_st, {k: 0 for k in FK}), out_axes=(axes_out, 0))


def body(st, f):
    ns, out = vone(parj, st, f)
    keep = {k: ns[k] for k in axes_st}
    flat = [ns[k] for k in ("cmc", "t1", "snowh", "sneqv", "albedo", "z0", "emissi")] + \
           [ns["stc"][i] for i in range(4)] + [ns["smc"][i] for i in range(4)] + [ns["sh2o"][i] for i in range(4)] + \
           [out[k] for k in ("eta", "sheat", "ssoil", "etp")]
    return keep, jnp.stack(flat)


run = jax.jit(lambda s, f: lax.scan(body, s, f))
t = time.time(); _, o = run(st0j, fseq); o.block_until_ready(); print(f"free-running week for {ntile} tiles: {time.time()-t:.1f}s incl compile")
o = np.asarray(o)
names = "cmc t1 snowh sneqv albedo z0 emissi stc1 stc2 stc3 stc4 smc1 smc2 smc3 smc4 sh2o1 sh2o2 sh2o3 sh2o4 eta sheat ssoil etp".split()
per_day = nt // 7
print("max |JAX - LIS| over all tiles, by day (state fields) ; range of the field in parentheses")
print(f"{'field':8s} {'range':>9s} " + " ".join(f"{'day'+str(d+1):>9s}" for d in range(7)) + "   p99(all)")
for i, n in enumerate(names):
    ref = R(post[n]).astype(np.float64); jx = o[:, i].astype(np.float64)
    e = np.abs(ref - jx)
    rng = max(ref.max() - ref.min(), 1e-30)
    days = [e[d*per_day:(d+1)*per_day].max() for d in range(7)]
    print(f"{n:8s} {rng:9.3g} " + " ".join(f"{x:9.2e}" for x in days) + f"   {np.percentile(e, 99):9.2e}")

if len(sys.argv) > 2:
    i = names.index(sys.argv[2])
    ref = R(post[sys.argv[2]]).astype(np.float64); jx = o[:, i].astype(np.float64)
    e = np.abs(ref - jx); k = np.unravel_index(np.argmax(e), e.shape)
    tl = k[1]
    print("worst tile (0-based)", tl, "veg", par["vegt"][tl], "soil", par["soiltyp"][tl], "at step", k[0])
    for s in list(range(0, nt, nt // 14)):
        print(s, "ref %.6f jax %.6f | sneqv ref %.3e jax %.3e | t1 ref %.3f jax %.3f" % (
            ref[s, tl], jx[s, tl], R(post["sneqv"])[s, tl], o[s, names.index("sneqv"), tl], R(post["t1"])[s, tl], o[s, names.index("t1"), tl]))
