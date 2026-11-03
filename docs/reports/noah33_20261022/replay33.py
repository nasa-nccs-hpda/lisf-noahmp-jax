"""Open-loop replay of the real LIS Noah-3.3 SFLX calls (dump made with lis_patches/make_noah33_hook.py) through the JAX port.
usage: python lis/replay33.py DUMP.bin [first_record] [n_records]   (records are time-ordered, one per tile and step)"""
import sys, numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, "lis"); sys.path.insert(0, ".")
import dumpio, replay
import noah39_jax as nj

path = sys.argv[1]
with open(path, "rb") as f:
    nh, npre, npost = (int(x) for x in np.fromfile(f, ">i4", 3))
dt = np.dtype([("hdr", ">i4", nh), ("pre", ">f4", npre), ("post", ">f4", npost)])
mm = np.memmap(path, dtype=dt, mode="r", offset=12)
first = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(sys.argv[3]) if len(sys.argv) > 3 else len(mm) - first
rec = np.array(mm[first:first + n])
pre = {k: rec["pre"][:, i].astype(np.float32) for i, k in enumerate(dumpio.PRE_NAMES)}
post = {k: rec["post"][:, i].astype(np.float32) for i, k in enumerate(dumpio.POST_NAMES)}
hdr = rec["hdr"]
print(len(rec), "records; first", hdr[0][:6], "last", hdr[-1][:6])
par, st, forc = replay.build(pre)
sldpth = [jnp.float32(x) for x in (0.1, 0.3, 0.6, 1.0)]
isurban, opt = int(pre["isurban"][0]), 0
one = lambda p, s, f: nj.sflx(f, p, s, jnp.float32(pre["dt"][0]), jnp.float32(2.0), sldpth, opt_thcnd=opt, isurban=isurban,
                              aoasis=jnp.float32(1.0), ua=False, v36=True, v33=True)
st_axes = {k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in "cmc t1 snowh sneqv albedo z0 snotime1 xlai emissi stc smc sh2o".split()}
fn = jax.jit(jax.vmap(one, in_axes=({k: 0 for k in replay.PAR_KEYS}, st_axes, {k: 0 for k in replay.FORC_KEYS}),
                      out_axes=({k: (1 if k in ('stc', 'smc', 'sh2o') else 0) for k in 'cmc t1 snowh sneqv albedo sncovr z0 snotime1 q1 emissi xlai stc smc sh2o'.split()}, 0)))
outs = []
for i in range(0, len(rec), 100000):
    sl = slice(i, i + 100000)
    outs.append(replay.flatten(*fn({k: jnp.asarray(v[sl]) for k, v in par.items()}, {k: jnp.asarray(v[..., sl]) for k, v in st.items()},
                                 {k: jnp.asarray(v[sl]) for k, v in forc.items()})))
d = {k: np.concatenate([o[k] for o in outs], axis=-1) for k in outs[0]}
replay.report(d, post, "OPEN LOOP: JAX sflx() (Noah 3.3 mode) vs real LIS Noah-3.3 SFLX")
np.savez("/tmp/replay33_last.npz", **{f"d_{k}": v for k, v in d.items()}, hdr=hdr, **{f"p_{k}": v for k, v in post.items()}, **{f"i_{k}": v for k, v in pre.items()})
