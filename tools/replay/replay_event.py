"""Replay of one cell's calls from a real-LIS Noah-MP 5.0 dump through the JAX port, every step from the Fortran pre-state (open loop).
Used to decide whether a closed-loop flux outlier is a port error or state sensitivity (docs/reports/flux_outliers_mp5_60d_20261111.md).
usage: MPVER=50 python3 replay_event.py <dumpdir> <lat> <lon> [first_step last_step]"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
import dumpio_mp, dump50, replay_mp
from lisf_noahmp_jax.physics import noahmp50_jax as nm

ddir, lat, lon = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
s0, s1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (1, 10 ** 6)
pn = dumpio_mp.read_layout(f"{ddir}/jaxdump_mp_pre.layout"); qn = dumpio_mp.read_layout(f"{ddir}/jaxdump_mp_post.layout")
npre, npost = sum(pn[1]), sum(qn[1])
dt = np.dtype([("w1", ">i4"), ("n1", ">i4"), ("pre", ">f4", (npre,)), ("w2", ">i4"), ("n2", ">i4"), ("post", ">f4", (npost,))])
mm = np.memmap(f"{ddir}/jaxdump_mp.bin", dtype=dt, mode="r", shape=(os.path.getsize(f"{ddir}/jaxdump_mp.bin") // dt.itemsize,))
off = dict(zip(pn[0], np.concatenate([[0], np.cumsum(pn[1])[:-1]])))
col = lambda k: np.asarray(mm["pre"][:, off[k]])
la, lo, ts = col("xlat"), col("xlon"), col("itimestep")
d = (la - lat) ** 2 + (lo - lon) ** 2
j = int(np.argmin(d)); la0, lo0 = la[j], lo[j]
idx = np.nonzero((la == la0) & (lo == lo0) & (ts >= s0) & (ts <= s1))[0]
print(f"tile lat {la0:.4f} lon {lo0:.4f}: {len(idx)} records, steps {int(ts[idx].min())}-{int(ts[idx].max())}")
pre, post = dump50.load50(ddir, idx=idx)
o = replay_mp.options(pre); P, x = replay_mp.split_inputs(pre)
r = jax.jit(jax.vmap(lambda P, x: nm.sflx_mp(P, x, o)))({k: jnp.asarray(v) for k, v in P.items()}, {k: jnp.asarray(v) for k, v in x.items()})
r = {k: np.stack([np.asarray(e) for e in v], -1) if isinstance(v, list) else np.asarray(v) for k, v in r.items()}
F = ["FSH", "SSOIL", "FSA", "FIRA", "FCEV", "FGEV", "FCTR", "TG", "TV", "SWE", "SNDPTH", "FSNO", "ALBOLD", "QSNBOT"]
print("step " + " ".join(f"{k:>8s}(F/J)" for k in F[:7]))
for i in range(len(idx)):
    st = int(pre["ITIMESTEP"][i])
    if st >= 30 or st % 6 == 0:
        print(f"{st:3d} SWE0 {pre['SWE'][i]:6.2f} Tair {pre['T_ML'][i]:6.1f} " + " ".join(f"{post[k][i]:8.1f}/{r[k][i]:8.1f}" for k in F[:7] if k in r and k in post)
              + f" | TG {post['TG'][i]:.2f}/{r['TG'][i]:.2f} SWE {post['SWE'][i]:.3f}/{r['SWE'][i]:.3f} FSNO {post['FSNO'][i] if 'FSNO' in post else float('nan'):.3f}/{r['FSNO'][i]:.3f}")
print("max |J-F| over all replayed steps:")
for k in F:
    if k in r and k in post:
        e = np.abs(post[k].astype(float) - r[k].astype(float)); print(f"  {k:8s} max {e.max():.3g} at step {int(pre['ITIMESTEP'][int(e.argmax())])}")
