"""List the replayed 5.0 calls with the largest surface runoff difference (debug aid for runoff options). usage: MPVER=50 python lis/dbg_runoff.py <dumpdir> [N=15000] [top=12]"""
import os, sys
import numpy as np, jax, jax.numpy as jnp
import dumpio_mp, dump50, replay_mp
dumpio_mp.load = dump50.load50
from lisf_noahmp_jax.physics import noahmp50_jax as nm
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 15000; top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
p0, q0 = dumpio_mp.load(d); idx = np.sort(np.random.default_rng(2).choice(len(p0["DT"]), min(n, len(p0["DT"])), replace=False)); del p0, q0
pre, post = dumpio_mp.load(d, 300000, 0, idx)
o = replay_mp.options(pre); P, x = replay_mp.split_inputs(pre)
r = jax.jit(jax.vmap(lambda P, x: nm.sflx_mp(P, x, o)))({k: jnp.asarray(v) for k, v in P.items()}, {k: jnp.asarray(v) for k, v in x.items()})
SMJ = np.stack([np.asarray(e) for e in r["SMC"]], -1)
a = post["RUNSF"].astype(np.float64); b = np.asarray(r["RUNSF"]).astype(np.float64)
e = np.abs(a - b); w = np.argsort(-e)[:top]
print("options", o)
for i in w:
    print("rec %6d  fort %.4e jax %.4e  diff %.2e | SMC pre %s | SMC post F %s J %s | PRCP %.2e" % (i, a[i], b[i], e[i], np.round(pre["SMC"][i], 4), np.round(post["SMC"][i], 4), np.round(SMJ[i], 4), pre["PRCPNONC"][i]))
print("n diff>1e-7:", int((e > 1e-7).sum()), "of", len(e), "; with Fortran runoff>0:", int((a > 0).sum()))
