"""List the replayed 5.0 calls with the largest difference in one output field (debug aid). usage: MPVER=50 python lis/dbg_field.py <dumpdir> FIELD [N=15000] [top=8]"""
import sys
import numpy as np, jax, jax.numpy as jnp
import dumpio_mp, dump50, replay_mp
dumpio_mp.load = dump50.load50
from lisf_noahmp_jax.physics import noahmp50_jax as nm
d, fld = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 15000; top = int(sys.argv[4]) if len(sys.argv) > 4 else 8
p0, q0 = dumpio_mp.load(d); idx = np.sort(np.random.default_rng(2).choice(len(p0["DT"]), min(n, len(p0["DT"])), replace=False)); del p0, q0
pre, post = dumpio_mp.load(d, 300000, 0, idx)
o = replay_mp.options(pre); P, x = replay_mp.split_inputs(pre)
r = jax.jit(jax.vmap(lambda P, x: nm.sflx_mp(P, x, o)))({k: jnp.asarray(v) for k, v in P.items()}, {k: jnp.asarray(v) for k, v in x.items()})
a = post[fld].astype(np.float64); b = np.asarray(r[fld]).astype(np.float64); e = np.abs(a - b)
for i in np.argsort(-e)[:top]:
    print("rec %6d veg %2d %s F %.5g J %.5g | LAIin %.4f SAIin %.4f FVEG %.3f FVGMAX %.3f LAIout F %.4f J %.4f FVEGMP F %.3f J %.3f SWDN %.0f TV %.2f" % (
        i, pre["VEGTYP"][i], fld, a[i], b[i], pre["PLAI"][i], pre["PSAI"][i], pre["FVEG"][i], pre["FVGMAX"][i], post["PLAI"][i], np.asarray(r["PLAI"])[i],
        post["FVEGMP"][i], np.asarray(r["FVEGMP"])[i], pre["SWDN"][i], pre["TV"][i]))
