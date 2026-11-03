"""Open-loop replay of real LIS SFLX_GLACIAL calls (records tagged hdr[5]==99)."""
import sys
import numpy as np, jax, jax.numpy as jnp
import dumpio, replay
from lisf_noahmp_jax.physics import noah39_jax as nj

hdr, pre, post = dumpio.load(sys.argv[1])
m = hdr[:, 5] == 99
pre = {k: v[m] for k, v in pre.items()}; post = {k: v[m] for k, v in post.items()}
print("glacier records:", int(m.sum()))
sld = [jnp.float32(x) for x in (0.1, 0.3, 0.6, 1.0)]
dt = jnp.float32(pre["dt"][0])

def one(forc, par, st):
    return nj.sflx_glacial(forc, par, st, dt, sld)

forc = {k: jnp.asarray(pre[k]) for k in ("lwdn solnet sfcprs prcp sfctmp q2 th2 q2sat dqsdt2 ffrozp ch ribb").split()}
par = {"alb": jnp.asarray(pre["alb"]), "snoalb": jnp.asarray(pre["snoalb"]), "tbot": jnp.asarray(pre["tbot"]),
       "z0brd": jnp.asarray(pre["z0brd"]), "embrd": jnp.asarray(pre["embrd"]), "lvcoef": jnp.asarray(pre["lvcoef"])}
st = {"t1": jnp.asarray(pre["t1"]), "stc": jnp.asarray(np.stack([pre[f"stc{k}"] for k in range(1, 5)])),
      "snowh": jnp.asarray(pre["snowh"]), "sneqv": jnp.asarray(pre["sneqv"]), "snotime1": jnp.asarray(pre["snotime1"]),
      "sncovr": jnp.asarray(pre["sca_pre"])}
f = jax.jit(jax.vmap(one, in_axes=({k: 0 for k in forc}, {k: 0 for k in par}, {"t1": 0, "stc": 1, "snowh": 0, "sneqv": 0, "snotime1": 0, "sncovr": 0}),
                     out_axes=({"t1": 0, "stc": 1, "snowh": 0, "sneqv": 0, "albedo": 0, "z0": 0, "emissi": 0, "snotime1": 0, "q1": 0, "sncovr": 0}, 0)))
ns, out = f(forc, par, st)
d = {k: np.asarray(v) for k, v in ns.items() if k != "stc"}
for i in range(4):
    d[f"stc{i+1}"] = np.asarray(ns["stc"])[i]
d.update({k: np.asarray(v) for k, v in out.items()})
d["sheat"] = d["sheat"]
replay.report(d, post, "GLACIAL: JAX sflx_glacial vs real LIS SFLX_GLACIAL (same inputs)")
