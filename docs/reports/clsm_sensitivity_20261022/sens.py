import sys; sys.path.insert(0,'.'); sys.path.insert(0,'lis')
import numpy as np, jax.numpy as jnp
import dumpio_clsm as D, clsm_jax as C
from replay_clsm import build_x
d="/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run/conus_clsm/d6i"
idx,recs,lay,dn=D.load(d,maxstep=2)
pre=recs[0][2]
x=build_x(pre); o0=C.clsmf25(jnp.float32(pre["dtstep"]),jnp.float32(pre["sfrac"]),x)
rng=np.random.default_rng(0)
for amp in (1.,8.):
    x2=build_x(pre); x2["ght"]=[g+jnp.asarray(rng.uniform(-amp,amp,g.shape).astype(np.float32)) for g in x2["ght"]]
    o=C.clsmf25(jnp.float32(pre["dtstep"]),jnp.float32(pre["sfrac"]),x2)
    e=np.abs(np.asarray(o["tc2"])-np.asarray(o0["tc2"]))
    print("ght noise +-%g J/m2: tc2 change max %.3g, cells>0.05: %d"%(amp,e.max(),(e>0.05).sum()))
for k in ("hlatn","shflux","tsurf","swland"):
    e=np.abs(np.asarray(o[k])-np.asarray(o0[k])); print(k,"noise8: p99 %.3g max %.3g cells>2: %d"%(np.percentile(e,99),e.max(),(e>2).sum()))
