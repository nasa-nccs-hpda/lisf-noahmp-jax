"""JAX Noah 3.3, 2 days CONUS, with a rounding-size perturbation of the initial state (stc, smc, sh2o, t1 multiplied by 1 + u*1e-6,
u uniform in [-1, 1], fixed seed), for comparison with the unperturbed JAX run. usage: python this.py BASE OUT"""
import sys, datetime as dt, numpy as np
sys.path.insert(0, ".")
from lisjax import run, params
base, out = sys.argv[1], sys.argv[2]
orig = params.Setup.initial_state
def noisy(self, y, m, d):
    st = orig(self, y, m, d); rng = np.random.default_rng(1)
    for k in ("stc", "smc", "sh2o", "t1"):
        st[k] = (st[k] * (1 + 1e-6 * rng.uniform(-1, 1, st[k].shape))).astype(np.float32)
    return st
params.Setup.initial_state = noisy
s = dt.datetime(2017, 2, 1)
run.run(f"{base}/lis.config", base, f"{base}/INPUT/NLDAS2.FORCING", out, s, s + dt.timedelta(hours=48), progress=False)
