"""Time the JAX Noah 3.9 day on the current default device. Usage: python bench.py case.bin [reps]
Prints one JSON line. Timing covers the whole 48-step jitted scan (physics + SFCDIF + all outputs
stacked) with block_until_ready; input transfer and file output are excluded. Also writes the
outputs of the last run next to the case as *_jax_free.bin for the correctness gate."""
import sys, time, json, platform
import numpy as np
import jax, jax.numpy as jnp
import run_case as rc

case = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N, NT, par, st0, forc = rc.load_case(case)
day, parj, step = rc.make_runner(par, N, NT)
forcj = {k: jax.device_put(jnp.asarray(v)) for k, v in forc.items()}
stj = jax.device_put(rc.to_st(st0))
t = time.perf_counter(); _, o = day(stj, forcj); o.block_until_ready(); first = time.perf_counter() - t
times = []
for _ in range(reps):
    t = time.perf_counter(); _, o = day(stj, forcj); o.block_until_ready(); times.append(time.perf_counter() - t)
np.asarray(o).astype(np.float32).tofile(case.replace(".bin", "_jax_free.bin"))
d = jax.devices()[0]
print(json.dumps(dict(case=case, N=N, steps=NT, device=str(d), kind=d.device_kind, jax=jax.__version__,
                      first_call_s=first, steady_min_s=min(times), steady_median_s=float(np.median(times)),
                      us_per_col_step=1e6 * min(times) / (N * NT), host=platform.node())))
