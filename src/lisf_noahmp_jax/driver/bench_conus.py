"""Benchmark the Python/JAX LIS run on the CONUS domain. Usage: python bench_conus.py base_dir out_dir days [days2]
Prints one JSON line with phase timings (cold run, including JIT compilation) and steady-state per-chunk device time."""
import os; os.environ.setdefault("LISJAX_FORCING_PROCS", "1")   # no __main__ guard in this script: keep the forcing reader serial (spawned workers would re-run it)
import sys, json, time, datetime as dt, platform
import jax
from lisf_noahmp_jax.driver import run

base, out, days = sys.argv[1], sys.argv[2], int(sys.argv[3])
start = dt.datetime(2017, 2, 1)
tm = {}
t0 = time.time()
run.run(f"{base}/lis.config", base, f"{base}/INPUT/NLDAS2.FORCING", out, start, start + dt.timedelta(days=days), timing=tm, progress=False)
ch = tm.pop("chunk_device_s")
d = jax.devices()[0]
res = dict(base=base, days=days, device=str(d), kind=d.device_kind, jax=jax.__version__, host=platform.node(),
           total_s=tm["total"], setup_s=tm["setup"], forcing_host_s=tm["forcing_host"], device_s=tm["device"], output_s=tm["output"], routing_s=tm.get("routing"),
           first_chunk_device_s=ch[0], steady_chunk_device_s_median=sorted(ch[1:])[len(ch[1:]) // 2] if len(ch) > 1 else None,
           n_chunks=len(ch))
if res["steady_chunk_device_s_median"]:
    res["steady_device_s_per_simulated_day"] = res["steady_chunk_device_s_median"] * (24 / 6)
print("BENCH", json.dumps(res))
