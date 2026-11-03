"""Compare two directories of LIS_HIST netCDF files (reference vs Python/JAX)."""
import sys, glob, os
import numpy as np
import netCDF4 as nc

ref_dir, new_dir = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{ref_dir}/SURFACEMODEL/*/LIS_HIST_*.nc"))
acc = {}
nan_extra = {}
n = 0
for f in files:
    g = f.replace(ref_dir, new_dir)
    if not os.path.exists(g):
        continue
    A, B = nc.Dataset(f), nc.Dataset(g); n += 1
    for v in A.variables:
        if v in ("lat", "lon", "time") or v not in B.variables:
            continue
        x = np.ma.filled(A[v][:].astype(float), np.nan); y = np.ma.filled(B[v][:].astype(float), np.nan)
        e = np.abs(x - y).ravel()
        nan_extra[v] = max(nan_extra.get(v, 0), int((np.isnan(y) & ~np.isnan(x)).sum()))   # cells that are NaN only in the new run
        e = e[np.isfinite(e)]
        rng = np.nanmax(x) - np.nanmin(x)
        acc.setdefault(v, []).append((e, rng))
print(f"{n} history files compared ({ref_dir} vs {new_dir})")
print(f"{'variable':16s} {'range':>10s} {'p50':>10s} {'p99':>10s} {'max':>10s} {'bitwise%':>9s}")
for v, l in acc.items():
    e = np.concatenate([a for a, _ in l]); r = max(b for _, b in l)
    if e.size == 0:
        print(f"{v:16s} no finite common values (all-NaN in one run)"); continue
    print(f"{v:16s} {r:10.3g} {np.percentile(e,50):10.2e} {np.percentile(e,99):10.2e} {e.max():10.2e} {100*(e==0).mean():9.2f}")
bad = {v: k for v, k in nan_extra.items() if k}
print("variables with NaN only in the new run (cells):", bad if bad else "none")
