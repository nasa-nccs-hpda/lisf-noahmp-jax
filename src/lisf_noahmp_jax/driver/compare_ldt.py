"""Compare two LDT parameter files variable by variable: python -m lisf_noahmp_jax.driver.compare_ldt reference.nc new.nc"""
import sys
import numpy as np, netCDF4 as nc
a, b = nc.Dataset(sys.argv[1]), nc.Dataset(sys.argv[2])
print(f"{'variable':14s} {'max|diff|':>12s} {'max rel':>10s} {'bitwise%':>9s}")
worst = 0.0
for v in a.variables:
    if v == "time": continue
    if v not in b.variables: print(f"{v:14s} MISSING in new file"); continue
    x = np.ma.filled(a[v][:].astype(np.float64), np.nan); y = np.ma.filled(b[v][:].astype(np.float64), np.nan)
    if x.shape != y.shape: print(f"{v:14s} SHAPE {x.shape} vs {y.shape}"); continue
    d = np.abs(x - y); d = d[np.isfinite(d)]
    rel = (np.abs(x - y) / np.maximum(np.abs(x), 1e-12))[np.isfinite(x - y)]
    print(f"{v:14s} {d.max() if d.size else 0:12.3e} {rel.max() if rel.size else 0:10.2e} {100*np.mean((x.astype(np.float32)==y.astype(np.float32))|(np.isnan(x)&np.isnan(y))):9.2f}")
    worst = max(worst, rel.max() if rel.size else 0)
print("worst relative difference:", worst)
