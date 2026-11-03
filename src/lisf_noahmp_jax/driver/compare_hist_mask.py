"""compare_hist with the land cells whose soil type is water (Soiltype 14, which LIS resets to sandy clay loam and which are numerically unstable in
the Fortran as well) excluded.   usage: python3 -m lisf_noahmp_jax.driver.compare_hist_mask REF_DIR NEW_DIR [first_file_glob_year]"""
import sys, glob, os
import numpy as np
import netCDF4 as nc

ref_dir, new_dir = sys.argv[1:3]
files = sorted(glob.glob(f"{ref_dir}/SURFACEMODEL/*/LIS_HIST_*.nc"))
VARS = ["SWE_inst", "SnowDepth_inst", "SoilMoist_inst", "SoilTemp_inst", "SmLiqFrac_inst", "CanopInt_inst", "Snowcover_inst"]
mask = None; acc = {}; n = 0
for f in files:
    g = f.replace(ref_dir, new_dir)
    if not os.path.exists(g):
        continue
    A, B = nc.Dataset(f), nc.Dataset(g); n += 1
    if mask is None:
        mask = np.ma.filled(B["Soiltype_inst"][:].astype(float), np.nan).squeeze() == 14
        print("excluded tiles (soil type water):", int(mask.sum()))
    for v in VARS:
        x = np.ma.filled(A[v][:].astype(float), np.nan); y = np.ma.filled(B[v][:].astype(float), np.nan)
        e = np.abs(x - y)
        e = e[..., ~mask] if e.ndim == mask.ndim + 1 or e.ndim > 2 else e[~mask]
        r = np.nanmax(x[..., ~mask]) - np.nanmin(x[..., ~mask]) if x.ndim > 2 else np.nanmax(x[~mask]) - np.nanmin(x[~mask])
        acc.setdefault(v, []).append((e[np.isfinite(e)], r))
print(f"{n} history files compared ({ref_dir} vs {new_dir})")
print(f"{'variable':16s} {'range':>10s} {'p50':>10s} {'p99':>10s} {'max':>10s} {'bitwise%':>9s}")
for v, l in acc.items():
    e = np.concatenate([a for a, _ in l]); r = max(b for _, b in l)
    print(f"{v:16s} {r:10.3g} {np.percentile(e,50):10.2e} {np.percentile(e,99):10.2e} {e.max():10.2e} {100*(e==0).mean():9.2f}")
