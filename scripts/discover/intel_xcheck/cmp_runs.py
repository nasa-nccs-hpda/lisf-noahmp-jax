"""Compare two LIS output directories (SURFACEMODEL/*/LIS_HIST_*.nc) of the same case: a wrong rank block shows as many cells with a large forcing difference.
usage: python cmp_runs.py REF_DIR NEW_DIR [VAR=Tair_f_inst]    prints files compared, bad cells (|diff| > 0.01 or NaN mismatch) summed over files, per-file maximum, bounding box"""
import sys, glob, os
import numpy as np
import netCDF4 as nc
ref, new = sys.argv[1], sys.argv[2]
var = sys.argv[3] if len(sys.argv) > 3 else "Tair_f_inst"
tot, nf, mx, box, worst = 0, 0, 0.0, [999, -1, 999, -1], 0
for f in sorted(glob.glob(f"{ref}/SURFACEMODEL/*/LIS_HIST_*.nc")):
    g = f.replace(ref, new)
    if not os.path.exists(g):
        continue
    a = np.ma.filled(nc.Dataset(f)[var][:].astype(float), np.nan); b = np.ma.filled(nc.Dataset(g)[var][:].astype(float), np.nan)
    d = np.abs(a - b)
    bad = (np.isfinite(d) & (d > 0.01)) | (np.isnan(a) != np.isnan(b))
    bad = bad.reshape((-1,) + bad.shape[-2:]).any(0)   # layers/time folded into one 2-D map of bad cells
    nf += 1; tot += int(bad.sum()); worst = max(worst, int(bad.sum()))
    if np.isfinite(d).any():
        mx = max(mx, float(np.nanmax(d)))
    if bad.any():
        r, c = np.where(bad); box = [min(box[0], r.min()), max(box[1], r.max()), min(box[2], c.min()), max(box[3], c.max())]
print(f"{os.path.basename(new.rstrip('/'))} vs {os.path.basename(ref.rstrip('/'))}: files {nf} bad cells (sum) {tot} worst file {worst} max abs diff {mx:.4g} bbox {box if tot else None}")
