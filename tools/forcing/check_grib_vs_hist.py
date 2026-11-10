"""Compare the driver's NLDAS-2 GRIB reading with the forcing fields that the Fortran LIS wrote to a history file (LIS_HIST_*.d01.nc, *_f_inst variables).
usage: python check_grib_vs_hist.py HIST.nc GRIB1.grb [GRIB2.grb ...]     (the GRIB files are candidate hours: at an exact hour the instantaneous forcing equals one of them)
Prints the _f_inst variables of the history file and, per variable and candidate hour, the number of cells equal bitwise, within 1e-4 relative, and the max abs difference."""
import sys
import numpy as np
import netCDF4 as nc
from lisf_noahmp_jax.driver.forcing import read_grib

hist, gribs = sys.argv[1], sys.argv[2:]
d = nc.Dataset(hist)
print("dims", {k: len(v) for k, v in d.dimensions.items()})
fv = sorted(k for k in d.variables if k.endswith("_f_inst"))
print("forcing variables in history:", fv)
MAP = {"Tair_f_inst": "Tair", "Qair_f_inst": "Qair", "Psurf_f_inst": "PSurf", "LWdown_f_inst": "LWdown", "SWdown_f_inst": "SWdown", "Rainf_f_inst": "Rainf",
       "CRainf_f_inst": "CRainf_frac", "CAPE_f_inst": "CAPE", "PotEvap_f_inst": "PotEvap"}
flds = {g: read_grib(g)[0] for g in gribs}
for hv in fv:
    a = np.ma.filled(np.ma.asarray(d[hv][:]).astype(np.float64), np.nan)
    a = a.reshape(a.shape[-2:])
    for g, f in flds.items():
        if hv == "Wind_f_inst":
            b = np.sqrt(f["Wind_E"].astype(np.float64) ** 2 + f["Wind_N"].astype(np.float64) ** 2)
        elif hv in MAP:
            b = f[MAP[hv]].astype(np.float64)
            if hv == "Rainf_f_inst":
                b = b / 3600.0
        else:
            continue
        ok = np.isfinite(a) & (a != -9999.0) & (b != -9999.0)
        if not ok.any():
            print(f"{hv:16s} {g[-26:]}: no valid cells"); continue
        diff = np.abs(a[ok] - b[ok])
        eq = int((a[ok].astype(np.float32) == b[ok].astype(np.float32)).sum())
        rel = int((diff <= 1e-4 * np.maximum(np.abs(b[ok]), 1e-30)).sum())
        print(f"{hv:16s} {g[-26:]}: cells {ok.sum():6d} bitwise(f32) {eq:6d} within 1e-4 rel {rel:6d} max abs diff {diff.max():.3e}")
