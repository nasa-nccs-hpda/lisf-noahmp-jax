"""Gate G5 (sanity): NaN/Inf, physical range and balance residuals of a run, next to the same numbers for the Fortran reference.

Pass rule (PORT_ASSURANCE.md, set in advance): no NaN/Inf and no range violation that the reference does not also have; the
energy residual of the JAX run no worse than the reference's (p99 and max within 1.1 x reference + a small absolute slack).
Energy residual R = Swnet + Lwnet - Qle - Qh - Qg (W/m2, per history file; sign of Lwnet fixed by the reference closure).
Water balance is NOT checked here: the JAX history has no TWS or instantaneous Qs/Qsb. It is checked per step on the one-step replay data
(tests/test_sanity_g5.py).
usage: python -m lisf_noahmp_jax.driver.sanity REF_DIR NEW_DIR      (each holds SURFACEMODEL/*/LIS_HIST_*.nc)
"""
import glob
import os
import sys
import numpy as np
import netCDF4 as nc

RANGES = dict(SoilMoist_inst=(0.0, 1.0), SWE_inst=(0.0, 1e4), SnowDepth_inst=(0.0, 100.0), CanopInt_inst=(0.0, 50.0), SoilTemp_inst=(150.0, 350.0),
              Albedo_inst=(0.0, 1.0), Snowcover_inst=(0.0, 1.0), SmLiqFrac_inst=(0.0, 1.0), Qle_inst=(-1000.0, 1500.0), Qh_inst=(-1000.0, 1500.0))
FLUX = ("Swnet_inst", "Lwnet_inst", "Qle_inst", "Qh_inst", "Qg_inst")


def _get(d, k):
    return np.ma.filled(d[k][:].astype(float), np.nan)


def check(run_dir):
    """Return dict: nonfinite, range violations, energy residual array, water residual array."""
    files = sorted(glob.glob(os.path.join(run_dir, "SURFACEMODEL", "*", "LIS_HIST_*.nc")))
    out = dict(n_files=len(files), nonfinite=0, nonfinite_fill=0, range={k: 0 for k in RANGES}, E=[])
    for f in files:
        d = nc.Dataset(f)
        for k in list(RANGES) + list(FLUX):
            if k not in d.variables:
                continue
            a = np.ma.filled(d[k][:].astype(float), np.nan)
            m = np.ma.getmaskarray(d[k][:])
            out["nonfinite"] += int((~np.isfinite(a) & ~m).sum())      # NaN/Inf that is not the fill value
            if k in RANGES:
                lo, hi = RANGES[k]
                out["range"][k] += int(((a < lo) | (a > hi)).sum())
        if all(k in d.variables for k in FLUX):
            sw, lw, qle, qh, qg = (_get(d, k) for k in FLUX)
            out["E"].append(np.stack([sw + lw - qle - qh - qg, sw - lw - qle - qh - qg]))
        d.close()
    return out


def _p(a, q):
    a = a[np.isfinite(a)]
    return float(np.percentile(np.abs(a), q)) if a.size else float("nan")


def report(ref_dir, new_dir):
    A, B = check(ref_dir), check(new_dir)
    ok = True
    lines = [f"G5 sanity: reference {A['n_files']} files, new run {B['n_files']} files"]
    lines.append(f"NaN/Inf (not fill): reference {A['nonfinite']}, new {B['nonfinite']}")
    ok &= B["nonfinite"] <= A["nonfinite"]
    for k in RANGES:
        lines.append(f"range {k:16s} {RANGES[k]}: reference {A['range'][k]}, new {B['range'][k]} violations")
        ok &= B["range"][k] <= A["range"][k]
    if A["E"] and B["E"]:
        EA, EB = np.stack(A["E"]), np.stack(B["E"])      # (file, sign, y, x)
        s = int(np.argmin([_p(EA[:, i], 99) for i in (0, 1)]))     # sign chosen by the reference closure
        a, b = EA[:, s], EB[:, s]
        for q, nm in ((99, "p99"), (100, "max")):
            ra, rb = _p(a, q), _p(b, q)
            good = rb <= 1.1 * ra + 0.05
            ok &= good
            lines.append(f"energy residual {nm}: reference {ra:.4g} W/m2, new {rb:.4g} W/m2 {'ok' if good else 'WORSE'}")
    lines.append("G5 PASS" if ok else "G5 FAIL")
    return ok, "\n".join(lines)


if __name__ == "__main__":
    good, txt = report(sys.argv[1], sys.argv[2])
    print(txt)
    sys.exit(0 if good else 1)
