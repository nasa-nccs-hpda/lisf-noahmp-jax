"""End-to-end check of examples/sgp_5d: run the CLI on the shipped case and compare the history files with the real Fortran LIS 7.8 output shipped beside it.

Limits fixed 2026-11-11, before the test existed in CI, by the rule used for the other regression limits: 10 x the p99 error measured on the first run
(rounded up to one digit); the forcing fields must be identical. Needs no /panfs paths. About 1.5 min on CPU."""
import glob, os, shutil, subprocess, sys
import numpy as np
import netCDF4 as nc

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(HERE, "..", "examples", "sgp_5d")
LIMITS = dict(Swnet_inst=2e-3, Lwnet_inst=3.0, Qle_inst=0.2, Qh_inst=20.0, Qg_inst=20.0, Evap_inst=5e-8, SWE_inst=4e-5, SnowDepth_inst=6e-7,
              SoilMoist_inst=2e-5, SoilTemp_inst=0.2, SmLiqFrac_inst=2e-3, CanopInt_inst=2e-5, Snowcover_inst=8e-6,
              Tair_f_inst=0.0, Qair_f_inst=0.0, Psurf_f_inst=0.0, SWdown_f_inst=0.0, LWdown_f_inst=0.0)


def test_example_sgp_5d(tmp_path):
    case = tmp_path / "sgp_5d"
    shutil.copytree(EX, case, ignore=shutil.ignore_patterns("out", "reference"))
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    r = subprocess.run([sys.executable, "-c", "from lisf_noahmp_jax.cli import main; main()", "-C", str(case), str(case / "lis.config")],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    ref = sorted(glob.glob(os.path.join(EX, "reference", "SURFACEMODEL", "*", "LIS_HIST_*.nc")))
    assert len(ref) == 20
    errs = {k: [] for k in LIMITS}
    for f in ref:
        g = os.path.join(case, "out", f.split("reference" + os.sep)[1])
        assert os.path.exists(g), f"missing {g}"
        A, B = nc.Dataset(f), nc.Dataset(g)
        for k in LIMITS:
            x = np.ma.filled(A[k][:].astype(float), np.nan); y = np.ma.filled(B[k][:].astype(float), np.nan)
            assert not (np.isnan(y) & ~np.isnan(x)).any(), f"{k}: NaN only in the new run"
            e = np.abs(x - y).ravel(); errs[k].append(e[np.isfinite(e)])
    for k, lim in LIMITS.items():
        e = np.concatenate(errs[k])
        assert e.size > 0, k
        assert np.percentile(e, 99) <= lim, f"{k}: p99 error {np.percentile(e, 99):.3g} above {lim:g}"

    from lisf_noahmp_jax.driver import sanity     # G5: NaN/Inf, range and energy residual no worse than the Fortran reference
    ok, txt = sanity.report(os.path.join(EX, "reference"), str(case / "out"))
    assert ok, txt
