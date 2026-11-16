"""LDT parameter-processing port (src/lisf_noahmp_jax/driver/ldt.py) against the parameter file written by the real LDT (GNU 14 build).
Needs the raw LDT inputs and the reference file under $LISF_RUN (not in the repository, see DATA_AND_PATHS.md); skipped otherwise."""
import os, sys
import numpy as np, pytest
import netCDF4 as nc

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get("LISF_RUN", "")
CASES = [("ldt_mp", "lis_input.sgp.noahmp401.nc"), ("conus_mp", "lis_input.conus.noahmp401.nc")]


@pytest.mark.parametrize("case,ref", CASES, ids=[c for c, _ in CASES])
def test_ldt(case, ref, tmp_path):
    cfg = f"{RUN}/{case}/ldt.config"
    if not (os.path.exists(cfg) and os.path.exists(f"{RUN}/{case}/out/{ref}")):
        pytest.skip("LDT inputs/reference not available")
    from lisf_noahmp_jax.driver import ldt
    out = str(tmp_path / "py.nc")
    ldt.write(ldt.process(cfg), out)
    a, b = nc.Dataset(f"{RUN}/{case}/out/{ref}"), nc.Dataset(out)
    for v in a.variables:
        if v == "time":
            continue
        x = np.ma.filled(a[v][:].astype(np.float64), np.nan); y = np.ma.filled(b[v][:].astype(np.float64), np.nan)
        assert np.array_equal(np.isnan(x), np.isnan(y)), v
        ok = np.isfinite(x)
        rel = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(x[ok]), 1e-12)
        # every field is bitwise identical except SLOPE and ASPECT (libm atan/cos differences, <= 1e-6 relative)
        assert rel.max(initial=0) <= (2e-6 if v in ("SLOPE", "ASPECT") else 0.0), (v, rel.max())
