"""GRIB NLDAS-2 forcing reader (driver/forcing.py): synthetic NLDAS-style GRIB1 files (eccodes) read back through read_grib and through the NLDAS2 class,
compared with the same fields written as netCDF (the format the driver was validated with). Skipped when pygrib/eccodes are not installed."""
import os, datetime as dt
import numpy as np
import pytest

pytest.importorskip("pygrib"); eccodes = pytest.importorskip("eccodes")
import netCDF4 as nc
from lisf_noahmp_jax.driver import forcing as F

NY, NX = 12, 16
LAT0, LON0 = 25.0625, -124.9375
ORDER = [(11, 2), (51, 2), (1, 0), (33, 10), (34, 10), (205, 0), (153, 0), (157, 180), (228, 0), (61, 0), (204, 0)]   # message order of the NLDAS FORA files


def fields(seed):
    r = np.random.default_rng(seed)
    out = {}
    for p, _ in ORDER:
        n = F.GRIB_PARAM[p]
        out[n] = (r.uniform(1, 300, (NY, NX))).astype(np.float32)
    out["Tair"][0, 0] = np.float32(-9999.0)          # an undefined cell
    return out


def write_grib(path, flds, south_to_north=True):
    with open(path, "wb") as f:
        for p, lev in ORDER:
            h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib1")
            eccodes.codes_set(h, "Ni", NX); eccodes.codes_set(h, "Nj", NY)
            eccodes.codes_set(h, "latitudeOfFirstGridPointInDegrees", LAT0 if south_to_north else LAT0 + 0.125 * (NY - 1))
            eccodes.codes_set(h, "latitudeOfLastGridPointInDegrees", LAT0 + 0.125 * (NY - 1) if south_to_north else LAT0)
            eccodes.codes_set(h, "longitudeOfFirstGridPointInDegrees", LON0)
            eccodes.codes_set(h, "longitudeOfLastGridPointInDegrees", LON0 + 0.125 * (NX - 1))
            eccodes.codes_set(h, "iDirectionIncrementInDegrees", 0.125); eccodes.codes_set(h, "jDirectionIncrementInDegrees", 0.125)
            eccodes.codes_set(h, "jScansPositively", 1 if south_to_north else 0)
            eccodes.codes_set(h, "indicatorOfParameter", p); eccodes.codes_set(h, "level", lev)
            v = flds[F.GRIB_PARAM[p]].astype(np.float64)
            v = v if south_to_north else v[::-1]
            eccodes.codes_set(h, "bitmapPresent", 0)
            eccodes.codes_set(h, "bitsPerValue", 24)
            eccodes.codes_set_values(h, np.where(v == -9999.0, 9999.0, v).ravel())
            eccodes.codes_write(h, f); eccodes.codes_release(h)


def write_nc(path, flds):
    with nc.Dataset(path, "w") as d:
        d.createDimension("lon", NX); d.createDimension("lat", NY); d.createDimension("time", 1)
        d.createVariable("lon", "f4", ("lon",))[:] = (LON0 + 0.125 * np.arange(NX)).astype(np.float32)
        d.createVariable("lat", "f4", ("lat",))[:] = (LAT0 + 0.125 * np.arange(NY)).astype(np.float32)
        for n, v in flds.items():
            d.createVariable(n, "f4", ("time", "lat", "lon"))[0] = v


@pytest.mark.parametrize("south_to_north", [True, False])
def test_read_grib_matches_fields(tmp_path, south_to_north):
    fl = fields(1)
    p = str(tmp_path / "a.grb"); write_grib(p, fl, south_to_north)
    got, la, lo = F.read_grib(p)
    assert np.all(np.diff(la) > 0) and la[0] == np.float32(LAT0) and lo[0] == np.float32(LON0) and np.all(np.diff(lo) == 0.125)
    for n, v in fl.items():
        # the GRIB1 sample packs with a decimal scale of 3 digits: error up to 5e-4
        ok = v != -9999.0
        assert np.allclose(got[n][ok], v[ok], rtol=0, atol=1e-3), n
        assert np.all(got[n][~ok] == -9999.0), n


def test_class_grib_equals_netcdf(tmp_path):
    t = dt.datetime(2008, 10, 1, 3)
    doy = t.timetuple().tm_yday
    fl = fields(2)
    stem = f"NLDAS_FORA0125_H.A{t:%Y%m%d}.{t:%H}00"
    for root, ext in (("g", "002.grb"), ("n", "020.nc")):
        os.makedirs(tmp_path / root / str(t.year) / f"{doy:03d}")
    # use the values the GRIB decodes to, so that both formats hold the same numbers
    write_grib(str(tmp_path / "g" / str(t.year) / f"{doy:03d}" / (stem + ".002.grb")), fl)
    dec, _, _ = F.read_grib(str(tmp_path / "g" / str(t.year) / f"{doy:03d}" / (stem + ".002.grb")))
    write_nc(str(tmp_path / "n" / str(t.year) / f"{doy:03d}" / (stem + ".020.nc")), dec)
    lats = np.array([LAT0, LAT0 + 0.125 * 3, LAT0 + 0.125 * 11, LAT0 + 0.125 * 3 + 0.04]); lons = np.array([LON0, LON0 + 0.125 * 5, LON0 + 0.125 * 15, LON0 + 0.125 * 5 - 0.04])
    os.environ["LISJAX_FORCING_PROCS"] = "1"
    Fg = F.NLDAS2(str(tmp_path / "g"), lats, lons, fmt="grib", t_ref=t)
    Fn = F.NLDAS2(str(tmp_path / "n"), lats, lons, fmt="netcdf", t_ref=t)
    assert Fg._path(t).endswith(".002.grb") and Fn._path(t).endswith(".020.nc")
    a, b = F._read_hour(Fg._job(t)), F._read_hour(Fn._job(t))
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_forcing_source_keys():
    c = {"NLDAS2 forcing directory": "/g", "NLDAS-2.0 FORA forcing directory": "/n"}.get
    assert F.forcing_source(c, '"NLDAS2 grib"') == ("grib", "/g")
    assert F.forcing_source(c, '"NLDAS2 netcdf"') == ("netcdf", "/n")
