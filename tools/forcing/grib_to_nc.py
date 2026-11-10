"""NLDAS-2 FORA hourly GRIB (NLDAS_FORA0125_H.A<date>.<HH>00.002.grb) -> hourly NetCDF4 that the JAX driver reads (NLDAS_FORA0125_H.A<date>.<HH>00.020.nc).
usage: python grib_to_nc.py GRIB_ROOT OUT_ROOT START END [NPROC]    START/END as YYYYMMDD (inclusive days); GRIB_ROOT/<yr>/<doy>/ -> OUT_ROOT/<yr>/<doy>/
Needs pygrib and netCDF4. Existing outputs are skipped (restartable). Missing values are -9999, rows run south to north, as in the v020 files.
Note: the ".020" in the output name only matches the driver's file pattern; the data come from the GRIB (.002) product."""
import sys, os, datetime as dt
import numpy as np
import pygrib
import netCDF4 as nc
from multiprocessing import Pool

# GRIB1 indicatorOfParameter -> NetCDF variable, units (message order in the file: 11 51 1 33 34 205 153 157 228 61 204)
PARAM = {11: ("Tair", "K"), 51: ("Qair", "kg kg-1"), 1: ("PSurf", "Pa"), 33: ("Wind_E", "m s-1"), 34: ("Wind_N", "m s-1"), 205: ("LWdown", "W m-2"),
         153: ("CRainf_frac", "fraction"), 157: ("CAPE", "J kg-1"), 228: ("PotEvap", "kg m-2"), 61: ("Rainf", "kg m-2"), 204: ("SWdown", "W m-2")}
MISS = np.float32(-9999.0)


def convert(job):
    src, dst, t = job
    if os.path.exists(dst):
        return 0
    if not os.path.exists(src):
        return -1
    g = pygrib.open(src)
    data = {}
    for m in g:
        p = m.indicatorOfParameter
        if p not in PARAM:
            continue
        lats, lons = m.latlons()
        v = np.ma.filled(np.ma.masked_values(np.ma.asarray(m.values), 9999.0), 9999.0)
        v = np.where(v == 9999.0, MISS, v).astype(np.float32)
        if lats[0, 0] > lats[-1, 0]:
            v = v[::-1]; lats = lats[::-1]
        data[PARAM[p][0]] = v
    g.close()
    if len(data) != len(PARAM):
        raise RuntimeError(f"{src}: found {sorted(data)}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with nc.Dataset(tmp, "w", format="NETCDF4") as d:
        ny, nx = lats.shape
        d.createDimension("lon", nx); d.createDimension("lat", ny); d.createDimension("time", 1); d.createDimension("bnds", 2)
        x = d.createVariable("lon", "f4", ("lon",), fill_value=MISS); x[:] = lons[0].astype(np.float32); x.units = "degrees_east"
        y = d.createVariable("lat", "f4", ("lat",), fill_value=MISS); y[:] = lats[:, 0].astype(np.float32); y.units = "degrees_north"
        tv = d.createVariable("time", "f8", ("time",)); tv.units = "hours since 1979-01-01 00:00:00"
        tv[:] = (t - dt.datetime(1979, 1, 1)).total_seconds() / 3600.0
        for p, (name, units) in PARAM.items():
            var = d.createVariable(name, "f4", ("time", "lat", "lon"), fill_value=MISS)
            var.units = units; var.missing_value = MISS
            var[0] = data[name]
        d.history = "converted from NLDAS-2 FORA GRIB by tools/forcing/grib_to_nc.py"
    os.replace(tmp, dst)
    return 1


if __name__ == "__main__":
    groot, oroot, a, b = sys.argv[1:5]
    npr = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    t = dt.datetime.strptime(a, "%Y%m%d"); end = dt.datetime.strptime(b, "%Y%m%d") + dt.timedelta(hours=23)
    jobs = []
    while t <= end:
        doy = t.timetuple().tm_yday
        stem = f"NLDAS_FORA0125_H.A{t:%Y%m%d}.{t:%H}00"
        jobs.append((f"{groot}/{t.year}/{doy:03d}/{stem}.002.grb", f"{oroot}/{t.year}/{doy:03d}/{stem}.020.nc", t))
        t += dt.timedelta(hours=1)
    with Pool(npr) as p:
        r = p.map(convert, jobs, chunksize=4)
    print("converted", r.count(1), "existing", r.count(0), "missing", r.count(-1), "of", len(r))
