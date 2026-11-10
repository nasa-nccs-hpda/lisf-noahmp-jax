"""NLDAS-2 hourly netCDF forcing as processed by LIS for a lat/lon domain (neighbor spatial interpolation,
linear time interpolation of state variables, zenith-angle interpolation of SWdown, previous-hour precip).
Mirrors lis/metforcing/nldas2/{get,read_nldas2a,timeinterp}_nldas2.F90 and lis/interp/zterp.F90."""
import datetime as dt
import os
import numpy as np
import netCDF4 as nc
from . import timeutil as tu

f32 = np.float32
PI = f32(3.141592)


def _coszenith(lon, lat, lhour, zone, julian):
    julian = np.asarray(julian)
    gamma = f32(2) * PI * (julian - 1).astype(f32) / f32(365.0)
    dec = (f32(0.006918) - f32(0.399912) * np.cos(gamma) + f32(0.070257) * np.sin(gamma)
           - f32(0.006758) * np.cos(f32(2) * gamma) + f32(0.000907) * np.sin(f32(2) * gamma)
           - f32(0.002697) * np.cos(f32(3) * gamma) + f32(0.00148) * np.sin(f32(3) * gamma))
    et = (f32(0.000075) + f32(0.001868) * np.cos(gamma) - f32(0.032077) * np.sin(gamma)
          - f32(0.014615) * np.cos(f32(2) * gamma) - f32(0.04089) * np.sin(f32(2) * gamma)) * f32(229.18)
    ls = ((zone - 1) * 15 - 180).astype(f32)
    lcorr = f32(4.0) * (ls - lon) * f32(-1.0)
    latime = lhour + lcorr / f32(60.0) + et / f32(60.0)
    latime = np.where(latime < 0, latime + f32(24), latime)
    latime = np.where(latime > 24, latime - f32(24), latime)
    omega = (latime - f32(12.0)) * f32(-15.0) * PI / f32(180.0)
    llat = lat * PI / f32(180.0)
    cz = np.sin(dec) * np.sin(llat) + np.cos(dec) * np.cos(llat) * np.cos(omega)
    return np.minimum(np.maximum(f32(0.0), cz), f32(1.0))


def _localtime(gmt, lon):
    zone = np.full(np.shape(lon), 25, dtype=np.int32)
    zone = np.where(zone == 25, 1, zone)      # fallthrough value (lon beyond range) -> 1
    found = np.zeros(np.shape(lon), bool)
    for i in range(1, 26):
        hit = (~found) & (lon < f32(-187.5 + 15 * i))
        zone = np.where(hit, 1 if i == 25 else i, zone)
        found |= hit
    change = (zone - 13).astype(f32)
    lhour = gmt + change
    lhour = np.where(lhour < 0, lhour + f32(24), lhour)
    lhour = np.where(lhour > 23, lhour - f32(24), lhour)
    return lhour.astype(f32), zone


def zterp1(lat, lon, btime, etime, mbtime, julianb):
    """zterp with iflag=1: returns (weight1, weight2, czbegdata, czenddata)."""
    juliane = julianb + (1 if etime <= btime else 0)
    julianmb = julianb + (1 if mbtime < btime else 0)
    lb, zb = _localtime(f32(btime), lon)
    le, ze = _localtime(f32(etime), lon)
    lm, zm = _localtime(f32(mbtime), lon)
    jb = np.where(lb > btime, julianb - 1, julianb)
    je = np.where(le > etime, juliane - 1, juliane)
    jm = np.where(lm > mbtime, julianmb - 1, julianmb)
    czb = _coszenith(lon, lat, lb, zb, jb)
    cze = _coszenith(lon, lat, le, ze, je)
    czm = _coszenith(lon, lat, lm, zm, jm)
    w1 = np.zeros_like(czm); w2 = np.zeros_like(czm)
    dbm = mbtime - btime if btime <= mbtime else 24 - btime + mbtime
    dem = etime - mbtime if etime >= mbtime else 24 - mbtime + etime
    deb = etime - btime if etime > btime else (24 if etime == btime else 24 - btime + etime)
    ok = czm > 0.01
    bothpos = (czb > 0.01) & (cze > 0.01)
    onlye = (czb <= 0.01) & (cze > 0.01)
    onlyb = (czb > 0.01) & (cze <= 0.01)
    weighte = f32(dbm / deb); weightb = f32(dem / deb)
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.where(ok & bothpos, czm / czb * weightb, w1)
        w2 = np.where(ok & bothpos, czm / cze * weighte, w2)
        w1 = np.where(ok & onlyb, czm / czb, w1)
        w2 = np.where(ok & onlye, czm / cze, w2)
    return w1.astype(f32), w2.astype(f32), czb, cze



# NLDAS-2 FORA GRIB1 (NLDAS_FORA0125_H.A<date>.<HH>00.002.grb): indicatorOfParameter -> file variable (the names of the netCDF files)
GRIB_PARAM = {11: "Tair", 51: "Qair", 1: "PSurf", 33: "Wind_E", 34: "Wind_N", 205: "LWdown", 153: "CRainf_frac", 157: "CAPE", 228: "PotEvap", 61: "Rainf",
              204: "SWdown"}


def read_grib(path):
    """One hourly NLDAS-2 FORA GRIB file -> ({variable: float32 [lat, lon] array, rows south to north, -9999 where undefined}, lat 1-D, lon 1-D)."""
    import pygrib
    out = {}
    la = lo = None
    g = pygrib.open(path)
    try:
        for m in g:
            name = GRIB_PARAM.get(m.indicatorOfParameter)
            if name is None:
                continue
            lats, lons = m.latlons()
            v = np.ma.filled(np.ma.masked_values(np.ma.asarray(m.values), 9999.0), 9999.0)
            v = np.where(v == 9999.0, f32(-9999.0), v).astype(f32)
            if lats[0, 0] > lats[-1, 0]:
                v = v[::-1]; lats = lats[::-1]
            out[name] = v
            # GRIB1 keeps lat/lon in millidegrees (25.063 for 25.0625): snap to the exact cell centres, which are odd multiples of 1/16 degree on the 1/8 degree NLDAS grid
            la, lo = (np.round(lats[:, 0] * 16) / 16).astype(f32), (np.round(lons[0] * 16) / 16).astype(f32)
    finally:
        g.close()
    miss = sorted(set(GRIB_PARAM.values()) - set(out))
    if miss:
        raise ValueError(f"{path}: GRIB messages for {miss} not found")
    return out, la, lo


def forcing_source(cfg_get, src):
    """(format, forcing directory) for a LIS 'Met forcing sources' value: "NLDAS2 grib" uses the key `NLDAS2 forcing directory` (metforcing/nldas2),
    "NLDAS2 netcdf" the key `NLDAS-2.0 FORA forcing directory` (metforcing/nldas-2.0)."""
    if "grib" in src.lower():
        return "grib", cfg_get("NLDAS2 forcing directory")
    return "netcdf", cfg_get("NLDAS-2.0 FORA forcing directory")


def _read_hour(job):
    """Worker: one NLDAS-2 hourly file -> per-tile float32 values (None if the file does not exist: look-ahead past the run end)."""
    path, vars_, j, i, alt = job[:5]
    fmt = job[5] if len(job) > 5 else "netcdf"
    if not os.path.exists(path):
        return None
    if fmt == "grib":
        fld = read_grib(path)[0]
        out = {v: fld[v][j, i].astype(f32) for v in vars_}
        if alt is not None:
            for v in ("Wind_E", "Wind_N", "Rainf"):
                out[v + "_alt"] = fld[v][alt[0], alt[1]].astype(f32)
        return out
    with nc.Dataset(path) as d:
        out = {v: np.array(d[v][0])[j, i].astype(f32) for v in vars_}
        if alt is not None:
            for v in ("Wind_E", "Wind_N", "Rainf"):
                out[v + "_alt"] = np.array(d[v][0])[alt[0], alt[1]].astype(f32)
    return out


class NLDAS2:
    LOOKAHEAD = 24
    VARS = ("Tair", "Qair", "PSurf", "Wind_E", "Wind_N", "LWdown", "SWdown", "Rainf")

    def __init__(self, root, lats, lons, extra=(), alt=None, fmt="netcdf", t_ref=None):
        """lats/lons: 1-D arrays (per tile) of domain-cell centers. fmt: \"netcdf\" (v020 files) or \"grib\" (v002 GRIB, as read by metforcing/nldas2); t_ref: a time whose file exists (grid read from it; default 2017-01-01). Uses LIS 'neighbor' rule (half rounds up).
        extra: additional file variables to read (e.g. 'CRainf_frac').
        alt: optional (lats, lons) of the domain cell whose GRID index equals the TILE index of each tile. Noah 2.7.1's noah271_f2t
        reads Wind_E, Wind_N and the precipitation-validity test at index t instead of tile_id; the values are returned as
        Wind_E_alt / Wind_N_alt (-9999 where undefined) and Rainf_valid_alt."""
        self.root = root
        self.fmt = fmt
        self.vars = tuple(self.VARS) + tuple(extra)
        self.lats = np.asarray(lats, f32); self.lons = np.asarray(lons, f32)
        t0 = t_ref or dt.datetime(2017, 1, 1, 0)
        if fmt == "grib":
            _, la, lo = read_grib(self._path(t0))
        else:
            with nc.Dataset(self._path(t0)) as d:
                la = np.array(d["lat"][:]); lo = np.array(d["lon"][:])
        # LIS 'neighbor' interpolation: nearest source cell, nint() (halves round up)
        self.j = np.floor((self.lats.astype(np.float64) - la[0]) / 0.125 + 0.5 + 1e-7).astype(int)
        self.i = np.floor((self.lons.astype(np.float64) - lo[0]) / 0.125 + 0.5 + 1e-7).astype(int)
        self.alt = alt
        if alt is not None:
            alat = np.asarray(alt[0], f32); alon = np.asarray(alt[1], f32)
            self.aj = np.floor((alat.astype(np.float64) - la[0]) / 0.125 + 0.5 + 1e-7).astype(int)
            self.ai = np.floor((alon.astype(np.float64) - lo[0]) / 0.125 + 0.5 + 1e-7).astype(int)
        self.cache = {}
        self._ex = None
        self.nproc = int(os.environ.get("LISJAX_FORCING_PROCS", min(8, len(os.sched_getaffinity(0)))))

    def _path(self, t):
        ext = "002.grb" if self.fmt == "grib" else "020.nc"
        return f"{self.root}/{t.year}/{tu.doy(t):03d}/NLDAS_FORA0125_H.A{t:%Y%m%d}.{t:%H}00.{ext}"

    def hourly(self, t):
        """Hourly file values at the tiles. A cache miss reads the next LOOKAHEAD hours in parallel (worker processes: netCDF-C is
        not thread-safe), so the sequential time loop mostly hits the cache. Values are identical to a serial read."""
        if t not in self.cache:
            want = [t + dt.timedelta(hours=k) for k in range(self.LOOKAHEAD) if t + dt.timedelta(hours=k) not in self.cache]
            pool = self._pool()
            if pool is None:
                got = [(w, _read_hour(self._job(w))) for w in want[:1]]
            else:
                got = list(zip(want, pool.map(_read_hour, [self._job(w) for w in want])))
            while len(self.cache) + len(got) > 3 * self.LOOKAHEAD:
                self.cache.pop(next(iter(self.cache)))
            for w, r in got:
                if r is not None:
                    self.cache[w] = r
            if t not in self.cache:
                raise FileNotFoundError(self._path(t))
        return self.cache[t]

    def _job(self, t):
        return (self._path(t), self.vars, self.j, self.i, None if self.alt is None else (self.aj, self.ai), self.fmt)

    def _pool(self):
        if self.nproc <= 1:
            return None
        if self._ex is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self._ex = ProcessPoolExecutor(self.nproc, mp_context=mp.get_context("spawn"))
        return self._ex

    def close(self):
        if self._ex is not None:
            self._ex.shutdown()
            self._ex = None

    def at(self, now):
        """Forcing for all tiles at LIS time `now` (a datetime at the end of the model step)."""
        t1 = now.replace(minute=0, second=0, microsecond=0)
        t2 = t1 + dt.timedelta(hours=1)
        a, b = self.hourly(t1), self.hourly(t2)
        span = (t2 - t1).total_seconds()
        wt1 = f32((t2 - now).total_seconds() / span); wt2 = f32(1.0) - wt1
        out = {k: a[k] * wt1 + b[k] * wt2 for k in ("Tair", "Qair", "PSurf", "Wind_E", "Wind_N", "LWdown")}
        if os.environ.get("LISJAX_ULP_TAIR"):                 # chaos test: air temperature moved by one float32 ulp (default off)
            out["Tair"] = np.nextafter(out["Tair"].astype(f32), f32(np.inf))
        gmt1, gmt2 = tu.gmt_hours(t1), tu.gmt_hours(t2)
        w1, w2, czb, cze = zterp1(self.lats, self.lons, gmt1, gmt2, tu.gmt_hours(now), tu.doy(now))
        sw = a["SWdown"] * w1 + b["SWdown"] * w2
        alt = (sw > a["SWdown"]) & (sw > b["SWdown"]) & ((czb < 0.1) | (cze < 0.1))
        sw = np.where(alt, a["SWdown"] * wt1 + b["SWdown"] * wt2, sw)
        out["SWdown"] = sw.astype(f32)
        out["Rainf"] = (b["Rainf"] / f32(3600.0)).astype(f32)
        if self.alt is not None:
            for v in ("Wind_E", "Wind_N"):
                ok = (a[v + "_alt"] != f32(-9999.0)) & (b[v + "_alt"] != f32(-9999.0))
                out[v + "_alt"] = np.where(ok, a[v + "_alt"] * wt1 + b[v + "_alt"] * wt2, f32(-9999.0)).astype(f32)
            out["Rainf_valid_alt"] = (b["Rainf_alt"] != f32(-9999.0))
        if "CRainf_frac" in b:
            out["CRainf_frac"] = b["CRainf_frac"]
        return out


class MERRA2:
    """MERRA-2 hourly (time-averaged, centered at hh:30) forcing as processed by LIS (lis/metforcing/merra2):
    T2M/QV2M/PS linear in time, SWGDN by zenith-angle interpolation between the bracketing hours, winds, LW and
    precipitation taken from the hour that started at or before `now`. Config options of the walkthrough-style
    run: lowest-model-level off, 2m wind off, uncorrected PRECTOT, no scaling/sampling."""

    def __init__(self, root, lats, lons):
        self.root = root
        self.lats = np.asarray(lats, f32); self.lons = np.asarray(lons, f32)
        self.j = np.rint((self.lats.astype(np.float64) + 90.0) / 0.5).astype(int)
        self.i = np.rint((self.lons.astype(np.float64) + 180.0) / 0.625).astype(int) % 576
        self.cache = {}

    def _prefix(self, y):
        return "MERRA2_100" if y <= 1991 else "MERRA2_200" if y <= 2000 else "MERRA2_300" if y <= 2010 else "MERRA2_400"

    def _path(self, d, coll):
        p = self._prefix(d.year)
        return f"{self.root}/{p}/Y{d.year}/M{d.month:02d}/{p}.tavg1_2d_{coll}_Nx.{d:%Y%m%d}.nc4"

    def day(self, d):
        """24 hourly fields (time-averaged, centers hh:30) for the tiles: dict name -> (24, ntile) float32."""
        if d not in self.cache:
            if len(self.cache) > 3:
                self.cache.pop(next(iter(self.cache)))
            out = {}
            for coll, names in (("slv", ("T2M", "QV2M", "U10M", "V10M", "PS")), ("flx", ("PRECTOT",)),
                                ("rad", ("SWGDN", "LWGAB"))):
                ds = nc.Dataset(self._path(d, coll))
                for n in names:
                    out[n] = np.array(ds[n][:, :, :])[:, self.j, self.i].astype(f32)
            self.cache[d] = out
        return self.cache[d]

    def hour_value(self, center):
        """Fields of the hour whose center is `center` (a datetime at hh:30)."""
        d = center.replace(hour=0, minute=0, second=0, microsecond=0)
        day = self.day(d)
        h = center.hour
        return {k: v[h] for k, v in day.items()}

    def at(self, now):
        t1 = now.replace(minute=30, second=0, microsecond=0)
        if t1 > now:
            t1 -= dt.timedelta(hours=1)
        t2 = t1 + dt.timedelta(hours=1)
        a, b = self.hour_value(t1), self.hour_value(t2)
        wt1 = f32((t2 - now).total_seconds() / 3600.0); wt2 = f32(1.0) - wt1
        gmt1, gmt2 = tu.gmt_hours(t1), tu.gmt_hours(t2)
        w1, w2, czb, cze = zterp1(self.lats, self.lons, gmt1, gmt2, tu.gmt_hours(now), tu.doy(now))
        sw = a["SWGDN"] * w1 + b["SWGDN"] * w2
        sw = np.where(sw > 1367.0, b["SWGDN"], sw)
        sw = np.where((sw < 0) & (sw > -0.00001), f32(0.0), sw)
        out = {"Tair": a["T2M"] * wt1 + b["T2M"] * wt2, "Qair": a["QV2M"] * wt1 + b["QV2M"] * wt2,
               "PSurf": a["PS"] * wt1 + b["PS"] * wt2, "Wind_E": a["U10M"], "Wind_N": a["V10M"],
               "LWdown": a["LWGAB"], "SWdown": sw.astype(f32), "Rainf": np.maximum(a["PRECTOT"], f32(0.0))}
        return out
