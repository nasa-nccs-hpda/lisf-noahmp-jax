"""End-to-end LIS-style Noah-MP 4.0.1 retrospective run in Python/JAX: config + LDT parameter file + NLDAS-2 forcing ->
time-averaged / accumulated / instantaneous history files with the variables of the LIS Noah-MP output table."""
import sys, os, time, datetime as dt
import numpy as np
import jax, jax.numpy as jnp
import jax.experimental
import netCDF4 as nc
from lisf_noahmp_jax.driver.params_mp import SetupMP
from lisf_noahmp_jax.driver.forcing import NLDAS2, MERRA2
from lisf_noahmp_jax.driver import model_mp, model_mp36, pert as pertmod

f32 = np.float32
# name -> (units, standard_name, long_name)   [as written by LIS for Noah-MP.4.0.1]
META = {
    "Qle": ("W m-2", "surface_upward_latent_heat_flux", "latent heat flux"),
    "Qh": ("W m-2", "surface_upward_sensible_heat_flux", "sensible heat flux"),
    "Qg": ("W m-2", "downward_heat_flux_in_soil", "soil heat flux"),
    "Evap": ("kg m-2 s-1", "total_evapotranspiration", "total evapotranspiration"),
    "Qs": ("kg m-2", "surface_runoff_amount", "surface runoff"),
    "Qsb": ("kg m-2", "subsurface_runoff_amount", "subsurface runoff amount"),
    "AvgSurfT": ("K", "surface_temperature", "surface temperature"),
    "Albedo": ("%", "surface_albedo", "surface albedo"),
    "SWE": ("kg m-2", "liquid_water_content_of_surface_snow", "snow water equivalent"),
    "SnowDepth": ("m", "snow_depth", "snow depth"),
    "SoilMoist": ("m^3 m-3", "soil_moisture_content", "soil moisture content"),
    "SoilTemp": ("K", "soil_temperature", "soil temperature"),
    "SmLiqFrac": ("m^3 m-3", "liquid_fraction_of_soil_moisture", "average layer fraction of liquid moisture"),
    "CanopInt": ("kg m-2", "total_canopy_water_storage", "total canopy water storage"),
    "Snowcover": ("%", "surface_snow_area_fraction", "snow cover"),
    "Wind_f": ("m s-1", "wind_speed", "wind speed"),
    "Tair_f": ("K", "air_temperature", "air temperature"),
    "Qair_f": ("kg kg-1", "specific_humidity", "specific humidity"),
    "Psurf_f": ("Pa", "surface_air_pressure", "surface pressure"),
    "SWdown_f": ("W m-2", "surface_downwelling_shortwave_flux_in_air", "surface downward shortwave radiation"),
    "LWdown_f": ("W m-2", "surface_downwelling_longwave_flux_in_air", "surface downward longwave radiation"),
    "RelSMC": ("-", "relative_soil_moisture", "relative soil moisture"),
    "RHMin": ("%", "min_relative_humidity", "minimum relative humidity"),
    "TotalPrecip": ("kg m-2", "total_precipitation_amount", "total precipitation amount"),
    "Landmask": ("-", "landmask", "landmask"), "Landcover": ("-", "landcover", "landcover"),
    "Soiltype": ("-", "soiltype", "soiltype"), "Elevation": ("m", "elevation", "elevation"),
    "Greenness": ("-", "green_vegetation_fraction", "green vegetation fraction"),
}
TAVG = "Qle Qh Qg Evap AvgSurfT Albedo SoilMoist SoilTemp Wind_f Tair_f Qair_f Psurf_f SWdown_f LWdown_f".split()
INST = ("AvgSurfT SWE SnowDepth SoilMoist SoilTemp SmLiqFrac CanopInt Snowcover Wind_f Tair_f Qair_f Psurf_f "
        "SWdown_f LWdown_f Landmask Landcover Soiltype Elevation Greenness RelSMC RHMin").split()
ACC = "Qs Qsb TotalPrecip".split()
LAYERED = ("SoilMoist", "SoilTemp", "SmLiqFrac", "RelSMC")
UDEF = f32(-9999.0)


ROUTING_VARS = [  # (file name, source key, units, standard_name, long_name, tavg)
    ("Streamflow_tavg", "rivout", "m3 s-1", "streamflow", "streamflow"),
    ("RiverStor_inst", "rivsto", "m3", "River_Water_Storage", "River Water Storage"),
    ("RiverDepth_inst", "rivdph", "m", "River_Depth", "River Depth"),
    ("RiverFlowVelocity_inst", "rivvel", "m s-1", "River_Flow_Velocity", "River Flow Velocity"),
    ("FloodQ_inst", "fldout", "m3 s-1", "Floodplain_Water_Discharge", "Floodplain Water Discharge"),
    ("FloodStor_inst", "fldsto", "m3", "Floodplain_Water_Storage", "Floodplain Water Storage"),
    ("FloodDepth_inst", "flddph", "m", "Floodplain_Depth", "Floodplain Depth"),
    ("FloodVelocity_inst", "fldvel", "m s-1", "Floodplain Flow Velocity", "Floodplain_Flow_Velocity"),
    ("FloodedFrac_inst", "fldfrc", "-", "Flooded Fraction", "Flooded_Fraction"),
    ("FloodedArea_inst", "fldare", "m2", "Flooded Area", "Flooded_Area"),
    ("SurfElev_inst", "sfcelv", "m", "Surface Water Elevation", "Surface_Water_Elevation"),
    ("RunoffStor_inst", "roffsto", "mm", "Runoff Reservoir Storage", "Runoff_Reservoir_Storage"),
    ("BaseflowStor_inst", "basfsto", "mm", "Baseflow Reservoir Storage", "Baseflow_Reservoir_Storage"),
    ("SWS_inst", "surfws", "mm", "Surface Water Storage", "Surface_Water_Storage"),
]


def write_routing_hist(path, S, HP, label, vals):
    """LIS_HIST file of the HyMAP2 router (ROUTING/): vals maps source key -> (nseq,) array."""
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("east_west", S.nc); ds.createDimension("north_south", S.nr); ds.createDimension("time", 1)
    ds.missing_value = f32(-9999.0); ds.title = "LIS routing output (Python/JAX port)"
    ds.MAP_PROJECTION = "EQUIDISTANT CYLINDRICAL"; ds.DX = f32(0.125); ds.DY = f32(0.125)
    ds.SOUTH_WEST_CORNER_LAT = f32(S.glat1d.min()); ds.SOUTH_WEST_CORNER_LON = f32(S.glon1d.min())
    la = ds.createVariable("lat", "f4", ("north_south",)); lo = ds.createVariable("lon", "f4", ("east_west",))
    tv = ds.createVariable("time", "f4", ("time",)); tv.units = f"minutes since {label:%Y-%m-%d %H:%M:%S}"; tv[:] = 0.0
    la[:] = S.glat1d; lo[:] = S.glon1d
    for name, key, u, sn, ln in ROUTING_VARS:
        x = ds.createVariable(name, "f4", ("north_south", "east_west"), fill_value=f32(-9999.0))
        x.units, x.standard_name, x.long_name = u, sn, ln
        g = np.full((S.nr, S.nc), -9999.0, f32); g[HP.rows, HP.cols] = vals[key]
        x[:] = g
    ds.close()


def write_hist(path, S, label, tavg, acc, inst, tmin, tmax, static):
    nr, ncol = S.nr, S.nc
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("east_west", ncol); ds.createDimension("north_south", nr); ds.createDimension("time", 1)
    for v in LAYERED:
        ds.createDimension(f"{v}_profiles", 4)
    ds.missing_value = f32(-9999.0); ds.NUM_SOIL_LAYERS = np.int32(4)
    ds.SOIL_LAYER_THICKNESSES = (S.sldpth * 100).astype(f32)
    ds.title = "LIS land surface model output (Python/JAX port)"
    ds.MAP_PROJECTION = "EQUIDISTANT CYLINDRICAL"; ds.DX = f32(0.25); ds.DY = f32(0.25)
    ds.SOUTH_WEST_CORNER_LAT = f32(S.glat1d.min()); ds.SOUTH_WEST_CORNER_LON = f32(S.glon1d.min())
    la = ds.createVariable("lat", "f4", ("north_south",)); lo = ds.createVariable("lon", "f4", ("east_west",))
    tv = ds.createVariable("time", "f4", ("time",)); tv.units = f"minutes since {label:%Y-%m-%d %H:%M:%S}"; tv[:] = 0.0
    la[:] = S.glat1d; lo[:] = S.glon1d

    def grid(a):
        g = np.full((nr, ncol), -9999.0, f32); g[S.rows, S.cols] = a; return g

    def put(name, arr, layered, base):
        u, sn, ln = META[base]
        dims = (f"{base}_profiles", "north_south", "east_west") if layered else ("north_south", "east_west")
        x = ds.createVariable(name, "f4", dims, fill_value=f32(-9999.0))
        x.units, x.standard_name, x.long_name = u, sn, ln
        x[:] = np.stack([grid(a) for a in arr]) if layered else grid(arr)
    for v in TAVG:
        if v in LAYERED:
            put(v + "_tavg", [tavg[f"{v}{k}"] for k in range(1, 5)], True, v)
        else:
            put(v + "_tavg", tavg[v], False, v)
    for v in ACC:
        put(v + "_acc", acc[v], False, v)
    for v in INST:
        if v in LAYERED:
            put(v + "_inst", [inst[f"{v}{k}"] for k in range(1, 5)], True, v)
        else:
            put(v + "_inst", inst[v] if v in inst else static[v], False, v)
    put("Tair_f_min", tmin, False, "Tair_f"); put("Tair_f_max", tmax, False, "Tair_f")
    ds.close()


def month_d_new_index(now):
    """noahmp_driver_50 month_d_new: months (0-based) and weight for the 12 monthly values taken as valid on the 15th (leap February)."""
    import calendar
    nd = lambda m, y=now.year: 29 if (m == 2 and calendar.isleap(y)) else [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    m, d = now.month, now.day
    if d == 15:
        return m - 1, m - 1, 0.0
    if d < 15:
        pm = 12 if m == 1 else m - 1
        return pm - 1, m - 1, float(np.float32(nd(pm) - 15 + d) / np.float32(nd(pm)))
    qm = 1 if m == 12 else m + 1
    return m - 1, qm - 1, float(np.float32(d - 15) / np.float32(nd(m)))


class LaiTracker:
    """Input LAI of Noah-MP dveg 7, 8, 9 as LIS supplies it (LIS_read_lai, 'LAI data source: LDT', monthly climatology): the two monthly maps are
    re-read when the mid-month alarm rings (first step of the 16th, and at the start), so they always are the pair of the interpolation weights
    (mid-month rule of LIS_computeTemporalWeights); LAI = wt1*map1 + wt2*map2 in double precision, then cast to single."""
    JULM = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)

    def __init__(self, lai_mon, dt_s):
        self.L = np.asarray(lai_mon, np.float64)           # (12, n)

    @classmethod
    def weights(cls, now):
        """(mo1, mo2, wt1, wt2) with 1-based months, as LIS_computeTemporalWeights('monthly')"""
        mo, da = now.month, now.day
        mo2 = mo + (1 if da > 15 else 0)
        if mo2 == 1:
            mo2 = 13
        mo1 = mo2 - 1
        juld = cls.JULM[mo - 1] + da
        if mo == 3 and now.year % 4 == 0:
            juld += 1
        if juld <= 15:
            juld += 365
        day2 = float(cls.JULM[mo2 - 1] + 15)
        if mo2 == 3 and now.year % 4 == 0:
            day2 += 1
        day1 = float(cls.JULM[mo1 - 1] + 15)
        wt1 = np.float32((np.float32(day2) - np.float32(juld)) / np.float32(day2 - day1))
        wt2 = np.float32((np.float32(juld) - np.float32(day1)) / np.float32(day2 - day1))
        return (1 if mo1 == 13 else mo1), (1 if mo2 == 13 else mo2), wt1, wt2

    def tlai(self, now):
        """LAI of the model step that starts at `now`"""
        mo1, mo2, wt1, wt2 = self.weights(now)
        t = np.float64(wt1) * self.L[mo1 - 1] + np.float64(wt2) * self.L[mo2 - 1]
        return t.astype(f32)


def make_chunk(S, step):
    def chunk(par, stat, st, rawK, tmK):
        def body(carry, x):
            st, acc, cnt, last, tmin, tmax, rh = carry
            raw, tm = x
            st, diag = step(par, stat, st, raw, tm)
            acc = {k: acc[k] + diag[k] for k in acc}
            cnt = cnt + diag["Albedo_n"]
            # RHMin bookkeeping (NoahMP401_main): updated when tair is below the running minimum of the interval
            tair, qair, ps = raw["tair"], raw["qair"], raw["psurf"]
            lower = tair < rh["tmin"]
            es = 611.0 * jnp.exp(2.501E6 / 461.0 * (1. / 273.15 - 1. / tair))
            q2sat = 0.622 * es / (ps - (1. - 0.622) * es)
            rh = {"tmin": jnp.where(lower, tair, rh["tmin"]), "rhmin": jnp.where(lower, qair / q2sat, rh["rhmin"])}
            tmin = jnp.minimum(tmin, tair); tmax = jnp.maximum(tmax, tair)
            return (st, acc, cnt, diag, tmin, tmax, rh), (diag["RunsfRate"], diag["RunsbRate"])
        d0 = jax.eval_shape(lambda: step(par, stat, st, jax.tree_util.tree_map(lambda a: a[0], rawK),
                                         jax.tree_util.tree_map(lambda a: a[0], tmK))[1])
        acc0 = {k: jnp.zeros(a.shape, a.dtype) for k, a in d0.items()}
        rh0 = {"tmin": jnp.full_like(rawK["tair"][0], 999.0), "rhmin": st["RHMIN"]}
        init = (st, acc0, jnp.zeros_like(rawK["tair"][0]), {k: jnp.zeros(a.shape, a.dtype) for k, a in d0.items()},
                jnp.full_like(rawK["tair"][0], 999.0), jnp.full_like(rawK["tair"][0], -999.0), rh0)
        (st2, acc, cnt, last, tmin, tmax, rh), ys = jax.lax.scan(body, init, (rawK, tmK))
        st2 = dict(st2); st2["RHMIN"] = rh["rhmin"]
        return st2, acc, cnt, last, tmin, tmax, ys
    return jax.jit(chunk)


UNDEF = f32(-9999.0)
# (restart name, units, source) ; 2-D variables are (nlayer, ntiles)
RST_SCALARS = [("SFCRUNOFF", "m", "SFCRUNOFF"), ("UDRRUNOFF", "m", "UDRRUNOFF"), ("SNEQV", "mm", "SWE"), ("SNOWH", "m", "SNDPTH"),
               ("ACSNOM", "-", "ACSNOM"), ("ACSNOW", "mm", "ACSNOW"), ("ISNOW", "-", "ISNOW"), ("TV", "K", "TV"), ("TG", "K", "TG"),
               ("CANICE", "mm", "CANICE"), ("CANLIQ", "mm", "CANLIQ"), ("EAH", "Pa", "EAH"), ("TAH", "K", "TAH"), ("CM", "-", "CM"),
               ("CH", "-", "CH"), ("FWET", "-", "FWET"), ("SNEQVO", "mm h2o", "SNEQVO"), ("ALBOLD", "-", "ALBOLD"),
               ("QSNOW", "mm/s", "QSNOW"), ("WSLAKE", "mm", "WSLAKE"), ("ZWT", "m", "ZWT"), ("WA", "mm", "WA"), ("WT", "mm", "WT"),
               ("LFMASS", "g/m2", "LFMASS"), ("RTMASS", "g/m2", "RTMASS"), ("STMASS", "g/m2", "STMASS"), ("WOOD", "g/m2", "WOOD"),
               ("STBLCP", "g/m2", "STBLCP"), ("FASTCP", "g/m2", "FASTCP"), ("LAI", "-", "PLAI"), ("SAI", "-", "PSAI"),
               ("TAUSS", "-", "TAUSS"), ("SMCWTD", "-", "SMCWTD"), ("DEEPRECH", "-", "DEEPRECH"), ("RECH", "-", "RECH"),
               ("GRAIN", "g/m2", "GRAIN"), ("GDD", "-", "GDD")]
GLACIER_UNDEF = "TV CANICE CANLIQ EAH TAH FWET WA WT LFMASS RTMASS STMASS WOOD STBLCP FASTCP LAI SAI GRAIN GDD".split()


def write_restart(path, S, st, now):
    """LIS_RST_NOAHMP401-style restart (same variable names/layout as the Fortran writer)."""
    st = {k: np.asarray(v) for k, v in st.items()}
    ntl = len(st["SWE"])
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("ntiles", len(st["SWE"])); ds.createDimension("dim1", 7); ds.createDimension("dim2", 4)   # ensemble members are contiguous tiles
    ds.createDimension("dim3", 3); ds.createDimension("dim4", 1); ds.createDimension("time", 1)
    v50 = getattr(S, "model", "") == "Noah-MP.5.0"
    ds.missing_value = f32(-9999.0); ds.title = "LIS land surface model restart (Python/JAX port)"; ds.source = "NoahMP50" if v50 else "NOAHMP401"
    g = S.glacier_idx
    DROP50 = ("SMCWTD", "DEEPRECH", "RECH") if v50 else ()

    def put(name, a, dims, units):
        x = ds.createVariable(name, "f4", dims); x.units = units; x[:] = np.asarray(a, f32)
    canwat = (st["CANLIQ"] + st["CANICE"]).astype(f32)
    canwat[g] = UNDEF
    for name, units, key in RST_SCALARS[:2]:
        put(name, st[key], ("ntiles",), units)
    put("SMC", st["SMC"].T, ("dim2", "ntiles"), "m3/m3"); put("SH2O", st["SMH2O"].T, ("dim2", "ntiles"), "m3/m3")
    put("TSLB", st["STC"][:, 3:].T, ("dim2", "ntiles"), "K")
    put("SNEQV", st["SWE"], ("ntiles",), "mm"); put("SNOWH", st["SNDPTH"], ("ntiles",), "m"); put("CANWAT", canwat, ("ntiles",), "mm")
    for name, units, key in RST_SCALARS[4:]:
        if name in DROP50:
            continue
        a = np.asarray(st[key], f32).copy()
        if name in GLACIER_UNDEF:
            a[g] = UNDEF
        put(name, a, ("ntiles",), units)
        if name == "WT":
            put("TSNO", st["STC"][:, :3].T, ("dim3", "ntiles"), "K"); put("ZSS", st["ZSNSO"].T, ("dim1", "ntiles"), "m")
            put("SNOWICE", st["SNICE"].T, ("dim3", "ntiles"), "mm"); put("SNOWLIQ", st["SNLIQ"].T, ("dim3", "ntiles"), "mm")
    if not v50:
        put("SMOISEQ", st["SMOISEQ"].T, ("dim2", "ntiles"), "m3/m3")
    put("PGS", np.asarray(st["PGS"], f32) if "PGS" in st else np.zeros(ntl, f32), ("ntiles",), "-")
    if v50:      # 5.0 restart also carries the soil-timestep accumulators (reset at every step when the soil step equals the model step)
        for nm_ in ("ACC_SSOIL", "ACC_QINSUR", "ACC_QSEVA", "ACC_DWATER", "ACC_PRCP", "ACC_ECAN", "ACC_ETRAN", "ACC_EDIR"):
            put(nm_, np.zeros(ntl, f32), ("ntiles",), "-")
        put("ACC_ETRANI", np.zeros((4, ntl), f32), ("dim2", "ntiles"), "-")
    la = ds.createVariable("lat", "f4", ("ntiles",)); lo = ds.createVariable("lon", "f4", ("ntiles",))
    la[:] = np.repeat(S.lat, ntl // S.ntile); lo[:] = np.repeat(S.lon, ntl // S.ntile)
    tv = ds.createVariable("time", "f4", ("time",)); tv.units = f"minutes since {now:%Y-%m-%d %H:%M:%S}"; tv[:] = 0.0
    ds.close()


def read_restart(path, S, st):
    """Warm start from a LIS_RST_NOAHMP401 file; returns updated state dict (numpy)."""
    d = nc.Dataset(path)
    A = lambda k: np.array(d[k][:], f32)
    st = {k: np.array(v) for k, v in st.items()}
    for name, units, key in RST_SCALARS:
        if name not in d.variables:             # 5.0 restarts have no SMCWTD / DEEPRECH / RECH / SMOISEQ
            continue
        st[key] = A(name).astype(np.int32) if key == "ISNOW" else A(name)
    if "PGS" in st and "PGS" in d.variables:                # dynamic crop model: growth stage
        st["PGS"] = A("PGS")
    st["SMC"] = A("SMC").T; st["SMH2O"] = A("SH2O").T
    st["STC"] = np.concatenate([A("TSNO").T, A("TSLB").T], 1)
    st["ZSNSO"] = A("ZSS").T; st["SNICE"] = A("SNOWICE").T; st["SNLIQ"] = A("SNOWLIQ").T
    if "SMOISEQ" in d.variables:
        st["SMOISEQ"] = A("SMOISEQ").T
    return st


FORC_FIELDS = {"Incident Shortwave Radiation Level 001": "swd", "Incident Longwave Radiation Level 001": "lwd",
               "Rainfall Rate Level 001": "pcp"}


def read_forc_attribs(path):
    """forcing_attribs.txt: min/max of each perturbed forcing field (names as in the file)."""
    L = [l.rstrip("\n") for l in open(path) if not l.startswith("#")]
    n = int(L[0]); out = {}
    for i in range(n):
        out[L[1 + 2 * i].strip()] = tuple(float(x) for x in L[2 + 2 * i].split())
    return out


def parse_interval(s):
    s = s.strip().lower()
    for suf, mul in (("hr", 3600), ("mn", 60), ("da", 86400), ("ss", 1)):
        if s.endswith(suf):
            return int(s[: -len(suf)]) * mul
    return int(s)


def ens_mean(v, nens):
    """grid-space output of an ensemble run: mean over the members of each cell, ignoring undefined (-9999) members."""
    a = np.asarray(v, f32).reshape(-1, nens)
    ok = a > -9000.0
    s = np.zeros(a.shape[0], f32)
    for m in range(nens):
        s = s + np.where(ok[:, m], a[:, m], f32(0))
    cnt = ok.sum(1)
    return np.where(cnt > 0, s / np.maximum(cnt, 1).astype(f32), UDEF).astype(f32)


def day0_and_len(now):
    y = now.year
    yl = 365
    if y % 4 == 0:
        yl = 366
        if y % 100 == 0:
            yl = 365
            if y % 400 == 0:
                yl = 366
    return (now.date() - dt.date(y, 1, 1)).days, yl


def run(cfg_path, base_dir, forcing_dir, out_dir, start, end, out_interval_h=6, progress=True, timing=None,
        restart_in=None, restart_out=None):
    tm_ = {"setup": 0.0, "forcing_host": 0.0, "device": 0.0, "output": 0.0, "chunk_device_s": []}
    t_all = time.time(); t = time.time()
    from lisf_noahmp_jax.driver.config import Config as _Cfg
    lsm = _Cfg(cfg_path).s("Land surface model")
    v36 = lsm == "NoahMP.3.6"
    v50 = lsm == "Noah-MP.5.0"
    if v36:
        from lisf_noahmp_jax.driver.params_mp36 import SetupMP36
        S = SetupMP36(cfg_path, base_dir)
    elif v50:
        from lisf_noahmp_jax.driver.params_mp50 import SetupMP50
        S = SetupMP50(cfg_path, base_dir)
    else:
        S = SetupMP(cfg_path, base_dir)
    src = S.cfg.s("Met forcing sources")
    P_ = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
    if src.startswith("NLDAS2"):
        F = NLDAS2(P_(S.cfg.s("NLDAS-2.0 FORA forcing directory")), S.lat, S.lon)
    elif src == "MERRA2":
        F = MERRA2(P_(S.cfg.s("MERRA2 forcing directory")), S.lat, S.lon)
    else:
        raise ValueError(f"forcing source {src!r} not ported")
    nens = S.cfg.i("Number of ensembles per tile") if "Number of ensembles per tile" in S.cfg.d else 1
    ncell = S.ntile
    rep = (lambda a: np.repeat(a, nens, axis=0)) if nens > 1 else (lambda a: a)
    if nens > 1:
        S.glacier_idx = (S.glacier_idx[:, None] * nens + np.arange(nens)[None, :]).ravel().astype(np.int32)
    if v50:
        from ..physics import noahmp50_jax, noahmp50_glacier_jax
        step = model_mp.make_step(S, jit=False, phys=noahmp50_jax, glac=noahmp50_glacier_jax)
    else:
        step = (model_mp36 if v36 else model_mp).make_step(S, jit=False)
    chunk = make_chunk(S, step)
    FP = None
    if S.cfg.s("Forcing perturbation algorithm").lower() != "none":
        assert S.cfg.s("Forcing perturbation algorithm") == "GMAO scheme"
        names, spec = pertmod.read_pert_attrs(P_(S.cfg.s("Forcing perturbation attributes file")), 3)
        fattr = read_forc_attribs(P_(S.cfg.s("Forcing attributes file")))
        FP = dict(P=pertmod.ForcingPerturb(S.nc, S.nr, nens, spec, S.dt, names=names), names=names, spec=spec, attr=fattr,
                  interval=parse_interval(S.cfg.s("Forcing perturbation frequency")))
    DA = None
    if "Data assimilation algorithm" in S.cfg.d and S.cfg.s("Data assimilation algorithm").lower() != "none":
        from lisf_noahmp_jax.driver.da import SoilMoistureDA
        DA = SoilMoistureDA(S, S.cfg, P_, nens, S.dt, parse_interval)
    HY = None
    if "Routing model" in S.cfg.d and S.cfg.s("Routing model").lower() != "none":
        from lisf_noahmp_jax.driver import hymap2
        assert nens == 1, "routing is ported for single-member runs"
        hc = hymap2.hymap2_config(S.cfg, parse_interval)
        HP = hymap2.HyMAP2Params(P_(S.cfg.s("LIS domain and parameter data file")), hc)
        tile_of = np.full((S.nr, S.nc), -1, np.int64); tile_of[S.rows, S.cols] = np.arange(S.ntile)
        HY = dict(P=HP, step=hymap2.make_router(HP), st=hymap2.initial_state(HP), cfg=hc,
                  tile=jnp.asarray(tile_of[HP.rows, HP.cols]), acc=None, n=0)
    lai_trk = LaiTracker(S.lai_mon, S.dt) if S.o["dveg"] in (7, 8, 9) else None
    par = {k: jnp.asarray(rep(v)) for k, v in S.par.items()}
    stat = {"lat": jnp.asarray(rep(S.lat)), "lon": jnp.asarray(rep(S.lon)), "tbot": jnp.asarray(rep(S.tbot)),
            "fvgmax": jnp.asarray(rep(S.fvgmax)), "vegt": jnp.asarray(rep(S.vegt)),
            "green": jnp.asarray(rep((S.green * f32(100.0)).T))}          # monthly green fraction [%] (n, 12), for dveg 1, 6, 7
    if getattr(S, "td_fraction", None) is not None:
        stat["td_fraction"] = jnp.asarray(rep(S.td_fraction))
    st0 = S.initial_state()
    st0["RHMIN"] = np.full(ncell, 999.0, f32)
    if restart_in:
        st0 = read_restart(restart_in, S, st0)
    elif not v36:
        # coldstart first-step fix-ups (noahmp_driver_401: itimestep == 1)
        first = F.at(start + dt.timedelta(seconds=S.dt))
        q2 = first["Qair"] / (f32(1) - first["Qair"])
        st0["EAH"] = (first["PSurf"] * (q2 / (f32(0.622) + q2))).astype(f32)
        st0["TAH"] = first["Tair"].astype(f32)
        st0["CM"] = np.full(ncell, 0.1, f32); st0["CH"] = np.full(ncell, 0.1, f32)
    st0 = {k: rep(v) for k, v in st0.items()}
    st = {k: jnp.asarray(v) for k, v in st0.items()}
    static_out = {"Landmask": np.ones(S.ntile, f32), "Landcover": S.vegt.astype(f32), "Soiltype": S.soiltyp.astype(f32),
                  "Elevation": S.static["ELEVATION"]}
    from types import SimpleNamespace
    from lisf_noahmp_jax.driver.asyncout import AsyncWriter
    W = AsyncWriter()
    grid_ = SimpleNamespace(nr=S.nr, nc=S.nc, rows=S.rows, cols=S.cols, glat1d=S.glat1d, glon1d=S.glon1d, sldpth=S.sldpth)    # what write_hist needs
    tm_["setup"] += time.time() - t
    nstep = int((end - start).total_seconds() // S.dt)
    K = out_interval_h * 3600 // S.dt
    for c0 in range(0, nstep, K):
        t = time.time()
        times = [start + dt.timedelta(seconds=S.dt * (s + 1)) for s in range(c0, min(c0 + K, nstep))]
        raws, tms = [], []
        for now in times:
            o = F.at(now)
            raw = {"tair": o["Tair"], "qair": o["Qair"], "swd": o["SWdown"], "lwd": o["LWdown"],
                   "psurf": o["PSurf"], "pcp": o["Rainf"], "u": o["Wind_E"], "v": o["Wind_N"]}
            raw = {k: rep(v) for k, v in raw.items()}
            if lai_trk is not None:
                raw["lai_in"] = rep(lai_trk.tlai(now - dt.timedelta(seconds=S.dt)))
            if FP is not None and (now.hour * 3600 + now.minute * 60 + now.second) % FP["interval"] == 0:
                pr = FP["P"].get_pert()
                for item, fi in enumerate(FP["P"].order):
                    nm_ = FP["names"][fi]
                    key = FORC_FIELDS[nm_]
                    p = pr[item][S.cols, S.rows, :]                                  # (ncell, nens)
                    base = {"swd": o["SWdown"], "lwd": o["LWdown"], "pcp": o["Rainf"]}[key]
                    mn, mx = FP["attr"][nm_]
                    raw[key] = pertmod.apply_forcing_perturbation(base, p, FP["spec"][fi]["typ"], mn, mx).reshape(-1)
            raws.append(raw)
            d0, yl = day0_and_len(now)
            pm, qm, fac = month_d_new_index(now)
            tms.append({"julian0": f32(d0), "gmt": f32(now.hour) + f32(now.minute) / f32(60.) + f32(now.second) / f32(3600.),
                        "yearlen": f32(yl), "hour": f32(now.hour), "gprev": f32(pm), "gpost": f32(qm), "gfac": f32(fac)})
        rawK = {k: np.stack([r[k] for r in raws]).astype(f32) for k in raws[0]}
        tmK = {k: np.array([r[k] for r in tms], f32) for k in tms[0]}
        tm_["forcing_host"] += time.time() - t
        t = time.time()
        st, acc, cnt, last, tmin, tmax, ys = chunk(par, stat, st, {k: jnp.asarray(v) for k, v in rawK.items()},
                                                {k: jnp.asarray(v) for k, v in tmK.items()})
        jax.block_until_ready(acc)
        tm_["device"] += time.time() - t; tm_["chunk_device_s"].append(time.time() - t)
        t = time.time()
        n = f32(len(times))
        tavg = {k: np.asarray(v) / n for k, v in acc.items()}
        cn = np.asarray(cnt)
        tavg["Albedo"] = np.where(cn > 0, np.asarray(acc["Albedo"]) / np.maximum(cn, 1), UDEF).astype(f32)
        accum = {k: np.asarray(acc[k]) for k in ACC}
        inst = {k: np.asarray(v) for k, v in last.items()}
        last_np = dict(inst)
        inst["RHMin"] = np.asarray(st["RHMIN"]) * f32(100.0)
        tmin_, tmax_ = np.asarray(tmin), np.asarray(tmax)
        if nens > 1:
            em = lambda d: {k: ens_mean(v, nens) for k, v in d.items()}
            tavg, accum, inst = em(tavg), em(accum), em(inst)
            tmin_, tmax_ = ens_mean(tmin_, nens), ens_mean(tmax_, nens)
        now = times[-1]
        d = f"{out_dir}/SURFACEMODEL/{now:%Y%m}"; os.makedirs(d, exist_ok=True)
        W.submit(write_hist, f"{d}/LIS_HIST_{now:%Y%m%d%H%M}.d01.nc", grid_, now, tavg, accum, inst, tmin_, tmax_,
                 static_out)
        if HY is not None:
            t_r = time.time()
            rsf, rsb = np.asarray(ys[0]), np.asarray(ys[1])                     # (K, ntile) mm/s
            dtr = int(HY["cfg"]["dt"])
            for si, now_s in enumerate(times):
                if int((now_s - start).total_seconds()) % dtr != 0:
                    continue
                ti = HY["tile"]
                vec = lambda a: jnp.where(ti >= 0, jnp.asarray(a)[jnp.clip(ti, 0)], jnp.float32(-9999.0))
                with jax.experimental.enable_x64():
                    HY["st"], o = HY["step"](HY["st"], vec(rsf[si]), vec(rsb[si]))
                HY["last"] = o
                if os.environ.get("HYDEBUG"):
                    for kk, vv in o.items():
                        nn = int(jnp.isnan(vv).sum())
                        if nn and not HY.get("dbg_" + kk):
                            HY["dbg_" + kk] = True
                            idx = np.nonzero(np.asarray(jnp.isnan(vv)))[0][:5]
                            print(f"[HYDEBUG] {now_s}: first NaN in {kk}: {nn} cells, seq idx {idx}, row/col "
                                  f"{[(int(HY['P'].rows[i]), int(HY['P'].cols[i])) for i in idx]}", flush=True)
                HY["acc"] = o["rivout"] if HY["acc"] is None else HY["acc"] + o["rivout"]
                HY["n"] += 1
            if HY["n"] > 0 and int((now - start).total_seconds()) % int(HY["cfg"]["out_interval"]) == 0:
                vals = {k: np.asarray(v) for k, v in HY["last"].items()}
                vals["rivout"] = np.asarray(HY["acc"]) / f32(HY["n"])
                vals["roffsto"] = np.asarray(HY["st"]["roffsto"]); vals["basfsto"] = np.asarray(HY["st"]["basfsto"])
                d = f"{out_dir}/ROUTING/{now:%Y%m}"; os.makedirs(d, exist_ok=True)
                write_routing_hist(f"{d}/LIS_HIST_{now:%Y%m%d%H%M}.d01.nc", S, HY["P"], now, vals)
                HY["acc"], HY["n"] = None, 0
            tm_["routing"] = tm_.get("routing", 0.0) + time.time() - t_r
        if DA is not None:
            stn = {k: np.asarray(v) for k, v in st.items()}
            stn = DA.perturb_states(now, stn)
            stn = DA.analysis(now, stn, last_np, stn["TG"], stn["SWE"])
            st = {k: jnp.asarray(v) for k, v in stn.items()}
        tm_["output"] += time.time() - t
        if progress:
            print(f"[{now}] wrote history  ({time.time()-t_all:.1f}s elapsed)", flush=True)
    t = time.time()
    W.close()
    tm_["output"] += time.time() - t
    if restart_out:
        write_restart(restart_out, S, st, end)
    tm_["total"] = time.time() - t_all
    if timing is not None:
        timing.update(tm_)
    return S, st


if __name__ == "__main__":
    base = sys.argv[1]
    cfg = sys.argv[3] if len(sys.argv) > 3 else "lis.config.mp"
    start = dt.datetime.strptime(sys.argv[4], "%Y%m%d") if len(sys.argv) > 4 else dt.datetime(2017, 2, 1)
    end = dt.datetime.strptime(sys.argv[5], "%Y%m%d") if len(sys.argv) > 5 else dt.datetime(2017, 2, 2)
    tm = {}
    run(f"{base}/{cfg}", base, f"{base}/INPUT/NLDAS2.FORCING", sys.argv[2], start, end, timing=tm)
    print("TIMING", {k: (round(v, 2) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in tm.items()})
