"""End-to-end LIS-style Noah-3.9 retrospective run in Python/JAX: config + LDT parameter file + NLDAS-2 forcing ->
6-hourly time-averaged history files with the same variables as LIS (walkthrough output table)."""
import sys, os, time, datetime as dt
import numpy as np
import jax, jax.numpy as jnp
import netCDF4 as nc
from lisf_noahmp_jax.driver.params import Setup
from lisf_noahmp_jax.driver.forcing import NLDAS2, MERRA2
from lisf_noahmp_jax.driver import model

f32 = np.float32
TAVG_VARS = ["Swnet", "Lwnet", "Qle", "Qh", "Qg", "Snowf", "Rainf", "Evap", "Qs", "Qsb", "Albedo", "SWE", "SnowDepth",
             "SoilMoist1", "SoilMoist2", "SoilMoist3", "SoilMoist4", "SoilTemp1", "SoilTemp2", "SoilTemp3", "SoilTemp4",
             "Rainf_f", "Tair_f", "SWdown_f", "LWdown_f"]
META = {  # name: (units, standard_name, long_name)
    "Swnet": ("W m-2", "surface_net_downward_shortwave_flux", "net downward shortwave radiation"),
    "Lwnet": ("W m-2", "surface_net_downward_longwave_flux", "net downward longwave radiation"),
    "Qle": ("W m-2", "surface_upward_latent_heat_flux", "latent heat flux"),
    "Qh": ("W m-2", "surface_upward_sensible_heat_flux", "sensible heat flux"),
    "Qg": ("W m-2", "downward_heat_flux_in_soil", "soil heat flux"),
    "Snowf": ("kg m-2 s-1", "snowfall_rate", "snowfall rate"),
    "Rainf": ("kg m-2 s-1", "precipitation_rate", "precipitation rate"),
    "Evap": ("kg m-2 s-1", "total_evapotranspiration", "total evapotranspiration"),
    "Qs": ("kg m-2 s-1", "surface_runoff_amount", "surface runoff"),
    "Qsb": ("kg m-2 s-1", "subsurface_runoff_amount", "subsurface runoff amount"),
    "Albedo": ("-", "surface_albedo", "surface albedo"),
    "SWE": ("kg m-2", "liquid_water_content_of_surface_snow", "snow water equivalent"),
    "SnowDepth": ("m", "snow_depth", "snow depth"),
    "SoilMoist": ("m^3 m-3", "soil_moisture_content", "soil moisture content"),
    "SoilTemp": ("K", "soil_temperature", "soil temperature"),
    "Rainf_f": ("kg m-2 s-1", "rainfall_flux", "rainfall flux"),
    "Tair_f": ("K", "air_temperature", "air temperature"),
    "SWdown_f": ("W m-2", "surface_downwelling_shortwave_flux_in_air", "surface downward shortwave radiation"),
    "LWdown_f": ("W m-2", "surface_downwelling_longwave_flux_in_air", "surface downward longwave radiation"),
}


def write_hist(path, S, label, avg):
    nr, ncol = S.nr, S.nc
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("east_west", ncol); ds.createDimension("north_south", nr); ds.createDimension("time", 1)
    ds.createDimension("SoilMoist_profiles", 4); ds.createDimension("SoilTemp_profiles", 4)
    ds.title = "LIS land surface model output (Python/JAX port)"; ds.missing_value = "-9999.0"
    ds.MAP_PROJECTION = "EQUIDISTANT CYLINDRICAL"; ds.DX = f32(0.25); ds.DY = f32(0.25)
    ds.SOUTH_WEST_CORNER_LAT = f32(S.glat1d.min()); ds.SOUTH_WEST_CORNER_LON = f32(S.glon1d.min())
    la = ds.createVariable("lat", "f4", ("north_south",)); lo = ds.createVariable("lon", "f4", ("east_west",))
    tv = ds.createVariable("time", "f4", ("time",)); tv.units = f"minutes since {label:%Y-%m-%d %H:%M:%S}"
    tv[:] = 0.0
    la[:] = S.glat1d; lo[:] = S.glon1d
    def grid(a):
        g = np.full((nr, ncol), -9999.0, f32); g[S.rows, S.cols] = a; return g
    for v in ("Swnet Lwnet Qle Qh Qg Snowf Rainf Evap Qs Qsb Albedo SWE SnowDepth Rainf_f Tair_f SWdown_f LWdown_f").split():
        u, sn, ln = META[v]
        x = ds.createVariable(v + "_tavg", "f4", ("north_south", "east_west"), fill_value=f32(-9999.0))
        x.units, x.standard_name, x.long_name = u, sn, ln
        x[:] = grid(avg[v])
    for v in ("SoilMoist", "SoilTemp"):
        u, sn, ln = META[v]
        x = ds.createVariable(v + "_tavg", "f4", (v + "_profiles", "north_south", "east_west"), fill_value=f32(-9999.0))
        x.units, x.standard_name, x.long_name = u, sn, ln
        x[:] = np.stack([grid(avg[f"{v}{k}"]) for k in range(1, 5)])
    ds.close()


RST_VARS = [("T1", "t1", "K", "skin temperature"), ("CMC", "cmc", "-", "canopy moisture content"),
            ("SNOWH", "snowh", "m", "snow depth"), ("SNEQV", "sneqv", "m", "snow water equivalent"),
            ("STC", "stc", "K", "soil temperature"), ("SMC", "smc", "m^3 m-3", "total soil moisture"),
            ("SH2O", "sh2o", "m^3 m-3", "liquid soil moisture"), ("CM", "cm", "-", "exchange coefficient for momentum"),
            ("CH", "ch", "-", "exchange coefficient for heat"), ("Z0", "z0", "m", "roughness length"),
            ("EMISS", "emiss", "-", "surface emissivity"), ("ALBEDO", "albedo", "-", "surface albedo"),
            ("SNOTIME1", "snotime1", "-", "Snotime1"), ("Q1", "q1", "-", "Q1")]


def write_restart(path, S, st, now):
    """LIS_RST_NOAH39/36-style restart (same variable names/layout as the Fortran writer)."""
    ds = nc.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("ntiles", S.ntile); ds.createDimension("dim1", 4); ds.createDimension("dim2", 4)
    ds.createDimension("time", 1)
    ds.title = "LIS land surface model restart (Python/JAX port)"; ds.source = f"{S.model}"
    for name, key, u, ln in RST_VARS:
        a = np.asarray(st[key], f32)
        dims = ("dim2", "ntiles") if name == "STC" else (("dim1", "ntiles") if a.ndim == 2 else ("ntiles",))
        x = ds.createVariable(name, "f4", dims); x.units, x.long_name = u, ln; x[:] = a
    la = ds.createVariable("lat", "f4", ("ntiles",)); lo = ds.createVariable("lon", "f4", ("ntiles",))
    la[:] = S.lat; lo[:] = S.lon
    tv = ds.createVariable("time", "f4", ("time",)); tv.units = f"minutes since {now:%Y-%m-%d %H:%M:%S}"; tv[:] = 0.0
    ds.close()


def read_restart(path, S):
    """Warm start: state from a restart file; quantities LIS does not store are re-initialised as at coldstart."""
    d = nc.Dataset(path)
    st = {}
    for name, key, u, ln in RST_VARS:
        st[key] = np.array(d[name][:], f32)
    return st


def run(cfg_path, base_dir, forcing_dir, out_dir, start, end, out_interval_h=6, progress=True, timing=None,
        restart_in=None, restart_out=None):
    """Chunked run: forcing for one output interval is prepared on the host, then the whole interval (24 steps of 15 min)
    is run on the device in one lax.scan. `timing` (dict) receives wall-clock seconds per phase."""
    from lisf_noahmp_jax.driver.config import Config
    if Config(cfg_path).s("Land surface model").startswith("Noah-MP") or Config(cfg_path).s("Land surface model") == "NoahMP.3.6":
        from lisf_noahmp_jax.driver import run_mp
        return run_mp.run(cfg_path, base_dir, forcing_dir, out_dir, start, end, out_interval_h, progress, timing, restart_in, restart_out)
    tm = {"setup": 0.0, "forcing_host": 0.0, "device": 0.0, "output": 0.0, "chunk_device_s": []}
    t_all = time.time()
    t = time.time()
    if Config(cfg_path).s("Land surface model") == "Noah.2.7.1":
        from lisf_noahmp_jax.driver.params271 import Setup271
        S = Setup271(cfg_path, base_dir)
    else:
        S = Setup(cfg_path, base_dir)
    src = S.cfg.s("Met forcing sources")
    if src.startswith("NLDAS2"):
        F = NLDAS2(forcing_dir or S.cfg.s("NLDAS-2.0 FORA forcing directory"), S.lat, S.lon, alt=getattr(S, "f2t_alt", None))
    elif src == "MERRA2":
        F = MERRA2(forcing_dir or S.cfg.s("MERRA2 forcing directory"), S.lat, S.lon)
    else:
        raise ValueError(f"forcing source {src!r} not ported")
    chunk = model.make_chunk(S)
    par = {k: jnp.asarray(v) for k, v in S.par.items()}
    st0 = S.initial_state(start.year, start.month, start.day)
    if restart_in:
        st0.update(read_restart(restart_in, S))
        st0["z0_old"] = st0["z0"].copy()
    st = {k: jnp.asarray(v) for k, v in st0.items()}
    tm["setup"] += time.time() - t
    nstep = int((end - start).total_seconds() // S.dt)
    K = out_interval_h * 3600 // S.dt
    dyn_cache = {}
    from types import SimpleNamespace
    from lisf_noahmp_jax.driver.asyncout import AsyncWriter
    W = AsyncWriter()
    grid_ = SimpleNamespace(nr=S.nr, nc=S.nc, rows=S.rows, cols=S.cols, glat1d=S.glat1d, glon1d=S.glon1d)    # what write_hist needs
    for c0 in range(0, nstep, K):
        t = time.time()
        times = [start + dt.timedelta(seconds=S.dt * (s + 1)) for s in range(c0, min(c0 + K, nstep))]
        raws, dyns = [], []
        for now in times:
            key = (now.year, now.month, now.day)
            if key not in dyn_cache:
                dyn_cache[key] = S.dyn(*key, newday=True)
                dyn_cache[key + ("cont",)] = S.dyn(*key, newday=False)
            first_of_day = (len(dyns) == 0 and now == start + dt.timedelta(seconds=S.dt)) or \
                (now.hour == 0 and now.minute == 0 and now.second == 0) or (now.hour == 0 and now.minute == S.dt // 60)
            # dynsetup runs when the label date changes: the 00:00 label belongs to the new day, 00:15 does not re-trigger
            dyns.append(dyn_cache[key] if (now.hour == 0 and now.minute == 0) or now == start + dt.timedelta(seconds=S.dt)
                        else dyn_cache[key + ("cont",)])
            o = F.at(now)
            if "Wind_E_alt" in o:      # Noah 2.7.1 noah271_f2t quirk (tile index instead of tile_id for wind and precipitation validity)
                u, v, pcp = o["Wind_E_alt"], o["Wind_N_alt"], np.where(o["Rainf_valid_alt"], o["Rainf"], f32(0.0))
            else:
                u, v, pcp = o["Wind_E"], o["Wind_N"], o["Rainf"]
            raws.append({"tair": o["Tair"], "qair": o["Qair"], "swd": o["SWdown"], "lwd": o["LWdown"],
                         "psurf": o["PSurf"], "pcp": pcp, "u": u, "v": v, "pcp_f": o["Rainf"]})
        rawK = {k: np.stack([r[k] for r in raws]) for k in raws[0]}
        dynK = {k: np.stack([d[k] for d in dyns]) for k in dyns[0]}
        tm["forcing_host"] += time.time() - t
        t = time.time()
        st, acc = chunk(par, {k: jnp.asarray(v) for k, v in dynK.items()}, st, {k: jnp.asarray(v) for k, v in rawK.items()})
        jax.block_until_ready(acc)
        tm["device"] += time.time() - t
        tm["chunk_device_s"].append(time.time() - t)
        t = time.time()
        now = times[-1]
        avg = {k: np.asarray(v) / f32(len(times)) for k, v in acc.items()}
        d = f"{out_dir}/SURFACEMODEL/{now:%Y%m}"; os.makedirs(d, exist_ok=True)
        W.submit(write_hist, f"{d}/LIS_HIST_{now:%Y%m%d%H%M}.d01.nc", grid_, now, avg)
        tm["output"] += time.time() - t
        if progress:
            print(f"[{now}] wrote history  ({time.time()-t_all:.1f}s elapsed)", flush=True)
    t = time.time()
    W.close()
    tm["output"] += time.time() - t
    if restart_out:
        write_restart(restart_out, S, {k: np.asarray(v) for k, v in st.items()}, end)
    tm["total"] = time.time() - t_all
    if timing is not None:
        timing.update(tm)
    return S, st


if __name__ == "__main__":
    base = sys.argv[1]
    cfg = sys.argv[3] if len(sys.argv) > 3 else "lis.config.n39"
    start = dt.datetime.strptime(sys.argv[4], "%Y%m%d") if len(sys.argv) > 4 else dt.datetime(2017, 2, 1)
    end = dt.datetime.strptime(sys.argv[5], "%Y%m%d") if len(sys.argv) > 5 else dt.datetime(2017, 2, 8)
    tm = {}
    run(f"{base}/{cfg}", base, f"{base}/INPUT/NLDAS2.FORCING", sys.argv[2], start, end, timing=tm)
    print("TIMING", {k: (round(v, 2) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in tm.items()})
