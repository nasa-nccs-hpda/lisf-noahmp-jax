"""One Noah-MP 3.6 LIS time step for all tiles in JAX: the NoahMP401_main / noahmp_driver_401 preparation
(forcing, zenith angle with the hour-resolution Julian day of calc_declin_36, FICEOLD, TV clamp), NOAHMP_SFLX_36, and the output diagnostics."""
import numpy as np
import jax, jax.numpy as jnp
from ..physics import noahmp36_jax as nm

f32 = jnp.float32
DEGRAD = np.float32(3.14159265) / np.float32(180.)
DPD = np.float32(360.) / np.float32(365.)
STATE_SCALARS = ("ALBOLD SNEQVO TAH EAH FWET CANLIQ CANICE TV TG QSNOW SNDPTH SWE ZWT WA WT WSLAKE LFMASS RTMASS "
                 "STMASS WOOD STBLCP FASTCP PLAI PSAI CM CH TAUSS SMCWTD").split()
STATE_ARRAYS = ("STC", "SMH2O", "SMC", "ZSNSO", "SNICE", "SNLIQ")


def cosz_julian(lat, lon, julian_day0, gmt, hour):
    """calc_declin_36. julian_day0: days since Jan 1; gmt: hours incl. minutes and seconds; hour: integer hour of the day (float32).
    JULIAN uses the whole hour only (Noah-MP 3.6), the local hour angle uses the minutes."""
    julian = julian_day0 + hour / f32(24.)
    obecl = f32(23.5) * DEGRAD
    sinob = jnp.sin(obecl)
    sx = jnp.where(julian >= 80., DPD * (julian - f32(80.)) * DEGRAD, DPD * (julian + f32(285.)) * DEGRAD)
    declin = jnp.arcsin(sinob * jnp.sin(sx))
    tl = gmt + lon / f32(15.)
    tl = jnp.mod(tl + f32(24.), f32(24.))
    hr = f32(15.) * (tl - f32(12.)) * DEGRAD
    cosz = jnp.sin(lat * DEGRAD) * jnp.sin(declin) + jnp.cos(lat * DEGRAD) * jnp.cos(declin) * jnp.cos(hr)
    return jnp.clip(cosz, -1.0, 1.0), julian


def make_step(S, jit=True):
    """step(par, stat, st, raw, tm) -> (st, diag): vmapped over tiles.
    stat: dict lat, lon, tbot, fvgmax, vegt (per tile). raw: tair qair swd lwd psurf pcp u v (per tile).
    tm: dict julian0 (days since Jan 1), gmt (hours), yearlen."""
    o = dict(S.o)
    dt = f32(S.dt)
    zsoil = jnp.asarray(S.zsoil)
    zref = f32(0.5) * f32(S.dz8w)
    ISURB = S.veg["ISURBAN"]

    def one(par, x):
        return nm.sflx_mp(par, x, o)
    vone = jax.vmap(one)

    def step(par, stat, st, raw, tm):
        n = raw["tair"].shape[0]
        cosz, julian = cosz_julian(stat["lat"], stat["lon"], tm["julian0"], tm["gmt"], tm["hour"])
        q_ml = raw["qair"]
        prcp = raw["pcp"]
        p_in = dt * prcp
        ice = st["SNICE"]; liq = st["SNLIQ"]
        active = (st["ISNOW"][:, None] + 3) <= jnp.arange(3)[None, :]
        ficeold = jnp.where(active, ice / jnp.where(active, ice + liq, 1.0), 0.0)
        zero = jnp.zeros(n, f32)
        x = dict(
            DT=jnp.full(n, dt), LAT=stat["lat"], YEARLEN=jnp.full(n, tm["yearlen"], f32), JULIAN=jnp.full(n, julian),
            COSZ=cosz, FVGMAX=stat["fvgmax"], VEGTYP=stat["vegt"].astype(f32), T_ML=raw["tair"], P_ML=raw["psurf"],
            PSFC=raw["psurf"], U_ML=raw["u"], V_ML=raw["v"], Q_ML=q_ml, LWDN=raw["lwd"], SWDN=raw["swd"],
            PRCP=prcp, TBOT=stat["tbot"],
            CO2PP=par["CO2"] * raw["psurf"], O2PP=par["O2"] * raw["psurf"],
            FOLN=jnp.ones(n, f32), Z_ML=jnp.full(n, zref), rivsto=zero, fldsto=zero, fldfrc=zero,
            ZSOIL=jnp.tile(zsoil[None, :], (n, 1)), FICEOLD=ficeold, VEGTYPE_=stat["vegt"].astype(f32))
        for k in STATE_SCALARS:
            x[k] = st[k]
        x["TV"] = jnp.where(st["TV"] > st["TG"] + f32(20.0), st["TG"] + f32(20.0), st["TV"])      # noahmp_driver_36: if (tv > tg+20) tv = tg+20
        for k in STATE_ARRAYS:
            x[k] = st[k]
        x["ISNOW"] = st["ISNOW"]
        out = vone(par, x)
        new = dict(st)
        for k in ("ALBOLD SNEQVO TAH EAH FWET CANLIQ CANICE TV TG QSNOW SNDPTH SWE ZWT WA WT PLAI PSAI CM CH TAUSS").split():
            new[k] = out[k]
        for k in ("STC", "SMH2O", "SMC", "ZSNSO", "SNICE", "SNLIQ"):
            new[k] = jnp.stack(out[k], -1) if isinstance(out[k], (list, tuple)) else out[k]
        new["ISNOW"] = out["ISNOW"].astype(jnp.int32)
        new["SMCWTD"] = st["SMCWTD"]
        new["SFCRUNOFF"] = st["SFCRUNOFF"] + out["RUNSF"] * dt
        new["UDRRUNOFF"] = st["UDRRUNOFF"] + out["RUNSB"] * dt
        new["ACSNOW"] = st["ACSNOW"] + p_in * out["FPICE"]
        new["ACSNOM"] = st["ACSNOM"] + out["QSNBOT"] * dt + out["PONDING"] + out["PONDING1"] + out["PONDING2"]
        sal = out["SALB"]
        valid_alb = sal > -999.
        lh = out["FCEV"] + out["FGEV"] + out["FCTR"]
        qfx = out["ECAN"] + out["ESOIL"] + out["ETRAN"]
        fveg = out["FVEGMP"]
        stc_soil = new["STC"][:, 3:]
        diag = {
            "Qle": lh, "Qh": out["FSH"], "Qg": out["SSOIL"], "Evap": qfx,
            "Qs": out["RUNSF"] * dt, "Qsb": out["RUNSB"] * dt, "RunsfRate": out["RUNSF"], "RunsbRate": out["RUNSB"],
            "AvgSurfT": fveg * out["TV"] + (f32(1.0) - fveg) * out["TGB"],
            "Albedo": jnp.where(valid_alb, sal * f32(100.0), 0.0), "Albedo_n": valid_alb.astype(f32),
            "SWE": new["SWE"], "SnowDepth": new["SNDPTH"],
            "CanopInt": jnp.where(new["CANLIQ"] < -9000.0, -9999.0, new["CANLIQ"] + new["CANICE"]), "Snowcover": out["FSNO"] * f32(100.0),
            "TotalPrecip": raw["pcp"] * dt, "Greenness": fveg,
            "Wind_f": jnp.sqrt(raw["u"] * raw["u"] + raw["v"] * raw["v"]), "Tair_f": raw["tair"],
            "Qair_f": raw["qair"], "Psurf_f": raw["psurf"], "SWdown_f": raw["swd"], "LWdown_f": raw["lwd"],
        }
        for k in range(4):
            diag[f"SoilMoist{k+1}"] = new["SMC"][:, k]
            diag[f"SoilTemp{k+1}"] = stc_soil[:, k]
            diag[f"SmLiqFrac{k+1}"] = new["SMH2O"][:, k]
            relsmc = out["RELSMC"][k] if isinstance(out["RELSMC"], (list, tuple)) else out["RELSMC"][:, k]
            relsmc = jnp.clip(relsmc, 0.01, 1.0)
            diag[f"RelSMC{k+1}"] = jnp.where(stat["vegt"] == ISURB, -9999.0, relsmc)
        return new, diag

    return jax.jit(step) if jit else step
