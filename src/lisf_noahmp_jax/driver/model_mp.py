"""One Noah-MP 4.0.1 LIS time step for all tiles in JAX: the NoahMP401_main / noahmp_driver_401 preparation
(forcing conversion, zenith angle, FICEOLD, first-step coldstart fix-ups), NOAHMP_SFLX, and the output diagnostics."""
import numpy as np
import jax, jax.numpy as jnp
from ..physics import noahmp401_jax as nm
from ..physics import noahmp401_glacier_jax as ng

f32 = jnp.float32
DEGRAD = np.float32(3.14159265) / np.float32(180.)
DPD = np.float32(360.) / np.float32(365.)
STATE_SCALARS = ("ALBOLD SNEQVO TAH EAH FWET CANLIQ CANICE TV TG QSNOW SNDPTH SWE ZWT WA WT WSLAKE LFMASS RTMASS "
                 "STMASS WOOD STBLCP FASTCP PLAI PSAI CM CH TAUSS SMCWTD").split()
STATE_ARRAYS = ("STC", "SMH2O", "SMC", "ZSNSO", "SNICE", "SNLIQ")


def cosz_julian(lat, lon, julian_day0, gmt):
    """calc_declin_401. julian_day0: days since Jan 1 (integer-valued float32 scalar); gmt: hours (float32 scalar)."""
    julian = julian_day0 + gmt / f32(24.)
    obecl = f32(23.5) * DEGRAD
    sinob = jnp.sin(obecl)
    sx = jnp.where(julian >= 80., DPD * (julian - f32(80.)) * DEGRAD, DPD * (julian + f32(285.)) * DEGRAD)
    declin = jnp.arcsin(sinob * jnp.sin(sx))
    tl = gmt + lon / f32(15.)
    tl = jnp.mod(tl + f32(24.), f32(24.))
    hr = f32(15.) * (tl - f32(12.)) * DEGRAD
    cosz = jnp.sin(lat * DEGRAD) * jnp.sin(declin) + jnp.cos(lat * DEGRAD) * jnp.cos(declin) * jnp.cos(hr)
    return jnp.clip(cosz, -1.0, 1.0), julian


def make_step(S, jit=True, phys=None, glac=None):
    """step(par, stat, st, raw, tm) -> (st, diag): vmapped over tiles.
    stat: dict lat, lon, tbot, fvgmax, vegt (per tile). raw: tair qair swd lwd psurf pcp u v (per tile).
    tm: dict julian0 (days since Jan 1), gmt (hours), yearlen. phys/glac: the physics modules (default: Noah-MP 4.0.1)."""
    nm_, ng_ = phys or nm, glac or ng
    o = dict(S.o)
    dt = f32(S.dt)
    zsoil = jnp.asarray(S.zsoil)
    zref = f32(0.5) * f32(S.dz8w)
    ISURB = S.veg["ISURBAN"]
    v50 = getattr(S, "model", "") == "Noah-MP.5.0"      # 5.0: snow cover is a fraction in the history files (4.0.1: percent)

    def one(par, x):
        return nm_.sflx_mp(par, x, o)
    vone = jax.vmap(one)
    gidx = np.asarray(getattr(S, "glacier_idx", np.zeros(0, np.int32)), np.int32)
    og = dict(alb=o["alb"], snf=o["snf"], tbot=o["tbot"], stc=o["stc"], gla=S.o["gla"], sndpth=2000.0)
    gone = jax.vmap(lambda P, x: ng_.sflx_glacier(P, x, og))

    def step(par, stat, st, raw, tm):
        n = raw["tair"].shape[0]
        cosz, julian = cosz_julian(stat["lat"], stat["lon"], tm["julian0"], tm["gmt"])
        q2mr = raw["qair"] / (f32(1.0) - raw["qair"])
        q_ml = q2mr / (f32(1.0) + q2mr)
        p_in = dt * raw["pcp"]
        prcp = p_in / dt
        ice = st["SNICE"]; liq = st["SNLIQ"]
        active = (st["ISNOW"][:, None] + 3) <= jnp.arange(3)[None, :]
        ficeold = jnp.where(active, ice / jnp.where(active, ice + liq, 1.0), 0.0)
        zero = jnp.zeros(n, f32)
        x = dict(
            DT=jnp.full(n, dt), LAT=stat["lat"], YEARLEN=jnp.full(n, tm["yearlen"], f32), JULIAN=jnp.full(n, julian),
            COSZ=cosz, FVGMAX=stat["fvgmax"], VEGTYP=stat["vegt"].astype(f32), T_ML=raw["tair"], P_ML=(raw["psurf"] + raw["psurf"]) * f32(0.5),
            PSFC=raw["psurf"], U_ML=raw["u"], V_ML=raw["v"], Q_ML=q_ml, LWDN=raw["lwd"], SWDN=raw["swd"],
            PRCPCONV=zero, PRCPNONC=prcp, PRCPSHCV=zero, PRCPSNOW=zero, PRCPGRPL=zero, PRCPHAIL=zero, TBOT=stat["tbot"],
            CO2PP=par["CO2"] * ((raw["psurf"] + raw["psurf"]) * f32(0.5)), O2PP=par["O2"] * ((raw["psurf"] + raw["psurf"]) * f32(0.5)),
            FOLN=jnp.ones(n, f32), Z_ML=jnp.full(n, zref), rivsto=zero, fldsto=zero, fldfrc=zero,
            ZSOIL=jnp.tile(zsoil[None, :], (n, 1)), FICEOLD=ficeold, VEGTYPE_=stat["vegt"].astype(f32))
        if o.get("tdrn", 0) > 0:                                      # tile drainage fraction (LDT) and grid size
            x["TD_FRACTION"] = stat["td_fraction"]; x["DX"] = jnp.full(n, S.dx, f32)
        if "green" in stat and o["dveg"] in (1, 6, 7):               # VEGFRA = month_d_new(SHDFAC_MONTHLY) (noahmp_driver_50)
            ip, iq = tm["gprev"].astype(jnp.int32), tm["gpost"].astype(jnp.int32)
            g = stat["green"]
            x["FVEG"] = (g[:, ip] * (f32(1.0) - tm["gfac"]) + g[:, iq] * tm["gfac"]) / f32(100.0)
        for k in STATE_SCALARS:
            x[k] = st[k]
        if o.get("crop", 0) > 0:                                      # dynamic crop model states
            for k in ("GRAIN", "GDD", "PGS"):
                x[k] = st[k]
        for k in STATE_ARRAYS:
            x[k] = st[k]
        x["ISNOW"] = st["ISNOW"]
        if o["dveg"] in (7, 8, 9):                                    # input LAI (undefined tiles keep the carried LAI)
            x["PLAI"] = jnp.where(raw["lai_in"] > f32(-900.0), raw["lai_in"], st["PLAI"])
        out = vone(par, x)
        if len(gidx):
            xg = {k: v[gidx] for k, v in x.items()}
            xg["TBOT"] = jnp.minimum(xg["TBOT"], f32(263.15))
            og_ = gone({k: v[gidx] for k, v in par.items()} if glac is not None else {"ZBOT": par["ZBOT"][gidx]}, xg)
            out = dict(out)
            for k, v in og_.items():
                if isinstance(v, (list, tuple)):
                    out[k] = [jnp.asarray(a).at[...].get() for a in out[k]]
                    out[k] = [a.at[gidx].set(b) for a, b in zip(out[k], v)]
                else:
                    out[k] = out[k].at[gidx].set(v)
            undef = jnp.full(len(gidx), -9999.0, f32)
            zg = jnp.zeros(len(gidx), f32)
            for k in ("TV", "CANLIQ", "CANICE", "TAH", "EAH", "FWET", "WA", "WT"):
                out[k] = out[k].at[gidx].set(zg if (v50 and k in ("CANLIQ", "CANICE")) else undef)     # 5.0 writes zero canopy water on land ice
            for k in ("FCEV", "FCTR", "ECAN", "ETRAN", "FVEGMP"):
                out[k] = out[k].at[gidx].set(zg)
            out["TGB"] = out["TGB"].at[gidx].set(og_["TG"])
            out["FSNO"] = out["FSNO"].at[gidx].set(jnp.ones(len(gidx), f32))
            out["FGEV"] = out["FGEV"].at[gidx].set(og_["FGEV"])
            out["RELSMC"] = [a.at[gidx].set(jnp.ones(len(gidx), f32)) for a in out["RELSMC"]]
        new = dict(st)
        for k in ("ALBOLD SNEQVO TAH EAH FWET CANLIQ CANICE TV TG QSNOW SNDPTH SWE ZWT WA WT PLAI PSAI CM CH TAUSS").split():
            new[k] = out[k]
        for k in ("LFMASS", "RTMASS", "STMASS", "WOOD", "STBLCP", "FASTCP"):          # carbon pools (5.0 dveg 2, 5, 6)
            if k in out:
                new[k] = out[k]
        if o.get("crop", 0) > 0:
            for k in ("GRAIN", "GDD", "PGS"):
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
            "Swnet": out["FSA"], "Lwnet": -out["FIRA"],
            "Qle": lh, "Qh": out["FSH"], "Qg": out["SSOIL"], "Evap": qfx,
            "Qs": out["RUNSF"] * dt, "Qsb": out["RUNSB"] * dt, "RunsfRate": out["RUNSF"], "RunsbRate": out["RUNSB"],
            "AvgSurfT": fveg * out["TV"] + (f32(1.0) - fveg) * out["TGB"],
            "Albedo": jnp.where(valid_alb, sal * f32(100.0), 0.0), "Albedo_n": valid_alb.astype(f32),
            "SWE": new["SWE"], "SnowDepth": new["SNDPTH"],
            "CanopInt": jnp.where(new["CANLIQ"] < -9000.0, -9999.0, new["CANLIQ"] + new["CANICE"]), "Snowcover": out["FSNO"] * f32(1.0 if v50 else 100.0),
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
