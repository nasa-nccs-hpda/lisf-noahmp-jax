"""One Noah-2.7.1 LIS time step for all tiles in JAX: noah271_main preparation (forcing floors, saturation humidity, wind-height
scaling, DQSDT, SFCDIF, FFROZP, snow depth fix), SFLX, and the instantaneous output diagnostics."""
import numpy as np
import jax, jax.numpy as jnp
from ..physics import noah271_jax as n2

f32 = jnp.float32
RHOFW = 1000.0
TKFRZ, LATVAP = 273.16, 2.501e6


def esat_fn(t):
    """noah271_main FUNCTION E (uses LIS_CONST_LATVAP and LIS_CONST_TKFRZ)."""
    cpv, rv, cw, eso = 1870.0, 461.5, 4187.0, 611.2
    lw = LATVAP - (cw - cpv) * (t - TKFRZ)
    return eso * jnp.exp(lw * (1.0 / TKFRZ - 1.0 / t) / rv)


def dqsdt_fn(t, prs):
    cpv, rv, cw, eps, eso = 1870.0, 461.5, 4187.0, 0.622, 611.2
    lw = LATVAP - (cw - cpv) * (t - TKFRZ)
    es = eso * jnp.exp(lw * (1.0 / TKFRZ - 1.0 / t) / rv)
    desdt = lw * es / (rv * t * t)
    dqs = eps * desdt
    return jnp.where((t >= 173.0) & (t <= 373.0), dqs / prs, 0.0)


def make_step(S, jit=True):
    sldpth = [f32(x) for x in S.sldpth]
    dt, zlvl, zlvl_wind = f32(S.dt), f32(S.zlvl), f32(S.zlvl_wind)
    c1mq = np.float32(1.0) - np.float32(0.622)          # (1.-0.622) folded at single precision by the compiler

    def land(par, dyn, st, raw):
        sfctmp, q2 = raw["tair"], raw["qair"]
        soldn, lwdn = raw["swd"], raw["lwd"] * par["emissi"]
        sfcprs, prcp = raw["psurf"], raw["pcp"]
        uwind, vwind = raw["u"] * raw["u"], raw["v"] * raw["v"]
        sfcspd = jnp.sqrt(uwind + vwind)
        sfcspd = jnp.where(sfcspd <= 0.01, 0.01, sfcspd)
        q2 = jnp.where(q2 < 0.1e-5, 0.1e-5, q2)
        esat = esat_fn(sfctmp)
        q2sat = 0.622 * esat / (sfcprs - c1mq * esat)
        q2 = jnp.where(q2 >= q2sat, q2sat * 0.99, q2)
        z0 = par["z0"]
        if S.zlvl < S.zlvl_wind:
            zfac = jnp.log((zlvl + z0) / z0) / jnp.log((zlvl_wind + z0) / z0)
            sfcspd = zfac * sfcspd
            sfcspd = jnp.where(sfcspd <= 0.01, 0.01, sfcspd)
        dqsdt2 = dqsdt_fn(sfctmp, sfcprs)
        th2 = sfctmp + 0.0098 * zlvl
        t1v = st["t1"] * (1.0 + 0.61 * q2)
        th2v = th2 * (1.0 + 0.61 * q2)
        cm, ch, ustar = n2.sfcdif(zlvl, z0, t1v, th2v, sfcspd, par["czil"], st["cm"], st["ch"])
        ffrozp = jnp.where((prcp > 0.0) & (sfctmp <= 273.16), 1.0, 0.0)
        sneqv, snowh = st["sneqv"], st["snowh"]
        snowh = jnp.where(sneqv == 0.0, 0.0, jnp.where(snowh == 0.0, sneqv * 10.0, snowh))
        forc = dict(sfctmp=sfctmp, q2=q2, soldn=soldn, lwdn=lwdn, sfcprs=sfcprs, prcp=prcp, ffrozp=ffrozp, q2sat=q2sat,
                    dqsdt2=dqsdt2, th2=th2, ch=ch)
        p = dict(par)
        p.update(shdfac=dyn["shdfac"], alb=dyn["alb"])
        s = {k: st[k] for k in ("cmc", "t1", "stc", "smc", "sh2o", "sneqv")}
        s["snowh"] = snowh
        ns, out = n2.sflx(forc, p, s, dt, sldpth)
        new = {"cmc": ns["cmc"], "t1": ns["t1"], "stc": ns["stc"], "smc": ns["smc"], "sh2o": ns["sh2o"],
               "snowh": jnp.where(ns["sneqv"] > 0.0, ns["snowh"], 0.0), "sneqv": ns["sneqv"], "ch": ch, "cm": cm}
        t1 = ns["t1"]
        diag = {
            "Swnet": soldn * (1.0 - out["albedo"]),
            "Lwnet": -1.0 * (5.67e-8 * jnp.power(t1, 4.0) * out["emissi_out"] - lwdn),
            "Qle": out["eta"], "Qh": out["sheat"], "Qg": -out["gflx"],
            "Snowf": jnp.where(sfctmp <= TKFRZ, prcp, 0.0), "Rainf": jnp.where(sfctmp <= TKFRZ, 0.0, prcp),
            "Evap": out["evp"], "Qs": out["ro1"] * RHOFW, "Qsb": out["ro2"] * RHOFW,
            "Albedo": out["albedo"], "SWE": ns["sneqv"] * RHOFW, "SnowDepth": ns["snowh"],
            "Rainf_f": raw["pcp_f"], "Tair_f": sfctmp, "SWdown_f": soldn, "LWdown_f": raw["lwd"],
        }
        for k in range(4):
            diag[f"SoilMoist{k+1}"] = ns["smc"][k]
            diag[f"SoilTemp{k+1}"] = ns["stc"][k]
        return new, diag

    ax_st = {k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in "cmc t1 stc smc sh2o snowh sneqv ch cm".split()}
    vland = jax.vmap(land, in_axes=(0, 0, ax_st, 0), out_axes=(ax_st, 0))
    return jax.jit(vland) if jit else vland
