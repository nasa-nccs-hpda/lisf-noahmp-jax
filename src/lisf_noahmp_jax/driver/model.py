"""One Noah-3.9 LIS time step for all tiles in JAX: noah39_main preparation (forcing floors, saturation humidity,
FFROZP, SFCDIF_OFF, KDT/FRZX), SFLX, persistent-parameter carry, and the instantaneous output diagnostics."""
import numpy as np
import jax, jax.numpy as jnp
from ..physics import noah39_jax as nj

f32 = jnp.float32
T0 = 273.15
LVH2O, LSUBS = 2.501e6, 2.83e6
RHOFW = 1000.0


def make_step(S, jit=True):
    """S: driver.params.Setup. Returns step(par, dyn, st, raw) -> (new_st, diag), vmapped over tiles (jitted if jit).
    Land-ice tiles (xice = -1) use SFLX_GLACIAL, run on the gathered subset S.glacier_idx."""
    sldpth = [f32(x) for x in S.sldpth]
    dt, zlvl, zlvl_wind = f32(S.dt), f32(S.zlvl), f32(S.zlvl_wind)
    opt, isurban = S.opt_thcnd, S.isurban
    refdk, frzk = f32(S.gen["refdk"]), f32(S.frzk)
    ua, sfcdif_opt, v36 = S.ua_phys, S.sfcdif_option, getattr(S, "v36", False)
    v33 = getattr(S, "v33", False)
    v32 = getattr(S, "model", "") == "Noah.3.2"          # Noah 3.2: wind speed scaled to the temperature height (ZFAC), SFCDIF with one height
    tab = nj.myj_tables() if sfcdif_opt == 1 else None
    aoasis = f32(1.0)
    gidx = np.asarray(S.glacier_idx, np.int32)

    def common(par, dyn, st, raw):
        sfctmp, qair, swd, lwraw = raw["tair"], raw["qair"], raw["swd"], raw["lwd"]
        sfcprs, prcp = raw["psurf"], raw["pcp"]
        emiss = st["emiss"]
        q1_in = jnp.where(st["q1"] < 0.0, qair, st["q1"])
        lwdn = lwraw * emiss
        sfcspd = jnp.sqrt(raw["u"] * raw["u"] + raw["v"] * raw["v"])
        ffrozp = jnp.where((prcp > 0.0) & (sfctmp < T0), 1.0, 0.0)
        sfcspd = jnp.where(sfcspd <= 0.01, 0.01, sfcspd)
        if v32 and float(S.zlvl) < float(S.zlvl_wind):
            zfac = jnp.log((zlvl + st["z0"]) / st["z0"]) / jnp.log((zlvl_wind + st["z0"]) / st["z0"])
            sfcspd = jnp.where(zfac * sfcspd <= 0.01, 0.01, zfac * sfcspd)
        q2 = jnp.where(qair < 0.1e-5, 0.1e-5, qair)
        a2, a3, a4, elwv, e0, rv, eps = 17.67, 273.15, 29.65, 2.501e6, 611.0, 461.0, 0.622
        es = e0 * jnp.exp(elwv / rv * (1.0 / a3 - 1.0 / sfctmp))
        q2sat = eps * es / (sfcprs - (1 - eps) * es)
        dqsdt2 = q2sat * (a2 * (a3 - a4)) / ((sfctmp - a4) * (sfctmp - a4))
        th2 = sfctmp + 0.0098 * zlvl
        t1v = st["t1"] * (1.0 + 0.61 * q2)
        th2v = th2 * (1.0 + 0.61 * q2)
        if sfcdif_opt == 0:
            cm, ch, ustar = nj.sfcdif_off(zlvl, zlvl if v32 else zlvl_wind, st["z0_old"], t1v, th2v, sfcspd, par["czil"], st["cm"], st["ch"])
            ribb = jnp.zeros_like(cm)
        else:
            cm, ch, ribb, ustar = nj.sfcdif_myj(zlvl, zlvl_wind, st["z0_old"], st["z0brd_old"], sfcprs, st["t1"], sfctmp,
                                               q1_in, q2, sfcspd, par["czil"], st["cm"], st["ch"], tab)
        forc = dict(sfctmp=sfctmp, q2=q2, soldn=swd, lwdn=lwdn, sfcprs=sfcprs, prcp=prcp, ffrozp=ffrozp, q2sat=q2sat,
                    dqsdt2=dqsdt2, th2=th2, solnet=swd * (1.0 - st["albedo"]), ch=ch, cm=cm, ribb=ribb)
        return forc

    def diagnostics(raw, ns_albedo, ns_emissi, t1, eta, sheat, ssoil, evp, runoff1, runoff2, sneqv, snowh, smc, stc):
        sfctmp, swd, lwraw, prcp = raw["tair"], raw["swd"], raw["lwd"], raw["pcp"]
        diag = {
            "Swnet": swd * (1.0 - ns_albedo),
            "Lwnet": -1.0 * (ns_emissi * (5.67e-8 * jnp.power(t1, 4.0)) - lwraw * ns_emissi),
            "Qle": eta, "Qh": sheat, "Qg": -ssoil,
            "Snowf": jnp.where(sfctmp < T0, prcp, 0.0), "Rainf": jnp.where(sfctmp < T0, 0.0, prcp),
            "Evap": evp, "Qs": runoff1 * RHOFW, "Qsb": runoff2 * RHOFW,
            "Albedo": ns_albedo, "SWE": sneqv * RHOFW, "SnowDepth": snowh,
            "Rainf_f": prcp, "Tair_f": sfctmp, "SWdown_f": swd, "LWdown_f": lwraw,
        }
        for k in range(4):
            diag[f"SoilMoist{k+1}"] = smc[k]
            diag[f"SoilTemp{k+1}"] = stc[k]
        return diag

    def land(par, dyn, st, raw):
        forc = common(par, dyn, st, raw)
        ch, cm = forc["ch"], forc["cm"]
        kdt = par["refkdt"] * par["dksat"] / refdk
        frzx = jnp.where(st["smcref"] == 0.0, 0.0, frzk * ((st["smcmax"] / st["smcref"]) * (0.412 / 0.468)))
        p = dict(par)
        p.update(shdfac=dyn["shdfac"], alb=dyn["alb"], z0brd=dyn["z0brd"], embrd=dyn["embrd"], kdt=kdt, frzx=frzx,
                 rsmin=st["rsmin"], smcmax=st["smcmax"], smcref=st["smcref"], smcwlt=st["smcwlt"], smcdry=st["smcdry"])
        s = {k: st[k] for k in ("cmc", "t1", "snowh", "sneqv", "albedo", "z0", "snotime1", "stc", "smc", "sh2o")}
        s["xlai"] = dyn["lai"]; s["emissi"] = st["emiss"]   # LIS runs dynsetup every step: LAI is reset each step
        ns, out = nj.sflx(forc, p, s, dt, zlvl, sldpth, opt_thcnd=opt, isurban=isurban, aoasis=aoasis, ua=ua, v36=v36, v33=v33)
        new = {"cmc": ns["cmc"], "t1": ns["t1"], "snowh": ns["snowh"], "sneqv": ns["sneqv"], "albedo": ns["albedo"],
               "ch": out["ch_o"], "cm": cm, "z0": ns["z0"], "snotime1": ns["snotime1"], "emiss": ns["emissi"],
               "stc": ns["stc"], "smc": ns["smc"], "sh2o": ns["sh2o"], "smcmax": out["smcmax_o"],
               "smcref": out["smcref_o"], "smcwlt": out["smcwlt_o"], "smcdry": out["smcdry_o"], "rsmin": out["rsmin_o"],
               "q1": ns["q1"], "xlai": dyn["lai"], "z0brd_old": dyn["z0brd"], "sca": ns["sncovr"], "z0_old": ns["z0"]}
        diag = diagnostics(raw, ns["albedo"], ns["emissi"], ns["t1"], out["eta"], out["sheat"], out["ssoil"], out["evp"],
                           out["runoff1"], out["runoff2"], ns["sneqv"], ns["snowh"], ns["smc"], ns["stc"])
        return new, diag

    def glacier(par, dyn, st, raw):
        forc = common(par, dyn, st, raw)
        ch, cm = forc["ch"], forc["cm"]
        p = {"alb": dyn["alb"], "snoalb": par["snoalb"], "tbot": par["tbot"], "z0brd": dyn["z0brd"],
             "embrd": dyn["embrd"], "lvcoef": par["lvcoef"]}
        s = {"t1": st["t1"], "stc": st["stc"], "snowh": st["snowh"], "sneqv": st["sneqv"], "snotime1": st["snotime1"],
             "sncovr": st["sca"]}
        ns, out = nj.sflx_glacial(forc, p, s, dt, sldpth)
        one_ = jnp.ones_like(st["stc"]); z = jnp.zeros_like(ns["t1"])
        new = {"cmc": st["cmc"], "t1": ns["t1"], "snowh": ns["snowh"], "sneqv": ns["sneqv"], "albedo": ns["albedo"],
               "ch": ch, "cm": cm, "z0": ns["z0"], "snotime1": ns["snotime1"], "emiss": ns["emissi"],
               "stc": ns["stc"], "smc": one_, "sh2o": one_, "smcmax": st["smcmax"], "smcref": st["smcref"],
               "smcwlt": st["smcwlt"], "smcdry": st["smcdry"], "rsmin": st["rsmin"], "q1": ns["q1"],
               "xlai": dyn["lai"], "z0brd_old": dyn["z0brd"], "sca": ns["sncovr"], "z0_old": st["z0_old"]}   # LIS does not update z0_old on land-ice tiles
        diag = diagnostics(raw, ns["albedo"], ns["emissi"], ns["t1"], out["eta"], out["sheat"], out["ssoil"], z,
                           out["runoff1"], z, ns["sneqv"], ns["snowh"], one_, ns["stc"])
        return new, diag

    ax_st = {k: (1 if k in ("stc", "smc", "sh2o") else 0) for k in
             ("cmc t1 snowh sneqv albedo ch cm z0 snotime1 emiss stc smc sh2o smcmax smcref smcwlt smcdry rsmin "
              "q1 xlai z0brd_old sca z0_old").split()}
    vland = jax.vmap(land, in_axes=(0, 0, ax_st, 0), out_axes=(ax_st, 0))
    vglac = jax.vmap(glacier, in_axes=(0, 0, ax_st, 0), out_axes=(ax_st, 0))

    def step(par, dyn, st, raw):
        new, diag = vland(par, dyn, st, raw)
        if len(gidx):
            take = lambda tree, axes=None: jax.tree_util.tree_map(lambda a: a[..., gidx] if a.ndim > 1 and False else a[gidx], tree)
            tk = lambda d: {k: v[gidx] for k, v in d.items()}
            tkst = {k: (v[:, gidx] if ax_st[k] == 1 else v[gidx]) for k, v in st.items()}
            ng, dg = vglac(tk(par), tk(dyn), tkst, tk(raw))
            new = {k: (new[k].at[:, gidx].set(ng[k]) if ax_st[k] == 1 else new[k].at[gidx].set(ng[k])) for k in new}
            diag = {k: diag[k].at[gidx].set(dg[k]) for k in diag}
        return new, diag

    return jax.jit(step) if jit else step


def make_chunk(S):
    """Jitted function running K model steps on device: chunk(par, dynK, st, rawK) -> (st, sum_of_diags).
    dynK / rawK have a leading time axis of length K (per-step dynamic parameters and forcing)."""
    if getattr(S, "v271", False):
        from . import model271
        v = model271.make_step(S, jit=False)
    else:
        v = make_step(S, jit=False)

    def chunk(par, dynK, st, rawK):
        def body(carry, x):
            st, acc = carry
            dyn, raw = x
            st, diag = v(par, dyn, st, raw)
            acc = {k: acc[k] + diag[k] for k in acc}
            return (st, acc), None
        d0 = jax.eval_shape(lambda: v(par, jax.tree_util.tree_map(lambda a: a[0], dynK), st,
                                       jax.tree_util.tree_map(lambda a: a[0], rawK))[1])
        acc0 = {k: jnp.zeros(a.shape, a.dtype) for k, a in d0.items()}
        (st, acc), _ = jax.lax.scan(body, (st, acc0), (dynK, rawK))
        return st, acc
    return jax.jit(chunk)
