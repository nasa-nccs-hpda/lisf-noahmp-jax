"""JAX port of the LISF Noah-MP 4.0.1 column physics (NOAHMP_SFLX and the routines it calls).

Source: lis/surfacemodels/land/noahmp.4.0.1/phys/module_sf_noahmplsm_401.F90 (unmodified, read-only).
Scope (reduced, see jax-port/LOG.md 2026-10-22): the option set of the USAF 557WW Noah-MP configuration
  dveg=4 (table LAI, FVEG=shdmax), crs=1 (Ball-Berry), btr=1, run=1 (SIMGM), sfc=1 (M-O), frz=1, inf=1,
  rad=1|2|3, alb=1|2, snf=1|2|3, tbot=1|2, stc=1, rsf=1|2|3|4, soil=1 (dominant texture), crop=0|1 (dynamic crop model), urban=0|flag,
  IST=1 (land), ICE=0 (no glacier branch, no lake). 4 soil layers, 3 snow layers.
Not ported: GECROS, Jarvis canopy resistance, SFCDIF2,
  runoff options 2-5, frz/inf option 2, WRF-Hydro/PARFLOW hooks, glacier and lake branches, the error checks.

Style: one function per Fortran routine on a single column with scalars. Snow+soil layers are Python lists of
7 scalars, position p = layer index + 2 (p 0..2 = snow layers -2..0, p 3..6 = soil layers 1..4); ISNOW is a
traced int in {0,-1,-2,-3}; layer p is an active snow layer when p >= ISNOW + 3. Data-dependent IFs are
jnp.where selects. float32 throughout like the Fortran build. The step is vmap-ed over tiles by the caller.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

W = jnp.where
f32 = jnp.float32

GRAV = 9.80616
SB = 5.67e-08
VKC = 0.40
TFRZ = 273.16
HSUB = 2.8440e06
HVAP = 2.5104e06
HFUS = 0.3336e06
CWAT = 4.188e06
CICE = 2.094e06
CPAIR = 1004.64
TKWAT = 0.6
TKICE = 2.2
RAIR = 287.04
RW = 461.269
DENH2O = 1000.
DENICE = 917.
MPE = 1.0e-6
PSIWLT = -150.
Z0_BARE = 0.002
NSOIL = 4
NSNOW = 3
NL = 7
NITERC = 20
NITERG = 5
NITERB = 5


def at(lst, idx):
    """lst[idx] for a traced integer index."""
    return jnp.stack(lst)[idx]


def sel_write(lst, idx, val, cond=True):
    """returns a copy of lst with lst[idx] = val where cond (idx traced)."""
    return [W((p == idx) & cond, val, lst[p]) for p in range(len(lst))]


# ----------------------------------------------------------------------------------------------
def esat(t):
    a = (6.107799961, 4.436518521e-01, 1.428945805e-02, 2.650648471e-04, 3.031240396e-06, 2.034080948e-08, 6.136820929e-11)
    b = (6.109177956, 5.034698970e-01, 1.886013408e-02, 4.176223716e-04, 5.824720280e-06, 4.838803174e-08, 1.838826904e-10)
    c = (4.438099984e-01, 2.857002636e-02, 7.938054040e-04, 1.215215065e-05, 1.036561403e-07, 3.532421810e-10, -7.090244804e-13)
    d = (5.030305237e-01, 3.773255020e-02, 1.267995369e-03, 2.477563108e-05, 3.005693132e-07, 2.158542548e-09, 7.131097725e-12)

    def poly(k):
        return 100. * (k[0] + t * (k[1] + t * (k[2] + t * (k[3] + t * (k[4] + t * (k[5] + t * k[6]))))))
    return poly(a), poly(b), poly(c), poly(d)


def tdc(t):
    return jnp.fmin(50., jnp.fmax(-50., t - TFRZ))


def calhum(sfctmp, sfcprs):
    a2, a3, a4, elwv, e0, rv, eps = 17.67, 273.15, 29.65, 2.501e6, 0.611, 461.0, 0.622
    a23m4 = a2 * (a3 - a4)
    es = e0 * jnp.exp(elwv / rv * (1. / a3 - 1. / sfctmp))
    sfcprsx = sfcprs * 1.e-3
    q2sat = eps * es / (sfcprsx - es)
    q2sat = q2sat * 1.e3
    dqsdt2 = (q2sat / (1 + q2sat)) * a23m4 / (sfctmp - a4) ** 2
    return q2sat / 1.e3, dqsdt2


# ----------------------------------------------------------------------------------------------
def fpice_wetbulb(sfctmp, sfcprs, eair):
    """rain-snow partition option 5 (wet bulb temperature, Wang et al. 2019); eair = vapor pressure at reference height"""
    tdc_ = jnp.fmin(50.0, jnp.fmax(-50.0, sfctmp - TFRZ))
    lat = W(sfctmp > TFRZ, HVAP, HSUB)
    psy = CPAIR * sfcprs / (0.622 * lat)
    tw = tdc_ - 5.0
    for _ in range(10):
        es = 610.8 * jnp.exp((17.27 * tw) / (237.3 + tw))
        tw = tw - (es - eair) / psy
    return W(sfctmp >= TFRZ + 8., 0., 1.0 / (1.0 + 6.99e-5 * jnp.exp(2.0 * (tw + 3.97))))


def atm(P, o, sfcprs, sfctmp, q2, prcpconv, prcpnonc, prcpshcv, prcpsnow, prcpgrpl, prcphail, soldn, cosz):
    pair = sfcprs
    thair = sfctmp * jnp.power(sfcprs / pair, RAIR / CPAIR)
    qair = q2
    eair = qair * sfcprs / (0.622 + 0.378 * qair)
    rhoair = (sfcprs - 0.378 * eair) / (RAIR * sfctmp)
    swdown = W(cosz <= 0., 0., soldn)
    solad = [swdown * 0.7 * 0.5, swdown * 0.7 * 0.5]
    solai = [swdown * 0.3 * 0.5, swdown * 0.3 * 0.5]
    prcp = prcpconv + prcpnonc + prcpshcv
    if o["snf"] == 4:
        qprecc = prcpconv + prcpshcv
        qprecl = prcpnonc
    else:
        qprecc = 0.10 * prcp
        qprecl = 0.90 * prcp
    fp = W(qprecc + qprecl > 0., (qprecc + qprecl) / (10. * qprecc + qprecl), 0.)
    fpice = f32(0.)
    if o["snf"] == 1:
        fpice = W(sfctmp > TFRZ + P["T_ulimit"], 0.,
                  W(sfctmp <= TFRZ + P["T_llimit"], 1.0,
                    W(sfctmp <= TFRZ + P["T_mlimit"], 1. - (-54.632 + 0.2 * sfctmp), 0.6)))
    elif o["snf"] == 2:
        fpice = W(sfctmp >= TFRZ + 2.2, 0., 1.0)
    elif o["snf"] == 3:
        fpice = W(sfctmp >= TFRZ, 0., 1.0)
    elif o["snf"] == 5:
        fpice = fpice_wetbulb(sfctmp, sfcprs, eair)
    bdfall = jnp.fmin(120., 67.92 + 51.25 * jnp.exp((sfctmp - TFRZ) / 2.59))
    if o["snf"] == 4:
        prcp_frozen = prcpsnow + prcpgrpl + prcphail
        ok = (prcpnonc > 0.) & (prcp_frozen > 0.)
        pf = jnp.fmax(0.0, jnp.fmin(1.0, prcp_frozen / jnp.fmax(prcpnonc, 1e-30)))
        bd = bdfall * (prcpsnow / jnp.fmax(prcp_frozen, 1e-30)) + 500.0 * (prcpgrpl / jnp.fmax(prcp_frozen, 1e-30)) \
            + 917.0 * (prcphail / jnp.fmax(prcp_frozen, 1e-30))
        fpice = W(ok, pf, 0.0)
        bdfall = W(ok, bd, bdfall)
    rain = prcp * (1. - fpice)
    snow = prcp * fpice * P["snowf_scalef"]
    prcp = rain + snow
    return dict(thair=thair, qair=qair, eair=eair, rhoair=rhoair, swdown=swdown, solad=solad, solai=solai,
                qprecc=qprecc, qprecl=qprecl, fp=fp, fpice=fpice, bdfall=bdfall, rain=rain, snow=snow, prcp=prcp)


def phenology(P, o, vegtyp, snowh, tv, lat, yearlen, julian, lai, sai, pgs=None):
    """PhenologyMain. Crop tiles (crop model on and CropType > 0, P["CROPTYPE"]) keep the LAI/SAI state and have no minimum-area cleaning; their
    growing season follows the crop growth stage. returns lai, sai, elai, esai, igs"""
    crop = (P["CROPTYPE"] > 0.5) if (o.get("crop", 0) > 0) else jnp.bool_(False)
    lai0, sai0 = lai, sai
    yl = f32(yearlen)
    day = W(lat >= 0., julian, jnp.mod(julian + (0.5 * yl), yl))
    t = 12. * day / yl
    it1 = jnp.floor(t + 0.5).astype(jnp.int32)
    it2 = it1 + 1
    wt1 = (it1.astype(f32) + 0.5) - t
    wt2 = 1. - wt1
    it1 = W(it1 < 1, 12, it1)
    it2 = W(it2 > 12, 1, it2)
    if o["dveg"] in (1, 3, 4):
        lai = wt1 * P["LAIM"][it1 - 1] + wt2 * P["LAIM"][it2 - 1]
        sai = wt1 * P["SAIM"][it1 - 1] + wt2 * P["SAIM"][it2 - 1]
    if o["dveg"] in (7, 8, 9):
        sai = jnp.fmax(0.05, 0.1 * lai)
        sai = W(lai < 0.05, 0.0, sai)
    sai = W(sai < 0.05, 0.0, sai)
    lai = W((lai < 0.05) | (sai == 0.0), 0.0, lai)
    kill = (vegtyp == P["ISWATER"]) | (vegtyp == P["ISBARREN"]) | (vegtyp == P["ISICE"]) | (P["URBAN_FLAG"] > 0.5)
    lai = W(kill, 0., lai)
    sai = W(kill, 0., sai)
    if o.get("crop", 0) > 0:
        lai = W(crop, lai0, lai)
        sai = W(crop, sai0, sai)
    db = jnp.fmin(jnp.fmax(snowh - P["HVB"], 0.), P["HVT"] - P["HVB"])
    fb = db / jnp.fmax(1.e-06, P["HVT"] - P["HVB"])
    snowhc = P["HVT"] * jnp.exp(-snowh / 0.2)
    fb2 = W(snowhc > 1.e-06, jnp.fmin(snowh, snowhc) / jnp.fmax(snowhc, 1e-30), 1.0)
    fb = W((P["HVT"] > 0.) & (P["HVT"] <= 1.0), fb2, fb)
    elai = lai * (1. - fb)
    esai = sai * (1. - fb)
    esai_c = W(esai < 0.05, 0.0, esai)
    elai_c = W((elai < 0.05) | (esai_c == 0.0), 0.0, elai)
    igs = W(tv > P["TMIN"], 1., 0.)
    if o.get("crop", 0) > 0:
        esai, elai = W(crop, esai, esai_c), W(crop, elai, elai_c)
        igs = W(crop, W((pgs > 2) & (pgs < 7), 1., 0.), igs)
    else:
        esai, elai = esai_c, elai_c
    return lai, sai, elai, esai, igs


def precip_heat(P, vegtyp, dt, uu, vv, elai, esai, fveg, bdfall, rain, snow, fp, canliq, canice, tv, sfctmp, tg):
    maxliq = fveg * P["CH2OP"] * (elai + esai)         # 4.1/5.0: canopy capacities are scaled by the vegetation fraction
    vai = elai + esai
    has = vai > 0.
    qintr = fveg * rain * fp
    qintr = jnp.fmin(qintr, (maxliq - canliq) / dt * (1. - jnp.exp(-rain * dt / jnp.fmax(maxliq, 1e-30))))
    qintr = jnp.fmax(qintr, 0.)
    qdripr = fveg * rain - qintr
    qthror = (1. - fveg) * rain
    canliq_a = jnp.fmax(0., canliq + qintr * dt)
    # no canopy: buried
    qdripr_b = W(canliq > 0., canliq / dt, 0.)
    canliq_b = W(canliq > 0., 0.0, canliq)
    qintr = W(has, qintr, 0.)
    qdripr = W(has, qdripr, qdripr_b)
    qthror = W(has, qthror, rain)
    canliq = W(has, canliq_a, canliq_b)
    pah_ac = fveg * rain * (CWAT / 1000.0) * (sfctmp - tv)
    pah_cg = qdripr * (CWAT / 1000.0) * (tv - tg)
    pah_ag = qthror * (CWAT / 1000.0) * (sfctmp - tg)
    maxsno = fveg * 6.6 * (0.27 + 46. / bdfall) * (elai + esai)
    qints = fveg * snow * fp
    qints = jnp.fmin(qints, (maxsno - canice) / dt * (1. - jnp.exp(-snow * dt / jnp.fmax(maxsno, 1e-30))))
    qints = jnp.fmax(qints, 0.)
    ft = jnp.fmax(0.0, (tv - 270.15) / 1.87E5)
    fv = jnp.sqrt(uu * uu + vv * vv) / 1.56E5
    icedrip = jnp.fmax(0., canice) * (fv + ft)
    icedrip = jnp.fmin(canice / dt + qints, icedrip)   # 4.x: constraint to keep the water balance
    qdrips = (fveg * snow - qints) + icedrip
    qthros = (1.0 - fveg) * snow
    canice_a = jnp.fmax(0., canice + (qints - icedrip) * dt)
    qdrips_b = W(canice > 0., canice / dt, 0.)
    canice_b = W(canice > 0., 0.0, canice)
    qints = W(has, qints, 0.)
    qdrips = W(has, qdrips, qdrips_b)
    qthros = W(has, qthros, snow)
    canice = W(has, canice_a, canice_b)
    fwet = W(canice > 0., jnp.fmax(0., canice) / jnp.fmax(maxsno, 1.e-06),
             jnp.fmax(0., canliq) / jnp.fmax(maxliq, 1.e-06))
    fwet = jnp.power(jnp.fmin(fwet, 1.), 0.667)
    cmc = canliq + canice
    pah_ac = pah_ac + fveg * snow * (CICE / 1000.0) * (sfctmp - tv)
    pah_cg = pah_cg + qdrips * (CICE / 1000.0) * (tv - tg)
    pah_ag = pah_ag + qthros * (CICE / 1000.0) * (sfctmp - tg)
    pahv = pah_ac - pah_cg
    pahg = pah_cg
    pahb = pah_ag
    mid = (fveg > 0.0) & (fveg < 1.0)
    lo = fveg <= 0.0
    pahg2 = W(mid, pahg / jnp.fmax(fveg, 1e-30), W(lo, 0.0, pahg))
    pahb2 = W(mid, pahb / jnp.fmax(1.0 - fveg, 1e-30), W(lo, pahg + pahb, W(fveg >= 1.0, 0.0, pahb)))
    pahv2 = W(lo, 0.0, pahv)
    pahv = jnp.fmin(jnp.fmax(pahv2, -20.0), 20.0)
    pahg = jnp.fmin(jnp.fmax(pahg2, -20.0), 20.0)
    pahb = jnp.fmin(jnp.fmax(pahb2, -20.0), 20.0)
    qrain = qdripr + qthror
    qsnow = qdrips + qthros
    snowhin = qsnow / bdfall
    return dict(canliq=canliq, canice=canice, fwet=fwet, cmc=cmc, pahv=pahv, pahg=pahg, pahb=pahb, qrain=qrain,
                qsnow=qsnow, snowhin=snowhin)


# ----------------------------------------------------------------------------------------------
def tdfcnd(P, k, smc, sh2o):
    smcmax = P["SMCMAX"][k]
    quartz = P["QUARTZ"][k]
    satratio = smc / smcmax
    thkw, thko, thkqtz = 0.57, 2.0, 7.7
    thks = jnp.power(thkqtz, quartz) * jnp.power(thko, 1. - quartz)
    xunfroz = W(smc > 0., sh2o / jnp.fmax(smc, 1e-30), 1.0)
    xu = xunfroz * smcmax
    thksat = jnp.power(thks, 1. - smcmax) * jnp.power(TKICE, smcmax - xu) * jnp.power(thkw, xu)
    gammd = (1. - smcmax) * 2700.
    thkdry = (0.135 * gammd + 64.7) / (2700. - 0.947 * gammd)
    ake = W((sh2o + 0.0005) < smc, satratio,
            W(satratio > 0.1, jnp.log10(jnp.fmax(satratio, 1e-30)) + 1.0, 0.0))
    return ake * (thksat - thkdry) + thkdry


def thermoprop(P, isnow, dt, snowh, snice, sniq, dz, smc, sh2o, tksno=1):
    """returns df[7], hcpct[7], fact[7]; snice/sniq length 3 (snow layers), dz length 7"""
    df = [None] * NL
    hc = [None] * NL
    fact = [None] * NL
    for p in range(3):
        a = isnow + 3 <= p
        dzs = W(a, dz[p], 1.0)
        snicev = jnp.fmin(1., snice[p] / (dzs * DENICE))
        epore = 1. - snicev
        snliqv = jnp.fmin(epore, sniq[p] / (dzs * DENH2O))
        bdsnoi = (snice[p] + sniq[p]) / dzs
        hc[p] = W(a, CICE * snicev + CWAT * snliqv, 0.)
        if tksno == 1:
            k_ = 3.2217E-6 * bdsnoi ** 2.                            # Stieglitz (Yen 1965)
        elif tksno == 2:
            k_ = 2e-2 + 2.5e-6 * bdsnoi * bdsnoi                     # Anderson 1976
        elif tksno == 3:
            k_ = 0.35 + 0. * bdsnoi                                  # constant
        elif tksno == 4:
            k_ = 2.576e-6 * bdsnoi ** 2. + 0.074                     # Verseghy 1991
        else:
            k_ = 2.22 * (bdsnoi / 1000.0) ** 1.88                    # Douville (Yen 1981)
        df[p] = W(a, k_, 0.)
    for k in range(NSOIL):
        p = k + 3
        sice = smc[k] - sh2o[k]
        hc[p] = sh2o[k] * CWAT + (1.0 - P["SMCMAX"][k]) * P["CSOIL"] + (P["SMCMAX"][k] - smc[k]) * CPAIR + sice * CICE
        d = tdfcnd(P, k, smc[k], sh2o[k])
        df[p] = W(P["URBAN_FLAG"] > 0.5, 3.24, d)
    for p in range(NL):
        a = isnow + 3 <= p
        fact[p] = W(a, dt / (hc[p] * W(a, dz[p], 1.0)), 0.)
    # top-layer conductivity adjustment
    df3_0 = (df[3] * dz[3] + 0.35 * snowh) / (snowh + dz[3])
    dfl0 = df[2]
    dzl0 = dz[2]
    df3_1 = (df[3] * dz[3] + dfl0 * dzl0) / (dzl0 + dz[3])
    df[3] = W(isnow == 0, df3_0, df3_1)
    return df, hc, fact


# ----------------------------------------------------------------------------------------------
def twostream(P, o, ib, ic, cosz, vai, fwet, t, albgrd, albgri, rho, tau, fveg):
    """returns fab, fre, ftd, fti, frev, freg, gdir, bgap, wgap for waveband ib (0/1), ic 0 (direct) or 1 (diffuse)"""
    PAI = 3.14159265
    bgap = f32(0.)
    wgap = f32(0.)
    noveg = vai == 0.0
    if o["rad"] == 1:
        denfveg = -jnp.log(jnp.fmax(1.0 - fveg, 0.01)) / (PAI * P["RC"] ** 2)
        hd = P["HVT"] - P["HVB"]
        bb = 0.5 * hd
        thetap = jnp.arctan(bb / P["RC"] * jnp.tan(jnp.arccos(jnp.fmax(0.01, cosz))))
        bgap1 = jnp.exp(-denfveg * PAI * P["RC"] ** 2 / jnp.cos(thetap))
        fa = vai / (1.33 * PAI * jnp.power(P["RC"], 3.0) * (bb / P["RC"]) * denfveg)
        newvai = hd * fa
        wgap1 = (1.0 - bgap1) * jnp.exp(-0.5 * newvai / cosz)
        gap1 = jnp.fmin(1.0 - fveg, bgap1 + wgap1)
        gap = W(noveg, 1.0, gap1)
        kopen = W(noveg, 1.0, 0.05)
        bgap = W(noveg, 0., bgap1)
        wgap = W(noveg, 0., wgap1)
    elif o["rad"] == 2:
        gap = W(noveg, 1.0, 0.0)
        kopen = W(noveg, 1.0, 0.0)
    else:
        gap = W(noveg, 1.0, 1.0 - fveg)
        kopen = W(noveg, 1.0, 1.0 - fveg)
    coszi = jnp.fmax(0.001, cosz)
    chil = jnp.fmin(jnp.fmax(P["XL"], -0.4), 0.6)
    chil = W(jnp.abs(chil) <= 0.01, 0.01, chil)
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil * chil
    phi2 = 0.877 * (1. - 2. * phi1)
    gdir = phi1 + phi2 * coszi
    ext = gdir / coszi
    avmu = (1. - phi1 / phi2 * jnp.log((phi1 + phi2) / phi1)) / phi2
    omegal = rho[ib] + tau[ib]
    tmp0 = gdir + phi2 * coszi
    tmp1 = phi1 * coszi
    asu = 0.5 * omegal * gdir / tmp0 * (1. - tmp1 / tmp0 * jnp.log((tmp1 + tmp0) / tmp1))
    betadl = (1. + avmu * ext) / (omegal * avmu * ext) * asu
    betail = 0.5 * (rho[ib] + tau[ib] + (rho[ib] - tau[ib]) * ((1. + chil) / 2.) ** 2) / omegal
    warm = t > TFRZ
    t0 = (1. - fwet) * omegal + fwet * P["OMEGAS"][ib]
    t1 = ((1. - fwet) * omegal * betadl + fwet * P["OMEGAS"][ib] * P["BETADS"]) / t0
    t2 = ((1. - fwet) * omegal * betail + fwet * P["OMEGAS"][ib] * P["BETAIS"]) / t0
    omega = W(warm, omegal, t0)
    betad = W(warm, betadl, t1)
    betai = W(warm, betail, t2)
    b = 1. - omega + omega * betai
    c = omega * betai
    tmp0 = avmu * ext
    d = tmp0 * omega * betad
    f = tmp0 * omega * (1. - betad)
    tmp1 = b * b - c * c
    h = jnp.sqrt(tmp1) / avmu
    sigma = tmp0 * tmp0 - tmp1
    sigma = W(jnp.abs(sigma) < 1.e-6, W(sigma >= 0., 1.e-6, -1.e-6), sigma)
    p1 = b + avmu * h
    p2 = b - avmu * h
    p3 = b + tmp0
    p4 = b - tmp0
    s1 = jnp.exp(-h * vai)
    s2 = jnp.exp(-ext * vai)
    alb = albgrd[ib] if ic == 0 else albgri[ib]
    u1 = b - c / alb
    u2 = b - c * alb
    u3 = f + c * alb
    tmp2 = u1 - avmu * h
    tmp3 = u1 + avmu * h
    d1 = p1 * tmp2 / s1 - p2 * tmp3 * s1
    tmp4 = u2 + avmu * h
    tmp5 = u2 - avmu * h
    d2 = tmp4 / s1 - tmp5 * s1
    h1 = -d * p4 - c * f
    tmp6 = d - h1 * p3 / sigma
    tmp7 = (d - c - h1 / sigma * (u1 + tmp0)) * s2
    h2 = (tmp6 * tmp2 / s1 - p2 * tmp7) / d1
    h3 = -(tmp6 * tmp3 * s1 - p1 * tmp7) / d1
    h4 = -f * p3 - c * d
    tmp8 = h4 / sigma
    tmp9 = (u3 - tmp8 * (u2 - tmp0)) * s2
    h5 = -(tmp8 * tmp4 / s1 + tmp9) / d2
    h6 = (tmp8 * tmp5 * s1 + tmp9) / d2
    h7 = (c * tmp2) / (d1 * s1)
    h8 = (-c * tmp3 * s1) / d1
    h9 = tmp4 / (d2 * s1)
    h10 = (-tmp5 * s1) / d2
    if ic == 0:
        ftds = s2 * (1.0 - gap) + gap
        ftis = (h4 * s2 / sigma + h5 * s1 + h6 / s1) * (1.0 - gap)
        fres = (h1 / sigma + h2 + h3) * (1.0 - gap) + albgrd[ib] * gap
        freveg = (h1 / sigma + h2 + h3) * (1.0 - gap)
        frebar = albgrd[ib] * gap
    else:
        ftds = f32(0.)
        ftis = (h9 * s1 + h10 / s1) * (1.0 - kopen) + kopen
        fres = (h7 + h8) * (1.0 - kopen) + albgri[ib] * kopen
        freveg = (h7 + h8) * (1.0 - kopen) + albgri[ib] * kopen
        frebar = f32(0.)
    fab = 1. - fres - (1. - albgrd[ib]) * ftds - (1. - albgri[ib]) * ftis
    return fab, fres, ftds, ftis, freveg, frebar, gdir, bgap, wgap


def albedo(P, o, dt, cosz, elai, esai, tg, tv, fsno, fwet, smc1, sneqvo, sneqv, qsnow, fveg, albold, tauss):
    vai = elai + esai
    wl = elai / jnp.fmax(vai, MPE)
    ws = esai / jnp.fmax(vai, MPE)
    rho = [jnp.fmax(P["RHOL"][ib] * wl + P["RHOS"][ib] * ws, MPE) for ib in range(2)]
    tau = [jnp.fmax(P["TAUL"][ib] * wl + P["TAUS"][ib] * ws, MPE) for ib in range(2)]
    # snow age
    dela0 = 1.E-6 * dt
    arg = 5.E3 * (1. / TFRZ - 1. / tg)
    age1 = jnp.exp(arg)
    age2 = jnp.exp(jnp.fmin(0., 10. * arg))
    tage = age1 + age2 + 0.3
    dela = dela0 * tage
    dels = jnp.fmax(0.0, sneqv - sneqvo) / P["SWEMX"]
    sge = (tauss + dela) * (1.0 - dels)
    tauss_n = W(sneqv <= 0.0, 0., W(sneqv > 800., 0., jnp.fmax(0., sge)))
    fage = tauss_n / (tauss_n + 1.)
    albold_n = albold
    if o["alb"] == 1:
        sl = 2.0
        sl1 = 1. / sl
        sl2 = 2. * sl
        cf1 = ((1. + sl1) / (1. + sl2 * cosz) - sl1)
        fzen = jnp.fmax(cf1, 0.)
        albsni = [0.95 * (1. - 0.2 * fage), 0.65 * (1. - 0.5 * fage)]
        albsnd = [albsni[0] + 0.4 * fzen * (1. - albsni[0]), albsni[1] + 0.4 * fzen * (1. - albsni[1])]
    else:
        alb = P["MNSNALB"] + (albold - P["MNSNALB"]) * jnp.exp(-P["sndecayexp"] * dt / 3600.)
        alb = W(qsnow > 0., alb + jnp.fmin(qsnow, P["SWEMX"] / dt) * (P["MXSNALB"] - alb) / (P["SWEMX"] / dt), alb)
        albsni = [alb, alb]
        albsnd = [alb, alb]
        albold_n = alb
    inc = jnp.fmax(0.11 - 0.40 * smc1, 0.)
    albgrd = []
    albgri = []
    for ib in range(2):
        albsod = jnp.fmin(P["ALBSAT"][ib] + inc, P["ALBDRY"][ib])
        albsoi = albsod
        albgrd.append(albsod * (1. - fsno) + albsnd[ib] * fsno)
        albgri.append(albsoi * (1. - fsno) + albsni[ib] * fsno)
    fabd = [0, 0]; fabi = [0, 0]; ftdd = [0, 0]; ftid = [0, 0]; ftii = [0, 0]
    albd = [0, 0]; albi = [0, 0]; frevd = [0, 0]; fregd = [0, 0]; frevi = [0, 0]; fregi = [0, 0]
    bgap = f32(0.); wgap = f32(0.)
    gdir = f32(0.)
    for ib in range(2):
        fa, fr, ftd, fti, frv, frg, gdir, bg, wg = twostream(P, o, ib, 0, cosz, vai, fwet, tv, albgrd, albgri, rho, tau, fveg)
        fabd[ib], albd[ib], ftdd[ib], ftid[ib], frevd[ib], fregd[ib] = fa, fr, ftd, fti, frv, frg
        bgap, wgap = bg, wg
        fa, fr, ftd, fti, frv, frg, gdir, bg, wg = twostream(P, o, ib, 1, cosz, vai, fwet, tv, albgrd, albgri, rho, tau, fveg)
        fabi[ib], albi[ib], ftii[ib], frevi[ib], fregi[ib] = fa, fr, fti, frv, frg
        bgap, wgap = bg, wg
    ext = gdir / cosz * jnp.sqrt(1. - rho[0] - tau[0])
    fsun = (1. - jnp.exp(-ext * vai)) / jnp.fmax(ext * vai, MPE)
    fsun = W(fsun < 0.01, 0., fsun)
    zero = f32(0.)
    pos = cosz > 0.
    def Z(x):
        return W(pos, x, zero)
    return dict(albgrd=[Z(x) for x in albgrd], albgri=[Z(x) for x in albgri], albd=[Z(x) for x in albd],
                albi=[Z(x) for x in albi], fabd=[Z(x) for x in fabd], fabi=[Z(x) for x in fabi],
                ftdd=[Z(x) for x in ftdd], ftid=[Z(x) for x in ftid], ftii=[Z(x) for x in ftii],
                frevd=[Z(x) for x in frevd], frevi=[Z(x) for x in frevi], fregd=[Z(x) for x in fregd],
                fregi=[Z(x) for x in fregi], fsun=Z(fsun), bgap=Z(bgap), wgap=Z(wgap),
                tauss=W(pos, tauss_n, tauss), albold=W(pos, albold_n, albold))


def radiation(P, o, dt, cosz, tg, tv, fsno, qsnow, fwet, elai, esai, smc1, solad, solai, fveg, albold, tauss,
              sneqvo, sneqv):
    A = albedo(P, o, dt, cosz, elai, esai, tg, tv, fsno, fwet, smc1, sneqvo, sneqv, qsnow, fveg, albold, tauss)
    fsun = A["fsun"]
    fsha = 1. - fsun
    laisun = elai * fsun
    laisha = elai * fsha
    vai = elai + esai
    sag = f32(0.); sav = f32(0.); fsa = f32(0.)
    cad = []; cai = []
    for ib in range(2):
        cad.append(solad[ib] * A["fabd"][ib])
        cai.append(solai[ib] * A["fabi"][ib])
        sav = sav + cad[ib] + cai[ib]
        fsa = fsa + cad[ib] + cai[ib]
        trd = solad[ib] * A["ftdd"][ib]
        tri = solad[ib] * A["ftid"][ib] + solai[ib] * A["ftii"][ib]
        ab = trd * (1. - A["albgrd"][ib]) + tri * (1. - A["albgri"][ib])
        sag = sag + ab
        fsa = fsa + ab
    laifra = elai / jnp.fmax(vai, MPE)
    parsun = W(fsun > 0., (cad[0] + fsun * cai[0]) * laifra / jnp.fmax(laisun, MPE), 0.)
    parsha = W(fsun > 0., (fsha * cai[0]) * laifra / jnp.fmax(laisha, MPE),
               (cad[0] + cai[0]) * laifra / jnp.fmax(laisha, MPE))
    rvis = A["albd"][0] * solad[0] + A["albi"][0] * solai[0]
    rnir = A["albd"][1] * solad[1] + A["albi"][1] * solai[1]
    fsr = rvis + rnir
    return dict(fsun=fsun, laisun=laisun, laisha=laisha, parsun=parsun, parsha=parsha, sav=sav, sag=sag, fsr=fsr,
                fsa=fsa, bgap=A["bgap"], wgap=A["wgap"], tauss=A["tauss"], albold=A["albold"])


# ----------------------------------------------------------------------------------------------
def sfcdif2(P, it, z0, thz0, thlm, sfcspd, zlm, akms, akhs, rlmo, wstar2, ustar_prev):
    """Noah-MP SFCDIF2 (Chen et al. 1997, OPT_SFC = 2; ILECH = 0). akms/akhs come in and go out multiplied by the wind speed
    (the caller divides by UR); ustar is INTENT(OUT) but only set at ITER == 1 and in the final block, so the caller's
    previous FV is what the first block of later iterations sees."""
    c = np.float32
    wwst2 = c(1.2) * c(1.2)
    vkrm, excm = c(0.40), c(0.001)
    btg = c(1.0) / c(270.0) * c(GRAV)
    elfc = vkrm * btg
    wold = c(0.15); wnew = c(1.0) - wold
    pihf = c(3.14159265) / c(2.)
    epsu2, epsust, ztmin, ztmax, hpbl, sqvisc = c(1.e-4), c(0.07), c(-5.0), c(1.0), c(1000.0), c(258.2)
    zilfc = -P["CZIL"] * vkrm * sqvisc
    zu = z0
    cxch = excm / zlm
    dthv = thlm - thz0
    du2 = jnp.fmax(sfcspd * sfcspd, epsu2)
    btgh = btg * hpbl
    first = it == 1
    wst = W(btgh * akhs * dthv != 0.0, wwst2 * jnp.power(jnp.abs(btgh * akhs * dthv), 2. / 3.), 0.)
    ustar1 = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wst)), epsust)
    rlmo1 = elfc * akhs * dthv / ustar1 ** 3
    wstar2 = W(first, wst, wstar2)
    ustar = W(first, ustar1, ustar_prev)
    rlmo = W(first, rlmo1, rlmo)
    zt = jnp.fmax(1.e-6, jnp.exp(zilfc * jnp.sqrt(ustar * z0)) * z0)
    zslu = zlm + zu
    zslt = zlm + zt
    rlogu = jnp.log(zslu / zu)
    rlogt = jnp.log(zslt / zt)
    zetalt = jnp.fmax(zslt * rlmo, ztmin)
    rlmo = zetalt / zslt
    zetalu = zslu * rlmo
    zetau = zu * rlmo
    zetat = zt * rlmo
    neg = rlmo < 0.
    pspmu = lambda xx: -2. * jnp.log((xx + 1.) * 0.5) - jnp.log((xx * xx + 1.) * 0.5) + 2. * jnp.arctan(xx) - pihf
    psphu = lambda xx: -2. * jnp.log((xx * xx + 1.) * 0.5)
    q4 = lambda z_: jnp.sqrt(jnp.sqrt(1. - 16. * z_))
    xlu, xlt, xu, xt = q4(zetalu), q4(zetalt), q4(zetau), q4(zetat)
    simm_u = pspmu(xlu) - pspmu(xu) + rlogu
    simh_u = psphu(xlt) - psphu(xt) + rlogt
    zetalu_s = jnp.fmin(zetalu, ztmax)
    zetalt_s = jnp.fmin(zetalt, ztmax)
    zetau_s = jnp.fmin(zetau, ztmax / (zslu / zu))
    zetat_s = jnp.fmin(zetat, ztmax / (zslt / zt))
    simm_s = 5. * zetalu_s - 5. * zetau_s + rlogu
    simh_s = 5. * zetalt_s - 5. * zetat_s + rlogt
    simm = W(neg, simm_u, simm_s)
    simh = W(neg, simh_u, simh_s)
    ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
    ustark = ustar * vkrm
    simm = W(simm < 1.e-6, 1.e-6, simm)
    akms = jnp.fmax(ustark / simm, cxch)
    simh = W(simh < 1.e-6, 1.e-6, simh)
    akhs = jnp.fmax(ustark / simh, cxch)
    wstar2 = W(btgh * akhs * dthv != 0.0, wwst2 * jnp.power(jnp.abs(btgh * akhs * dthv), 2. / 3.), 0.)
    rlmn = elfc * akhs * dthv / ustar ** 3
    rlmo = rlmo * wold + rlmn * wnew
    return ustar, akms, akhs, rlmo, wstar2


def sfcdif1(it, sfctmp, rhoair, h, qair, zlvl, zpd, z0m, z0h, ur, moz, mozsgn, fm, fh, fm2, fh2, fv_prev):
    mozold = moz
    tmpcm = jnp.log((zlvl - zpd) / z0m)
    tmpch = jnp.log((zlvl - zpd) / z0h)
    tmpcm2 = jnp.log((2.0 + z0m) / z0m)
    tmpch2 = jnp.log((2.0 + z0h) / z0h)
    first = it == 1
    tvir = (1. + 0.61 * qair) * sfctmp
    tmp1 = VKC * (GRAV / tvir) * h / (rhoair * CPAIR)
    tmp1 = W(jnp.abs(tmp1) <= MPE, MPE, tmp1)
    mol = -1. * fv_prev ** 3 / tmp1
    moz_n = jnp.fmin((zlvl - zpd) / mol, 1.)
    moz2_n = jnp.fmin((2.0 + z0h) / mol, 1.)
    moz = W(first, 0.0, moz_n)
    moz2 = W(first, 0.0, moz2_n)
    mozsgn = mozsgn + W(mozold * moz < 0., 1, 0)
    z2 = mozsgn >= 2
    moz = W(z2, 0., moz)
    fm = W(z2, 0., fm)
    fh = W(z2, 0., fh)
    moz2 = W(z2, 0., moz2)
    fm2 = W(z2, 0., fm2)
    fh2 = W(z2, 0., fh2)
    neg = moz < 0.
    t1 = (1. - 16. * moz) ** 0.25
    t2 = jnp.log((1. + t1 * t1) / 2.)
    t3 = jnp.log((1. + t1) / 2.)
    fmnew = W(neg, 2. * t3 + t2 - 2. * jnp.arctan(t1) + 1.5707963, -5. * moz)
    fhnew = W(neg, 2 * t2, -5. * moz)
    neg2 = moz2 < 0.
    t12 = (1. - 16. * moz2) ** 0.25
    t22 = jnp.log((1. + t12 * t12) / 2.)
    t32 = jnp.log((1. + t12) / 2.)
    fm2new = W(neg2, 2. * t32 + t22 - 2. * jnp.arctan(t12) + 1.5707963, -5. * moz2)
    fh2new = W(neg2, 2 * t22, -5. * moz2)
    fm = W(first, fmnew, 0.5 * (fm + fmnew))
    fh = W(first, fhnew, 0.5 * (fh + fhnew))
    fm2 = W(first, fm2new, 0.5 * (fm2 + fm2new))
    fh2 = W(first, fh2new, 0.5 * (fh2 + fh2new))
    fh = jnp.fmin(fh, 0.9 * tmpch)
    fm = jnp.fmin(fm, 0.9 * tmpcm)
    fh2 = jnp.fmin(fh2, 0.9 * tmpch2)
    fm2 = jnp.fmin(fm2, 0.9 * tmpcm2)
    cmfm = tmpcm - fm
    chfh = tmpch - fh
    cm2fm2 = tmpcm2 - fm2
    ch2fh2 = tmpch2 - fh2
    cmfm = W(jnp.abs(cmfm) <= MPE, MPE, cmfm)
    chfh = W(jnp.abs(chfh) <= MPE, MPE, chfh)
    cm2fm2 = W(jnp.abs(cm2fm2) <= MPE, MPE, cm2fm2)
    ch2fh2 = W(jnp.abs(ch2fh2) <= MPE, MPE, ch2fh2)
    cm = VKC * VKC / (cmfm * cmfm)
    ch = VKC * VKC / (cmfm * chfh)
    fv = ur * jnp.sqrt(cm)
    ch2 = VKC * fv / ch2fh2
    return moz, mozsgn, fm, fh, fm2, fh2, cm, ch, fv, ch2


def ragrb(P, it, vai, rhoair, hg, tah, zpd, z0mg, z0hg, hcan, uc, z0h, fv, cwp, fhg_prev):
    first = it == 1
    tmp1 = VKC * (GRAV / tah) * hg / (rhoair * CPAIR)
    tmp1 = W(jnp.abs(tmp1) <= MPE, MPE, tmp1)
    molg = -1. * fv ** 3 / tmp1
    mozg = W(first, 0., jnp.fmin((zpd - z0mg) / molg, 1.))
    fhgnew = W(mozg < 0., (1. - 15. * mozg) ** (-0.25), 1. + 4.7 * mozg)
    fhg = W(first, fhgnew, 0.5 * (fhg_prev + fhgnew))
    cwpc = jnp.sqrt(cwp * vai * hcan * fhg)
    t1 = jnp.exp(-cwpc * z0hg / hcan)
    t2 = jnp.exp(-cwpc * (z0h + zpd) / hcan)
    tmprah2 = hcan * jnp.exp(cwpc) / cwpc * (t1 - t2)
    kh = jnp.fmax(VKC * fv * (hcan - zpd), MPE)
    rahg = tmprah2 / kh
    tmprb = cwpc * 50. / (1. - jnp.exp(-cwpc / 2.))
    rb = tmprb * jnp.sqrt(P["DLEAF"] / uc)
    rb = jnp.fmin(jnp.fmax(rb, 5.0), 50.0)           # 4.3: limit the bulk leaf boundary-layer resistance to 5-50 s/m
    return rahg, rahg, rb, fhg


def frh2o_mp(P, k, tkelv, smc, sh2o):
    """Noah-MP FRH2O (Flerchinger, supercooled liquid water; OPT_FRZ = 2): Noah-MP constants GRAV, HFUS, TFRZ."""
    ck, blim, error = 8.0, 5.5, 0.005
    bx = W(P["BEXP"][k] > blim, blim, P["BEXP"][k])
    swl0 = smc - sh2o
    swl0 = W(swl0 > (smc - 0.02), smc - 0.02, swl0)
    swl0 = W(swl0 < 0.0, 0.0, swl0)

    def cond(st):
        return (st[1] < 10) & (st[2] == 0)

    def body(st):
        swl, nlog, kcount = st
        nlog = nlog + 1
        df = jnp.log((P["PSISAT"][k] * GRAV / HFUS) * jnp.power(1.0 + ck * swl, 2.0)
                     * jnp.power(P["SMCMAX"][k] / (smc - swl), bx)) - jnp.log(-(tkelv - TFRZ) / tkelv)
        denom = 2.0 * ck / (1.0 + ck * swl) + bx / (smc - swl)
        swlk = swl - df / denom
        swlk = W(swlk > (smc - 0.02), smc - 0.02, swlk)
        swlk = W(swlk < 0.0, 0.0, swlk)
        dswl = jnp.abs(swlk - swl)
        kcount = W(dswl <= error, kcount + 1, kcount)
        return swlk, nlog, kcount

    swl, nlog, kcount = jax.lax.while_loop(cond, body, (swl0, jnp.int32(0), jnp.int32(0)))
    free = smc - swl
    fk = jnp.power(((HFUS / (GRAV * (-P["PSISAT"][k]))) * ((tkelv - TFRZ) / tkelv)), -1.0 / bx) * P["SMCMAX"][k]
    fk = W(fk < 0.02, 0.02, fk)
    free = W(kcount == 0, jnp.fmin(fk, smc), free)
    return W(tkelv > (TFRZ - 1.0e-3), smc, free)


def canres(P, par, sfctmp, rcsoil, eah, sfcprs):
    """Noah-MP CANRES (Jarvis, OPT_CRS = 2): canopy resistance per unit LAI; PSN is not computed (-999.99)."""
    q2 = 0.622 * eah / (sfcprs - 0.378 * eah)
    q2 = q2 / (1.0 - q2)                               # 5.0: specific humidity -> mixing ratio (4.0.1: q/(1+q))
    q2sat, _ = calhum(sfctmp, sfcprs)
    ff = 2.0 * par / P["RGL"]
    rcs = (ff + P["RSMIN"] / P["RSMAX"]) / (1.0 + ff)
    rcs = jnp.fmax(rcs, 0.0001)
    rct = 1.0 - 0.0016 * jnp.power(P["TOPT"] - sfctmp, 2.0)
    rct = jnp.fmax(rct, 0.0001)
    rcq = 1.0 / (1.0 + P["HS"] * jnp.fmax(0., q2sat - q2))
    rcq = jnp.fmax(rcq, 0.01)
    rc = P["RSMIN"] / (rcs * rct * rcq * rcsoil)
    return rc, f32(-999.99) + 0. * rc


def stomata(P, apar, foln, tv, ei, ea, sfctmp, sfcprs, o2, co2, igs, btran, rb):
    niter = 3
    cf = sfcprs / (8.314 * sfctmp) * 1.e06
    rs0 = 1. / P["BP"] * cf
    fnf = jnp.fmin(foln / jnp.fmax(MPE, P["FOLNMX"]), 1.0)
    tc = tv - TFRZ
    ppf = 4.6 * apar
    j = ppf * P["QE25"]
    f1 = lambda ab, bc: jnp.power(ab, (bc - 25.) / 10.)
    f2 = lambda ab: 1. + jnp.exp((-2.2E05 + 710. * (ab + 273.16)) / (8.314 * (ab + 273.16)))
    kc = P["KC25"] * f1(P["AKC"], tc)
    ko = P["KO25"] * f1(P["AKO"], tc)
    awc = kc * (1. + o2 / ko)
    cp = 0.5 * kc / ko * o2 * 0.21
    vcmx = P["VCMX25"] / f2(tc) * fnf * btran * f1(P["AVCMX"], tc)
    c3 = P["C3PSN"]
    ci = 0.7 * co2 * c3 + 0.4 * co2 * (1. - c3)
    rlb = rb / cf
    cea = jnp.fmax(0.25 * ei * c3 + 0.40 * ei * (1. - c3), jnp.fmin(ea, ei))
    rs = rs0
    psn = f32(0.)
    for _ in range(niter):
        wj = jnp.fmax(ci - cp, 0.) * j / (ci + 2. * cp) * c3 + j * (1. - c3)
        wc = jnp.fmax(ci - cp, 0.) * vcmx / (ci + awc) * c3 + vcmx * (1. - c3)
        we = 0.5 * vcmx * c3 + 4000. * vcmx * ci / sfcprs * (1. - c3)
        psn = jnp.fmin(jnp.fmin(wj, wc), we) * igs
        cs = jnp.fmax(co2 - 1.37 * rlb * sfcprs * psn, MPE)
        a = P["MP"] * psn * sfcprs * cea / (cs * ei) + P["BP"]
        b = (P["MP"] * psn * sfcprs / cs + P["BP"]) * rlb - 1.
        c = -rlb
        q = W(b >= 0., -0.5 * (b + jnp.sqrt(b * b - 4. * a * c)), -0.5 * (b - jnp.sqrt(b * b - 4. * a * c)))
        r1 = q / a
        r2 = c / q
        rs = jnp.fmax(r1, r2)
        ci = jnp.fmax(cs - psn * sfcprs * 1.65 * rs, 0.)
    rs = rs * cf
    ok = apar > 0.
    return W(ok, rs, rs0), W(ok, psn, 0.)


# ----------------------------------------------------------------------------------------------
def _esat_pair(t):
    tc = tdc(t)
    esw, esi, desw, desi = esat(tc)
    liq = tc > 0.
    return W(liq, esw, esi), W(liq, desw, desi)


def vege_flux(P, o, c, pet=False):
    """Newton-Raphson solution for vegetation (tv) and ground (tg) temperatures. c: dict of inputs."""
    g = lambda k: c[k]
    fveg = g("fveg"); vai = g("vai"); ur = g("ur"); sfctmp = g("sfctmp"); rhoair = g("rhoair")
    zlvl, zpd, z0m = g("zlvl"), g("zpd"), g("z0m")
    z0mg, emv, emg, lwdn = g("z0mg"), g("emv"), g("emg"), g("lwdn")
    rsurf = g("rsurf_pet") if pet else g("rsurf")
    rhsur = g("rhsur_pet") if pet else g("rhsur")
    vaie = jnp.fmin(6., vai)                          # 4.1: LAI is stand LAI and is not rescaled by FVEG
    laisune = jnp.fmin(6., g("laisun"))
    laishae = jnp.fmin(6., g("laisha"))
    estg0, _ = _esat_pair(g("tg"))
    hcan = P["HVT"]
    # 4.4/4.5: canopy heat storage in the vegetation temperature solution (CBIOM from the table)
    hcapc = P["CBIOM"] * vaie * CWAT + g("canliq") * CWAT / DENH2O + g("canice") * CICE / DENICE
    uc = ur * jnp.log((hcan - zpd + z0m) / z0m) / jnp.log(zlvl / z0m)
    tg0 = g("tg")
    air = -emv * (1. + (1. - emv) * (1. - emg)) * lwdn - emv * emg * SB * tg0 ** 4
    cir = (2. - emv * (1. - emg)) * emv * SB
    qsfc0 = 0.622 * g("eair") / (g("psfc") - 0.378 * g("eair"))
    z = f32(0.)
    carry = dict(it=jnp.int32(1), fin=jnp.bool_(False), liter=jnp.int32(0),
                 tah=g("tah"), eah=g("eah"), tv=g("tv"), moz=z, mozsgn=jnp.int32(0), fm=z, fh=z, fm2=z, fh2=z, wstar=z,
                 fv=f32(0.1), cm=g("cm"), ch=g("ch"), h=z, hg=z, qsfc=qsfc0, fhg=z,
                 rssun=z, rssha=z, psnsun=z, psnsha=z, irc=z, shc=z, evc=z, tr=z, dtv=z,
                 rahc=f32(1.), rahg=f32(1.), rawg=f32(1.), cah=f32(1.), cvh=f32(1.), rb=f32(100.))
    sav, sag, pahv = g("sav"), g("sag"), g("pahv")

    def cond(s):
        return (~s["fin"]) & (s["it"] <= NITERC)

    def body(s):
        it = s["it"]
        if o["sfc"] == 2:
            fv, cm, ch, moz, wstar = sfcdif2(P, it, z0m, s["tah"], g("thair"), ur, zlvl, s["cm"], s["ch"], s["moz"],
                                             s["wstar"], s["fv"])
            cm = cm / ur; ch = ch / ur
            mozsgn, fm, fh, fm2, fh2 = s["mozsgn"], s["fm"], s["fh"], s["fm2"], s["fh2"]
        else:
            moz, mozsgn, fm, fh, fm2, fh2, cm, ch, fv, ch2 = sfcdif1(
                it, sfctmp, rhoair, s["h"], g("qair"), zlvl, zpd, z0m, z0m, ur, s["moz"], s["mozsgn"],
                s["fm"], s["fh"], s["fm2"], s["fh2"], s["fv"])
            wstar = s["wstar"]
        ramc = jnp.fmax(1., 1. / (cm * ur))
        rahc = jnp.fmax(1., 1. / (ch * ur))
        rawc = rahc
        ramg, rahg, rb, fhg = ragrb(P, it, vaie, rhoair, s["hg"], s["tah"], zpd, z0mg, z0mg, hcan, uc, z0m, fv,
                                    g("cwp"), s["fhg"])
        rawg = rahg
        estv, destv = _esat_pair(s["tv"])
        if o["crs"] == 2:
            fvmin = jnp.fmax(fveg, 1.0e-6)                       # 5.0: absorbed PAR per unit vegetated area
            rs_su, ps_su = canres(P, g("parsun") / fvmin, s["tv"], g("btran"), s["eah"], g("sfcprs"))
            rs_sh, ps_sh = canres(P, g("parsha") / fvmin, s["tv"], g("btran"), s["eah"], g("sfcprs"))
        else:
            fvmin = jnp.fmax(fveg, 1.0e-6)                       # 4.1: absorbed PAR per unit vegetated area
            rs_su, ps_su = stomata(P, g("parsun") / fvmin, g("foln"), s["tv"], estv, s["eah"], sfctmp, g("sfcprs"), g("o2air"),
                                   g("co2air"), g("igs"), g("btran"), rb)
            rs_sh, ps_sh = stomata(P, g("parsha") / fvmin, g("foln"), s["tv"], estv, s["eah"], sfctmp, g("sfcprs"), g("o2air"),
                                   g("co2air"), g("igs"), g("btran"), rb)
        first = it == 1
        if pet and o["crs"] == 1:
            rs_su = P["RSMIN"] + 0. * rs_su
            rs_sh = P["RSMIN"] + 0. * rs_sh
        rssun = W(first, rs_su, s["rssun"])
        rssha = W(first, rs_sh, s["rssha"])
        psnsun = W(first, ps_su, s["psnsun"])
        psnsha = W(first, ps_sh, s["psnsha"])
        cah = 1. / rahc
        cvh = 2. * vaie / rb
        cgh = 1. / rahg
        cond_ = cah + cvh + cgh
        ata = (sfctmp * cah + tg0 * cgh) / cond_
        bta = cvh / cond_
        csh = (1. - bta) * rhoair * CPAIR * cvh
        caw = 1. / rawc
        cew = g("fwet") * vaie / rb
        ctw = (1. - g("fwet")) * (laisune / (rb + rssun) + laishae / (rb + rssha))
        cgw = 1. / (rawg + rsurf)
        cond2 = caw + cew + ctw + cgw
        aea = (g("eair") * caw + estg0 * cgw) / cond2
        bea = (cew + ctw) / cond2
        cev = (1. - bea) * cew * rhoair * CPAIR / g("gammav")
        ctr = (1. - bea) * ctw * rhoair * CPAIR / g("gammav")
        tah = ata + bta * s["tv"]
        eah = aea + bea * estv
        tv = s["tv"]
        irc = fveg * (air + cir * tv ** 4)
        shc = fveg * rhoair * CPAIR * cvh * (tv - tah)
        evc = fveg * rhoair * CPAIR * cew * (estv - eah) / g("gammav")
        tr = fveg * rhoair * CPAIR * ctw * (estv - eah) / g("gammav")
        evc = W(tv > TFRZ, jnp.fmin(g("canliq") * g("lathev") / g("dt"), evc),
                jnp.fmin(g("canice") * g("lathev") / g("dt"), evc))
        b = sav - irc - shc - evc - tr + pahv
        a = fveg * (4. * cir * tv ** 3 + csh + (cev + ctr) * destv + hcapc / g("dt"))
        dtv = b / a
        irc = irc + fveg * 4. * cir * tv ** 3 * dtv
        shc = shc + fveg * csh * dtv
        evc = evc + fveg * cev * destv * dtv
        tr = tr + fveg * ctr * destv * dtv
        tv = tv + dtv
        h = rhoair * CPAIR * (tah - sfctmp) / rahc
        hg = rhoair * CPAIR * (tg0 - tah) / rahg
        qsfc = (0.622 * eah) / (g("sfcprs") - 0.378 * eah)
        fin = s["liter"] == 1
        liter = W((it >= 5) & (jnp.abs(dtv) <= 0.01) & (s["liter"] == 0), 1, s["liter"])
        return dict(it=it + 1, fin=fin, liter=liter, tah=tah, eah=eah, tv=tv, moz=moz, mozsgn=mozsgn, fm=fm, fh=fh,
                    fm2=fm2, fh2=fh2, wstar=wstar, fv=fv, cm=cm, ch=ch, h=h, hg=hg, qsfc=qsfc, fhg=fhg, rssun=rssun,
                    rssha=rssha, psnsun=psnsun, psnsha=psnsha, irc=irc, shc=shc, evc=evc, tr=tr, dtv=dtv,
                    rahc=rahc, rahg=rahg, rawg=rawg, cah=cah, cvh=cvh, rb=rb)

    s = lax.while_loop(cond, body, carry)
    tv, tah, eah = s["tv"], s["tah"], s["eah"]
    # ground
    air2 = -emg * (1. - emv) * lwdn - emg * emv * SB * tv ** 4
    cir2 = emg * SB
    csh = rhoair * CPAIR / s["rahg"]
    cev = rhoair * CPAIR / (g("gammag") * (s["rawg"] + rsurf))
    cgh = 2. * g("df1") / g("dz1")
    tg = tg0
    irg = shg = evg = gh = f32(0.)
    estg = estg0
    for _ in range(NITERG):
        estg, destg = _esat_pair(tg)
        irg = cir2 * tg ** 4 + air2
        shg = csh * (tg - tah)
        evg = cev * (estg * rhsur - eah)
        gh = cgh * (tg - g("stc1"))
        b = sag - irg - shg - evg - gh + g("pahg")
        a = 4. * cir2 * tg ** 3 + csh + cev * destg + cgh
        dtg = b / a
        irg = irg + 4. * cir2 * tg ** 3 * dtg
        shg = shg + csh * dtg
        evg = evg + cev * destg * dtg
        gh = gh + cgh * dtg
        tg = tg + dtg
    adj = (g("snowh") > 0.05) & (tg > TFRZ) & (o["stc"] != 2)
    tg_a = f32(TFRZ) if o["stc"] == 1 else (1. - g("fsno")) * tg + g("fsno") * TFRZ      # OPT_STC 3: allow TG > 0 C during melt
    irg_a = cir2 * tg_a ** 4 - emg * (1. - emv) * lwdn - emg * emv * SB * tv ** 4
    shg_a = csh * (tg_a - tah)
    evg_a = cev * (estg * rhsur - eah)
    gh_a = sag + g("pahg") - (irg_a + shg_a + evg_a)
    tg, irg, shg, evg, gh = W(adj, tg_a, tg), W(adj, irg_a, irg), W(adj, shg_a, shg), W(adj, evg_a, evg), W(adj, gh_a, gh)
    # 2 m diagnostics
    cah2 = s["fv"] * VKC / (jnp.log((2. + z0m) / z0m) - s["fh2"])
    small = cah2 < 1.e-5
    t2mv = W(small, tah, tah - (shg + s["shc"] / fveg) / (rhoair * CPAIR) * 1. / cah2)
    q2v = W(small, s["qsfc"], s["qsfc"] - ((s["evc"] + s["tr"]) / fveg + evg) / (g("lathev") * rhoair) * 1. / cah2)
    rssun, rssha = s["rssun"], s["rssha"]
    return dict(tah=tah, eah=eah, tv=tv, tg=tg, cm=s["cm"], ch=s["cah"], irg=irg, irc=s["irc"], shg=shg, shc=s["shc"],
                evg=evg, evc=s["evc"], tr=s["tr"], ghv=gh, t2mv=t2mv, psnsun=s["psnsun"], psnsha=s["psnsha"],
                qsfc=s["qsfc"], q2v=q2v, chv2=cah2, chleaf=s["cvh"], chuc=1. / s["rahg"], rssun=rssun, rssha=rssha)


def bare_flux(P, o, c):
    g = lambda k: c[k]
    sfctmp, rhoair, ur, snowh = g("sfctmp"), g("rhoair"), g("ur"), g("snowh")
    zlvl, zpd, z0m, emg, lwdn = g("zlvl"), g("zpdg"), g("z0mg"), g("emg"), g("lwdn")
    z = f32(0.)
    cir = emg * SB
    cgh = 2. * g("df1") / g("dz1")
    tgb = g("tg")
    s = dict(moz=z, mozsgn=jnp.int32(0), fm=z, fh=z, fm2=z, fh2=z, wstar=z, fv=f32(0.1), cm=g("cm"), ch=g("ch"), h=z,
             qsfc=z, irb=z, shb=z, evb=z, ghb=z, csh=z, cev=z, estg=z, ehb=z)
    for it in range(1, NITERB + 1):
        if o["sfc"] == 2:
            fv, cm, ch, moz, wstar = sfcdif2(P, jnp.int32(it), z0m, tgb, g("thair"), ur, zlvl, s["cm"], s["ch"], s["moz"],
                                             s["wstar"], s["fv"])
            cm = cm / ur; ch = ch / ur
            cm = W(g("snowh") > 0., jnp.fmin(0.01, cm), cm)      # CM & CH are too large over snow (Fortran comment)
            ch = W(g("snowh") > 0., jnp.fmin(0.01, ch), ch)
            mozsgn, fm, fh, fm2, fh2 = s["mozsgn"], s["fm"], s["fh"], s["fm2"], s["fh2"]
        else:
            moz, mozsgn, fm, fh, fm2, fh2, cm, ch, fv, ch2 = sfcdif1(
                jnp.int32(it), sfctmp, rhoair, s["h"], g("qair"), zlvl, zpd, z0m, z0m, ur, s["moz"], s["mozsgn"],
                s["fm"], s["fh"], s["fm2"], s["fh2"], s["fv"])
            wstar = s["wstar"]
        ramb = jnp.fmax(1., 1. / (cm * ur))
        rahb = jnp.fmax(1., 1. / (ch * ur))
        rawb = rahb
        ehb = 1. / rahb
        estg, destg = _esat_pair(tgb)
        csh = rhoair * CPAIR / rahb
        cev = rhoair * CPAIR / g("gammag") / (g("rsurf") + rawb)
        irb = cir * tgb ** 4 - emg * lwdn
        shb = csh * (tgb - sfctmp)
        evb = cev * (estg * g("rhsur") - g("eair"))
        ghb = cgh * (tgb - g("stc1"))
        b = g("sag") - irb - shb - evb - ghb + g("pahb")
        a = 4. * cir * tgb ** 3 + csh + cev * destg + cgh
        dtg = b / a
        irb = irb + 4. * cir * tgb ** 3 * dtg
        shb = shb + csh * dtg
        evb = evb + cev * destg * dtg
        ghb = ghb + cgh * dtg
        tgb = tgb + dtg
        h = csh * (tgb - sfctmp)
        estg, _ = _esat_pair(tgb)
        qsfc = 0.622 * (estg * g("rhsur")) / (g("psfc") - 0.378 * (estg * g("rhsur")))
        s = dict(moz=moz, mozsgn=mozsgn, fm=fm, fh=fh, fm2=fm2, fh2=fh2, wstar=wstar, fv=fv, cm=cm, ch=ch, h=h, qsfc=qsfc, irb=irb,
                 shb=shb, evb=evb, ghb=ghb, csh=csh, cev=cev, estg=estg, ehb=ehb)
    irb, shb, evb, ghb = s["irb"], s["shb"], s["evb"], s["ghb"]
    adj = (snowh > 0.05) & (tgb > TFRZ) & (o["stc"] != 2)
    tg_a = f32(TFRZ) if o["stc"] == 1 else (1. - g("fsno")) * tgb + g("fsno") * TFRZ
    irb_a = cir * tg_a ** 4 - emg * lwdn
    shb_a = s["csh"] * (tg_a - sfctmp)
    evb_a = s["cev"] * (s["estg"] * g("rhsur") - g("eair"))
    ghb_a = g("sag") + g("pahb") - (irb_a + shb_a + evb_a)
    tgb, irb, shb, evb, ghb = W(adj, tg_a, tgb), W(adj, irb_a, irb), W(adj, shb_a, shb), W(adj, evb_a, evb), W(adj, ghb_a, ghb)
    ehb2 = s["fv"] * VKC / (jnp.log((2. + z0m) / z0m) - s["fh2"])
    small = ehb2 < 1.e-5
    t2mb = W(small, tgb, tgb - shb / (rhoair * CPAIR) * 1. / ehb2)
    q2b = W(small, s["qsfc"], s["qsfc"] - evb / (g("latheag") * rhoair) * (1. / ehb2 + g("rsurf")))
    q2b = W(P["URBAN_FLAG"] > 0.5, s["qsfc"], q2b)
    return dict(tgb=tgb, cm=s["cm"], ch=s["ehb"], irb=irb, shb=shb, evb=evb, ghb=ghb, t2mb=t2mb, qsfc=s["qsfc"],
                q2b=q2b, chb2=ehb2)


# ----------------------------------------------------------------------------------------------
def rosr12(a, b, c, d, pf):
    """tridiagonal solve on 7 positions; first active position pf (traced)."""
    n = len(a)
    p = [None] * n
    dl = [None] * n
    for k in range(n):
        first = pf == k
        if k == 0:
            p[k] = -c[k] / b[k]
            dl[k] = d[k] / b[k]
        else:
            den = 1.0 / (b[k] + a[k] * p[k - 1])
            p[k] = W(first, -c[k] / b[k], -c[k] * den)
            dl[k] = W(first, d[k] / b[k], (d[k] - a[k] * dl[k - 1]) * den)
    p[n - 1] = dl[n - 1]
    for kk in range(n - 2, -1, -1):
        p[kk] = p[kk] * p[kk + 1] + dl[kk]
    return p


def tsnosoi(P, o, isnow, tbot, zsnso, ssoil, df, hcpct, dt, snowh, stc):
    zbot = P["ZBOT"] - snowh
    pf = isnow + 3
    act = [pf <= p for p in range(NL)]
    denom = [None] * NL; ddz = [None] * NL; dtsdz = [None] * NL; eflux = [None] * NL
    for k in range(NL):
        first = pf == k
        if k < NL - 1:
            den_f = -zsnso[k] * hcpct[k]
            t1_f = -zsnso[k + 1]
            den_m = (zsnso[k - 1] - zsnso[k]) * hcpct[k] if k > 0 else den_f
            t1_m = (zsnso[k - 1] - zsnso[k + 1]) if k > 0 else t1_f
            denom[k] = W(first, den_f, den_m)
            temp1 = W(first, t1_f, t1_m)
            ddz[k] = 2.0 / temp1
            dtsdz[k] = 2.0 * (stc[k] - stc[k + 1]) / temp1
        else:
            denom[k] = (zsnso[k - 1] - zsnso[k]) * hcpct[k]
            if o["tbot"] == 2:
                dtsdz[k] = (stc[k] - tbot) / (0.5 * (zsnso[k - 1] + zsnso[k]) - zbot)
                botflx = -df[k] * dtsdz[k]
            else:
                dtsdz[k] = f32(0.)
                botflx = f32(0.)
    for k in range(NL):
        first = pf == k
        if k < NL - 1:
            ef = df[k] * dtsdz[k] - ssoil
            em = (df[k] * dtsdz[k] - df[k - 1] * dtsdz[k - 1]) if k > 0 else ef
            eflux[k] = W(first, ef, em)
        else:
            eflux[k] = (-botflx - df[k - 1] * dtsdz[k - 1])
    ai = [None] * NL; bi = [None] * NL; ci = [None] * NL; rhs = [None] * NL
    for k in range(NL):
        first = pf == k
        if k < NL - 1:
            c_f = -df[k] * ddz[k] / denom[k]
            a_m = -df[k - 1] * ddz[k - 1] / denom[k] if k > 0 else f32(0.)
            c_m = -df[k] * ddz[k] / denom[k]
            ai[k] = W(first, 0.0, a_m)
            ci[k] = c_f
            b_first = -c_f + df[k] / (0.5 * zsnso[k] * zsnso[k] * hcpct[k]) if o["stc"] == 2 else -c_f
            bi[k] = W(first, b_first, -(a_m + c_m))
        else:
            ai[k] = -df[k - 1] * ddz[k - 1] / denom[k]
            ci[k] = f32(0.)
            bi[k] = -(ai[k] + ci[k])
        rhs[k] = eflux[k] / (-denom[k])
    # identity rows for inactive layers
    A = [W(act[k], ai[k] * dt, 0.) for k in range(NL)]
    B = [W(act[k], 1. + bi[k] * dt, 1.) for k in range(NL)]
    C = [W(act[k], ci[k] * dt, 0.) for k in range(NL)]
    D = [W(act[k], rhs[k] * dt, 0.) for k in range(NL)]
    sol = rosr12(A, B, C, D, pf)
    return [W(act[k], stc[k] + sol[k], stc[k]) for k in range(NL)]


def phasechange(P, o, isnow, dt, fact, dz, hcpct, stc, snice, snliq, sneqv, snowh, smc, sh2o):
    """returns stc[7], snice[3], snliq[3], sneqv, snowh, smc[4], sh2o[4], qmelt, imelt[7], ponding"""
    pf = isnow + 3
    act = [pf <= p for p in range(NL)]
    mice = [None] * NL; mliq = [None] * NL; supercool = [f32(0.)] * NL
    for p in range(3):
        mice[p] = snice[p]; mliq[p] = snliq[p]
    for k in range(NSOIL):
        p = k + 3
        mliq[p] = sh2o[k] * dz[p] * 1000.
        mice[p] = (smc[k] - sh2o[k]) * dz[p] * 1000.
    wice0 = list(mice); wliq0 = list(mliq); wmass0 = [mice[p] + mliq[p] for p in range(NL)]
    for k in range(NSOIL):
        p = k + 3
        cold = stc[p] < TFRZ
        if o["frz"] == 1:
            smp = HFUS * (TFRZ - stc[p]) / (GRAV * stc[p])
            sc = P["SMCMAX"][k] * jnp.power(smp / P["PSISAT"][k], -1. / P["BEXP"][k])
            supercool[p] = W(cold, sc * dz[p] * 1000., 0.)
        else:
            supercool[p] = frh2o_mp(P, k, stc[p], smc[k], sh2o[k]) * dz[p] * 1000.
    imelt = [None] * NL
    for p in range(NL):
        im = jnp.int32(0)
        im = W((mice[p] > 0.) & (stc[p] >= TFRZ), 1, im)
        im = W((mliq[p] > supercool[p]) & (stc[p] < TFRZ), 2, im)
        if p == 3:
            im = W((isnow == 0) & (sneqv > 0.) & (stc[p] >= TFRZ), 1, im)
        imelt[p] = W(act[p], im, 0)
    hm = [None] * NL; xm = [None] * NL
    stc = list(stc)
    for p in range(NL):
        h = W(imelt[p] > 0, (stc[p] - TFRZ) / jnp.where(act[p], fact[p], 1.0), 0.)
        stc[p] = W(imelt[p] > 0, TFRZ, stc[p])
        bad1 = (imelt[p] == 1) & (h < 0.)
        bad2 = (imelt[p] == 2) & (h > 0.)
        h = W(bad1 | bad2, 0., h)
        imelt[p] = W(bad1 | bad2, 0, imelt[p])
        hm[p] = h
        xm[p] = h * dt / HFUS
    # surface snow melt special case
    sp = (isnow == 0) & (sneqv > 0.) & (xm[3] > 0.)
    temp1 = sneqv
    sneqv_n = jnp.fmax(0., temp1 - xm[3])
    propor = sneqv_n / jnp.fmax(temp1, 1e-30)
    snowh_n = jnp.fmax(0., propor * snowh)
    heatr = hm[3] - HFUS * (temp1 - sneqv_n) / dt
    xm3 = W(heatr > 0., heatr * dt / HFUS, 0.)
    hm3 = W(heatr > 0., heatr, 0.)
    qmelt = W(sp, jnp.fmax(0., (temp1 - sneqv_n)) / dt, 0.)
    xmf = W(sp, HFUS * qmelt, 0.)
    ponding = W(sp, temp1 - sneqv_n, 0.)
    sneqv = W(sp, sneqv_n, sneqv)
    snowh = W(sp, snowh_n, snowh)
    xm[3] = W(sp, xm3, xm[3])
    hm[3] = W(sp, hm3, hm[3])
    for p in range(NL):
        go = (imelt[p] > 0) & (jnp.abs(hm[p]) > 0.) & act[p]
        pos = xm[p] > 0.
        neg = xm[p] < 0.
        mi_pos = jnp.fmax(0., wice0[p] - xm[p])
        he_pos = hm[p] - HFUS * (wice0[p] - mi_pos) / dt
        if p <= 2:
            mi_neg = jnp.fmin(wmass0[p], wice0[p] - xm[p])
        else:
            mi_neg = W(wmass0[p] < supercool[p], 0.,
                       jnp.fmax(jnp.fmin(wmass0[p] - supercool[p], wice0[p] - xm[p]), 0.0))
        he_neg = hm[p] - HFUS * (wice0[p] - mi_neg) / dt
        mi = W(pos, mi_pos, W(neg, mi_neg, mice[p]))
        he = W(pos, he_pos, W(neg, he_neg, 0.))
        ml = jnp.fmax(0., wmass0[p] - mi)
        mice_p = W(go, mi, mice[p])
        mliq_p = W(go, ml, mliq[p])
        upd = go & (jnp.abs(he) > 0.)
        st = W(upd, stc[p] + fact[p] * he, stc[p])
        if p <= 2:
            st = W(upd & (ml * mi > 0.), TFRZ, st)
        stc[p] = st
        xmf = xmf + W(go, HFUS * (wice0[p] - mi) / dt, 0.)
        if p < 3:
            qmelt = qmelt + W(go, jnp.fmax(0., (wice0[p] - mi)) / dt, 0.)
        mice[p] = mice_p
        mliq[p] = mliq_p
    snliq_o = [mliq[p] for p in range(3)]
    snice_o = [mice[p] for p in range(3)]
    snliq_o = [W(act[p], snliq_o[p], snliq[p]) for p in range(3)]
    snice_o = [W(act[p], snice_o[p], snice[p]) for p in range(3)]
    sh2o_o = [mliq[k + 3] / (1000. * dz[k + 3]) for k in range(NSOIL)]
    smc_o = [(mliq[k + 3] + mice[k + 3]) / (1000. * dz[k + 3]) for k in range(NSOIL)]
    return stc, snice_o, snliq_o, sneqv, snowh, smc_o, sh2o_o, qmelt, imelt, ponding


# ----------------------------------------------------------------------------------------------
def _msel(cond, a, b):
    if isinstance(b, list):
        return [W(cond, x, y) for x, y in zip(a, b)]
    return W(cond, a, b)


def _dsel(cond, new, old):
    return {k: _msel(cond, new[k], old[k]) for k in old}


def combo(dz, wliq, wice, t, dz2, wliq2, wice2, t2):
    dzc = dz + dz2
    wicec = wice + wice2
    wliqc = wliq + wliq2
    h = (CICE * wice + CWAT * wliq) * (t - TFRZ) + HFUS * wliq
    h2 = (CICE * wice2 + CWAT * wliq2) * (t2 - TFRZ) + HFUS * wliq2
    hc = h + h2
    den = CICE * wicec + CWAT * wliqc
    tc = W(hc < 0., TFRZ + hc / den, W(hc <= HFUS * wliqc, TFRZ, TFRZ + (hc - HFUS * wliqc) / den))
    return dzc, wliqc, wicec, tc


def combine(P, sn, gla=False):
    """snow layer combination. sn: dict with isnow, dz[3], ice[3], liq[3], stc[3], sice1, sh2o1, dzs1, snowh, sneqv,
    ponding1, ponding2. Returns updated copy (all selects)."""
    sn = dict(sn)
    dz = list(sn["dz"]); ice = list(sn["ice"]); liq = list(sn["liq"]); stc = list(sn["stc"])
    isnow = sn["isnow"]
    sice1, sh2o1 = sn["sice1"], sn["sh2o1"]
    snowh, sneqv, pond1, pond2 = sn["snowh"], sn["sneqv"], sn["ponding1"], sn["ponding2"]
    isnow_old = isnow
    # part 1: merge thin ice layers, J = -2..0 (p 0..2)
    for p in range(3):
        j = p - 2
        a = (j >= isnow_old + 1)
        cond = a & (ice[p] <= 0.1)
        if p < 2:
            liq[p + 1] = W(cond, liq[p + 1] + liq[p], liq[p + 1])
            ice[p + 1] = W(cond, ice[p + 1] + ice[p], ice[p + 1])
        else:
            c1 = cond & (isnow_old < -1)
            liq[1] = W(c1, liq[1] + liq[2], liq[1])
            ice[1] = W(c1, ice[1] + ice[2], ice[1])
            c2 = cond & ~(isnow_old < -1)
            if gla:
                pond1 = W(c2, pond1 + liq[2], pond1)
                sneqv = W(c2, ice[2], sneqv)
                snowh = W(c2, dz[2], snowh)
            else:
                ge = ice[2] >= 0.
                c2a = c2 & ge
                c2b = c2 & ~ge
                pond1_b = liq[2] + ice[2]
                neg = pond1_b < 0.
                sice1_b = jnp.fmax(0.0, sice1 + pond1_b / (sn["dzs1"] * 1000.))
                pond1 = W(c2a, liq[2], W(c2b, W(neg, 0.0, pond1_b), pond1))
                sice1 = W(c2b & neg, sice1_b, sice1)
                sneqv = W(c2a, ice[2], W(c2b, 0.0, sneqv))
                snowh = W(c2a, dz[2], W(c2b, 0.0, snowh))
            liq[2] = W(c2, 0.0, liq[2])
            ice[2] = W(c2, 0.0, ice[2])
            dz[2] = W(c2, 0.0, dz[2])
        # shift
        shift = cond & (j > isnow + 1) & (isnow < -1)
        for i in range(p, 0, -1):
            ii = i - 2
            ok = shift & (ii >= isnow + 2)
            stc[i] = W(ok, stc[i - 1], stc[i])
            liq[i] = W(ok, liq[i - 1], liq[i])
            ice[i] = W(ok, ice[i - 1], ice[i])
            dz[i] = W(ok, dz[i - 1], dz[i])
        isnow = isnow + W(cond, 1, 0)
    neg1 = sice1 < 0.
    sh2o1 = W(neg1, sh2o1 + sice1, sh2o1)
    sice1 = W(neg1, 0.0, sice1)
    ret = isnow == 0
    # sums over active layers
    sneqv2 = f32(0.); snowh2 = f32(0.); zwice = f32(0.); zwliq = f32(0.)
    for p in range(3):
        a = (p - 2) >= isnow + 1
        sneqv2 = sneqv2 + W(a, ice[p] + liq[p], 0.)
        snowh2 = snowh2 + W(a, dz[p], 0.)
        zwice = zwice + W(a, ice[p], 0.)
        zwliq = zwliq + W(a, liq[p], 0.)
    sneqv = W(ret, sneqv, sneqv2)
    snowh = W(ret, snowh, snowh2)
    thin = (~ret) & (snowh < (0.05 if gla else 0.025)) & (isnow < 0)
    isnow = W(thin, 0, isnow)
    sneqv = W(thin, zwice, sneqv)
    pond2 = W(thin, (pond2 + zwliq) if gla else zwliq, pond2)
    snowh = W(thin & (sneqv <= 0.), 0.0, snowh)
    # part 3: thickness-based combine for isnow < -1
    DZMIN = (0.045, 0.05, 0.2) if gla else (0.025, 0.025, 0.1)
    run3 = (~ret) & (~thin) & (isnow < -1)
    isnow_old2 = isnow
    mssi = jnp.int32(1)
    done = ~run3
    for p in range(3):
        i = p - 2
        a = (i >= isnow_old2 + 1) & ~done
        dzmin = W(mssi == 1, DZMIN[0], W(mssi == 2, DZMIN[1], DZMIN[2]))
        small = a & (dz[p] < dzmin)
        # neighbour selection (positions are static for a given p; choice between p-1 and p+1 is traced)
        nb_up_ok = i == isnow + 1
        if p == 2:
            neibor_hi = jnp.bool_(False)       # i == 0 -> neighbour = i-1
            use_lo = jnp.bool_(True)
        else:
            neibor_hi = nb_up_ok
            tmp_lo = (dz[p - 1] + dz[p]) < (dz[p + 1] + dz[p]) if p >= 1 else jnp.bool_(False)
            use_lo = (~nb_up_ok) & tmp_lo
            if p == 0:
                use_lo = jnp.bool_(False)
        # case HI: J = p+1, L = p ; case LO: J = p, L = p-1
        new_dz, new_liq, new_ice, new_stc = list(dz), list(liq), list(ice), list(stc)
        res = {}
        if p < 2:
            cdz, cliq, cice, cst = combo(dz[p + 1], liq[p + 1], ice[p + 1], stc[p + 1], dz[p], liq[p], ice[p], stc[p])
            res["hi"] = (p + 1, cdz, cliq, cice, cst)
        if p >= 1:
            cdz, cliq, cice, cst = combo(dz[p], liq[p], ice[p], stc[p], dz[p - 1], liq[p - 1], ice[p - 1], stc[p - 1])
            res["lo"] = (p, cdz, cliq, cice, cst)
        # apply the selected case sequentially via masks
        for case, (jp, cdz, cliq, cice, cst) in res.items():
            if case == "hi":
                sel = small & (~use_lo)
            else:
                sel = small & use_lo
            dz[jp] = W(sel, cdz, dz[jp]); liq[jp] = W(sel, cliq, liq[jp]); ice[jp] = W(sel, cice, ice[jp])
            stc[jp] = W(sel, cst, stc[jp])
            # shift K = J-1 .. isnow+2 (descending)
            for kp in range(jp - 1, 0, -1):
                kk = kp - 2
                ok = sel & (kk >= isnow + 2)
                stc[kp] = W(ok, stc[kp - 1], stc[kp])
                ice[kp] = W(ok, ice[kp - 1], ice[kp])
                liq[kp] = W(ok, liq[kp - 1], liq[kp])
                dz[kp] = W(ok, dz[kp - 1], dz[kp])
        isnow = isnow + W(small, 1, 0)
        done = done | (small & (isnow >= -1))
        mssi = mssi + W(a & ~small, 1, 0)
    sn.update(isnow=isnow, dz=dz, ice=ice, liq=liq, stc=stc, sice1=sice1, sh2o1=sh2o1, snowh=snowh, sneqv=sneqv,
              ponding1=pond1, ponding2=pond2)
    return sn


def divide(isnow, stc, snice, snliq, dz, thr2=0.20):
    """snow layer division. inputs: lists of 3 (p 0..2). returns isnow, stc, snice, snliq, dz"""
    # top-aligned copies: top[j] = layer p = isnow + 3 + j
    s = isnow + 3
    def top(lst):
        out = []
        for j in range(3):
            idx = jnp.clip(s + j, 0, 2)
            out.append(jnp.stack(lst)[idx])
        return out
    dzt, swice, swliq, tsno = top(dz), top(snice), top(snliq), top(stc)
    msno = -isnow
    c1 = msno == 1
    big = dzt[0] > 0.05
    m1 = c1 & big
    msno = W(m1, 2, msno)
    h = dzt[0] / 2.
    dzt[1] = W(m1, h, dzt[1]); swice[1] = W(m1, swice[0] / 2., swice[1]); swliq[1] = W(m1, swliq[0] / 2., swliq[1])
    tsno[1] = W(m1, tsno[0], tsno[1])
    dzt[0] = W(m1, h, dzt[0]); swice[0] = W(m1, swice[0] / 2., swice[0]); swliq[0] = W(m1, swliq[0] / 2., swliq[0])
    # msno > 1
    c2 = (msno > 1) & (dzt[0] > 0.05)
    drr = dzt[0] - 0.05
    propor = drr / dzt[0]
    zwice = propor * swice[0]
    zwliq = propor * swliq[0]
    propor2 = 0.05 / dzt[0]
    new_swice0 = propor2 * swice[0]
    new_swliq0 = propor2 * swliq[0]
    cdz, cliq, cice, cst = combo(dzt[1], swliq[1], swice[1], tsno[1], drr, zwliq, zwice, tsno[0])
    dzt[1] = W(c2, cdz, dzt[1]); swliq[1] = W(c2, cliq, swliq[1]); swice[1] = W(c2, cice, swice[1]); tsno[1] = W(c2, cst, tsno[1])
    swice[0] = W(c2, new_swice0, swice[0]); swliq[0] = W(c2, new_swliq0, swliq[0]); dzt[0] = W(c2, 0.05, dzt[0])
    c3 = c2 & (msno <= 2) & (dzt[1] > thr2)
    msno = W(c3, 3, msno)
    dtdz = (tsno[0] - tsno[1]) / ((dzt[0] + dzt[1]) / 2.)
    h2 = dzt[1] / 2.
    ts3 = tsno[1] - dtdz * h2 / 2.
    warm = ts3 >= TFRZ
    new_t3 = W(warm, tsno[1], ts3)
    new_t2 = W(warm, tsno[1], tsno[1] + dtdz * h2 / 2.)
    dzt[2] = W(c3, h2, dzt[2]); swice[2] = W(c3, swice[1] / 2., swice[2]); swliq[2] = W(c3, swliq[1] / 2., swliq[2])
    tsno[2] = W(c3, new_t3, tsno[2])
    dzt[1] = W(c3, h2, dzt[1]); swice[1] = W(c3, swice[1] / 2., swice[1]); swliq[1] = W(c3, swliq[1] / 2., swliq[1])
    tsno[1] = W(c3, new_t2, tsno[1])
    # msno > 2
    c4 = (msno > 2) & (dzt[1] > 0.2)
    drr = dzt[1] - 0.2
    propor = drr / dzt[1]
    zwice = propor * swice[1]
    zwliq = propor * swliq[1]
    propor2 = 0.2 / dzt[1]
    new_swice1 = propor2 * swice[1]
    new_swliq1 = propor2 * swliq[1]
    cdz, cliq, cice, cst = combo(dzt[2], swliq[2], swice[2], tsno[2], drr, zwliq, zwice, tsno[1])
    dzt[2] = W(c4, cdz, dzt[2]); swliq[2] = W(c4, cliq, swliq[2]); swice[2] = W(c4, cice, swice[2]); tsno[2] = W(c4, cst, tsno[2])
    swice[1] = W(c4, new_swice1, swice[1]); swliq[1] = W(c4, new_swliq1, swliq[1]); dzt[1] = W(c4, 0.2, dzt[1])
    isnow_n = W(isnow < 0, -msno, isnow)
    s2 = isnow_n + 3
    do = isnow < 0
    out_dz = []; out_ice = []; out_liq = []; out_stc = []
    for p in range(3):
        a = do & (p >= s2)
        j = jnp.clip(p - s2, 0, 2)
        out_dz.append(W(a, jnp.stack(dzt)[j], dz[p]))
        out_ice.append(W(a, jnp.stack(swice)[j], snice[p]))
        out_liq.append(W(a, jnp.stack(swliq)[j], snliq[p]))
        out_stc.append(W(a, jnp.stack(tsno)[j], stc[p]))
    return isnow_n, out_stc, out_ice, out_liq, out_dz


def compact(P, dt, isnow, stc, snice, snliq, imelt, ficeold, dz):
    C2, C3, C4, C5, DM, ETA0 = [P.get(k, d) for k, d in (("C2_SNOWCOMPACT", 21.e-3), ("C3_SNOWCOMPACT", 2.5e-6), ("C4_SNOWCOMPACT", 0.04),
                                                          ("C5_SNOWCOMPACT", 2.0), ("DM_SNOWCOMPACT", 100.0), ("ETA0_SNOWCOMPACT", 1.33e+6))]   # 5.0 table (ETA0 1.33e6)
    burden = f32(0.)
    dz = list(dz)
    for p in range(3):
        a = (p - 2) >= isnow + 1
        wx = snice[p] + snliq[p]
        wxs = W(a, wx, 1.0)
        dzs = W(a, dz[p], 1.0)
        fice = snice[p] / wxs
        void = 1. - (snice[p] / DENICE + snliq[p] / DENH2O) / dzs
        go = a & (void > 0.001) & (snice[p] > 0.1)
        bi = snice[p] / dzs
        td = jnp.fmax(0., TFRZ - stc[p])
        dexpf = jnp.exp(-C4 * td)
        ddz1 = -C3 * dexpf
        ddz1 = W(bi > DM, ddz1 * jnp.exp(-46.0E-3 * (bi - DM)), ddz1)
        ddz1 = W(snliq[p] > 0.01 * dzs, ddz1 * C5, ddz1)
        ddz2 = -(burden + 0.5 * wx) * jnp.exp(-0.08 * td - C2 * bi) / ETA0
        ddz3 = W(imelt[p] == 1, -(jnp.fmax(0., (ficeold[p] - fice) / jnp.fmax(1.E-6, ficeold[p]))) / dt, 0.)
        pdzdtc = (ddz1 + ddz2 + ddz3) * dt
        pdzdtc = jnp.fmax(-0.5, pdzdtc)
        dzn = dz[p] * (1. + pdzdtc)
        dzn = jnp.fmax(dzn, snice[p] / DENICE + snliq[p] / DENH2O)               # 5.0: density limits 50-500 kg/m3
        dzn = jnp.fmin(jnp.fmax(dzn, wx / 500.), wx / 50.)
        dz[p] = W(go, dzn, dz[p])
        burden = burden + W(a, wx, 0.)
    return dz


# ----------------------------------------------------------------------------------------------
def canwater(P, dt, fcev, fctr, elai, esai, bdfall, frozen_canopy, canliq, canice, tv, fwet, fveg):
    maxliq = fveg * P["CH2OP"] * (elai + esai)
    fr = frozen_canopy
    etran = W(fr, jnp.fmax(fctr / HSUB, 0.), jnp.fmax(fctr / HVAP, 0.))
    qevac = W(fr, 0., jnp.fmax(fcev / HVAP, 0.))
    qdewc = W(fr, 0., jnp.abs(jnp.fmin(fcev / HVAP, 0.)))
    qsubc = W(fr, jnp.fmax(fcev / HSUB, 0.), 0.)
    qfroc = W(fr, jnp.abs(jnp.fmin(fcev / HSUB, 0.)), 0.)
    qevac = jnp.fmin(canliq / dt, qevac)
    canliq = jnp.fmax(0., canliq + (qdewc - qevac) * dt)
    canliq = W(canliq <= 1.E-06, 0.0, canliq)
    maxsno = fveg * 6.6 * (0.27 + 46. / bdfall) * (elai + esai)
    qsubc = jnp.fmin(canice / dt, qsubc)
    canice = jnp.fmax(0., canice + (qfroc - qsubc) * dt)
    canice = W(canice <= 1.E-6, 0., canice)
    fwet = W((canice > 0.) & (canice >= canliq), jnp.fmax(0., canice) / jnp.fmax(maxsno, 1.E-06),     # 4.4: ice-based only when ice >= liquid
             jnp.fmax(0., canliq) / jnp.fmax(maxliq, 1.E-06))
    fwet = jnp.power(jnp.fmin(fwet, 1.), 0.667)
    melt = (canice > 1.E-6) & (tv > TFRZ)
    qmeltc = jnp.fmin(canice / dt, (tv - TFRZ) * CICE * canice / DENICE / (dt * HFUS))
    canice = W(melt, jnp.fmax(0., canice - qmeltc * dt), canice)
    canliq = W(melt, jnp.fmax(0., canliq + qmeltc * dt), canliq)
    tv = W(melt, fwet * TFRZ + (1. - fwet) * tv, tv)
    frz = (canliq > 1.E-6) & (tv < TFRZ)
    qfrzc = jnp.fmin(canliq / dt, (TFRZ - tv) * CWAT * canliq / DENH2O / (dt * HFUS))
    canliq = W(frz, jnp.fmax(0., canliq - qfrzc * dt), canliq)
    canice = W(frz, jnp.fmax(0., canice + qfrzc * dt), canice)
    tv = W(frz, fwet * TFRZ + (1. - fwet) * tv, tv)
    cmc = canliq + canice
    ecan = qevac + qsubc - qdewc - qfroc
    return canliq, canice, tv, fwet, cmc, ecan, etran


def snowfall(dt, qsnow, snowhin, sfctmp, isnow, snowh, dz, stc, snice, snliq, sneqv, thr=0.025):
    dz = list(dz); stc = list(stc); snice = list(snice); snliq = list(snliq)
    top0 = (isnow == 0) & (qsnow > 0.)
    snowh = W(top0, snowh + snowhin * dt, snowh)
    sneqv = W(top0, sneqv + qsnow * dt, sneqv)
    newn = top0 & (snowh >= thr)
    dz[2] = W(newn, snowh, dz[2])
    stc[2] = W(newn, jnp.fmin(273.16, sfctmp), stc[2])
    snice[2] = W(newn, sneqv, snice[2])
    snliq[2] = W(newn, 0.0, snliq[2])
    snowh = W(newn, 0.0, snowh)
    isnow = W(newn, -1, isnow)
    add = (isnow < 0) & (~newn) & (qsnow > 0.)
    for p in range(3):
        m = add & (p == isnow + 3)
        snice[p] = W(m, snice[p] + qsnow * dt, snice[p])
        dz[p] = W(m, dz[p] + snowhin * dt, dz[p])
    return isnow, snowh, dz, stc, snice, snliq, sneqv


def snowh2o(P, dt, qsnfro, qsnsub, qrain, sn):
    """sn: dict isnow, dz[3], ice[3], liq[3], stc[3], sice1, sh2o1, dzs1, snowh, sneqv, ponding1, ponding2"""
    sn = dict(sn)
    isnow, sneqv, snowh = sn["isnow"], sn["sneqv"], sn["snowh"]
    sice1, sh2o1 = sn["sice1"], sn["sh2o1"]
    ice = list(sn["ice"]); liq = list(sn["liq"])
    dzs1 = sn["dzs1"]
    z0 = sneqv == 0.
    sice1 = W(z0, sice1 + (qsnfro - qsnsub) * dt / (dzs1 * 1000.), sice1)
    n1 = z0 & (sice1 < 0.)
    sh2o1 = W(n1, sh2o1 + sice1, sh2o1)
    sice1 = W(n1, 0., sice1)
    top = (isnow == 0) & (sneqv > 0.)
    temp = sneqv
    sneqv_t = sneqv - qsnsub * dt + qsnfro * dt
    propor = sneqv_t / jnp.fmax(temp, 1e-30)
    snowh_t = jnp.fmax(0., propor * snowh)
    neg = sneqv_t < 0.
    sice1_t = W(neg, sice1 + sneqv_t / (dzs1 * 1000.), sice1)
    sneqv_t2 = W(neg, 0., sneqv_t)
    snowh_t2 = W(neg, 0., snowh_t)
    n2 = sice1_t < 0.
    sh2o1_t = W(n2, sh2o1 + sice1_t, sh2o1)
    sice1_t2 = W(n2, 0., sice1_t)
    sneqv = W(top, sneqv_t2, sneqv); snowh = W(top, snowh_t2, snowh)
    sice1 = W(top, sice1_t2, sice1); sh2o1 = W(top, sh2o1_t, sh2o1)
    tiny = (snowh <= 1.E-8) | (sneqv <= 1.E-6)
    snowh = W(tiny, 0.0, snowh)
    sneqv = W(tiny, 0.0, sneqv)
    # layered snow
    lay = isnow < 0
    for p in range(3):
        m = lay & (p == isnow + 3)
        ice[p] = W(m, ice[p] - qsnsub * dt + qsnfro * dt, ice[p])
    wgdif = at(ice, jnp.clip(isnow + 3, 0, 2))
    sn.update(isnow=isnow, sneqv=sneqv, snowh=snowh, sice1=sice1, sh2o1=sh2o1, ice=ice, liq=liq)
    cmb = lay & (wgdif < 1.e-6)
    sn_c = combine(P, sn)
    sn = _dsel(cmb, sn_c, sn)
    isnow, ice, liq = sn["isnow"], list(sn["ice"]), list(sn["liq"])
    lay2 = isnow < 0
    for p in range(3):
        m = lay2 & (p == isnow + 3)
        liq[p] = W(m, jnp.fmax(0., liq[p] + qrain * dt), liq[p])
    dz = list(sn["dz"])
    qin = f32(0.)
    qout = f32(0.)
    fmx = P["SNLIQMAXFRAC"]
    for p in range(3):
        a = isnow + 3 <= p
        dzs = W(a, dz[p], 1.0)
        vol_ice = jnp.fmin(1., ice[p] / (dzs * DENICE))
        epore = 1. - vol_ice
        liq_n = liq[p] + qin
        vol_liq = liq_n / (dzs * DENH2O)
        qo = jnp.fmax(0., (vol_liq - P["SSI"] * epore) * dzs)
        if p == 2:                                    # bottom layer: release timescale (5.0 SnowpackHydrologyMod)
            qo = jnp.fmax((vol_liq - epore) * dzs, P["SNOW_RET_FAC"] * dt * qo)
        qo = qo * DENH2O
        l2 = liq_n - qo
        over = (l2 / jnp.fmax(ice[p] + l2, 1e-30)) > fmx
        qo = W(over, qo + (l2 - fmx / (1. - fmx) * ice[p]), qo)
        l2 = W(over, fmx / (1. - fmx) * ice[p], l2)
        liq[p] = W(a, l2, liq[p])
        dz[p] = W(a, jnp.fmax(dz[p], l2 / DENH2O + ice[p] / DENICE), dz[p])
        qin = W(a, qo, qin)
        qout = W(a, qo, qout)
    sn.update(ice=ice, liq=liq, dz=dz)
    return sn, qout / dt


def snowwater(P, dt, zsoil, sfctmp, snowhin, qsnow, qsnfro, qsnsub, qrain, ficeold, imelt, isnow, snowh, sneqv,
              snice, snliq, sh2o1, sice1, stc, zsnso, dzsnso):
    """returns updated snow state; stc/zsnso/dzsnso lists of 7; snice/snliq lists of 3"""
    snoflow = f32(0.)
    dzsnso = list(dzsnso); stc = list(stc)
    snice = list(snice); snliq = list(snliq)
    isnow, snowh, dzs3, stc3, snice, snliq, sneqv = snowfall(dt, qsnow, snowhin, sfctmp, isnow, snowh, dzsnso[:3], stc[:3],
                                                              snice, snliq, sneqv)
    dzsnso[:3] = dzs3; stc[:3] = stc3
    lay = isnow < 0
    # compact
    dzc = compact(P, dt, isnow, stc[:3], snice, snliq, imelt[:3], ficeold, dzsnso[:3])
    dzsnso[:3] = [W(lay, a, b) for a, b in zip(dzc, dzsnso[:3])]
    sn = dict(isnow=isnow, dz=list(dzsnso[:3]), ice=list(snice), liq=list(snliq), stc=list(stc[:3]), sice1=sice1,
              sh2o1=sh2o1, dzs1=dzsnso[3], snowh=snowh, sneqv=sneqv, ponding1=f32(0.), ponding2=f32(0.))
    cm = combine(P, sn)
    sn = _dsel(lay, cm, sn)
    isn2, stc2, ice2, liq2, dz2 = divide(sn["isnow"], sn["stc"], sn["ice"], sn["liq"], sn["dz"])
    lay = sn["isnow"] < 0
    sn["isnow"] = W(lay, isn2, sn["isnow"])
    sn["stc"] = [W(lay, a, b) for a, b in zip(stc2, sn["stc"])]
    sn["ice"] = [W(lay, a, b) for a, b in zip(ice2, sn["ice"])]
    sn["liq"] = [W(lay, a, b) for a, b in zip(liq2, sn["liq"])]
    sn["dz"] = [W(lay, a, b) for a, b in zip(dz2, sn["dz"])]
    sn, qsnbot = snowh2o(P, dt, qsnfro, qsnsub, qrain, sn)
    isnow = sn["isnow"]
    ice, liq, stc3, dz3 = sn["ice"], sn["liq"], sn["stc"], sn["dz"]
    # zero unused layers
    for p in range(3):
        un = p < isnow + 3
        ice[p] = W(un, 0., ice[p]); liq[p] = W(un, 0., liq[p]); stc3[p] = W(un, 0., stc3[p]); dz3[p] = W(un, 0., dz3[p])
    sneqv = sn["sneqv"]
    big = sneqv > 2000.
    bdsnow = ice[2] / jnp.fmax(dz3[2], 1e-30)
    snoflow_x = sneqv - 2000.
    ice[2] = W(big, ice[2] - snoflow_x, ice[2])
    dz3[2] = W(big, dz3[2] - snoflow_x / bdsnow, dz3[2])
    snoflow = W(big, snoflow_x / dt, 0.)
    lay = isnow < 0
    s_new = f32(0.)
    for p in range(3):
        a = isnow + 3 <= p
        s_new = s_new + W(a, ice[p] + liq[p], 0.)
    sneqv = W(lay, s_new, sneqv)
    # depths
    dzf = [None] * NL
    for p in range(3):
        a = isnow + 3 <= p
        dzf[p] = W(a, -dz3[p], 0.)
    dzf[3] = zsoil[0]
    for k in range(1, NSOIL):
        dzf[k + 3] = zsoil[k] - zsoil[k - 1]
    zs = [None] * NL
    for p in range(NL):
        first = p == isnow + 3
        prev = zs[p - 1] if p > 0 else f32(0.)
        zs[p] = W(first, dzf[p], W(isnow + 3 < p, prev + dzf[p], 0.))
    dznew = [W(isnow + 3 <= p, -dzf[p], 0.) for p in range(NL)]
    # snow layer dz are positive again: -(-dz) = dz ; soil -(dz<0) positive
    dzout = dznew
    return dict(isnow=isnow, snowh=sn["snowh"], sneqv=sneqv, snice=ice, snliq=liq, stc=list(stc3) + stc[3:], zsnso=zs,
                dzsnso=dzout, sh2o1=sn["sh2o1"], sice1=sn["sice1"], qsnbot=qsnbot, snoflow=snoflow,
                ponding1=sn["ponding1"], ponding2=sn["ponding2"])


# ----------------------------------------------------------------------------------------------
def _wdfcnd1(P, smc, fcr, k):
    factr = jnp.fmax(0.01, smc / P["SMCMAX"][k])
    wdf = P["DWSAT"][k] * jnp.power(factr, P["BEXP"][k] + 2.0)
    wdf = wdf * (1.0 - fcr)
    wcnd = P["DKSAT"][k] * jnp.power(factr, 2.0 * P["BEXP"][k] + 3.0)
    wcnd = wcnd * (1.0 - fcr)
    return wdf, wcnd


def _wdfcnd2(P, smc, sicemax, k):
    factr1 = 0.05 / P["SMCMAX"][k]
    factr2 = jnp.fmax(0.01, smc / P["SMCMAX"][k])
    factr1 = jnp.fmin(factr1, factr2)
    expon = P["BEXP"][k] + 2.0
    wdf = P["DWSAT"][k] * jnp.power(factr2, expon)
    vkwgt = 1.0 / (1.0 + jnp.power(500.0 * sicemax, 3.0))
    wdf2 = vkwgt * wdf + (1.0 - vkwgt) * P["DWSAT"][k] * jnp.power(factr1, expon)
    wdf = W(sicemax > 0.0, wdf2, wdf)
    expon = 2.0 * P["BEXP"][k] + 3.0
    wcnd = P["DKSAT"][k] * jnp.power(factr2, expon)
    return wdf, wcnd


def infil(P, dt, zsoil, sh2o, sice, sicemax, qinsur):
    """Noah-MP INFIL (runoff option 3, Schaake96). Returns pddum, runsrf (both 0 where QINSUR <= 0: the Fortran leaves its outputs untouched)."""
    cvfrz = 3
    dt1 = dt / 86400.
    smcav = P["SMCMAX"][0] - P["SMCWLT"][0]
    dmax = -zsoil[0] * smcav
    dice = -zsoil[0] * sice[0]
    dmax = dmax * (1.0 - (sh2o[0] + sice[0] - P["SMCWLT"][0]) / smcav)
    dd = dmax
    for k in range(1, NSOIL):
        dice = dice + (zsoil[k - 1] - zsoil[k]) * sice[k]
        dm = (zsoil[k - 1] - zsoil[k]) * smcav
        dm = dm * (1.0 - (sh2o[k] + sice[k] - P["SMCWLT"][k]) / smcav)
        dd = dd + dm
    val = 1. - jnp.exp(-P["KDT"] * dt1)
    ddt = dd * val
    px = jnp.fmax(0., qinsur * dt)
    infmax = (px * (ddt / (px + ddt))) / dt
    acrt = cvfrz * P["FRZX"] / dice
    sm = 1.0 + (acrt * acrt) / 2.0 + acrt / 1.0
    fcr = W(dice > 1.e-2, 1. - jnp.exp(-acrt) * sm, 1.0)
    infmax = infmax * fcr
    wdf, wcnd = _wdfcnd2(P, sh2o[0], sicemax, 0)
    infmax = jnp.fmax(infmax, wcnd)
    infmax = jnp.fmin(infmax, px)
    runsrf = jnp.fmax(0., qinsur - infmax)
    pddum = qinsur - runsrf
    pos = qinsur > 0.
    return W(pos, pddum, 0.), W(pos, runsrf, 0.)


def zwteq(P, zsoil, dz, sh2o):
    nfine = 100
    wd1 = f32(0.)
    for k in range(NSOIL):
        wd1 = wd1 + (P["SMCMAX"][0] - sh2o[k]) * dz[k]
    dzfine = 3.0 * (-zsoil[NSOIL - 1]) / nfine
    # sequential search with early exit (EXIT leaves ZWT = ZFINE(K)); implemented as a scan over the fine layers
    zinit = -3. * zsoil[NSOIL - 1] - 0.001
    zfound = zinit
    wd2 = f32(0.)
    done = jnp.bool_(False)
    for kk in range(1, nfine + 1):
        zf = kk * dzfine
        temp = 1. + (zinit - zf) / P["PSISAT"][0]
        wd2n = wd2 + P["SMCMAX"][0] * (1. - jnp.power(temp, -1. / P["BEXP"][0])) * dzfine
        wd2 = W(done, wd2, wd2n)
        hit = (~done) & (jnp.abs(wd2 - wd1) <= 0.01)
        zfound = W(hit, zf, zfound)
        done = done | hit
    return zfound


def runoff_vic(P, smc, zsoil, qin, ts):
    """RunoffSurfaceVIC (surface runoff option 6). The Fortran weights the top layers by the layer-bottom depth. Returns (runoff, infiltration) in m/s."""
    top = f32(0.); topmax = f32(0.)
    for k in range(NSOIL - 2):
        top = top + smc[k] * (-zsoil[k]); topmax = topmax + P["SMCMAX"][k] * (-zsoil[k])
    b = P["BVIC"]
    expf = b / (1.0 + b)
    satfrac = jnp.clip(1.0 - jnp.power(jnp.fmax(0.0, 1.0 - top / topmax), expf), 0.0, 1.0)
    wmax = (1.0 + b) * topmax
    winit = wmax * (1.0 - jnp.power(1.0 - satfrac, 1.0 / b))
    tot = qin * ts
    cap = tot - topmax + top
    inv = 1.0 - (winit + tot) / jnp.where(wmax == 0.0, 1.0, wmax)
    mid = cap + topmax * jnp.power(jnp.fmax(inv, 0.0), 1.0 + b)
    r = W(qin == 0.0, 0.0, W(wmax == 0.0, tot, W(winit + tot > wmax, cap, mid)))
    r = jnp.fmin(jnp.fmax(r / ts, 0.0), qin)
    return r, qin - r


def runoff_xaj(P, smc, zsoil, fimp, qin, ts):
    """RunoffSurfaceXinAnJiang (surface runoff option 7). Returns (runoff, infiltration) in m/s."""
    wt = f32(0.); wmx = f32(0.); wf = f32(0.); wfmx = f32(0.)
    for k in range(NSOIL - 2):
        d = -zsoil[k]
        over = smc[k] - P["SMCREF"][k]
        pos = over > 0.0
        wf = wf + W(pos, over * d, 0.)
        wt = wt + W(pos, P["SMCREF"][k] * d, smc[k] * d)
        wmx = wmx + P["SMCREF"][k] * d
        wfmx = wfmx + (P["SMCMAX"][k] - P["SMCREF"][k]) * d
    wt = jnp.fmin(wt, wmx); wf = jnp.fmin(wf, wfmx)
    rimp = fimp * qin * ts
    a = P["AXAJ"]; bs = P["BXAJ"]; xs = P["XXAJ"]
    rat = wt / wmx
    low = (1.0 - fimp) * qin * ts * jnp.power(0.5 - a, 1.0 - bs) * jnp.power(rat, bs)
    high = (1.0 - fimp) * qin * ts * (1.0 - jnp.power(0.5 + a, 1.0 - bs) * jnp.power(jnp.fmax(1.0 - rat, 0.0), bs))
    perv = W(rat <= 0.5 - a, low, high)
    r = W(qin == 0.0, 0.0, perv * (1.0 - jnp.power(jnp.fmax(1.0 - wf / wfmx, 0.0), xs)) + rimp)
    r = jnp.fmin(jnp.fmax(r / ts, 0.0), qin)
    return r, qin - r


def _dvic_rse(P, winit, wmax, y):
    """RunoffSatExcessDynamicVic (saturation excess runoff, Eq. 5)."""
    fac = P["BDVIC"]
    wt = jnp.fmin(winit + y, wmax)
    r = y - (wmax / (fac + 1.0)) * (jnp.power(1.0 - winit / wmax, fac + 1.0) - jnp.power(1.0 - wt / wmax, fac + 1.0))
    return jnp.fmax(r, 0.0)


def _dvic_rie(y, yinit, rs, imax, itmp, ts, win, b):
    """RunoffInfilExcessDynamicVic."""
    a = win - rs - imax * ts * (1.0 - jnp.power(1.0 - (win - rs) / (imax * ts), b + 1.0))
    r = W(y >= yinit, a, win - rs - imax * ts)
    return jnp.fmax(r, 0.0)


def runoff_dvic(P, o, smc, sice0, zsoil, qin, ts, acc):
    """RunoffSurfaceDynamicVic (surface runoff option 8, infiltration option o["infdv"]: 1 Philip, 2 Green-Ampt, 3 Smith-Parlange).
    All branches are evaluated and selected. The Fortran evaluates the saturation-excess helper in two branches with a
    never-initialised depth (DepthYTmp, taken as 0 here). Returns (runoff, infiltration, new accumulated infiltration)."""
    nit = 20
    err = 1.388889E-07 * ts
    b = P["BBVIC"]; fac = P["BDVIC"]
    wtop = f32(0.); wsat = f32(0.)
    for k in range(NSOIL - 2):
        wtop = wtop + smc[k] * (-zsoil[k]); wsat = wsat + P["SMCMAX"][k] * (-zsoil[k])
    wtop = jnp.fmin(wtop, wsat)
    win = qin * ts
    wmax = wsat * (fac + 1.0)
    winit = wmax * (1.0 - (1.0 - jnp.power(wtop / wsat, 1.0 / (1.0 + fac))))
    wdf, wcnd = _wdfcnd2(P, smc[0], sice0, 0)
    ks = P["DKSAT"][0]
    if o["infdv"] == 1:
        sorp = jnp.sqrt(2.0 * jnp.fmax(0.0, P["SMCMAX"][0] - smc[0]) * (P["DWSAT"][0] - wdf))
        a = jnp.fmax(jnp.fmin(wcnd, (2.0 / 3.0) * ks), (1.0 / 3.0) * ks)
        itmp = 0.5 * sorp * jnp.power(ts, -0.5) + a
    else:
        ft = P["GDVIC"] * jnp.fmax(0.0, P["SMCMAX"][0] - smc[0]) * (-zsoil[0])
        if o["infdv"] == 2:
            itmp = ks + (ft / acc) * (ks - wcnd)
        else:
            itmp = W(ft == 0.0, wcnd, ks + 0.82 * (ks - wcnd) / (jnp.exp(0.82 * acc / jnp.where(ft == 0.0, 1.0, ft)) - 1.0))
    itmp = W(ks < qin, jnp.fmin(qin, itmp), qin)
    acc_new = acc + itmp
    imax = (b + 1.0) * itmp
    its = imax * ts
    zero = f32(0.)

    def iterate(y0, step, stopfun=None):
        y = y0; done = jnp.bool_(False)
        for i in range(nit):
            yn = step(y)
            stop = (jnp.abs(yn - y) <= err) if stopfun is None else stopfun(yn, y)
            stop = stop | (i == nit - 1)
            y = W(done, y, yn); done = done | stop
        return y

    def clip(y):
        return jnp.fmin(jnp.fmax(y, 0.0), win)

    def it12(y0, with_exp):
        if with_exp:
            return iterate(y0, lambda y: _dvic_rse(P, winit, wmax, y) + itmp * ts * jnp.power((win - _dvic_rse(P, winit, wmax, y)) / its, b + 1.0))
        return iterate(y0, lambda y: _dvic_rse(P, winit, wmax, y) + itmp * ts)

    def it3(y0, stop30):
        def s30(y):
            return clip(y - _dvic_rse(P, winit, wmax, y) - itmp * ts + win)
        if stop30 == "y":
            yi = iterate(y0, s30)
        else:
            yi = iterate(y0, s30, lambda yn, y: jnp.abs(_dvic_rse(P, winit, wmax, yn) + its - win) <= err)
        def s3(y):
            rs = _dvic_rse(P, winit, wmax, y)
            return win - _dvic_rie(y, yi, rs, imax, itmp, ts, win, b)
        return iterate(yi, s3)

    c1 = win + winit > wmax
    d = its >= win
    # B1: wet, capacity above the input
    y_b = wmax - winit
    rs_b = _dvic_rse(P, winit, wmax, y_b)
    tmp_b = wmax - winit - rs_b - itmp * ts * jnp.power((win - rs_b) / its, b + 1.0)       # Fortran (1-(1-x**(B+1))) = x**(B+1)
    sat_b1 = c1 & d & (tmp_b <= 0.0)
    y_b1 = it12(zero, True)
    # B2 / C2: input above the capacity; the saturation excess is evaluated with an undefined depth (0)
    rs_u = _dvic_rse(P, winit, wmax, zero)
    e = rs_u + its <= win
    sat_b2 = c1 & ~d & e & (wmax - winit - rs_u - its <= 0.0)
    y_b2a = it12(zero, False)
    y_b2c = it3(win / 2.0, "y")
    y_c1 = it12(win / 2.0, True)
    y_c2a = it12(win / 2.0, False)
    y_c2c = it3(zero, "c")
    y = W(c1, W(d, y_b1, W(e, y_b2a, y_b2c)), W(d, y_c1, W(e, y_c2a, y_c2c)))
    y = clip(y)
    runsat = _dvic_rse(P, winit, wmax, y)
    runinf = win - y
    sat_full = sat_b1 | sat_b2
    runsat = W(sat_full, win - (wsat - wtop), runsat); runinf = W(sat_full, 0.0, runinf)
    full = (wtop >= wsat) & (winit >= wmax)
    runsat = W(full, win, runsat); runinf = W(full, 0.0, runinf)
    r = (runsat + runinf) / ts
    r = W(win <= 0.0, 0.0, r)
    r = jnp.fmax(jnp.fmin(r, qin), 0.0)
    return r, qin - r, acc_new


def philip_infil(P, ts, sh2o1, smc1, sice):
    """IrrigationInfilPhilip: surface infiltration rate [m/s] of the first soil layer (SoilDiffusivityConductivityOpt2)."""
    sicemax = f32(0.)
    for k in range(NSOIL):
        sicemax = W(sice[k] > sicemax, sice[k], sicemax)
    wdf, wcnd = _wdfcnd2(P, sh2o1, sicemax, 0)
    sorp = jnp.sqrt(2.0 * jnp.fmax(0.0, P["SMCMAX"][0] - smc1) * (P["DWSAT"][0] - wdf))
    cinit = jnp.fmin(wcnd, (2.0 / 3.0) * P["DKSAT"][0])
    cinit = jnp.fmax(cinit, (1.0 / 3.0) * P["DKSAT"][0])
    return jnp.fmax(0.0, 0.5 * sorp * jnp.power(ts, -0.5) + cinit)


def irrigation_prepare(P, o, x, rain, fveg, lai, sh2o, cropland):
    """IrrigationPrepare + IrrigationTrigger (OPT_IRR 1 = on, 2 = within the crop season [planting day, harvest day - IRR_HAR], 3 = on above a LAI threshold)
    and the state it keeps. Returns a dict of the fractions and amounts [m]."""
    thr = P["IRR_FRAC"]
    fgrid = x["IRFRACT"]
    fs, fm, ff = x["SIFRACT"], x["MIFRACT"], x["FIFRACT"]
    amt_s, amt_m, amt_f = x["IRWATSI"], x["IRWATMI"], x["IRWATFI"]
    cs, cm, cf = x["IRNUMSI"], x["IRNUMMI"], x["IRNUMFI"]
    method = o["irrm"]
    if method == 0:
        fs = W((fs == 0.0) & (fm == 0.0) & (ff == 0.0) & (fgrid >= thr), 1.0, fs)
    elif method == 1:
        fs, fm, ff = f32(1.0) + 0. * fs, 0. * fm, 0. * ff
    elif method == 2:
        fs, fm, ff = 0. * fs, f32(1.0) + 0. * fm, 0. * ff
    elif method == 3:
        fs, fm, ff = 0. * fs, 0. * fm, f32(1.0) + 0. * ff
    rain_ok = rain < (P["IR_RAIN"] / 3600.0)
    trig = cropland & (fgrid >= thr) & rain_ok & ((amt_s + amt_m + amt_f) == 0.0)
    flag = jnp.bool_(o["irr"] in (1, 3))
    if o["irr"] == 2:
        flag = ~((x["JULIAN"] < P["PLTDAY"]) | (x["JULIAN"] > (P["HSDAY"] - P["IRR_HAR"])))
    if o["irr"] == 3:
        flag = flag & ~(lai < P["IRR_LAI"])
    zr = [-z for z in x["ZSOIL"]]
    nroot = P["NROOT"].astype(jnp.int32)
    avail = (sh2o[0] - P["SMCWLT"][0]) * zr[0]
    availmax = (P["SMCREF"][0] - P["SMCWLT"][0]) * zr[0]
    for k in range(1, NSOIL):
        inr = k < nroot
        dzk = x["ZSOIL"][k - 1] - x["ZSOIL"][k]
        avail = W(inr, avail + (sh2o[k] - P["SMCWLT"][k]) * dzk, avail)
        availmax = W(inr, availmax + (P["SMCREF"][k] - P["SMCWLT"][k]) * dzk, availmax)
    need = trig & flag & ((avail / availmax) <= P["IRR_MAD"])
    water = (availmax - avail) * fgrid * fveg
    if method == 0:
        ns = need & (amt_s == 0.0) & (fs > 0.0); w_s = fs * water
        nm_ = need & (amt_m == 0.0) & (fm > 0.0); w_m = fm * water
        nf = need & (amt_f == 0.0) & (ff > 0.0); w_f = ff * water * (1.0 / (1.0 - P["FILOSS"]))
    else:
        ns = need & (amt_s == 0.0) & (method == 1); w_s = water
        nm_ = need & (amt_m == 0.0) & (method == 2); w_m = water
        nf = need & (amt_f == 0.0) & (method == 3); w_f = water * (1.0 / (1.0 - P["FILOSS"]))
    amt_s = W(ns, w_s, amt_s); cs = W(ns, cs + 1.0, cs)
    amt_m = W(nm_, w_m, amt_m); cm = W(nm_, cm + 1.0, cm)
    amt_f = W(nf, w_f, amt_f); cf = W(nf, cf + 1.0, cf)
    clear = trig & flag & ~need                              # trigger said "no deficit": zero the amounts
    amt_s = W(clear, 0.0, amt_s); amt_m = W(clear, 0.0, amt_m); amt_f = W(clear, 0.0, amt_f)
    stop = (rain >= (P["IR_RAIN"] / 3600.0)) | (fgrid < thr)
    amt_s = W(stop, 0.0, amt_s); amt_m = W(stop, 0.0, amt_m); amt_f = W(stop, 0.0, amt_f)
    return dict(fs=fs, fm=fm, ff=ff, amt_s=amt_s, amt_m=amt_m, amt_f=amt_f, cs=cs, cm=cm, cf=cf)


def irrigation_sprinkler(P, dt, ir, sfctmp, uu, vv, eair, sh2o1, smc1, sice):
    """IrrigationSprinkler: returns the new amount, the rain added [mm/s], the latent heat [W/m2] and the evaporation [mm/s]."""
    inf = philip_infil(P, dt, sh2o1, smc1, sice)
    rate_tmp = P["SPRIR_RATE"] * (1.0 / 1000.0) * dt / 3600.0
    rate = jnp.fmin(jnp.fmin(inf * dt, ir["amt_s"]), rate_tmp)
    wind = jnp.sqrt(uu ** 2.0 + vv ** 2.0)
    esat_ = 610.8 * jnp.exp((17.27 * (sfctmp - 273.15)) / (237.3 + (sfctmp - 273.15)))
    loss_w = 4.375 * jnp.exp(0.106 * wind) * jnp.power((esat_ - eair) * 0.01, -0.092) * jnp.power(sfctmp - 273.15, -0.102)
    loss_c = 4.337 * jnp.exp(0.077 * wind) * jnp.power((esat_ - eair) * 0.01, -0.098)
    loss = W(sfctmp > 273.15, loss_w, loss_c)
    loss = W(jnp.isnan(loss), 4.0, loss)
    loss = W((loss > 100.0) | (loss < 0.0), 4.0, loss)
    rate = rate * ir["fs"]
    full = rate >= ir["amt_s"]
    rate = W(full, ir["amt_s"], rate)
    amt = W(full, 0.0, ir["amt_s"] - rate)
    evap_loss = rate * loss * (1.0 / 100.0)
    rate = rate - evap_loss
    return amt, rate * 1000.0 / dt, evap_loss * 1000.0 * HVAP / dt, evap_loss * 1000.0 / dt


def irrigation_micro(P, dt, ir, zsoil, sh2o, smc1, sice):
    inf = philip_infil(P, dt, sh2o[0], smc1, sice)
    rate_tmp = P["MICIR_RATE"] * (1.0 / 1000.0) * dt / 3600.0
    rate = jnp.fmin(jnp.fmin(0.5 * inf * dt, ir["amt_m"]), rate_tmp)
    rate = rate * ir["fm"]
    full = rate >= ir["amt_m"]
    rate = W(full, ir["amt_m"], rate)
    amt = W(full, 0.0, ir["amt_m"] - rate)
    return amt, sh2o[0] + (rate / (-1.0 * zsoil[0]))


def irrigation_flood(P, dt, ir, sh2o1, smc1, sice):
    inf = philip_infil(P, dt, sh2o1, smc1, sice)
    rate = inf * dt * P["FIRTFAC"]
    rate = rate * ir["ff"]
    full = rate >= ir["amt_f"]
    rate = W(full, ir["amt_f"], rate)
    amt = W(full, 0.0, ir["amt_f"] - rate)
    return amt, rate / dt


def tile_equi_depth(dd, dist, rad):
    """TileDrainageEquiDepth: height of the water table in the drain above the impermeable layer (Hooghoudt equivalent depth)."""
    PI = f32(22.0) / f32(7.0)
    aspect = (2.0 * PI * dd) / dist
    fac = f32(0.)
    res = f32(0.)
    done = jnp.zeros_like(dd, dtype=bool)
    last = jnp.zeros_like(dd, dtype=jnp.int32)
    for li in range(1, 46, 2):
        e = jnp.exp(-2.0 * li * aspect)
        ft = (4.0 * e) / (li * (1.0 - e))
        act = ~done
        fac = W(act, fac + ft, fac)
        hit = act & (ft < 1.0e-6)
        res = W(hit, ((PI * dist) / 8.0) / (jnp.log(dist / (PI * rad)) + fac), res)
        last = W(act, li, last)
        done = done | hit
    res_big = res
    res_small = dd
    fac_mid = ((PI * PI) / (4.0 * aspect)) + jnp.log(aspect / (2.0 * PI))
    res_mid = ((PI * dist) / 8.0) / (jnp.log(dist / (PI * rad)) + fac_mid)
    out = W(aspect > 0.5, res_big, W(aspect < 1.0e-8, res_small, res_mid))
    return W((aspect > 0.5) & (out < 0.0) & (last <= 2), dd, out)


def tile_drainage(P, o, dt, zsoil, dz, sh2o, sice, wcnd, frac, dx):
    """TileDrainageSimple (OPT_TDRN 1) and TileDrainageHooghoudt (OPT_TDRN 2) after the soil water solve; surface runoff option 3 only.
    Returns the new liquid water per layer and the tile drainage [mm/s]."""
    sh2o = list(sh2o)
    fc = [P["SMCREF"][k] for k in range(NSOIL)]
    zero = f32(0.)
    if o["tdrn"] == 1:
        active = (frac > 0.3)
        coef = P["TD_DC"] * dt / (24.0 * 3600.0)
        lo = P["DRAIN_LAYER_OPT"]
        tube = P["TD_DEPTH"]
        masks = []
        for k in range(NSOIL):
            L = k + 1
            m = W(lo == 0, tube == L,
                  W(lo == 1, L <= 2, W(lo == 2, L <= 3, W(lo == 3, (L >= 2) & (L <= 3), W(lo == 4, (L >= 3) & (L <= 4), lo == 5)))))
            masks.append(m)
        exc = []
        for k in range(NSOIL):
            fcl = fc[k] - sice[k]
            e = (sh2o[k] - (P["TDSMC_FAC"] * fcl)) * dz[k] * 1000.0
            e = W(lo == 0, e, jnp.fmax(e, 0.0))
            exc.append(W(masks[k], e, 0.0))
        tot = zero
        for k in range(NSOIL):
            tot = tot + jnp.fmax(exc[k], 0.0)
        go = active & (tot > 0.0)
        tot_c = W(tot > coef, coef, tot)
        drain = W(go, tot_c, 0.0)
        for k in range(NSOIL):
            fr = W(exc[k] > 0.0, exc[k] / jnp.where(tot > 0.0, tot, 1.0), 0.0)
            take = fr * tot_c
            sh2o[k] = W(go & masks[k], sh2o[k] - take / (dz[k] * 1000.0), sh2o[k])
        return sh2o, drain / dt, None
    # ---- Hooghoudt
    active = (frac > 0.1)
    smc = [sh2o[k] + sice[k] for k in range(NSOIL)]
    zl = [-zsoil[k] for k in range(NSOIL)]                    # depth of the layer bottom, positive
    sat = jnp.int32(0)
    for k in range(NSOIL - 1, -1, -1):
        cond = (smc[k] >= fc[k]) & (fc[k] > P["SMCWLT"][k])
        sat = W(cond & ((sat == k + 2) | (k == NSOIL - 1)), k + 1, sat)
    wt_part = W(sat > 1, at(zl, jnp.fmax(sat - 2, 0)), 0.0)
    wtd = W(sat != 0, wt_part, zl[NSOIL - 1])
    wt2 = W(wtd > P["TD_ADEPTH"], P["TD_ADEPTH"], wtd)
    thick = []
    top = zero
    for k in range(NSOIL):
        t = W(wt2 > zl[k], 0.0, zl[k] - wt2)
        wt1 = zl[k] - top
        t = W((wt2 <= zl[k]) & (t > wt1), wt1, t)
        thick.append(t)
        top = zl[k]
    exc = []
    exctot = zero
    for k in range(NSOIL):
        e = (sh2o[k] - (fc[k] - sice[k])) * dz[k] * 1000.0
        e = W(e < 0.0, 0.0, e)
        exc.append(e)
        exctot = exctot + e
    lat = zero
    tz = zero
    for k in range(NSOIL):
        lk = wcnd[k] * P["KLAT_FAC"] * dt
        lat = lat + thick[k] * lk
        tz = tz + thick[k]
    tz = W(tz < 0.001, 0.001, tz)
    lat = W(lat < 0.001, 0.0, lat)
    lat_ave = lat / tz
    dd = P["TD_ADEPTH"] - P["TD_DDRAIN"]
    hab = tile_equi_depth(dd, P["TD_SPAC"], P["TD_RADI"])
    hgt = P["TD_DDRAIN"] - wt2
    td = ((8.0 * lat_ave * hab * hgt) + (4.0 * lat_ave * hgt * hgt)) / (P["TD_SPAC"] * P["TD_SPAC"])
    td = W(hgt <= 0.0, 0.0, td)
    td = td * 1000.0
    td = W(td <= 0.0, 0.0, td)
    cap = P["TD_DCOEF"] * 1000.0 * dt / (24.0 * 3600.0)
    td = W(td > cap, cap, td)
    nd = jnp.floor(dx / P["TD_SPAC"])
    td = td * nd
    td = W(td > exctot, exctot, td)
    tmp = td
    done = jnp.zeros_like(td, dtype=bool)
    for k in range(NSOIL):
        cond = active & ~done & (tmp > 0.0) & (thick[k] > 0.0) & (exc[k] > 0.0)
        rem = exc[k] - tmp
        pos = rem > 0.0
        new_pos = (fc[k] - sice[k]) + rem / (dz[k] * 1000.0)
        new_neg = fc[k] - sice[k]
        sh2o[k] = W(cond, W(pos, new_pos, new_neg), sh2o[k])
        tmp = W(cond & ~pos, tmp - exc[k], tmp)
        done = done | (cond & pos)
    return sh2o, W(active, td / dt, 0.0), (active, wtd)          # Hooghoudt also overwrites the water table depth (WaterTableDepthSearch)


def soilwater(P, o, dt, zsoil, dz, qinsur, qseva, etrani, sice, sh2o, smc, zwt, td=None):
    """OPT_RUN (surface runoff) 1-4, 6, 7, OPT_INF=1. Returns sh2o, smc, runsrf, qdrain, wcnd, fcrmax, runsub, zwt"""
    opt_run = o["run"]
    sh2o = list(sh2o)
    rsat = f32(0.)
    for k in range(NSOIL):
        epore = jnp.fmax(1.E-4, P["SMCMAX"][k] - sice[k])
        rsat = rsat + jnp.fmax(0., sh2o[k] - epore) * dz[k]
        sh2o[k] = jnp.fmin(epore, sh2o[k])
    fcr = []
    A = 4.0
    for k in range(NSOIL):
        fice = jnp.fmin(1.0, sice[k] / P["SMCMAX"][k])
        fcr.append(jnp.fmax(0.0, jnp.exp(-A * (1. - fice)) - jnp.exp(-A)) / (1.0 - jnp.exp(-A)))
    sicemax = f32(0.)
    fcrmax = f32(0.)
    for k in range(NSOIL):
        sicemax = W(sice[k] > sicemax, sice[k], sicemax)
        fcrmax = W(fcr[k] > fcrmax, fcr[k], fcrmax)
    runsub = f32(0.)
    if opt_run == 2:
        zwt = zwteq(P, zsoil, dz, sh2o)
        runsub = (1.0 - fcrmax) * 4.0 * jnp.exp(-P["TIMEAN"]) * jnp.exp(-2.0 * zwt)
    fcr[0] = W(P["URBAN_FLAG"] > 0.5, 0.95, fcr[0])
    pos = qinsur > 0.
    runsrf0 = f32(0.)
    pddum = f32(0.)
    if opt_run == 1:
        fsat = P["FSATMX"] * jnp.exp(-0.5 * (P["BEXP"][0] / 3.0) * zwt)       # 4.4/5.0 (GY Niu 2022): decay factor from the soil exponent
    elif opt_run == 2:
        fsat = P["FSATMX"] * jnp.exp(-0.5 * 2.0 * zwt)
    elif opt_run == 4:
        smctot = f32(0.)
        dztot = f32(0.)
        run = jnp.bool_(True)
        for k in range(NSOIL):
            dztot = W(run, dztot + dz[k], dztot)
            smctot = W(run, smctot + smc[k] / P["SMCMAX"][k] * dz[k], smctot)
            run = run & (dztot < 2.0)
        smctot = smctot / dztot
        fsat = jnp.power(jnp.fmax(0.01, smctot), 4.)
    free = opt_run in (3, 6, 7, 8)                          # free-drainage bottom boundary and drainage runoff (subsurface 3, 6, 7, 8)
    if opt_run == 3:
        pddum, runsrf0 = infil(P, dt, zsoil, sh2o, sice, sicemax, qinsur)
    elif opt_run == 6:
        runsrf0, pddum = runoff_vic(P, smc, zsoil, qinsur, dt)
    elif opt_run == 7:
        runsrf0, pddum = runoff_xaj(P, smc, zsoil, fcr[0], qinsur, dt)
    elif opt_run == 8:
        runsrf0, pddum, _ = runoff_dvic(P, o, smc, sice[0], zsoil, qinsur, dt, f32(1.0e-6))
    else:
        runsrf0 = W(pos, qinsur * ((1.0 - fcr[0]) * fsat + fcr[0]), 0.)
        pddum = W(pos, qinsur - runsrf0, 0.)
    niter = W(pddum * dt > dz[0] * P["SMCMAX"][0], 6, 3)     # 4.0.1 has the OPT_INF == 1 test commented out: always sub-steps
    dtfine = dt / niter.astype(f32)
    smc0 = list(smc)

    def one(i, c):
        sh2o_, smc_, qd_s, rs_s, rsat_, acc_ = c
        act = i < niter
        pdd = pddum
        rs_i = runsrf0
        if opt_run == 3:
            pdd_n, rs_n = infil(P, dtfine, zsoil, sh2o_, sice, sicemax, qinsur)
            pdd = pdd_n
            rs_i = rs_n
        elif opt_run == 6:
            rs_i, pdd = runoff_vic(P, smc_, zsoil, qinsur, dtfine)
        elif opt_run == 7:
            rs_i, pdd = runoff_xaj(P, smc_, zsoil, fcr[0], qinsur, dtfine)
        acc_n = acc_
        if opt_run == 8:
            rs_i, pdd, acc_n = runoff_dvic(P, o, smc_, sice[0], zsoil, qinsur, dtfine, acc_)
        if opt_run in (6, 7, 8):
            rs_i = W(qinsur > 0., rs_i, runsrf0); pdd = W(qinsur > 0., pdd, pddum)
            acc_n = W(qinsur > 0., acc_n, acc_)
        wdf = []; wcnd = []
        for k in range(NSOIL):
            if o["inf"] == 2:
                a, b = _wdfcnd2(P, sh2o_[k], sicemax, k)
            else:
                a, b = _wdfcnd1(P, smc_[k], fcr[k], k)
            wdf.append(a); wcnd.append(b)
        smx = list(sh2o_) if o["inf"] == 2 else list(smc_)
        denom = [None] * NSOIL; ddz = [None] * NSOIL; dsmdz = [None] * NSOIL; wflux = [None] * NSOIL
        for k in range(NSOIL):
            if k == 0:
                denom[k] = -zsoil[k]
                temp1 = -zsoil[k + 1]
                ddz[k] = 2.0 / temp1
                dsmdz[k] = 2.0 * (smx[k] - smx[k + 1]) / temp1
                wflux[k] = wdf[k] * dsmdz[k] + wcnd[k] - pdd + etrani[k] + qseva
            elif k < NSOIL - 1:
                denom[k] = zsoil[k - 1] - zsoil[k]
                temp1 = zsoil[k - 1] - zsoil[k + 1]
                ddz[k] = 2.0 / temp1
                dsmdz[k] = 2.0 * (smx[k] - smx[k + 1]) / temp1
                wflux[k] = wdf[k] * dsmdz[k] + wcnd[k] - wdf[k - 1] * dsmdz[k - 1] - wcnd[k - 1] + etrani[k]
            else:
                denom[k] = zsoil[k - 1] - zsoil[k]
                if free:
                    qdrain = P["SLOPE"] * wcnd[k]
                elif opt_run == 4:
                    qdrain = (1.0 - fcrmax) * wcnd[k]
                else:
                    qdrain = f32(0.)
                wflux[k] = -(wdf[k - 1] * dsmdz[k - 1]) - wcnd[k - 1] + etrani[k] + qdrain
        ai = []; bi = []; ci = []; rhs = []
        for k in range(NSOIL):
            if k == 0:
                a_ = f32(0.); b_ = wdf[k] * ddz[k] / denom[k]; c_ = -b_
            elif k < NSOIL - 1:
                a_ = -wdf[k - 1] * ddz[k - 1] / denom[k]; c_ = -wdf[k] * ddz[k] / denom[k]; b_ = -(a_ + c_)
            else:
                a_ = -wdf[k - 1] * ddz[k - 1] / denom[k]; c_ = f32(0.); b_ = -(a_ + c_)
            ai.append(a_ * dtfine); bi.append(1. + b_ * dtfine); ci.append(c_ * dtfine); rhs.append(wflux[k] / (-denom[k]) * dtfine)
        sol = rosr12(ai, bi, ci, rhs, jnp.int32(0))
        new = [sh2o_[k] + sol[k] for k in range(NSOIL)]
        # redistribute excess above porosity
        wplus = f32(0.)
        for k in range(NSOIL - 1, 0, -1):
            epore = jnp.fmax(1.E-4, P["SMCMAX"][k] - sice[k])
            wplus = jnp.fmax((new[k] - epore), 0.0) * dz[k]
            new[k] = jnp.fmin(epore, new[k])
            new[k - 1] = new[k - 1] + wplus / dz[k - 1]
        epore = jnp.fmax(1.E-4, P["SMCMAX"][0] - sice[0])
        wplus = jnp.fmax((new[0] - epore), 0.0) * dz[0]
        new[0] = jnp.fmin(epore, new[0])
        up = wplus > 0.0
        n2 = list(new)
        n2[1] = n2[1] + wplus / dz[1]
        for k in range(1, NSOIL - 1):
            epore = jnp.fmax(1.E-4, P["SMCMAX"][k] - sice[k])
            wplus = jnp.fmax((n2[k] - epore), 0.0) * dz[k]
            n2[k] = jnp.fmin(epore, n2[k])
            n2[k + 1] = n2[k + 1] + wplus / dz[k + 1]
        epore = jnp.fmax(1.E-4, P["SMCMAX"][NSOIL - 1] - sice[NSOIL - 1])
        wplus2 = jnp.fmax((n2[NSOIL - 1] - epore), 0.0) * dz[NSOIL - 1]
        n2[NSOIL - 1] = jnp.fmin(epore, n2[NSOIL - 1])
        new = [W(up, n2[k], new[k]) for k in range(NSOIL)]
        wplus = W(up, wplus2, wplus)
        smc_new = [new[k] + sice[k] for k in range(NSOIL)]
        out_sh = [W(act, new[k], sh2o_[k]) for k in range(NSOIL)]
        out_smc = [W(act, smc_new[k], smc_[k]) for k in range(NSOIL)]
        return (out_sh, out_smc, W(act, qd_s + qdrain, qd_s), W(act, rs_s + rs_i, rs_s), W(act, rsat_ + wplus, rsat_), W(act, acc_n, acc_)), \
            [W(act, wcnd[k], 0.) for k in range(NSOIL)]

    c = (sh2o, smc0, f32(0.), f32(0.), rsat, f32(1.0e-6))
    wcnd_last = [f32(0.)] * NSOIL
    for i in range(6):
        c, wc = one(i, c)
        wcnd_last = [W(i < niter, wc[k], wcnd_last[k]) for k in range(NSOIL)]
    sh2o, smc, qd_s, rs_s, rsat, _ = c
    niterf = niter.astype(f32)
    runsrf = rs_s / niterf
    qdrain = qd_s / niterf
    runsrf = runsrf * 1000. + rsat * 1000. / dt
    qdrain = qdrain * 1000.
    if o.get("tdrn", 0) > 0 and opt_run == 3 and td is not None:
        sh2o, _, wt_new = tile_drainage(P, o, dt, zsoil, dz, sh2o, sice, wcnd_last, td["frac"], td["dx"])
        if wt_new is not None:
            zwt = W(wt_new[0], wt_new[1], zwt)
    if opt_run == 2:
        wtsub = f32(0.)
        for k in range(NSOIL):
            wtsub = wtsub + wcnd_last[k] * dz[k]
        sh2o = list(sh2o)
        for k in range(NSOIL):
            mh2o = runsub * dt * (wcnd_last[k] * dz[k]) / wtsub
            sh2o[k] = sh2o[k] - mh2o / (dz[k] * 1000.)
    if opt_run != 1:
        sh2o = list(sh2o)
        mliq = [sh2o[iz] * dz[iz] * 1000. for iz in range(NSOIL)]
        watmin = 0.01
        for iz in range(NSOIL - 1):
            xs = W(mliq[iz] < 0., watmin - mliq[iz], 0.)
            mliq[iz] = mliq[iz] + xs
            mliq[iz + 1] = mliq[iz + 1] - xs
        xs = W(mliq[NSOIL - 1] < watmin, watmin - mliq[NSOIL - 1], 0.)
        mliq[NSOIL - 1] = mliq[NSOIL - 1] + xs
        runsub = runsub - xs / dt
        sh2o = [mliq[iz] / (dz[iz] * 1000.) for iz in range(NSOIL)]
    return sh2o, smc, runsrf, qdrain, wcnd_last, fcrmax, runsub, zwt


def groundwater(P, dt, sice, zsoil, wcnd, fcrmax, sh2o, zwt, wa, wt):
    ROUS = P["ROUS"]
    CMIC = P["CMIC"]
    dzmm = [-zsoil[0] * 1.E3] + [1.E3 * (zsoil[k - 1] - zsoil[k]) for k in range(1, NSOIL)]
    znode = [-zsoil[0] / 2.] + [-zsoil[k - 1] + 0.5 * (zsoil[k - 1] - zsoil[k]) for k in range(1, NSOIL)]
    smc = [sh2o[k] + sice[k] for k in range(NSOIL)]
    mliq = [sh2o[k] * dzmm[k] for k in range(NSOIL)]
    epore = [jnp.fmax(0.01, P["SMCMAX"][k] - sice[k]) for k in range(NSOIL)]
    hk = [1.E3 * wcnd[k] for k in range(NSOIL)]
    iwt = jnp.int32(NSOIL)           # 1-based
    found = jnp.bool_(False)
    for iz in range(2, NSOIL + 1):
        hit = (~found) & (zwt <= -zsoil[iz - 1])
        iwt = W(hit, iz - 1, iwt)
        found = found | hit
    hk_i = at(hk, iwt - 1)
    decay = at(list(P["BEXP"]), iwt - 1) / 3.0                                 # 4.4/5.0 (GY Niu 2022)
    qdis = (1.0 - fcrmax) * (hk_i * 1.E3 * jnp.exp(f32(3.0))) * jnp.exp(-P["TIMEAN"]) * jnp.exp(-decay * zwt)
    smc_i = at(smc, iwt - 1)
    smcmax_i = at(list(P["SMCMAX"]), iwt - 1)
    s_node = jnp.fmin(1.0, smc_i / smcmax_i)
    s_node = jnp.fmax(s_node, 0.01)
    smpfz = -at(list(P["PSISAT"]), iwt - 1) * 1000. * jnp.power(s_node, -at(list(P["BEXP"]), iwt - 1))
    smpfz = jnp.fmax(-120000.0, CMIC * smpfz)
    dks_i = at(list(P["DKSAT"]), iwt - 1) * 1.E3
    ka = 2.0 * (hk_i * dks_i) / (hk_i + dks_i)                                 # harmonic average (5.0)
    wh_zwt = -zwt * 1.E3
    zn_i = at(znode, iwt - 1)
    wh = smpfz - zn_i * 1.E3
    qin = -ka * (wh_zwt - wh) / ((zwt - zn_i) * 1.E3)
    qin = jnp.fmax(-10.0 / dt, jnp.fmin(10. / dt, qin))
    wt = wt + (qin - qdis) * dt
    bottom = iwt == NSOIL
    # bottom-layer case
    wa_b = wa + (qin - qdis) * dt
    wt_b = wa_b
    zwt_b = (-zsoil[NSOIL - 1] + 25.) - wa_b / 1000. / ROUS
    mliq_b = list(mliq)
    mliq_b[NSOIL - 1] = mliq[NSOIL - 1] - qin * dt
    mliq_b[NSOIL - 1] = mliq_b[NSOIL - 1] + jnp.fmax(0., (wa_b - 5000.))
    wa_b2 = jnp.fmin(wa_b, 5000.)
    # interior case
    ws = f32(0.)
    for iz in range(1, NSOIL + 1):
        ws = ws + W(iz >= iwt + 2, epore[iz - 1] * dzmm[iz - 1], 0.)
    zs_n = at(list(zsoil), jnp.clip(iwt, 0, NSOIL - 1))           # zsoil(iwt+1) 1-based
    ep_n = at(epore, jnp.clip(iwt, 0, NSOIL - 1))
    zwt_i = -zs_n - (wt - ROUS * 1000. * 25. - ws) / ep_n / 1000.
    wtsub = f32(0.)
    for iz in range(NSOIL):
        wtsub = wtsub + hk[iz] * dzmm[iz]
    mliq_i = [W(wtsub != 0.0, mliq[iz] - qdis * dt * hk[iz] * dzmm[iz] / jnp.where(wtsub != 0.0, wtsub, 1.0), mliq[iz])
              for iz in range(NSOIL)]
    zwt = W(bottom, zwt_b, zwt_i)
    wt = W(bottom, wt_b, wt)
    wa = W(bottom, wa_b2, wa)
    mliq = [W(bottom, mliq_b[iz], mliq_i[iz]) for iz in range(NSOIL)]
    zwt = jnp.fmax(1.5, zwt)
    watmin = 0.01
    for iz in range(NSOIL - 1):
        xs = W(mliq[iz] < 0., watmin - mliq[iz], 0.)
        mliq[iz] = mliq[iz] + xs
        mliq[iz + 1] = mliq[iz + 1] - xs
    xs = W(mliq[NSOIL - 1] < watmin, watmin - mliq[NSOIL - 1], 0.)
    mliq[NSOIL - 1] = mliq[NSOIL - 1] + xs
    wa = wa - xs
    wt = wt - xs
    sh2o = [mliq[iz] / dzmm[iz] for iz in range(NSOIL)]
    return sh2o, zwt, wa, wt, qin, qdis


# ----------------------------------------------------------------------------------------------
def water(P, o, dt, zsoil, qvap, qdew, fcev, fctr, bdfall, frozen_canopy, frozen_ground, qsnow, qrain, snowhin,
          ponding, elai, esai, btrani, ficeold, imelt, sfctmp, isnow, canliq, canice, tv, snowh, sneqv, snice, snliq,
          stc, zsnso, sh2o, smc, sice, zwt, wa, wt, dzsnso, wslake, fwet, rivsto, fldsto, fldfrc, fveg, td=None, ir=None, cropland=None):
    """returns dict of updated water state and fluxes"""
    canliq, canice, tv, fwet, cmc, ecan, etran = canwater(P, dt, fcev, fctr, elai, esai, bdfall, frozen_canopy,
                                                         canliq, canice, tv, fwet, fveg)
    has = sneqv > 0.
    qsnsub = W(has, jnp.fmin(qvap, sneqv / dt), 0.)
    subsnow = qsnsub
    qseva = qvap - qsnsub
    qsnfro = W(has, qdew, 0.)
    qsdew = qdew - qsnfro
    sh2o = list(sh2o); sice = list(sice)
    sw = snowwater(P, dt, zsoil, sfctmp, snowhin, qsnow, qsnfro, qsnsub, qrain, ficeold, imelt, isnow, snowh, sneqv,
                   snice, snliq, sh2o[0], sice[0], stc, zsnso, dzsnso)
    isnow = sw["isnow"]; snowh = sw["snowh"]; sneqv = sw["sneqv"]; snice = sw["snice"]; snliq = sw["snliq"]
    stc = sw["stc"]; zsnso = sw["zsnso"]; dzsnso = sw["dzsnso"]
    sh2o[0] = sw["sh2o1"]; sice[0] = sw["sice1"]
    qsnbot = sw["qsnbot"]; snoflow = sw["snoflow"]; ponding1 = sw["ponding1"]; ponding2 = sw["ponding2"]
    # frozen ground surface exchange
    sice1 = sice[0] + (qsdew - qseva) * dt / (dzsnso[3] * 1000.)
    sh2o1 = sh2o[0]
    neg = sice1 < 0.
    sh2o1 = W(neg, sh2o1 + sice1, sh2o1)
    sice1 = W(neg, 0., sice1)
    sice[0] = W(frozen_ground, sice1, sice[0])
    sh2o[0] = W(frozen_ground, sh2o1, sh2o[0])
    qsdew = W(frozen_ground, 0.0, qsdew)
    qseva = W(frozen_ground, 0.0, qseva)
    qinsur = (ponding + ponding1 + ponding2) / dt * 0.001
    qinsur = qinsur + W(isnow == 0, (qsnbot + qsdew + qrain) * 0.001, (qsnbot + qsdew) * 0.001)
    qseva = qseva * 0.001
    etrani = [etran * btrani[k] * 0.001 for k in range(NSOIL)]
    qinsur = W(fldfrc == 1, qinsur + (rivsto + fldsto), qinsur)
    dz4 = dzsnso[3:]
    ir_out = {}
    if ir is not None:
        smc1_ir = sh2o[0] + sice[0]                     # SoilMoisture(1) at this point of the step
        ir = dict(ir)
        flood_on = cropland & (ir["amt_f"] > 0.0)
        amt_f, rate_f = irrigation_flood(P, dt, ir, sh2o[0], smc1_ir, sice)
        ir["amt_f"] = W(flood_on, amt_f, ir["amt_f"])
        qinsur = qinsur + W(flood_on, rate_f, 0.0)
        micro_on = cropland & (ir["amt_m"] > 0.0)
        amt_m, sh1 = irrigation_micro(P, dt, ir, zsoil, sh2o, smc1_ir, sice)
        ir["amt_m"] = W(micro_on, amt_m, ir["amt_m"])
        sh2o[0] = W(micro_on, sh1, sh2o[0])
        ir_out = ir
    sh2o, smc, runsrf, qdrain, wcnd, fcrmax, runsub, zwt = soilwater(P, o, dt, zsoil, dz4, qinsur, qseva, etrani, sice, sh2o, smc, zwt, td)
    if o["run"] == 1:
        sh2o, zwt, wa, wt, qin, qdis = groundwater(P, dt, sice, zsoil, wcnd, fcrmax, sh2o, zwt, wa, wt)
        runsub = qdis
    elif o["run"] in (3, 4, 6, 7, 8):
        runsub = runsub + qdrain
    smc = [sh2o[k] + sice[k] for k in range(NSOIL)]
    runsub = runsub + snoflow
    runsrf = W(fldfrc == 0, runsrf + (rivsto + fldsto) * 1000., runsrf)
    return dict(canliq=canliq, canice=canice, tv=tv, fwet=fwet, cmc=cmc, ecan=ecan, etran=etran, subsnow=subsnow,
                isnow=isnow, snowh=snowh, sneqv=sneqv, snice=snice, snliq=snliq, stc=stc, zsnso=zsnso, dzsnso=dzsnso,
                sh2o=sh2o, smc=smc, sice=sice, zwt=zwt, wa=wa, wt=wt, runsrf=runsrf, runsub=runsub, qsnbot=qsnbot,
                ponding1=ponding1, ponding2=ponding2, ir=ir_out)


def carbon_nature(P, x, dt, lai, sai, tv, smc, stc1, dzsnso, zsoil, nroot, transfac_acc, psn, igs):
    """BiochemNatureVegMain + CarbonFluxNatureVeg (dveg 2, 5, 6), run after the water step. nonvegetated points zero every pool."""
    veg_t = x["VEGTYP"]
    bare = (veg_t == P["ISWATER"]) | (veg_t == P["ISBARREN"]) | (veg_t == P["ISICE"]) | (P["URBAN_FLAG"] > 0.5)
    lap = P["SLA"] / 1000.0                                   # m2/kg -> m2/g
    stress = 1.0 - transfac_acc
    zroot = at(list(zsoil), nroot - 1)
    rootz = f32(0.)
    for iz in range(NSOIL):
        rootz = rootz + W(iz < nroot, smc[iz] / P["SMCMAX"][iz] * dzsnso[iz + 3] / (-zroot), 0.)
    lfm, rtm, stm, wdm = x["LFMASS"], x["RTMASS"], x["STMASS"], x["WOOD"]
    fcp, scp = x["FASTCP"], x["STBLCP"]
    sapm = 3.0 * 0.001
    lfmin = P["LAIMIN"] / lap
    stmin = P["XSAMIN"] / sapm
    redfac = W(igs == 0.0, 0.5, 1.0)
    fn_ = jnp.fmin(x["FOLN"] / jnp.fmax(1.0e-06, P["FOLNMX"]), 1.0)
    rtemp = jnp.power(P["ARM"], (tv - 298.16) / 10.0)
    resp_leaf = P["RMF25"] * rtemp * fn_ * lai * redfac * (1.0 - stress)
    resp_leafm = jnp.fmin((lfm - lfmin) / dt, resp_leaf * 12.0e-6)
    resp_root = P["RMR25"] * (rtm * 1.0e-3) * rtemp * redfac * 12.0e-6
    resp_stem = P["RMS25"] * ((stm - stmin) * 1.0e-3) * rtemp * redfac * 12.0e-6
    resp_wood = P["RSWOODC"] * jnp.exp(0.08 * (tv - 298.16)) * wdm * P["WDPOOL"]
    cassim = psn * 12.0e-6
    fleaf = jnp.exp(0.01 * (1.0 - jnp.exp(0.75 * lai)) * lai)
    fleaf = W(veg_t == P["EBLFOREST"], jnp.exp(0.01 * (1.0 - jnp.exp(0.50 * lai)) * lai), fleaf)
    fwoodroot = 1.0 - fleaf
    fstem = lai / 10.0 * fleaf
    fleaf = fleaf - fstem
    wcf = W(wdm > 1.0e-6, (1.0 - jnp.exp(-P["BF"] * (P["WRRAT"] * rtm / jnp.fmax(wdm, 1e-30))) / P["BF"]) * P["WDPOOL"], P["WDPOOL"])
    froot = fwoodroot * (1.0 - wcf)
    fwood = fwoodroot * wcf
    to_leaf = P["LTOVRC"] * 5.0e-7 * lfm
    to_stem = P["LTOVRC"] * 5.0e-7 * stm
    to_root = P["RTOVRC"] * rtm
    to_wood = 9.5e-10 * wdm
    dcoef_t = jnp.exp(-0.3 * jnp.fmax(0.0, tv - P["TDLEF"])) * (lfm / 120.0)
    dcoef_w = jnp.exp((stress - 1.0) * P["WSTRC"])
    death_leaf = lfm * 1.0e-6 * (P["DILEFW"] * dcoef_w + P["DILEFC"] * dcoef_t)
    death_stem = stm * 1.0e-6 * (P["DILEFW"] * dcoef_w + P["DILEFC"] * dcoef_t)
    gr_leaf = jnp.fmax(0.0, P["FRAGR"] * (fleaf * cassim - resp_leafm))
    gr_stem = jnp.fmax(0.0, P["FRAGR"] * (fstem * cassim - resp_stem))
    gr_root = jnp.fmax(0.0, P["FRAGR"] * (froot * cassim - resp_root))
    gr_wood = jnp.fmax(0.0, P["FRAGR"] * (fwood * cassim - resp_wood))
    npp_leaf_add = jnp.fmax(0.0, fleaf * cassim - gr_leaf - resp_leafm)
    npp_stem_add = jnp.fmax(0.0, fstem * cassim - gr_stem - resp_stem)
    npp_leaf_add = W(tv < P["TMIN"], 0.0, npp_leaf_add)
    npp_stem_add = W(tv < P["TMIN"], 0.0, npp_stem_add)
    lf_maxchg = (lfm - lfmin) / dt
    st_maxchg = (stm - stmin) / dt
    death_leaf = jnp.fmin(death_leaf, lf_maxchg + npp_leaf_add - to_leaf)
    death_stem = jnp.fmin(death_stem, st_maxchg + npp_stem_add - to_stem)
    npp_leaf = jnp.fmax(npp_leaf_add, -lf_maxchg)
    npp_stem = jnp.fmax(npp_stem_add, -st_maxchg)
    npp_root = froot * cassim - resp_root - gr_root
    npp_wood = fwood * cassim - resp_wood - gr_wood
    lfm_n = lfm + (npp_leaf - to_leaf - death_leaf) * dt
    stm_n = stm + (npp_stem - to_stem - death_stem) * dt
    rtm_n = rtm + (npp_root - to_root) * dt
    to_root = W(rtm_n < 0.0, npp_root, to_root)
    rtm_n = W(rtm_n < 0.0, 0.0, rtm_n)
    wdm_n = (wdm + (npp_wood - to_wood) * dt) * P["WDPOOL"]
    fcp_n = fcp + (to_root + to_leaf + to_stem + to_wood + death_leaf + death_stem) * dt
    mr_t = jnp.power(2.0, (stc1 - 283.16) / 10.0)
    mr_w = rootz / (0.20 + rootz) * 0.23 / (0.23 + rootz)
    resp_soil = mr_w * mr_t * P["MRP"] * jnp.fmax(0.0, fcp_n * 1.0e-3) * 12.0e-6
    decay = 0.1 * resp_soil
    fcp_n = fcp_n - (resp_soil + decay) * dt
    scp_n = scp + decay * dt
    lai_n = jnp.fmax(lfm_n * lap, P["LAIMIN"])
    sai_n = jnp.fmax(stm_n * sapm, P["XSAMIN"])
    z = f32(0.)
    return dict(LFMASS=W(bare, z, lfm_n), RTMASS=W(bare, z, rtm_n), STMASS=W(bare, z, stm_n), WOOD=W(bare, z, wdm_n),
                STBLCP=W(bare, z, scp_n), FASTCP=W(bare, z, fcp_n), lai=W(bare, z, lai_n), sai=W(bare, z, sai_n))


def carbon_crop(P, x, dt, lai, sai, tv, smc, stc1, dzsnso, zsoil, nroot, transfac_acc, psn, t2m):
    """BiochemCropMain: CropGrowDegreeDay + CarbonFluxCrop (crop model option 1), run after the water step on crop tiles."""
    jul = x["JULIAN"]
    plt_, hsd = P["PLTDAY"], P["HSDAY"]
    tc = t2m - 273.15
    tdiff = W(tc < P["GDDTBASE"], 0.0, W(tc >= P["GDDTCUT"], P["GDDTCUT"] - P["GDDTBASE"], tc - P["GDDTBASE"]))
    ipl = W(jul < plt_, 0.0, 1.0)
    ihv = W(jul >= hsd, 0.0, 1.0)
    gdd = (x["GDD"] + tdiff * dt / 86400.0) * ipl * ihv
    pgs = W(gdd > 0.0, 2, 1)
    for k, thr in ((3, "GDDS1"), (4, "GDDS2"), (5, "GDDS3"), (6, "GDDS4"), (7, "GDDS5")):
        pgs = W(gdd >= P[thr], k, pgs)
    pgs = W(jul >= hsd, 8, pgs)
    pgs = W(jul < plt_, 1, pgs)
    st = pgs - 1
    stage = lambda name: jnp.take(P[name], st)
    # BiochemCropMain: water stress and root-zone soil water
    stress = 1.0 - transfac_acc
    zroot = at(list(zsoil), nroot - 1)
    rootz = f32(0.)
    for iz in range(NSOIL):
        rootz = rootz + W(iz < nroot, smc[iz] / P["SMCMAX"][iz] * dzsnso[iz + 3] / (-zroot), 0.)
    lfm, rtm, stm, grm = x["LFMASS"], x["RTMASS"], x["STMASS"], x["GRAIN"]
    scp, fcp = x["STBLCP"], x["FASTCP"]
    sapm = 3.0 * 0.001
    lfmin = P["LAIMIN"] / 0.035
    stmin = P["XSAMIN"] / sapm
    cassim = psn * 12.0e-6
    chassim = psn * 30.0e-6
    fn_ = jnp.fmin(1.0 / jnp.fmax(1.0e-06, P["FOLNMX"]), 1.0)
    rtemp = jnp.power(P["ARM"], (tv - 298.16) / 10.0)
    resp_leaf = P["RMF25"] * rtemp * fn_ * lai * (1.0 - stress)
    resp_leafm = jnp.fmin((lfm - lfmin) / dt, resp_leaf * 30.0e-6)
    resp_root = P["RMR25"] * (rtm * 1.0e-3) * rtemp * 30.0e-6
    resp_stem = P["RMS25"] * (stm * 1.0e-3) * rtemp * 30.0e-6
    resp_grain = P["GRAINMR25"] * (grm * 1.0e-3) * rtemp * 30.0e-6
    fl, fs, fr, fg = stage("LFPT") * chassim, stage("STPT") * chassim, stage("RTPT") * chassim, stage("GRAINPT") * chassim
    gr_leaf = jnp.fmax(0.0, P["FRAGR"] * (fl - resp_leafm))
    gr_stem = jnp.fmax(0.0, P["FRAGR"] * (fs - resp_stem))
    gr_root = jnp.fmax(0.0, P["FRAGR"] * (fr - resp_root))
    gr_grain = jnp.fmax(0.0, P["FRAGR"] * (fg - resp_grain))
    to_leaf = stage("LF_OVRC") * 1.0e-6 * lfm
    to_root = stage("RT_OVRC") * 1.0e-6 * rtm
    to_stem = stage("ST_OVRC") * 1.0e-6 * stm
    dc_t = jnp.exp(-0.3 * jnp.fmax(0.0, tv - P["TDLEF"])) * (lfm / 120.0)
    dc_w = jnp.exp((stress - 1.0) * P["WSTRC"])
    death_leaf = lfm * 1.0e-6 * (stage("DILE_FW") * dc_w + stage("DILE_FC") * dc_t)
    npp_leaf_add = fl - gr_leaf - resp_leafm
    npp_stem_add = fs - gr_stem - resp_stem
    lf_maxchg = (lfm - lfmin) / dt
    st_maxchg = (stm - stmin) / dt
    to_leaf = jnp.fmin(to_leaf, lf_maxchg + npp_leaf_add)
    to_stem = jnp.fmin(to_stem, st_maxchg + npp_stem_add)
    death_leaf = jnp.fmin(death_leaf, lf_maxchg + npp_leaf_add - to_leaf)
    npp_leaf = npp_leaf_add
    npp_stem = npp_stem_add
    npp_root = fr - resp_root - gr_root
    npp_grain = fg - resp_grain - gr_grain
    lfm = lfm + (npp_leaf - to_leaf - death_leaf) * dt
    stm = stm + (npp_stem - to_stem) * dt
    rtm = rtm + (npp_root - to_root) * dt
    grm = grm + npp_grain * dt
    gpp = chassim * 0.4
    cv_l = lfm * (stage("LFCT") * dt / 3600.0)
    cv_s = stm * (stage("STCT") * dt / 3600.0)
    cv_r = rtm * (stage("RTCT") * dt / 3600.0)
    lfm = lfm - cv_l
    stm = stm - cv_s
    rtm = rtm - cv_r
    grm = grm + cv_s + cv_r + cv_l
    to_root = W(rtm < 0.0, npp_root, to_root)
    rtm = W(rtm < 0.0, 0.0, rtm)
    grm = W(grm < 0.0, 0.0, grm)
    fcp_n = fcp + (to_root + to_leaf + to_stem + death_leaf) * dt * 0.4
    mr_t = jnp.power(2.0, (stc1 - 283.16) / 10.0)
    mr_w = rootz / (0.20 + rootz) * 0.23 / (0.23 + rootz)
    resp_soil = mr_w * mr_t * P["MRP"] * jnp.fmax(0.0, fcp_n * 1.0e-3) * 30.0e-6
    decay = 0.1 * resp_soil
    fcp_n = fcp_n - (resp_soil + decay) * dt * 0.4
    scp_n = scp + decay * dt * 0.4
    npp_t = (npp_leaf + npp_stem + npp_root + npp_grain) * 0.4
    resp_pl = (resp_root + resp_grain + resp_leafm + resp_stem + gr_leaf + gr_root + gr_grain + gr_stem) * 0.4
    resp_so = 0.9 * resp_soil * 0.4
    nee = (resp_pl + resp_so - gpp) * 44.0 / 12.0
    lai_n = jnp.fmax(lfm * P["BIO2LAI"], P["LAIMIN"])
    sai_n = jnp.fmax(stm * sapm, P["XSAMIN"])
    harv = (pgs == 8) & ((grm > 0) | (lfm > 0) | (stm > 0) | (rtm > 0))
    lai_n = W(harv, 0.05, lai_n)
    sai_n = W(harv, 0.05, sai_n)
    lfm = W(harv, lfmin, lfm)
    stm = W(harv, stmin, stm)
    rtm = W(harv, 0.0, rtm)
    grm = W(harv, 0.0, grm)
    return dict(LFMASS=lfm, RTMASS=rtm, STMASS=stm, STBLCP=scp_n, FASTCP=fcp_n, GRAIN=grm, GDD=gdd, PGS=pgs.astype(f32), lai=lai_n, sai=sai_n,
                GPP=gpp, NPP=npp_t, NEE=nee)


def sflx_mp(P, x, o):
    """One Noah-MP 4.0.1 time step for one tile.
    P: parameter dict (scalars; BEXP... arrays of 4; LAIM/SAIM 12; RHOL.. 2). x: inputs/states dict. o: static options.
    Returns dict with all post-call quantities (names as in the Fortran dump)."""
    dt = x["DT"]
    sfctmp, sfcprs, psfc, uu, vv, q2 = x["T_ML"], x["P_ML"], x["PSFC"], x["U_ML"], x["V_ML"], x["Q_ML"]
    lwdn, cosz, zref = x["LWDN"], x["COSZ"], x["Z_ML"]
    zsoil = x["ZSOIL"]
    isnow = x["ISNOW"]
    A = atm(P, o, sfcprs, sfctmp, q2, x["PRCPCONV"], x["PRCPNONC"], x["PRCPSHCV"], x["PRCPSNOW"], x["PRCPGRPL"],
            x["PRCPHAIL"], x["SWDN"], cosz)
    swdown = A["swdown"]
    zsnso = list(x["ZSNSO"])
    dzsnso = []
    for p in range(NL):
        a = isnow + 3 <= p
        first = p == isnow + 3
        prev = zsnso[p - 1] if p > 0 else f32(0.)
        dzsnso.append(W(a, W(first, -zsnso[p], prev - zsnso[p]), 0.))
    lai, sai, elai, esai, igs = phenology(P, o, x["VEGTYP"], x["SNDPTH"], x["TV"], x["LAT"], x["YEARLEN"], x["JULIAN"],
                                          x["PLAI"], x["PSAI"], x.get("PGS"))
    if o["dveg"] in (1, 6, 7):          # PhenologyMain: input green vegetation fraction
        fveg = x["FVEG"]
    elif o["dveg"] in (2, 3, 8):        # computed from LAI + SAI
        fveg = 1. - jnp.exp(-0.52 * (lai + sai))
    else:                               # 4, 5, 9: yearly maximum vegetation fraction
        fveg = x["FVGMAX"]
    crop = (P["CROPTYPE"] > 0.5) if o.get("crop", 0) > 0 else None
    if crop is not None:                # crop run: maximum vegetation fraction
        fveg = W(crop, x["FVGMAX"], fveg)
    fveg = W(fveg <= 0.05, 0.05, fveg)
    fveg = W((P["URBAN_FLAG"] > 0.5) | (x["VEGTYP"] == P["ISBARREN"]), 0.0, fveg)
    fveg = W(elai + esai == 0.0, 0.0, fveg)
    ir = None
    cropland = (x["VEGTYP"] == 12) | (x["VEGTYP"] == 14)                      # IrrigationPrepare: MODIFIED_IGBP_MODIS_NOAH croplands
    heat_irri = f32(0.); evap_irri = f32(0.)
    if o.get("irr", 0) > 0:
        sh2o_in = list(x["SMH2O"])
        ir = irrigation_prepare(P, o, x, A["rain"], fveg, lai, sh2o_in, cropland)
        sice_in = [jnp.fmax(0.0, x["SMC"][k] - sh2o_in[k]) for k in range(NSOIL)]
        spr_on = cropland & (ir["amt_s"] > 0.0)
        amt_s, add_rain, hl, ev = irrigation_sprinkler(P, dt, ir, sfctmp, uu, vv, A["eair"], sh2o_in[0], x["SMC"][0], sice_in)
        ir = dict(ir, amt_s=W(spr_on, amt_s, ir["amt_s"]))
        A = dict(A, rain=A["rain"] + W(spr_on, add_rain, 0.0))
        heat_irri = W(spr_on, hl, 0.0); evap_irri = W(spr_on, ev, 0.0)
    PH = precip_heat(P, x["VEGTYP"], dt, uu, vv, elai, esai, fveg, A["bdfall"], A["rain"], A["snow"], A["fp"],
                     x["CANLIQ"], x["CANICE"], x["TV"], sfctmp, x["TG"])
    canliq, canice, fwet = PH["canliq"], PH["canice"], PH["fwet"]
    qsnow = PH["qsnow"]
    tv, tg = x["TV"], x["TG"]
    snowh, sneqv = x["SNDPTH"], x["SWE"]
    stc = list(x["STC"]); smc = list(x["SMC"]); sh2o = list(x["SMH2O"])
    snice = list(x["SNICE"]); snliq = list(x["SNLIQ"])
    sneqvo = x["SNEQVO"]
    # ----------------------------------------------------------- ENERGY
    ur = jnp.fmax(jnp.sqrt(uu ** 2. + vv ** 2.), 1.)
    vai = elai + esai
    veg = vai > 0.
    bdsno = sneqv / jnp.fmax(snowh, 1e-30)
    fmelt = jnp.power(bdsno / 100., P["MFSNO"])
    fsno = W(snowh > 0., jnp.tanh(snowh / (P["SCFFAC"] * jnp.fmax(fmelt, 1e-30))), 0.)      # 4.3: SCFFAC replaces the hard-coded 2.5*z0
    z0mg = P["Z0SOIL"] * (1.0 - fsno) + fsno * P["Z0SNO"]
    zpdg = snowh
    urb = P["URBAN_FLAG"] > 0.5                       # 4.1: bulk urban roughness length and displacement from the table
    z0mg = W(urb, P["Z0MVT"], z0mg)
    zpdg = W(urb, 0.65 * P["HVT"], zpdg)
    zpd_v = 0.65 * P["HVT"]
    zpd_v = W(snowh > zpd_v, snowh, zpd_v)
    z0m = W(veg, P["Z0MVT"], z0mg)
    zpd = W(veg, zpd_v, zpdg)
    zlvl = jnp.fmax(zpd, P["HVT"]) + zref
    zlvl = W(zpdg >= zlvl, zpdg + zref, zlvl)
    cwp = P["CWPVT"]
    df, hcpct, fact = thermoprop(P, isnow, dt, snowh, snice, snliq, dzsnso, smc, sh2o, o.get("tksno", 1))
    R = radiation(P, o, dt, cosz, tg, tv, fsno, qsnow, fwet, elai, esai, smc[0], A["solad"], A["solai"], fveg,
                  x["ALBOLD"], x["TAUSS"], sneqvo, sneqv)
    emv = 1. - jnp.exp(-(elai + esai) / 1.0)
    emg = P["EG"][0] * (1. - fsno) + P["SNOW_EMIS"] * fsno          # 4.3: snow emissivity is a table parameter
    # soil moisture stress
    nroot = P["NROOT"].astype(jnp.int32)
    zsoil_root = at(list(zsoil), nroot - 1)
    btrani = []
    btran = f32(0.)
    for iz in range(NSOIL):
        inroot = iz < nroot
        if o["btr"] == 1:
            gx = (sh2o[iz] - P["SMCWLT"][iz]) / (P["SMCREF"][iz] - P["SMCWLT"][iz])
        else:
            psi = jnp.fmax(PSIWLT, -P["PSISAT"][iz] * jnp.power(jnp.fmax(0.01, sh2o[iz]) / P["SMCMAX"][iz], -P["BEXP"][iz]))
            if o["btr"] == 2:
                gx = (1. - psi / PSIWLT) / (1. + P["PSISAT"][iz] / PSIWLT)
            else:
                gx = 1. - jnp.exp(-5.8 * jnp.log(PSIWLT / psi))
        gx = jnp.fmin(1., jnp.fmax(0., gx))
        b = jnp.fmax(MPE, dzsnso[iz + 3] / (-zsoil_root) * gx)
        b = W(inroot, b, 0.)
        btrani.append(b)
        btran = btran + b
    btran = jnp.fmax(MPE, btran)
    btrani = [W(iz < nroot, btrani[iz] / btran, 0.) for iz in range(NSOIL)]
    bevap = jnp.fmax(0.0, sh2o[0] / P["SMCMAX"][0])
    if o["rsf"] in (1, 4):
        l_rsurf = (-zsoil[0]) * (jnp.exp((1.0 - jnp.fmin(1.0, sh2o[0] / P["SMCMAX"][0])) ** P["RSURF_EXP"]) - 1.0) / (2.71828 - 1.0)
        d_rsurf = 2.2E-5 * P["SMCMAX"][0] * P["SMCMAX"][0] * jnp.power(1.0 - P["SMCWLT"][0] / P["SMCMAX"][0], 2.0 + 3.0 / P["BEXP"][0])
        rsurf = l_rsurf / d_rsurf
        l_pet = (-zsoil[0]) * (jnp.exp((1.0 - jnp.fmin(1.0, P["SMCMAX"][0] / P["SMCMAX"][0])) ** P["RSURF_EXP"]) - 1.0) / (2.71828 - 1.0)
        rsurf_pet = l_pet / d_rsurf
    elif o["rsf"] == 2:
        rsurf = fsno * 1. + (1. - fsno) * jnp.exp(8.25 - 4.225 * bevap)
        rsurf_pet = rsurf
    else:
        rsurf = fsno * 1. + (1. - fsno) * jnp.exp(8.25 - 6.0 * bevap)
        rsurf_pet = rsurf
    if o["rsf"] == 4:
        rsurf = 1. / (fsno * (1. / P["RSURF_SNOW"]) + (1. - fsno) * (1. / jnp.fmax(rsurf, 0.001)))
    rsurf = W((sh2o[0] < 0.01) & (snowh == 0.), 1.E6, rsurf)
    psi1 = -P["PSISAT"][0] * jnp.power(jnp.fmax(0.01, sh2o[0]) / P["SMCMAX"][0], -P["BEXP"][0])
    rhsur = fsno + (1. - fsno) * jnp.exp(psi1 * GRAV / (RW * tg))
    psi_pet = -P["PSISAT"][0] * jnp.power(jnp.fmax(0.01, P["SMCMAX"][0]) / P["SMCMAX"][0], -P["BEXP"][0])
    rhsur_pet = fsno + (1. - fsno) * jnp.exp(psi_pet * GRAV / (RW * tg))
    rsurf = W((P["URBAN_FLAG"] > 0.5) & (snowh == 0.), 1.E6, rsurf)
    frozen_canopy = ~(tv > TFRZ)
    lathev = W(tv > TFRZ, HVAP, HSUB)
    gammav = CPAIR * sfcprs / (0.622 * lathev)
    frozen_ground = ~(tg > TFRZ)
    latheag = W(tg > TFRZ, HVAP, HSUB)
    gammag = CPAIR * sfcprs / (0.622 * latheag)
    pf = isnow + 3
    stc1 = at(stc, pf); df1 = at(df, pf); dz1 = at(dzsnso, pf)
    cm0, ch0 = x["CM"], x["CH"]
    cdict = dict(fveg=fveg, vai=vai, ur=ur, sfctmp=sfctmp, rhoair=A["rhoair"], zlvl=zlvl, zpd=zpd, z0m=z0m, z0mg=z0mg,
                 emv=emv, emg=emg, lwdn=lwdn, rsurf=rsurf, rsurf_pet=rsurf_pet, rhsur=rhsur, rhsur_pet=rhsur_pet,
                 laisun=R["laisun"], laisha=R["laisha"], tg=tg, eair=A["eair"], psfc=psfc, cwp=cwp, tah=x["TAH"],
                 eah=x["EAH"], tv=tv, cm=cm0, ch=ch0, qair=A["qair"], parsun=R["parsun"], parsha=R["parsha"],
                 foln=x["FOLN"], o2air=x["O2PP"], co2air=x["CO2PP"], igs=igs, btran=btran, sfcprs=sfcprs, fwet=fwet,
                 gammav=gammav, gammag=gammag, canliq=canliq, canice=canice, lathev=lathev, dt=dt, sav=R["sav"],
                 sag=R["sag"], pahv=PH["pahv"], pahg=PH["pahg"], df1=df1, dz1=dz1, stc1=stc1, snowh=snowh, fsno=fsno, thair=A["thair"])
    VF = vege_flux(P, o, cdict, pet=False)
    VP = vege_flux(P, o, cdict, pet=True)
    bdict = dict(sfctmp=sfctmp, rhoair=A["rhoair"], ur=ur, snowh=snowh, zlvl=zlvl, zpdg=zpdg, z0mg=z0mg, emg=emg,
                 lwdn=lwdn, rsurf=rsurf, rhsur=rhsur, gammag=gammag, latheag=latheag, tg=tg, cm=cm0, ch=ch0,
                 qair=A["qair"], eair=A["eair"], psfc=psfc, sag=R["sag"], pahb=PH["pahb"], df1=df1, dz1=dz1, stc1=stc1, fsno=fsno, thair=A["thair"])
    BF = bare_flux(P, o, bdict)
    vv_ok = veg & (fveg > 0)
    fv_ = fveg
    one = 1.0 - fveg
    tgb = BF["tgb"]
    fira = W(vv_ok, fv_ * VF["irg"] + one * BF["irb"] + VF["irc"], BF["irb"])
    fsh = W(vv_ok, fv_ * VF["shg"] + one * BF["shb"] + VF["shc"], BF["shb"])
    fgev = W(vv_ok, fv_ * VF["evg"] + one * BF["evb"], BF["evb"])
    ssoil = W(vv_ok, fv_ * VF["ghv"] + one * BF["ghb"], BF["ghb"])
    fcev = W(vv_ok, VF["evc"], 0.)
    fctr = W(vv_ok, VF["tr"], 0.)
    pah = W(vv_ok, fv_ * PH["pahg"] + one * PH["pahb"] + PH["pahv"], PH["pahb"])
    tg_n = W(vv_ok, fv_ * VF["tg"] + one * tgb, tgb)
    cm_n = W(vv_ok, fv_ * VF["cm"] + one * BF["cm"], BF["cm"])
    ch_n = W(vv_ok, fv_ * VF["ch"] + one * BF["ch"], BF["ch"])
    z0wrf = W(vv_ok, z0m, z0mg)
    fgev_pet = W(vv_ok, fv_ * VP["evg"] + one * BF["evb"], BF["evb"])
    fcev_pet = W(vv_ok, VP["evc"], 0.)
    fctr_pet = W(vv_ok, VP["tr"], 0.)
    tv_n = W(vv_ok, VF["tv"], tv)
    tah_n = W(vv_ok, VF["tah"], x["TAH"])
    eah_n = W(vv_ok, VF["eah"], x["EAH"])
    rssun = W(vv_ok, VF["rssun"], 0.)             # 5.0 reports the stomatal resistance of the main vegetated pass (4.0.1: of the PET pass)
    rssha = W(vv_ok, VF["rssha"], 0.)
    tgv = W(vv_ok, VF["tg"], tgb)
    tgv_pre = tgv
    chv = W(vv_ok, VF["ch"], BF["ch"])
    fire = lwdn + fira
    emissi = fveg * (emg * (1 - emv) + emv + emv * (1 - emv) * (1 - emg)) + (1 - fveg) * emg
    emissi = W(vv_ok, emissi, emg)
    trad = jnp.power((fire - (1 - emissi) * lwdn) / (emissi * SB), 0.25)
    psnsun = W(vv_ok, VF["psnsun"], 0.)
    psnsha = W(vv_ok, VF["psnsha"], 0.)
    apar = R["parsun"] * R["laisun"] + R["parsha"] * R["laisha"]
    psn = psnsun * R["laisun"] + psnsha * R["laisha"]
    stc_n = tsnosoi(P, o, isnow, x["TBOT"], zsnso, ssoil, df, hcpct, dt, snowh, stc)
    if o["stc"] == 2:             # adjusting snow surface temperature after the soil/snow temperature solve
        adj2 = (snowh > 0.05) & (tg_n > TFRZ)
        tgv_a = f32(TFRZ) + 0. * tgv_pre
        tgv = W(adj2, tgv_a, tgv_pre)
        tgb = W(adj2, tgv_a, tgb)
        tg_n = W(adj2, W(vv_ok, fv_ * tgv_a + one * tgv_a, tgv_a), tg_n)
    pc = phasechange(P, o, isnow, dt, fact, dzsnso, hcpct, stc_n, snice, snliq, sneqv, snowh, smc, sh2o)
    stc, snice, snliq, sneqv, snowh, smc, sh2o, qmelt, imelt, ponding = pc
    tg = tg_n; tv = tv_n
    sice = [jnp.fmax(0.0, smc[k] - sh2o[k]) for k in range(NSOIL)]
    sneqvo_n = sneqv
    qvap = jnp.fmax(fgev / latheag, 0.)
    qdew = jnp.abs(jnp.fmin(fgev / latheag, 0.))
    edir = qvap - qdew
    # ----------------------------------------------------------- WATER
    Wt = water(P, o, dt, zsoil, qvap, qdew, fcev, fctr, A["bdfall"], frozen_canopy, frozen_ground, qsnow, PH["qrain"],
               PH["snowhin"], ponding, elai, esai, btrani, x["FICEOLD"], imelt, sfctmp, isnow, canliq, canice, tv,
               snowh, sneqv, snice, snliq, stc, zsnso, sh2o, smc, sice, x["ZWT"], x["WA"], x["WT"], dzsnso, x["WSLAKE"],
               fwet, x["rivsto"], x["fldsto"], x["fldfrc"], fveg,
               td=dict(frac=x["TD_FRACTION"], dx=x["DX"]) if o.get("tdrn", 0) > 0 else None, ir=ir, cropland=cropland)
    if ir is not None:
        ir = Wt["ir"]
        irri_out = dict(IRWATSI=ir["amt_s"], IRWATMI=ir["amt_m"], IRWATFI=ir["amt_f"], IRNUMSI=ir["cs"], IRNUMMI=ir["cm"], IRNUMFI=ir["cf"],
                        IRRSPLH=x.get("IRRSPLH", f32(0.)) + heat_irri * dt, EVAPIRRI=evap_irri)          # IRRSPLH accumulates [J/m2]
        fsh = fsh - W(cropland & (x["IRFRACT"] >= P["IRR_FRAC"]), heat_irri, 0.0)
    else:
        irri_out = dict(IRWATSI=x["IRWATSI"], IRWATMI=x["IRWATMI"], IRWATFI=x["IRWATFI"], IRNUMSI=x["IRNUMSI"], IRNUMMI=x["IRNUMMI"],
                        IRNUMFI=x["IRNUMFI"], IRRSPLH=f32(0.), EVAPIRRI=f32(0.)) if "IRWATSI" in x else {}
    snowh = Wt["snowh"]; sneqv = Wt["sneqv"]
    if o["dveg"] in (2, 5, 6):                              # FlagDynamicVeg: BiochemNatureVegMain after WaterMain
        CB = carbon_nature(P, x, dt, lai, sai, Wt["tv"], Wt["smc"], Wt["stc"][3], dzsnso, zsoil, nroot, btran, psn, igs)
        lai, sai = CB["lai"], CB["sai"]
    else:
        CB = dict(LFMASS=x["LFMASS"], RTMASS=x["RTMASS"], STMASS=x["STMASS"], WOOD=x["WOOD"], STBLCP=x["STBLCP"], FASTCP=x["FASTCP"])
    crop_out = {}
    if crop is not None:
        t2m = W(vv_ok, fveg * VF["t2mv"] + (1.0 - fveg) * BF["t2mb"], BF["t2mb"])
        CC = carbon_crop(P, x, dt, lai, sai, Wt["tv"], Wt["smc"], Wt["stc"][3], dzsnso, zsoil, nroot, btran, psn, t2m)
        # crop tiles take the crop model's states; other tiles keep the ones of the nature-vegetation path
        for k in ("LFMASS", "RTMASS", "STMASS", "STBLCP", "FASTCP"):
            CB[k] = W(crop, CC[k], CB[k])
        lai, sai = W(crop, CC["lai"], lai), W(crop, CC["sai"], sai)
        crop_out = dict(GRAIN=W(crop, CC["GRAIN"], x["GRAIN"]), GDD=W(crop, CC["GDD"], x["GDD"]), PGS=W(crop, CC["PGS"], x["PGS"]),
                        GPP=W(crop, CC["GPP"], f32(0.)), NPP=W(crop, CC["NPP"], f32(0.)), NEE=W(crop, CC["NEE"], f32(0.)))
    clean = (snowh <= 1.E-6) | (sneqv <= 1.E-6)          # 5.0: SWE limit 1e-6 (4.0.1: 1e-3)
    snowh = W(clean, 0.0, snowh)
    sneqv = W(clean, 0.0, sneqv)
    albedo_ = W(swdown != 0., R["fsr"] / jnp.where(swdown != 0., swdown, 1.0), -999.9)
    relsmc = [(Wt["smc"][k] - P["SMCWLT"][k]) / (P["SMCMAX"][k] - P["SMCWLT"][k]) for k in range(NSOIL)]
    out = dict(ALBOLD=R["albold"], SNEQVO=sneqvo_n, TAH=tah_n, EAH=eah_n, FWET=Wt["fwet"], CANLIQ=Wt["canliq"],
               CANICE=Wt["canice"], TV=Wt["tv"], TG=tg, QSFC1D=BF["qsfc"], QSNOW=qsnow, ISNOW=Wt["isnow"], SNDPTH=snowh,
               SWE=sneqv, ZWT=Wt["zwt"], WA=Wt["wa"], WT=Wt["wt"], PLAI=lai, PSAI=sai, LFMASS=CB["LFMASS"], RTMASS=CB["RTMASS"],
               STMASS=CB["STMASS"], WOOD=CB["WOOD"], STBLCP=CB["STBLCP"], FASTCP=CB["FASTCP"], CM=cm_n, CH=ch_n,
               TAUSS=R["tauss"], Z0WRF=z0wrf, FSA=R["fsa"], FSR=R["fsr"], FIRA=fira, FSH=fsh, SSOIL=ssoil,
               FCEV=fcev, FGEV=fgev, FCTR=fctr, ECAN=Wt["ecan"], ETRAN=Wt["etran"], ESOIL=edir, TRAD=trad,
               SUBSNOW=Wt["subsnow"], TGB=tgb, TGV=tgv, T2MV=W(vv_ok, VF["t2mv"], 0.), T2MB=BF["t2mb"],
               Q2MV=W(vv_ok, VP["q2v"], 0.), Q2MB=BF["q2b"], RUNSF=Wt["runsrf"], RUNSB=Wt["runsub"], APAR=apar, PSN=psn,
               SAV=R["sav"], SAG=R["sag"], FSNO=fsno, SALB=albedo_, QSNBOT=Wt["qsnbot"], PONDING=ponding,
               PONDING1=Wt["ponding1"], PONDING2=Wt["ponding2"], RSSUN=rssun, RSSHA=rssha, BGAP=R["bgap"],
               WGAP=R["wgap"], CHV=chv, CHB=BF["ch"], EMISSI=emissi, SHG=W(vv_ok, VF["shg"], 0.),
               SHC=W(vv_ok, VF["shc"], 0.), SHB=BF["shb"], EVG=W(vv_ok, VF["evg"], 0.), EVB=BF["evb"],
               GHV=W(vv_ok, VF["ghv"], 0.), GHB=BF["ghb"], IRG=W(vv_ok, VF["irg"], 0.), IRC=W(vv_ok, VF["irc"], 0.),
               IRB=BF["irb"], TR=W(vv_ok, VF["tr"], 0.), EVC=W(vv_ok, VF["evc"], 0.), FGEV_PET=fgev_pet,
               FCEV_PET=fcev_pet, FCTR_PET=fctr_pet, CHLEAF=W(vv_ok, VP["chleaf"], 0.), CHUC=W(vv_ok, VP["chuc"], 0.),
               CHV2=W(vv_ok, VP["chv2"], 0.), CHB2=BF["chb2"], FPICE=A["fpice"], PAHV=PH["pahv"], PAHG=PH["pahg"],
               PAHB=PH["pahb"], PAH=pah, LAISUN=R["laisun"], LAISHA=R["laisha"], RB=f32(0.),
               STC=Wt["stc"], SMH2O=Wt["sh2o"], SMC=Wt["smc"], ZSNSO=Wt["zsnso"], SNICE=Wt["snice"], SNLIQ=Wt["snliq"],
               RELSMC=relsmc, FVEGMP=fveg, **irri_out, **crop_out)
    return out

