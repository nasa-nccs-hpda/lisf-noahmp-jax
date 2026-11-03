"""JAX port of the LISF Noah-MP 3.6 column physics (NOAHMP_SFLX_36 and the routines it calls).

Derived from noahmp401_jax.py (the Noah-MP 4.0.1 port) and checked routine by routine against
lis/surfacemodels/land/noahmp.3.6/module_sf_noahmplsm_36.F90 (unmodified, read-only). Differences of 3.6 relative to 4.0.1 that matter here:
rain/snow partition and canopy interception are done inside CANWATER (during WATER) with QPRECC/QPRECL = 0.1/0.9 PRCP; the ENERGY step uses the
snowfall QSNOW and the canopy water of the PREVIOUS step; FVEG floor 0.01; melt factor M = 2.5 and bare-ground roughness 0.01; UC without the
displacement height; CLASS snow albedo constants; SOILWATER keeps SMC fixed over the fine sub-steps and SSTEP does not redistribute upwards;
snow depth bookkeeping in SNOWH2O, SNOWWATER and PHASECHANGE; no PET calls, no precipitation heat (PAH), no OPT_STC 3, no crop/GECROS.
Scope: dveg=4, crs=1, btr=1..3, run=1, sfc=1, frz=1, inf=1, rad=1..3, alb=1|2, snf=1..3, tbot=1|2, stc=1, soil color 4,
IST=1 (land), ICE=0, 4 soil layers, 3 snow layers; the 3.6 land-ice branch (ICE=1) is not ported.

Style: see noahmp401_jax.py (layers as Python lists of 7 scalars, jnp.where selects, float32).
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
Z0_BARE = 0.01
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
def atm(P, o, sfcprs, sfctmp, q2, prcp, soldn, cosz):
    pair = sfcprs
    thair = sfctmp * jnp.power(sfcprs / pair, RAIR / CPAIR)
    qair = q2
    eair = qair * sfcprs / (0.622 + 0.378 * qair)
    rhoair = (sfcprs - 0.378 * eair) / (RAIR * sfctmp)
    swdown = W(cosz <= 0., 0., soldn)
    solad = [swdown * 0.7 * 0.5, swdown * 0.7 * 0.5]
    solai = [swdown * 0.3 * 0.5, swdown * 0.3 * 0.5]
    qprecc = 0.10 * prcp
    qprecl = 0.90 * prcp
    return dict(thair=thair, qair=qair, eair=eair, rhoair=rhoair, swdown=swdown, solad=solad, solai=solai,
                qprecc=qprecc, qprecl=qprecl)


def phenology(P, o, vegtyp, snowh, tv, lat, yearlen, julian, lai, sai):
    """Noah-MP 3.6 PHENOLOGY, dveg 1|3|4 (table LAI/SAI). returns lai, sai, elai, esai, igs"""
    yl = f32(yearlen)
    day = W(lat >= 0., julian, jnp.mod(julian + (0.5 * yl), yl))
    t = 12. * day / yl
    it1 = jnp.floor(t + 0.5).astype(jnp.int32)
    it2 = it1 + 1
    wt1 = (it1.astype(f32) + 0.5) - t
    wt2 = 1. - wt1
    it1 = W(it1 < 1, 12, it1)
    it2 = W(it2 > 12, 1, it2)
    lai = wt1 * P["LAIM"][it1 - 1] + wt2 * P["LAIM"][it2 - 1]
    sai = wt1 * P["SAIM"][it1 - 1] + wt2 * P["SAIM"][it2 - 1]
    sai = W(sai < 0.05, 0.0, sai)
    lai = W((lai < 0.05) | (sai == 0.0), 0.0, lai)
    kill = (vegtyp == P["ISWATER"]) | (vegtyp == P["ISBARREN"]) | (vegtyp == P["ISICE"]) | (P["URBAN_FLAG"] > 0.5)
    lai = W(kill, 0., lai)
    sai = W(kill, 0., sai)
    db = jnp.fmin(jnp.fmax(snowh - P["HVB"], 0.), P["HVT"] - P["HVB"])
    fb = db / jnp.fmax(1.e-06, P["HVT"] - P["HVB"])
    snowhc = P["HVT"] * jnp.exp(-snowh / 0.2)
    fb2 = jnp.fmin(snowh, snowhc) / jnp.fmax(snowhc, 1e-30)
    fb = W((P["HVT"] > 0.) & (P["HVT"] <= 1.0), fb2, fb)
    elai = lai * (1. - fb)
    esai = sai * (1. - fb)
    esai = W(esai < 0.05, 0.0, esai)
    elai = W((elai < 0.05) | (esai == 0.0), 0.0, elai)
    igs = W(tv > P["TMIN"], 1., 0.)
    return lai, sai, elai, esai, igs


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


def thermoprop(P, isnow, dt, snowh, snice, sniq, dz, smc, sh2o):
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
        df[p] = W(a, 3.2217E-6 * bdsnoi ** 2., 0.)
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
        alb = 0.55 + (albold - 0.55) * jnp.exp(-0.01 * dt / 3600.)
        alb = W(qsnow > 0., alb + jnp.fmin(qsnow * dt, P["SWEMX"]) * (0.84 - alb) / P["SWEMX"], alb)
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
    """Noah-MP 3.6 SFCDIF2 (Chen et al. 1997, OPT_SFC = 2; ILECH = 0). Unlike 4.0.1: no extra limits on ZETAU/ZETAT, SIMM/SIMH are not clamped (zero -> CXCH). akms/akhs come in and go out multiplied by the wind speed
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
    zetau_s = zetau
    zetat_s = zetat
    simm_s = 5. * zetalu_s - 5. * zetau_s + rlogu
    simh_s = 5. * zetalt_s - 5. * zetat_s + rlogt
    simm = W(neg, simm_u, simm_s)
    simh = W(neg, simh_u, simh_s)
    ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
    ustark = ustar * vkrm
    akms = W(simm != 0.0, jnp.fmax(ustark / jnp.where(simm != 0.0, simm, 1.0), cxch), cxch)
    akhs = W(simh != 0.0, jnp.fmax(ustark / jnp.where(simh != 0.0, simh, 1.0), cxch), cxch)
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
    rb = tmprb * jnp.sqrt(P["DLEAF"] / uc)          # 3.6 has no lower bound of 100 s/m on RB
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
    q2 = q2 / (1.0 + q2)
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
    vaie = jnp.fmin(6., vai / fveg)
    laisune = jnp.fmin(6., g("laisun") / fveg)
    laishae = jnp.fmin(6., g("laisha") / fveg)
    estg0, _ = _esat_pair(g("tg"))
    hcan = P["HVT"]
    uc = ur * jnp.log(hcan / z0m) / jnp.log(zlvl / z0m)
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
            rs_su, ps_su = canres(P, g("parsun"), s["tv"], g("btran"), s["eah"], g("sfcprs"))
            rs_sh, ps_sh = canres(P, g("parsha"), s["tv"], g("btran"), s["eah"], g("sfcprs"))
        else:
            rs_su, ps_su = stomata(P, g("parsun"), g("foln"), s["tv"], estv, s["eah"], sfctmp, g("sfcprs"), g("o2air"),
                                   g("co2air"), g("igs"), g("btran"), rb)
            rs_sh, ps_sh = stomata(P, g("parsha"), g("foln"), s["tv"], estv, s["eah"], sfctmp, g("sfcprs"), g("o2air"),
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
        a = fveg * (4. * cir * tv ** 3 + csh + (cev + ctr) * destv)
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
    adj = (g("snowh") > 0.05) & (tg > TFRZ) & (o["stc"] == 1)
    tg_a = f32(TFRZ)
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
            cm = W(snowh > 0., jnp.fmin(0.01, cm), cm)      # CM & CH are too large over snow (Fortran comment)
            ch = W(snowh > 0., jnp.fmin(0.01, ch), ch)
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
    adj = (snowh > 0.05) & (tgb > TFRZ) & (o["stc"] == 1)
    tg_a = f32(TFRZ)
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
    bdsnow = sneqv / snowh
    temp1 = sneqv
    sneqv_n = jnp.fmax(0., temp1 - xm[3])
    propor = (sneqv_n - temp1) / bdsnow
    snowh_n = jnp.fmax(0., snowh + propor)
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
    C2, C3, C4, C5, DM, ETA0 = 21.e-3, 2.5e-6, 0.04, 2.0, 100.0, 0.8e+6
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
        dz[p] = W(go, dz[p] * (1. + pdzdtc), dz[p])
        burden = burden + W(a, wx, 0.)
    return dz


# ----------------------------------------------------------------------------------------------
def canwater(P, o, dt, sfctmp, uu, vv, fcev, fctr, qprecc, qprecl, elai, esai, fveg, frozen_canopy, canliq, canice, tv):
    """Noah-MP 3.6 CANWATER: rain/snow partition, canopy interception, evaporation/dew, melt and freeze of canopy water."""
    prc = qprecc + qprecl
    if o["snf"] == 1:
        fpice = W(sfctmp > TFRZ + 2.5, 0., W(sfctmp <= TFRZ + 0.5, 1.0, W(sfctmp <= TFRZ + 2., 1. - (-54.632 + 0.2 * sfctmp), 0.6)))
    elif o["snf"] == 2:
        fpice = W(sfctmp >= TFRZ + 2.2, 0., 1.0)
    else:
        fpice = W(sfctmp >= TFRZ, 0., 1.0)
    bdfall = jnp.fmax(120., 67.92 + 51.25 * jnp.exp(jnp.fmin(2.5, sfctmp - TFRZ) / 2.59))
    rain = prc * (1. - fpice)
    snow = prc * fpice
    fp = W(prc > 0., prc / (10. * qprecc + qprecl), 0.)
    vai = elai + esai
    has = vai > 0.
    maxliq = P["CH2OP"] * vai
    qintr = fveg * rain * fp
    qintr = jnp.fmin(qintr, (maxliq - canliq) / dt * (1. - jnp.exp(-rain * dt / jnp.fmax(maxliq, 1e-30))))
    qintr = jnp.fmax(qintr, 0.)
    qdripr = fveg * rain - qintr
    qthror = (1. - fveg) * rain
    qintr = W(has, qintr, 0.)
    qdripr = W(has, qdripr, 0.)
    qthror = W(has, qthror, rain)
    fr = frozen_canopy
    etran = W(fr, jnp.fmax(fctr / HSUB, 0.), jnp.fmax(fctr / HVAP, 0.))
    qevac = W(fr, 0., jnp.fmax(fcev / HVAP, 0.))
    qdewc = W(fr, 0., jnp.abs(jnp.fmin(fcev / HVAP, 0.)))
    qsubc = W(fr, jnp.fmax(fcev / HSUB, 0.), 0.)
    qfroc = W(fr, jnp.abs(jnp.fmin(fcev / HSUB, 0.)), 0.)
    qevac = jnp.fmin(canliq / dt, qevac)
    canliq = jnp.fmax(0., canliq + (qintr + qdewc - qevac) * dt)
    canliq = W(canliq <= 1.E-06, 0.0, canliq)
    maxsno = 6.6 * (0.27 + 46. / bdfall) * vai
    qints = fveg * snow * fp
    qints = jnp.fmin(qints, (maxsno - canice) / dt * (1. - jnp.exp(-snow * dt / jnp.fmax(maxsno, 1e-30))))
    qints = jnp.fmax(qints, 0.)
    ft = jnp.fmax(0.0, (tv - 270.15) / 1.87E5)
    fv = jnp.sqrt(uu * uu + vv * vv) / 1.56E5
    qdrips = jnp.fmax(0., canice) * (fv + ft)
    qthros = (1.0 - fveg) * snow + (fveg * snow - qints)
    qints = W(has, qints, 0.)
    qdrips = W(has, qdrips, 0.)
    qthros = W(has, qthros, snow)
    qsubc = jnp.fmin(canice / dt, qsubc)
    canice = jnp.fmax(0., canice + (qints - qdrips) * dt + (qfroc - qsubc) * dt)
    canice = W(canice <= 1.E-6, 0., canice)
    fwet = W(canice > 0., jnp.fmax(0., canice) / jnp.fmax(maxsno, 1.E-06),
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
    qrain = qdripr + qthror
    qsnow = qdrips + qthros
    snowhin = qsnow / bdfall
    return dict(canliq=canliq, canice=canice, tv=tv, fwet=fwet, cmc=cmc, ecan=ecan, etran=etran, qrain=qrain, qsnow=qsnow,
                snowhin=snowhin, fpice=fpice)


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
    top = (isnow == 0) & (sneqv > 0.) & (snowh > 0.)
    bdsnow = sneqv / snowh
    temp = sneqv
    sneqv_t = jnp.fmax(0., sneqv - qsnsub * dt + qsnfro * dt)
    propor = (sneqv_t - temp) / bdsnow
    snowh_t = jnp.fmax(0., snowh + propor)
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
    dz = sn["dz"]
    qin = f32(0.)
    qout = f32(0.)
    for p in range(3):
        a = isnow + 3 <= p
        dzs = W(a, dz[p], 1.0)
        vol_ice = jnp.fmin(1., ice[p] / (dzs * DENICE))
        epore = 1. - vol_ice
        liq_n = liq[p] + qin
        vol_liq = liq_n / (dzs * DENH2O)
        qo = jnp.fmax(0., (vol_liq - P["SSI"] * epore) * dzs) * DENH2O
        liq[p] = W(a, liq_n - qo, liq[p])
        qin = W(a, qo, qin)
        qout = W(a, qo, qout)
    sn.update(ice=ice, liq=liq)
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
    snowh_l = f32(0.)
    for p in range(3):
        a = isnow + 3 <= p
        snowh_l = snowh_l + W(a, dz3[p], 0.)
    snowh_f = W(lay, snowh_l, sn["snowh"])
    bdsnow = sneqv / jnp.where(snowh_f > 0., snowh_f, 1.0)
    dense = (isnow == 0) & (sneqv > 0.) & (bdsnow >= DENH2O)
    snowh_f = W(dense, snowh_f * (bdsnow / 1000.), snowh_f)
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
    return dict(isnow=isnow, snowh=snowh_f, sneqv=sneqv, snice=ice, snliq=liq, stc=list(stc3) + stc[3:], zsnso=zs,
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
    """Noah-MP 3.6 WDFCND2: ice weighting toward (0.2/SMCMAX), no FACTR1 clamp (4.0.1 uses MIN(0.05/SMCMAX, FACTR2))."""
    factr = jnp.fmax(0.01, smc / P["SMCMAX"][k])
    expon = P["BEXP"][k] + 2.0
    wdf = P["DWSAT"][k] * jnp.power(factr, expon)
    vkwgt = 1.0 / (1.0 + jnp.power(500.0 * sicemax, 3.0))
    wdf2 = vkwgt * wdf + (1.0 - vkwgt) * P["DWSAT"][k] * jnp.power(0.2 / P["SMCMAX"][k], expon)
    wdf = W(sicemax > 0.0, wdf2, wdf)
    expon = 2.0 * P["BEXP"][k] + 3.0
    wcnd = P["DKSAT"][k] * jnp.power(factr, expon)
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


def soilwater(P, o, dt, zsoil, dz, qinsur, qseva, etrani, sice, sh2o, smc, zwt):
    """OPT_RUN 1-4, OPT_INF=1. Returns sh2o, smc, runsrf, qdrain, wcnd, fcrmax, runsub, zwt"""
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
        fsat = P["FSATMX"] * jnp.exp(-0.5 * 6.0 * (zwt - 2.0))
    elif opt_run == 2:
        fsat = P["FSATMX"] * jnp.exp(-0.5 * 2.0 * zwt)
    elif opt_run == 4:
        smctot = f32(0.)
        dztot = f32(0.)
        run = jnp.bool_(True)
        for k in range(NSOIL):
            dztot = W(run, dztot + dz[k], dztot)
            smctot = W(run, smctot + smc[k] * dz[k], smctot)
            run = run & (dztot < 2.0)
        smctot = smctot / dztot
        fsat = jnp.power(jnp.fmax(0.01, smctot / P["SMCMAX"][0]), 4.)
    if opt_run == 3:
        pddum, runsrf0 = infil(P, dt, zsoil, sh2o, sice, sicemax, qinsur)
    else:
        runsrf0 = W(pos, qinsur * ((1.0 - fcr[0]) * fsat + fcr[0]), 0.)
        pddum = W(pos, qinsur - runsrf0, 0.)
    if o["inf"] == 2:
        niter = jnp.int32(1)                       # 3.6: sub-stepping only for OPT_INF = 1
    else:
        niter = W(pddum * dt > dz[0] * P["SMCMAX"][0], 6, 3)
    dtfine = dt / niter.astype(f32)
    smc0 = list(smc)

    def one(i, c):
        sh2o_, smc_, qd_s, rs_s, rsat_ = c
        act = i < niter
        pdd = pddum
        rs_i = runsrf0
        wdf = []; wcnd = []
        for k in range(NSOIL):
            if o["inf"] == 2:
                a, b = _wdfcnd2(P, sh2o_[k], sicemax, k)
            else:
                a, b = _wdfcnd1(P, smc0[k], fcr[k], k)      # Noah-MP 3.6: SMC is not updated inside the fine sub-steps
            wdf.append(a); wcnd.append(b)
        smx = list(sh2o_) if o["inf"] == 2 else list(smc0)
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
                if opt_run == 3:
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
        smc_new = [new[k] + sice[k] for k in range(NSOIL)]
        out_sh = [W(act, new[k], sh2o_[k]) for k in range(NSOIL)]
        out_smc = [W(act, smc_new[k], smc_[k]) for k in range(NSOIL)]
        return (out_sh, out_smc, W(act, qd_s + qdrain, qd_s), W(act, rs_s + rs_i, rs_s), W(act, rsat_ + wplus, rsat_)), \
            [W(act, wcnd[k], 0.) for k in range(NSOIL)]

    c = (sh2o, smc0, f32(0.), f32(0.), rsat)
    wcnd_last = [f32(0.)] * NSOIL
    for i in range(6):
        c, wc = one(i, c)
        wcnd_last = [W(i < niter, wc[k], wcnd_last[k]) for k in range(NSOIL)]
    sh2o, smc, qd_s, rs_s, rsat = c
    niterf = niter.astype(f32)
    runsrf = runsrf0                  # Noah-MP 3.6 computes the surface runoff once, before the sub-steps
    qdrain = qd_s / niterf
    runsrf = runsrf * 1000. + rsat * 1000. / dt
    qdrain = qdrain * 1000.
    if opt_run == 2:
        wtsub = f32(0.)
        for k in range(NSOIL):
            wtsub = wtsub + wcnd_last[k] * dz[k]
        sh2o = list(sh2o)
        for k in range(NSOIL):
            mh2o = W(wtsub == 0.0, 0., runsub * dt * (wcnd_last[k] * dz[k]) / jnp.where(wtsub == 0.0, 1.0, wtsub))
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
    ROUS = 0.2
    CMIC = 0.20
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
    fff = 6.0
    rsbmx = 5.0
    qdis = (1.0 - fcrmax) * rsbmx * jnp.exp(-P["TIMEAN"]) * jnp.exp(-fff * (zwt - 2.0))
    smc_i = at(smc, iwt - 1)
    smcmax_i = at(list(P["SMCMAX"]), iwt - 1)
    s_node = jnp.fmin(1.0, smc_i / smcmax_i)
    s_node = jnp.fmax(s_node, 0.01)
    smpfz = -at(list(P["PSISAT"]), iwt - 1) * 1000. * jnp.power(s_node, -at(list(P["BEXP"]), iwt - 1))
    smpfz = jnp.fmax(-120000.0, CMIC * smpfz)
    ka = at(hk, iwt - 1)
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
def water(P, o, dt, zsoil, qvap, qdew, fcev, fctr, qprecc, qprecl, uu, vv, frozen_canopy, frozen_ground,
          ponding, elai, esai, fveg, btrani, ficeold, imelt, sfctmp, isnow, canliq, canice, tv, snowh, sneqv, snice, snliq,
          stc, zsnso, sh2o, smc, sice, zwt, wa, wt, dzsnso, wslake, rivsto, fldsto, fldfrc):
    """returns dict of updated water state and fluxes (Noah-MP 3.6 WATER)"""
    CW = canwater(P, o, dt, sfctmp, uu, vv, fcev, fctr, qprecc, qprecl, elai, esai, fveg, frozen_canopy, canliq, canice, tv)
    canliq, canice, tv, fwet, cmc, ecan, etran = CW["canliq"], CW["canice"], CW["tv"], CW["fwet"], CW["cmc"], CW["ecan"], CW["etran"]
    qrain, qsnow, snowhin, fpice = CW["qrain"], CW["qsnow"], CW["snowhin"], CW["fpice"]
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
    sh2o, smc, runsrf, qdrain, wcnd, fcrmax, runsub, zwt = soilwater(P, o, dt, zsoil, dz4, qinsur, qseva, etrani, sice, sh2o, smc, zwt)
    if o["run"] == 1:
        sh2o, zwt, wa, wt, qin, qdis = groundwater(P, dt, sice, zsoil, wcnd, fcrmax, sh2o, zwt, wa, wt)
        runsub = qdis
    elif o["run"] in (3, 4):
        runsub = runsub + qdrain
    smc = [sh2o[k] + sice[k] for k in range(NSOIL)]
    runsub = runsub + snoflow
    runsrf = W(fldfrc == 0, runsrf + (rivsto + fldsto) * 1000., runsrf)
    return dict(canliq=canliq, canice=canice, tv=tv, fwet=fwet, cmc=cmc, ecan=ecan, etran=etran, subsnow=subsnow,
                isnow=isnow, snowh=snowh, sneqv=sneqv, snice=snice, snliq=snliq, stc=stc, zsnso=zsnso, dzsnso=dzsnso,
                sh2o=sh2o, smc=smc, sice=sice, zwt=zwt, wa=wa, wt=wt, runsrf=runsrf, runsub=runsub, qsnbot=qsnbot,
                ponding1=ponding1, ponding2=ponding2, qsnow=qsnow, fpice=fpice)


def sflx_mp(P, x, o):
    """One Noah-MP 3.6 time step for one tile.
    P: parameter dict (scalars; BEXP... arrays of 4; LAIM/SAIM 12; RHOL.. 2). x: inputs/states dict. o: static options.
    Returns dict with all post-call quantities (names as in the Fortran dump)."""
    dt = x["DT"]
    sfctmp, sfcprs, psfc, uu, vv, q2 = x["T_ML"], x["P_ML"], x["PSFC"], x["U_ML"], x["V_ML"], x["Q_ML"]
    lwdn, cosz, zref = x["LWDN"], x["COSZ"], x["Z_ML"]
    zsoil = x["ZSOIL"]
    isnow = x["ISNOW"]
    A = atm(P, o, sfcprs, sfctmp, q2, x["PRCP"], x["SWDN"], cosz)
    swdown = A["swdown"]
    zsnso = list(x["ZSNSO"])
    dzsnso = []
    for p in range(NL):
        a = isnow + 3 <= p
        first = p == isnow + 3
        prev = zsnso[p - 1] if p > 0 else f32(0.)
        dzsnso.append(W(a, W(first, -zsnso[p], prev - zsnso[p]), 0.))
    lai, sai, elai, esai, igs = phenology(P, o, x["VEGTYP"], x["SNDPTH"], x["TV"], x["LAT"], x["YEARLEN"], x["JULIAN"],
                                          x["PLAI"], x["PSAI"])
    fveg = x["FVGMAX"] if o["dveg"] == 4 else (x["SHDFAC"] if o["dveg"] == 1 else 1. - jnp.exp(-0.52 * (lai + sai)))      # DVEG 3: FVEG from LAI + SAI
    fveg = W(fveg <= 0.01, 0.01, fveg)
    fveg = W((P["URBAN_FLAG"] > 0.5) | (x["VEGTYP"] == P["ISBARREN"]), 0.0, fveg)
    fveg = W(elai + esai == 0.0, 0.0, fveg)
    zero_ = f32(0.)
    PH = dict(pahv=zero_, pahg=zero_, pahb=zero_)                       # no precipitation heat in 3.6
    canliq, canice, fwet = x["CANLIQ"], x["CANICE"], x["FWET"]          # canopy water of the previous step
    qsnow = x["QSNOW"]                                                  # snowfall of the previous step
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
    fsno = W(snowh > 0., jnp.tanh(snowh / (2.5 * Z0_BARE * jnp.fmax(fmelt, 1e-30))), 0.)
    z0mg = Z0_BARE * (1.0 - fsno) + fsno * P["Z0SNO"]
    zpdg = snowh
    zpd_v = 0.65 * P["HVT"]
    zpd_v = W(snowh > zpd_v, snowh, zpd_v)
    z0m = W(veg, P["Z0MVT"], z0mg)
    zpd = W(veg, zpd_v, zpdg)
    zlvl = jnp.fmax(zpd, P["HVT"]) + zref
    zlvl = W(zpdg >= zlvl, zpdg + zref, zlvl)
    cwp = P["CWPVT"]
    df, hcpct, fact = thermoprop(P, isnow, dt, snowh, snice, snliq, dzsnso, smc, sh2o)
    R = radiation(P, o, dt, cosz, tg, tv, fsno, qsnow, fwet, elai, esai, smc[0], A["solad"], A["solai"], fveg,
                  x["ALBOLD"], x["TAUSS"], sneqvo, sneqv)
    emv = 1. - jnp.exp(-(elai + esai) / 1.0)
    emg = P["EG"][0] * (1. - fsno) + 1.0 * fsno
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
        l_rsurf = (-zsoil[0]) * (jnp.exp((1.0 - jnp.fmin(1.0, sh2o[0] / P["SMCMAX"][0])) ** 5) - 1.0) / (2.71828 - 1.0)
        d_rsurf = 2.2E-5 * P["SMCMAX"][0] * P["SMCMAX"][0] * jnp.power(1.0 - P["SMCWLT"][0] / P["SMCMAX"][0], 2.0 + 3.0 / P["BEXP"][0])
        rsurf = l_rsurf / d_rsurf
        l_pet = (-zsoil[0]) * (jnp.exp((1.0 - jnp.fmin(1.0, P["SMCMAX"][0] / P["SMCMAX"][0])) ** 5) - 1.0) / (2.71828 - 1.0)
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
                 sag=R["sag"], pahv=PH["pahv"], pahg=PH["pahg"], df1=df1, dz1=dz1, stc1=stc1, snowh=snowh, thair=A["thair"])
    VF = vege_flux(P, o, cdict, pet=False)
    bdict = dict(sfctmp=sfctmp, rhoair=A["rhoair"], ur=ur, snowh=snowh, zlvl=zlvl, zpdg=zpdg, z0mg=z0mg, emg=emg,
                 lwdn=lwdn, rsurf=rsurf, rhsur=rhsur, gammag=gammag, latheag=latheag, tg=tg, cm=cm0, ch=ch0,
                 qair=A["qair"], eair=A["eair"], psfc=psfc, sag=R["sag"], pahb=PH["pahb"], df1=df1, dz1=dz1, stc1=stc1, thair=A["thair"])
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
    tv_n = W(vv_ok, VF["tv"], tv)
    tah_n = W(vv_ok, VF["tah"], x["TAH"])
    eah_n = W(vv_ok, VF["eah"], x["EAH"])
    rssun = W(vv_ok, VF["rssun"], 0.)
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
    Wt = water(P, o, dt, zsoil, qvap, qdew, fcev, fctr, A["qprecc"], A["qprecl"], uu, vv, frozen_canopy, frozen_ground,
               ponding, elai, esai, fveg, btrani, x["FICEOLD"], imelt, sfctmp, isnow, canliq, canice, tv,
               snowh, sneqv, snice, snliq, stc, zsnso, sh2o, smc, sice, x["ZWT"], x["WA"], x["WT"], dzsnso, x["WSLAKE"],
               x["rivsto"], x["fldsto"], x["fldfrc"])
    snowh = Wt["snowh"]; sneqv = Wt["sneqv"]
    clean = (snowh <= 1.E-6) | (sneqv <= 1.E-3)
    snowh = W(clean, 0.0, snowh)
    sneqv = W(clean, 0.0, sneqv)
    albedo_ = W(swdown != 0., R["fsr"] / jnp.where(swdown != 0., swdown, 1.0), -999.9)
    relsmc = [(Wt["smc"][k] - P["SMCWLT"][k]) / (P["SMCMAX"][k] - P["SMCWLT"][k]) for k in range(NSOIL)]
    out = dict(ALBOLD=R["albold"], SNEQVO=sneqvo_n, TAH=tah_n, EAH=eah_n, FWET=Wt["fwet"], CANLIQ=Wt["canliq"],
               CANICE=Wt["canice"], TV=Wt["tv"], TG=tg, QSFC1D=BF["qsfc"], QSNOW=Wt["qsnow"], ISNOW=Wt["isnow"], SNDPTH=snowh,
               SWE=sneqv, ZWT=Wt["zwt"], WA=Wt["wa"], WT=Wt["wt"], PLAI=lai, PSAI=sai, CM=cm_n, CH=ch_n,
               TAUSS=R["tauss"], Z0WRF=z0wrf, FSA=R["fsa"], FSR=R["fsr"], FIRA=fira, FSH=fsh, SSOIL=ssoil,
               FCEV=fcev, FGEV=fgev, FCTR=fctr, ECAN=Wt["ecan"], ETRAN=Wt["etran"], ESOIL=edir, TRAD=trad,
               SUBSNOW=Wt["subsnow"], TGB=tgb, TGV=tgv, T2MV=W(vv_ok, VF["t2mv"], 0.), T2MB=BF["t2mb"],
               Q2MV=W(vv_ok, VF["q2v"], 0.), Q2MB=BF["q2b"], RUNSF=Wt["runsrf"], RUNSB=Wt["runsub"], APAR=apar, PSN=psn,
               SAV=R["sav"], SAG=R["sag"], FSNO=fsno, SALB=albedo_, QSNBOT=Wt["qsnbot"], PONDING=ponding,
               PONDING1=Wt["ponding1"], PONDING2=Wt["ponding2"], RSSUN=rssun, RSSHA=rssha, BGAP=R["bgap"],
               WGAP=R["wgap"], CHV=chv, CHB=BF["ch"], EMISSI=emissi, SHG=W(vv_ok, VF["shg"], 0.),
               SHC=W(vv_ok, VF["shc"], 0.), SHB=BF["shb"], EVG=W(vv_ok, VF["evg"], 0.), EVB=BF["evb"],
               GHV=W(vv_ok, VF["ghv"], 0.), GHB=BF["ghb"], IRG=W(vv_ok, VF["irg"], 0.), IRC=W(vv_ok, VF["irc"], 0.),
               IRB=BF["irb"], TR=W(vv_ok, VF["tr"], 0.), EVC=W(vv_ok, VF["evc"], 0.), 
               CHLEAF=W(vv_ok, VF["chleaf"], 0.), CHUC=W(vv_ok, VF["chuc"], 0.),
               CHV2=W(vv_ok, VF["chv2"], 0.), CHB2=BF["chb2"], FPICE=Wt["fpice"], PAHV=PH["pahv"], PAHG=PH["pahg"],
               PAHB=PH["pahb"], PAH=pah, LAISUN=R["laisun"], LAISHA=R["laisha"], RB=f32(0.),
               STC=Wt["stc"], SMH2O=Wt["sh2o"], SMC=Wt["smc"], ZSNSO=Wt["zsnso"], SNICE=Wt["snice"], SNLIQ=Wt["snliq"],
               RELSMC=relsmc, FVEGMP=fveg)
    return out

