"""JAX port of the LISF Noah 2.7.1 column physics (SFLX and the routines it calls).

Source: lis/surfacemodels/land/noah.2.7.1/module_sf_noah271lsm.F90 and noah271_main.F90 (unmodified, read-only).
Scope: ICE = 0 (water-class tiles belong to the open-water model in LIS), 4 soil layers, uncoupled mode, SFCDIF with ILECH = 0.
Style as noah39_jax.py: one function per Fortran routine on a single column; layers are Python lists of 4 scalars; IFs are jnp.where
selects; float32 throughout. Routines identical to the Noah 3.9 port (DEVAP/TRANSP, CANRES, SNKSRC/FRH2O, TBND, TMPAVG,
ROSR12/HSTEP, SSTEP, SNOWPACK, SNFRAC) are reused from noah39_jax (checked line by line against the 2.7.1 source).
"""
import jax.numpy as jnp
from . import noah39_jax as nj

W = jnp.where
f32 = jnp.float32
NS = 4
CP = 1004.5
RD = 287.04
SIGMA = 5.67e-8
LVH2O, LSUBS, LSUBC, LSUBF = 2.501e6, 2.83e6, 2.501e6, 3.335e5
CPH2O, CPICE = 4.218e3, 2.106e3
TFREEZ = 273.15
T0 = 273.15
SALP = 4.0
LIS_TKFRZ, LIS_LATVAP = 273.16, 2.501e6


def csnow(dsnow):
    c = 0.328 * jnp.power(10.0, 2.25 * dsnow)
    return 0.11631 * c


def snow_new(temp, newsn, snowh, sndens):
    snowhc = snowh * 100.0
    newsnc = newsn * 100.0
    tempc = temp - 273.15
    dsnew = W(tempc <= -15.0, 0.05, 0.05 + 0.0017 * jnp.power(tempc + 15.0, 1.5))
    hnewc = newsnc / dsnew
    sndens = (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc)
    snowhc = snowhc + hnewc
    return snowhc * 0.01, sndens


def tdfcnd(smc, qz, smcmax, sh2o):
    satratio = smc / smcmax
    thkice, thkw, thko, thkqtz = 2.2, 0.57, 2.0, 7.7
    thks = jnp.power(thkqtz, qz) * jnp.power(thko, 1.0 - qz)
    xunfroz = (sh2o + 1.0e-9) / (smc + 1.0e-9)
    xu = xunfroz * smcmax
    thksat = jnp.power(thks, 1.0 - smcmax) * jnp.power(thkice, smcmax - xu) * jnp.power(thkw, xu)
    gammd = (1.0 - smcmax) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    ake = W((sh2o + 0.0005) < smc, satratio, W(satratio > 0.1, jnp.log10(satratio) + 1.0, 0.0))
    return ake * (thksat - thkdry) + thkdry


def wdfcnd(smc, smcmax, bexp, dksat, dwsat, sicemax):
    factr1 = 0.2 / smcmax
    factr2 = smc / smcmax
    expon = bexp + 2.0
    wdf = dwsat * jnp.power(factr2, expon)
    vkwgt = 1.0 / (1.0 + jnp.power(500.0 * sicemax, 3.0))
    wdf2 = vkwgt * wdf + (1.0 - vkwgt) * dwsat * jnp.power(factr1, expon)
    wdf = W(sicemax > 0.0, wdf2, wdf)
    expon = (2.0 * bexp) + 3.0
    wcnd = dksat * jnp.power(factr2, expon)
    return wdf, wcnd


def penman(sfctmp, sfcprs, ch, t2v, th2, prcp, fdown, ssoil, q2, q2sat, snowng, frzgra, dqsdt2, emissi):
    cp, elcp = 1004.6, 2.4888e3
    delta = elcp * dqsdt2
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    rr = emissi * t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    rho = sfcprs / (RD * t2v)
    rch = rho * cp * ch
    rr = W(snowng, rr + CPICE * prcp / rch, W(prcp > 0.0, rr + CPH2O * prcp / rch, rr))
    fnet = fdown - emissi * SIGMA * t24 - ssoil
    flx2 = W(frzgra, -LSUBF * prcp, 0.0)
    fnet = W(frzgra, fnet - flx2, fnet)
    rad = fnet / rch + th2 - sfctmp
    a = elcp * (q2sat - q2)
    epsca = (a * rr + rad * delta) / (delta + rr)
    etp = epsca * rch / LSUBC
    return etp, rch, epsca, rr, t24, flx2


def evapo(smc, cmc, etp1, dt, zsoil, sh2o, smcmax, pc, smcwlt, smcref, shdfac, cmcmax, smcdry, cfactr, nroot, rtdis, fxexp):
    """EVAPO of Noah 2.7.1: direct evaporation uses the LIQUID water of layer 1 (SH2O(1)); otherwise as in noah39_jax.evapo."""
    pos = etp1 > 0.0
    edir = W(pos & (shdfac < 1.0), nj.devap(etp1, sh2o[0], shdfac, smcmax, smcdry, fxexp), 0.0)
    et = nj.transp(etp1, sh2o, cmc, zsoil, shdfac, smcwlt, cmcmax, pc, cfactr, smcref, nroot, rtdis)
    tr_on = pos & (shdfac > 0.0)
    et = [W(tr_on, e, 0.0) for e in et]
    ett = nj._lsum(et)
    ec = W(cmc > 0.0, shdfac * jnp.power(jnp.fmax(cmc, 1.0e-30) / cmcmax, cfactr) * etp1, 0.0)
    ec = jnp.fmin(cmc / dt, ec)
    ec = W(tr_on, ec, 0.0)
    return edir + ett + ec, edir, ec, et, ett


def hrt(stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot, psisat, sh2o, dt, bexp, quartz, csoil, df1):
    cair, ch2o, cice = 1004.0, 4.2e6, 2.106e6
    sh2o = list(sh2o)
    rhsts, ai, bi, ci = [None] * NS, [None] * NS, [None] * NS, [None] * NS
    hcpct = sh2o[0] * ch2o + (1.0 - smcmax) * csoil + (smcmax - smc[0]) * cair + (smc[0] - sh2o[0]) * cice
    ddz = 1.0 / (-0.5 * zsoil[1])
    ai[0] = f32(0.0)
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct)
    bi[0] = -ci[0] + df1 / (0.5 * zsoil[0] * zsoil[0] * hcpct * zz1)
    dtsdz = (stc[0] - stc[1]) / (-0.5 * zsoil[1])
    ssoil = df1 * (stc[0] - yy) / (0.5 * zsoil[0] * zz1)
    rhsts[0] = (df1 * dtsdz - ssoil) / (zsoil[0] * hcpct)
    qtot = ssoil - df1 * dtsdz
    tsurf = (yy + (zz1 - 1) * stc[0]) / zz1
    tbk = nj.tbnd(stc[0], stc[1], zsoil, zbot, 0)
    sice = smc[0] - sh2o[0]
    cond = (sice > 0.0) | (tsurf < T0) | (stc[0] < T0) | (tbk < T0)
    tavg = nj.tmpavg(tsurf, stc[0], tbk, zsoil, 0)
    tsnsr, xh = nj.snksrc(tavg, smc[0], sh2o[0], zsoil, smcmax, psisat, bexp, dt, 0, qtot)
    rhsts[0] = W(cond, rhsts[0] - tsnsr / (zsoil[0] * hcpct), rhsts[0])
    sh2o[0] = W(cond, xh, sh2o[0])
    ddz2 = f32(0.0)
    df1k = df1
    for k in range(1, NS):
        hcpct = sh2o[k] * ch2o + (1.0 - smcmax) * csoil + (smcmax - smc[k]) * cair + (smc[k] - sh2o[k]) * cice
        df1n = tdfcnd(smc[k], quartz, smcmax, sh2o[k])
        if k != NS - 1:
            denom = 0.5 * (zsoil[k - 1] - zsoil[k + 1])
            dtsdz2 = (stc[k] - stc[k + 1]) / denom
            ddz2 = 2.0 / (zsoil[k - 1] - zsoil[k + 1])
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct)
            tbk1 = nj.tbnd(stc[k], stc[k + 1], zsoil, zbot, k)
        else:
            denom = 0.5 * (zsoil[k - 1] + zsoil[k]) - zbot
            dtsdz2 = (stc[k] - tbot) / denom
            ci[k] = f32(0.0)
            tbk1 = nj.tbnd(stc[k], tbot, zsoil, zbot, k)
        denom = (zsoil[k] - zsoil[k - 1]) * hcpct
        rhsts[k] = (df1n * dtsdz2 - df1k * dtsdz) / denom
        qtot = -1.0 * denom * rhsts[k]
        sice = smc[k] - sh2o[k]
        cond = (sice > 0.0) | (tbk < T0) | (stc[k] < T0) | (tbk1 < T0)
        tavg = nj.tmpavg(tbk, stc[k], tbk1, zsoil, k)
        tsnsr, xh = nj.snksrc(tavg, smc[k], sh2o[k], zsoil, smcmax, psisat, bexp, dt, k, qtot)
        rhsts[k] = W(cond, rhsts[k] - tsnsr / denom, rhsts[k])
        sh2o[k] = W(cond, xh, sh2o[k])
        ai[k] = -df1 * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct)
        bi[k] = -(ai[k] + ci[k])
        tbk = tbk1
        df1k = df1n
        dtsdz = dtsdz2
        ddz = ddz2
    return rhsts, ai, bi, ci, sh2o


def shflx(stc, smc, smcmax, dt, yy, zz1, zsoil, tbot, zbot, psisat, sh2o, bexp, df1, quartz, csoil):
    rhsts, ai, bi, ci, sh2o = hrt(stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot, psisat, sh2o, dt, bexp, quartz, csoil, df1)
    stc = nj.hstep(stc, rhsts, dt, ai, bi, ci)
    t1 = (yy + (zz1 - 1.0) * stc[0]) / zz1
    ssoil = df1 * (stc[0] - t1) / (0.5 * zsoil[0])
    return ssoil, stc, t1, sh2o


def srt(edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice):
    cvfrz = 3
    sicemax = f32(0.0)
    for ks in range(NS):
        sicemax = W(sice[ks] > sicemax, sice[ks], sicemax)
    dt1 = dt / 86400.0
    smcav = smcmax - smcwlt
    dmax0 = -zsoil[0] * smcav
    dice = -zsoil[0] * sice[0]
    dmax0 = dmax0 * (1.0 - (sh2oa[0] + sice[0] - smcwlt) / smcav)
    dd = dmax0
    for ks in range(1, NS):
        dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks]
        dm = (zsoil[ks - 1] - zsoil[ks]) * smcav
        dm = dm * (1.0 - (sh2oa[ks] + sice[ks] - smcwlt) / smcav)
        dd = dd + dm
    val = 1.0 - jnp.exp(-kdt * dt1)
    ddt = dd * val
    px = pcpdrp * dt
    px = W(px < 0.0, 0.0, px)
    infmax = W((px + ddt) == 0.0, 0.0, (px * (ddt / (px + ddt))) / dt)
    acrt = cvfrz * frzx / dice
    sm = 1.0 + (acrt * acrt) / 2.0 + acrt / 1.0
    fcr = W(dice > 1.0e-2, 1.0 - jnp.exp(-acrt) * sm, 1.0)
    infmax = infmax * fcr
    wdf, wcnd = wdfcnd(sh2oa[0], smcmax, bexp, dksat, dwsat, sicemax)
    infmax = jnp.fmax(infmax, wcnd)
    infmax = jnp.fmin(infmax, px)
    over = (pcpdrp != 0.0) & (pcpdrp > infmax)
    runoff1 = W(over, pcpdrp - infmax, 0.0)
    pddum = W(over, infmax, pcpdrp)
    ddz = 1.0 / (-0.5 * zsoil[1])
    ai, bi, ci, rhstt = [None] * NS, [None] * NS, [None] * NS, [None] * NS
    ai[0] = f32(0.0)
    bi[0] = wdf * ddz / (-zsoil[0])
    ci[0] = -bi[0]
    dsmdz = (sh2o[0] - sh2o[1]) / (-0.5 * zsoil[1])
    rhstt[0] = (wdf * dsmdz + wcnd - pddum + edir + et[0]) / zsoil[0]
    runoff2 = f32(0.0)
    for k in range(1, NS):
        denom2 = zsoil[k - 1] - zsoil[k]
        if k != NS - 1:
            slopx = 1.0
            wdf2, wcnd2 = wdfcnd(sh2oa[k], smcmax, bexp, dksat, dwsat, sicemax)
            denom = zsoil[k - 1] - zsoil[k + 1]
            dsmdz2 = (sh2o[k] - sh2o[k + 1]) / (denom * 0.5)
            ddz2 = 2.0 / denom
            ci[k] = -wdf2 * ddz2 / denom2
        else:
            slopx = slope
            wdf2, wcnd2 = wdfcnd(sh2oa[NS - 1], smcmax, bexp, dksat, dwsat, sicemax)
            dsmdz2 = f32(0.0)
            ci[k] = f32(0.0)
            ddz2 = ddz
        numer = (wdf2 * dsmdz2) + slopx * wcnd2 - (wdf * dsmdz) - wcnd + et[k]
        rhstt[k] = numer / (-denom2)
        ai[k] = -wdf * ddz / denom2
        bi[k] = -(ai[k] + ci[k])
        if k == NS - 1:
            runoff2 = slopx * wcnd2
        else:
            wdf, wcnd, dsmdz, ddz = wdf2, wcnd2, dsmdz2, ddz2
    return rhstt, ai, bi, ci, runoff1, runoff2


def smflx(smc, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzx, smcmax, bexp, smcwlt, dksat, dwsat, shdfac, cmcmax, edir, ec, et):
    rhsct = shdfac * prcp1 - ec
    trhsct = dt * rhsct
    excess = cmc + trhsct
    drip = W(excess > cmcmax, excess - cmcmax, 0.0)
    pcpdrp = (1.0 - shdfac) * prcp1 + drip / dt
    sice = [smc[i] - sh2o[i] for i in range(NS)]
    cond = (pcpdrp * dt) > (1.0 * (-zsoil[0]) * smcmax)
    rhstt, ai, bi, ci, ro1, ro2 = srt(edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice)
    sh2o_b, smc_b, cmc_b, ro3_b = nj.sstep(sh2o, cmc, rhstt, rhsct, dt, smcmax, cmcmax, zsoil, sice, ai, bi, ci)
    sh2ofg, _, _, _ = nj.sstep(sh2o, f32(0.0), rhstt, rhsct, dt, smcmax, cmcmax, zsoil, sice, ai, bi, ci)
    sh2oa = [(sh2o[k] + sh2ofg[k]) * 0.5 for k in range(NS)]
    rhstt2, ai2, bi2, ci2, ro1_a, ro2_a = srt(edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice)
    sh2o_a, smc_a, cmc_a, ro3_a = nj.sstep(sh2o, cmc, rhstt2, rhsct, dt, smcmax, cmcmax, zsoil, sice, ai2, bi2, ci2)
    pick = lambda a, b: W(cond, a, b)
    smc_o = [pick(smc_a[k], smc_b[k]) for k in range(NS)]
    sh2o_o = [pick(sh2o_a[k], sh2o_b[k]) for k in range(NS)]
    return smc_o, sh2o_o, pick(cmc_a, cmc_b), drip, pick(ro1_a, ro1), pick(ro2_a, ro2), pick(ro3_a, ro3_b)


def nopac(etp, prcp, smc, smcmax, smcwlt, smcref, smcdry, cmc, cmcmax, dt, shdfac, sbeta, sfctmp, t24, th2, fdown, stc, epsca,
          bexp, pc, rch, rr, cfactr, sh2o, slope, kdt, frzx, psisat, zsoil, dksat, dwsat, tbot, zbot, nroot, rtdis, quartz,
          fxexp, csoil, emissi, q2):
    prcp1 = prcp * 0.001
    etp1 = etp * 0.001
    pos = etp > 0.0
    eta1, edir1, ec1, et1, ett1 = evapo(smc, cmc, etp1, dt, zsoil, sh2o, smcmax, pc, smcwlt, smcref, shdfac, cmcmax, smcdry,
                                          cfactr, nroot, rtdis, fxexp)
    z = f32(0.0)
    edir1 = W(pos, edir1, z)
    ec1 = W(pos, ec1, z)
    et1 = [W(pos, e, z) for e in et1]
    ett1 = W(pos, ett1, z)
    eta1 = W(pos, eta1, z)
    dew = W(pos, z, -etp1)
    prcp1 = W(pos, prcp1, prcp1 + dew)
    smc, sh2o, cmc, drip, ro1, ro2, ro3 = smflx(smc, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzx, smcmax, bexp, smcwlt, dksat,
                                               dwsat, shdfac, cmcmax, edir1, ec1, et1)
    eta = eta1 * 1000.0
    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    et = [e * 1000.0 for e in et1]
    ett = ett1 * 1000.0
    beta = W(etp <= 0.0, W(etp < 0.0, 1.0, 0.0), eta / etp)
    df1 = tdfcnd(smc[0], quartz, smcmax, sh2o[0])
    df1 = df1 * jnp.exp(sbeta * shdfac)
    yynum = fdown - emissi * SIGMA * t24
    yy = sfctmp + (yynum / rch + th2 - sfctmp - beta * epsca) / rr
    zz1 = df1 / (-0.5 * zsoil[0] * rch * rr) + 1.0
    ssoil, stc, t1, sh2o = shflx(stc, smc, smcmax, dt, yy, zz1, zsoil, tbot, zbot, psisat, sh2o, bexp, df1, quartz, csoil)
    return dict(eta=eta, beta=beta, edir=edir, ec=ec, et=et, ett=ett, drip=drip, ro1=ro1, ro2=ro2, ro3=ro3, ssoil=ssoil,
                tsoil=t1, t1=t1, stc=stc, smc=smc, sh2o=sh2o, cmc=cmc, dew=dew, flx1=z, flx3=z, esnow=z, snomlt=z, etanrg=z)


def snopac(etp, prcp, prcp1, snowng, smc, smcmax, smcwlt, smcref, smcdry, cmc, cmcmax, dt, df1, q2, t1, sfctmp, t24, th2, fdown,
           stc, rch, rr, cfactr, sncovr, esd, sndens, snowh, sh2o, slope, kdt, frzx, psisat, zsoil, dwsat, dksat, tbot, zbot,
           shdfac, nroot, rtdis, quartz, fxexp, csoil, bexp, pc, flx2):
    esdmin, snoexp, emissi_s = 1.0e-6, 2.0, 0.95
    z = f32(0.0)
    prcp1 = prcp1 * 0.001
    etp1 = etp * 0.001
    neg = etp < 0.0
    dew = W(neg, -etp1, z)
    esnow2_neg = etp1 * dt
    etanrg_neg = etp * ((1.0 - sncovr) * LSUBC + sncovr * LSUBS)
    run_ev = (~neg) & (sncovr < 1.0)
    eta1, edir1, ec1, et1, ett1 = evapo(smc, cmc, etp1, dt, zsoil, sh2o, smcmax, pc, smcwlt, smcref, shdfac, cmcmax, smcdry,
                                          cfactr, nroot, rtdis, fxexp)
    etns1 = eta1
    f = 1.0 - sncovr
    edir1 = W(run_ev, edir1 * f, z)
    ec1 = W(run_ev, ec1 * f, z)
    et1 = [W(run_ev, e * f, z) for e in et1]
    ett1 = W(run_ev, ett1 * f, z)
    etns1 = W(run_ev, etns1 * f, z)
    edir, ec, ett, etns = edir1 * 1000.0, ec1 * 1000.0, ett1 * 1000.0, etns1 * 1000.0
    et = [e * 1000.0 for e in et1]
    esnow = W(neg, z, etp * sncovr)
    esnow1 = esnow * 0.001
    esnow2 = W(neg, esnow2_neg, esnow1 * dt)
    etanrg = W(neg, etanrg_neg, esnow * LSUBS + etns * LSUBC)
    flx1 = W(snowng, CPICE * prcp * (t1 - sfctmp), W(prcp > 0.0, CPH2O * prcp * (t1 - sfctmp), z))
    dsoil = -(0.5 * zsoil[0])
    dtot = snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    emf = emissi_s * sncovr + (1.0 - sncovr)
    t12a = ((fdown - flx1 - flx2 - emf * SIGMA * t24) / rch + th2 - sfctmp - etanrg / rch) / rr
    t12b = df1 * stc[0] / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom
    cold = t12 <= TFREEZ
    t1_c = t12
    ssoil_c = df1 * (t1_c - stc[0]) / dtot
    esd_c = jnp.fmax(0.0, esd - esnow2)
    sn2 = jnp.power(sncovr, snoexp)
    t1_w = TFREEZ * sn2 + t12 * (1.0 - sn2)
    ssoil_w = df1 * (t1_w - stc[0]) / dtot
    gone = (esd - esnow2) <= esdmin
    esd_w1 = esd - esnow2
    seh = rch * (t1_w - th2)
    t14 = t1_w * t1_w
    t14 = t14 * t14
    flx3 = fdown - flx1 - flx2 - emf * SIGMA * t14 - ssoil_w - seh - etanrg
    flx3 = W(flx3 <= 0.0, 0.0, flx3)
    ex = flx3 * 0.001 / LSUBF
    snomlt = ex * dt
    enough = (esd_w1 - snomlt) >= esdmin
    esd_w2 = W(enough, esd_w1 - snomlt, 0.0)
    ex = W(enough, ex, esd_w1 / dt)
    flx3 = W(enough, flx3, ex * 1000.0 * LSUBF)
    snomlt = W(enough, snomlt, esd_w1)
    esd_w = W(gone, 0.0, esd_w2)
    ex_w = W(gone, 0.0, ex)
    snomlt_w = W(gone, 0.0, snomlt)
    flx3_w = W(gone, 0.0, flx3)            # not assigned by the Fortran when the snow is gone: the caller's initial 0
    t1 = W(cold, t1_c, t1_w)
    ssoil = W(cold, ssoil_c, ssoil_w)
    esd = W(cold, esd_c, esd_w)
    flx3 = W(cold, 0.0, flx3_w)
    ex = W(cold, 0.0, ex_w)
    snomlt = W(cold, 0.0, snomlt_w)
    prcp1 = W(cold, prcp1, prcp1 + ex)
    smc, sh2o, cmc, drip, ro1, ro2, ro3 = smflx(smc, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzx, smcmax, bexp, smcwlt, dksat,
                                               dwsat, shdfac, cmcmax, edir1, ec1, et1)
    zz1 = f32(1.0)
    yy = stc[0] - 0.5 * ssoil * zsoil[0] * zz1 / df1
    _, stc, _, sh2o = shflx(stc, smc, smcmax, dt, yy, zz1, zsoil, tbot, zbot, psisat, sh2o, bexp, df1, quartz, csoil)
    has = esd > 0.0
    snowh_n, sndens_n = nj.snowpack(esd, dt, snowh, sndens, t1, yy)
    snowh = W(has, snowh_n, 0.0)
    sndens = W(has, sndens_n, 0.0)
    sncovr = W(has, sncovr, 0.0)
    esd = W(has, esd, 0.0)
    return dict(edir=edir, ec=ec, et=et, ett=ett, drip=drip, ro1=ro1, ro2=ro2, ro3=ro3, ssoil=ssoil, tsoil=yy, t1=t1, stc=stc,
                smc=smc, sh2o=sh2o, cmc=cmc, dew=dew, flx1=flx1, flx3=flx3, esnow=esnow, snomlt=snomlt, esd=esd, snowh=snowh,
                sndens=sndens, sncovr=sncovr, etanrg=etanrg)


def sfcdif(zlm, z0, thz0, thlm, sfcspd, czil, akms, akhs):
    """Noah 2.7.1 SFCDIF (ILECH = 0). Returns updated (akms, akhs) and ustar."""
    wwst, vkrm, excm = 1.2, 0.40, 0.001
    wwst2 = wwst * wwst
    g = 9.8
    beta = 1.0 / 270.0
    btg = beta * g
    elfc = vkrm * btg
    wold = 0.15
    wnew = 1.0 - wold
    pihf = 3.14159265 / 2.0
    epsu2, epsust, hpbl, sqvisc = 1.0e-4, 0.07, 1000.0, 258.2
    ztmin, ztmax = -5.0, 1.0
    zilfc = -czil * vkrm * sqvisc
    zu = z0
    rdz = 1.0 / zlm
    cxch = excm * rdz
    dthv = thlm - thz0
    du2 = jnp.fmax(sfcspd * sfcspd, epsu2)
    btgh = btg * hpbl

    def wstar(akhs):
        x = btgh * akhs * dthv
        return W(x != 0.0, wwst2 * jnp.power(jnp.abs(W(x != 0.0, x, 1.0)), 2.0 / 3.0), 0.0)

    wstar2 = wstar(akhs)
    ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
    zt = jnp.exp(zilfc * jnp.sqrt(ustar * z0)) * z0
    zt = W(zt < 1.0e-15, 1.0e-6, zt)
    zslu = zlm + zu
    zslt = zlm + zt
    rlogu = jnp.log(zslu / zu)
    rlogt = jnp.log(zslt / zt)
    rlmo = elfc * akhs * dthv / (ustar * ustar * ustar)
    pspmu = lambda xx: -2.0 * jnp.log((xx + 1.0) * 0.5) - jnp.log((xx * xx + 1.0) * 0.5) + 2.0 * jnp.arctan(xx) - pihf
    psphu = lambda xx: -2.0 * jnp.log((xx * xx + 1.0) * 0.5)
    for _ in range(5):
        zetalt = jnp.fmax(zslt * rlmo, ztmin)
        rlmo = zetalt / zslt
        zetalu = zslu * rlmo
        zetau = zu * rlmo
        zetat = zt * rlmo
        unst = rlmo < 0.0
        xlu = jnp.sqrt(jnp.sqrt(W(unst, 1.0 - 16.0 * zetalu, 1.0)))
        xlt = jnp.sqrt(jnp.sqrt(W(unst, 1.0 - 16.0 * zetalt, 1.0)))
        xu = jnp.sqrt(jnp.sqrt(W(unst, 1.0 - 16.0 * zetau, 1.0)))
        xt = jnp.sqrt(jnp.sqrt(W(unst, 1.0 - 16.0 * zetat, 1.0)))
        simm_u = pspmu(xlu) - pspmu(xu) + rlogu
        simh_u = psphu(xlt) - psphu(xt) + rlogt
        zetalu_s = jnp.fmin(zetalu, ztmax)
        zetalt_s = jnp.fmin(zetalt, ztmax)
        simm_s = 5.0 * zetalu_s - 5.0 * zetau + rlogu
        simh_s = 5.0 * zetalt_s - 5.0 * zetat + rlogt
        simm = W(unst, simm_u, simm_s)
        simh = W(unst, simh_u, simh_s)
        ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
        zt = jnp.exp(zilfc * jnp.sqrt(ustar * z0)) * z0
        zt = W(zt < 1.0e-15, 1.0e-6, zt)
        zslt = zlm + zt
        rlogt = jnp.log(zslt / zt)
        ustark = ustar * vkrm
        akms = W(simm != 0.0, jnp.fmax(ustark / W(simm != 0.0, simm, 1.0), cxch), cxch)
        akhs = W(simh != 0.0, jnp.fmax(ustark / W(simh != 0.0, simh, 1.0), cxch), cxch)
        wstar2 = wstar(akhs)
        rlmn = elfc * akhs * dthv / (ustar * ustar * ustar)
        rlmo = rlmo * wold + rlmn * (1.0 - wold)
    return akms, akhs, ustar


def sflx(forc, par, st, dt, sldpth):
    """One Noah 2.7.1 step for one column (ICE = 0).
    forc: sfctmp q2 soldn lwdn sfcprs prcp ffrozp q2sat dqsdt2 th2 ch (scalars); par: static/dynamic parameters; st: state
    (cmc t1 stc[4] smc[4] sh2o[4] snowh sneqv). Returns (new_state, outputs)."""
    sfctmp, q2, soldn, lwdn = forc["sfctmp"], forc["q2"], forc["soldn"], forc["lwdn"]
    sfcprs, prcp, ffrozp = forc["sfcprs"], forc["prcp"], forc["ffrozp"]
    q2sat, dqsdt2, th2, ch = forc["q2sat"], forc["dqsdt2"], forc["th2"], forc["ch"]
    nroot = par["nroot"]
    shdfac, alb, snoalb, snup = par["shdfac"], par["alb"], par["snoalb"], par["snup"]
    smcwlt, smcdry, smcref, smcmax = par["smcwlt"], par["smcdry"], par["smcref"], par["smcmax"]
    emissi = par["emissi"]
    cmc, t1 = st["cmc"], st["t1"]
    stc = [st["stc"][k] for k in range(NS)]
    smc = [st["smc"][k] for k in range(NS)]
    sh2o = [st["sh2o"][k] for k in range(NS)]
    snowh, sneqv = st["snowh"], st["sneqv"]
    zsoil = [-sldpth[0]]
    for k in range(1, NS):
        zsoil.append(-sldpth[k] + zsoil[k - 1])
    zs_root = jnp.stack(zsoil)[nroot - 1]
    rtdis = [-sldpth[i] / zs_root for i in range(NS)]
    has0 = sneqv != 0.0
    sndens = W(has0, sneqv / W(has0, snowh, 1.0), 0.0)
    snowh = W(has0, snowh, 0.0)
    sncond = W(has0, csnow(sndens), 1.0)
    snowng = (prcp > 1.0e-20) & (ffrozp > 0.5)
    frzgra = (prcp > 1.0e-20) & (~(ffrozp > 0.5)) & (t1 <= TFREEZ)
    addsn = snowng | frzgra
    sn_new = prcp * dt * 0.001
    sneqv_n = sneqv + sn_new
    snowh_n, sndens_n = snow_new(sfctmp, sn_new, snowh, sndens)
    sncond_n = csnow(sndens_n)
    sneqv = W(addsn, sneqv_n, sneqv)
    snowh = W(addsn, snowh_n, snowh)
    sndens = W(addsn, sndens_n, sndens)
    sncond = W(addsn, sncond_n, sncond)
    prcp1 = W(addsn, 0.0, prcp)
    nosnow = sneqv == 0.0
    sncovr_s = nj.snfrac(sneqv, snup, SALP)
    albedo_s = alb + sncovr_s * (snoalb - alb)
    albedo_s = W(albedo_s > snoalb, snoalb, albedo_s)
    sncovr = W(nosnow, 0.0, sncovr_s)
    albedo = W(nosnow, alb, albedo_s)
    emissi_out = W(nosnow, par["emiss_out0"], 0.95 * sncovr_s + emissi * (1.0 - sncovr_s))
    df1 = tdfcnd(smc[0], par["quartz"], smcmax, sh2o[0])
    df1 = df1 * jnp.exp(par["sbeta"] * shdfac)
    dsoil = -(0.5 * zsoil[0])
    dtot = snowh + dsoil
    frcsno = snowh / dtot
    frcsoi = dsoil / dtot
    df1a = frcsno * sncond + frcsoi * df1
    df1_sn = df1a * sncovr + df1 * (1.0 - sncovr)
    ssoil = W(nosnow, df1 * (t1 - stc[0]) / dsoil, df1_sn * (t1 - stc[0]) / dtot)
    df1 = W(nosnow, df1, df1_sn)
    t2v = sfctmp * (1.0 + 0.61 * q2)
    fdown = soldn * (1.0 - albedo) + lwdn
    etp, rch, epsca, rr, t24, flx2 = penman(sfctmp, sfcprs, ch, t2v, th2, prcp, fdown, ssoil, q2, q2sat, snowng, frzgra, dqsdt2, emissi)
    can = shdfac > 0.0
    rc_c, pc_c, rcs, rct, rcq, rcsoil = nj.canres(soldn, ch, sfctmp, q2, sfcprs, sh2o, zsoil, smcwlt, smcref, par["rsmin"], nroot,
                                                  q2sat, dqsdt2, par["topt"], par["rsmax"], par["rgl"], par["hs"], par["xlai"], emissi)
    rc = W(can, rc_c, 0.0)
    pc = W(can, pc_c, 0.0)
    rcs, rct, rcq, rcsoil = [W(can, v, 0.0) for v in (rcs, rct, rcq, rcsoil)]
    common = dict(smc=smc, smcmax=smcmax, smcwlt=smcwlt, smcref=smcref, smcdry=smcdry, cmc=cmc, cmcmax=par["cmcmax"], dt=dt,
                  sfctmp=sfctmp, t24=t24, th2=th2, fdown=fdown, stc=stc, bexp=par["bexp"], pc=pc, rch=rch, rr=rr,
                  cfactr=par["cfactr"], sh2o=sh2o, slope=par["slope"], kdt=par["kdt"], frzx=par["frzx"], psisat=par["psisat"],
                  zsoil=zsoil, dksat=par["dksat"], dwsat=par["dwsat"], tbot=par["tbot"], zbot=par["zbot"], nroot=nroot,
                  rtdis=rtdis, quartz=par["quartz"], fxexp=par["fxexp"], csoil=par["csoil"], shdfac=shdfac)
    npc = nopac(etp=etp, prcp=prcp, sbeta=par["sbeta"], epsca=epsca, emissi=emissi, q2=q2, **common)
    spc = snopac(etp=etp, prcp=prcp, prcp1=prcp1, snowng=snowng, df1=df1, q2=q2, t1=t1, sncovr=sncovr, esd=sneqv, sndens=sndens,
                 snowh=snowh, flx2=flx2, **common)
    sel = lambda a, b: W(nosnow, a, b)
    selL = lambda a, b: [W(nosnow, a[k], b[k]) for k in range(NS)]
    t1 = sel(npc["t1"], spc["t1"])
    stc = selL(npc["stc"], spc["stc"])
    smc = selL(npc["smc"], spc["smc"])
    sh2o = selL(npc["sh2o"], spc["sh2o"])
    cmc = sel(npc["cmc"], spc["cmc"])
    et = selL(npc["et"], spc["et"])
    edir, ec, ett = sel(npc["edir"], spc["edir"]), sel(npc["ec"], spc["ec"]), sel(npc["ett"], spc["ett"])
    drip = sel(npc["drip"], spc["drip"])
    ro1, ro2, ro3 = sel(npc["ro1"], spc["ro1"]), sel(npc["ro2"], spc["ro2"]), sel(npc["ro3"], spc["ro3"])
    ssoil = sel(npc["ssoil"], spc["ssoil"])
    tsoil = sel(npc["tsoil"], spc["tsoil"])
    dew = sel(npc["dew"], spc["dew"])
    esnow = sel(npc["esnow"], spc["esnow"])
    snomlt = sel(npc["snomlt"], spc["snomlt"])
    flx1 = sel(npc["flx1"], spc["flx1"])
    flx3 = sel(npc["flx3"], spc["flx3"])
    etanrg = sel(npc["etanrg"], spc["etanrg"])
    eta_kin = sel(npc["eta"], f32(0.0))                  # SNOPAC never sets ETA: the caller's 0
    sneqv = sel(sneqv, spc["esd"])
    snowh = sel(snowh, spc["snowh"])
    sncovr = sel(sncovr, spc["sncovr"])
    q1 = q2 + eta_kin * CP / rch
    sheat = -(ch * CP * sfcprs) / (RD * t2v) * (th2 - t1)
    evp = W(etp > 0.0, edir + ec + ett + esnow, etp)
    edir = edir * LVH2O
    ec = ec * LVH2O
    et = [e * LVH2O for e in et]
    ett = ett * LVH2O
    esnow = esnow * LSUBS
    etp = etp * ((1.0 - sncovr) * LVH2O + sncovr * LSUBS)
    eta = W(etp > 0.0, edir + ec + ett + esnow, etp)
    beta = W(etp == 0.0, 0.0, eta / W(etp == 0.0, 1.0, etp))
    ssoil = -1.0 * ssoil
    ro3 = ro3 / dt
    ro2 = ro2 + ro3
    soilm = -1.0 * smc[0] * zsoil[0]
    for k in range(1, NS):
        soilm = soilm + smc[k] * (zsoil[k - 1] - zsoil[k])
    soilwm = -1.0 * (smcmax - smcwlt) * zsoil[0]
    soilww = -1.0 * (smc[0] - smcwlt) * zsoil[0]
    for k in range(1, NS):
        inroot = k < nroot
        soilwm = W(inroot, soilwm + (smcmax - smcwlt) * (zsoil[k - 1] - zsoil[k]), soilwm)
        soilww = W(inroot, soilww + (smc[k] - smcwlt) * (zsoil[k - 1] - zsoil[k]), soilww)
    soilw = soilww / soilwm
    soiltm = -1.0 * (smcmax - smcwlt) * zsoil[0]
    soiltw = -1.0 * (smc[0] - smcwlt) * zsoil[0]
    for k in range(1, NS):
        soiltm = soiltm + (smcmax - smcwlt) * (zsoil[k - 1] - zsoil[k])
        soiltw = soiltw + (smc[k] - smcwlt) * (zsoil[k - 1] - zsoil[k])
    soilt = soiltw / soiltm
    clr = (t1 >= TFREEZ + 2.0) & ((snowh <= 1.0e-6) | (sneqv <= 1.0e-3))
    snowh = W(clr, 0.0, snowh)
    sneqv = W(clr, 0.0, sneqv)
    new_st = dict(cmc=cmc, t1=t1, stc=jnp.stack(stc), smc=jnp.stack(smc), sh2o=jnp.stack(sh2o), snowh=snowh, sneqv=sneqv)
    out = dict(albedo=albedo, sncovr=sncovr, evp=evp, eta=eta, sheat=sheat, eta_kin=eta_kin, ec=ec, edir=edir, et=et, ett=ett,
               esnow=esnow, drip=drip, dew=dew, beta=beta, etp=etp, gflx=ssoil, flx1=flx1, flx2=flx2, flx3=flx3, snomlt=snomlt,
               ro1=ro1, ro2=ro2, ro3=ro3, rc=rc, pc=pc, rcs=rcs, rct=rct, rcq=rcq, rcsoil=rcsoil, q1=q1, soilm=soilm,
               etanrg=etanrg, tsoil=tsoil, emissi_out=emissi_out, soilw=soilw, soilt=soilt)
    return new_st, out
