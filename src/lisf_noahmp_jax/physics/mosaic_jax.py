"""JAX port of the LIS Mosaic land surface model (lis/surfacemodels/land/mosaic: mos_main.F90 preparation, MOSTILE and its
callees in mos_tile.F90, umd_sibalb.F90). All routines work on arrays over tiles (float32, as the default REAL of the Fortran).
Branches are written with jnp.where; MIN/MAX use fmin/fmax (gfortran returns the non-NaN argument)."""
import numpy as np
import jax.numpy as jnp

f32 = np.float32
W = jnp.where
fmin, fmax = jnp.fmin, jnp.fmax
# parameters of mos_tile.F90 (compile-time REAL constants)
ALHE, ALHS = f32(2.4548e6), f32(2.8368e6)
ALHM = ALHS - ALHE
TF = f32(273.16)
RGAS_T = f32(0.286) * f32(1003.5)          # RGAS of MOSTILE
RHOW = f32(1000.)
EPSILON = f32(18.01) / f32(28.97)
PIE = f32(3.14159265)
DELTC, DELEA = f32(0.01), f32(0.001)
CSOIL0 = f32(70000.)
NLAY = 3


def m_qsat(t, pr, alhx):
    return f32(3.797915) * jnp.exp((alhx / ALHE) * (f32(21.18123) - f32(1.33445) - f32(5418.) / t)) / pr


def m_esat(t):
    return jnp.exp(f32(21.18123) - f32(5418.) / t) / EPSILON


def umd_sibalb(sib, vlai, vgrn, zth, ityp):
    """sib: array (6, 13, 2, 14) = ALVDR BTVDR GMVDR ALIDR BTIDR GMIDR [ityp-1, green index, lai index].
    Returns avisdr, anirdr, avisdf, anirdf."""
    templai = W((ityp != 12) & (vlai < f32(0.5)), f32(0.5), vlai)
    alatrm = f32(0.5) + f32(13) * f32(0.5) - f32(1.e-6)
    ala = fmin(fmax(f32(0.), templai), alatrm)
    lai = 1 + jnp.maximum(0, ((ala - f32(0.5)) / f32(0.5)).astype(jnp.int32))
    dx = (ala - (f32(0.5) + (lai - 1).astype(jnp.float32) * f32(0.5))) * (f32(1.) / f32(0.5))
    dy = (vgrn - f32(0.33)) * (f32(1.) / (f32(0.67) - f32(0.33)))
    it = ityp - 1

    def coeff(k):
        t = lambda li, g: sib[k, it, g, li - 1]
        return ((t(lai, 0) + (t(lai, 1) - t(lai, 0)) * dy) * (f32(1.) - dx) + (t(lai + 1, 0) + (t(lai + 1, 1) - t(lai + 1, 0)) * dy) * dx)
    out = []
    for kk in (0, 3):
        alpha, beta, gamma = coeff(kk), coeff(kk + 1), coeff(kk + 2)
        dr = alpha - zth * beta / (gamma + zth)
        df = alpha - beta + f32(2.) * beta * gamma * (f32(1.) - gamma * jnp.log((f32(1.) + gamma) / gamma))
        out.append((dr, df))
    (avisdr, avisdf), (anirdr, anirdf) = out
    return avisdr, anirdr, avisdf, anirdf


def m_pmonth(green, lai, vgrdrs, vgz2, vgrotd, vgroca, vgrdc, vgrotl, vgz0, vgdd, vgtr12, vgtr11, vgrf11, vgrf12):
    vkc = f32(0.35)
    rdc = vgrdc
    rootl = vgrotl
    rlengt = vgz0
    z0 = rlengt
    dparam = vgdd
    vroot = rootl * vgroca
    dum1 = jnp.log(vroot / (1 - vroot))
    dum2 = f32(1.) / (f32(8.) * f32(3.14159) * rootl)
    alphaf = dum2 * (vroot - f32(3.) - f32(2.) * dum1)
    rsoil1 = vgrdrs / (rootl * vgrotd)
    rsoil2 = alphaf / vgrotd
    scat = green * (vgtr11 + vgrf11) + (f32(1.) - green) * (vgtr12 + vgrf12)
    sqscat = jnp.sqrt(f32(1.) - scat)
    dzm = f32(600.)
    cuni = (jnp.log(f32(0.025) * dzm / rlengt) / vkc) + f32(8.4)
    dparam = W(dparam > vgz2, f32(0.66) * vgz2, dparam)
    u2fac = jnp.log((vgz2 - dparam) / rlengt) / (cuni * vkc)
    z2 = vgz2
    satcap = f32(0.2) * lai
    return z0, sqscat, z2, dzm, rsoil1, rsoil2, satcap, rdc, u2fac


# ---------------------------------------------------------------------------------------- MOSTILE callees
def m_rcunst(sunang, sqscat, pdir, par, zlai, green, vgchil, vgzmew, vgrst1, vgrst2, vgrst3):
    aa = f32(0.5) - (f32(0.633) + f32(0.330) * vgchil) * vgchil
    bb = f32(0.877) * (f32(1.) - f32(2.) * aa)
    cc = (aa + bb * sunang) / sunang
    extk1 = cc * sqscat
    extk2 = (f32(1.) / vgzmew) * sqscat
    dum1 = pdir * cc
    dum2 = (f32(1.) - pdir) * (bb * (f32(1.) / f32(3.) + PIE / f32(4.)) + aa * f32(1.5))
    zk = pdir * fmin(extk1, f32(50.) / zlai) + (f32(1.) - pdir) * fmin(extk2, f32(50.) / zlai)
    gamma = vgrst1 / vgrst3 + vgrst2
    ekat = jnp.exp(zk * zlai)
    par0 = fmax(par, f32(.000001))
    rho4 = gamma / (par0 * (dum1 + dum2))
    dum1 = (vgrst2 - gamma) / (gamma + f32(1.e-20))
    dum2 = (rho4 * ekat + f32(1.)) / (rho4 + f32(1.))
    dum3 = zk * vgrst3
    rcinv = (dum1 * jnp.log(dum2) + zk * zlai) / dum3
    return f32(1.) / (rcinv * green + f32(1.e-10))


def m_vpdfac(esattc, ea, vgdfac):
    v = f32(1.) - (esattc - ea) * vgdfac
    return fmin(f32(1.), fmax(v, f32(1.e-10)))


def m_tmpfac(tc, vgtll, vgtu, vgtcf1, vgtcf2, vgtcf3):
    ft = (tc - vgtll) * (tc - vgtu) * (vgtcf1 * tc * tc + vgtcf2 * tc + vgtcf3)
    ft = W((tc <= vgtll) | (tc >= vgtu), f32(1.e-10), ft)
    return fmin(f32(1.), fmax(ft, f32(1.e-10)))


def soil(wet, vgbeex, vgpsax, vgcsax, delz):
    wet0 = fmax(wet, f32(0.01))
    wexpb = wet0 ** vgbeex
    phr = vgpsax / wexpb
    soilco = vgcsax * wexpb * wexpb * wet0 * wet0 * wet0
    pheq = phr - delz
    weteq = (pheq / vgpsax) ** (-f32(1.) / vgbeex)
    return phr, soilco, weteq


def smfac(esattc, ea, phr, soilco, rcun, vpdstr, ftemp, tc, psur, z2, rsoil1, rsoil2, vgph1x, vgph2x, vgrplx):
    rcuntd = rcun / (vpdstr * ftemp)
    rhoair = psur * f32(100.) / (RGAS_T * tc)
    lis_const = rhoair * EPSILON / psur
    de = (esattc - ea) * lis_const
    d12 = vgph1x - vgph2x
    dr2 = phr - z2 - vgph2x
    rsoil = rsoil1 + rsoil2 / (soilco + f32(1.0e-08))
    r0 = (vgrplx + rsoil) / RHOW
    den = rcuntd * d12 + de * r0
    small = jnp.abs(den) < f32(0.00000001)
    eest = W(small, de * dr2 / f32(0.00000001), de * dr2 / W(small, f32(1.), den))
    rstfac = (dr2 - r0 * eest) / d12
    rstfac = fmin(f32(1.), fmax(f32(0.001), rstfac))
    return rcuntd / rstfac


def rsurfp(um, u2fac, rdc, wet, esattc, ea, rc):
    u2 = um * u2fac
    rsurf = rdc / u2 + f32(26.) + f32(6.) / (f32(1.E-08) + wet) ** 2
    hesat = esattc * jnp.minimum(f32(1.), wet * f32(2.))
    diff = hesat - ea
    r1 = rsurf * (f32(1.) + (esattc - hesat) / W(diff != 0, diff, f32(1e-08)))
    rsurf = W(ea < hesat, r1, f32(1.E08))
    return rc * rsurf / (rc + rsurf), rc, rsurf       # (new RC, RX1, RX2)


def m_rcanop(capac, snow, satcap, ra, eturb, snwfrc, potfrc, rc):
    rc = W(satcap > f32(.001), rc * (f32(1.) - potfrc) / (f32(1.) + potfrc * rc / ra), rc)
    rc = W(satcap <= f32(.001), rc * (f32(1.) - snwfrc) / (f32(1.) + snwfrc * rc / ra), rc)
    rampfc = eturb / f32(-2.E-6)
    rc = W(rampfc >= 0, rc * (f32(1.) - rampfc), rc)
    rc = W(rampfc > 1, f32(0.), rc)
    return rc


def m_fluxes(dtstep, esattc, desdtc, alhx, eturb, dedea, dedtc, hsturb, dhsdea, dhsdtc, rc, drcdea, drcdtc, swnet, hlwdwn,
             alwrad, blwrad, dtcanal, esnfrc, em, csoil, psur, emaxrt, tc, td, ea, snow, vgwmax_unused=None):
    """Returns tc, td, ea, snow, evap, shflux, smelt, hlwup, ghflux (updated)."""
    hlwtc = alwrad + blwrad * tc
    cdeeps = PIE * csoil * f32(2.) / f32(86400.)
    rhoair = psur * f32(100.) / (RGAS_T * tc)
    lis_const = rhoair * EPSILON / psur
    dhlwtc = blwrad
    a11 = csoil / dtstep + dhlwtc + dhsdtc + alhx * dedtc + cdeeps
    a12 = dhsdea + alhx * dedea
    q0 = swnet + hlwdwn - hlwtc - hsturb - alhx * eturb - cdeeps * (tc - td) + csoil / dtstep * dtcanal
    rcpos = rc > 0
    # RC > 0 branch
    eplant = lis_const * (esattc - ea) / rc
    eharmn = W(eplant * eturb > 0, f32(2.) * eplant * eturb / (eplant + eturb), f32(0.))
    a21p = -dedtc * rc + fmax(f32(0.), lis_const * desdtc - eharmn * drcdtc)
    a22p = -(rc * dedea + fmax(f32(0.), lis_const + eharmn * drcdea))
    f0p = rc * (eturb - eplant)
    determ = jnp.minimum(a12 * a21p / (a11 * a22p) - f32(1.), f32(-0.1))
    deap = (q0 * a21p - a11 * f0p) / (determ * a11 * a22p)
    denomp = determ * a11 * a22p
    # RC <= 0 branch
    a21n = -desdtc
    a22n = f32(1.)
    f0n = esattc - ea
    denn = a12 * a21n - a11 * a22n
    dean = (q0 * a21n - a11 * f0n) / denn
    a21 = W(rcpos, a21p, a21n); a22 = W(rcpos, a22p, a22n); f0 = W(rcpos, f0p, f0n)
    dea = W(rcpos, deap, dean)
    denom = W(rcpos, denomp, denn)
    dtc = (q0 - a12 * dea) / a11
    evap = eturb + dedea * dea + dedtc * dtc
    shflux = hsturb + dhsdea * dea + dhsdtc * dtc
    choke = rcpos
    smelt = f32(0.) * tc
    snleft = snow - evap * dtstep * esnfrc
    c1 = (tc + dtc > TF) & (snleft > 0)
    q0x = q0 - ALHM * snleft / dtstep
    smelt1 = snleft / dtstep
    dea1 = (q0x * a21 - a11 * f0) / denom
    dtc1 = (q0x - a12 * dea1) / a11
    c1b = c1 & (tc + dtc1 < TF)
    dtc1 = W(c1b, TF - tc, dtc1)
    dea1 = W(c1b, (f0 - a21 * dtc1) / a22, dea1)
    q0snow = a11 * dtc1 + a12 * dea1
    smelt1 = W(c1b, (q0 - q0snow) / ALHM, smelt1)
    evap1 = eturb + dedea * dea1 + dedtc * dtc1
    shflux1 = hsturb + dhsdea * dea1 + dhsdtc * dtc1
    dea = W(c1, dea1, dea); dtc = W(c1, dtc1, dtc); smelt = W(c1, smelt1, smelt)
    evap = W(c1, evap1, evap); shflux = W(c1, shflux1, shflux)
    # EVAP > EMAXRT
    c2 = evap > emaxrt
    choke = choke & ~c2
    dea2 = em - ea
    dtc2 = (q0 + alhx * (eturb - emaxrt) - dhsdea * dea2) / (a11 - alhx * dedtc)
    shflux2 = hsturb + dhsdea * dea2 + dhsdtc * dtc2
    dea = W(c2, dea2, dea); dtc = W(c2, dtc2, dtc); evap = W(c2, emaxrt, evap); shflux = W(c2, shflux2, shflux)
    smelt = W(c2, f32(0.), smelt)
    esatnw = esattc + desdtc * dtc
    # EVAP < 0 and EM < ESATNW
    c3 = (evap < 0) & (em < esatnw)
    choke = choke & ~c3
    dea3 = em - ea
    dtc3 = (q0 + alhx * eturb - dhsdea * dea3) / (a11 - alhx * dedtc)
    shflux3 = hsturb + dhsdea * dea3 + dhsdtc * dtc3
    c3m = c3 & (tc + dtc3 > TF) & (snow > 0)
    q0x3 = q0 - ALHM * snow / dtstep
    smelt3 = snow / dtstep
    dtc3m = (q0x3 + alhx * eturb - dhsdea * dea3) / (a11 - alhx * dedtc)
    c3mm = c3m & (tc + dtc3m < TF)
    dtc3m = W(c3mm, TF - tc, dtc3m)
    q0snow3 = a11 * dtc3m + a12 * dea3
    smelt3 = W(c3mm, (q0 - q0snow3) / ALHM, smelt3)
    shflux3m = hsturb + dhsdea * dea3 + dhsdtc * dtc3m
    dea = W(c3, dea3, dea)
    dtc = W(c3, W(c3m, dtc3m, dtc3), dtc)
    evap = W(c3, f32(0.), evap)
    shflux = W(c3, W(c3m, shflux3m, shflux3), shflux)
    smelt = W(c3, W(c3m, smelt3, smelt), smelt)
    # CHOKE and |DEA| > 0.5 EA
    c4 = choke & (jnp.abs(dea) > f32(0.5) * ea)
    dea4 = jnp.sign(dea) * f32(.5) * ea
    dea4 = W(dea == 0, f32(.5) * ea, dea4)             # SIGN(a, 0.) = +|a|
    dtc4 = (q0 - a12 * dea4) / a11
    evap4 = eturb + dedea * dea4 + dedtc * dtc4
    shflux4 = hsturb + dhsdea * dea4 + dhsdtc * dtc4
    c4m = c4 & (tc + dtc4 > TF) & (snow > 0)
    snleft4 = snow - evap4 * dtstep * esnfrc
    q0x4 = q0 - ALHM * snleft4 / dtstep
    smelt4 = snleft4 / dtstep
    dtc4m = (q0x4 - a12 * dea4) / a11
    c4mm = c4m & (tc + dtc4m < TF)
    dtc4m = W(c4mm, TF - tc, dtc4m)
    q0snow4 = a11 * dtc4m + a12 * dea4
    smelt4 = W(c4mm, (q0 - q0snow4) / ALHM, smelt4)
    evap4m = eturb + dedea * dea4 + dedtc * dtc4m
    shflux4m = hsturb + dhsdea * dea4 + dhsdtc * dtc4m
    dea = W(c4, dea4, dea)
    dtc = W(c4, W(c4m, dtc4m, dtc4), dtc)
    evap = W(c4, W(c4m, evap4m, evap4), evap)
    shflux = W(c4, W(c4m, shflux4m, shflux4), shflux)
    smelt = W(c4m, smelt4, smelt)
    tc = tc + dtc
    ea = ea + dea
    td = td + dtstep * cdeeps * (tc - td) / (csoil * f32(67.73))
    hlwup = hlwtc + dhlwtc * dtc
    snow = snow - smelt * dtstep
    hlwtc = alwrad + blwrad * tc
    ghflux = swnet + hlwdwn - hlwtc - alhx * evap - shflux - smelt * ALHM
    ea = fmax(ea, f32(0.0))
    return tc, td, ea, snow, evap, shflux, smelt, hlwup, ghflux


def m_wupdat(dtstep, evap, satcap, vgwmax, rx1, rx2, esnfrc, eirfrc, capac, snow, swet, runoff, runsrf, fwsoil):
    """swet: list of 3. Returns capac, snow, swet, runoff, eint, esoi, eveg, esno, runsrf, fwsoil."""
    wlay1 = swet[0] * vgwmax[0]
    wlay2 = swet[1] * vgwmax[1]
    wtotal = wlay1 + wlay2
    esno = esnfrc * evap * dtstep
    eint = eirfrc * evap * dtstep
    egro = evap * dtstep - esno - eint
    c = esno > snow
    eint = W(c, eint + (esno - snow), eint); esno = W(c, snow, esno)
    egroi = egro + eint
    c = egroi > capac + wtotal
    esno = W(c, esno + egroi - (capac + wtotal), esno)
    egroi = W(c, capac + wtotal, egroi)
    eint = egroi - egro
    c = eint > capac
    egro = W(c, egro + eint - capac, egro); eint = W(c, capac, eint)
    c = egro > wtotal
    eint = W(c, eint + egro - wtotal, eint); egro = W(c, wtotal, egro)
    s = rx1 + rx2
    esoi = W(s != 0, egro * rx1 / W(s != 0, s, f32(1.)), egro / f32(2.))
    eveg = W(s != 0, egro - esoi, egro / f32(2.))
    fws = W(wtotal > 0, eveg / W(wtotal > 0, wtotal, f32(1.)), f32(1.))
    elay1 = wlay1 * fws + esoi
    elay2 = wlay2 * fws
    c = elay1 > wlay1
    elay2 = W(c, elay2 + (elay1 - wlay1), elay2); elay1 = W(c, wlay1, elay1)
    c = elay2 > wlay2
    elay1 = W(c, elay1 + (elay2 - wlay2), elay1); elay2 = W(c, wlay2, elay2)
    neg = evap < 0
    eint = W(neg, (f32(1.) - esnfrc) * evap * dtstep, eint)
    elay1 = W(neg, f32(0.), elay1); elay2 = W(neg, f32(0.), elay2)
    esoi = W(neg, f32(0.), esoi); eveg = W(neg, f32(0.), eveg)
    esno = W(neg, esnfrc * evap * dtstep, esno)
    snow = snow - esno
    capac = capac - eint
    s0 = fmin(f32(1.), fmax(f32(0.), (wlay1 - elay1) / vgwmax[0]))
    s1 = fmin(f32(1.), fmax(f32(0.), (wlay2 - elay2) / vgwmax[1]))
    c = capac > satcap
    thru = capac - satcap
    capac = W(c, satcap, capac)
    s0n = s0 + thru / vgwmax[0]
    dewrun = W(c & (s0n > 1.), (s0n - f32(1.)) * vgwmax[0], f32(0.))
    s0 = W(c, W(s0n > 1., f32(1.), s0n), s0)
    runoff = runoff + dewrun / dtstep
    runsrf = runsrf + dewrun / dtstep
    fwsoil = W(c, fwsoil + (thru - dewrun) / dtstep, fwsoil)
    return capac, snow, [s0, s1, swet[2]], runoff, eint, esoi, eveg, esno, runsrf, fwsoil


def gwater(wsmax, phlay, aklay, tc, dtstep, vgzdex, vgslox, weteq1, weteq2, phsat, aksat, swet, runoff):
    s = list(swet)
    zdep12 = vgzdex[0] + vgzdex[1]
    c = s[0] > weteq1
    farea = W(c, (s[0] - weteq1) / (f32(1.) - weteq1), f32(1.))
    dhdz = W(c, f32(1.) + f32(2.) * (phsat - phlay[1]) / zdep12, f32(1.) + f32(2.) * (phlay[0] - phlay[1]) / zdep12)
    akave = W(c, aksat, aklay[1])
    gwflux = f32(1000.) * dtstep * akave * dhdz * farea
    halfmx = f32(0.5) * jnp.abs(s[0] - weteq1) * wsmax[0]
    pos = gwflux >= 0
    gp = fmin(gwflux, s[0] * wsmax[0]); gp = fmin(gp, wsmax[1] - s[1] * wsmax[1]); gp = fmin(gp, halfmx)
    gn = -fmin(jnp.abs(gwflux), s[1] * wsmax[1]); gn = -fmin(jnp.abs(gn), wsmax[0] - s[0] * wsmax[0]); gn = -fmin(jnp.abs(gn), halfmx)
    gwflux = W(pos, gp, gn)
    gwflux = W(tc < TF, f32(0.), gwflux)
    s[0] = s[0] - gwflux / wsmax[0]
    s[1] = s[1] + gwflux / wsmax[1]
    zdep23 = vgzdex[1] + vgzdex[2]
    dhdz = f32(1.) + f32(2.) * (phlay[1] - phlay[2]) / zdep23
    akave = W(dhdz >= 0, aklay[1], aklay[2])
    gwflux = f32(1000.) * dtstep * akave * dhdz
    halfmx = f32(0.5) * jnp.abs(s[1] - weteq2) * wsmax[1]
    pos = gwflux >= 0
    gp = fmin(gwflux, s[1] * wsmax[1]); gp = fmin(gp, wsmax[2] - s[2] * wsmax[2]); gp = fmin(gp, halfmx)
    gn = -fmin(jnp.abs(gwflux), s[2] * wsmax[2]); gn = -fmin(jnp.abs(gn), wsmax[1] - s[1] * wsmax[1]); gn = -fmin(jnp.abs(gn), halfmx)
    gwflux = W(pos, gp, gn)
    gwflux = W(tc <= TF, f32(0.), gwflux)
    s[1] = s[1] - gwflux / wsmax[1]
    s[2] = s[2] + gwflux / wsmax[2]
    gwflux = vgslox * aklay[2] * f32(1000.) * dtstep
    gwflux = W(tc <= TF, f32(0.), gwflux)
    gwflux = fmin(gwflux, s[2] * wsmax[2])
    s[2] = s[2] - gwflux / wsmax[2]
    gdrain = gwflux / dtstep
    runoff = runoff + gwflux / dtstep
    return s, runoff, gdrain


def m_interc(dtstep, trainl, trainc, tsnow, satcap, wmax1, csoil, weteq1, tc, capac, snow, swet1, runoff, runsrf, smelt, fwsoil, ghflux):
    fwetl, fwetc, timfrl, timfrc = f32(1.0), f32(0.20), f32(1.0), f32(0.125)
    snow = snow + tsnow * dtstep
    snowm = f32(0.) * tc
    c = (snow > 0) & (tc > TF)
    sm = fmin(snow, fmax(f32(0.), (tc - TF) * csoil / ALHM))
    full = c & (sm == snow)
    tc_new = W(full, tc - sm * ALHM / csoil, W(c, TF, tc))
    snow_new = W(full, f32(0.), W(c, snow - sm, snow))
    smpers = sm / dtstep
    smelt = W(c, smelt + smpers, smelt)
    ghflux = W(c, ghflux - smpers * ALHM, ghflux)
    tc, snow = tc_new, snow_new

    def stage(capac, watadd, fwet, timfr):
        xtcorr = (f32(1.) - timfr) * fmin(f32(1.), (capac / satcap) / fwet)
        cavail = (satcap - capac) * fwet
        wetint = capac / satcap
        thru1 = W(watadd * (f32(1.) - wetint) < cavail, watadd * wetint, watadd - cavail)
        thru1 = thru1 * (f32(1.) - xtcorr)
        thru2 = xtcorr * watadd
        thru = thru1 + thru2
        return capac + watadd - thru1 - thru2, thru
    capac, thrul = stage(capac, trainl * dtstep + smelt * dtstep, fwetl, timfrl)
    capac, thruc = stage(capac, trainc * dtstep, fwetc, timfrc)

    def soil_stage(swet1, thru, fwet):
        wetfrc = W((swet1 > weteq1) & (weteq1 != 1.), (swet1 - weteq1) / (f32(1.) - weteq1), f32(0.))
        cavail = (f32(1.) - wetfrc) * fwet * wmax1 * (f32(1.) - weteq1)
        c = thru * (f32(1.) - wetfrc) < cavail
        wrun = W(c, thru * wetfrc, thru - cavail)
        swet1 = W(c, swet1 + thru * (f32(1.) - wetfrc) / wmax1, swet1 + cavail / wmax1)
        return swet1, wrun
    swet1, wrunl = soil_stage(swet1, thrul, fwetl)
    swet1, wrunc = soil_stage(swet1, thruc, fwetc)
    runoff = runoff + (wrunc + wrunl) / dtstep
    runsrf = runsrf + (wrunc + wrunl) / dtstep
    fwsoil = fwsoil + (thruc + thrul - wrunc - wrunl) / dtstep
    return tc, capac, snow, swet1, runoff, runsrf, smelt, fwsoil, ghflux


def mostile(dtstep, trainl, trainc, tsnow, um, eturb, dedqa, dedtc, hsturb, dhsdqa, dhsdtc, tm, qm, cd, sunang, pardir, pardif,
            swnet, hlwdwn, psur, zlai, green, z2, sqscat, rsoil1, rsoil2, rdc, u2fac, qsattc, dqsdtc, alwrad, blwrad, dtcanal,
            tc, td, qa, swet_in, capac, snow, vgbeex, vgpsax, vgcsax, vgzdex, vgslox, vgph1x, vgph2x, vgrplx, vgwmax, snwmid,
            vgchil, vgzmew, vgrst1, vgrst2, vgrst3, vgdfac, vgtll, vgtu, vgtcf1, vgtcf2, vgtcf3, ra):
    """vgzdex, vgwmax: lists of 3 arrays; swet_in list of 3. Returns dict of states and outputs."""
    delz12 = (vgzdex[0] + vgzdex[1]) / f32(2.0)
    delz23 = (vgzdex[1] + vgzdex[2]) / f32(2.0)
    por = [vgwmax[k] / (vgzdex[k] * f32(1000.)) for k in range(3)]
    dedqa = fmax(dedqa, f32(500.) / ALHE)
    dedtc = fmax(dedtc, f32(0.))
    dhsdqa = fmax(dhsdqa, f32(0.))
    dhsdtc = fmax(dhsdtc, f32(-10.))
    em = qm * psur / EPSILON
    ea = qa * psur / EPSILON
    esattc = qsattc * psur / EPSILON
    desdtc = dqsdtc * psur / EPSILON
    dedea = dedqa * EPSILON / psur
    dhsdea = dhsdqa * EPSILON / psur
    par = pardir + pardif + f32(1.E-20)
    pdir = pardir / par
    zlai = W(zlai == 0, f32(0.00001), zlai)
    satcap = f32(0.2) * zlai
    csoil = f32(CSOIL0)
    swet = [fmax(fmin(s, f32(1.)), f32(0.)) for s in swet_in]
    capac = fmax(fmin(capac, satcap), f32(0.))
    snwfrc = snow / (snow + snwmid)
    fcan = fmin(f32(1.), fmax(f32(0.), capac / satcap))
    potfrc = f32(1.) - (f32(1.) - snwfrc) * (f32(1.) - fcan)
    wsoi12 = swet[0] * vgwmax[0] + swet[1] * vgwmax[1]
    swet12 = wsoi12 / (vgwmax[0] + vgwmax[1])
    emaxrt = (snow + capac + wsoi12) / dtstep
    runoff = f32(0.) * tc; runsrf = runoff; fwsoil = runoff
    # canopy / surface resistance and its derivatives
    rcun = m_rcunst(sunang, sqscat, pdir, par, zlai, green, vgchil, vgzmew, vgrst1, vgrst2, vgrst3)
    vpdstr = m_vpdfac(esattc, ea, vgdfac)
    ftemp = m_tmpfac(tc, vgtll, vgtu, vgtcf1, vgtcf2, vgtcf3)
    phr, soilco, _ = soil(swet12, vgbeex, vgpsax, vgcsax, delz12)
    rc = smfac(esattc, ea, phr, soilco, rcun, vpdstr, ftemp, tc, psur, z2, rsoil1, rsoil2, vgph1x, vgph2x, vgrplx)
    temprc = rc
    rc, rx1, rx2 = rsurfp(um, u2fac, rdc, swet[0], esattc, ea, rc)
    rc = m_rcanop(capac, snow, satcap, ra, eturb, snwfrc, potfrc, rc)
    tx = tc + DELTC
    esattx = esattc + desdtc * DELTC
    eax = ea + DELEA
    vpdstx = m_vpdfac(esattx, ea, vgdfac)
    ftempx = m_tmpfac(tx, vgtll, vgtu, vgtcf1, vgtcf2, vgtcf3)
    rcx = smfac(esattx, ea, phr, soilco, rcun, vpdstx, ftempx, tx, psur, z2, rsoil1, rsoil2, vgph1x, vgph2x, vgrplx)
    rcx, _, _ = rsurfp(um, u2fac, rdc, swet[0], esattx, ea, rcx)
    rcx = m_rcanop(capac, snow, satcap, ra, eturb, snwfrc, potfrc, rcx)
    drcdtc = (rcx - rc) / DELTC
    vpdstx = m_vpdfac(esattc, eax, vgdfac)
    rcx = smfac(esattc, eax, phr, soilco, rcun, vpdstx, ftemp, tc, psur, z2, rsoil1, rsoil2, vgph1x, vgph2x, vgrplx)
    rcx, _, _ = rsurfp(um, u2fac, rdc, swet[0], esattc, eax, rcx)
    rcx = m_rcanop(capac, snow, satcap, ra, eturb, snwfrc, potfrc, rcx)
    drcdea = (rcx - rc) / DELEA
    rtbs = rx1 * rx2 / (rx1 + rx2)
    epfrc = potfrc * (ra + rtbs) / (ra + potfrc * rtbs)
    esnfrc = epfrc * snwfrc / (snwfrc + fcan + f32(1.E-20))
    eirfrc = epfrc * fcan / (snwfrc + fcan + f32(1.E-20))
    alhx = (f32(1.) - esnfrc) * ALHE + esnfrc * ALHS
    condry = f32(1.) / (rtbs + f32(1.E-20))
    tc, td, ea, snow, evap, shflux, smelt, hlwup, ghflux = m_fluxes(
        dtstep, esattc, desdtc, alhx, eturb, dedea, dedtc, hsturb, dhsdea, dhsdtc, rc, drcdea, drcdtc, swnet, hlwdwn, alwrad,
        blwrad, dtcanal, esnfrc, em, csoil, psur, emaxrt, tc, td, ea, snow)
    capac, snow, swet, runoff, eint, esoi, eveg, esno, runsrf, fwsoil = m_wupdat(
        dtstep, evap, satcap, vgwmax, rx1, rx2, esnfrc, eirfrc, capac, snow, swet, runoff, runsrf, fwsoil)
    eadjst = (esno / dtstep) - esnfrc * evap
    shflux = W(evap > 0, shflux - eadjst * (ALHS - ALHE), shflux)
    phlay0, aklay0, _ = soil(swet[0], vgbeex, vgpsax, vgcsax, delz12)
    phlay1, aklay1, weteq1 = soil(swet[1], vgbeex, vgpsax, vgcsax, delz12)
    phlay2, aklay2, weteq2 = soil(swet[2], vgbeex, vgpsax, vgcsax, delz23)
    swet, runoff, gdrain = gwater(vgwmax, [phlay0, phlay1, phlay2], [aklay0, aklay1, aklay2], tc, dtstep, vgzdex, vgslox,
                                  weteq1, weteq2, vgpsax, vgcsax, swet, runoff)
    _, _, weteq1 = soil(swet[1], vgbeex, vgpsax, vgcsax, delz12)
    swet1 = swet[0]
    tc, capac, snow, swet1, runoff, runsrf, smelt, fwsoil, ghflux = m_interc(
        dtstep, trainl, trainc, tsnow, satcap, vgwmax[0], csoil, weteq1, tc, capac, snow, swet1, runoff, runsrf, smelt, fwsoil, ghflux)
    qa = ea * EPSILON / psur
    hlatn = evap * alhx
    s1, s2, s3 = swet1, swet[1], swet[2]
    s1 = W(s1 < f32(1.E-10), f32(0.), s1); s2 = W(s2 < f32(1.E-10), f32(0.), s2); s3 = W(s3 < f32(1.E-10), f32(0.), s3)
    capac = W(capac < f32(1.E-10), f32(0.), capac)
    snow = W(snow < f32(1.E-10), f32(0.), snow)
    runoff = W(runoff < f32(1.E-10), f32(0.), runoff)
    eint = eint * ALHE / dtstep
    esoi = esoi * ALHE / dtstep
    eveg = eveg * ALHE / dtstep
    esno = esno * ALHS / dtstep
    water = [s1 * vgwmax[0], s2 * vgwmax[1], s3 * vgwmax[2]]
    c1 = (temprc >= f32(0.000001)) & (temprc <= f32(1000000))
    tempcc = W(c1, f32(1.) / W(c1, temprc, f32(1.)), f32(-1.))
    tempcc = W(tempcc < f32(0.000001), f32(0.), tempcc)
    tempcc = W(tempcc > 1, f32(-1.), tempcc)
    return dict(tc=tc, td=td, qa=qa, swet=[s1, s2, s3], capac=capac, snow=snow, evap=evap, shflux=shflux, runoff=runoff, eint=eint,
                esoi=esoi, eveg=eveg, esno=esno, smelt=smelt, hlatn=hlatn, hlwup=hlwup, gdrain=gdrain, runsrf=runsrf, fwsoil=fwsoil,
                ghflux=ghflux, water=water, por=por, tempcc=tempcc)
