"""JAX port of the LISF Noah 3.9 column physics (SFLX and the routines it calls).

Source: lis/surfacemodels/land/noah.3.9/phys/module_sf_noah39lsm.F90 (unmodified, read-only).
Scope (reduced, see jax-port/GOAL.md): UA_PHYS off, FASDAS off, WRF_HYDRO off, ICE=0 (no glacial
branch), 4 soil layers, OPT_THCND 1 or 2, SFCDIF_OFF with ILECH=0 and IZ0TLND=0. The LIS wrapper,
I/O and forcing readers are not ported; forcing, parameters and initial state are inputs.

Style: one function per Fortran routine, written on a single column with scalars; layers are Python
lists of 4 scalars (unrolled). Data-dependent IFs are jnp.where selects, so both branches are
evaluated. The whole step is vmap-ed over columns. All floats are float32 like the Fortran build.
"""
import jax
import jax.numpy as jnp
from jax import lax

W = jnp.where
f32 = jnp.float32
NS = 4


# constants (module_model_constants_39 and file-level PARAMETERs of module_sf_noah39lsm)
CP_MOD = 1004.5            # module constant cp = 7*r_d/2 with r_d = 287.
RD = 287.04                # file-level RD
SIGMA = 5.67e-8
CPH2O = 4.218e3
CPICE = 2.106e3
LSUBF = 3.335e5
EMISSI_S = 0.95
LVH2O = 2.501e6
LSUBS = 2.83e6
LSUBC = 2.501e6
TFREEZ = 273.15
T0 = 273.15


def _lsum(lst):
    """Sequential float32 sum in layer order (matches the Fortran DO loops)."""
    s = lst[0]
    for v in lst[1:]:
        s = s + v
    return s


# ----------------------------------------------------------------------------------------------
def csnow(dsnow):
    c = 0.328 * jnp.power(10.0, 2.25 * dsnow)
    return 2.0 * 0.11631 * c


def snow_new(temp, newsn, snowh, sndens):
    snowhc = snowh * 100.0
    newsnc = newsn * 100.0
    tempc = temp - 273.15
    dsnew = W(tempc <= -15.0, 0.05, 0.05 + 0.0017 * jnp.power(tempc + 15.0, 1.5))
    hnewc = newsnc / dsnew
    sndens_new = W(snowhc + hnewc < 1.0e-3, jnp.fmax(dsnew, sndens),
                   (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc))
    snowhc = snowhc + hnewc
    return snowhc * 0.01, sndens_new


def snfrac(sneqv, snup, salp, snowh=None, xlai=None, shdfac=None, ztopv=None, zbotv=None, ua=False):
    """Snow cover fraction. Returns sncovr; with ua=True returns (sncovr, xlai, fvb, gama, fbur, fgsn)."""
    rsnow = sneqv / snup
    sncovr = W(sneqv < snup, 1.0 - (jnp.exp(-salp * rsnow) - rsnow * jnp.exp(-salp)), 1.0)
    if not ua:
        return sncovr
    snupgrd = 0.02
    fgsn = W(sneqv < snupgrd, sneqv / snupgrd, 1.0)
    fbur = W((zbotv > 0.0) & (snowh > zbotv),
             W(zbotv <= 0.5, (snowh - 0.4 * zbotv) / (0.4 * (ztopv - zbotv)), (snowh - zbotv) / (ztopv - zbotv)), 0.0)
    fbur = jnp.fmin(jnp.fmax(fbur, 0.0), 1.0)
    xlai = xlai * (1.0 - fbur)
    fvb = shdfac * fgsn * (1.0 - fbur)
    gama = jnp.exp(-1.0 * xlai)
    return sncovr, xlai, fvb, gama, fbur, fgsn


def alcalc(alb, snoalb, embrd, sncovr, dt, snowng, snotime1, lvcoef):
    snoalb1 = snoalb + lvcoef * (0.85 - snoalb)
    snotime1 = W(snowng, 0.0, snotime1 + dt)
    snoalb2 = W(snowng, snoalb1,
                snoalb1 * jnp.power(0.94, jnp.power(snotime1 / 86400.0, 0.58)))
    snoalb2 = jnp.fmax(snoalb2, alb)
    albedo = alb + sncovr * (snoalb2 - alb)
    albedo = W(albedo > snoalb2, snoalb2, albedo)
    emissi = embrd + sncovr * (EMISSI_S - embrd)
    return albedo, emissi, snotime1


def snowz0(sncovr, z0brd, snowh, ua=False, fbur=None, fgsn=None, shdmax=None):
    z0s = 0.001
    if ua:
        z0g = 0.01
        fv = shdmax * (1.0 - fbur)
        a1 = (1.0 - fv) ** 2 * ((1.0 - fgsn ** 2) * jnp.log(z0g) + (fgsn ** 2) * jnp.log(z0s))
        a2 = (1.0 - (1.0 - fv) ** 2) * jnp.log(z0brd)
        return jnp.exp(a1 + a2)
    burial = 7.0 * z0brd - snowh
    z0eff = W(burial <= 0.0007, z0s, burial / 7.0)
    return (1.0 - sncovr) * z0brd + sncovr * z0eff


def snowpack(esd, dtsec, snowh, sndens, tsnow, tsoil, snomlt=None, ua=False):
    c1, c2 = 0.01, 21.0
    esdc = esd * 100.0
    dthr = dtsec / 3600.0
    tsnowc = tsnow - 273.15
    tsoilc = tsoil - 273.15
    tavgc = 0.5 * (tsnowc + tsoilc)
    esdcx = W(esdc > 1.0e-2, esdc, 1.0e-2)
    bfac = dthr * c1 * jnp.exp(0.08 * tavgc - c2 * sndens)
    pexp = 0.0
    for j in (4, 3, 2, 1):
        pexp = (1.0 + pexp) * bfac * esdcx / float(j + 1)
    pexp = pexp + 1.0
    dsx = sndens * pexp
    dsx = W(dsx > 0.40, 0.40, dsx)
    dsx = W(dsx < 0.05, 0.05, dsx)
    sndens = dsx
    dw = 0.13 * dthr / 24.0
    if ua:
        snomltc = snomlt * 100.0
        dw = W(tsoilc >= 0.0, jnp.fmin(dw, 0.13 * snomltc / (esdcx + 0.13 * snomltc)), dw)
    sd2 = sndens * (1.0 - dw) + dw
    sd2 = W(sd2 >= 0.40, 0.40, sd2)
    sndens = W(tsnowc >= 0.0, sd2, sndens)
    snowhc = esdc / sndens
    return snowhc * 0.01, sndens


# ----------------------------------------------------------------------------------------------
def tdfcnd(smc, qz, smcmax, sh2o, bexp, psisat, soiltyp, opt_thcnd):
    satratio = smc / smcmax
    thkice, thkw, thko, thkqtz = 2.2, 0.57, 2.0, 7.7
    thks = jnp.power(thkqtz, qz) * jnp.power(thko, 1.0 - qz)
    xunfroz = sh2o / smc
    xu = xunfroz * smcmax
    thksat = jnp.power(thks, 1.0 - smcmax) * jnp.power(thkice, smcmax - xu) * jnp.power(thkw, xu)
    gammd = (1.0 - smcmax) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    akei = satratio
    akel = W(satratio > 0.1, jnp.log10(satratio) + 1.0, 0.0)
    ake = ((smc - sh2o) * akei + sh2o * akel) / smc
    df1 = ake * (thksat - thkdry) + thkdry
    if opt_thcnd == 0:        # Noah 3.6 formulation
        ake0 = W(sh2o + 0.0005 < smc, satratio, W(satratio > 0.1, jnp.log10(satratio) + 1.0, 0.0))
        return ake0 * (thksat - thkdry) + thkdry
    if opt_thcnd == 1:
        return df1
    psif = psisat * 100.0 * jnp.power(smcmax / smc, bexp)
    pf = jnp.log10(jnp.abs(psif))
    df2 = W(pf <= 5.1, 420.0 * jnp.exp(-(pf + 2.7)), 0.1744)
    return W((soiltyp != 4) & (soiltyp != 3), df1, df2)


def wdfcnd(smc, smcmax, bexp, dksat, dwsat, sicemax):
    factr1 = 0.05 / smcmax
    factr2 = smc / smcmax
    factr1 = jnp.fmin(factr1, factr2)
    expon = bexp + 2.0
    wdf = dwsat * jnp.power(factr2, expon)
    vkwgt = 1.0 / (1.0 + jnp.power(500.0 * sicemax, 3.0))
    wdf2 = vkwgt * wdf + (1.0 - vkwgt) * dwsat * jnp.power(factr1, expon)
    wdf = W(sicemax > 0.0, wdf2, wdf)
    expon = (2.0 * bexp) + 3.0
    wcnd = dksat * jnp.power(factr2, expon)
    return wdf, wcnd


def fac2mit(smcmax):
    fl = f32(0.90)
    fl = W(smcmax == f32(0.395), 0.59, fl)
    fl = W((smcmax == f32(0.434)) | (smcmax == f32(0.404)), 0.85, fl)
    fl = W((smcmax == f32(0.465)) | (smcmax == f32(0.406)), 0.86, fl)
    fl = W((smcmax == f32(0.476)) | (smcmax == f32(0.439)), 0.74, fl)
    fl = W((smcmax == f32(0.200)) | (smcmax == f32(0.464)), 0.80, fl)
    return fl


def frh2o(tkelv, smc, sh2o, smcmax, bexp, psis):
    ck, blim, error = 8.0, 5.5, 0.005
    hlice, gs = 3.335e5, 9.81
    bx = W(bexp > blim, blim, bexp)
    swl0 = smc - sh2o
    swl0 = W(swl0 > (smc - 0.02), smc - 0.02, swl0)
    swl0 = W(swl0 < 0.0, 0.0, swl0)

    def cond(s):
        swl, nlog, kcount = s
        return (nlog < 10) & (kcount == 0)

    def body(s):
        swl, nlog, kcount = s
        nlog = nlog + 1
        df = jnp.log((psis * gs / hlice) * jnp.power(1.0 + ck * swl, 2.0)
                     * jnp.power(smcmax / (smc - swl), bx)) - jnp.log(-(tkelv - T0) / tkelv)
        denom = 2.0 * ck / (1.0 + ck * swl) + bx / (smc - swl)
        swlk = swl - df / denom
        swlk = W(swlk > (smc - 0.02), smc - 0.02, swlk)
        swlk = W(swlk < 0.0, 0.0, swlk)
        dswl = jnp.abs(swlk - swl)
        kcount = W(dswl <= error, kcount + 1, kcount)
        return swlk, nlog, kcount

    swl, nlog, kcount = lax.while_loop(cond, body, (swl0, jnp.int32(0), jnp.int32(0)))
    free = smc - swl
    fk = jnp.power(((hlice / (gs * (-psis))) * ((tkelv - T0) / tkelv)), -1.0 / bx) * smcmax
    fk = W(fk < 0.02, 0.02, fk)
    free_fk = jnp.fmin(fk, smc)
    free = W(kcount == 0, free_fk, free)
    return W(tkelv > (T0 - 1.0e-3), smc, free)


def snksrc(tavg, smc, sh2o, zsoil, smcmax, psisat, bexp, dt, k, qtot):
    dh2o, hlice = 1.0e3, 3.335e5
    dz = -zsoil[0] if k == 0 else zsoil[k - 1] - zsoil[k]
    free = frh2o(tavg, smc, sh2o, smcmax, bexp, psisat)
    xh2o = sh2o + qtot * dt / (dh2o * hlice * dz)
    c1 = (xh2o < sh2o) & (xh2o < free)
    xh2o = W(c1, W(free > sh2o, sh2o, free), xh2o)
    c2 = (xh2o > sh2o) & (xh2o > free)
    xh2o = W(c2, W(free < sh2o, sh2o, free), xh2o)
    xh2o = W(xh2o < 0.0, 0.0, xh2o)
    xh2o = W(xh2o > smc, smc, xh2o)
    tsnsr = -dh2o * hlice * dz * (xh2o - sh2o) / dt
    return tsnsr, xh2o


def tbnd(tu, tb, zsoil, zbot, k):
    zup = 0.0 if k == 0 else zsoil[k - 1]
    zb = 2.0 * zbot - zsoil[k] if k == NS - 1 else zsoil[k + 1]
    return tu + (tb - tu) * (zup - zsoil[k]) / (zup - zb)


def tmpavg(tup, tm, tdn, zsoil, k):
    t0 = 2.7315e2
    dz = -zsoil[0] if k == 0 else zsoil[k - 1] - zsoil[k]
    dzh = dz * 0.5
    # tup < t0
    a_tm_lt_tdn_lt = (tup + 2.0 * tm + tdn) / 4.0
    x0 = (t0 - tm) * dzh / (tdn - tm)
    a_tm_lt_tdn_ge = 0.5 * (tup * dzh + tm * (dzh + x0) + t0 * (2.0 * dzh - x0)) / dz
    xup1 = (t0 - tup) * dzh / (tm - tup)
    xdn1 = dzh - (t0 - tm) * dzh / (tdn - tm)
    a_tm_ge_tdn_lt = 0.5 * (tup * xup1 + t0 * (2.0 * dz - xup1 - xdn1) + tdn * xdn1) / dz
    a_tm_ge_tdn_ge = 0.5 * (tup * xup1 + t0 * (2.0 * dz - xup1)) / dz
    r_tup_lt = W(tm < t0, W(tdn < t0, a_tm_lt_tdn_lt, a_tm_lt_tdn_ge),
                 W(tdn < t0, a_tm_ge_tdn_lt, a_tm_ge_tdn_ge))
    # tup >= t0
    xup2 = dzh - (t0 - tup) * dzh / (tm - tup)
    b_tm_lt_tdn_lt = 0.5 * (t0 * (dz - xup2) + tm * (dzh + xup2) + tdn * dzh) / dz
    xdn2 = (t0 - tm) * dzh / (tdn - tm)
    b_tm_lt_tdn_ge = 0.5 * (t0 * (2.0 * dz - xup2 - xdn2) + tm * (xup2 + xdn2)) / dz
    xdn3 = dzh - (t0 - tm) * dzh / (tdn - tm)
    b_tm_ge_tdn_lt = (t0 * (dz - xdn3) + 0.5 * (t0 + tdn) * xdn3) / dz
    b_tm_ge_tdn_ge = (tup + 2.0 * tm + tdn) / 4.0
    r_tup_ge = W(tm < t0, W(tdn < t0, b_tm_lt_tdn_lt, b_tm_lt_tdn_ge),
                 W(tdn < t0, b_tm_ge_tdn_lt, b_tm_ge_tdn_ge))
    return W(tup < t0, r_tup_lt, r_tup_ge)


def rosr12(a, b, c, d):
    n = NS
    c = list(c)
    c[n - 1] = f32(0.0)
    p = [None] * n
    delta = [None] * n
    p[0] = -c[0] / b[0]
    delta[0] = d[0] / b[0]
    for k in range(1, n):
        p[k] = -c[k] * (1.0 / (b[k] + a[k] * p[k - 1]))
        delta[k] = (d[k] - a[k] * delta[k - 1]) * (1.0 / (b[k] + a[k] * p[k - 1]))
    p[n - 1] = delta[n - 1]
    for kk in range(n - 2, -1, -1):
        p[kk] = p[kk] * p[kk + 1] + delta[kk]
    return p, delta


# ----------------------------------------------------------------------------------------------
def hrt(stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot, psisat, sh2o, dt, bexp, soiltyp,
        opt_thcnd, quartz, csoil, df1, vegtyp, isurban):
    cair, cice, ch2o = 1004.0, 2.106e6, 4.2e6
    csoil_loc = W(vegtyp == isurban, 3.0e6, csoil)
    sh2o = list(sh2o)
    rhsts = [None] * NS
    ai = [None] * NS
    bi = [None] * NS
    ci = [None] * NS
    hcpct = sh2o[0] * ch2o + (1.0 - smcmax) * csoil_loc + (smcmax - smc[0]) * cair \
        + (smc[0] - sh2o[0]) * cice
    ddz = 1.0 / (-0.5 * zsoil[1])
    ai[0] = f32(0.0)
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct)
    bi[0] = -ci[0] + df1 / (0.5 * zsoil[0] * zsoil[0] * hcpct * zz1)
    dtsdz = (stc[0] - stc[1]) / (-0.5 * zsoil[1])
    ssoil = df1 * (stc[0] - yy) / (0.5 * zsoil[0] * zz1)
    denom = zsoil[0] * hcpct
    rhsts[0] = (df1 * dtsdz - ssoil) / denom
    qtot = -1.0 * rhsts[0] * denom
    sice = smc[0] - sh2o[0]
    tsurf = (yy + (zz1 - 1) * stc[0]) / zz1
    tbk = tbnd(stc[0], stc[1], zsoil, zbot, 0)
    cond = (sice > 0.0) | (stc[0] < T0) | (tsurf < T0) | (tbk < T0)
    tavg = tmpavg(tsurf, stc[0], tbk, zsoil, 0)
    tsnsr, xh = snksrc(tavg, smc[0], sh2o[0], zsoil, smcmax, psisat, bexp, dt, 0, qtot)
    rhsts[0] = W(cond, rhsts[0] - tsnsr / denom, rhsts[0])
    sh2o[0] = W(cond, xh, sh2o[0])
    ddz2 = f32(0.0)
    df1k = df1
    for k in range(1, NS):
        hcpct = sh2o[k] * ch2o + (1.0 - smcmax) * csoil_loc + (smcmax - smc[k]) * cair \
            + (smc[k] - sh2o[k]) * cice
        df1n = tdfcnd(smc[k], quartz, smcmax, sh2o[k], bexp, psisat, soiltyp, opt_thcnd)
        df1n = W(vegtyp == isurban, 3.24, df1n)
        if k != NS - 1:
            denom = 0.5 * (zsoil[k - 1] - zsoil[k + 1])
            dtsdz2 = (stc[k] - stc[k + 1]) / denom
            ddz2 = 2.0 / (zsoil[k - 1] - zsoil[k + 1])
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct)
            tbk1 = tbnd(stc[k], stc[k + 1], zsoil, zbot, k)
        else:
            denom = 0.5 * (zsoil[k - 1] + zsoil[k]) - zbot
            dtsdz2 = (stc[k] - tbot) / denom
            ci[k] = f32(0.0)
            tbk1 = tbnd(stc[k], tbot, zsoil, zbot, k)
        denom = (zsoil[k] - zsoil[k - 1]) * hcpct
        rhsts[k] = (df1n * dtsdz2 - df1k * dtsdz) / denom
        qtot = -1.0 * denom * rhsts[k]
        sice = smc[k] - sh2o[k]
        tavg = tmpavg(tbk, stc[k], tbk1, zsoil, k)
        cond = (sice > 0.0) | (stc[k] < T0) | (tbk < T0) | (tbk1 < T0)
        tsnsr, xh = snksrc(tavg, smc[k], sh2o[k], zsoil, smcmax, psisat, bexp, dt, k, qtot)
        rhsts[k] = W(cond, rhsts[k] - tsnsr / denom, rhsts[k])
        sh2o[k] = W(cond, xh, sh2o[k])
        ai[k] = -df1k * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct)
        bi[k] = -(ai[k] + ci[k])
        tbk = tbk1
        df1k = df1n
        dtsdz = dtsdz2
        ddz = ddz2
    return rhsts, ai, bi, ci, sh2o


def hstep(stcin, rhsts, dt, ai, bi, ci):
    rhsts = [r * dt for r in rhsts]
    ai = [a * dt for a in ai]
    bi = [1.0 + b * dt for b in bi]
    ci = [c * dt for c in ci]
    p, _ = rosr12(ai, bi, ci, rhsts)
    return [stcin[k] + p[k] for k in range(NS)]


def shflx(stc, smc, smcmax, t1, dt, yy, zz1, zsoil, tbot, zbot, psisat, sh2o, bexp, df1,
          quartz, csoil, vegtyp, isurban, soiltyp, opt_thcnd):
    rhsts, ai, bi, ci, sh2o = hrt(stc, smc, smcmax, zsoil, yy, zz1, tbot, zbot, psisat, sh2o, dt,
                                  bexp, soiltyp, opt_thcnd, quartz, csoil, df1, vegtyp, isurban)
    stc = hstep(stc, rhsts, dt, ai, bi, ci)
    t1 = (yy + (zz1 - 1.0) * stc[0]) / zz1
    ssoil = df1 * (stc[0] - t1) / (0.5 * zsoil[0])
    return ssoil, stc, t1, sh2o


# ----------------------------------------------------------------------------------------------
def srt(edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat, dksat, smcmax, bexp, dt, smcwlt, slope,
        kdt, frzx, sice):
    cvfrz = 3
    sicemax = W(sice[0] > 0.0, sice[0], 0.0)
    for ks in range(1, NS):
        sicemax = W(sice[ks] > sicemax, sice[ks], sicemax)
    pddum = pcpdrp
    runoff1 = f32(0.0)
    dt1 = dt / 86400.0
    smcav = smcmax - smcwlt
    dmax = [None] * NS
    dmax[0] = -zsoil[0] * smcav
    dice = -zsoil[0] * sice[0]
    dmax[0] = dmax[0] * (1.0 - (sh2oa[0] + sice[0] - smcwlt) / smcav)
    dd = dmax[0]
    for ks in range(1, NS):
        dice = dice + (zsoil[ks - 1] - zsoil[ks]) * sice[ks]
        dmax[ks] = (zsoil[ks - 1] - zsoil[ks]) * smcav
        dmax[ks] = dmax[ks] * (1.0 - (sh2oa[ks] + sice[ks] - smcwlt) / smcav)
        dd = dd + dmax[ks]
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
    infmax = jnp.fmin(infmax, px / dt)
    blk = pcpdrp != 0.0
    over = blk & (pcpdrp > infmax)
    runoff1 = W(over, pcpdrp - infmax, runoff1)
    pddum = W(over, infmax, pddum)
    # soil water flux terms
    ddz = 1.0 / (-0.5 * zsoil[1])
    ai = [None] * NS
    bi = [None] * NS
    ci = [None] * NS
    rhstt = [None] * NS
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


def sstep(sh2oin, cmc, rhstt, rhsct, dt, smcmax, cmcmax, zsoil, sice, ai, bi, ci):
    rhstt = [r * dt for r in rhstt]
    ai = [a * dt for a in ai]
    bi = [1.0 + b * dt for b in bi]
    ci = [c * dt for c in ci]
    p, _ = rosr12(ai, bi, ci, rhstt)
    wplus = f32(0.0)
    sh2oout = [None] * NS
    smc = [None] * NS
    for k in range(NS):
        ddz = -zsoil[0] if k == 0 else zsoil[k - 1] - zsoil[k]
        sh2oout[k] = sh2oin[k] + p[k] + wplus / ddz
        stot = sh2oout[k] + sice[k]
        ddz2 = -zsoil[0] if k == 0 else -zsoil[k] + zsoil[k - 1]
        wplus = W(stot > smcmax, (stot - smcmax) * ddz2, 0.0)
        smc[k] = jnp.fmax(jnp.fmin(stot, smcmax), 0.02)
        sh2oout[k] = jnp.fmax(smc[k] - sice[k], 0.0)
    runoff3 = wplus
    cmc = cmc + dt * rhsct
    cmc = W(cmc < 1.0e-20, 0.0, cmc)
    cmc = jnp.fmin(cmc, cmcmax)
    return sh2oout, smc, cmc, runoff3


def smflx(smc, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzx, smcmax, bexp, smcwlt, dksat,
          dwsat, shdfac, cmcmax, edir, ec, et):
    rhsct = shdfac * prcp1 - ec
    trhsct = dt * rhsct
    excess = cmc + trhsct
    drip = W(excess > cmcmax, excess - cmcmax, 0.0)
    pcpdrp = (1.0 - shdfac) * prcp1 + drip / dt
    sice = [smc[i] - sh2o[i] for i in range(NS)]
    fac2 = f32(0.0)
    for i in range(NS):
        fac2 = jnp.fmax(fac2, sh2o[i] / smcmax)
    flimit = fac2mit(smcmax)
    cond = ((pcpdrp * dt) > (0.0001 * 1000.0 * (-zsoil[0]) * smcmax)) | (fac2 > flimit)
    args = (edir, et)
    # first SRT/SSTEP pass with SH2O (used by both paths)
    rhstt, ai, bi, ci, ro1, ro2 = srt(edir, et, sh2o, sh2o, pcpdrp, zsoil, dwsat, dksat, smcmax,
                                      bexp, dt, smcwlt, slope, kdt, frzx, sice)
    # else path: single pass
    sh2o_b, smc_b, cmc_b, ro3_b = sstep(sh2o, cmc, rhstt, rhsct, dt, smcmax, cmcmax, zsoil, sice,
                                        ai, bi, ci)
    # if path: first-guess step then midpoint SRT
    sh2ofg, _, _, _ = sstep(sh2o, f32(0.0), rhstt, rhsct, dt, smcmax, cmcmax, zsoil, sice,
                            ai, bi, ci)
    sh2oa = [(sh2o[k] + sh2ofg[k]) * 0.5 for k in range(NS)]
    rhstt2, ai2, bi2, ci2, ro1_a, ro2_a = srt(edir, et, sh2o, sh2oa, pcpdrp, zsoil, dwsat, dksat,
                                              smcmax, bexp, dt, smcwlt, slope, kdt, frzx, sice)
    sh2o_a, smc_a, cmc_a, ro3_a = sstep(sh2o, cmc, rhstt2, rhsct, dt, smcmax, cmcmax, zsoil, sice,
                                        ai2, bi2, ci2)
    pick = lambda a, b: W(cond, a, b)
    smc_o = [pick(smc_a[k], smc_b[k]) for k in range(NS)]
    sh2o_o = [pick(sh2o_a[k], sh2o_b[k]) for k in range(NS)]
    return (smc_o, sh2o_o, pick(cmc_a, cmc_b), drip, pick(ro1_a, ro1), pick(ro2_a, ro2),
            pick(ro3_a, ro3_b))


# ----------------------------------------------------------------------------------------------
def devap(etp1, smc1, shdfac, smcmax, smcdry, fxexp):
    sratio = (smc1 - smcdry) / (smcmax - smcdry)
    fx = jnp.power(jnp.fmax(sratio, 1.0e-30), fxexp)
    fx = W(sratio > 0.0, jnp.fmax(jnp.fmin(fx, 1.0), 0.0), 0.0)
    return fx * (1.0 - shdfac) * etp1


def transp(etp1, sh2o, cmc, zsoil, shdfac, smcwlt, cmcmax, pc, cfactr, smcref, nroot, rtdis):
    etp1a = W(cmc != 0.0, shdfac * pc * etp1 * (1.0 - jnp.power(cmc / cmcmax, cfactr)),
              shdfac * pc * etp1)
    gx = []
    sgx = f32(0.0)
    for i in range(NS):
        g = (sh2o[i] - smcwlt) / (smcref - smcwlt)
        g = jnp.fmax(jnp.fmin(g, 1.0), 0.0)
        g = W(i < nroot, g, 0.0)
        gx.append(g)
        sgx = sgx + g
    sgx = sgx / nroot.astype(f32)
    denom = f32(0.0)
    for i in range(NS):
        rtx = rtdis[i] + gx[i] - sgx
        g = gx[i] * jnp.fmax(rtx, 0.0)
        g = W(i < nroot, g, 0.0)
        gx[i] = g
        denom = denom + g
    denom = W(denom <= 0.0, 1.0, denom)
    return [W(i < nroot, etp1a * gx[i] / denom, 0.0) for i in range(NS)]


def evapo(smc, cmc, etp1, dt, zsoil, sh2o, smcmax, pc, smcwlt, smcref, shdfac, cmcmax, smcdry,
          cfactr, nroot, rtdis, fxexp):
    pos = etp1 > 0.0
    edir = W(pos & (shdfac < 1.0), devap(etp1, smc[0], shdfac, smcmax, smcdry, fxexp), 0.0)
    et = transp(etp1, sh2o, cmc, zsoil, shdfac, smcwlt, cmcmax, pc, cfactr, smcref, nroot, rtdis)
    tr_on = pos & (shdfac > 0.0)
    et = [W(tr_on, e, 0.0) for e in et]
    ett = _lsum(et)
    ec = W(cmc > 0.0, shdfac * jnp.power(jnp.fmax(cmc, 1.0e-30) / cmcmax, cfactr) * etp1, 0.0)
    ec = jnp.fmin(cmc / dt, ec)
    ec = W(tr_on, ec, 0.0)
    eta1 = edir + ett + ec
    return eta1, edir, ec, et, ett


def canres(solar, ch, sfctmp, q2, sfcprs, sh2o, zsoil, smcwlt, smcref, rsmin, nroot, q2sat,
           dqsdt2, topt, rsmax, rgl, hs, xlai, emissi):
    slv = 2.501e6
    ff = 0.55 * 2.0 * solar / (rgl * xlai)
    rcs = jnp.fmax((ff + rsmin / rsmax) / (1.0 + ff), 0.0001)
    rct = jnp.fmax(1.0 - 0.0016 * jnp.power(topt - sfctmp, 2.0), 0.0001)
    rcq = jnp.fmax(1.0 / (1.0 + hs * (q2sat - q2)), 0.01)
    zs_root = jnp.stack(zsoil)[nroot - 1]
    part = []
    for k in range(NS):
        gx = (sh2o[k] - smcwlt) / (smcref - smcwlt)
        gx = W(gx > 1.0, 1.0, gx)
        gx = W(gx < 0.0, 0.0, gx)
        if k == 0:
            p = (zsoil[0] / zs_root) * gx
        else:
            p = ((zsoil[k] - zsoil[k - 1]) / zs_root) * gx
        part.append(W(k < nroot, p, 0.0))
    rcsoil = f32(0.0)
    for k in range(NS):
        rcsoil = rcsoil + part[k]
    rcsoil = jnp.fmax(rcsoil, 0.0001)
    rc = rsmin / (xlai * rcs * rct * rcq * rcsoil)
    rr = (4.0 * emissi * SIGMA * RD / CP_MOD) * jnp.power(sfctmp, 4.0) / (sfcprs * ch) + 1.0
    delta = (slv / CP_MOD) * dqsdt2
    pc = (rr + delta) / (rr * (1.0 + rc * ch) + delta)
    return rc, pc, rcs, rct, rcq, rcsoil


def penman(sfctmp, sfcprs, ch, t2v, th2, prcp, fdown, ssoil, q2, q2sat, snowng, frzgra, dqsdt2,
           emissi, sncovr, aoasis, ua=False, sneqv=None, albedo=None, soldn=None, fvb=None, gama=None, stc1=None):
    elcp, lsubc, cp = 2.4888e3, 2.501e6, 1004.6
    algdsn, alvgsn = 0.5, 0.13
    elcp1 = (1.0 - sncovr) * elcp + sncovr * elcp * LSUBS / lsubc
    lvs = (1.0 - sncovr) * lsubc + sncovr * LSUBS
    flx2 = f32(0.0)
    delta = elcp1 * dqsdt2
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    rr = emissi * t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    rho = sfcprs / (RD * t2v)
    rch = rho * cp * ch
    rr = W(snowng, rr + CPICE * prcp / rch, W(prcp > 0.0, rr + CPH2O * prcp / rch, rr))
    fnet = fdown - emissi * SIGMA * t24 - ssoil
    flx4 = f32(0.0)
    fnetn = f32(0.0)
    if ua:
        c1 = (sneqv > 0.0) & (fnet > 0.0) & (soldn > 0.0)
        totabs = (1.0 - albedo) * soldn * fvb
        ucabs = jnp.fmin(totabs, ((1.0 - algdsn) * (1.0 - alvgsn) * soldn * gama) * fvb)
        flx4 = W(c1, jnp.fmin(totabs - ucabs, jnp.fmin(250.0, 0.5 * (1.0 - albedo) * soldn)), 0.0)
        signck = (stc1 - 273.15) * (sfctmp - 273.15)
        c2 = (flx4 > 0.0) & ((signck <= 0.0) | (stc1 < 273.15))
        ge = fnet >= flx4
        fnetn = W(c2, W(ge, fnet - flx4, 0.0), 0.0)
        flx4 = W(c2, W(ge, flx4, fnet), 0.0)
    flx2 = W(frzgra, -LSUBF * prcp, flx2)
    fnet = W(frzgra, fnet - flx2, fnet)
    if ua:
        fnetn = W(frzgra, fnetn - flx2, fnetn)
    rad = fnet / rch + th2 - sfctmp
    a = elcp1 * (q2sat - q2)
    epsca = (a * rr + rad * delta) / (delta + rr)
    epsca = W(epsca > 0.0, epsca * aoasis, epsca)
    etp = epsca * rch / lvs
    etpn = f32(0.0)
    if ua:
        radn = fnetn / rch + th2 - sfctmp
        epscan = (a * rr + radn * delta) / (delta + rr)
        etpn = epscan * rch / lvs
    return etp, rch, epsca, rr, t24, flx2, etpn, flx4


# ----------------------------------------------------------------------------------------------
def nopac(etp, prcp, smc, smcmax, smcwlt, smcref, smcdry, cmc, cmcmax, dt, shdfac, sbeta, q2,
          t1, sfctmp, t24, th2, fdown, f1, emissi, stc, epsca, bexp, pc, rch, rr, cfactr, sh2o,
          slope, kdt, frzx, psisat, zsoil, dksat, dwsat, tbot, zbot, nroot, rtdis, quartz, fxexp,
          csoil, vegtyp, isurban, soiltyp, opt_thcnd):
    prcp1 = prcp * 0.001
    etp1 = etp * 0.001
    pos = etp > 0.0
    eta1, edir1, ec1, et1, ett1 = evapo(smc, cmc, etp1, dt, zsoil, sh2o, smcmax, pc, smcwlt,
                                        smcref, shdfac, cmcmax, smcdry, cfactr, nroot, rtdis,
                                        fxexp)
    z = f32(0.0)
    edir1 = W(pos, edir1, z)
    ec1 = W(pos, ec1, z)
    et1 = [W(pos, e, z) for e in et1]
    ett1 = W(pos, ett1, z)
    dew = W(pos, z, -etp1)
    prcp1 = W(pos, prcp1, prcp1 + dew)
    smc, sh2o, cmc, drip, ro1, ro2, ro3 = smflx(smc, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzx,
                                                smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
                                                cmcmax, edir1, ec1, et1)
    eta = W(pos, eta1 * 1000.0, etp)
    beta = W(etp <= 0.0, W(etp < 0.0, 1.0, 0.0), eta / etp)
    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    et = [e * 1000.0 for e in et1]
    ett = ett1 * 1000.0
    df1 = tdfcnd(smc[0], quartz, smcmax, sh2o[0], bexp, psisat, soiltyp, opt_thcnd)
    df1 = W(vegtyp == isurban, 3.24, df1)
    df1 = df1 * jnp.exp(sbeta * shdfac)
    yynum = fdown - emissi * SIGMA * t24
    yy = sfctmp + (yynum / rch + th2 - sfctmp - beta * epsca) / rr
    tsoil = yy
    zz1 = df1 / (-0.5 * zsoil[0] * rch * rr) + 1.0
    ssoil, stc, t1, sh2o = shflx(stc, smc, smcmax, t1, dt, yy, zz1, zsoil, tbot, zbot, psisat,
                                 sh2o, bexp, df1, quartz, csoil, vegtyp, isurban, soiltyp,
                                 opt_thcnd)
    flx1 = CPH2O * prcp * (t1 - sfctmp)
    flx3 = f32(0.0)
    return dict(eta=eta, beta=beta, edir=edir, ec=ec, et=et, ett=ett, drip=drip, ro1=ro1,
                ro2=ro2, ro3=ro3, ssoil=ssoil, tsoil=tsoil, t1=t1, stc=stc, smc=smc, sh2o=sh2o,
                cmc=cmc, dew=dew, flx1=flx1, flx3=flx3, esnow=f32(0.0), etns=f32(0.0),
                snomlt=f32(0.0), eta_kin=eta, etp=etp)


def snopac(etp, prcp, prcpf, snowng, smc, smcmax, smcwlt, smcref, smcdry, cmc, cmcmax, dt, sbeta,
           df1, q2, t1, sfctmp, t24, th2, fdown, f1, stc, epsca, sfcprs, bexp, pc, rch, rr, cfactr,
           sncovr, esd, sndens, snowh, sh2o, slope, kdt, frzx, psisat, zsoil, dwsat, dksat, tbot,
           zbot, shdfac, nroot, rtdis, quartz, fxexp, csoil, emissi, ribb, soldn, isurban, vegtyp,
           soiltyp, opt_thcnd, flx2, ua=False, etpn=None, flx4=None, v36=False):
    esdmin, snoexp = 1.0e-6, 2.0
    z = f32(0.0)
    prcp1 = prcpf * 0.001
    # --- potential evaporation branches
    neg = etp <= 0.0
    cond_r = (ribb >= 0.1) & (fdown > 150.0)
    etp_adj = (jnp.fmin(etp * (1.0 - ribb), 0.0) * sncovr / 0.980 + etp * (0.980 - sncovr)) / 0.980
    etp = W(neg & cond_r, etp_adj, etp)
    beta0 = W(neg & (etp == 0.0), 0.0, 1.0)
    etp1 = etp * 0.001
    dew = W(neg, -etp1, z)
    esnow2_neg = etp1 * dt
    etanrg_neg = etp * ((1.0 - sncovr) * LSUBC + sncovr * LSUBS)
    # positive-ETP: evaporation from the snow-free part
    run_ev = (~neg) & (sncovr < 1.0)
    etns1, edir1, ec1, et1, ett1 = None, None, None, None, None
    eta1, edir1, ec1, et1, ett1 = evapo(smc, cmc, etp1, dt, zsoil, sh2o, smcmax, pc, smcwlt, smcref,
                                        shdfac, cmcmax, smcdry, cfactr, nroot, rtdis, fxexp)
    etns1 = eta1
    f = 1.0 - sncovr
    edir1 = W(run_ev, edir1 * f, z)
    ec1 = W(run_ev, ec1 * f, z)
    et1 = [W(run_ev, e * f, z) for e in et1]
    ett1 = W(run_ev, ett1 * f, z)
    etns1 = W(run_ev, etns1 * f, z)
    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    et = [e * 1000.0 for e in et1]
    ett = ett1 * 1000.0
    etns = etns1 * 1000.0
    esnow_pos = (etpn if ua else etp) * sncovr
    esnow = W(neg, z, esnow_pos)
    esnow1 = esnow * 0.001
    esnow2 = W(neg, esnow2_neg, esnow1 * dt)
    etanrg = W(neg, etanrg_neg, esnow * LSUBS + etns * LSUBC)
    # --- precipitation heat flux
    flx1 = W(snowng, CPICE * prcp * (t1 - sfctmp), W(prcp > 0.0, CPH2O * prcp * (t1 - sfctmp), z))
    dsoil = -(0.5 * zsoil[0])
    dtot = snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    t12a = ((fdown - flx1 - flx2 - emissi * SIGMA * t24) / rch + th2 - sfctmp - etanrg / rch) / rr
    t12b = df1 * stc[0] / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom
    cold = t12 <= TFREEZ
    # cold branch
    t1_c = t12
    ssoil_c = df1 * (t1_c - stc[0]) / dtot
    esd_c = jnp.fmax(0.0, esd - esnow2)
    # warm branch
    sn2 = jnp.power(sncovr, snoexp) if v36 else jnp.fmax(0.01, jnp.power(sncovr, snoexp))
    t1_w = TFREEZ * sn2 + t12 * (1.0 - sn2)
    ssoil_w = df1 * (t1_w - stc[0]) / dtot
    gone = (esd - esnow2) <= esdmin
    esd_w1 = esd - esnow2
    seh = rch * (t1_w - th2)
    t14 = t1_w * t1_w
    t14 = t14 * t14
    flx3 = fdown - flx1 - flx2 - emissi * SIGMA * t14 - ssoil_w - seh - etanrg
    flx3 = W(flx3 <= 0.0, 0.0, flx3)
    if ua:
        c4 = (flx4 > 0.0) & (flx3 > 0.0)
        ge4 = flx3 >= flx4
        flx4_w = W(c4, W(ge4, flx4, flx3), 0.0)
        flx3 = W(c4, W(ge4, flx3 - flx4, 0.0), flx3)
    else:
        flx4_w = f32(0.0)
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
    flx3_w = W(gone, 0.0, flx3)
    flx4_w = W(gone, 0.0, flx4_w)
    t1 = W(cold, t1_c, t1_w)
    ssoil = W(cold, ssoil_c, ssoil_w)
    esd = W(cold, esd_c, esd_w)
    flx3 = W(cold, 0.0, flx3_w)
    ex = W(cold, 0.0, ex_w)
    flx4_o = W(cold, 0.0, flx4_w) if ua else f32(0.0)
    snomlt = W(cold, 0.0, snomlt_w)
    beta = W(cold, beta0, 1.0)
    prcp1 = W(cold, prcp1, prcp1 + ex)
    smc, sh2o, cmc, drip, ro1, ro2, ro3 = smflx(smc, cmc, dt, prcp1, zsoil, sh2o, slope, kdt, frzx,
                                                smcmax, bexp, smcwlt, dksat, dwsat, shdfac,
                                                cmcmax, edir1, ec1, et1)
    zz1 = f32(1.0)
    yy = stc[0] - 0.5 * ssoil * zsoil[0] * zz1 / df1
    tsoil = yy
    _, stc, t11, sh2o = shflx(stc, smc, smcmax, t1, dt, yy, zz1, zsoil, tbot, zbot, psisat, sh2o,
                              bexp, df1, quartz, csoil, vegtyp, isurban, soiltyp, opt_thcnd)
    has = esd > 0.0
    snowh_n, sndens_n = snowpack(esd, dt, snowh, sndens, t1, yy, snomlt=snomlt, ua=ua)
    snowh = W(has, snowh_n, 0.0)
    sndens = W(has, sndens_n, 0.0)
    sncovr = W(has, sncovr, 0.0)
    esd = W(has, esd, 0.0)
    return dict(beta=beta, edir=edir, ec=ec, et=et, ett=ett, drip=drip, ro1=ro1, ro2=ro2, ro3=ro3,
                ssoil=ssoil, tsoil=tsoil, t1=t1, stc=stc, smc=smc, sh2o=sh2o, cmc=cmc, dew=dew,
                flx1=flx1, flx3=flx3, esnow=esnow, etns=etns, snomlt=snomlt, esd=esd,
                snowh=snowh, sndens=sndens, sncovr=sncovr, etp=etp, flx4=flx4_o,
                eta_kin=(esnow + etns if v36 else esnow + etns - 1000.0 * dew))


# ----------------------------------------------------------------------------------------------
def sfcdif_off(zlm, zlm_wind, z0, thz0, thlm, sfcspd, czil, akms, akhs):
    """SFCDIF_off with ILECH=0, IZ0TLND=0. Returns updated (akms, akhs) and ustar."""
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
    rdz = 1.0 / zlm_wind
    cxch = excm * rdz
    dthv = thlm - thz0
    du2 = jnp.fmax(sfcspd * sfcspd, epsu2)
    btgh = btg * hpbl

    def wstar(akhs):
        x = btgh * akhs * dthv
        return W(x != 0.0, wwst2 * jnp.power(jnp.abs(jnp.where(x != 0.0, x, 1.0)), 2.0 / 3.0), 0.0)

    wstar2 = wstar(akhs)
    ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
    zt = jnp.exp(zilfc * jnp.sqrt(ustar * z0)) * z0
    zslu = zlm_wind + zu
    zslt = zlm + zt
    rlogu = jnp.log(zslu / zu)
    rlogt = jnp.log(zslt / zt)
    rlmo = elfc * akhs * dthv / (ustar * ustar * ustar)
    for _ in range(5):
        zetalt = jnp.fmax(zslt * rlmo, ztmin)
        rlmo = zetalt / zslt
        zetalu = zslu * rlmo
        zetau = zu * rlmo
        zetat = zt * rlmo
        unst = rlmo < 0.0
        # unstable
        xlu = jnp.sqrt(jnp.sqrt(jnp.where(unst, 1.0 - 16.0 * zetalu, 1.0)))
        xlt = jnp.sqrt(jnp.sqrt(jnp.where(unst, 1.0 - 16.0 * zetalt, 1.0)))
        xu = jnp.sqrt(jnp.sqrt(jnp.where(unst, 1.0 - 16.0 * zetau, 1.0)))
        xt = jnp.sqrt(jnp.sqrt(jnp.where(unst, 1.0 - 16.0 * zetat, 1.0)))
        pspmu = lambda xx: (-2.0 * jnp.log((xx + 1.0) * 0.5) - jnp.log((xx * xx + 1.0) * 0.5)
                            + 2.0 * jnp.arctan(xx) - pihf)
        psphu = lambda xx: -2.0 * jnp.log((xx * xx + 1.0) * 0.5)
        psmz_u = pspmu(xu)
        simm_u = pspmu(xlu) - psmz_u + rlogu
        pshz_u = psphu(xt)
        simh_u = psphu(xlt) - pshz_u + rlogt
        # stable
        zetalu_s = jnp.fmin(zetalu, ztmax)
        zetalt_s = jnp.fmin(zetalt, ztmax)
        psmz_s = 5.0 * zetau
        simm_s = 5.0 * zetalu_s - psmz_s + rlogu
        pshz_s = 5.0 * zetat
        simh_s = 5.0 * zetalt_s - pshz_s + rlogt
        simm = W(unst, simm_u, simm_s)
        simh = W(unst, simh_u, simh_s)
        ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
        zt = jnp.exp(zilfc * jnp.sqrt(ustar * z0)) * z0
        zslt = zlm + zt
        rlogt = jnp.log(zslt / zt)
        ustark = ustar * vkrm
        akms = jnp.fmax(ustark / simm, cxch)
        akhs = jnp.fmax(ustark / simh, cxch)
        wstar2 = wstar(akhs)
        rlmn = elfc * akhs * dthv / (ustar * ustar * ustar)
        rlmo = rlmo * wold + rlmn * wnew
    return akms, akhs, ustar


# ----------------------------------------------------------------------------------------------
# SFLX_GLACIAL (module_sf_noah39lsm_glacial): land-ice tiles (xice = -1)
def sflx_glacial(forc, par, st, dt, sldpth):
    """One step for a glacier column. forc: lwdn solnet sfcprs prcp sfctmp q2 th2 q2sat dqsdt2 ffrozp ch ribb;
    par: alb snoalb tbot z0brd embrd lvcoef; st: t1 stc[4] snowh sneqv emissi snotime1 sncovr."""
    lwdn, solnet, sfcprs, prcp = forc["lwdn"], forc["solnet"], forc["sfcprs"], forc["prcp"]
    sfctmp, q2, th2, q2sat, dqsdt2 = forc["sfctmp"], forc["q2"], forc["th2"], forc["q2sat"], forc["dqsdt2"]
    ffrozp, ch, ribb = forc["ffrozp"], forc["ch"], forc["ribb"]
    alb, snoalb, tbot, z0brd, embrd, lvcoef = par["alb"], par["snoalb"], par["tbot"], par["z0brd"], par["embrd"], par["lvcoef"]
    t1, snowh, sneqv, snotime1 = st["t1"], st["snowh"], st["sneqv"], st["snotime1"]
    stc = [st["stc"][k] for k in range(NS)]
    zsoil = [-sldpth[0]]
    for k in range(1, NS):
        zsoil.append(-sldpth[k] + zsoil[k - 1])
    low = sneqv < 0.10
    sneqv = W(low, 0.10, sneqv)
    snowh = W(low, 0.50, snowh)
    sndens = sneqv / snowh
    sncond = csnow(sndens)
    snowng = (prcp > 0.0) & (ffrozp > 0.5)
    frzgra = (prcp > 0.0) & (~(ffrozp > 0.5)) & (t1 <= TFREEZ)
    addsn = snowng | frzgra
    sn_new = prcp * dt * 0.001
    sneqv_n = sneqv + sn_new
    snowh_n, sndens_n = snow_new(sfctmp, sn_new, snowh, sndens)
    sncovr0 = st["sncovr"]
    c1 = sncovr0 > 0.99
    sndens_n = W(c1 & (stc[0] < (TFREEZ - 5.0)), 0.2, sndens_n)
    sndens_n = W(c1 & snowng & (t1 < 273.0) & (sfctmp < 273.0), 0.2, sndens_n)
    sncond_n = csnow(sndens_n)
    sneqv = W(addsn, sneqv_n, sneqv)
    snowh = W(addsn, snowh_n, snowh)
    sndens = W(addsn, sndens_n, sndens)
    sncond = W(addsn, sncond_n, sncond)
    prcpf = W(addsn, 0.0, prcp)
    sncovr = f32(1.0)
    # ALCALC (glacial): albedo = snow albedo, emissivity = snow emissivity
    snoalb1 = snoalb + lvcoef * (0.85 - snoalb)
    snotime1 = W(snowng, 0.0, snotime1 + dt)
    snoalb2 = W(snowng, snoalb1, snoalb1 * jnp.power(0.94, jnp.power(snotime1 / 86400.0, 0.58)))
    snoalb2 = jnp.fmax(snoalb2, alb)
    albedo = alb + (snoalb2 - alb)
    albedo = W(albedo > snoalb2, snoalb2, albedo)
    emissi = embrd + (EMISSI_S - embrd)
    df1 = sncond
    dsoil = -(0.5 * zsoil[0])
    dtot = snowh + dsoil
    frcsno = snowh / dtot
    frcsoi = dsoil / dtot
    df1 = frcsno * sncond + frcsoi * df1
    dtot = W(dtot > 2.0 * dsoil, 2.0 * dsoil, dtot)
    ssoil = df1 * (t1 - stc[0]) / dtot
    burial = 7.0 * z0brd - snowh
    z0 = W(burial <= 0.0007, 0.001, burial / 7.0)
    fdown = solnet + lwdn
    t2v = sfctmp * (1.0 + 0.61 * q2)
    rho = sfcprs / (RD * t2v)
    rch = rho * 1004.6 * ch
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    # PENMAN (glacial)
    elcp, lsubc = 2.4888e3, 2.501e6
    warm = t1 > 273.15
    elcp1 = W(warm, elcp, elcp * LSUBS / lsubc)
    lvs = W(warm, lsubc, LSUBS)
    delta = elcp1 * dqsdt2
    a = elcp1 * (q2sat - q2)
    rr = emissi * t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    rr = W(snowng, rr + CPICE * prcp / rch, W(prcp > 0.0, rr + CPH2O * prcp / rch, rr))
    flx2 = W(frzgra, -LSUBF * prcp, 0.0)
    fnet = fdown - (emissi * SIGMA * t24) - ssoil - flx2
    rad = fnet / rch + th2 - sfctmp
    epsca = (a * rr + rad * delta) / (delta + rr)
    etp = epsca * rch / lvs
    # SNOPAC (glacial)
    esdmin, snoexp = 1.0e-6, 2.0
    z = f32(0.0)
    prcp1 = prcpf * 0.001
    neg = etp <= 0.0
    etp_adj = (jnp.fmin(etp * (1.0 - ribb), 0.0) / 0.980 + etp * (0.980 - 1.0)) / 0.980
    etp = W(neg & (ribb >= 0.1) & (fdown > 150.0), etp_adj, etp)
    etp1 = etp * 0.001
    dew = W(neg, -etp1, z)
    esnow = W(neg, z, etp)
    esnow2 = W(neg, etp1 * dt, esnow * 0.001 * dt)
    etanrg = W(neg, etp * LSUBS, esnow * LSUBS)
    flx1 = W(snowng, CPICE * prcp * (t1 - sfctmp), W(prcp > 0.0, CPH2O * prcp * (t1 - sfctmp), z))
    dsoil = -(0.5 * zsoil[0])
    dtot = snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    t12a = ((fdown - flx1 - flx2 - emissi * SIGMA * t24) / rch + th2 - sfctmp - etanrg / rch) / rr
    t12b = df1 * stc[0] / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom
    cold = t12 <= TFREEZ
    t1_c = t12
    ssoil_c = df1 * (t1_c - stc[0]) / dtot
    sneqv_c = jnp.fmax(0.0, sneqv - esnow2)
    t1_w = f32(TFREEZ)
    dtot_w = W(dtot > 2.0 * dsoil, 2.0 * dsoil, dtot)
    ssoil_w = df1 * (t1_w - stc[0]) / dtot_w
    gone = (sneqv - esnow2) <= esdmin
    sneqv_w1 = sneqv - esnow2
    seh = rch * (t1_w - th2)
    t14 = (t1_w * t1_w) * (t1_w * t1_w)
    flx3_w = fdown - flx1 - flx2 - emissi * SIGMA * t14 - ssoil_w - seh - etanrg
    flx3_w = W(flx3_w <= 0.0, 0.0, flx3_w)
    ex = flx3_w * 0.001 / LSUBF
    snomlt_w = ex * dt
    enough = (sneqv_w1 - snomlt_w) >= esdmin
    sneqv_w2 = W(enough, sneqv_w1 - snomlt_w, 0.0)
    flx3_w = W(enough, flx3_w, (sneqv_w1 / dt) * 1000.0 * LSUBF)
    snomlt_w = W(enough, snomlt_w, sneqv_w1)
    sneqv_w = W(gone, 0.0, sneqv_w2)
    snomlt_w = W(gone, 0.0, snomlt_w)
    flx3_w = W(gone, 0.0, flx3_w)
    t1_n = W(cold, t1_c, t1_w)
    ssoil = W(cold, ssoil_c, ssoil_w)
    sneqv = W(cold, sneqv_c, sneqv_w)
    flx3 = W(cold, 0.0, flx3_w)
    snomlt = W(cold, 0.0, snomlt_w)
    zz1 = f32(1.0)
    yy = stc[0] - 0.5 * ssoil * zsoil[0] * zz1 / df1
    tsoil = yy
    # SHFLX / HRTICE / HSTEP (ice column)
    hcpct = 1.0e6 * (0.8194 - 0.1309 * 0.5 * zsoil[0])
    df1k = df1
    ddz = 1.0 / (-0.5 * zsoil[1])
    ai = [None] * NS; bi = [None] * NS; ci = [None] * NS; rhsts = [None] * NS
    ai[0] = f32(0.0)
    ci[0] = (df1 * ddz) / (zsoil[0] * hcpct)
    bi[0] = -ci[0] + df1 / (0.5 * zsoil[0] * zsoil[0] * hcpct * zz1)
    dtsdz = (stc[0] - stc[1]) / (-0.5 * zsoil[1])
    ssoil_h = df1 * (stc[0] - yy) / (0.5 * zsoil[0] * zz1)
    rhsts[0] = (df1 * dtsdz - ssoil_h) / (zsoil[0] * hcpct)
    ddz2 = f32(0.0)
    zbot_g = -25.0
    for k in range(1, NS):
        zmd = 0.5 * (zsoil[k] + zsoil[k - 1])
        hcpct = 1.0e6 * (0.8194 - 0.1309 * zmd)
        df1n = 0.32333 - (0.10073 * zmd)
        if k != NS - 1:
            denom_ = 0.5 * (zsoil[k - 1] - zsoil[k + 1])
            dtsdz2 = (stc[k] - stc[k + 1]) / denom_
            ddz2 = 2.0 / (zsoil[k - 1] - zsoil[k + 1])
            ci[k] = -df1n * ddz2 / ((zsoil[k - 1] - zsoil[k]) * hcpct)
        else:
            dtsdz2 = (stc[k] - tbot) / (0.5 * (zsoil[k - 1] + zsoil[k]) - zbot_g)
            ci[k] = f32(0.0)
        denom_ = (zsoil[k] - zsoil[k - 1]) * hcpct
        rhsts[k] = (df1n * dtsdz2 - df1k * dtsdz) / denom_
        ai[k] = -df1k * ddz / ((zsoil[k - 1] - zsoil[k]) * hcpct)
        bi[k] = -(ai[k] + ci[k])
        df1k = df1n
        dtsdz = dtsdz2
        ddz = ddz2
    stc = hstep(stc, rhsts, dt, ai, bi, ci)
    ok = sneqv >= 0.10
    snowh_n, sndens_n2 = snowpack(sneqv, dt, snowh, sndens, t1_n, yy)
    snowh = W(ok, snowh_n, 0.50)
    sneqv = W(ok, sneqv, 0.10)
    # SFLX_GLACIAL epilogue
    eta_kin = etp
    q1 = q2 + eta_kin * CP_MOD / rch
    sheat = -(ch * CP_MOD * sfcprs) / (287.04 * t2v) * (th2 - t1_n)
    esnow = esnow * LSUBS
    etp = etp * LSUBS
    eta = W(etp > 0.0, esnow, etp)
    ssoil = -1.0 * ssoil
    runoff1 = snomlt / dt
    new = dict(t1=t1_n, stc=jnp.stack(stc), snowh=snowh, sneqv=sneqv, albedo=albedo, z0=z0, emissi=emissi,
               snotime1=snotime1, q1=q1, sncovr=sncovr)
    out = dict(eta=eta, sheat=sheat, ssoil=ssoil, etp=etp, esnow=esnow, runoff1=runoff1, snomlt=snomlt, dew=dew,
               flx1=flx1, flx2=flx2, flx3=flx3, fdown=fdown, tsoil=tsoil)
    return new, out


# ----------------------------------------------------------------------------------------------
# SFCDIF_MYJ (module_sfcdif_wrf_39, "KWM version"): Monin-Obukhov with tabulated stability functions
import numpy as _np
KZTM = 10001
_G, _RD, _CP, _P0, _RV = 9.81, 287.0, 1004.5, 1.0e5, 461.6
_RCP = _RD / _CP
_P608 = _RV / _RD - 1.0
_ELOCP = 2.72e6 / _CP
_ZTMIN2 = -5.0


def myj_tables():
    """MYJSFCINIT: returns (ztmax2, dzeta2, psih2, psim2) as float32 (sequential float32 accumulation as in Fortran)."""
    f = _np.float32
    ztmin1, ztmax1 = f(-5.0), f(1.0)
    ztmax2 = f(1.0); zrng2 = f(ztmax2 - f(_ZTMIN2))
    dzeta2 = f(zrng2 / f(KZTM - 1))
    zeta2 = f(_ZTMIN2)
    pihf = f(3.1415926) / f(2.0)
    psim = _np.zeros(KZTM, f); psih = _np.zeros(KZTM, f)
    zt_last = zeta2
    for k in range(KZTM):
        if zeta2 < 0:
            x = _np.sqrt(_np.sqrt(f(1.0) - f(16.0) * zeta2))
            psim[k] = f(-2.0) * _np.log((x + f(1.0)) / f(2.0)) - _np.log((x * x + f(1.0)) / f(2.0)) + f(2.0) * _np.arctan(x) - pihf
            psih[k] = f(-2.0) * _np.log((x * x + f(1.0)) / f(2.0))
        else:
            psim[k] = f(0.7) * zeta2 + f(0.75) * zeta2 * (f(6.0) - f(0.35) * zeta2) * _np.exp(f(-0.35) * zeta2)
            psih[k] = psim[k]
        if k == KZTM - 1:
            ztmax2 = zeta2
        zeta2 = f(zeta2 + dzeta2)
    ztmax2 = f(ztmax2 - f(1.0e-6))
    return ztmax2, dzeta2, psih, psim


def sfcdif_myj(zsl, zsl_wind, z0, z0base, sfcprs, tz0, tlow, qz0, qlow, sfcspd, czil, akms, akhs, tab):
    """Returns (akms, akhs, rib, ustar). tab = (ztmax2, dzeta2, psih2, psim2)."""
    ztmax2, dzeta2, psih2, psim2 = tab
    psih2 = jnp.asarray(psih2); psim2 = jnp.asarray(psim2)
    vkarman, excml, epsu2, epsust, epszt = 0.4, 1.0e-4, 1.0e-6, 1.0e-9, 1.0e-28
    sqvisc, ric, ztfc, wwst = 258.2, 0.505, 1.0, 1.2
    wwst2 = wwst * wwst
    thlow = tlow * jnp.power(_P0 / sfcprs, _RCP)
    thz0 = tz0 * jnp.power(_P0 / sfcprs, _RCP)
    cxchl = excml / zsl
    btgx = _G / thlow
    elfc = vkarman * btgx
    btgh = btgx * 1000.0
    thm = (thlow + thz0) * 0.5
    tem = (tlow + tz0) * 0.5
    a = thm * _P608
    b = (_ELOCP / tem - 1.0 - _P608) * thm
    cwmlow = 0.0
    dthv = ((thlow - thz0) * ((qlow + qz0 + cwmlow) * (0.5 * _P608) + 1.0) + (qlow - qz0 + cwmlow) * a + cwmlow * b)
    du2 = jnp.fmax(sfcspd * sfcspd, epsu2)
    rib = btgx * dthv * zsl_wind * zsl_wind / du2 / zsl
    zu = z0
    zslu = zsl_wind + zu
    rzsu = zslu / zu
    rlogu = jnp.log(rzsu)
    zslt = zsl + zu
    zilfc = -czil * vkarman * sqvisc
    czetmax = 10.0
    zzil = W(dthv > 0.0, W(rib < ric, zilfc * (1.0 + (rib / ric) * (rib / ric) * czetmax), zilfc * (1.0 + czetmax)), zilfc)
    x0 = btgh * akhs * dthv
    wstar2 = W(x0 != 0.0, wwst2 * jnp.power(jnp.abs(jnp.where(x0 != 0.0, x0, 1.0)), 2.0 / 3.0), 0.0)
    ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
    kmax = KZTM - 2

    def interp(tab_, zeta):
        rz = (zeta - _ZTMIN2) / dzeta2
        k = jnp.trunc(rz).astype(jnp.int32)
        rdzt = rz - k.astype(jnp.float32)
        k = jnp.clip(k, 0, kmax)
        return (tab_[k + 1] - tab_[k]) * rdzt + tab_[k]

    for _ in range(5):
        zt = jnp.fmax(jnp.exp(zzil * jnp.sqrt(ustar * z0base)) * z0base, epszt)
        rzst = zslt / zt
        rlogt = jnp.log(rzst)
        rlmo = elfc * akhs * dthv / (ustar * ustar * ustar)
        zetalu = zslu * rlmo
        zetalt = zslt * rlmo
        zetau = zu * rlmo
        zetat = zt * rlmo
        zetalu = jnp.fmin(jnp.fmax(zetalu, _ZTMIN2), ztmax2)
        zetalt = jnp.fmin(jnp.fmax(zetalt, _ZTMIN2), ztmax2)
        zetau = jnp.fmin(jnp.fmax(zetau, _ZTMIN2 / rzsu), ztmax2 / rzsu)
        zetat = jnp.fmin(jnp.fmax(zetat, _ZTMIN2 / rzst), ztmax2 / rzst)
        psmz = interp(psim2, zetau)
        psmzl = interp(psim2, zetalu)
        simm = psmzl - psmz + rlogu
        pshz = interp(psih2, zetat)
        pshzl = interp(psih2, zetalt)
        simh = pshzl - pshz + rlogt
        ustark = ustar * vkarman
        akms = jnp.fmax(ustark / simm, cxchl)
        akhs = jnp.fmax(ustark / simh, cxchl)
        wstar2 = W(dthv <= 0.0, wwst2 * jnp.power(jnp.abs(btgh * akhs * dthv), 2.0 / 3.0), 0.0)
        ustar = jnp.fmax(jnp.sqrt(akms * jnp.sqrt(du2 + wstar2)), epsust)
    return akms, akhs, rib, ustar


# ----------------------------------------------------------------------------------------------
def sflx(forc, par, st, dt, zlvl, sldpth, opt_thcnd=1, isurban=1, aoasis=1.0, ua=False, v36=False, v33=False):
    """One Noah 3.9 step for one column. forc/par/st are dicts of scalars (see step_fn)."""
    sfctmp, q2, soldn, lwdn = forc["sfctmp"], forc["q2"], forc["soldn"], forc["lwdn"]
    sfcprs, prcp, ffrozp = forc["sfcprs"], forc["prcp"], forc["ffrozp"]
    q2sat, dqsdt2, th2, solnet = forc["q2sat"], forc["dqsdt2"], forc["th2"], forc["solnet"]
    ch, cm = forc["ch"], forc["cm"]
    vegtyp, soiltyp, nroot = par["vegt"], par["soiltyp"], par["nroot"]
    shdfac, shdmin, shdmax = par["shdfac"], par["shdmin"], par["shdmax"]
    smcwlt, smcdry, smcref, smcmax = par["smcwlt"], par["smcdry"], par["smcref"], par["smcmax"]
    rsmin = par["rsmin"]
    urban = vegtyp == isurban
    shdfac = W(urban, 0.05, shdfac)
    rsmin = W(urban, 400.0, rsmin)
    smcmax = W(urban, 0.45, smcmax)
    smcref = W(urban, 0.42, smcref)
    smcwlt = W(urban, 0.40, smcwlt)
    smcdry = W(urban, 0.40, smcdry)
    cmc, t1 = st["cmc"], st["t1"]
    stc = [st["stc"][k] for k in range(NS)]
    smc = [st["smc"][k] for k in range(NS)]
    sh2o = [st["sh2o"][k] for k in range(NS)]
    snowh, sneqv, albedo = st["snowh"], st["sneqv"], st["albedo"]
    z0, snotime1, xlai, emissi = st["z0"], st["snotime1"], st["xlai"], st["emissi"]
    ribb = forc.get("ribb", f32(0.0))
    zsoil = [-sldpth[0]]
    for k in range(1, NS):
        zsoil.append(-sldpth[k] + zsoil[k - 1])
    zs_root = jnp.stack(zsoil)[nroot - 1]
    rtdis = [-sldpth[i] / zs_root for i in range(NS)]
    # --- snow bookkeeping
    tiny = sneqv <= 1.0e-7
    sneqv = W(tiny, 0.0, sneqv)
    sndens = W(tiny, 0.0, sneqv / snowh)
    snowh = W(tiny, 0.0, snowh)
    sncond = W(tiny, 1.0, csnow(sndens))
    snowng = (prcp > 0.0) & (ffrozp > 0.5)
    frzgra = (prcp > 0.0) & (~(ffrozp > 0.5)) & (t1 <= TFREEZ)
    addsn = snowng | frzgra
    sn_new = prcp * dt * 0.001
    sneqv_n = sneqv + sn_new
    snowh_n, sndens_n = snow_new(sfctmp, sn_new, snowh, sndens)
    sncond_n = csnow(sndens_n)
    sneqv = W(addsn, sneqv_n, sneqv)
    snowh = W(addsn, snowh_n, snowh)
    sndens = W(addsn, sndens_n, sndens)
    sncond = W(addsn, sncond_n, sncond)
    prcpf = W(addsn, 0.0, prcp)
    # --- albedo / emissivity / snow cover
    nosnow = sneqv == 0.0
    if ua:
        sncovr_r, xlai_s, fvb_s, gama_s, fbur_s, fgsn_s = snfrac(sneqv, par["snup"], par["salp"], snowh, xlai, shdfac,
                                                                 par["ztopv"], par["zbotv"], ua=True)
        ru = W(sfctmp <= t1, 0.0, 100.0 * shdfac * fgsn_s * jnp.fmin((sfctmp - t1) / 5.0, 1.0) * (1.0 - jnp.exp(-xlai_s)))
        ch_s = ch / (1.0 + ru * ch)
        sncovr_s = jnp.fmin(sncovr_r, 0.98)
        xlai = W(sneqv == 0.0, xlai, xlai_s)
        fvb = W(sneqv == 0.0, 0.0, fvb_s); gama = W(sneqv == 0.0, 0.0, gama_s)
        fbur = W(sneqv == 0.0, 0.0, fbur_s); fgsn = W(sneqv == 0.0, 0.0, fgsn_s)
        ch = W(sneqv == 0.0, ch, ch_s)
    else:
        sncovr_s = jnp.fmin(snfrac(sneqv, par["snup"], par["salp"]), 0.98)
        fvb = gama = fbur = fgsn = f32(0.0)
    alb_s, emis_s, snot_s = alcalc(par["alb"], par["snoalb"], par["embrd"], sncovr_s, dt, snowng,
                                   snotime1, par["lvcoef"])
    sncovr = W(nosnow, 0.0, sncovr_s)
    albedo = W(nosnow, par["alb"], alb_s)
    emissi = W(nosnow, par["embrd"], emis_s)
    snotime1 = W(nosnow, snotime1, snot_s)
    # --- thermal conductivity
    df1 = tdfcnd(smc[0], par["quartz"], smcmax, sh2o[0], par["bexp"], par["psisat"], soiltyp,
                 opt_thcnd)
    df1 = W(urban, 3.24, df1)
    df1 = df1 * jnp.exp(par["sbeta"] * shdfac)
    df1 = W(sncovr > 0.97, sncond, df1)
    dsoil = -(0.5 * zsoil[0])
    dtot = snowh + dsoil
    frcsno = snowh / dtot
    frcsoi = dsoil / dtot
    df1a = frcsno * sncond + frcsoi * df1
    df1_sn = df1a * sncovr + df1 * (1.0 - sncovr)
    ssoil = W(nosnow, df1 * (t1 - stc[0]) / dsoil, df1_sn * (t1 - stc[0]) / dtot)
    df1 = W(nosnow, df1, df1_sn)
    if ua:
        z0 = snowz0(sncovr, par["z0brd"], snowh, ua=True, fbur=fbur, fgsn=fgsn, shdmax=shdmax)
    else:
        z0 = W(sncovr > 0.0, snowz0(sncovr, par["z0brd"], snowh), par["z0brd"])
    fdown = solnet + lwdn
    t2v = sfctmp * (1.0 + 0.61 * q2)
    etp, rch, epsca, rr, t24, flx2, etpn, flx4 = penman(
        sfctmp, sfcprs, ch, t2v, th2, prcp, fdown, ssoil, q2, q2sat, snowng, frzgra, dqsdt2, emissi, sncovr, aoasis,
        ua=ua, sneqv=sneqv, albedo=albedo, soldn=soldn, fvb=fvb, gama=gama, stc1=stc[0])
    can = (shdfac > 0.0) if v33 else ((shdfac > 0.0) & (xlai > 0.0))      # Noah 3.3 calls CANRES whenever SHDFAC > 0 (RC = inf, PC = 0 for LAI = 0)
    rc_c, pc_c, rcs, rct, rcq, rcsoil = canres(soldn, ch, sfctmp, q2, sfcprs, sh2o, zsoil, smcwlt,
                                               smcref, rsmin, nroot, q2sat, dqsdt2, par["topt"],
                                               par["rsmax"], par["rgl"], par["hs"], xlai, emissi)
    rc = W(can, rc_c, 0.0)
    pc = W(can, pc_c, 1.0)
    # --- NOPAC (no snow) and SNOPAC (snow)
    common = dict(smc=smc, smcmax=smcmax, smcwlt=smcwlt, smcref=smcref, smcdry=smcdry, cmc=cmc,
                  cmcmax=par["cmcmax"], dt=dt, sbeta=par["sbeta"], q2=q2, t1=t1, sfctmp=sfctmp,
                  t24=t24, th2=th2, fdown=fdown, f1=par["f1"], stc=stc, epsca=epsca,
                  bexp=par["bexp"], pc=pc, rch=rch, rr=rr, cfactr=par["cfactr"], sh2o=sh2o,
                  slope=par["slope"], kdt=par["kdt"], frzx=par["frzx"], psisat=par["psisat"],
                  zsoil=zsoil, dksat=par["dksat"], dwsat=par["dwsat"], tbot=par["tbot"],
                  zbot=par["zbot"], nroot=nroot, rtdis=rtdis, quartz=par["quartz"],
                  fxexp=par["fxexp"], csoil=par["csoil"], vegtyp=vegtyp, isurban=isurban,
                  soiltyp=soiltyp, opt_thcnd=opt_thcnd)
    npc = nopac(etp=etp, prcp=prcp, shdfac=shdfac, emissi=emissi, **common)
    spc = snopac(etp=etp, prcp=prcp, prcpf=prcpf, snowng=snowng, df1=df1, sfcprs=sfcprs,
                 sncovr=sncovr, esd=sneqv, sndens=sndens, snowh=snowh, shdfac=shdfac,
                 emissi=emissi, ribb=ribb, soldn=soldn, flx2=flx2, ua=ua, etpn=etpn, flx4=flx4, v36=v36, **common)
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
    tsoil = sel(npc["t1"] if v36 else npc["tsoil"], spc["tsoil"])
    flx4f = sel(flx4, spc["flx4"]) if ua else f32(0.0)
    dew = sel(npc["dew"], spc["dew"])
    esnow = sel(npc["esnow"], spc["esnow"])
    snomlt = sel(npc["snomlt"], spc["snomlt"])
    eta_kin = sel(npc["eta_kin"], spc["eta_kin"])
    flx1 = sel(npc["flx1"], spc["flx1"])
    flx3 = sel(npc["flx3"], spc["flx3"])
    etp = sel(etp, spc["etp"])
    sneqv = sel(sneqv, spc["esd"])
    snowh = sel(snowh, spc["snowh"])
    sncovr = sel(sncovr, spc["sncovr"])
    sndens = sel(sndens, spc["sndens"])
    q1 = q2 + eta_kin * CP_MOD / rch
    t2v_ = t2v
    sheat = -(ch * CP_MOD * sfcprs) / (287.04 * t2v_) * (th2 - t1) + flx4f
    evp = W(etp > 0.0, edir + ec + ett + esnow, etp)
    # --- unit conversions and diagnostics
    edir = edir * LVH2O
    ec = ec * LVH2O
    et = [e * LVH2O for e in et]
    ett = ett * LVH2O
    esnow = esnow * LSUBS
    etp = etp * ((1.0 - sncovr) * LVH2O + sncovr * LSUBS)
    eta = W(etp > 0.0, edir + ec + ett + esnow, etp)
    beta = W(etp == 0.0, 0.0, eta / jnp.where(etp == 0.0, 1.0, etp))
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
    small = soilwm < 1.0e-6
    soilw = W(small, 0.0, soilww / jnp.where(small, 1.0, soilwm))
    soilm = W(small, 0.0, soilm)
    new_st = dict(cmc=cmc, t1=t1, stc=jnp.stack(stc), smc=jnp.stack(smc), sh2o=jnp.stack(sh2o),
                  snowh=snowh, sneqv=sneqv, albedo=albedo, sncovr=sncovr, z0=z0,
                  snotime1=snotime1, q1=q1, emissi=emissi, xlai=xlai)
    out = dict(shdfac=shdfac, eta=eta, sheat=sheat, ssoil=ssoil, etp=etp, edir=edir, ec=ec, ett=ett,
               esnow=esnow, runoff1=ro1, runoff2=ro2, runoff3=ro3, snomlt=snomlt, drip=drip,
               dew=dew, beta=beta, fdown=fdown, rc=rc, pc=pc, soilm=soilm, soilw=soilw, tsoil=tsoil,
               flx1=flx1, flx2=flx2, flx3=flx3, evp=evp, eta_kin=eta_kin,
               smcmax_o=smcmax, smcref_o=smcref, smcwlt_o=smcwlt, smcdry_o=smcdry, rsmin_o=rsmin,
               ch_o=ch, flx4=flx4f)
    return new_st, out


# ----------------------------------------------------------------------------------------------
STATIC_FLOAT = ["slope", "shdfac", "shdmin", "shdmax", "z0brd", "embrd", "alb", "snoalb", "tbot",
                "czil", "sbeta", "fxexp", "csoil", "salp", "kdt", "cfactr", "zbot", "refkdt",
                "frzx", "lvcoef", "ztopv", "zbotv", "smcwlt", "smcdry", "smcref", "smcmax", "hs",
                "rgl", "snup", "cmcmax", "rsmax", "topt", "bexp", "dksat", "dwsat", "f1", "quartz",
                "psisat", "rsmin"]


def column_step(par, st, fc, dt=1800.0, zlvl=10.0, opt_thcnd=1, isurban=1):
    """Wrapper + SFLX for one column, one step. Mirrors the driver.F90 preparation."""
    sfctmp, q2, soldn = fc["sfctmp"], fc["q2"], fc["soldn"]
    sfcspd, sfcprs, prcp = fc["sfcspd"], fc["sfcprs"], fc["prcp"]
    emissi = st["emissi"]
    lwdn = fc["lwdn"] * emissi
    ffrozp = W((prcp > 0.0) & (sfctmp < 273.15), 1.0, 0.0)
    sfcspd = W(sfcspd <= 0.01, 0.01, sfcspd)
    q2 = W(q2 < 0.1e-5, 0.1e-5, q2)
    a2, a3, a4, elwv, e0, rv, eps = 17.67, 273.15, 29.65, 2.501e6, 611.0, 461.0, 0.622
    es = e0 * jnp.exp(elwv / rv * (1.0 / a3 - 1.0 / sfctmp))
    q2sat = eps * es / (sfcprs - (1 - eps) * es)
    dqsdt2 = q2sat * a2 * (a3 - a4) / ((sfctmp - a4) * (sfctmp - a4))
    th2 = sfctmp + 0.0098 * zlvl
    t1 = st["t1"]
    t1v = t1 * (1.0 + 0.61 * q2)
    th2v = th2 * (1.0 + 0.61 * q2)
    cm, ch, _ = sfcdif_off(zlvl, zlvl, st["z0"], t1v, th2v, sfcspd, par["czil"], st["cm"], st["ch"])
    solnet = soldn * (1.0 - st["albedo"])
    forc = dict(sfctmp=sfctmp, q2=q2, soldn=soldn, lwdn=lwdn, sfcprs=sfcprs, prcp=prcp,
                ffrozp=ffrozp, q2sat=q2sat, dqsdt2=dqsdt2, th2=th2, solnet=solnet, ch=ch, cm=cm)
    sldpth = [f32(0.1), f32(0.3), f32(0.6), f32(1.0)]
    new_st, out = sflx(forc, par, st, dt, zlvl, sldpth, opt_thcnd=opt_thcnd, isurban=isurban)
    new_st["ch"], new_st["cm"] = ch, cm
    return new_st, out
