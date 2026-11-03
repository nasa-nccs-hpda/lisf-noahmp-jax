"""JAX port of the Noah-MP 4.0.1 land-ice branch (NOAHMP_GLACIER and callees) for the option set glacier option 1
(phase change in the ice layers), snow albedo 1|2, snf 1-3, tbot 1|2, stc 1.
Source: lis/surfacemodels/land/noahmp.4.0.1/phys/module_sf_noahmp_glacier_401.F90. Layer conventions as in noahmp401_jax."""
import jax
import jax.numpy as jnp
from .noahmp401_jax import *          # constants, helpers
from . import noahmp401_jax as nm

NS = 4


def _csnow_thermo(isnow, snice, snliq, dz):
    df = [None] * NL; hc = [None] * NL
    for p in range(3):
        a = isnow + 3 <= p
        dzs = W(a, dz[p], 1.0)
        snicev = jnp.fmin(1., snice[p] / (dzs * DENICE))
        epore = 1. - snicev
        snliqv = jnp.fmin(epore, snliq[p] / (dzs * DENH2O))
        bdsnoi = (snice[p] + snliq[p]) / dzs
        hc[p] = W(a, CICE * snicev + CWAT * snliqv, 0.)
        df[p] = W(a, 3.2217E-6 * bdsnoi ** 2., 0.)
    return df, hc


def thermoprop(isnow, dt, snowh, snice, snliq, dz):
    df, hc = _csnow_thermo(isnow, snice, snliq, dz)
    for k in range(NS):
        p = k + 3
        zmid = 0.5 * dz[p]
        for k2 in range(k):
            zmid = zmid + dz[k2 + 3]
        hc[p] = 1.E6 * (0.8194 + 0.1309 * zmid)
        df[p] = 0.32333 + (0.10073 * zmid)
    fact = [W(isnow + 3 <= p, dt / (hc[p] * W(isnow + 3 <= p, dz[p], 1.0)), 0.) for p in range(NL)]
    df3_0 = (df[3] * dz[3] + 0.35 * snowh) / (snowh + dz[3])
    df3_1 = (df[3] * dz[3] + df[2] * dz[2]) / (dz[2] + dz[3])
    df[3] = W(isnow == 0, df3_0, df3_1)
    return df, hc, fact


def radiation(o, dt, tg, sneqvo, sneqv, cosz, qsnow, solad, solai, albold, tauss):
    SWEMX = 1.0
    dela0 = 1.E-6 * dt
    arg = 5.E3 * (1. / TFRZ - 1. / tg)
    age1 = jnp.exp(arg); age2 = jnp.exp(jnp.fmin(0., 10. * arg))
    tage = age1 + age2 + 0.3
    dela = dela0 * tage
    dels = jnp.fmax(0.0, sneqv - sneqvo) / SWEMX
    sge = (tauss + dela) * (1.0 - dels)
    tauss_n = W(sneqv <= 0.0, 0., W(sneqv > 800., 0., jnp.fmax(0., sge)))
    fage = tauss_n / (tauss_n + 1.)
    albold_n = albold
    if o["alb"] == 1:
        sl = 2.0; sl1 = 1. / sl; sl2 = 2. * sl
        cf1 = ((1. + sl1) / (1. + sl2 * cosz) - sl1)
        fzen = jnp.fmax(cf1, 0.)
        albsni = [0.95 * (1. - 0.2 * fage), 0.65 * (1. - 0.5 * fage)]
        albsnd = [albsni[0] + 0.4 * fzen * (1. - albsni[0]), albsni[1] + 0.4 * fzen * (1. - albsni[1])]
    else:
        alb = 0.55 + (albold - 0.55) * jnp.exp(-0.01 * dt / 3600.)
        alb = W(qsnow > 0., alb + jnp.fmin(qsnow * dt, SWEMX) * (0.84 - alb) / SWEMX, alb)
        albsni = [alb, alb]; albsnd = [alb, alb]
        albold_n = alb
    albice = [0.80, 0.55]
    fsno = W(sneqv > 0.0, 1.0, 0.0)
    sag = f32(0.); fsa = f32(0.); fsr = f32(0.)
    for ib in range(2):
        asnd = albice[ib] * (1. - fsno) + albsnd[ib] * fsno
        asni = albice[ib] * (1. - fsno) + albsni[ib] * fsno
        ab = solad[ib] * (1. - asnd) + solai[ib] * (1. - asni)
        sag = sag + ab; fsa = fsa + ab
        ref = solad[ib] * asnd + solai[ib] * asni
        fsr = fsr + ref
    return sag, fsr, fsa, tauss_n, albold_n


def glacier_flux(o, emg, isnow, df1, dz1, z0m, zlvl, zpd, qair, sfctmp, rhoair, sfcprs, ur, gamma, rsurf, lwdn, rhsur,
                 smc, eair, stc1, sag, snowh, lathea, sh2o, cm, ch, tgb):
    mpe = MPE
    cir = emg * SB
    cgh = 2. * df1 / dz1
    z = f32(0.)
    s = dict(moz=z, mozsgn=jnp.int32(0), fm=z, fh=z, fm2=z, fh2=z, fv=f32(0.1), h=z, qsfc=z, cm=cm, ch=ch)
    csh = cev = estg = z
    irb = shb = evb = ghb = z
    for it in range(1, 6):
        moz, mozsgn, fm, fh, fm2, fh2, cm_, ch_, fv, ch2 = nm.sfcdif1(
            jnp.int32(it), sfctmp, rhoair, s["h"], qair, zlvl, zpd, z0m, z0m, ur, s["moz"], s["mozsgn"], s["fm"],
            s["fh"], s["fm2"], s["fh2"], s["fv"])
        rahb = jnp.fmax(1., 1. / (ch_ * ur))
        rawb = rahb
        estg, destg = nm._esat_pair(tgb)
        csh = rhoair * CPAIR / rahb
        cev = W((snowh > 0.0) | (o["gla"] == 1), rhoair * CPAIR / gamma / (rsurf + rawb), 0.0)
        irb = cir * tgb ** 4 - emg * lwdn
        shb = csh * (tgb - sfctmp)
        evb = cev * (estg * rhsur - eair)
        ghb = cgh * (tgb - stc1)
        b = sag - irb - shb - evb - ghb
        a = 4. * cir * tgb ** 3 + csh + cev * destg + cgh
        dtg = b / a
        irb = irb + 4. * cir * tgb ** 3 * dtg
        shb = shb + csh * dtg
        evb = evb + cev * destg * dtg
        ghb = ghb + cgh * dtg
        tgb = tgb + dtg
        h = csh * (tgb - sfctmp)
        estg, _ = nm._esat_pair(tgb)
        qsfc = 0.622 * (estg * rhsur) / (sfcprs - 0.378 * (estg * rhsur))
        s = dict(moz=moz, mozsgn=mozsgn, fm=fm, fh=fh, fm2=fm2, fh2=fh2, fv=fv, h=h, qsfc=qsfc, cm=cm_, ch=ch_)
    sice_max = f32(-1e30)
    for k in range(NS):
        sice_max = jnp.fmax(sice_max, smc[k] - sh2o[k])
    adj = ((sice_max > 0.0) | (snowh > 0.0)) & (tgb > TFRZ) & (o["gla"] == 1)
    tg_a = f32(TFRZ)
    _, esi, _, _ = nm.esat(nm.tdc(tg_a))
    qsfc_a = 0.622 * (esi * rhsur) / (sfcprs - 0.378 * (esi * rhsur))
    irb_a = cir * tg_a ** 4 - emg * lwdn
    shb_a = csh * (tg_a - sfctmp)
    evb_a = cev * (esi * rhsur - eair)
    ghb_a = sag - (irb_a + shb_a + evb_a)
    tgb, irb, shb, evb, ghb, qsfc = W(adj, tg_a, tgb), W(adj, irb_a, irb), W(adj, shb_a, shb), W(adj, evb_a, evb), W(adj, ghb_a, ghb), W(adj, qsfc_a, s["qsfc"])
    ehb2 = s["fv"] * VKC / (jnp.log((2. + z0m) / z0m) - s["fh2"])
    small = ehb2 < 1.e-5
    t2mb = W(small, tgb, tgb - shb / (rhoair * CPAIR) * 1. / ehb2)
    q2b = W(small, qsfc, qsfc - evb / (lathea * rhoair) * (1. / ehb2 + rsurf))
    ch_out = 1. / rahb
    return dict(tg=tgb, cm=s["cm"], ch=ch_out, qsfc=qsfc, fira=irb, fsh=shb, fgev=evb, ssoil=ghb, t2m=t2mb, q2e=q2b,
                ch2b=ehb2)


def phasechange(o, isnow, dt, fact, dz, stc, snice, snliq, sneqv, snowh, smc, sh2o):
    pf = isnow + 3
    act = [pf <= p for p in range(3)]
    mice = [None] * NL; mliq = [None] * NL
    stc = list(stc)
    for p in range(3):
        mice[p] = snice[p]; mliq[p] = snliq[p]
    qmelt = f32(0.); ponding = f32(0.)
    imelt = [jnp.int32(0)] * NL
    hm = [f32(0.)] * NL; xm = [f32(0.)] * NL
    wice0 = [None] * NL; wmass0 = [None] * NL
    for p in range(3):
        wice0[p] = mice[p]; wmass0[p] = mice[p] + mliq[p]
        im = W((mice[p] > 0.) & (stc[p] >= TFRZ), 1, 0)
        im = W((mliq[p] > 0.) & (stc[p] < TFRZ), 2, im)
        im = W(act[p], im, 0)
        h = W(im > 0, (stc[p] - TFRZ) / W(act[p], fact[p], 1.0), 0.)
        stc[p] = W(im > 0, TFRZ, stc[p])
        bad = ((im == 1) & (h < 0.)) | ((im == 2) & (h > 0.))
        h = W(bad, 0., h); im = W(bad, 0, im)
        imelt[p] = im; hm[p] = h; xm[p] = h * dt / HFUS
    for p in range(3):
        go = (imelt[p] > 0) & (jnp.abs(hm[p]) > 0.) & act[p]
        pos = xm[p] > 0.; neg = xm[p] < 0.
        mi = W(pos, jnp.fmax(0., wice0[p] - xm[p]), W(neg, jnp.fmin(wmass0[p], wice0[p] - xm[p]), mice[p]))
        he = W(pos | neg, hm[p] - HFUS * (wice0[p] - mi) / dt, 0.)
        ml = jnp.fmax(0., wmass0[p] - mi)
        upd = go & (jnp.abs(he) > 0.)
        st = W(upd, stc[p] + fact[p] * he, stc[p])
        st = W(upd & (ml * mi > 0.), TFRZ, st)
        stc[p] = st
        qmelt = qmelt + W(go, jnp.fmax(0., (wice0[p] - mi)) / dt, 0.)
        mice[p] = W(go, mi, mice[p]); mliq[p] = W(go, ml, mliq[p])
    # ---- ice layers (OPT_GLA = 1)
    for k in range(NS):
        p = k + 3
        mliq[p] = sh2o[k] * dz[p] * 1000.
        mice[p] = (smc[k] - sh2o[k]) * dz[p] * 1000.
        wice0[p] = mice[p]; wmass0[p] = mice[p] + mliq[p]
        im = W((mice[p] > 0.) & (stc[p] >= TFRZ), 1, 0)
        im = W((mliq[p] > 0.) & (stc[p] < TFRZ), 2, im)
        if k == 0:
            im = W((isnow == 0) & (sneqv > 0.) & (stc[p] >= TFRZ), 1, im)
        h = W(im > 0, (stc[p] - TFRZ) / fact[p], 0.)
        stc[p] = W(im > 0, TFRZ, stc[p])
        bad = ((im == 1) & (h < 0.)) | ((im == 2) & (h > 0.))
        h = W(bad, 0., h); im = W(bad, 0, im)
        imelt[p] = im; hm[p] = h; xm[p] = h * dt / HFUS
    sp = (isnow == 0) & (sneqv > 0.) & (xm[3] > 0.)
    temp1 = sneqv
    sneqv_n = jnp.fmax(0., temp1 - xm[3])
    propor = sneqv_n / jnp.fmax(temp1, 1e-30)
    snowh_n = jnp.fmax(0., propor * snowh)
    heatr1 = hm[3] - HFUS * (temp1 - sneqv_n) / dt
    pos1 = heatr1 > 0.
    xm[3] = W(sp, W(pos1, heatr1 * dt / HFUS, 0.), xm[3])
    hm[3] = W(sp, W(pos1, heatr1, 0.), hm[3])
    imelt[3] = W(sp, W(pos1, 1, 0), imelt[3])
    qmelt = W(sp, jnp.fmax(0., (temp1 - sneqv_n)) / dt, qmelt)
    ponding = W(sp, temp1 - sneqv_n, ponding)
    sneqv = W(sp, sneqv_n, sneqv); snowh = W(sp, snowh_n, snowh)
    for k in range(NS):
        p = k + 3
        go = (imelt[p] > 0) & (jnp.abs(hm[p]) > 0.)
        pos = xm[p] > 0.; neg = xm[p] < 0.
        mi = W(pos, jnp.fmax(0., wice0[p] - xm[p]), W(neg, jnp.fmin(wmass0[p], wice0[p] - xm[p]), mice[p]))
        he = W(pos | neg, hm[p] - HFUS * (wice0[p] - mi) / dt, 0.)
        ml = jnp.fmax(0., wmass0[p] - mi)
        upd = go & (jnp.abs(he) > 0.)
        stc[p] = W(upd, stc[p] + fact[p] * he, stc[p])
        mice[p] = W(go, mi, mice[p]); mliq[p] = W(go, ml, mliq[p])
    sl = list(range(3, 7))
    heatr = {p: f32(0.) for p in sl}
    xmv = {p: f32(0.) for p in sl}
    anyp = lambda: jnp.any(jnp.stack([stc[p] > TFRZ for p in sl]))
    anyn = lambda: jnp.any(jnp.stack([stc[p] < TFRZ for p in sl]))
    # block 1
    c1 = anyp() & anyn()
    for j in sl:
        gj = c1 & (stc[j] > TFRZ)
        hj = W(gj, (stc[j] - TFRZ) / fact[j], heatr[j])
        for kk in sl:
            if kk == j:
                continue
            ok = gj & (stc[kk] < TFRZ) & (hj > 0.1)
            hk = (stc[kk] - TFRZ) / fact[kk]
            absorb = jnp.abs(hk) > hj
            hk2 = hk + hj
            stc_k = W(absorb, TFRZ + hk2 * fact[kk], TFRZ)
            hj_n = W(absorb, 0.0, hj + hk)
            stc[kk] = W(ok, stc_k, stc[kk])
            hj = W(ok, hj_n, hj)
        stc[j] = W(gj, TFRZ + hj * fact[j], stc[j])
        heatr[j] = hj
    # block 2
    c2 = anyp() & anyn()
    for j in sl:
        gj = c2 & (stc[j] < TFRZ)
        hj = W(gj, (stc[j] - TFRZ) / fact[j], heatr[j])
        for kk in sl:
            if kk == j:
                continue
            ok = gj & (stc[kk] > TFRZ) & (hj < -0.1)
            hk = (stc[kk] - TFRZ) / fact[kk]
            absorb = hk > jnp.abs(hj)
            hk2 = hk + hj
            stc_k = W(absorb, TFRZ + hk2 * fact[kk], TFRZ)
            hj_n = W(absorb, 0.0, hj + hk)
            stc[kk] = W(ok, stc_k, stc[kk])
            hj = W(ok, hj_n, hj)
        stc[j] = W(gj, TFRZ + hj * fact[j], stc[j])
        heatr[j] = hj
    # block 3: warm layers melt ice in other layers
    c3 = anyp() & jnp.any(jnp.stack([mice[p] > 0. for p in sl]))
    for j in sl:
        gj = c3 & (stc[j] > TFRZ)
        hj = (stc[j] - TFRZ) / fact[j]
        xj = W(gj, hj * dt / HFUS, xmv[j])
        for kk in sl:
            if kk == j:
                continue
            ok = gj & (mice[kk] > 0.) & (xj > 0.1)
            allab = mice[kk] > xj
            mice_k = W(allab, mice[kk] - xj, 0.0)
            xj_n = W(allab, 0.0, xj - mice[kk])
            mice[kk] = W(ok, mice_k, mice[kk])
            stc[kk] = W(ok, TFRZ, stc[kk])
            xj = W(ok, xj_n, xj)
            mliq[kk] = W(ok, jnp.fmax(0., wmass0[kk] - mice[kk]), mliq[kk])
        xmv[j] = xj
        heatr[j] = W(gj, xj * HFUS / dt, heatr[j])
        stc[j] = W(gj, TFRZ + heatr[j] * fact[j], stc[j])
    # block 4: cold layers freeze liquid in other layers
    c4 = anyn() & jnp.any(jnp.stack([mliq[p] > 0. for p in sl]))
    for j in sl:
        gj = c4 & (stc[j] < TFRZ)
        hj = (stc[j] - TFRZ) / fact[j]
        xj = W(gj, hj * dt / HFUS, xmv[j])
        for kk in sl:
            if kk == j:
                continue
            ok = gj & (mliq[kk] > 0.) & (xj < -0.1)
            allab = mliq[kk] > jnp.abs(xj)
            mice_k = W(allab, mice[kk] - xj, wmass0[kk])
            xj_n = W(allab, 0.0, xj + mliq[kk])
            mice[kk] = W(ok, mice_k, mice[kk])
            stc[kk] = W(ok, TFRZ, stc[kk])
            xj = W(ok, xj_n, xj)
            mliq[kk] = W(ok, jnp.fmax(0., wmass0[kk] - mice[kk]), mliq[kk])
        xmv[j] = xj
        heatr[j] = W(gj, xj * HFUS / dt, heatr[j])
        stc[j] = W(gj, TFRZ + heatr[j] * fact[j], stc[j])
    snliq_o = [W(act[p], mliq[p], snliq[p]) for p in range(3)]
    snice_o = [W(act[p], mice[p], snice[p]) for p in range(3)]
    sh2o_o = [jnp.fmax(0.0, jnp.fmin(1.0, mliq[k + 3] / (1000. * dz[k + 3]))) for k in range(NS)]
    smc_o = [f32(1.0)] * NS
    return stc, snice_o, snliq_o, sneqv, snowh, smc_o, sh2o_o, qmelt, imelt, ponding


def snowh2o(P, o, dt, qsnfro, qsnsub, qrain, sn):
    sn = dict(sn)
    isnow, sneqv, snowh = sn["isnow"], sn["sneqv"], sn["snowh"]
    sice1, sh2o1 = sn["sice1"], sn["sh2o1"]
    ice = list(sn["ice"]); liq = list(sn["liq"])
    dzs1 = sn["dzs1"]
    SSI = 0.03
    z0 = sneqv == 0.
    sice1 = W(z0, sice1 + (qsnfro - qsnsub) * dt / (dzs1 * 1000.), sice1)
    top = (isnow == 0) & (sneqv > 0.)
    temp = sneqv
    sneqv_t = sneqv - qsnsub * dt + qsnfro * dt
    propor = sneqv_t / jnp.fmax(temp, 1e-30)
    snowh_t = jnp.fmax(0., propor * snowh)
    neg = sneqv_t < 0.
    sice1_t = W(neg, sice1 + sneqv_t / (dzs1 * 1000.), sice1)
    sneqv_t2 = W(neg, 0., sneqv_t); snowh_t2 = W(neg, 0., snowh_t)
    n2 = sice1_t < 0.
    sh2o1_t = W(n2, sh2o1 + sice1_t, sh2o1)
    sice1_t2 = W(n2, 0., sice1_t)
    sneqv = W(top, sneqv_t2, sneqv); snowh = W(top, snowh_t2, snowh)
    sice1 = W(top, sice1_t2, sice1); sh2o1 = W(top, sh2o1_t, sh2o1)
    tiny = (snowh <= 1.E-8) | (sneqv <= 1.E-6)
    snowh = W(tiny, 0.0, snowh); sneqv = W(tiny, 0.0, sneqv)
    lay = isnow < 0
    for p in range(3):
        m = lay & (p == isnow + 3)
        ice[p] = W(m, ice[p] - qsnsub * dt + qsnfro * dt, ice[p])
    wgdif = nm.at(ice, jnp.clip(isnow + 3, 0, 2))
    sn.update(isnow=isnow, sneqv=sneqv, snowh=snowh, sice1=sice1, sh2o1=sh2o1, ice=ice, liq=liq)
    cmb = lay & (wgdif < 1.e-6)
    sn_c = nm.combine(P, sn, gla=True)
    sn = nm._dsel(cmb, sn_c, sn)
    isnow, ice, liq = sn["isnow"], list(sn["ice"]), list(sn["liq"])
    lay2 = isnow < 0
    for p in range(3):
        m = lay2 & (p == isnow + 3)
        liq[p] = W(m, jnp.fmax(0., liq[p] + qrain * dt), liq[p])
    dz = sn["dz"]
    vol_ice = []; epore = []; vol_liq = []
    for p in range(3):
        a = isnow + 3 <= p
        dzs = W(a, dz[p], 1.0)
        vi = jnp.fmin(1., ice[p] / (dzs * DENICE))
        ep = 1. - vi
        vol_ice.append(vi); epore.append(ep)
        vol_liq.append(jnp.fmin(ep, liq[p] / (dzs * DENH2O)))
    qin = f32(0.); qout = f32(0.)
    for p in range(3):
        a = isnow + 3 <= p
        liq_n = liq[p] + qin
        if p <= 1:
            blocked = (epore[p] < 0.05) | (epore[p + 1] < 0.05)
            qo = jnp.fmax(0., (vol_liq[p] - SSI * epore[p]) * W(a, dz[p], 1.0))
            qo = jnp.fmin(qo, (1. - vol_ice[p + 1] - vol_liq[p + 1]) * W(a, dz[p + 1], 1.0))
            qo = W(blocked, 0., qo)
        else:
            qo = jnp.fmax(0., (vol_liq[p] - SSI * epore[p]) * W(a, dz[p], 1.0))
        qo = qo * 1000.
        liq[p] = W(a, liq_n - qo, liq[p])
        qin = W(a, qo, qin)
        qout = W(a, qo, qout)
    sn.update(ice=ice, liq=liq)
    return sn, qout / dt


def snowwater(P, o, dt, zsoil, sfctmp, snowhin, qsnow, qsnfro, qsnsub, qrain, ficeold, imelt, isnow, snowh, sneqv,
              snice, snliq, sh2o1, sice1, stc, zsnso, dzsnso, sndpth_max):
    dzsnso = list(dzsnso); stc = list(stc)
    isnow, snowh, dzs3, stc3, snice, snliq, sneqv = nm.snowfall(dt, qsnow, snowhin, sfctmp, isnow, snowh, dzsnso[:3], stc[:3],
                                                                list(snice), list(snliq), sneqv, thr=0.05)
    dzsnso[:3] = dzs3; stc[:3] = stc3
    lay = isnow < 0
    dzc = nm.compact(P, dt, isnow, stc[:3], snice, snliq, imelt[:3], ficeold, dzsnso[:3])
    dzsnso[:3] = [W(lay, a, b) for a, b in zip(dzc, dzsnso[:3])]
    sn = dict(isnow=isnow, dz=list(dzsnso[:3]), ice=list(snice), liq=list(snliq), stc=list(stc[:3]), sice1=sice1,
              sh2o1=sh2o1, dzs1=dzsnso[3], snowh=snowh, sneqv=sneqv, ponding1=f32(0.), ponding2=f32(0.))
    cm = nm.combine(P, sn, gla=True)
    sn = nm._dsel(lay, cm, sn)
    isn2, stc2, ice2, liq2, dz2 = nm.divide(sn["isnow"], sn["stc"], sn["ice"], sn["liq"], sn["dz"], thr2=0.10)
    lay = sn["isnow"] < 0
    sn["isnow"] = W(lay, isn2, sn["isnow"])
    sn["stc"] = [W(lay, a, b) for a, b in zip(stc2, sn["stc"])]
    sn["ice"] = [W(lay, a, b) for a, b in zip(ice2, sn["ice"])]
    sn["liq"] = [W(lay, a, b) for a, b in zip(liq2, sn["liq"])]
    sn["dz"] = [W(lay, a, b) for a, b in zip(dz2, sn["dz"])]
    sn, qsnbot = snowh2o(P, o, dt, qsnfro, qsnsub, qrain, sn)
    isnow = sn["isnow"]
    ice, liq, stc3, dz3 = sn["ice"], sn["liq"], sn["stc"], sn["dz"]
    for p in range(3):
        un = p < isnow + 3
        ice[p] = W(un, 0., ice[p]); liq[p] = W(un, 0., liq[p]); stc3[p] = W(un, 0., stc3[p]); dz3[p] = W(un, 0., dz3[p])
    sneqv = sn["sneqv"]
    big = sneqv > sndpth_max
    bdsnow = ice[2] / jnp.fmax(dz3[2], 1e-30)
    snoflow_x = sneqv - sndpth_max
    ice[2] = W(big, ice[2] - snoflow_x, ice[2])
    dz3[2] = W(big, dz3[2] - snoflow_x / bdsnow, dz3[2])
    snoflow = W(big, snoflow_x / dt, 0.)
    lay = isnow != 0
    s_new = f32(0.)
    for p in range(3):
        a = isnow + 3 <= p
        s_new = s_new + W(a, ice[p] + liq[p], 0.)
    sneqv = W(lay, s_new, sneqv)
    dzf = [None] * NL
    for p in range(3):
        a = isnow + 3 <= p
        dzf[p] = W(a, -dz3[p], 0.)
    dzf[3] = zsoil[0]
    for k in range(1, NS):
        dzf[k + 3] = zsoil[k] - zsoil[k - 1]
    zs = [None] * NL
    for p in range(NL):
        first = p == isnow + 3
        prev = zs[p - 1] if p > 0 else f32(0.)
        zs[p] = W(first, dzf[p], W(isnow + 3 < p, prev + dzf[p], 0.))
    dzout = [W(isnow + 3 <= p, -dzf[p], 0.) for p in range(NL)]
    return dict(isnow=isnow, snowh=sn["snowh"], sneqv=sneqv, snice=ice, snliq=liq, stc=list(stc3) + stc[3:], zsnso=zs,
                dzsnso=dzout, sh2o1=sn["sh2o1"], sice1=sn["sice1"], qsnbot=qsnbot, snoflow=snoflow,
                ponding1=sn["ponding1"], ponding2=sn["ponding2"])


def sflx_glacier(P, x, o):
    """One Noah-MP land-ice step (NOAHMP_GLACIER). x: inputs/states as in the dump; o: options (alb, snf, tbot, stc,
    gla, sndpth)."""
    dt = x["DT"]
    sfctmp, sfcprs, uu, vv, q2 = x["T_ML"], x["P_ML"], x["U_ML"], x["V_ML"], x["Q_ML"]
    lwdn, cosz, zref = x["LWDN"], x["COSZ"], x["Z_ML"]
    zsoil = x["ZSOIL"]
    isnow = x["ISNOW"]
    pair = sfcprs
    thair = sfctmp * jnp.power(sfcprs / pair, RAIR / CPAIR)
    qair = q2
    eair = qair * sfcprs / (0.622 + 0.378 * qair)
    rhoair = (sfcprs - 0.378 * eair) / (RAIR * sfctmp)
    swdown = W(cosz <= 0., 0., x["SWDN"])
    solad = [swdown * 0.7 * 0.5, swdown * 0.7 * 0.5]
    solai = [swdown * 0.3 * 0.5, swdown * 0.3 * 0.5]
    zsnso = list(x["ZSNSO"])
    dzsnso = []
    for p in range(NL):
        a = isnow + 3 <= p
        first = p == isnow + 3
        prev = zsnso[p - 1] if p > 0 else f32(0.)
        dzsnso.append(W(a, W(first, -zsnso[p], prev - zsnso[p]), 0.))
    snowh, sneqv = x["SNDPTH"], x["SWE"]
    smc, sh2o = list(x["SMC"]), list(x["SMH2O"])
    stc = list(x["STC"]); snice = list(x["SNICE"]); snliq = list(x["SNLIQ"])
    ur = jnp.fmax(jnp.sqrt(uu ** 2. + vv ** 2.), 1.)
    z0mg = 0.002
    zpd = snowh
    zlvl = zpd + zref
    df, hcpct, fact = thermoprop(isnow, dt, snowh, snice, snliq, dzsnso)
    sag, fsr, fsa, tauss, albold = radiation(o, dt, x["TG"], x["SNEQVO"], sneqv, cosz, x["QSNOW"], solad, solai, x["ALBOLD"], x["TAUSS"])
    emg = 0.98
    lathea = f32(HSUB)
    gamma = CPAIR * sfcprs / (0.622 * lathea)
    pf = isnow + 3
    df1 = nm.at(df, pf); dz1 = nm.at(dzsnso, pf); stc1 = nm.at(stc, pf)
    oo = dict(o); oo["gla"] = o["gla"]
    GF = glacier_flux(oo, emg, isnow, df1, dz1, z0mg, zlvl, zpd, qair, sfctmp, rhoair, sfcprs, ur, gamma, 1.0, lwdn, 1.0,
                      smc, eair, stc1, sag, snowh, lathea, sh2o, x["CM"], x["CH"], x["TG"])
    fira, fsh, fgev, ssoil = GF["fira"], GF["fsh"], GF["fgev"], GF["ssoil"]
    fire = lwdn + fira
    emissi = f32(emg)
    trad = jnp.power((fire - (1 - emissi) * lwdn) / (emissi * SB), 0.25)
    stc_n = nm.tsnosoi(P, o, isnow, x["TBOT"], zsnso, ssoil, df, hcpct, dt, snowh, stc)
    pc = phasechange(o, isnow, dt, fact, dzsnso, stc_n, snice, snliq, sneqv, snowh, smc, sh2o)
    stc, snice, snliq, sneqv, snowh, smc, sh2o, qmelt, imelt, ponding = pc
    tg = GF["tg"]
    sice = [jnp.fmax(0.0, smc[k] - sh2o[k]) for k in range(NS)]
    sneqvo = sneqv
    qvap = jnp.fmax(fgev / lathea, 0.)
    qdew = jnp.abs(jnp.fmin(fgev / lathea, 0.))
    edir = qvap - qdew
    # ---------------------------------------------------- WATER_GLACIER
    prcp = x["PRCPNONC"] + x["PRCPCONV"] + x["PRCPSHCV"]
    sice_save = list(sice); sh2o_save = list(sh2o)
    if o["snf"] in (1, 4):
        fpice = W(sfctmp > TFRZ + 2.5, 0., W(sfctmp <= TFRZ + 0.5, 1.0, W(sfctmp <= TFRZ + 2., 1. - (-54.632 + 0.2 * sfctmp), 0.6)))
    elif o["snf"] == 2:
        fpice = W(sfctmp >= TFRZ + 2.2, 0., 1.0)
    else:
        fpice = W(sfctmp >= TFRZ, 0., 1.0)
    bdfall = jnp.fmin(120., 67.92 + 51.25 * jnp.exp((sfctmp - TFRZ) / 2.59))
    qrain = prcp * (1. - fpice)
    qsnow = prcp * fpice
    snowhin = qsnow / bdfall
    sw = snowwater(P, o, dt, zsoil, sfctmp, snowhin, qsnow, qdew, qvap, qrain, x["FICEOLD"], imelt, isnow, snowh, sneqv,
                   snice, snliq, sh2o[0], sice[0], stc, zsnso, dzsnso, f32(o["sndpth"]))
    isnow = sw["isnow"]; snowh = sw["snowh"]; sneqv = sw["sneqv"]; snice = sw["snice"]; snliq = sw["snliq"]
    stc = sw["stc"]; zsnso = sw["zsnso"]; dzsnso = sw["dzsnso"]
    sh2o[0] = sw["sh2o1"]; sice[0] = sw["sice1"]
    runsrf = (ponding + sw["ponding1"] + sw["ponding2"]) / dt
    runsrf = runsrf + W(isnow == 0, sw["qsnbot"] + qrain, sw["qsnbot"])
    replace = f32(0.)
    for k in range(NS):
        replace = replace + dzsnso[k + 3] * (sice[k] - sice_save[k] + sh2o[k] - sh2o_save[k])
    replace = replace * 1000.0 / dt
    sice = [jnp.fmin(1.0, sice_save[k]) for k in range(NS)]
    sh2o = [1.0 - sice[k] for k in range(NS)]
    runsub = sw["snoflow"] + replace
    clean = (snowh <= 1.E-6) | (sneqv <= 1.E-3)
    snowh = W(clean, 0.0, snowh); sneqv = W(clean, 0.0, sneqv)
    albedo_ = W(swdown != 0., fsr / jnp.where(swdown != 0., swdown, 1.0), -999.9)
    out = dict(ALBOLD=albold, SNEQVO=sneqvo, TG=tg, QSFC1D=GF["qsfc"], QSNOW=qsnow, ISNOW=isnow, SNDPTH=snowh, SWE=sneqv,
               CM=GF["cm"], CH=GF["ch"], TAUSS=tauss, FSA=fsa, FSR=fsr, FIRA=fira, FSH=fsh, FGEV=fgev, SSOIL=ssoil,
               TRAD=trad, ESOIL=edir, RUNSF=runsrf, RUNSB=runsub, SAG=sag, SALB=albedo_, QSNBOT=sw["qsnbot"], PONDING=ponding,
               PONDING1=sw["ponding1"], PONDING2=sw["ponding2"], T2MB=GF["t2m"], Q2MB=GF["q2e"], EMISSI=emissi, FPICE=fpice,
               CHB2=GF["ch2b"], SUBSNOW=qvap, STC=stc, SMH2O=sh2o, SMC=smc, ZSNSO=zsnso, SNICE=snice, SNLIQ=snliq)
    return out
