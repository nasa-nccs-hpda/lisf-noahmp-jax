"""Catchment Land Surface Model, Fortuna 2.5 (CLSM F2.5) of LIS in JAX: the core `clsmf25` (clsmf25_model.F90) with its
helpers, the Louis surface layer (clsmf25_turb/surfacelayer, DragCoefficients), PMONTH vegetation parameters and the driver
`clsm_step` (clsmf25_main.F90).  Everything is float32 and vectorised over catchment tiles; Fortran branches become
masks (jnp.where).  jnp.fmin/fmax replace MIN/MAX/AMIN1/AMAX1 so that NaN handling matches gfortran.
Arrays are (N,) per tile, (3,N) for snow layers and (6,N) for the ground-heat layers."""
import numpy as np
import jax
import jax.numpy as jnp
from ..driver import clsm_tables as TB

f32 = jnp.float32
n32 = np.float32
W = jnp.where
fmin, fmax = jnp.fmin, jnp.fmax

# ---- constants (clsmf25_MAPL_constants / clsmf25_constants) --------------------------------------------------------
ALHE = n32(2.4665e6); ALHM = n32(3.3370e5); ALHS = n32(ALHE + ALHM)
TF = n32(273.16); STEFAN = n32(5.6734e-8)
AIRMW = n32(28.97); H2OMW = n32(18.01)
RGAS = n32(n32(8314.3) / AIRMW); CPAIR = n32(RGAS / n32(n32(2.0) / n32(7.0)))
SHW = n32(4218.); SHI = n32(2000.); SHR = n32(2400.); RHOW = n32(1000.)
GRAV = n32(9.80); EPSILON = n32(H2OMW / AIRMW); VIREPS = n32(AIRMW / H2OMW - n32(1.0))
KARMAN = n32(0.40); USMIN = n32(1.00); NUAIR = n32(1.533e-5)
PIE = n32(3.14159265358979323846)
SCONST = n32(n32(1.9e6) / n32(920.))
CSOIL_1 = n32(70000.); CSOIL_2 = n32(200.)
RHOFS = n32(150.); RHOMA = n32(500.); WEMIN = n32(26.); MINSWE = n32(0.013); DZ1MAX = n32(0.08)
FWETL = n32(0.02); FWETC = n32(0.02); SATCAPFR = n32(0.2)
DZGT = [n32(x) for x in (0.0988, 0.1952, 0.3859, 0.7626, 1.5071, 10.0)]
FSN = n32(n32(1.e3) * ALHM)
CPW = n32(2065.22)
N_SNOW, N_GT = 3, 6



def nofma(p, ref):
    """Return p unchanged but opaque to the compiler, so that a following add cannot be fused with the multiply that produced p into a
    fused multiply-add (the Fortran rounds the product first). The integer xor uses a zero the compiler cannot prove: 0 if ref is not NaN."""
    z = (ref != ref).astype(jnp.int32)
    return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(p, jnp.int32) ^ z, jnp.float32)

def _t(a):
    return jnp.asarray(np.asarray(a, np.float32))


# ---- small helpers --------------------------------------------------------------------------------------------------
def sign(a, b):
    """Fortran SIGN(a, b): |a| with the sign of b (b = +0 counts as positive)."""
    return W(b >= 0, jnp.abs(a), -jnp.abs(a))


def get_tf0d(h, w):
    """returns t, f, ice1, tzero for one snow layer."""
    tfac = 1. / CPW; ffac = 1. / ALHM
    hbw = W(w > 0., h / jnp.where(w > 0., w, 1.), 0.)
    c1 = hbw < -1.00001 * ALHM
    c2 = hbw > -0.99999 * ALHM
    t = W(c1, (hbw + ALHM) * tfac, 0.)
    f = W(c1, 1., W(c2, -hbw * ffac, 1.))
    ice1 = W(c1, True, W(c2, False, True))
    tzero = W(c1, False, True)
    neg = f < 0.
    t = W(neg, hbw * tfac, t); f = W(neg, 0., f)
    t = W(w == 0., 0., t); f = W(w == 0., 0., f)
    return t, f, ice1, tzero


def get_tf(h, w):
    """get_tf_nd for one layer (array)."""
    tfac = 1. / CPW; ffac = 1. / ALHM
    hbw = W(w > 0., h / jnp.where(w > 0., w, 1.), 0.)
    lt = hbw < -ALHM
    gt = hbw > -ALHM
    t = W(lt, (hbw + ALHM) * tfac, 0.)
    f = W(lt, 1., W(gt, -hbw * ffac, 1.))
    neg = f < 0.
    t = W(neg, hbw * tfac, t); f = W(neg, 0., f)
    t = W(w == 0., 0., t); f = W(w == 0., 0., f)
    return t, f


def get_tf_nd(h, w):
    out = [get_tf(h[i], w[i]) for i in range(len(h))]
    return [o[0] for o in out], [o[1] for o in out]


def relayer(htsnn, wesn, sndz):
    """htsnn, wesn, sndz: lists of 3 arrays. Returns new lists."""
    small = 1.e-20
    areasc = fmin(sum_list(wesn) / WEMIN, 1.)
    h = [[0. * wesn[0], 0. * wesn[0]] for _ in range(N_SNOW + 1)]
    for i in range(N_SNOW):
        ok = sndz[i] > 0.
        den = W(ok, sndz[i], 1.)
        h[i][0] = W(ok, htsnn[i] / den, 0.)
        h[i][1] = W(ok, wesn[i] / den, 0.)
    sdold = [None] * (N_SNOW + 1)
    sdold[N_SNOW] = 0. * wesn[0]
    for i in range(N_SNOW - 1, -1, -1):
        sdold[i] = sdold[i + 1] + sndz[i]
    new = sdold[0] / float(N_SNOW)
    sn = [new for _ in range(N_SNOW)]
    big = new > DZ1MAX
    rest = (sdold[0] - DZ1MAX) / float(N_SNOW - 1)
    sn = [W(big, DZ1MAX, sn[0])] + [W(big, rest, sn[i]) for i in range(1, N_SNOW)]
    sdnew = [None] * (N_SNOW + 1)
    sdnew[N_SNOW] = 0. * wesn[0]
    for i in range(N_SNOW - 1, -1, -1):
        sdnew[i] = sdnew[i + 1] + sn[i]
    s = [[0. * wesn[0], 0. * wesn[0]] for _ in range(N_SNOW + 1)]
    ds = [None] * N_SNOW
    for i in range(N_SNOW):
        dz = sdnew[i + 1] - sdold[i + 1]
        use_next = dz < 0.
        for c in range(2):
            hj = W(use_next, h[i + 1][c], h[i][c]) if i + 1 < N_SNOW + 1 else h[i][c]
            s[i + 1][c] = hj * dz
        ds[i] = [s[i][c] - s[i + 1][c] for c in range(2)]
    htn = [htsnn[i] + ds[i][0] for i in range(N_SNOW)]
    wen = [wesn[i] + ds[i][1] for i in range(N_SNOW)]
    low = sum_list(wen) < WEMIN
    sn = [W(low, sn[i] / (areasc + small), sn[i]) for i in range(N_SNOW)]
    return htn, wen, sn


def sum_list(a):
    s = a[0]
    for x in a[1:]:
        s = s + x
    return s


def trid(dd, d, rd, b):
    """TRID for N = 3 (lists of arrays); returns x."""
    n = len(d)
    d = list(d); b = list(b)
    rsf = 0. * d[0]
    for i in range(2, n + 1):
        j = n + 1 - i                      # 1-based
        j0 = j - 1
        nz = d[j0 + 1] != 0.
        rsf = W(nz, rd[j0] / W(nz, d[j0 + 1], 1.), rsf)
        d[j0] = d[j0] - dd[j0 + 1] * rsf
        b[j0] = b[j0] - b[j0 + 1] * rsf
    x = [None] * n
    x[0] = W(d[0] != 0., b[0] / W(d[0] != 0., d[0], 1.), 0.)
    for j in range(1, n):
        nz = d[j] != 0.
        x[j] = W(nz, (b[j] - dd[j] * x[j - 1]) / W(nz, d[j], 1.), x[j - 1])
    return x


# ---- soil moisture / water balance routines -----------------------------------------------------------------------------
def rzequil(catdef, vgwmax, cdcr1, cdcr2, wpwet, ars, ara, arw):
    wilt = wpwet
    catdefx = fmin(catdef, cdcr1)
    cdi = W(ara[0] != ara[2], (ara[3] - ara[1]) / W(ara[0] != ara[2], ara[0] - ara[2], 1.), 0.)
    ax = W(catdefx >= cdi, ara[2] * catdefx + ara[3], ara[0] * catdefx + ara[1])
    wmin = fmin(1., fmax(0., arw[3] + (1. - arw[3]) * (1. + arw[0] * catdefx) /
                          (1. + arw[1] * catdefx + arw[2] * catdefx * catdefx)))
    arg1 = fmax(-40., fmin(40., -ax * (1. - wmin)))
    exparg1 = jnp.exp(arg1)
    rzeq = (wmin - 1. - (2. / ax)) * exparg1 + wmin + (2. / ax)
    factor = (cdcr2 - catdef) / (cdcr2 - cdcr1)
    rzeq = W(catdef > cdcr1, wilt + (rzeq - wilt) * factor, rzeq)
    rzeq = fmin(1., fmax(0., rzeq))
    return rzeq * vgwmax


def partition(dzsf, rzexc, rzeq, vgwmax, cdcr1, cdcr2, psis, bee, poros, wpwet, ars, ara, arw, srfexc, catdef, runsrf, dtstep):
    """returns dict(ar1, ar2, ar4, srfmx, srfmn, swsrf1, swsrf2, swsrf4, rzi, srfexc, catdef, runsrf)."""
    wilt = wpwet
    wrz = rzexc / vgwmax
    catdefx = fmin(catdef, cdcr1)
    cdi = W(ara[0] != ara[2], (ara[3] - ara[1]) / W(ara[0] != ara[2], ara[0] - ara[2], 1.), 0.)
    ar1 = fmin(1., fmax(0., (1. + ars[0] * catdefx) / (1. + ars[1] * catdefx + ars[2] * catdefx * catdefx)))
    ax = W(catdefx >= cdi, ara[2] * catdefx + ara[3], ara[0] * catdefx + ara[1])
    wmin = fmin(1., fmax(0., arw[3] + (1. - arw[3]) * (1. + arw[0] * catdefx) /
                          (1. + arw[1] * catdefx + arw[2] * catdefx * catdefx)))
    arg1 = fmax(-40., fmin(40., -ax * (1. - wmin)))
    exparg1 = jnp.exp(arg1)
    rzeqx = (wmin - 1. - (2. / ax)) * exparg1 + wmin + (2. / ax)
    rzeqxi = ax * exparg1 * (-1. - (2. / ax) - (2. / (ax * ax)) + wmin + (wmin / ax)) + wmin + 2. / ax
    ar20 = 1. + (-ax - 1. + ax * wmin) * exparg1
    rzeqxi = rzeqxi / (ar20 + 1.E-20)
    arg2 = fmax(-40., fmin(40., -ax * (1. - wilt)))
    exparg2 = jnp.exp(arg2)
    rzeqw = (wilt - 1. - (2. / ax)) * exparg2 + wilt + (2. / ax)
    rzeqwi = ax * exparg2 * (-1. - (2. / ax) - (2. / (ax * ax)) + wilt + (wilt / ax)) + wilt + 2. / ax
    ar20 = 1. + (-ax - 1. + ax * wilt) * exparg2
    rzeqwi = rzeqwi / (ar20 + 1.E-20)
    le = catdef <= cdcr1
    # --- CATDEF <= CDCR1
    rzeqy_a = rzeqx + wrz
    rzeqyi_a = rzeqxi + wrz
    wmnew = wmin + wrz
    exparg3 = jnp.exp(fmax(-40., fmin(40., -ax * (1. - wmnew))))
    area1 = (1. + ax - ax * wmin) * exparg1
    area2 = (1. + ax - ax * wmnew) * exparg3
    hi = wmnew >= wilt
    ar1_h = ar1 + area2 - area1
    ar2_h = 1. - ar1_h
    area3 = (1. + ax - ax * wilt) * exparg2
    ar1_l = ar1 + area3 - area1
    ar2_l = 1. - ar1_l
    fac_l = (rzeqx + wrz - wilt) / (rzeqw - wilt)
    ar1_l, ar2_l = ar1_l * fac_l, ar2_l * fac_l
    ar4_l = 1. - fac_l
    ar1_a = W(hi, ar1_h, ar1_l); ar2_a = W(hi, ar2_h, ar2_l); ar4_a = W(hi, 0., ar4_l)
    # --- CATDEF > CDCR1
    factor = (cdcr2 - catdef) / (cdcr2 - cdcr1)
    rzeqy_b = wilt + (rzeqx - wilt) * factor + wrz
    rzeqyi_b = wilt + (rzeqxi - wilt) * factor + wrz
    low = rzeqy_b < wilt
    rzeqy_b = W(low, wilt, rzeqy_b); rzeqyi_b = W(low, wilt, rzeqyi_b)
    # B1
    b1 = rzeqy_b >= rzeqx
    wmnew1 = wmin + (rzeqy_b - rzeqx)
    exparg3b = jnp.exp(fmax(-40., fmin(40., -ax * (1. - wmnew1))))
    area2b = (1. + ax - ax * wmnew1) * exparg3b
    ar1_b1 = ar1 + (area2b - area1)
    # B2/B3 share catdefw
    catdefw = cdcr2 + ((rzeqw - wilt) / (rzeqx - wilt)) * (cdcr1 - cdcr2)
    ar1w = fmin(1., fmax(0., (1. + ars[0] * catdefw) / (1. + ars[1] * catdefw + ars[2] * catdefw * catdefw)))
    b2 = (~b1) & (rzeqy_b >= rzeqw)
    fac2 = (rzeqy_b - rzeqw) / (rzeqx - rzeqw)
    ar1_b2 = ar1w + fac2 * (ar1 - ar1w)
    b3 = (~b1) & (~b2)
    fac3 = (rzeqy_b - wilt) / (rzeqw - wilt)
    ar1_b3 = ar1w * fac3
    ar2_b3 = (1. - ar1w) * fac3
    ar4_b3 = 1. - fac3
    ar1_b = W(b1, ar1_b1, W(b2, ar1_b2, ar1_b3))
    ar2_b = W(b3, ar2_b3, 1. - ar1_b)
    ar4_b = W(b3, ar4_b3, 0.)
    ar1 = W(le, ar1_a, ar1_b); ar2 = W(le, ar2_a, ar2_b); ar4 = W(le, ar4_a, ar4_b)
    rzeqyi = W(le, rzeqyi_a, rzeqyi_b)
    swsrf1 = 1. + 0. * ar1
    swsrf2 = fmin(1., fmax(0.01, rzeqyi))
    swsrf4 = fmin(1., fmax(0.01, wilt))
    swsrf2 = jnp.power(jnp.power(swsrf2, -bee) - (.5 / psis), -1. / bee)
    swsrf4 = jnp.power(jnp.power(swsrf4, -bee) - (.5 / psis), -1. / bee)
    srfmx = ar2 * (1. - swsrf2) * (dzsf * poros)
    srfmx = srfmx + ar4 * (1. - swsrf4) * (dzsf * poros)
    srfmn = -(ar2 * swsrf2 + ar4 * swsrf4) * (dzsf * poros)
    over = srfexc > srfmx
    under = (~over) & (srfexc < srfmn)
    cor = W(over, srfexc - srfmx, W(under, srfexc - srfmn, 0.))
    srfexc = W(over, srfmx, W(under, srfmn, srfexc))
    catdef = catdef - cor
    neg = over & (catdef < 0.)
    runsrf = W(neg, runsrf - catdef / dtstep, runsrf)
    catdef = W(neg, 0., catdef)
    swsrf2 = swsrf2 + srfexc / (dzsf * poros * (1. - ar1) + 1.e-20)
    swsrf2 = fmin(1., fmax(1.E-5, swsrf2))
    swsrf4 = swsrf4 + srfexc / (dzsf * poros * (1. - ar1) + 1.e-20)
    swsrf4 = fmin(1., fmax(1.E-5, swsrf4))
    full = ar1 >= 1. - 1.E-5
    ar1 = W(full, 1., ar1); ar2 = W(full, 0., ar2); ar4 = W(full, 0., ar4)
    swsrf2 = W(full, 1., swsrf2); swsrf4 = W(full, wilt, swsrf4)
    ar1 = W(ar1 < 0., 0., ar1)
    ar1 = fmax(0., fmin(1., ar1)); ar2 = fmax(0., fmin(1., ar2)); ar4 = fmax(0., fmin(1., ar4))
    return dict(ar1=ar1, ar2=ar2, ar4=ar4, srfmx=srfmx, srfmn=srfmn, swsrf1=swsrf1, swsrf2=swsrf2, swsrf4=swsrf4,
                rzi=rzeqyi, srfexc=srfexc, catdef=catdef, runsrf=runsrf)


def calc_soil_moist(dzsf, vgwmax, cdcr1, cdcr2, wpwet, poros, psis, bee, ars, ara, arw, srfexc, rzexc, catdef):
    """returns sfmc, rzmc, prmc, werror and the (modified) srfexc, rzexc, catdef (side effects of the Fortran call)."""
    prmc_orig = cdcr2 / (1. - wpwet) - catdef + rzexc + srfexc
    catdef = fmax(1., fmin(cdcr2, catdef))
    rzeq = rzequil(catdef, vgwmax, cdcr1, cdcr2, wpwet, ars, ara, arw)
    rzexc = fmax(wpwet * vgwmax - rzeq, fmin(vgwmax - rzeq, rzexc))
    pt = partition(dzsf, rzexc, rzeq, vgwmax, cdcr1, cdcr2, psis, bee, poros, wpwet, ars, ara, arw, srfexc, catdef,
                   0. * catdef, f32(-9999.0))
    srfexc = pt["srfexc"]
    sfmc = poros * (pt["swsrf1"] * pt["ar1"] + pt["swsrf2"] * pt["ar2"] + pt["swsrf4"] * pt["ar4"])
    rzmc = (rzeq + rzexc + srfexc) * poros / vgwmax
    prmc = cdcr2 / (1. - wpwet) - catdef + rzexc + srfexc
    werror = prmc - prmc_orig
    prmc = prmc * poros / (cdcr2 / (1. - wpwet))
    return dict(sfmc=sfmc, rzmc=rzmc, prmc=prmc, werror=werror, srfexc=srfexc, rzexc=rzexc, catdef=catdef, rzeq=rzeq)


def wupdat(dtstep, evap, satcap, ar1, ar2, ar4, cdcr1, eirfrc, rzeq, srfmn, wpwet, vgwmax, rx11, rx21, rx12, rx22, rx14, rx24,
           capac, rzexc, catdef, srfexc):
    cndsat = (ar1 / rx11) + (ar1 / rx21)
    cnduns = (ar2 / rx12) + (ar2 / rx22) + (ar4 / rx14) + (ar4 / rx24)
    esatfr = cndsat / (cndsat + cnduns)
    eint = eirfrc * evap * dtstep
    egro = evap * dtstep - eint
    gt = eint > capac
    egro = W(gt, egro + eint - capac, egro)
    eint = W(gt, capac, eint)
    wilt = wpwet * vgwmax
    rzemax = fmax(0., rzexc + rzeq - wilt)
    egromx = rzemax + (srfexc - srfmn)
    gt = egro > egromx
    eint = W(gt, fmin(capac, eint + egro - egromx), eint)
    egro = W(gt, egromx, egro)
    esoi = fmin(srfexc - srfmn, egro * (ar1 * rx11 / (rx11 + rx21 + 1.E-20) + ar2 * rx12 / (rx12 + rx22 + 1.E-20)
                                          + ar4 * rx14 / (rx14 + rx24 + 1.E-20)))
    eveg = egro - esoi
    c = esoi > srfexc - srfmn
    esoi = W(c, srfexc - srfmn, esoi)
    eveg = W(c, egro - esoi, eveg)
    c = eveg > rzemax
    eveg = W(c, rzemax, eveg)
    esoi = W(c, egro - eveg, esoi)
    neg = evap < 0.
    eint = W(neg, fmin(0., evap * dtstep), eint)
    esoi = W(neg, 0., esoi); eveg = W(neg, 0., eveg)
    wet = catdef < cdcr1
    capac_n = fmax(0., capac - eint)
    rzexc_n = W(wet, rzexc - eveg * (1. - esatfr), rzexc - eveg)
    srfexc_n = W(wet, srfexc - esoi * (1. - esatfr), srfexc - esoi)
    catdef_n = W(wet, catdef + (esoi + eveg) * esatfr, catdef)
    return dict(capac=capac_n, rzexc=rzexc_n, catdef=catdef_n, srfexc=srfexc_n, eint=eint, esoi=esoi, eveg=eveg)


def rzdrain(dtstep, vgwmax, satcap, rzeq, ar1, wpwet, tsa1, tsa2, tsb1, tsb2, atau, btau, cdcr2, poros, srfexc, capac, rzexc,
            catdef, runsrf):
    rztot = rzeq + rzexc
    ok = ar1 != 1.
    rzave = (rztot - ar1 * vgwmax) / W(ok, 1. - ar1, 1.)
    rzave = rzave * poros / vgwmax
    rzave = W(ok, rzave, poros)
    rzave = W(rzave == 0., 1.e-4, rzave)
    btaux = W(srfexc < 0., btau * (poros / rzave), btau)
    rz0 = fmax(0.001, rzave - srfexc / (1000. * (-btaux)))
    tsc0 = atau / jnp.power(rz0, 3.)
    tsc0 = tsc0 * 3600.
    tsc0 = W(tsc0 < dtstep, dtstep, tsc0)
    srflw = srfexc * dtstep / tsc0
    rzdif = rzave / poros - wpwet
    srflw = W((rzdif <= 0.) & (srflw < 0.), 0., srflw)
    srflw = W((rzdif > 0.) & (rzdif < 0.01) & (srflw < 0.), srflw * (rzdif / 0.01), srflw)
    rzexc = rzexc + srflw
    srfexc = srfexc - srflw
    rzx = rzexc / vgwmax
    ax = W(rzx > .01, tsa1, W(rzx < -.01, tsa2, tsa2 + (rzx + .01) * (tsa1 - tsa2) / .02))
    bx = W(rzx > .01, tsb1, W(rzx < -.01, tsb2, tsb2 + (rzx + .01) * (tsb1 - tsb2) / .02))
    tsc2 = jnp.exp(ax + bx * catdef)
    rzflw = rzexc * tsc2 * dtstep / 3600.
    rzflw = W(catdef - rzflw > cdcr2, catdef - cdcr2, rzflw)
    catdef = catdef - rzflw
    rzexc = rzexc - rzflw
    c = capac > satcap
    rzexc = W(c, rzexc + capac - satcap, rzexc)
    capac = W(c, satcap, capac)
    c = rzeq + rzexc > vgwmax
    excess = rzeq + rzexc - vgwmax
    rzexc = W(c, vgwmax - rzeq, rzexc)
    catdef = W(c, catdef - excess, catdef)
    c = catdef < 0.
    runsrf = W(c, runsrf - catdef, runsrf)
    catdef = W(c, 0., catdef)
    return dict(capac=capac, rzexc=rzexc, srfexc=srfexc, catdef=catdef, runsrf=runsrf)


def base_flow(dtstep, bf1, bf2, bf3, cdcr1, frice, cond, gnu, catdef):
    zbar = jnp.sqrt(1.e-20 + catdef / bf1) - bf2
    bflow = (1. - frice) * cond * jnp.exp(-(bf3 - 0.) - gnu * zbar) / gnu
    bflow = W(catdef >= cdcr1, 0., bflow)
    bflow = fmin(cond, bflow)
    bflow = bflow * 1000.0
    return bflow, catdef + bflow * dtstep


def interc(dtstep, trainl, trainc, smelt, satcap, sfrac, capac):
    timfrl, timfrc = 1.00, 0.333
    xtcorr = (1. - timfrl) * fmin(1., (capac / satcap) / (FWETL * sfrac))
    watadd = trainl * dtstep + smelt * dtstep
    cavail = (satcap - capac) * (FWETL * sfrac)
    wetint = capac / satcap
    thru1 = W(watadd * (1. - wetint) < cavail, watadd * wetint, watadd - cavail)
    thru1 = thru1 * (1. - xtcorr)
    thru2 = xtcorr * watadd
    thrul = thru1 + thru2
    capac = capac + watadd - thru1 - thru2
    xtcorr = (1. - timfrc) * fmin(1., (capac / satcap) / (FWETC * sfrac))
    watadd = trainc * dtstep
    cavail = (satcap - capac) * (FWETC * sfrac)
    wetint = capac / satcap
    thru1 = W(watadd * (1. - wetint) < cavail, watadd * wetint, watadd - cavail)
    thru1 = thru1 * (1. - xtcorr)
    thru2 = xtcorr * watadd
    thruc = thru1 + thru2
    capac = capac + watadd - thru1 - thru2
    return fmax(0., thrul + thruc), capac


def srunoff(dtstep, ar1, ar2, ar4, thru, srfmx, srfexc, runsrf):
    ptotal = thru
    frun = ar1
    pos = srfexc > 0.
    r = srfexc / (srfmx + 1.e-20)
    frun = W(pos, frun + ar2 * jnp.power(r, 2.) + ar4 * jnp.power(r, 4.), frun)
    srun0 = ptotal * frun
    srun0 = W(ptotal - srun0 > srfmx - srfexc, ptotal - (srfmx - srfexc), srun0)
    srun0 = W(srun0 > ptotal, ptotal, srun0)
    runsrf = runsrf + srun0
    qin = ptotal - srun0
    srfexc = srfexc + qin
    return runsrf / dtstep, srfexc, qin / dtstep


# ---- vegetation / surface resistance helpers ------------------------------------------------------------------------------
_VGCHIL = _t([.1, .25, .01, -.3, .01, .2, 0., 0., 0., 0.])
_VGZMEW = _t([.9809, .9638, .9980, 1.0773, .9980, .9676, 1., 1., 1., 1.])
_VGRST1 = _t([2335.9, 9802.2, 2869.7, 2582.0, 93989.4, 9802.2, 0., 0., 0., 0.])
_VGRST2 = _t([0., 10.6, 3.7, 1.1, .01, 10.6, 0., 0., 0., 0.])
_VGRST3 = _t([153.5, 180., 233., 110., 855., 180., 1., 1., 1., 1.])
_VGTLL = _t([273., 273., 268., 283., 283., 273., 0., 0., 0., 0.])
_VGTU = _t([318., 318., 313., 328., 323., 323., 0., 0., 0., 0.])
_VGTCF1 = _t([-1.43549E-06, -6.83584E-07, 1.67699E-07, -1.43465E-06, -2.76097E-06, -1.58094E-07, 0., 0., 0., 0.])
_VGTCF2 = _t([7.95859E-04, 3.72064E-04, -7.65944E-05, 8.24060E-04, 1.57617E-03, 8.44847E-05, 0., 0., 0., 0.])
_VGTCF3 = _t([-1.11575E-01, -5.21533E-02, 6.14960E-03, -1.19602E-01, -2.26109E-01, -1.27272E-02, 0., 0., 0., 0.])


def rcunst(ityp, sunang, sqscat, pdir, par, zlai, green):
    k = ityp - 1
    chil = _VGCHIL[k]
    aa = 0.5 - (0.633 + 0.330 * chil) * chil
    bb = 0.877 * (1. - 2. * aa)
    cc = (aa + bb * sunang) / sunang
    extk1 = cc * sqscat
    extk2 = (1. / _VGZMEW[k]) * sqscat
    dum1 = pdir * cc
    dum2 = (1. - pdir) * (bb * (1. / 3. + PIE / 4.) + aa * 1.5)
    zk = pdir * fmin(extk1, 50. / zlai) + (1. - pdir) * fmin(extk2, 50. / zlai)
    gamma = _VGRST1[k] / _VGRST3[k] + _VGRST2[k]
    ekat = jnp.exp(zk * zlai)
    rho4 = gamma / (par * (dum1 + dum2))
    dum1 = (_VGRST2[k] - gamma) / (gamma + 1.E-20)
    dum2 = (rho4 * ekat + 1.) / (rho4 + 1.)
    dum3 = zk * _VGRST3[k]
    rcinv = (dum1 * jnp.log(dum2) + zk * zlai) / dum3
    rcinv = fmax(rcinv, 0.)
    return 1. / (rcinv * green + 1.E-10)


def tmpfac(ityp, tc):
    k = ityp - 1
    ll, uu = _VGTLL[k], _VGTU[k]
    ft = (tc - ll) * (tc - uu) * (_VGTCF1[k] * tc * tc + _VGTCF2[k] * tc + _VGTCF3[k])
    ft = W((tc <= ll) | (tc >= uu), 1.E-10, ft)
    return fmin(1., fmax(ft, 1.E-10))


def rcanop(ra, potfrc, rc):
    return rc * (1. - potfrc) / (1. + potfrc * rc / ra)


def rsurfp1(um, rdc, rc):
    rsurf = rdc / um
    return rc * rsurf / (rc + rsurf), rc, rsurf


def rsurfp2(um, rdc, wet, esattc, ea, wpwet, rc):
    rswilt, rssat = 500.0, 25.0
    bterm = (rswilt - rssat) / (1. / jnp.power(wpwet, 2.) - 1.)       # WPWET**2 (integer power)
    aterm = rssat - bterm
    rsurf = rdc / um + aterm + bterm / jnp.power(1.E-10 + wet, 2.)
    hesat = esattc * fmin(1., wet * 2.)
    rsurf = W(ea < hesat, rsurf * (1. + (esattc - hesat) / (hesat - ea)), 1.E10)
    return rc * rsurf / (rc + rsurf), rc, rsurf


def fluxes(dtstep, esattc, desdtc, eturb, dedea, dedtc, hsturb, dhsdea, dhsdtc, rc, drcdea, drcdtc, swnet, hlwdwn, alwrad,
           blwrad, em, csoil, psur, emaxrt, hftds, dhftds, tc, ea):
    hlwtc = alwrad + blwrad * tc
    rhoair = psur * 100. / (RGAS * tc)
    const = rhoair * EPSILON / psur
    dhlwtc = blwrad
    a11 = csoil / dtstep + dhlwtc + dhsdtc + ALHE * dedtc + dhftds
    a12 = dhsdea + ALHE * dedea
    q0 = swnet + hlwdwn - hlwtc - hsturb - ALHE * eturb - hftds
    pos = rc > 0.
    rcs = W(pos, rc, 1.)
    eplant = const * (esattc - ea) / rcs
    eharmn = W(eplant * eturb > 0., 2. * eplant * eturb / W(eplant * eturb > 0., eplant + eturb, 1.), 0.)
    a21p = -dedtc * rcs + fmax(0., const * desdtc - eharmn * drcdtc)
    a22p = -(rcs * dedea + fmax(0., const + eharmn * drcdea))
    f0p = rcs * (eturb - eplant)
    determ = fmin(a12 * a21p / (a11 * a22p) - 1., -0.1)
    dea_p = (q0 * a21p - a11 * f0p) / (determ * a11 * a22p)
    dtc_p = (q0 - a12 * dea_p) / a11
    # RC <= 0 branch
    a21n = -desdtc; a22n = 1.; f0n = esattc - ea
    dea_n = (q0 * a21n - a11 * f0n) / (a12 * a21n - a11 * a22n)
    dtc_n = (q0 - a12 * dea_n) / a11
    dea = W(pos, dea_p, dea_n)
    dtc = W(pos, dtc_p, dtc_n)
    evap = eturb + dedea * dea + dedtc * dtc
    shflux = hsturb + dhsdea * dea + dhsdtc * dtc
    choke = pos
    # EVAP > EMAXRT
    c = evap > emaxrt
    choke = W(c, False, choke)
    dea1 = sign(eturb, ea)
    dea1 = W(dedea != 0., (emaxrt - eturb) / W(dedea != 0., dedea, 1.), dea1)
    dea1 = em - ea
    dtc1 = (q0 + ALHE * (eturb - emaxrt) - dhsdea * dea1) / (a11 - ALHE * dedtc)
    dea = W(c, dea1, dea); dtc = W(c, dtc1, dtc)
    evap = W(c, emaxrt, evap)
    shflux = W(c, hsturb + dhsdea * dea + dhsdtc * dtc, shflux)
    # EVAP*(ETURB+DEDEA*DEA) < 0
    c = evap * (eturb + dedea * dea) < 0.
    choke = W(c, False, choke)
    dea2 = sign(eturb, ea)
    dea2 = W(dedea != 0., -eturb / W(dedea != 0., dedea, 1.), dea2)
    dtc2 = (q0 + ALHE * eturb - dhsdea * dea2) / (a11 - ALHE * dedtc)
    dea = W(c, dea2, dea); dtc = W(c, dtc2, dtc)
    evap = W(c, 0., evap)
    shflux = W(c, hsturb + dhsdea * dea + dhsdtc * dtc, shflux)
    # CHOKE and |DEA| > 0.5 EA
    c = choke & (jnp.abs(dea) > 0.5 * ea)
    dea3 = sign(.5 * ea, dea)
    dtc3 = (q0 - a12 * dea3) / a11
    evap3 = eturb + dedea * dea3 + dedtc * dtc3
    shflux3 = hsturb + dhsdea * dea3 + dhsdtc * dtc3
    cc = evap3 > emaxrt
    edif = evap3 - emaxrt
    shflux3 = W(cc, shflux3 + edif * ALHE, shflux3)
    evap3 = W(cc, emaxrt, evap3)
    dea = W(c, dea3, dea); dtc = W(c, dtc3, dtc)
    evap = W(c, evap3, evap); shflux = W(c, shflux3, shflux)
    tc_n = tc + dtc
    ea_n = ea + dea
    hlwup = hlwtc + dhlwtc * dtc
    ghflux = hftds + dhftds * dtc
    ea_n = fmax(ea_n, 0.0)
    return dict(evap=evap, shflux=shflux, hlwup=hlwup, ghflux=ghflux, tc=tc_n, ea=ea_n)


def _energy_common(qm, psur, qa, qsattc, dqsdtc, dedqa, dhsdqa):
    em = qm * psur / EPSILON
    ea = qa * psur / EPSILON
    esattc = qsattc * psur / EPSILON
    desdtc = dqsdtc * psur / EPSILON
    dedea = dedqa * EPSILON / psur
    dhsdea = dhsdqa * EPSILON / psur
    return em, ea, esattc, desdtc, dedea, dhsdea


def energy(kind, dtstep, ityp, um, rcin, eturb, dedqa, dedtc, hsturb, dhsdqa, dhsdtc, qm, ra, swnet, hlwdwn, psur, rdc, hftds,
           dhftds, qsattc, dqsdtc, alwrad, blwrad, emaxrt, csoil, swsrf, potfrc, tc, qa, rzi=None, wpwet=None):
    """kind 1 (saturated), 2 (unsaturated), 4 (wilting). returns dict(evap, shflux, hlwup, rx1, rx2, ghflux, tc, qa)."""
    deltc, delea = 0.01, 0.001
    em, ea, esattc, desdtc, dedea, dhsdea = _energy_common(qm, psur, qa, qsattc, dqsdtc, dedqa, dhsdqa)
    vpd = 1.0
    ftemp = tmpfac(ityp, tc)
    if kind == 4:
        rc = rcin
        rc, rx1, rx2 = rsurfp2(um, rdc, swsrf, esattc, ea, wpwet, rc)
        rc = rcanop(ra, potfrc, rc)
        drcdtc = 0. * rc; drcdea = 0. * rc
    else:
        rc = rcin / (vpd * ftemp + 1.E-20)
        if kind == 2:
            astrfr, stexp = 0.333, 2.0
            atrans = wpwet + astrfr * (1. - wpwet)
            rstfac = fmax(1.E-3, (rzi - wpwet) / (atrans - wpwet))
            rstfac = fmin(1., rstfac)
            rc = rc / jnp.power(rstfac, stexp)
            rc = fmin(rc, 1.E10)
            surf = lambda rcc, esat, eaa: rsurfp2(um, rdc, swsrf, esat, eaa, wpwet, rcc)
        else:
            surf = lambda rcc, esat, eaa: rsurfp1(um, rdc, rcc)
        rc, rx1, rx2 = surf(rc, esattc, ea)
        rc = rcanop(ra, potfrc, rc)
        tx = tc + deltc
        esattx = esattc + desdtc * deltc
        eax = ea + delea
        ftempx = tmpfac(ityp, tx)
        rcx = rcin / (vpd * ftempx + 1.E-20)
        if kind == 2:
            rcx = fmin(rcx / jnp.power(rstfac, stexp), 1.E10)
        rcx, _, _ = surf(rcx, esattx, ea)
        rcx = rcanop(ra, potfrc, rcx)
        drcdtc = (rcx - rc) / deltc
        rcx = rcin / (vpd * ftemp + 1.E-20)
        if kind == 2:
            rcx = fmin(rcx / jnp.power(rstfac, stexp), 1.E10)
        rcx, _, _ = surf(rcx, esattc, eax)
        rcx = rcanop(ra, potfrc, rcx)
        drcdea = (rcx - rc) / delea
    fl = fluxes(dtstep, esattc, desdtc, eturb, dedea, dedtc, hsturb, dhsdea, dhsdtc, rc, drcdea, drcdtc, swnet, hlwdwn,
                alwrad, blwrad, em, csoil, psur, emaxrt, hftds, dhftds, tc, ea)
    fl["qa"] = fl["ea"] * EPSILON / psur
    fl["rx1"] = rx1; fl["rx2"] = rx2
    return fl


# ---- ground heat: GNDTP0 / GNDTMP -------------------------------------------------------------------------------------
def _gt_props(phi, zbar, thetaf, ht, nlayer):
    """common to gndtp0 (first layer only) and gndtmp. returns tp[k], fice[k], xklh[k], zb, zc."""
    shw0 = SHW * 1000.; shi0 = SHI * 1000.; shr0 = SHR * 1000.
    zb = [-0.05]
    for l in range(N_GT):
        zb.append(zb[l] - DZGT[l])
    zc = [0.5 * (zb[l] + zb[l + 1]) for l in range(N_GT)]
    shc = [shr0 * (1. - phi) * DZGT[l] for l in range(N_GT)]
    tp, fice, xklh = [], [], []
    for k in range(nlayer):
        ws = phi * DZGT[k]
        xw = 0.5 * ws
        fk = fmax(0., fmin(1., -ht[k] / (FSN * xw)))
        t = W(fk == 1., (ht[k] + xw * FSN) / (shc[k] + xw * shi0), W(fk == 0., ht[k] / (shc[k] + xw * shw0), 0.))
        tp.append(t); fice.append(fk)
        a1 = 1. - phi
        tk1 = 1.01692 + a1 * (0.89865 + 1.06211 * a1)
        xw2 = phi * (1. - fk)
        a2 = phi - xw2
        tk2 = 1.00543 + a2 * (0.723371 + .464342 * a2)
        tk3 = 0.998899 + xw2 * (-0.548043 + 0.120291 * xw2)
        tksat = tk1 * tk2 * tk3
        xwi = 1.0 + 0. * phi
        xwi = W(zbar <= zb[k + 1], thetaf, xwi)
        mid = (zbar >= zb[k + 1]) & (zbar <= zb[k])
        xd1 = zb[k] - zbar
        xd2 = zbar - zb[k + 1]
        xwi = W(mid & (zbar > zb[k + 1]), ((xd1 * thetaf) + xd2) / (xd1 + xd2), xwi)
        xwi = fmin(xwi, 1.)
        tkdry = 0.226
        xklh.append((tksat - tkdry) * xwi + tkdry)
    return tp, fice, xklh, zb, zc


def gndtp0(t1, phi, zbar, thetaf, ht):
    tp, fice, xklh, zb, zc = _gt_props(phi, zbar, thetaf, ht, 1)
    # first-layer values only; xklhw uses tksat of the same layer
    a1 = 1. - phi
    tk1 = 1.01692 + a1 * (0.89865 + 1.06211 * a1)
    xw = phi * (1. - fice[0])
    a2 = phi - xw
    tk2 = 1.00543 + a2 * (0.723371 + .464342 * a2)
    tk3 = 0.998899 + xw * (-0.548043 + 0.120291 * xw)
    tksat = tk1 * tk2 * tk3
    xklhw = tksat
    denom = -.025 - zc[0]
    fh21w = -xklhw * (t1[0] - 273.16 - tp[0]) / denom
    fh21i = -xklh[0] * (t1[1] - 273.16 - tp[0]) / denom
    fh21d = -xklh[0] * (t1[2] - 273.16 - tp[0]) / denom
    dfh21w = -xklhw / denom
    dfh21i = -xklh[0] / denom
    return fh21w, fh21i, fh21d, dfh21w, dfh21i, dfh21i


def gndtmp(dts, phi, zbar, thetaf, fh21, ht):
    """returns new ht list, tp list, xfice, fice list."""
    shw0 = SHW * 1000.; shi0 = SHI * 1000.; shr0 = SHR * 1000.
    tp, fice, xklh, zb, zc = _gt_props(phi, zbar, thetaf, ht, N_GT)
    fh = [None] * (N_GT + 1)
    fh[N_GT] = 0. * phi
    fh[0] = fh21
    for k in range(1, N_GT):
        xkth = ((zb[k] - zc[k - 1]) * xklh[k - 1] + (zc[k] - zb[k]) * xklh[k]) / (zc[k] - zc[k - 1])
        fh[k] = -xkth * (tp[k - 1] - tp[k]) / (zc[k - 1] - zc[k])
    htn = [ht[k] + (fh[k + 1] - fh[k]) * dts for k in range(N_GT)]
    tpn, fice = [], []
    for k in range(N_GT):
        ws = phi * DZGT[k]
        xw = 0.5 * ws
        shc = shr0 * (1. - phi) * DZGT[k]
        fk = fmax(0., fmin(1., -htn[k] / (FSN * xw)))
        t = W(fk == 1., (htn[k] + xw * FSN) / (shc + xw * shi0), W(fk == 0., htn[k] / (shc + xw * shw0), 0.))
        tpn.append(t); fice.append(fk)
    lstart = jnp.full(phi.shape, N_GT, jnp.int32)
    for l in range(N_GT - 1, -1, -1):                      # DO L=N_GT,1,-1 ; last assignment = smallest L with zbar >= zb(l+1)
        lstart = W(zbar >= zb[l + 1], l + 1, lstart)
    xf = 0. * phi
    for l in range(N_GT):
        xf = xf + W(l + 1 >= lstart, fice[l], 0.)
    xfice = xf / ((N_GT + 1) - lstart).astype(f32)
    return htn, tpn, xfice, fice


# ---- snow ------------------------------------------------------------------------------------------------------------------
def snowrt(t1, area, tkgnd, precip, snowf, ts, dts, eturb, dedtc, hsturb, dhsdtc, hlwtc, dhlwtc, raddn, zc1, wesn_in, htsnn_in,
           sndz_in):
    snfr = 0.01
    small = 1.e-6
    wesn = list(wesn_in); htsnn = list(htsnn_in); sndz = list(sndz_in)
    zero = 0. * precip
    snowd0 = sum_list(wesn)
    snowin = snowd0
    tsx = fmin(ts - TF, 0.) * CPW
    rainf = precip - snowf
    nosnow = snowd0 <= MINSWE

    # ---------------- no snow at the start of the step ----------------
    areasc_ns = fmin(snowd0 / WEMIN, 1.)
    pre_ns = snowd0 / dts + areasc_ns * rainf
    hcorr_ns = raddn * areasc_ns + sum_list(htsnn) / dts
    sn_new = snowf > 0.
    w_ns = [W(sn_new, snowf * dts / float(N_SNOW), 0.) for _ in range(N_SNOW)]
    h_ns = [W(sn_new, (tsx - ALHM) * w_ns[i], 0.) for i in range(N_SNOW)]
    s_ns = [W(sn_new, w_ns[i] / RHOFS, 0.) for i in range(N_SNOW)]
    areasc0_ns = W(sn_new, fmin((snowf * dts) / WEMIN, 1.), 0.)
    hcorr_ns = W(sn_new, hcorr_ns - tsx * snowf, hcorr_ns)
    h_ns2, w_ns2, s_ns2 = relayer(h_ns, w_ns, s_ns)
    h_ns = [W(sn_new, h_ns2[i], 0.) for i in range(N_SNOW)]
    w_ns = [W(sn_new, w_ns2[i], 0.) for i in range(N_SNOW)]
    s_ns = [W(sn_new, s_ns2[i], 0.) for i in range(N_SNOW)]
    tp_ns, fi_ns = get_tf_nd(h_ns, w_ns)
    tp_ns = [W(sn_new, tp_ns[i], 0.) for i in range(N_SNOW)]
    fi_ns = [W(sn_new, fi_ns[i], 0.) for i in range(N_SNOW)]

    # ---------------- snow present ----------------
    areasc = fmin(snowd0 / WEMIN, 1.)
    dens = []
    for i in range(N_SNOW):
        ok = sndz[i] > 0.
        dens.append(W(ok, fmax(wesn[i] / (areasc * W(ok, sndz[i], 1.)), RHOFS), RHOFS))
    dens0 = list(dens)
    tksn = [3.2217e-06 * dens[i] * dens[i] for i in range(N_SNOW)]
    tpsn, fices = get_tf_nd(htsnn, wesn)
    denom = 1. / (sndz[N_SNOW - 1] * 0.5 - zc1)
    fhgnd = [-jnp.sqrt(tkgnd[i] * tksn[N_SNOW - 1]) * area[i] * denom * (tpsn[N_SNOW - 1] - t1[i]) for i in range(N_SNOW)]
    fhsn = [None] * (N_SNOW + 1)
    df = [None] * (N_SNOW + 1)
    fhsn[N_SNOW] = sum_list(fhgnd)
    dfn = zero
    for i in range(N_SNOW):
        dfn = dfn - jnp.sqrt(tkgnd[i] * tksn[N_SNOW - 1]) * area[i] * denom
    df[N_SNOW] = dfn
    t1ave = sum_list([t1[i] * area[i] for i in range(N_SNOW)]) / sum_list(area)
    htest = list(htsnn)
    htest[N_SNOW - 1] = htest[N_SNOW - 1] + fhsn[N_SNOW] * dts * areasc
    ttest, ftest = get_tf_nd(htest, wesn)
    scale = 1. + zero
    c = (t1ave - tpsn[-1]) * (t1ave - ttest[-1]) < 0.
    scale = W(c, 0.5 * (tpsn[-1] - t1ave) / (tpsn[-1] - ttest[-1]), scale)
    c = (tpsn[-1] == 0.) & (ttest[-1] > 0.) & (jnp.abs(fhsn[N_SNOW]) > 1.e-10)
    scale = W(c, (-0.5 * htsnn[-1] / (dts * areasc)) / W(c, fhsn[N_SNOW], 1.), scale)
    fhsn[N_SNOW] = fhsn[N_SNOW] * scale
    df[N_SNOW] = df[N_SNOW] * scale
    fhgnd = [f * scale for f in fhgnd]
    for i in range(1, N_SNOW):
        df[i] = -jnp.sqrt(tksn[i - 1] * tksn[i]) / ((sndz[i - 1] + sndz[i]) * 0.5)
        fhsn[i] = df[i] * (tpsn[i - 1] - tpsn[i])
    alhv = ALHE + ALHM
    lhturb = alhv * eturb
    dlhdtc = alhv * dedtc
    fhsn[0] = lhturb + hsturb + hlwtc - raddn
    df[0] = -(dlhdtc + dhsdtc + dhlwtc)
    ice1, tzero, cl, cd, cr, q = [], [], [], [], [], []
    for i in range(N_SNOW):
        _, _, ic, tz = get_tf0d(htsnn[i], wesn[i])
        ice1.append(ic); tzero.append(tz)
        cl.append(W(ic, df[i], 0.))
        cd.append(W(ic, CPW * wesn[i] / dts - df[i] - df[i + 1], 1.))
        cr.append(W(ic, df[i + 1], 0.))
        q.append(W(ic, fhsn[i + 1] - fhsn[i], 0.))
    cl[0] = zero
    cr[N_SNOW - 1] = zero
    for i in range(N_SNOW - 1):
        cl[i + 1] = W(ice1[i], cl[i + 1], 0.)
    for i in range(1, N_SNOW):
        cr[i - 1] = W(ice1[i], cr[i - 1], 0.)
    dtc = trid(cl, cd, cr, q)
    for i in range(N_SNOW):
        bad = (tpsn[i] + dtc[i] > 0.) | (htsnn[i] + wesn[i] * CPW * dtc[i] > 0.)
        dtc[i] = W(bad, -tpsn[i], dtc[i])
        dtc[i] = W(ice1[i], dtc[i], 0.)
    eold = sum_list(htsnn)

    def flxnet(i):
        if i == 0:
            return fhsn[1] + df[1] * (dtc[0] - dtc[1]) - fhsn[0] - df[0] * dtc[0]
        if i < N_SNOW - 1:
            return fhsn[i + 1] + df[i + 1] * (dtc[i] - dtc[i + 1]) - fhsn[i] - df[i] * (dtc[i - 1] - dtc[i])
        return fhsn[N_SNOW] + df[N_SNOW] * dtc[N_SNOW - 1] - fhsn[N_SNOW - 1] - df[N_SNOW - 1] * (dtc[N_SNOW - 2] - dtc[N_SNOW - 1])

    wes = list(wesn); sdz = list(sndz); hts = list(htsnn)
    tpn = [None] * N_SNOW; fin = [None] * N_SNOW
    hcorr = zero
    lhflux = shflux = hlwout = evap = zero
    fhg = list(fhgnd)
    for i in range(N_SNOW):
        fl = flxnet(i)
        hsprime = htsnn[i] + areasc * fl * dts
        _, fprime, _, _ = get_tf0d(hsprime, wesn[i])
        fprime = fmax(0., fmin(1., fprime))
        # cases
        cold = ~tzero[i]                       # ice1 & ~tzero
        melt = ~ice1[i]
        frz = ice1[i] & tzero[i]
        neg = dtc[i] < 0.
        tnew = W(cold, tpsn[i] + dtc[i], W(melt, 0., W(neg, tpsn[i] + dtc[i], 0.)))
        fnew = W(cold, 1., W(melt, fprime, W(neg, 1., fprime)))
        if i == 0:
            lhflux = lhturb + dlhdtc * dtc[0]
            shflux = hsturb + dhsdtc * dtc[0]
            hlwout = hlwtc + dhlwtc * dtc[0]
            evap = lhflux / alhv
            dw = -evap * dts * areasc
            cut = -dw > wesn[0]
            dw = W(cut, -wesn[0], dw)
            evap = W(cut, -dw / (dts * areasc), evap)
            hcorr = W(cut, hcorr + (lhflux - evap * alhv) * areasc, hcorr)
            lhflux = W(cut, evap * alhv, lhflux)
            wes[0] = wesn[0] + dw
            dn = W(dw > 0., 1. / RHOMA, 1. / dens[0])
            sdz[0] = sndz[0] + dw * dn
            dw0 = dw
        if i == N_SNOW - 1:
            fhg = [fhgnd[iz] + area[iz] * df[N_SNOW] * dtc[N_SNOW - 1] for iz in range(N_SNOW)]
        hts[i] = (CPW * tnew - fnew * ALHM) * wes[i]
        tpn[i] = tnew; fin[i] = fnew
    enew = sum_list(hts)
    hcorr = hcorr - ((enew - eold) / dts + areasc * (lhflux + shflux + hlwout - raddn)
                     - areasc * (fhsn[N_SNOW] + df[N_SNOW] * dtc[N_SNOW - 1]))
    tpsn2, fices2 = get_tf_nd(hts, wes)
    wes[0] = wes[0] + (rainf * areasc + snowf) * dts
    hts[0] = hts[0] + (tsx - ALHM) * (snowf * dts)
    sdz[0] = sdz[0] + (snowf / RHOFS) * dts
    hcorr = hcorr - tsx * snowf
    snowd = sum_list(wes)
    tpsn2, fices2 = get_tf_nd(hts, wes)
    flow = zero
    pre = zero
    for i in range(N_SNOW):
        upd = flow > 0.
        wes[i] = W(upd, wes[i] + flow, wes[i])
        t_, f_ = get_tf_nd(hts, wes)
        tpsn2 = [W(upd, t_[k], tpsn2[k]) for k in range(N_SNOW)]
        fices2 = [W(upd, f_[k], fices2[k]) for k in range(N_SNOW)]
        pre = fmax((1. - fices2[i]) * wes[i], 0.)
        big = snowd > WEMIN
        icedens = wes[i] * fices2[i] / (sdz[i] + 1.e-20)
        densfac = fmax(0., fmin(1., icedens / RHOFS))
        term = densfac * snfr * (sdz[i] * RHOW - wes[i] * fices2[i])
        take = big & (pre > term)
        pre_b = fmin(pre - term, wes[i])
        wes_b = W(take, wes[i] - pre_b, wes[i])
        flow_b = W(take, pre_b, 0.)
        wes_s = wes[i] - pre
        wes[i] = W(big, wes_b, wes_s)
        flow = W(big, flow_b, pre)
        if i == 0:
            dz = fmin(flow / dens[0], 0.5 * sdz[0])
            sdz[0] = sdz[0] - dz
    pre = flow / dts
    snowd = sum_list(wes)
    excs = zero
    mass = zero
    big = snowd > WEMIN
    dens2 = []
    for i in range(N_SNOW):
        ok = sdz[i] > 0.
        dens2.append(W(ok, fmax(wes[i] / W(ok, sdz[i], 1.), RHOFS), RHOFS))
    cmpc = [jnp.exp(14.643 - (4000. / fmin(tpsn2[i] + TF, TF)) - .02 * dens2[i]) for i in range(N_SNOW)]
    wprev = zero
    exc_sum = zero
    dens_fin = list(dens)
    for i in range(N_SNOW):
        wi = wes[i]
        mass = mass + 0.5 * (wi + wprev)
        wprev = wi
        d = dens2[i] * (1. + (dts * 0.5e-7 * 9.81) * mass * cmpc[i])
        ex = d > RHOMA
        exc = W(big & ex, (d - RHOMA) * sdz[i], 0.)
        w_new = W(big & ex, wes[i] - exc, wes[i])
        h_new = (CPW * tpsn2[i] - fices2[i] * ALHM) * w_new
        hcorr = W(big & ex, hcorr + (hts[i] - h_new) / dts, hcorr)
        hts[i] = W(big & ex, h_new, hts[i])
        wes[i] = w_new
        d = W(ex, RHOMA, d)
        dens_fin[i] = W(big, d, dens_fin[i])
        exc_sum = exc_sum + exc
    pre = pre + exc_sum / dts
    sdz = [wes[i] / dens_fin[i] for i in range(N_SNOW)]
    ice10, tzero0 = [], []
    for i in range(N_SNOW):
        _, _, ic, tz = get_tf0d(hts[i], wes[i])
        ice10.append(ic); tzero0.append(tz)
    hts, wes, sdz = relayer(hts, wes, sdz)
    tpsn3, fices3 = get_tf_nd(hts, wes)
    for i in range(N_SNOW):
        k1 = ice10[i] & tzero0[i] & ((fices3[i] != 1.) | (tpsn3[i] != 0.))
        k2 = (~ice10[i]) & tzero0[i] & (fices3[i] == 1.) & (tpsn3[i] < 0.)
        k3 = ice10[i] & (~tzero0[i]) & (fices3[i] != 1.) & (tpsn3[i] == 0.)
        kf = k1 | k2 | k3
        hnew = -ALHM * wes[i]
        hcorr = W(kf, hcorr + (hts[i] - hnew) / dts, hcorr)
        hts[i] = W(kf, hnew, hts[i])
        tpsn3[i] = W(kf, 0., tpsn3[i])
        fices3[i] = W(kf, 1., fices3[i])
    areasc0 = fmin(sum_list(wes) / WEMIN, 1.)
    out = dict(
        wesn=[W(nosnow, w_ns[i], wes[i]) for i in range(N_SNOW)],
        htsnn=[W(nosnow, h_ns[i], hts[i]) for i in range(N_SNOW)],
        sndz=[W(nosnow, s_ns[i], sdz[i]) for i in range(N_SNOW)],
        tpsn=[W(nosnow, tp_ns[i], tpsn3[i]) for i in range(N_SNOW)],
        fices=[W(nosnow, fi_ns[i], fices3[i]) for i in range(N_SNOW)],
        areasc=W(nosnow, areasc_ns, areasc), areasc0=W(nosnow, areasc0_ns, areasc0),
        pre=W(nosnow, pre_ns, pre), fhgnd=[W(nosnow, 0., fhg[i]) for i in range(N_SNOW)],
        evap=W(nosnow, 0., evap), shflux=W(nosnow, 0., shflux), lhflux=W(nosnow, 0., lhflux), hlwout=W(nosnow, 0., hlwout),
        hcorr=W(nosnow, hcorr_ns, hcorr))
    return out


# ---- albedo (SIBALB) -----------------------------------------------------------------------------------------------------
_ALVDRS, _ALIDRS, _ALVDRD, _ALIDRD, _ALVDRI, _ALIDRI = 0.100, 0.200, 0.300, 0.350, 0.700, 0.700
_NLAI, _BLAI, _DLAI = 14, 0.5, 0.5
_ALATRM = _BLAI + (_NLAI - 1) * _DLAI - 1.E-6
_TAB = {k: jnp.asarray(getattr(TB, k)) for k in ("ALVDR", "BTVDR", "GMVDR", "ALIDR", "BTIDR", "GMIDR")}
_SNWMID = jnp.asarray(TB.SNWMID)


def _coeffsib(table, ityp, lai, dx, dy):
    """table (14,2,9); lai 1-based index (N,), ityp 1-based."""
    tk = table[:, :, ityp - 1]                           # (14, 2, N)
    n = jnp.arange(lai.shape[0])
    t1a = tk[lai - 1, 0, n]; t1b = tk[lai - 1, 1, n]
    t2a = tk[lai, 0, n]; t2b = tk[lai, 1, n]
    return (t1a + (t1b - t1a) * dy) * (1.0 - dx) + (t2a + (t2b - t2a) * dy) * dx


def sibalb(ityp, vlai, vgrn, zth, scalvdr, scalvdf, scalidr, scalidf, wesn, sndz):
    grn1, grn2 = 0.33, 0.67
    ala = fmin(fmax(0., vlai), _ALATRM)
    lai = 1 + jnp.maximum(0, ((ala - _BLAI) / _DLAI).astype(jnp.int32))
    dx = (ala - (_BLAI + (lai - 1).astype(f32) * _DLAI)) * (1. / _DLAI)
    dy = (vgrn - grn1) * (1. / (grn2 - grn1))
    out = {}
    for nm, (a, b, g) in {"v": ("ALVDR", "BTVDR", "GMVDR"), "i": ("ALIDR", "BTIDR", "GMIDR")}.items():
        alpha = _coeffsib(_TAB[a], ityp, lai, dx, dy)
        beta = _coeffsib(_TAB[b], ityp, lai, dx, dy)
        gamma = fmax(_coeffsib(_TAB[g], ityp, lai, dx, dy), 0.01)
        dr = alpha - zth * beta / (gamma + zth)
        df = alpha - beta + 2. * beta * gamma * (1. - gamma * jnp.log((1. + gamma) / gamma))
        out[nm] = (dr, df)
    avisdr = out["v"][0] * scalvdr; anirdr = out["i"][0] * scalidr
    avisdf = out["v"][1] * scalvdf; anirdf = out["i"][1] * scalidf
    clip = lambda a: fmin(1., fmax(0., a))
    avisdr, anirdr, avisdf, anirdf = clip(avisdr), clip(anirdr), clip(avisdf), clip(anirdf)
    swe = wesn[0] + wesn[1] + wesn[2]
    totdep = sndz[0] + sndz[1] + sndz[2]
    areasc = fmin(swe / WEMIN, 1.)
    density = (swe / (areasc + 1.e-20)) / (totdep + 1.e-20)
    dens_exc = fmax(0., density - RHOFS)
    asnvdr = fmax(0.5, 0.70 - dens_exc * .0006); asnndr = fmax(0.3, 0.5 - dens_exc * .0006)
    asnvdf = asnvdr; asnndf = asnndr
    snwmid = _SNWMID[jnp.minimum(ityp - 1, 8)]
    fac = swe / (swe + snwmid)
    fveg = fmin(1., vlai / 2.)
    res = {}
    for nm, snow, free in (("asnvdr", asnvdr, avisdr), ("asnndr", asnndr, anirdr), ("asnvdf", asnvdf, avisdf), ("asnndf", asnndf, anirdf)):
        sveg = free + (snow - free) * fac
        res[nm] = snow * (1. - fveg) + sveg * fveg
    return dict(avisdr=avisdr, anirdr=anirdr, avisdf=avisdf, anirdf=anirdf, **res)


# ---- PMONTH --------------------------------------------------------------------------------------------------------------
_VGRDA = jnp.asarray(TB.VGRDA); _VGRDB = jnp.asarray(TB.VGRDB)
_AKDAY0 = [float(x) for x in TB.AKDAY0]


def pmonth(ityp, jday, alat, green, zlt, prev):
    """returns dict(sqscat, z2, dzm, rsoil1, rsoil2, satcap, rdc, z0) ; `prev` supplies values for ityp == 9 (unchanged)."""
    veg = ityp
    k = veg - 1
    south = alat < 0.
    kday = W(south, (182 + jday) % 365, (365 + jday) % 365).astype(f32)
    ak = jnp.asarray(np.array(_AKDAY0, np.float32))
    klow = jnp.full(kday.shape, 11, jnp.int32)           # 0-based 12
    for kk in range(12):
        klow = W(kday >= ak[kk], kk, klow)
    wlo_a = (16. - kday) / 31.
    nxt = jnp.minimum(klow + 1, 12)
    wlo_b = (ak[nxt] - kday) / (ak[nxt] - ak[klow])
    wlo = W((klow == 11) & (kday < ak[11]), wlo_a, wlo_b)
    whi = 1. - wlo
    vgz0 = jnp.asarray(TB.VGZ0); vgdd = jnp.asarray(TB.VGDD)
    rlengt = wlo * vgz0[klow, k] + whi * vgz0[nxt, k]
    rootl = jnp.asarray(TB.VGRT)[k]
    vroot = rootl * jnp.asarray(TB.VGROCA)[k]
    dum1 = jnp.log(vroot / (1 - vroot))
    dum2 = 1. / (8. * 3.14159 * rootl)
    alphaf = dum2 * (vroot - 3. - 2. * dum1)
    rsoil1 = jnp.asarray(TB.VGRDRS)[k] / (rootl * jnp.asarray(TB.VGROTD)[k])
    rsoil2 = alphaf / jnp.asarray(TB.VGROTD)[k]
    scat = green * (jnp.asarray(TB.VGTR11)[k] + jnp.asarray(TB.VGRF11)[k]) + (1. - green) * (jnp.asarray(TB.VGTR12)[k]
                                                                                             + jnp.asarray(TB.VGRF12)[k])
    sqscat = jnp.sqrt(1. - scat)
    z2 = jnp.asarray(TB.VGZ2)[k]
    satcap = 0.2 * zlt
    z0 = 0.13 * (z2 - (z2 - 0.01) * jnp.exp(-zlt))
    rk = jnp.minimum(k, 7)
    rdc = fmax(_VGRDA[rk] * fmin(1., zlt / _VGRDB[rk]), 0.001)
    ice = veg == 9
    return dict(sqscat=W(ice, prev["sqscat"], sqscat), z2=W(ice, prev["z2"], z2), dzm=W(ice, prev["dzm"], 600.),
                rsoil1=W(ice, prev["rsoil1"], rsoil1), rsoil2=W(ice, prev["rsoil2"], rsoil2),
                satcap=W(ice, prev["satcap"], satcap), rdc=W(ice, prev["rdc"], rdc), z0=W(ice, prev["z0"], z0))


# ---- Louis surface layer ----------------------------------------------------------------------------------------------------
LOUIS, MAXRI = n32(5.), n32(2.)


def getcdm(tva, ua, dz, tvs, z0):
    rm = dz / z0 + 1.0
    cn = KARMAN / jnp.log(rm)
    cn = cn * cn
    dridtv = (GRAV * dz) / (tva * (ua * ua))
    cp = RGAS / (n32(2.) / n32(7.))
    ri = fmin(dridtv * (tva - tvs + (GRAV / CPAIR) * dz), MAXRI)
    unst = ri < 0.
    r = (3.0 * LOUIS * LOUIS) * (cn * jnp.sqrt(W(unst, -ri * rm, 0.)))
    psi_u = ri / (1.0 + r)
    fm_u = 1.0 - (2.0 * LOUIS) * psi_u
    psi_s = jnp.sqrt(1.0 + LOUIS * W(unst, 0., ri))
    r_s = W(unst, 0., ri) / psi_s
    fm_s = 1.0 / (1.0 + (2.0 * LOUIS) * r_s)
    fm = W(unst, fm_u, fm_s)
    return cn * fm, cn, ri


def getcdh(tva, ua, dz, tvs, zh, zq, cn, ri):
    rt = dz / zh + 1.0
    ct = KARMAN / jnp.log(rt)
    rq = dz / zq + 1.0
    cq = KARMAN / jnp.log(rq)
    ct = jnp.sqrt(cn) * ct
    cq = jnp.sqrt(cn) * cq
    unst = ri < 0.
    r_u = (3.0 * LOUIS * LOUIS) * (ct * jnp.sqrt(W(unst, -ri * rt, 0.)))
    psi_u = ri / (1.0 + r_u)
    fh_u = 1.0 - (3.0 * LOUIS) * psi_u
    psi_s = jnp.sqrt(1.0 + LOUIS * W(unst, 0., ri))
    r_s = W(unst, 0., ri) / psi_s
    fh_s = 1.0 / (1.0 + (3.0 * LOUIS) * W(unst, 0., ri) * psi_s)
    fh = W(unst, fh_u, fh_s)
    fh = W(fh <= 0.20, 0.20, fh)
    ri_safe = W(unst, ri, 1.)
    dfh_u = psi_u / ri_safe
    dfh_u = -(3.0 * LOUIS) * dfh_u * (1.0 - 0.5 * r_u * dfh_u)
    dfh_s = -(3.0 * LOUIS) * (fh * fh) * (psi_s + (0.5 * LOUIS) * r_s)
    dfh = W(unst, dfh_u, dfh_s)
    dfh = W(fh == 0.20, 0.0, dfh)
    dridtv = (GRAV * dz) / (tva * (ua * ua))
    dfh = dfh * dridtv
    return ct * fh, cq * fh, ct * dfh, cq * dfh


def louis_land(uu, ps, ta, ts, qa_atm, qs, pcu, lai, z0, dz):
    """one sub-tile (index n): returns ch, cq, dch, dcq (already multiplied by URA as in louissurface for ch,cq)."""
    laicrt = 0.3
    ucn = pcu * 8640.
    ucn = jnp.power((19.8 * ucn * ucn) / (1.5 + ucn + ucn * ucn), 0.4)
    ucn = ucn * ucn
    tva = ta * (1.0 + VIREPS * qa_atm)
    uuu = fmax(jnp.sqrt(uu * uu + 0. + ucn), USMIN)
    ura = uuu * ps / (RGAS * tva)
    tvs = ts * (1 + VIREPS * qs)
    cm, cn, ri = getcdm(tva, uuu, dz, tvs, z0)
    re = fmax(fmin(z0 * (jnp.sqrt(cm) * uuu) / NUAIR, 1000.), 0.4)
    zt = 1.0 - (1.0 - jnp.exp(-0.13 * jnp.sqrt(re))) * jnp.exp(-fmax(lai, laicrt))
    zt = W(lai < laicrt, zt + (jnp.exp(-2.5 * jnp.sqrt(jnp.sqrt(re)) + jnp.log(7.4)) - zt) * (laicrt - lai) / laicrt, zt)
    zt = z0 * zt
    zq = zt
    ch, cq, dch, dcq = getcdh(tva, uuu, dz, tvs, zt, zq, cn, ri)
    return ch * ura, cq * ura, dch, dcq


def turb_louis(tair, qair, psurf, rainf_c, wind, lai, zol, dze, tc, qa):
    """tc, qa: lists of 4 arrays (tc1, tc2, tc4, tpsn1 / qa1, qa2, qa4, qsat(tpsn1)). returns the *_1,2,4,S terms."""
    ch, cq, dch, dcq = [], [], [], []
    for n in range(4):
        c_h, c_q, d_h, d_q = louis_land(wind, psurf, tair, tc[n], qair, qa[n], rainf_c, lai, zol, dze)
        ch.append(c_h); cq.append(c_q); dch.append(d_h); dcq.append(d_q)
    rhoair = RGAS * tair * (1. + VIREPS * qair)
    rhoair = psurf / rhoair
    out = {}
    for n, nm in enumerate(("1", "2", "4", "s")):
        dq = qa[n] - qair
        dt = tc[n] - tair
        dcqdq = -dcq[n] * VIREPS * tc[n]
        dcqdt = -dcq[n] * (1. + VIREPS * qa[n])
        dchdt = -dch[n] * (1. + VIREPS * qa[n])
        dchdq = -dch[n] * VIREPS * tc[n]
        out["ra" + nm] = rhoair / ch[n]
        out["eturb" + nm] = cq[n] * dq
        out["hsturb" + nm] = CPAIR * ch[n] * dt
        out["dedqa" + nm] = fmax(0., cq[n] + dcqdq * dq)
        out["dedtc" + nm] = fmax(0., dcqdt * dq)
        out["dhsdtc" + nm] = fmax(0., CPAIR * (ch[n] + dchdt * dt))
        out["dhsdqa" + nm] = fmax(0., CPAIR * dchdq * dt)
    return out


# ---- qsat (clsmf25_esat_qsat) -----------------------------------------------------------------------------------------------
def qsat(t, p):
    """qsat(temperature [K], pressure [mb]); no latent-heat argument: esat = exp(C4 - C5/t)/epsilon."""
    return EPSILON / p * (jnp.exp(n32(21.18123) - n32(5418.) / t) / EPSILON)


# ---- the core: CLSMF25 ----------------------------------------------------------------------------------------------------
def clsmf25(dtstep, sfrac, x):
    """x: dict of the arguments of the Fortran routine (names lower case, see CORE_INPUTS). Returns a dict with the updated
    prognostic state (tc1, tc2, tc4, qa1, qa2, qa4, capac, catdef, rzexc, srfexc, ght[6], wesn[3], htsn[3], sndz[3]) and the
    diagnostics (evap, shflux, runoff, eint, esoi, eveg, esno, bflow, runsrf, smelt, hlwup, swland, hlatn, qinfil, ar1, ar2,
    rzeq, ghflux, tpsn1, asnow0, tp[6], sfmc, rzmc, prmc, tsurf, ...)."""
    g = x.__getitem__
    ityp = g("ityp")
    zero = 0. * g("tm")
    tc1, tc2, tc4 = g("tc1"), g("tc2"), g("tc4")
    qa1, qa2, qa4 = g("qa1"), g("qa2"), g("qa4")
    capac, catdef, rzexc, srfexc = g("capac"), g("catdef"), g("rzexc"), g("srfexc")
    ght = list(g("ght")); wesnn = list(g("wesn")); htsnnn = list(g("htsn")); sndzn = list(g("sndz"))
    zlai, green = g("zlai"), g("green")
    bf1, bf2, bf3, vgwmax, cdcr1, cdcr2 = g("bf1"), g("bf2"), g("bf3"), g("vgwmax"), g("cdcr1"), g("cdcr2")
    psis, bee, poros, wpwet, cond, gnu = g("psis"), g("bee"), g("poros"), g("wpwet"), g("cond"), g("gnu")
    ars = [g("ars1"), g("ars2"), g("ars3")]
    ara = [g("ara1"), g("ara2"), g("ara3"), g("ara4")]
    arw = [g("arw1"), g("arw2"), g("arw3"), g("arw4")]
    tsa1, tsa2, tsb1, tsb2, atau, btau = g("tsa1"), g("tsa2"), g("tsb1"), g("tsb2"), g("atau"), g("btau")
    dzsf = g("dzsf")
    # ---- initial per-tile quantities
    satcap = nofma(SATCAPFR * zlai, g('tm')) + 1.e-5          # no FMA contraction: SATCAP feeds FCAN = CAPAC/SATCAP, which is ill-conditioned for a full canopy
    csoil = W(ityp != 1, CSOIL_2, CSOIL_1)
    fcan = fmin(1., fmax(0., capac / satcap))
    sclai = fmin(1., zlai / 2.)
    potfrc = fcan * sclai
    potfrc = W(fcan < .1, potfrc * (10. * fcan), potfrc)
    rdcx = g("rdc")
    dedqa1x = fmax(g("dedqa1"), 500. / ALHE); dedtc1x = fmax(g("dedtc1"), 0.)
    dhsdqa1x = fmax(g("dhsdqa1"), 0.); dhsdtc1x = fmax(g("dhsdtc1"), -10.)
    dedqa2x = fmax(g("dedqa2"), 500. / ALHE); dedtc2x = fmax(g("dedtc2"), 0.)
    dhsdqa2x = fmax(g("dhsdqa2"), 0.); dhsdtc2x = fmax(g("dhsdtc2"), -10.)
    dedqa4x = fmax(g("dedqa4"), 500. / ALHE); dedtc4x = fmax(g("dedtc4"), 0.)
    dhsdqa4x = fmax(g("dhsdqa4"), 0.); dhsdtc4x = fmax(g("dhsdtc4"), -10.)
    qa1_orig, qa2_orig, qa4_orig = qa1, qa2, qa4
    tc1_orig, tc2_orig, tc4_orig = tc1, tc2, tc4
    potfrc = W(ityp >= 7, 0., potfrc)
    runsrf = zero
    # glacier (ityp 9): reset the snow pack to a permanent pack
    ice = ityp == 9
    nosnowice = (sum_list(htsnnn) + sum_list(wesnn)) == 0.
    tsn = [W(nosnowice, tc1 - 273.16, (htsnnn[i] + wesnn[i] * ALHM) / (SCONST * wesnn[i] + 1.e-5)) for i in range(3)]
    wi = [0.1, 0.2, 0.1]
    wesnn = [W(ice, wi[i] + zero, wesnn[i]) for i in range(3)]
    htsnnn = [W(ice, -ALHM * wi[i] + tsn[0] * SCONST * wi[i], htsnnn[i]) for i in range(3)]
    sndzn = [W(ice, wi[i] / .9 + zero, sndzn[i]) for i in range(3)]
    potfrc = W(ice, 1., potfrc)
    lake = ityp == 10
    catdef = W(lake, 0., catdef); rzexc = W(lake, 0., rzexc); srfexc = W(lake, 0., srfexc)
    satcap = W(lake, 1000., satcap)
    capac = W(lake, satcap, capac)
    potfrc = W(lake, 1., potfrc)
    # ---- root zone equilibrium and partition
    rzeqol = rzequil(catdef, vgwmax, cdcr1, cdcr2, wpwet, ars, ara, arw)
    pt = partition(dzsf, rzexc, rzeqol, vgwmax, cdcr1, cdcr2, psis, bee, poros, wpwet, ars, ara, arw, srfexc, catdef, runsrf,
                   dtstep)
    srfexc, catdef, runsrf = pt["srfexc"], pt["catdef"], pt["runsrf"]
    ar1, ar2, ar4 = pt["ar1"], pt["ar2"], pt["ar4"]
    srfmx, srfmn = pt["srfmx"], pt["srfmn"]
    tsoil = ar1 * tc1 + ar2 * tc2 + ar4 * tc4
    wtot_orig = wesnn[0] + wesnn[1] + wesnn[2] + (cdcr2 - catdef) + rzexc + srfexc + capac
    emaxrz = fmax(0., rzeqol + rzexc - wpwet * vgwmax)
    emaxrt = (capac + emaxrz + (srfexc - srfmn)) / dtstep
    ar1old, ar2old, ar4old = ar1, ar2, ar4
    # ---- ground heat initial estimate
    phi = poros
    zbar = -jnp.sqrt(1.e-20 + catdef / bf1) - bf2
    thetaf = 0.5
    fh21w, fh21i, fh21d, dfh21w, dfh21i, dfh21d = gndtp0([tc1, tc2, tc4], phi, zbar, thetaf, ght)
    hftds1, hftds2, hftds4 = -fh21w, -fh21i, -fh21d
    dhft1, dhft2, dhft4 = -dfh21w, -dfh21i, -dfh21d
    par = g("pardir") + g("pardif") + 1.E-20
    pdir = g("pardir") / par
    tc1sf, tc2sf, tc4sf = tc1, tc2, tc4
    rcun = rcunst(ityp, g("sunang"), g("sqscat"), pdir, par, zlai, green)
    rzi = pt["rzi"]
    kw = dict(dtstep=dtstep, ityp=ityp, um=g("um"), qm=g("qm"), swnet=g("swnetf"), hlwdwn=g("hlwdwn"), psur=g("psur"),
              rdc=rdcx, emaxrt=emaxrt, csoil=csoil, potfrc=potfrc)
    e1 = energy(1, rcin=rcun, eturb=g("eturb1"), dedqa=dedqa1x, dedtc=dedtc1x, hsturb=g("hsturb1"), dhsdqa=dhsdqa1x,
                dhsdtc=dhsdtc1x, ra=g("ra1"), hftds=hftds1, dhftds=dhft1, qsattc=g("qsat1"), dqsdtc=g("dqs1"),
                alwrad=g("alw1"), blwrad=g("blw1"), swsrf=pt["swsrf1"], tc=tc1sf, qa=qa1, **kw)
    e2 = energy(2, rcin=rcun, eturb=g("eturb2"), dedqa=dedqa2x, dedtc=dedtc2x, hsturb=g("hsturb2"), dhsdqa=dhsdqa2x,
                dhsdtc=dhsdtc2x, ra=g("ra2"), hftds=hftds2, dhftds=dhft2, qsattc=g("qsat2"), dqsdtc=g("dqs2"),
                alwrad=g("alw2"), blwrad=g("blw2"), swsrf=pt["swsrf2"], tc=tc2sf, qa=qa2, rzi=rzi, wpwet=wpwet, **kw)
    e4 = energy(4, rcin=1.E10 + zero, eturb=g("eturb4"), dedqa=dedqa4x, dedtc=dedtc4x, hsturb=g("hsturb4"), dhsdqa=dhsdqa4x,
                dhsdtc=dhsdtc4x, ra=g("ra4"), hftds=hftds4, dhftds=dhft4, qsattc=g("qsat4"), dqsdtc=g("dqs4"),
                alwrad=g("alw4"), blwrad=g("blw4"), swsrf=pt["swsrf4"], tc=tc4sf, qa=qa4, wpwet=wpwet, **kw)
    tc1sf, tc2sf, tc4sf = e1["tc"], e2["tc"], e4["tc"]
    qa1, qa2, qa4 = e1["qa"], e2["qa"], e4["qa"]
    evap1, evap2, evap4 = e1["evap"], e2["evap"], e4["evap"]
    shflux1, shflux2, shflux4 = e1["shflux"], e2["shflux"], e4["shflux"]
    hlwup1, hlwup2, hlwup4 = e1["hlwup"], e2["hlwup"], e4["hlwup"]
    ghflux1, ghflux2, ghflux4 = e1["ghflux"], e2["ghflux"], e4["ghflux"]
    rx11, rx21, rx12, rx22, rx14, rx24 = e1["rx1"], e1["rx2"], e2["rx1"], e2["rx2"], e4["rx1"], e4["rx2"]
    ra1, ra2, ra4 = g("ra1"), g("ra2"), g("ra4")
    rtbs1 = rx11 * rx21 / (rx11 + rx21 + 1.E-20)
    epfrc1 = potfrc * (ra1 + rtbs1) / (ra1 + potfrc * rtbs1)
    rtbs2 = rx12 * rx22 / (rx12 + rx22 + 1.E-20)
    epfrc2 = potfrc * (ra2 + rtbs2) / (ra2 + potfrc * rtbs2)
    rtbs4 = rx14 * rx24 / (rx14 + rx24 + 1.E-20)
    epfrc4 = potfrc * (ra4 + rtbs4) / (ra4 + potfrc * rtbs4)
    sumep = epfrc1 * evap1 * ar1 + epfrc2 * evap2 * ar2 + epfrc4 * evap4 * ar4
    sume = evap1 * ar1 + evap2 * ar2 + evap4 * ar4
    eirfrc = W(sume != -1.e-20, sumep / (sume + 1.E-20), sumep / (sume + 2.E-20))
    inter = dict(ar1=ar1, ar2=ar2, ar4=ar4, rzeqol=rzeqol, rzi=rzi, swsrf1=pt["swsrf1"], swsrf2=pt["swsrf2"], swsrf4=pt["swsrf4"],
                 srfmx=srfmx, srfmn=srfmn, hftds1=hftds1, hftds2=hftds2, hftds4=hftds4, dhft1=dhft1, dhft2=dhft2, dhft4=dhft4,
                 rcun=rcun, emaxrt=emaxrt, potfrc=potfrc, tc1sf=tc1sf, tc2sf=tc2sf, tc4sf=tc4sf, qa1=qa1, qa2=qa2, qa4=qa4,
                 evap1=evap1, evap2=evap2, evap4=evap4, shflux1=shflux1, shflux2=shflux2, shflux4=shflux4, hlwup1=hlwup1,
                 hlwup2=hlwup2, hlwup4=hlwup4, ghflux1=ghflux1, ghflux2=ghflux2, ghflux4=ghflux4, rx11=rx11, rx21=rx21,
                 rx12=rx12, rx22=rx22, rx14=rx14, rx24=rx24, eirfrc=eirfrc, csoil=csoil, satcap=satcap, capac=capac,
                 catdef=catdef, rzexc=rzexc, srfexc=srfexc)
    # ---- snow
    trainc, trainl, tsnow = g("trainc"), g("trainl"), g("tsnow")
    qs_ = g("qsats") if "qsats" in x else None
    psur = g("psur"); qm = g("qm")
    pr = trainc + trainl + tsnow
    dedea = g("dedqas") * EPSILON / psur
    dhsdea = g("dhsdqas") * EPSILON / psur
    desdtc = g("dqss") * psur / EPSILON
    dedtc0 = g("dedtcs") + dedea * desdtc
    dhsdtc0 = g("dhsdtcs") + dhsdea * desdtc
    raddn = g("hlwdwn") + g("swnets")
    t1 = [tc1 - 273.16, tc2 - 273.16, tc4 - 273.16]
    tsnowsrf, _, _, _ = get_tf0d(htsnnn[0], wesnn[0])
    tcs_orig = tsnowsrf + TF
    tcs_orig = W(sum_list(wesnn) == 0., fmin(TF, tc1_orig * ar1 + tc2_orig * ar2 + tc4_orig * (1. - ar1 - ar2)), tcs_orig)
    hlwtc = g("alws") + g("blws") * (tsnowsrf + TF)
    dhlwtc = g("blws")
    sn = snowrt(t1, [ar1, ar2, ar4], [1.8 + zero, 1.8 + zero, 1.8 + zero], pr, tsnow, g("tm"), dtstep, g("eturbs"), dedtc0,
                g("hsturbs"), dhsdtc0, hlwtc, dhlwtc, raddn, -.025, wesnn, htsnnn, sndzn)
    tpsn1 = sn["tpsn"][0] + TF; tpsn2 = sn["tpsn"][1] + TF; tpsn3 = sn["tpsn"][2] + TF
    smelt = sn["pre"]
    fh31w, fh31i, fh31d = sn["fhgnd"]
    asnow = sn["areasc"]; asnow0 = sn["areasc0"]
    hsnacc = (1. - asnow) * zero + sn["hcorr"]                 # hsnacc1,2,4 are zero
    wesnn, htsnnn, sndzn = sn["wesn"], sn["htsn"] if "htsn" in sn else sn["htsnn"], sn["sndz"]
    areasc = sn["areasc"]
    traincx = trainc * (1. - areasc)
    trainlx = trainl * (1. - areasc)
    dtc1sn = ((-(fh31w / (ar1 + 1.e-20)) - hftds1) * dtstep / csoil) / (1. + dhft1 * dtstep / csoil)
    dtc2sn = ((-(fh31i / (ar2 + 1.e-20)) - hftds2) * dtstep / csoil) / (1. + dhft2 * dtstep / csoil)
    dtc4sn = ((-(fh31d / (ar4 + 1.e-20)) - hftds4) * dtstep / csoil) / (1. + dhft4 * dtstep / csoil)
    tc1sn = tc1 + dtc1sn
    c = (tc1sn - tpsn3) * (tc1 - tpsn3) < 0.
    hsnacc = W(c, hsnacc + areasc * ar1 * (tc1sn - tpsn3) * csoil / dtstep, hsnacc); tc1sn = W(c, tpsn3, tc1sn)
    tc2sn = tc2 + dtc2sn
    c = (tc2sn - tpsn3) * (tc2 - tpsn3) < 0.
    hsnacc = W(c, hsnacc + areasc * ar2 * (tc2sn - tpsn3) * csoil / dtstep, hsnacc); tc2sn = W(c, tpsn3, tc2sn)
    tc4sn = tc4 + dtc4sn
    c = (tc4sn - tpsn3) * (tc4 - tpsn3) < 0.
    hsnacc = W(c, hsnacc + areasc * ar4 * (tc4sn - tpsn3) * csoil / dtstep, hsnacc); tc4sn = W(c, tpsn3, tc4sn)
    tc1 = tc1sf * (1 - areasc) + tc1sn * areasc
    tc2 = tc2sf * (1 - areasc) + tc2sn * areasc
    tc4 = tc4sf * (1 - areasc) + tc4sn * areasc
    evsnow = sn["evap"]
    esno = evsnow * asnow * dtstep
    shfluxs = sn["shflux"]; hlwups = sn["hlwout"]
    ghfluxs = ar1 * (hftds1 + dhft1 * dtc1sn) + ar2 * (hftds2 + dhft2 * dtc2sn) + ar4 * (hftds4 + dhft4 * dtc4sn)
    hlatn = (1. - asnow) * (evap1 * ar1 + evap2 * ar2 + evap4 * ar4) * ALHE + asnow * evsnow * ALHS
    evap = (1. - asnow) * (evap1 * ar1 + evap2 * ar2 + evap4 * ar4) + asnow * evsnow
    evapfr = (1. - asnow) * (evap1 * ar1 + evap2 * ar2 + evap4 * ar4)
    shflux = (1. - asnow) * (shflux1 * ar1 + shflux2 * ar2 + shflux4 * ar4) + asnow * shfluxs
    hlwup = (1. - asnow) * (hlwup1 * ar1 + hlwup2 * ar2 + hlwup4 * ar4) + asnow * hlwups
    swland = (1. - asnow) * g("swnetf") + asnow * g("swnets")
    ghflux = (1. - asnow) * (ghflux1 * ar1 + ghflux2 * ar2 + ghflux4 * ar4) + asnow * ghfluxs
    # ---- deeper soil temperature update
    zbar = -jnp.sqrt(1.e-20 + catdef / bf1) - bf2
    ght, tp, xfice, _ = gndtmp(dtstep, phi, zbar, 0.5, -ghflux, ght)
    frice = xfice
    # ---- water balance
    eintx = eirfrc * evapfr * dtstep
    c = eintx > capac
    edif = (eintx - capac) / dtstep
    evapfr = W(c, evapfr - edif, evapfr)
    evap = W(c, evap - edif, evap)
    hlatn = W(c, hlatn - edif * ALHE, hlatn)
    shflux = W(c, shflux + edif * ALHE, shflux)
    eirfrc = W(c, capac / ((evapfr + 1.E-20) * dtstep), eirfrc)
    wu = wupdat(dtstep, evapfr, satcap, ar1, ar2, ar4, cdcr1, eirfrc, rzeqol, srfmn, wpwet, vgwmax, rx11, rx21, rx12, rx22,
                rx14, rx24, capac, rzexc, catdef, srfexc)
    capac, rzexc, catdef, srfexc = wu["capac"], wu["rzexc"], wu["catdef"], wu["srfexc"]
    eint, esoi, eveg = wu["eint"], wu["esoi"], wu["eveg"]
    rd = rzdrain(dtstep, vgwmax, satcap, rzeqol, ar1, wpwet, tsa1, tsa2, tsb1, tsb2, atau, btau, cdcr2, poros, srfexc, capac,
                 rzexc, catdef, runsrf)
    capac, rzexc, srfexc, catdef, runsrf = rd["capac"], rd["rzexc"], rd["srfexc"], rd["catdef"], rd["runsrf"]
    bflow, catdef = base_flow(dtstep, bf1, bf2, bf3, cdcr1, frice, cond, gnu, catdef)
    thru, capac = interc(dtstep, trainlx, traincx, smelt, satcap, sfrac, capac)
    runsrf, srfexc, qinfil = srunoff(dtstep, ar1, ar2, ar4, thru, srfmx, srfexc, runsrf)
    rzeq = rzequil(catdef, vgwmax, cdcr1, cdcr2, wpwet, ars, ara, arw)
    adj = 0.5 * (rzeqol - rzeq)
    rzexc = rzexc + adj
    catdef = catdef + adj
    c = catdef < 0.
    runsrf = W(c, runsrf - catdef / dtstep, runsrf)
    catdef = W(c, 0., catdef)
    sm = calc_soil_moist(dzsf, vgwmax, cdcr1, cdcr2, wpwet, poros, psis, bee, ars, ara, arw, srfexc, rzexc, catdef)
    srfexc, rzexc, catdef = sm["srfexc"], sm["rzexc"], sm["catdef"]
    werror = sm["werror"]
    neg = werror <= 0.
    runsrf = W(neg, runsrf - werror / dtstep, runsrf)
    pos = ~neg
    edif = werror / dtstep
    evap = W(pos, evap - edif, evap)
    hlatn = W(pos, hlatn - edif * ALHE, hlatn)
    evegfrc = eveg / (eveg + esoi + 1.E-20)
    eveg = W(pos, eveg - edif * evegfrc * dtstep, eveg)
    esoi = W(pos, esoi - edif * (1. - evegfrc) * dtstep, esoi)
    shflux = W(pos, shflux + edif * ALHE, shflux)
    pt2 = partition(dzsf, rzexc, rzeqol, vgwmax, cdcr1, cdcr2, psis, bee, poros, wpwet, ars, ara, arw, srfexc, catdef, runsrf,
                    dtstep)
    srfexc, catdef, runsrf = pt2["srfexc"], pt2["catdef"], pt2["runsrf"]
    ar1, ar2, ar4 = pt2["ar1"], pt2["ar2"], pt2["ar4"]
    hold = csoil * (ar1old * tc1 + ar2old * tc2 + ar4old * tc4)
    hnew = csoil * (ar1 * tc1 + ar2 * tc2 + ar4 * tc4)
    shflux = shflux - (hnew - hold) / dtstep
    runoff = runsrf + bflow
    c = capac < 1.E-10
    runoff = W(c, runoff + capac / dtstep, runoff)
    capac = W(c, 0., capac)
    eint = eint * ALHE / dtstep
    esoi = esoi * ALHE / dtstep
    eveg = eveg * ALHE / dtstep
    esno = esno * ALHS / dtstep
    tsoil = ar1 * tc1 + ar2 * tc2 + ar4 * tc4
    tsurf = (1. - asnow) * tsoil + asnow * tpsn1
    z = asnow == 0.
    tpsn1 = W(z, tsurf, tpsn1)
    c = ar1 + ar2 > 1.
    a1_ge = ar1 >= ar2
    ar1n = W(c & a1_ge, 1. - ar2, ar1)
    ar2n = W(c & (~a1_ge), 1. - ar1n, ar2)
    ar1, ar2 = ar1n, ar2n
    return dict(tc1=tc1, tc2=tc2, tc4=tc4, qa1=qa1, qa2=qa2, qa4=qa4, capac=capac, catdef=catdef, rzexc=rzexc, srfexc=srfexc,
                ght=ght, wesn=wesnn, htsn=htsnnn, sndz=sndzn, evap=evap, shflux=shflux, runoff=runoff, eint=eint, esoi=esoi,
                eveg=eveg, esno=esno, bflow=bflow, runsrf=runsrf, smelt=smelt, hlwup=hlwup, swland=swland, hlatn=hlatn,
                qinfil=qinfil, ar1=ar1, ar2=ar2, rzeq=rzeq, ghflux=ghflux, tpsn1=tpsn1, asnow0=asnow0, tp=tp, sfmc=sm["sfmc"],
                rzmc=sm["rzmc"], prmc=sm["prmc"], tsurf=tsurf, fices=sn["fices"], asnow=asnow, frice=frice, inter=inter)
