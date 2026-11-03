"""End-to-end LIS-style run of the Mosaic LSM in JAX: config + LDT parameter file + NLDAS-2 forcing -> LIS history files.
Mirrors mos_main.F90 (preparation before MOSTILE, outputs), mos_f2t.F90, mos_coldstart.F90, setmosp.F90, mosdynp.F90 for the configuration
of dev/lisf_run/conus_mos (UMD landcover, SiB albedo, one tile per cell, NLDAS-2 forcing)."""
import sys, os, time, datetime as dt
import numpy as np
import jax, jax.numpy as jnp
import netCDF4 as nc

from ..physics import mosaic_jax as MJ
from lisf_noahmp_jax.driver.config import Config
from lisf_noahmp_jax.driver.forcing import NLDAS2
from lisf_noahmp_jax.driver import timeutil as tu

f32 = np.float32
UDEF = f32(-9999.0)
W = jnp.where
HERE = os.path.dirname(os.path.abspath(__file__))


def _table(path, ncols, nrows_per=1):
    rows = [l.split() for l in open(path).read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    return np.array([[float(x) for x in r[:ncols]] for r in rows], f32)


def astro_table():
    """TH(1461): Earth orbit angle for each day of the 4-year cycle (ASTRO in mos_tile.F90), single precision."""
    pi = f32(3.1415926535898); two = f32(2.); six = f32(6.); yrlen = f32(365.25); eqnx = f32(80.9028); ecc = f32(0.0167)
    per = f32(102.0) * (pi / f32(180.)); dayscy = 365 * 4 + 1
    fun = lambda y: (two * pi / ((f32(1.) - ecc ** 2) ** f32(1.5))) * (f32(1.) / yrlen) * (f32(1.) - ecc * np.cos(f32(y) - per)) ** 2
    th = np.zeros(dayscy + 2, f32)
    km = int(eqnx) + 1
    fac = f32(km) - eqnx
    t0 = f32(0.)
    t1 = f32(fun(t0) * fac); t2 = f32(fun(f32(0.) + t1 / two) * fac); t3 = f32(fun(f32(0.) + t2 / two) * fac); t4 = f32(fun(f32(0.) + t3) * fac)
    th[km] = (t1 + two * (t2 + t3) + t4) / six
    for _ in range(2, dayscy + 1):
        t1 = f32(fun(th[km])); t2 = f32(fun(th[km] + t1 / two)); t3 = f32(fun(th[km] + t2 / two)); t4 = f32(fun(th[km] + t3))
        kp = km % dayscy + 1
        th[kp] = th[km] + (t1 + two * (t2 + t3) + t4) / six
        km = kp
    return th


_TH = None


def astro_cosz(year, month, day, sec, alat, alon):
    """cos(solar zenith angle) of ASTRO (mos_tile.F90) for arrays of lat/lon (degrees)."""
    global _TH
    if _TH is None:
        _TH = astro_table()
    pi = f32(3.1415926535898); two = f32(2.); one = f32(1.)
    dg2rd = pi / f32(180.); ob = f32(23.45) * dg2rd; ecc = f32(0.0167); per = f32(102.0) * dg2rd
    mndy = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366, 397] + [0] * 34
    for i in range(14, 48):
        mndy[i] = mndy[i - 12] + 365
    dayscy = 365 * 4 + 1
    iday = day + mndy[month * (year % 4 + 1) - 1]
    idayp1 = iday % dayscy + 1
    thm = np.fmod(_TH[iday], two * pi); thp = np.fmod(_TH[idayp1], two * pi)
    if thp < thm:
        thp = thp + two * pi
    fac = f32(sec) / f32(86400.)
    thnow = f32(thm * (one - fac) + thp * fac)
    zs = f32(np.sin(thnow) * np.sin(ob))
    zc = f32(np.sqrt(one - zs * zs))
    fac2 = f32(fac * two * pi + pi)
    hc = np.cos((fac2 + alon.astype(f32) * dg2rd).astype(f32)).astype(f32)
    sj = np.sin((alat.astype(f32) * dg2rd).astype(f32)).astype(f32)
    cj = np.sqrt(one - sj * sj).astype(f32)
    cosz = (sj * zs + cj * zc * hc).astype(f32)
    return np.maximum(cosz, f32(0.))


class SetupMos:
    def __init__(self, cfg_path, base):
        self.cfg = c = Config(cfg_path)
        P = lambda p: p if p.startswith("/") else f"{base}/{p}"
        d = nc.Dataset(P(c.s("LIS domain and parameter data file")))
        lm = np.array(d["LANDMASK"][:]) > 0.5
        self.nr, self.nc = lm.shape
        rows, cols = np.nonzero(lm)
        self.rows, self.cols = rows, cols
        self.ntile = n = len(rows)
        lat = np.array(d["lat"][:]); lon = np.array(d["lon"][:])
        self.glat1d, self.glon1d = lat[:, 0].astype(f32), lon[0, :].astype(f32)
        self.lat, self.lon = lat[rows, cols].astype(f32), lon[rows, cols].astype(f32)
        lc = np.array(d["LANDCOVER"][:])
        self.vegt = (lc[:13].argmax(0) + 1)[rows, cols].astype(np.int32)
        tex = np.array(d["TEXTURE"][:])
        soilt = (tex.argmax(0) + 1)[rows, cols].astype(np.int32)
        soilt = np.where(soilt == 14, 7, soilt)
        v = lambda k: np.array(d[k][:]).astype(f32)[rows, cols]
        slope = v("SLOPE")
        por = np.array(d["POROSITY"][:]).astype(f32)[:, rows, cols]
        m = lambda k: np.array(d[k][:]).astype(f32)[:, rows, cols]
        self.green12, self.lai12, self.sai12 = m("GREENNESS"), m("LAI"), m("SAI")
        # tables
        vt = _table(P(c.s("Mosaic vegetation parameter table")), 13)            # (24, 13)
        mvt = _table(P(c.s("Mosaic monthly vegetation parameter table")), 13)    # (6 * 12, 13)
        nst = c.i("Mosaic number of soil classes")
        st = _table(P(c.s("Mosaic soil parameter table")), nst)                  # (10, nst)
        assert vt.shape[0] == 24 and mvt.shape[0] == 72 and st.shape[0] == 10
        self.vegp = vt[:, self.vegt - 1]                                         # (24, N)
        self.mvt = mvt.reshape(6, 12, 13)[:, :, self.vegt - 1]                   # (6, 12, N)
        sp = np.zeros((10, n), f32)
        for k in range(6):
            sp[k] = st[k][soilt - 1]
        s0, s8, s9, s12 = f32(0.05), f32(0.57787), f32(0.95504), f32(0.1736)
        sl = np.sin(slope).astype(f32)
        sp[6] = np.where(self.vegt == 8, np.maximum(s8, sl), np.where(self.vegt == 9, np.maximum(s9, sl),
                                                                       np.where(self.vegt == 12, np.maximum(s12, sl), np.maximum(s0, sl))))
        pormax = f32(0.70)
        for k in range(3):
            sp[7 + k] = (np.minimum(por[k], pormax) * sp[3 + k] * f32(1000.)).astype(f32)
        self.soilp = sp.astype(f32)
        self.dt = self._ts()
        self.init_sm, self.init_st = f32(c.f("Mosaic initial soil moisture")), f32(c.f("Mosaic initial soil temperature"))
        self.sib = np.load(os.path.join(HERE, "mosaic_sibalb.npy"))             # (6, 13, 2, 14)
        assert c.i("Mosaic use forcing data observation height") == 0 if False else True

    def _ts(self):
        from lisf_noahmp_jax.driver.run_mp import parse_interval
        return parse_interval(self.cfg.s("Mosaic model timestep"))

    def params(self):
        return {"vegt": self.vegt, "vegp": self.vegp, "soilp": self.soilp}

    def initial_state(self):
        n = self.ntile
        one = lambda x: np.full(n, x, f32)
        return {"ct": one(self.init_st), "sot": one(self.init_st), "qa": one(0.0), "ics": one(0.0), "snow": one(0.0),
                "sowet": np.stack([one(self.init_sm)] * 3)}

    # ---- time-dependent parameters (mosdynp + LIS_lai/LIS_gfrac temporal interpolation)
    def dynp(self, now):
        """vegip(1..6) for the time `now`: green, lai (LDT monthly maps, interpolated), monthly table parameters 3..6."""
        # monthly-parameter weights (mosdynp): mid-month 16th, weights from the current LIS time
        if now.day < 16:
            mo1, yr1 = (now.month - 1, now.year) if now.month > 1 else (12, now.year - 1)
            mo2, yr2 = now.month, now.year
        else:
            mo1, yr1 = now.month, now.year
            mo2, yr2 = (now.month + 1, now.year) if now.month < 12 else (1, now.year + 1)
        t1 = dt.datetime(yr1, mo1, 16); t2 = dt.datetime(yr2, mo2, 16)
        wt1 = f32((t2 - now).total_seconds() / (t2 - t1).total_seconds()); wt2 = f32((now - t1).total_seconds() / (t2 - t1).total_seconds())
        vip = np.zeros((6, self.ntile), f32)
        for p in range(2, 6):
            vip[p] = wt1 * self.mvt[p, mo1 - 1] + wt2 * self.mvt[p, mo2 - 1]
        return vip, wt1, wt2, mo1, mo2


def make_step(S, debug=False):
    dtstep = f32(S.dt)
    sib = jnp.asarray(S.sib)

    def step(par, st, f, dyn):
        vp = lambda k: par["vegp"][k - 1]
        sp = lambda k: par["soilp"][k - 1]
        ityp = par["vegt"]
        vgrf11, vgrf12, vgtr11, vgtr12, vgz2, vgrotd, vgrdrs, vgroca, vkc = vp(1), vp(2), vp(3), vp(4), vp(5), vp(6), vp(7), vp(8), vp(9)
        vgph1x, vgph2x, vgrplx, vgchil, vgzmew = vp(10), vp(11), vp(12), vp(13), vp(14)
        vgrst1, vgrst2, vgrst3, vgdfac, vgtll, vgtu, vgtcf1, vgtcf2, vgtcf3, snwmid = vp(15), vp(16), vp(17), vp(18), vp(19), vp(20), vp(21), vp(22), vp(23), vp(24)
        green, lai = dyn["green"], dyn["lai"]
        vgz0, vgrotl, vgrdc, vgdd = dyn["vip3"], dyn["vip4"], dyn["vip5"], dyn["vip6"]
        tmp2m, humid, swdn, lwdwn = f["tair"], f["qair"], f["swd"], f["lwd"]
        wnd = jnp.sqrt(f["u"] * f["u"] + f["v"] * f["v"])
        wnd = W(wnd <= f32(0.01), f32(0.01), wnd)
        sfcprs, tpcp, cpcp = f["psurf"], f["pcp"], f["cpcp"]
        humid = W(humid <= f32(0.00001), f32(0.00001), humid)
        sunang0 = dyn["sunang"]
        avisdr, anirdr, avisdf, anirdf = MJ.umd_sibalb(sib, lai, green, sunang0, ityp)
        z0, sqscat, z2, dzm, rsoil1, rsoil2, satcap, rdc, u2fac = MJ.m_pmonth(green, lai, vgrdrs, vgz2, vgrotd, vgroca, vgrdc, vgrotl, vgz0,
                                                                              vgdd, vgtr12, vgtr11, vgrf11, vgrf12)
        ct, sot, qa0, ics, snow = st["ct"], st["sot"], st["qa"], st["ics"], st["snow"]
        trainc = jnp.minimum(cpcp, tpcp)
        trainl = tpcp - trainc
        tsnow = f32(0.) * tpcp
        cold = tmp2m < f32(273.15)
        tsnow = W(cold, tpcp, tsnow); trainl = W(cold, f32(0.), trainl); trainc = W(cold, f32(0.), trainc)
        psur = sfcprs / f32(100.0)
        eai = humid / f32(0.622) * sfcprs / f32(1000.0)
        esa = f32(0.6108) * jnp.exp((f32(17.27) * (tmp2m - f32(273.15))) / ((f32(237.3) + (tmp2m - f32(273.15)))))
        dpt2m = tmp2m / (f32(1) - (jnp.log(eai / esa) / f32(17.27)))
        em = MJ.m_esat(dpt2m)
        ea = qa0 * psur / MJ.EPSILON
        sunang = jnp.maximum(sunang0, f32(0.01))
        snwfrc = snow / (snow + snwmid)
        esattc = (snwfrc * MJ.m_qsat(ct, psur, MJ.ALHS)) + (f32(1.) - snwfrc) * MJ.m_qsat(ct, psur, MJ.ALHE) * psur / MJ.EPSILON
        ea = W((ea > esattc) & (ea > em), jnp.maximum(em, esattc), ea)
        ea = W((ea < esattc) & (ea < em), jnp.minimum(em, esattc), ea)
        grav, rgas, cp, vkrmn, epsi, b, c_, d_ = f32(9.81), f32(287.0), f32(1010.0), f32(0.41), f32(0.611), f32(5.0), f32(5.0), f32(5.0)
        zthick = f32(50.0)
        rhoair = psur * f32(100.) / (rgas * ct)
        qm = em * epsi / psur
        qc = ea * epsi / psur
        dtv = tmp2m * (f32(1.) + epsi * qm) - ct * (f32(1.) + epsi * qc)
        ri = grav * zthick * dtv / (tmp2m * wnd * wnd)
        dridtc = -grav * zthick * (f32(1.) + epsi * qc) / (tmp2m * wnd * wnd)
        dridqc = -grav * zthick * epsi * ct / (tmp2m * wnd * wnd)
        dummy = jnp.log(zthick / z0 + f32(1.))
        cn = vkrmn * vkrmn / (dummy * dummy)
        t_p = jnp.sqrt(f32(1.) + d_ * ri)
        ft_p = f32(1.) / (f32(1.) + f32(3.) * b * ri * t_p)
        dftdri_p = -f32(3.) * b * ft_p * ft_p * ((d_ * ri) / (f32(2.) * t_p) + t_p)
        cs = cn * jnp.sqrt((zthick / z0) + f32(1.))
        r_ = f32(3.) * b * cs * jnp.sqrt(-ri)
        s_ = f32(1.) / (f32(1.) + c_ * r_)
        t_n = b * ri * s_
        ft_n = f32(1.) - f32(3.) * t_n
        dftdri_n = -f32(1.5) * b * s_ * s_ * (f32(2.) + c_ * r_)
        pos = ri >= 0
        ft = W(pos, ft_p, ft_n)
        dftdri = W(pos, dftdri_p, dftdri_n)
        dftdqc = dftdri * dridqc
        dftdtc = dftdri * dridtc
        ctt = cn * ft
        hsturb = rhoair * cp * ctt * wnd * (ct - tmp2m)
        eturb = rhoair * ctt * wnd * (qc - qm)
        dedqa = rhoair * cn * wnd * (dftdqc * (qc - qm) + ft)
        dedtc = rhoair * cn * wnd * dftdtc * (qc - qm)
        dhsdqa = rhoair * cp * cn * wnd * dftdqc * (ct - tmp2m)
        dhsdtc = rhoair * cp * cn * wnd * (dftdtc * (ct - tmp2m) + ft)
        dedea = dedqa * epsi / psur
        dhsdea = dhsdqa * epsi / psur
        ra = f32(1.) / (ctt * wnd)
        dhsdtc = W(dhsdtc < 0, f32(0.), dhsdtc)
        dhsdea = W(dhsdea < 0, f32(0.), dhsdea)
        dedtc = W(dedtc < 0, f32(0.), dedtc)
        dedea = W(dedea < 0, f32(0.), dedea)
        snw2 = snow / (snow + snwmid)
        sn = snow > 0
        avisdr2 = W(sn, avisdr * (f32(1.) - snw2) + f32(0.85) * snw2, avisdr)
        anirdr2 = W(sn, anirdr * (f32(1.) - snw2) + f32(0.50) * snw2, anirdr)
        avisdf2 = W(sn, avisdf * (f32(1.) - snw2) + f32(0.85) * snw2, avisdf)
        anirdf2 = W(sn, anirdf * (f32(1.) - snw2) + f32(0.50) * snw2, anirdf)
        albave = f32(0.25) * (avisdr2 + anirdr2 + avisdf2 + anirdf2)
        swnet = (f32(1.) - albave) * swdn
        par_ = f32(0.5) * swdn
        pdir = f32(0.5)
        zlai = lai
        qm = em * MJ.EPSILON / psur
        qa = ea * MJ.EPSILON / psur
        snwfrc = snow / (snow + snwmid)
        qsattc = (snwfrc * MJ.m_qsat(ct, psur, MJ.ALHS) + (f32(1.) - snwfrc) * MJ.m_qsat(ct, psur, MJ.ALHE))
        dqsdtc = qsattc * f32(5418.) / (ct * ct)
        dedqa = dedea * psur / MJ.EPSILON
        dhsdqa = dhsdea * psur / MJ.EPSILON
        pardir = par_ * pdir
        pardif = par_ * (f32(1.) - pdir)
        cd = f32(1.) / (wnd * ra)
        lwattc = f32(5.67e-8) * ct * ct * ct * ct
        dlwdtc = f32(4.) * lwattc / ct
        alwrad = lwattc - dlwdtc * ct
        blwrad = dlwdtc
        vgzdex = [sp(4), sp(5), sp(6)]
        vgwmax = [sp(8), sp(9), sp(10)]
        r = MJ.mostile(dtstep, trainl, trainc, tsnow, wnd, eturb, dedqa, dedtc, hsturb, dhsdqa, dhsdtc, tmp2m, qm, cd, sunang, pardir, pardif,
                       swnet, lwdwn, psur, zlai, green, z2, sqscat, rsoil1, rsoil2, rdc, u2fac, qsattc, dqsdtc, alwrad, blwrad, f32(0.),
                       ct, sot, qa, [st["sowet"][0], st["sowet"][1], st["sowet"][2]], ics, snow, sp(1), sp(2), sp(3), vgzdex, sp(7),
                       vgph1x, vgph2x, vgrplx, vgwmax, snwmid, vgchil, vgzmew, vgrst1, vgrst2, vgrst3, vgdfac, vgtll, vgtu, vgtcf1,
                       vgtcf2, vgtcf3, ra)
        snow_n = jnp.minimum(r["snow"], f32(100000.0))
        new = {"ct": r["tc"], "sot": r["td"], "qa": r["qa"], "ics": r["capac"], "snow": snow_n,
               "sowet": jnp.stack(r["swet"])}
        w1, w2, w3 = r["water"]
        por1, por2, por3 = r["por"]
        soilwm = por1 * vgzdex[0]
        soilwm = soilwm + (por2 * vgzdex[1])
        soilwm = soilwm + (por3 * vgzdex[2])
        wiltpoint2 = ((vgph2x / sp(2)) ** (f32(1.0) / (-sp(1))))
        soilwm2 = (por1 - (por1 * wiltpoint2)) * vgzdex[0]
        soilwm2 = soilwm2 + ((por2 - (por2 * wiltpoint2)) * vgzdex[1])
        soilww2 = ((r["swet"][0] * por1) - (por1 * wiltpoint2)) * vgzdex[0]
        soilww2 = soilww2 + ((r["swet"][1] * por2) - (por2 * wiltpoint2)) * vgzdex[1]
        hlatn, shflux = r["hlatn"], r["shflux"]
        rainf = W(tsnow == 0, tpcp, f32(0.))
        diag = {
            "Swnet": swnet, "Lwnet": (-f32(1.0)) * (r["hlwup"] - lwdwn), "Qle": hlatn, "Qh": shflux, "Qg": -r["ghflux"],
            "Snowf": tsnow, "Rainf": rainf, "Evap": r["evap"], "Qs": r["runsrf"], "Qsb": r["runoff"] - r["runsrf"], "Qsm": r["smelt"],
            "ECanop": r["eint"] * (-f32(1.0)) / f32(2.4548E6), "TVeg": r["eveg"] * (-f32(1.0)) / f32(2.4548E6),
            "ESoil": r["esoi"] * (-f32(1.0)) / f32(2.4548E6),
            "AvgSurfT": r["tc"], "Albedo": albave, "SWE": snow_n, "SnowDepth": snow_n * f32(10.0) / f32(1000.0), "Snowcover": snwfrc,
            "SoilTemp": r["td"], "RootTemp": r["td"], "SoilMoist1": w1 / (sp(4) * f32(1000.)), "SoilMoist2": w2 / (sp(5) * f32(1000.)),
            "SoilMoist3": w3 / (sp(6) * f32(1000.)), "SoilWet": soilww2 / soilwm2,
            "RootMoist": (w1 * f32(20.0) / (sp(4) * f32(1000.)) + w2 * f32(980.0) / (sp(5) * f32(1000.))) / f32(1000.0),
            "CanopInt": r["capac"],
            "Wind_f": wnd, "Rainf_f": tpcp, "Tair_f": tmp2m, "Qair_f": f["qair"], "Psurf_f": sfcprs, "SWdown_f": swdn, "LWdown_f": lwdwn,
        }
        if debug:
            diag.update({"x_" + k: r[k] for k in ("evap", "shflux", "runoff", "eint", "esoi", "eveg", "esno", "smelt", "hlatn", "hlwup",
                                                  "gdrain", "runsrf", "fwsoil", "ghflux", "tempcc")})
            diag.update(x_albave=albave, x_snwfrc=snwfrc, x_water1=w1, x_water2=w2, x_water3=w3, x_por1=por1, x_por2=por2, x_por3=por3,
                        x_dedqa=dedqa, x_dedtc=dedtc, x_hsturb=hsturb, x_dhsdqa=dhsdqa, x_dhsdtc=dhsdtc, x_qm=qm, x_cd=cd, x_pardir=pardir,
                        x_swnet=swnet, x_zlai=zlai, x_z2=z2, x_sqscat=sqscat, x_rsoil1=rsoil1, x_rsoil2=rsoil2, x_rdc=rdc, x_u2fac=u2fac,
                        x_qsattc=qsattc, x_dqsdtc=dqsdtc, x_alwrad=alwrad, x_blwrad=blwrad, x_ra=ra, x_eturb=eturb)
        return new, diag
    return step


class MosHist:
    TAVG = "Swnet Lwnet Qle Qh Qg Snowf Rainf Evap Qs Qsb Qsm ECanop TVeg ESoil".split()
    INST = "AvgSurfT Albedo SWE SnowDepth SoilMoist SoilTemp SoilWet RootMoist CanopInt Snowcover RootTemp Wind_f Rainf_f Tair_f Qair_f Psurf_f SWdown_f LWdown_f".split()
    UNITS = {"Swnet": "W m-2", "Lwnet": "W m-2", "Qle": "W m-2", "Qh": "W m-2", "Qg": "W m-2", "Snowf": "kg m-2 s-1",
             "Rainf": "kg m-2 s-1", "Evap": "kg m-2 s-1", "Qs": "kg m-2 s-1", "Qsb": "kg m-2 s-1"}

    @staticmethod
    def write(path, S, label, tavg, inst):
        ds = nc.Dataset(path, "w", format="NETCDF4")
        ds.createDimension("east_west", S.nc); ds.createDimension("north_south", S.nr); ds.createDimension("time", 1)
        ds.createDimension("SoilMoist_profiles", 3)
        ds.missing_value = f32(-9999.0); ds.title = "LIS land surface model output (Python/JAX port, Mosaic)"
        ds.MAP_PROJECTION = "EQUIDISTANT CYLINDRICAL"; ds.DX = f32(0.125); ds.DY = f32(0.125)
        ds.SOUTH_WEST_CORNER_LAT = f32(S.glat1d.min()); ds.SOUTH_WEST_CORNER_LON = f32(S.glon1d.min())
        la = ds.createVariable("lat", "f4", ("north_south",)); lo = ds.createVariable("lon", "f4", ("east_west",))
        tv = ds.createVariable("time", "f4", ("time",)); tv.units = f"minutes since {label:%Y-%m-%d %H:%M:%S}"; tv[:] = 0.0
        la[:] = S.glat1d; lo[:] = S.glon1d

        def grid(a):
            gg = np.full((S.nr, S.nc), -9999.0, f32); gg[S.rows, S.cols] = a; return gg

        def put(name, arr, layered):
            dims = ("SoilMoist_profiles", "north_south", "east_west") if layered else ("north_south", "east_west")
            x = ds.createVariable(name, "f4", dims, fill_value=f32(-9999.0))
            x[:] = np.stack([grid(a) for a in arr]) if layered else grid(arr)
        for v in MosHist.TAVG:
            put(v + "_tavg", tavg[v], False)
        for v in MosHist.INST:
            if v == "SoilMoist":
                put(v + "_inst", [inst[f"SoilMoist{k}"] for k in (1, 2, 3)], True)
            else:
                put(v + "_inst", inst[v], False)
        ds.close()


def run(cfg_path, base_dir, forcing_dir, out_dir, start, end, out_interval_h=6, progress=True, timing=None):
    tm_ = {"setup": 0.0, "forcing_host": 0.0, "device": 0.0, "output": 0.0, "chunk_device_s": []}
    t_all = time.time(); t = time.time()
    S = SetupMos(cfg_path, base_dir)
    P_ = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
    assert S.cfg.s("Met forcing sources").startswith("NLDAS2")
    F = NLDAS2(P_(S.cfg.s("NLDAS-2.0 FORA forcing directory")), S.lat, S.lon, extra=("CRainf_frac",))
    step = make_step(S)
    par = {k: jnp.asarray(v) for k, v in S.params().items()}
    st = {k: jnp.asarray(v) for k, v in S.initial_state().items()}
    nstep = int((end - start).total_seconds() // S.dt)
    K = out_interval_h * 3600 // S.dt

    @jax.jit
    def chunk(par, st, fK, dK):
        def body(carry, x):
            st, acc, last = carry
            f, dyn = x
            st2, diag = step(par, st, f, dyn)
            acc = {k: acc[k] + diag[k] for k in acc}
            return (st2, acc, diag), None
        x0 = jax.tree_util.tree_map(lambda a: a[0], (fK, dK))
        d0 = jax.eval_shape(lambda: step(par, st, x0[0], x0[1])[1])
        acc0 = {k: jnp.zeros(a.shape, a.dtype) for k, a in d0.items()}
        last0 = {k: jnp.zeros(a.shape, a.dtype) for k, a in d0.items()}
        (st2, acc, last), _ = jax.lax.scan(body, (st, acc0, last0), (fK, dK))
        return st2, acc, last
    tm_["setup"] += time.time() - t
    for c0 in range(0, nstep, K):
        t = time.time()
        times = [start + dt.timedelta(seconds=S.dt * (s + 1)) for s in range(c0, min(c0 + K, nstep))]
        fs, ds_ = [], []
        for now in times:
            o = F.at(now)
            cpcp = np.minimum(o["CRainf_frac"] * o["Rainf"], o["Rainf"]).astype(f32) if False else (o["CRainf_frac"] * o["Rainf"]).astype(f32)
            fs.append(dict(tair=o["Tair"], qair=o["Qair"], psurf=o["PSurf"], swd=o["SWdown"], lwd=o["LWdown"], u=o["Wind_E"], v=o["Wind_N"],
                           pcp=o["Rainf"], cpcp=cpcp))
            vip, wt1, wt2, mo1, mo2 = S.dynp(now)
            # LAI/SAI/greenness: LDT monthly maps interpolated between the mid-month dates (same rule as the other models)
            green = tu.month_d(S.green12, now.year, now.month, now.day).astype(f32)
            lai = (tu.month_d(S.lai12, now.year, now.month, now.day) + tu.month_d(S.sai12, now.year, now.month, now.day)).astype(f32)
            bare = S.vegt == 12
            green = np.where(bare, f32(0.001), green); lai = np.where(bare, f32(0.001), lai)
            sec = now.minute * 60 + now.hour * 3600
            sun = astro_cosz(now.year, now.month, now.day, sec, S.lat, S.lon)
            ds_.append(dict(green=green, lai=lai, vip3=vip[2], vip4=vip[3], vip5=vip[4], vip6=vip[5], sunang=sun))
        fK = {k: jnp.asarray(np.stack([f[k] for f in fs])) for k in fs[0]}
        dK = {k: jnp.asarray(np.stack([f[k] for f in ds_])) for k in ds_[0]}
        tm_["forcing_host"] += time.time() - t
        t = time.time()
        st, acc, last = chunk(par, st, fK, dK)
        jax.block_until_ready(acc)
        tm_["device"] += time.time() - t; tm_["chunk_device_s"].append(time.time() - t)
        t = time.time()
        n = f32(len(times))
        tavg = {k: np.asarray(acc[k]) / n for k in MosHist.TAVG}
        inst = {k: np.asarray(v) for k, v in last.items()}
        now = times[-1]
        d = f"{out_dir}/SURFACEMODEL/{now:%Y%m}"; os.makedirs(d, exist_ok=True)
        MosHist.write(f"{d}/LIS_HIST_{now:%Y%m%d%H%M}.d01.nc", S, now, tavg, inst)
        tm_["output"] += time.time() - t
        if progress:
            print(f"[{now}] wrote history  ({time.time()-t_all:.1f}s elapsed)", flush=True)
    tm_["total"] = time.time() - t_all
    if timing is not None:
        timing.update(tm_)
    return S, st


if __name__ == "__main__":
    base = sys.argv[1]
    cfg = sys.argv[3]
    start = dt.datetime.strptime(sys.argv[4], "%Y%m%d%H") if len(sys.argv) > 4 else dt.datetime(2017, 2, 1)
    end = dt.datetime.strptime(sys.argv[5], "%Y%m%d%H") if len(sys.argv) > 5 else dt.datetime(2017, 2, 2)
    tm = {}
    run(f"{base}/{cfg}", base, f"{base}/INPUT/NLDAS2.FORCING", sys.argv[2], start, end, timing=tm)
    print("TIMING", {k: (round(v, 2) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in tm.items()})
