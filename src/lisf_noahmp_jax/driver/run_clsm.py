"""End-to-end LIS-style run of the Catchment LSM Fortuna-2.5 in JAX: config + LDT parameter file + NLDAS-2 forcing ->
LIS history files (SURFACEMODEL/). Mirrors clsmf25_main.F90 / clsmf25_f2t.F90 / clsmf25_setup.F90 / clsmf25_readrst.F90
(coldstart) for the configuration of dev/lisf_run/conus_clsm (Louis turbulence, MODIS albedo scale factors, one tile per cell)."""
import sys, os, time, datetime as dt
import numpy as np
import jax, jax.numpy as jnp
import netCDF4 as nc

from ..physics import clsm_jax as C
from lisf_noahmp_jax.driver.config import Config
from lisf_noahmp_jax.driver.forcing import NLDAS2, _coszenith, _localtime
from lisf_noahmp_jax.driver import timeutil as tu

f32 = np.float32
UDEF = f32(-9999.0)
N_GT, N_SNOW = 6, 3
D0_BY_ZVEG, Z0_BY_ZVEG, DZE_MIN = f32(0.66), f32(0.13), f32(10.)
EMSVEG = f32([0.99560, 0.99000, 0.99560, 0.99320, 0.99280, 0.99180, 0.94120, 0.94120])
EMSSNO = f32(0.99999)
TOTALB_MIN, TOTALB_MAX = f32(0.01), f32(0.90)


class SetupCLSM:
    def __init__(self, cfg_path, base):
        self.cfg = c = Config(cfg_path)
        self.base = base
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
        nveg = int(d.getncattr("NUMVEGTYPES"))
        self.vegcls = (lc[:nveg].argmax(0) + 1)[rows, cols].astype(np.int32)
        v = lambda k: np.array(d[k][:]).astype(f32)[rows, cols]
        self.bee, self.psis, self.poros, self.cond = v("BEXP"), v("PSISAT"), v("POROSITY"), v("KSAT")
        self.wpwet, self.dpth = v("WPWET"), v("BEDROCKDEPTH")
        for k in ("gnu:GNUCLSM", "ars1:ARS1CLSM", "ars2:ARS2CLSM", "ars3:ARS3CLSM", "ara1:ARA1CLSM", "ara2:ARA2CLSM",
                  "ara3:ARA3CLSM", "ara4:ARA4CLSM", "arw1:ARW1CLSM", "arw2:ARW2CLSM", "arw3:ARW3CLSM", "arw4:ARW4CLSM",
                  "bf1:BF1CLSM", "bf2:BF2CLSM", "bf3:BF3CLSM", "tsa1:TSA1CLSM", "tsa2:TSA2CLSM", "tsb1:TSB1CLSM",
                  "tsb2:TSB2CLSM", "atau:ATAUCLSM", "btau:BTAUCLSM"):
            a, b = k.split(":")
            setattr(self, a, v(b))
        m = lambda k: np.array(d[k][:]).astype(f32)[:, rows, cols]
        self.green12, self.lai12 = m("GREENNESS"), m("LAI")
        self.albscale = np.stack([m("ALBVISDIR"), m("ALBVISDIFF"), m("ALBNIRDIR"), m("ALBNIRDIFF")], 1)     # (12, 4, N)
        self.dt = int(round(c.f("CLSM F2.5 model timestep") if False else self._ts()))
        # clsmf25_compute_land_parameters
        self.dzsf = np.full(n, f32(c.f("CLSM F2.5 top soil layer depth")) * f32(1000.), f32)
        self.dzpr = np.maximum(f32(1000.), self.dpth).astype(f32)
        dzrz = np.full(n, f32(1000.), f32)
        dzrz = np.where(dzrz > f32(0.75) * self.dzpr, f32(0.75) * self.dzpr, dzrz).astype(f32)
        self.dzrz = dzrz
        self.vgwmax = (self.poros * dzrz).astype(f32)
        z = (self.dzpr / f32(1000.)).astype(f32)
        term1 = (f32(-1.) + np.power((self.psis - z) / self.psis, (self.bee - f32(1.)) / self.bee)).astype(f32)
        term2 = (self.psis * self.bee / (self.bee - f32(1.))).astype(f32)
        self.cdcr1 = (f32(1000.) * self.poros * (z - (-term2 * term1))).astype(f32)
        self.cdcr2 = ((f32(1.) - self.wpwet) * self.poros * self.dzpr).astype(f32)
        self.init_sm, self.init_st = f32(c.f("CLSM F2.5 initial soil moisture")), f32(c.f("CLSM F2.5 initial soil temperature"))
        self.refh = f32(c.f("CLSM F2.5 fixed reference height"))
        assert c.i("CLSM F2.5 turbulence scheme") == 0, "only the Louis scheme is ported"
        assert c.i("CLSM F2.5 use MODIS albedo flag") == 1

    def _ts(self):
        from lisf_noahmp_jax.driver.run_mp import parse_interval
        return parse_interval(self.cfg.s("CLSM F2.5 model timestep"))

    def params(self):
        keys = "bee psis poros cond wpwet gnu ars1 ars2 ars3 ara1 ara2 ara3 ara4 arw1 arw2 arw3 arw4 bf1 bf2 bf3 tsa1 tsa2 tsb1 tsb2 atau btau dzsf dzpr dzrz vgwmax cdcr1 cdcr2".split()
        P = {k: getattr(self, k) for k in keys}
        P["ityp"] = self.vegcls
        P["lat"] = self.lat
        P["albscale"] = self.albscale
        return P

    def initial_state(self):
        n = self.ntile
        one = lambda x: np.full(n, x, f32)
        st = {"tc1": one(self.init_st), "tc2": one(self.init_st), "tc4": one(self.init_st), "qa1": one(0.002), "qa2": one(0.002),
              "qa4": one(0.002), "capac": one(0.0), "rzexc": one(0.), "srfexc": one(0.)}
        st["catdef"] = (np.maximum(self.poros - self.init_sm, f32(0.05)) * self.dzpr).astype(f32)
        dzgt = [f32(x) for x in (0.0988, 0.1952, 0.3859, 0.7626, 1.5071, 10.0)]
        st["ght"] = np.stack([((self.init_st - f32(273.16)) * (f32(2.4e6) * f32(0.55) * dzgt[k] + f32(0.5) * f32(0.45) * dzgt[k]
                                                                  * f32(4.185e6))) * np.ones(n, f32) for k in range(N_GT)]).astype(f32)
        for k in ("wesn", "htsn", "sndz"):
            st[k] = np.zeros((N_SNOW, n), f32)
        return st


def make_step(S, sfrac=f32(1.0), debug=False):
    dtstep = f32(S.dt)

    def step(par, st, f):
        g = lambda k: par[k]
        n = par["bee"].shape[0]
        ityp = par["ityp"]
        tair, qair, psurf = f["tair"], f["qair"], f["psurf"]
        rainf, rainf_c, snowf = f["rainf"], f["rainf_c"], f["snowf"]
        lwd, swd, wind = f["lwd"], f["swd"], f["wind"]
        refh = f["refh"]
        sunang = f["sunang"]
        green = f["green"]
        lai = jnp.where(f["lai"] <= 0., 0.001, f["lai"])
        wesn = [st["wesn"][i] for i in range(3)]; htsn = [st["htsn"][i] for i in range(3)]; sndz = [st["sndz"][i] for i in range(3)]
        ght = [st["ght"][i] for i in range(6)]
        tc1, tc2, tc4, qa1, qa2, qa4 = st["tc1"], st["tc2"], st["tc4"], st["qa1"], st["qa2"], st["qa4"]
        # tpsn(1), ar1, ar2, asnow
        t0, _ = C.get_tf(htsn[0], wesn[0])
        tpsn1 = t0 + C.TF
        ars = [g("ars1"), g("ars2"), g("ars3")]; ara = [g("ara1"), g("ara2"), g("ara3"), g("ara4")]
        arw = [g("arw1"), g("arw2"), g("arw3"), g("arw4")]
        rzeq0 = C.rzequil(st["catdef"], g("vgwmax"), g("cdcr1"), g("cdcr2"), g("wpwet"), ars, ara, arw)
        pt0 = C.partition(g("dzsf"), st["rzexc"], rzeq0, g("vgwmax"), g("cdcr1"), g("cdcr2"), g("psis"), g("bee"), g("poros"),
                          g("wpwet"), ars, ara, arw, st["srfexc"], st["catdef"], 0. * lai, jnp.float32(-9999.))
        asnow_a = jnp.fmin(C.sum_list(wesn) / C.WEMIN, 1.)
        # vegetation parameters
        prev = {k: 0. * lai for k in ("sqscat", "z2", "dzm", "rsoil1", "rsoil2", "satcap", "rdc", "z0")}
        pm = C.pmonth(ityp, f["jday"], g("lat"), green, lai, prev)
        zol = pm["z0"]
        trainl = rainf - rainf_c
        d0 = D0_BY_ZVEG * zol / Z0_BY_ZVEG
        dze = jnp.fmax(refh - d0, DZE_MIN)
        qa_atm = qair
        qs_tpsn = C.qsat(tpsn1, psurf / 100.)
        tb = C.turb_louis(tair, qair, psurf, rainf_c, wind, lai, zol, dze, [tc1, tc2, tc4, tpsn1], [qa1, qa2, qa4, qs_tpsn])
        # keep qa between qair and qsat (clsmf25_main)
        def fixq(qa, tc):
            qsattc = C.qsat(tc, psurf / 100.)
            c1 = (qa > qsattc) & (qa > qair)
            qa = jnp.where(c1, jnp.fmax(qair, qsattc), qa)
            c2 = (qa < qsattc) & (qa < qair)
            return jnp.where(c2, jnp.fmin(qair, qsattc), qa)
        qa1, qa2, qa4 = fixq(qa1, tc1), fixq(qa2, tc2), fixq(qa4, tc4)
        # albedo
        al = C.sibalb(ityp, lai, green, sunang, f["albsc"][0], f["albsc"][1], f["albsc"][2], f["albsc"][3], wesn, sndz)
        albedo = jnp.fmin(1., 0.25 * (al["avisdr"] + al["anirdr"] + al["avisdf"] + al["anirdf"]))
        albsn = jnp.fmin(1., 0.25 * (al["asnvdr"] + al["asnndr"] + al["asnvdf"] + al["asnndf"]))
        asnow = asnow_a
        swnetf = (1. - albedo) * swd
        swnets = (1. - albsn) * swd
        tmpswnet = (1 - asnow) * swnetf + asnow * swnets
        daytime = (swd > 0.) & (sunang > 0.)
        swnetf = jnp.where(daytime, swnetf, 0.); swnets = jnp.where(daytime, swnets, 0.)
        totalb = jnp.where(daytime, jnp.fmin(jnp.fmax(1. - tmpswnet / jnp.where(daytime, swd, 1.), C.n32(0.01)), C.n32(0.90)), UDEF)
        swup = jnp.where(daytime, totalb * swd, 0.)
        pardir = 0.5 * 0.5 * swd
        pardif = pardir
        qsat1 = C.qsat(tc1, psurf / 100.); qsat2 = C.qsat(tc2, psurf / 100.); qsat4 = C.qsat(tc4, psurf / 100.)
        qsats = C.qsat(tpsn1, psurf / 100.)
        emsv = jnp.asarray(EMSVEG)
        emis = emsv[jnp.minimum(ityp - 1, 7)] + (emsv[7] - emsv[jnp.minimum(ityp - 1, 7)]) * jnp.exp(-lai)
        emis = emis * (1. - asnow) + EMSSNO * asnow

        def lw(t):
            b = emis * C.STEFAN * t * t * t
            a = -3. * t * b
            return a, 4. * b
        alw1, blw1 = lw(tc1); alw2, blw2 = lw(tc2); alw4, blw4 = lw(tc4); alws, blws = lw(tpsn1)
        zvg = zol / Z0_BY_ZVEG
        uuu = jnp.fmax(wind, C.USMIN) * (jnp.log((jnp.fmax(zvg - d0, 0.) + zol) / zol)
                                         / jnp.log((jnp.fmax(refh - d0, 10.) + zol) / zol))
        x = dict(ityp=ityp, dzsf=g("dzsf"), trainc=rainf_c, trainl=trainl, tsnow=snowf, um=uuu,
                 tm=tair, qm=qair, sunang=sunang, pardir=pardir, pardif=pardif, swnetf=swnetf, swnets=swnets,
                 hlwdwn=lwd, psur=psurf / 100., zlai=lai, green=green, sqscat=pm["sqscat"], rsoil1=pm["rsoil1"],
                 rsoil2=pm["rsoil2"], rdc=pm["rdc"],
                 qsat1=qsat1, dqs1=0. * lai, alw1=alw1, blw1=blw1, qsat2=qsat2, dqs2=0. * lai, alw2=alw2, blw2=blw2,
                 qsat4=qsat4, dqs4=0. * lai, alw4=alw4, blw4=blw4, qsats=qsats, dqss=0. * lai, alws=alws, blws=blws,
                 bf1=g("bf1"), bf2=g("bf2"), bf3=g("bf3"), vgwmax=g("vgwmax"), cdcr1=g("cdcr1"), cdcr2=g("cdcr2"),
                 psis=g("psis"), bee=g("bee"), poros=g("poros"), wpwet=g("wpwet"), cond=g("cond"), gnu=g("gnu"),
                 ars1=g("ars1"), ars2=g("ars2"), ars3=g("ars3"), ara1=g("ara1"), ara2=g("ara2"), ara3=g("ara3"),
                 ara4=g("ara4"), arw1=g("arw1"), arw2=g("arw2"), arw3=g("arw3"), arw4=g("arw4"), tsa1=g("tsa1"),
                 tsa2=g("tsa2"), tsb1=g("tsb1"), tsb2=g("tsb2"), atau=g("atau"), btau=g("btau"),
                 tc1=tc1, tc2=tc2, tc4=tc4, qa1=qa1, qa2=qa2, qa4=qa4, capac=st["capac"], catdef=st["catdef"],
                 rzexc=st["rzexc"], srfexc=st["srfexc"], ght=ght, wesn=wesn, htsn=htsn, sndz=sndz)
        x.update({k: v for k, v in tb.items()})
        out = C.clsmf25(dtstep, sfrac, x)
        new = dict(tc1=out["tc1"], tc2=out["tc2"], tc4=out["tc4"], qa1=out["qa1"], qa2=out["qa2"], qa4=out["qa4"],
                   capac=out["capac"], catdef=out["catdef"], rzexc=out["rzexc"], srfexc=out["srfexc"],
                   ght=jnp.stack(out["ght"]), wesn=jnp.stack(out["wesn"]), htsn=jnp.stack(out["htsn"]),
                   sndz=jnp.stack(out["sndz"]))
        tpsn = out["tpsn1"]
        snocovr = jnp.fmin(C.sum_list(out["wesn"]) / C.WEMIN, 1.)
        diag = {
            "Swnet": swd - swup, "Lwnet": lwd - out["hlwup"], "Qle": out["hlatn"] - out["esno"], "Qh": out["shflux"],
            "Qg": out["ghflux"], "Snowf": snowf, "Rainf": rainf, "Evap": out["evap"], "Qs": out["runsrf"], "Qsb": out["bflow"],
            "AvgSurfT": out["tsurf"], "Albedo": totalb, "SWE": C.sum_list(out["wesn"]),
            "SnowDepth": C.sum_list(out["sndz"]) * snocovr,
            "SoilMoist1": out["sfmc"] * g("dzsf"), "SoilMoist2": out["rzmc"] * g("dzrz"), "SoilMoist3": out["prmc"] * g("dzpr"),
            "CanopInt": out["capac"], "Snowcover": snocovr, "Wind_f": wind, "Rainf_f": rainf + snowf, "Tair_f": tair, "Qair_f": qair,
            "Psurf_f": psurf, "SWdown_f": swd, "LWdown_f": lwd, "Greenness": green,
        }
        for k in range(6):
            diag[f"SoilTemp{k+1}"] = out["tp"][k] + C.TF
        if debug:
            return new, diag, x
        return new, diag
    return step


class CLSMHist:
    TAVG = "Swnet Lwnet Qle Qh Qg Snowf Rainf Evap Qs Qsb".split()
    INST = ("AvgSurfT Albedo SWE SnowDepth SoilMoist SoilTemp CanopInt Snowcover Wind_f Rainf_f Tair_f Qair_f Psurf_f SWdown_f "
            "LWdown_f Greenness").split()
    LAY = {"SoilMoist": 3, "SoilTemp": 6}
    UNITS = {"Swnet": "W m-2", "Lwnet": "W m-2", "Qle": "W m-2", "Qh": "W m-2", "Qg": "W m-2", "Snowf": "kg m-2 s-1",
             "Rainf": "kg m-2 s-1", "Evap": "kg m-2 s-1", "Qs": "kg m-2 s-1", "Qsb": "kg m-2 s-1", "AvgSurfT": "K"}

    @staticmethod
    def write(path, S, label, tavg, inst):
        ds = nc.Dataset(path, "w", format="NETCDF4")
        ds.createDimension("east_west", S.nc); ds.createDimension("north_south", S.nr); ds.createDimension("time", 1)
        for v, k in CLSMHist.LAY.items():
            ds.createDimension(f"{v}_profiles", k)
        ds.missing_value = f32(-9999.0); ds.title = "LIS land surface model output (Python/JAX port, CLSM F2.5)"
        ds.MAP_PROJECTION = "EQUIDISTANT CYLINDRICAL"; ds.DX = f32(0.125); ds.DY = f32(0.125)
        ds.SOUTH_WEST_CORNER_LAT = f32(S.glat1d.min()); ds.SOUTH_WEST_CORNER_LON = f32(S.glon1d.min())
        la = ds.createVariable("lat", "f4", ("north_south",)); lo = ds.createVariable("lon", "f4", ("east_west",))
        tv = ds.createVariable("time", "f4", ("time",)); tv.units = f"minutes since {label:%Y-%m-%d %H:%M:%S}"; tv[:] = 0.0
        la[:] = S.glat1d; lo[:] = S.glon1d

        def grid(a):
            gg = np.full((S.nr, S.nc), -9999.0, f32); gg[S.rows, S.cols] = a; return gg

        def put(name, arr, layered):
            dims = (name.rsplit("_", 1)[0] + "_profiles", "north_south", "east_west") if layered else ("north_south", "east_west")
            x = ds.createVariable(name, "f4", dims, fill_value=f32(-9999.0))
            x[:] = np.stack([grid(a) for a in arr]) if layered else grid(arr)
        for v in CLSMHist.TAVG:
            put(v + "_tavg", tavg[v], False)
        for v in CLSMHist.INST:
            if v in CLSMHist.LAY:
                put(v + "_inst", [inst[f"{v}{k}"] for k in range(1, CLSMHist.LAY[v] + 1)], True)
            else:
                put(v + "_inst", inst[v], False)
        ds.close()


def run(cfg_path, base_dir, forcing_dir, out_dir, start, end, out_interval_h=6, progress=True, timing=None):
    tm_ = {"setup": 0.0, "forcing_host": 0.0, "device": 0.0, "output": 0.0, "chunk_device_s": []}
    t_all = time.time(); t = time.time()
    S = SetupCLSM(cfg_path, base_dir)
    P_ = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
    src = S.cfg.s("Met forcing sources")
    assert src.startswith("NLDAS2")
    F = NLDAS2(P_(S.cfg.s("NLDAS-2.0 FORA forcing directory")), S.lat, S.lon, extra=("CRainf_frac",))
    step = C.jax.jit(make_step(S))
    par = {k: jnp.asarray(v) for k, v in S.params().items()}
    st = {k: jnp.asarray(v) for k, v in S.initial_state().items()}
    zero = np.zeros(S.ntile, f32)
    nstep = int((end - start).total_seconds() // S.dt)
    K = out_interval_h * 3600 // S.dt

    @jax.jit
    def chunk(par, st, fK):
        def body(carry, f):
            st, acc, last = carry
            st2, diag = step(par, st, f)
            acc = {k: acc[k] + diag[k] for k in acc}
            # instantaneous output keeps the last defined value of the window (LIS ignores UDEF in diagnoseDataEntry)
            diag = dict(diag, Albedo=jnp.where(diag["Albedo"] != UDEF, diag["Albedo"], last["Albedo"]))
            return (st2, acc, diag), None
        d0 = jax.eval_shape(lambda: step(par, st, jax.tree_util.tree_map(lambda a: a[0], fK))[1])
        acc0 = {k: jnp.zeros(a.shape, a.dtype) for k, a in d0.items()}
        last0 = {k: jnp.zeros(a.shape, a.dtype) for k, a in d0.items()}
        last0["Albedo"] = jnp.full(d0["Albedo"].shape, UDEF, d0["Albedo"].dtype)
        (st2, acc, last), _ = jax.lax.scan(body, (st, acc0, last0), fK)
        return st2, acc, last
    tm_["setup"] += time.time() - t
    for c0 in range(0, nstep, K):
        t = time.time()
        times = [start + dt.timedelta(seconds=S.dt * (s + 1)) for s in range(c0, min(c0 + K, nstep))]
        fs = []
        for now in times:
            o = F.at(now)
            tair = o["Tair"]
            warm = tair >= f32(273.16)
            pcp = o["Rainf"]
            cpcp = o["CRainf_frac"] * pcp
            rainf = np.where(warm, pcp, f32(0)); rainf_c = np.where(warm, cpcp, f32(0)); snowf = np.where(warm, f32(0), pcp)
            rainf = np.maximum(f32(0), rainf); rainf_c = np.minimum(np.maximum(f32(0), rainf_c), rainf)
            wind = np.maximum(f32(0.0001), np.sqrt(o["Wind_E"] * o["Wind_E"] + o["Wind_N"] * o["Wind_N"]))
            gmt = f32(now.hour) + f32(now.minute) / f32(60.) + f32(now.second) / f32(3600.)
            lhour, zone = _localtime(gmt, S.lon)
            cosz = _coszenith(S.lon, S.lat, lhour, zone, tu.doy(now))
            sunang = np.maximum(cosz, f32(0.01))
            # monthly climatologies (LIS_computeTemporalWeights)
            julm = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
            mo2 = now.month + (1 if now.day > 15 else 0)
            if mo2 == 1:
                mo2 = 13
            mo1 = mo2 - 1
            juld = julm[now.month - 1] + now.day + (1 if (now.month == 3 and now.year % 4 == 0) else 0)
            if juld <= 15:
                juld += 365
            day2 = f32(julm[mo2 - 1] + 15 + (1 if (mo2 == 3 and now.year % 4 == 0) else 0)); day1 = f32(julm[mo1 - 1] + 15)
            wt1 = (day2 - f32(juld)) / (day2 - day1); wt2 = (f32(juld) - day1) / (day2 - day1)
            m1 = (mo1 - 1) % 12; m2 = (mo2 - 1) % 12
            green = wt1 * S.green12[m1] + wt2 * S.green12[m2]
            lai = wt1 * S.lai12[m1] + wt2 * S.lai12[m2]
            fs.append(dict(tair=tair, qair=o["Qair"], psurf=o["PSurf"], rainf=rainf, rainf_c=rainf_c, snowf=snowf, lwd=o["LWdown"],
                           swd=o["SWdown"], wind=wind, refh=np.full(S.ntile, S.refh, f32), sunang=sunang.astype(f32),
                           green=green.astype(f32), lai=lai.astype(f32), albsc=S.albscale[now.month - 1],
                           jday=np.full(S.ntile, tu.doy(now), np.int32)))
        fK = {k: jnp.asarray(np.stack([f[k] for f in fs])) for k in fs[0]}
        tm_["forcing_host"] += time.time() - t
        t = time.time()
        st, acc, last = chunk(par, st, fK)
        jax.block_until_ready(acc)
        tm_["device"] += time.time() - t; tm_["chunk_device_s"].append(time.time() - t)
        t = time.time()
        n = f32(len(times))
        tavg = {k: np.asarray(acc[k]) / n for k in CLSMHist.TAVG}
        inst = {k: np.asarray(v) for k, v in last.items()}
        now = times[-1]
        d = f"{out_dir}/SURFACEMODEL/{now:%Y%m}"; os.makedirs(d, exist_ok=True)
        CLSMHist.write(f"{d}/LIS_HIST_{now:%Y%m%d%H%M}.d01.nc", S, now, tavg, inst)
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
