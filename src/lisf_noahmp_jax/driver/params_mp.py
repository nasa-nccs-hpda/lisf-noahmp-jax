"""Tile/parameter setup for Noah-MP 4.0.1 from an LDT parameter file, mirroring NoahMP401_readcrd / _setup /
TRANSFER_MP_PARAMETERS / _coldstart (+ NOAHMP_INIT) for the dveg=4 style configuration (table LAI, FVEG=shdmax).
One tile per grid cell (dominant land cover). Parameters come from MPTABLE/SOILPARM/GENPARM; nothing time-varying
is needed except the date (phenology uses the table LAI/SAI by day of year)."""
import numpy as np
import netCDF4 as nc
from . import mp_tables as T
from .config import Config

f32 = np.float32
NL = 7
LANDUSE = {"IGBPNCEP": "MODIFIED_IGBP_MODIS_NOAH", "USGS": "USGS", "UMD": "UMD"}
OPTKEYS = [("dveg", "dynamic vegetation option"), ("crs", "canopy stomatal resistance option"),
           ("btr", "soil moisture factor for stomatal resistance"), ("run", "runoff and groundwater option"),
           ("sfc", "surface layer drag coefficient option"), ("frz", "supercooled liquid water option"),
           ("inf", "frozen soil permeability option"), ("rad", "radiation transfer option"),
           ("alb", "snow surface albedo option"), ("snf", "rainfall & snowfall option"),
           ("tbot", "lower boundary of soil temperature option"), ("stc", "snow&soil temperature time scheme option"),
           ("gla", "glacier option"), ("rsf", "surface resistance option"), ("soil", "soil configuration option"),
           ("pedo", "soil pedotransfer function option"), ("crop", "crop model option"), ("urban", "urban physics option")]


class SetupMP:
    def __init__(self, cfg_path, base_dir="."):
        self.cfg = c = Config(cfg_path)
        self.base = base_dir
        P = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
        self.nc_path = P(c.s("LIS domain and parameter data file"))
        self.model = c.s("Land surface model")
        assert self.model == "Noah-MP.4.0.1", f"unsupported land surface model {self.model}"
        M = "Noah-MP.4.0.1"
        self.o = {k: c.i(f"{M} {t}") for k, t in OPTKEYS}
        o = self.o
        supported = dict(dveg=(1, 3, 4), crs=(1, 2), btr=(1, 2, 3), run=(1, 2, 3, 4), sfc=(1, 2), frz=(1, 2), inf=(1, 2), rad=(1, 2, 3),
                         alb=(1, 2), snf=(1, 2, 3), tbot=(1, 2), stc=(1, 2, 3), rsf=(1, 2, 3, 4), soil=(1,), crop=(0,), urban=(0,))
        for k, ok in supported.items():
            assert o[k] in ok, f"Noah-MP option {k}={o[k]} not ported (supported: {ok})"
        self.nslay = c.i(f"{M} number of soil layers")
        assert self.nslay == 4
        self.sldpth = np.array(c.fl(f"{M} thickness of soil layers"), f32)
        ts = c.s(f"{M} model timestep")
        self.dt = int(ts.rstrip("mnMN")) * 60 if ts.lower().endswith("mn") else (int(ts[:-2]) * 3600 if ts.lower().endswith("hr") else int(ts))
        self.zref = c.f(f"{M} reference height of temperature and humidity")
        self.dz8w = f32(self.zref) * f32(2.0)
        d = nc.Dataset(self.nc_path)
        scheme = getattr(d, "LANDCOVER_SCHEME", "IGBPNCEP")
        d.close()
        self.lcscheme = scheme
        self.veg = T.read_mp_veg(P(c.s(f"{M} MP parameter table")), LANDUSE[scheme])
        self.radg = T.read_mp_rad_global(P(c.s(f"{M} MP parameter table")))
        self.soil, self.gen = T.read_mp_soil(P(c.s(f"{M} soil parameter table")), P(c.s(f"{M} general parameter table")))
        self.init = dict(tskin=c.f(f"{M} initial surface skin temperature"),
                         tslb=c.fl(f"{M} initial soil temperatures"), smc=c.fl(f"{M} initial total soil moistures"),
                         sneqv=c.f(f"{M} initial snow water equivalent"), snowh=c.f(f"{M} initial snow depth"),
                         canwat=c.f(f"{M} initial total canopy surface water"), lai=c.f(f"{M} initial leaf area index"),
                         zwt=c.f(f"{M} initial water table depth"), wa=c.f(f"{M} initial water in the aquifer"),
                         wt=c.f(f"{M} initial water in aquifer and saturated soil"))
        assert self.init["sneqv"] == 0.0 and self.init["snowh"] == 0.0, "coldstart with initial snow not ported"
        self.build_tiles()

    def build_tiles(self):
        d = nc.Dataset(self.nc_path)
        lc = np.array(d["LANDCOVER"][:]); tx = np.array(d["TEXTURE"][:])
        nr, ncol = lc.shape[1:]
        self.nr, self.nc = nr, ncol
        lat = np.array(d["lat"][:]); lon = np.array(d["lon"][:])
        landmask = np.array(d["LANDMASK"][:]) > 0.5
        water = self.veg["ISWATER"]
        lcl = lc.copy()
        for wc in (water, 21):          # water classes are open-water tiles, not LSM tiles
            if wc <= lcl.shape[0]:
                lcl[wc - 1] = -1.0
        dom = lcl.argmax(0) + 1
        rows, cols = np.nonzero(landmask)
        self.rows, self.cols = rows, cols
        self.glat1d, self.glon1d = lat[:, 0].astype(f32), lon[0, :].astype(f32)
        self.lat, self.lon = lat[rows, cols].astype(f32), lon[rows, cols].astype(f32)
        self.ntile = n = len(rows)
        self.vegt = dom[rows, cols].astype(np.int32)
        self.soiltyp = (tx.argmax(0) + 1)[rows, cols].astype(np.int32)
        self.tbot = np.array(d["TBOT"][:])[rows, cols].astype(f32)
        self.green = np.array(d["GREENNESS"][:])[:, rows, cols].astype(f32)        # (12, ntile) fractions
        self.static = {k: np.array(d[k][:])[rows, cols].astype(f32) for k in ("ELEVATION",)}
        self.build_param()

    def build_param(self):
        v, r, s, g = self.veg, self.radg, self.soil, self.gen
        n = self.ntile
        iv, isl = self.vegt - 1, self.soiltyp - 1
        one = lambda x: np.full(n, x, f32)
        urb = np.isin(self.vegt, [v["ISURBAN"], v["LOW_DENSITY_RESIDENTIAL"], v["HIGH_DENSITY_RESIDENTIAL"],
                                   v["HIGH_INTENSITY_INDUSTRIAL"]])
        P = {"ISWATER": one(v["ISWATER"]), "ISBARREN": one(v["ISBARREN"]), "ISICE": one(v["ISICE"]),
             "ISCROP": one(v["ISCROP"]), "EBLFOREST": one(v["EBLFOREST"]), "URBAN_FLAG": urb.astype(f32)}
        for k in ("CH2OP DLEAF Z0MVT HVT HVB DEN RC MFSNO SLA DILEFC DILEFW FRAGR LTOVRC C3PSN KC25 AKC KO25 AKO "
                  "VCMX25 AVCMX BP MP QE25 AQE RMF25 RMS25 RMR25 ARM FOLNMX TMIN XL MRP CWPVT WRRAT WDPOOL TDLEF "
                  "NROOT RGL HS TOPT RSMAX").split():
            P[k] = v[k][iv].astype(f32)
        P["RSMIN"] = v["RS"][iv].astype(f32)
        for k in ("SAIM", "LAIM", "RHOL", "RHOS", "TAUL", "TAUS"):
            P[k] = v[k][iv].astype(f32)
        soilcolor = 4
        P["ALBSAT"] = np.tile(r["ALBSAT"][soilcolor - 1], (n, 1)); P["ALBDRY"] = np.tile(r["ALBDRY"][soilcolor - 1], (n, 1))
        for k in ("ALBICE", "ALBLAK", "OMEGAS", "EG"):
            P[k] = np.tile(r[k], (n, 1))
        P["BETADS"], P["BETAIS"] = one(r["BETADS"]), one(r["BETAIS"])
        for k in ("CO2", "O2", "TIMEAN", "FSATMX", "Z0SNO", "SSI", "SWEMX", "RSURF_SNOW"):
            P[k] = one(r[k])
        for src, dst in (("BEXP", "BEXP"), ("SMCDRY", "SMCDRY"), ("SMCWLT", "SMCWLT"), ("SMCREF", "SMCREF"),
                         ("SMCMAX", "SMCMAX"), ("PSISAT", "PSISAT"), ("DKSAT", "DKSAT"), ("DWSAT", "DWSAT"),
                         ("QUARTZ", "QUARTZ")):
            P[dst] = np.tile(s[src][isl][:, None], (1, 4)).astype(f32)
        P["F1"] = s["F1"][isl].astype(f32)
        P["REFDK"], P["REFKDT"] = one(g["REFDK"]), one(g["REFKDT"])
        P["CSOIL"], P["ZBOT"], P["CZIL"] = one(g["CSOIL"]), one(g["ZBOT"]), one(g["CZIL"])
        P["KDT"] = (P["REFKDT"] * P["DKSAT"][:, 0] / P["REFDK"]).astype(f32)
        P["SLOPE"] = one(g["SLOPE"][0])
        u = urb[:, None]
        P["SMCMAX"] = np.where(u, f32(0.45), P["SMCMAX"]); P["SMCREF"] = np.where(u, f32(0.42), P["SMCREF"])
        P["SMCWLT"] = np.where(u, f32(0.40), P["SMCWLT"]); P["SMCDRY"] = np.where(u, f32(0.40), P["SMCDRY"])
        P["CSOIL"] = np.where(urb, f32(3.0e6), P["CSOIL"])
        frzfact = (P["SMCMAX"][:, 0] / P["SMCREF"][:, 0]) * f32(0.412 / 0.468)
        P["FRZX"] = np.where(self.soiltyp != 14, g["FRZK"] * frzfact, f32(0.0)).astype(f32)
        P["MXSNALB"], P["MNSNALB"], P["sndecayexp"] = one(0.84), one(0.55), one(0.01)
        P["T_ulimit"], P["T_mlimit"], P["T_llimit"], P["snowf_scalef"] = one(2.5), one(2.0), one(0.5), one(1.0)
        self.par = {k: np.ascontiguousarray(a, f32) for k, a in P.items()}
        self.glacier_idx = np.nonzero(self.vegt == v['ISICE'])[0].astype(np.int32)
        self.fvgmax = (np.max(self.green * f32(100.0), axis=0) / f32(100.0)).astype(f32)
        self.zsoil = -np.cumsum(self.sldpth).astype(f32)

    # ------------------------------------------------------------------
    def initial_state(self):
        """NoahMP401_coldstart + NOAHMP_INIT (no initial snow). Arrays have a leading tile axis."""
        n, c, v, s = self.ntile, self.init, self.veg, self.soil
        one = lambda x: np.full(n, x, f32)
        isl = self.soiltyp - 1
        iv = self.vegt - 1
        tsk = c["tskin"]
        st = {}
        tslb = np.tile(np.array(c["tslb"], f32), (n, 1))
        smc = np.tile(np.array(c["smc"], f32), (n, 1))
        smcmax = self.par["SMCMAX"]          # same table values as SMCMAX_TABLE(ISLTYP) except urban overrides
        smcmax_t = np.tile(s["SMCMAX"][isl][:, None], (1, 4))
        smc = np.minimum(smc, smcmax_t)
        HLICE, GRAV, T0 = f32(3.335e5), f32(9.80616), f32(273.16)
        bexp, psisat = s["BEXP"][isl][:, None], s["PSISAT"][isl][:, None]
        with np.errstate(all="ignore"):
            fk = (((HLICE / (GRAV * (-psisat))) * ((tslb - T0) / tslb)) ** (f32(-1.0) / bexp)) * smcmax_t
        fk = np.maximum(fk, f32(0.02))
        sh2o = np.where(tslb < 273.149, np.minimum(fk, smc), smc).astype(f32)
        st["SMH2O"], st["SMC"], st["STC"] = sh2o, smc.astype(f32), None
        lai0 = np.maximum(f32(c["lai"]), f32(0.05))
        bare = np.isin(self.vegt, [v["ISBARREN"], v["ISICE"], v["ISURBAN"], v["ISWATER"]])
        st["PLAI"] = np.where(bare, f32(0), lai0).astype(f32)
        st["PSAI"] = np.where(bare, f32(0), np.maximum(f32(0.1) * lai0, f32(0.05))).astype(f32)
        st["STC"] = np.concatenate([np.zeros((n, 3), f32), tslb], 1)        # snow layers 0, soil layers
        zs = np.concatenate([np.zeros(3, f32), self.zsoil])
        st["ZSNSO"] = np.tile(zs, (n, 1))
        st["SNICE"], st["SNLIQ"] = np.zeros((n, 3), f32), np.zeros((n, 3), f32)
        st["ISNOW"] = np.zeros(n, np.int32)
        st["TV"], st["TG"] = one(tsk), one(tsk)
        st["TAH"], st["EAH"] = one(tsk), one(2000.0)
        st["CM"], st["CH"] = one(0.0), one(0.0)
        for k in ("FWET CANLIQ CANICE QSNOW SNDPTH SWE SNEQVO WSLAKE TAUSS SMCWTD DEEPRECH RECH GDD SFCRUNOFF UDRRUNOFF "
                  "ACSNOM ACSNOW").split():
            st[k] = one(0.0)
        st["GRAIN"] = one(1e-10)
        # NOAHMP_INIT carbon pools (not used by dveg=4, but part of the LIS restart)
        masslai = f32(1000.) / np.maximum(v["SLA"][iv], f32(1.0))
        st["LFMASS"] = np.where(bare, f32(0), st["PLAI"] * masslai).astype(f32)
        st["STMASS"] = np.where(bare, f32(0), st["PSAI"] * (f32(1000.) / f32(3.0))).astype(f32)
        for k, val in (("RTMASS", 500.0), ("WOOD", 500.0), ("STBLCP", 1000.0), ("FASTCP", 1000.0)):
            st[k] = np.where(bare, f32(0), f32(val)).astype(f32)
        st["SMOISEQ"] = np.zeros((n, 4), f32)
        st["ALBOLD"] = one(0.65)
        wa = f32(4900.0)
        st["WA"], st["WT"] = one(wa), one(wa)
        st["ZWT"] = one((f32(25.0) + f32(2.0)) - wa / f32(1000) / f32(0.2))
        # land-ice tiles (NOAHMP_INIT + snow_init): ice starts all frozen with >= 10 mm snow
        g = self.vegt == v["ISICE"]
        if g.any():
            st["SMC"][g] = f32(1.0); st["SMH2O"][g] = f32(0.0)
            tslb_g = np.minimum(tslb[g], f32(263.15))
            tg0 = f32(tsk) if not (tsk > 273.15) else f32(273.15)       # snow > 0 and tsk > 273.15 -> 273.15
            st["TV"][g] = tg0; st["TG"][g] = tg0; st["TAH"][g] = tg0
            swe = f32(10.0); snodep = f32(swe * f32(0.01))
            st["SWE"][g] = swe; st["SNDPTH"][g] = snodep
            # snow_init_401 for snodep = 0.1 (two layers of 0.05 m); other depths not needed for the 10 mm start
            assert abs(snodep - 0.1) < 1e-6
            ng = int(g.sum())
            stc_g = np.zeros((ng, NL), f32)
            stc_g[:, 3:] = tslb_g
            stc_g[:, 1:3] = tg0
            st["STC"][g] = stc_g
            st["ISNOW"][g] = -2
            dzs = np.array([0.0, 0.05, 0.05], f32)
            snice = np.zeros((ng, 3), f32); snice[:, 1:] = f32(1.0) * f32(0.05) * (swe / snodep)
            st["SNICE"][g] = snice
            zs = np.zeros((ng, NL), f32)
            zs[:, 1] = -dzs[1]; zs[:, 2] = zs[:, 1] - dzs[2]
            dzsoil = np.concatenate([[self.zsoil[0]], np.diff(self.zsoil)]).astype(f32)
            for k in range(4):
                zs[:, 3 + k] = zs[:, 2 + k] + dzsoil[k]
            st["ZSNSO"][g] = zs
        return st
