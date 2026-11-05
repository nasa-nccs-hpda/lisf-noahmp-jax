"""Tile/parameter setup for Noah-MP 3.6 from an LDT parameter file, mirroring NoahMP36_readcrd / _setup / _coldstart and the
REDPRM / read_mp_veg_parameters / SOIL_VEG_GEN_PARM_36 table handling, for the dveg=4 style configuration (table LAI, FVEG=max
monthly greenness). One tile per grid cell (dominant land cover). Parameter names are those of the 4.0.1 parameter dictionary so that
noahmp36_jax.py and the replay tool can use them unchanged."""
import numpy as np
import netCDF4 as nc
from . import mp_tables as T, tables
from .config import Config
from .params_mp import SetupMP

f32 = np.float32
NL = 7
LUTYPE = {"IGBPNCEP": "MODIFIED_IGBP_MODIS_NOAH", "USGS": "USGS", "UMD": "UMD"}
GRP = {"MODIFIED_IGBP_MODIS_NOAH": "modis", "USGS": "usgs", "UMD": "umd"}
OPTKEYS = [("dveg", "vegetation model option"), ("crs", "canopy stomatal resistance option"),
           ("btr", "soil moisture factor for stomatal resistance option"), ("run", "runoff and groundwater option"),
           ("sfc", "surface layer drag coefficient option"), ("frz", "supercooled liquid water option"),
           ("inf", "frozen soil permeability option"), ("rad", "radiation transfer option"),
           ("alb", "snow surface albedo option"), ("snf", "rainfall and snowfall option"),
           ("tbot", "lower boundary of soil temperature option"), ("stc", "snow and soil temperature time scheme")]


def read_mp36_veg(path, scheme):
    g = T.read_namelists(path)
    key = GRP[scheme]
    cat = g[f"noah_mp_{key}_veg_categories"]
    par = g[f"noah_mp_{key}_parameters"]
    nveg = cat["NVEG"][0]
    d = {k: par[k][0] for k in ("ISURBAN", "ISWATER", "ISBARREN", "ISSNOW", "EBLFOREST")}
    arr = lambda k: np.array(par[k][:nveg], f32)
    for k in ("CH2OP DLEAF Z0MVT HVT HVB RC XL CWPVT C3PSN KC25 AKC KO25 AKO AVCMX AQE LTOVRC DILEFC DILEFW RMF25 SLA FRAGR "
              "TMIN VCMX25 TDLEF BP MP QE25 RMS25 RMR25 ARM FOLNMX WDPOOL WRRAT MRP").split():
        d[k] = arr(k)
    for k in ("RHOL", "RHOS", "TAUL", "TAUS"):
        d[k] = np.array(par[k][:2 * nveg], f32).reshape(2, nveg).T
    for k in ("SAIM", "LAIM"):
        d[k] = np.array(par[k][:12 * nveg], f32).reshape(12, nveg).T
    return d


class SetupMP36(SetupMP):
    def __init__(self, cfg_path, base_dir="."):
        self.cfg = c = Config(cfg_path)
        self.base = base_dir
        P = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
        self.nc_path = P(c.s("LIS domain and parameter data file"))
        self.model = c.s("Land surface model")
        if not (self.model == "NoahMP.3.6"):
            raise ValueError(f"unsupported land surface model {self.model}")
        M = "Noah-MP.3.6"
        self.v36mp = True
        self.o = {k: c.i(f"{M} {t}") for k, t in OPTKEYS}
        self.o["rsf"] = 1; self.o["soil"] = 1; self.o["crop"] = 0; self.o["gla"] = 1
        o = self.o
        supported = dict(dveg=(1, 3, 4), crs=(1, 2), btr=(1, 2, 3), run=(1, 2, 3, 4), sfc=(1, 2), frz=(1, 2), inf=(1, 2), rad=(1, 2, 3),
                         alb=(1, 2), snf=(1, 2, 3), tbot=(1, 2), stc=(1, 2))
        for k, ok in supported.items():
            if not (o[k] in ok):
                raise ValueError(f"Noah-MP 3.6 option {k}={o[k]} not ported (supported: {ok})")
        assert c.i(f"{M} soil color index") == 4 and c.i(f"{M} CZIL option (iz0tlnd)") == 0
        self.nslay = c.i(f"{M} number of soil layers")
        assert self.nslay == 4
        self.sldpth = np.array(c.fl(f"{M} soil layer thickness"), f32)
        ts = c.s(f"{M} model timestep")
        self.dt = int(ts.rstrip("mnMN")) * 60 if ts.lower().endswith("mn") else (int(ts[:-2]) * 3600 if ts.lower().endswith("hr") else int(ts))
        self.zref = c.f(f"{M} initial reference height of temperature and humidity")
        self.dz8w = f32(self.zref) * f32(2.0)
        d = nc.Dataset(self.nc_path)
        scheme = getattr(d, "LANDCOVER_SCHEME", "IGBPNCEP")
        d.close()
        self.lcscheme = scheme
        lu = LUTYPE[scheme]
        self.veg = read_mp36_veg(P(c.s(f"{M} MP parameter table")), lu)
        self.vegtbl = tables.read_veg(P(c.s(f"{M} landuse parameter table")), lu)      # NROOT RS RGL HS + TOPT RSMAX (VEGPARM.TBL)
        self.soil36 = tables.read_soil(P(c.s(f"{M} soil parameter table")), "STAS")
        self.gen = tables.read_gen(P(c.s(f"{M} general parameter table")))
        self.i36 = {k: c.f(f"{M} initial {t}") for k, t in (
            ("albold", "value of snow albedo at the last timestep"), ("sneqvo", "value of snow mass at the last timestep"),
            ("tah", "canopy air temperature"), ("eah", "canopy air vapor pressure"), ("fwet", "wetted or snowed fraction of canopy"),
            ("canliq", "intercepted liquid water"), ("canice", "intercepted ice mass"), ("tv", "vegetation temperature"),
            ("tg", "ground temperature"), ("qsnow", "snowfall on the ground"), ("snowh", "snow height"),
            ("sneqv", "snow water equivalent"), ("zwt", "depth to water table"), ("wa", "water storage in aquifer"),
            ("wt", "water in aquifer and saturated soil"), ("wslake", "lake water storage"), ("lfmass", "leaf mass"),
            ("rtmass", "mass of fine roots"), ("stmass", "stem mass"), ("wood", "mass of wood including woody roots"),
            ("stblcp", "stable carbon in deep soil"), ("fastcp", "short-lived carbon in shallow soil"), ("lai", "LAI"),
            ("sai", "SAI"), ("cm", "momentum drag coefficient"), ("ch", "sensible heat exchange coefficient"),
            ("tauss", "snow aging term"), ("smcwtd", "soil water content between bottom of the soil and water table"),
            ("deeprech", "recharge to or from the water table when deep"), ("rech", "recharge to or from the water table when shallow"))}
        self.i36["stc"] = c.fl(f"{M} initial soil temperatures")
        self.i36["smc"] = c.fl(f"{M} initial total soil moistures")
        self.i36["sh2o"] = c.fl(f"{M} initial liquid soil moistures")
        if not (self.i36["sneqv"] == 0.0 and self.i36["snowh"] == 0.0):
            raise ValueError("coldstart with initial snow not ported")
        self.build_tiles()

    def build_tiles(self):
        d = nc.Dataset(self.nc_path)
        lc = np.array(d["LANDCOVER"][:]); tx = np.array(d["TEXTURE"][:])
        nr, ncol = lc.shape[1:]
        self.nr, self.nc = nr, ncol
        lat = np.array(d["lat"][:]); lon = np.array(d["lon"][:])
        landmask = np.array(d["LANDMASK"][:]) > 0.5
        lcl = lc.copy()
        for wc in (self.veg["ISWATER"], 21):
            if wc <= lcl.shape[0]:
                lcl[wc - 1] = -1.0
        dom = lcl.argmax(0) + 1
        rows, cols = np.nonzero(landmask)
        self.rows, self.cols = rows, cols
        self.glat1d, self.glon1d = lat[:, 0].astype(f32), lon[0, :].astype(f32)
        self.lat, self.lon = lat[rows, cols].astype(f32), lon[rows, cols].astype(f32)
        self.ntile = len(rows)
        self.vegt = dom[rows, cols].astype(np.int32)
        self.soiltyp = (tx.argmax(0) + 1)[rows, cols].astype(np.int32)
        self.tbot = np.array(d["TBOT"][:])[rows, cols].astype(f32)
        self.green = np.array(d["GREENNESS"][:])[:, rows, cols].astype(f32)
        self.static = {k: np.array(d[k][:])[rows, cols].astype(f32) for k in ("ELEVATION",)}
        self.build_param()

    def build_param(self):
        v, s, g = self.veg, self.soil36, self.gen
        vt = self.vegtbl
        n = self.ntile
        iv, isl = self.vegt - 1, self.soiltyp - 1
        one = lambda x: np.full(n, x, f32)
        urban = self.vegt == v["ISURBAN"]
        P = {"ISWATER": one(v["ISWATER"]), "ISBARREN": one(v["ISBARREN"]), "ISICE": one(v["ISSNOW"]), "EBLFOREST": one(v["EBLFOREST"]),
             "URBAN_FLAG": urban.astype(f32)}
        for k in ("CH2OP DLEAF Z0MVT HVT HVB RC XL CWPVT C3PSN KC25 AKC KO25 AKO AVCMX AQE LTOVRC DILEFC DILEFW RMF25 SLA FRAGR "
                  "TMIN VCMX25 TDLEF BP MP QE25 RMS25 RMR25 ARM FOLNMX WDPOOL WRRAT MRP").split():
            P[k] = v[k][iv].astype(f32)
        for k in ("SAIM", "LAIM", "RHOL", "RHOS", "TAUL", "TAUS"):
            P[k] = v[k][iv].astype(f32)
        P["NROOT"] = vt["nroot"][iv].astype(f32)
        P["RGL"] = vt["rgl"][iv].astype(f32); P["HS"] = vt["hs"][iv].astype(f32)
        P["RSMIN"] = np.where(urban, f32(400.0), vt["rs"][iv]).astype(f32)
        P["TOPT"], P["RSMAX"] = one(vt["topt"]), one(vt["rsmax"])
        P["ALBSAT"] = np.tile(np.array([0.09, 0.18], f32), (n, 1)); P["ALBDRY"] = np.tile(np.array([0.18, 0.36], f32), (n, 1))
        P["ALBICE"] = np.tile(np.array([0.80, 0.55], f32), (n, 1)); P["ALBLAK"] = np.tile(np.array([0.60, 0.40], f32), (n, 1))
        P["OMEGAS"] = np.tile(np.array([0.8, 0.4], f32), (n, 1)); P["EG"] = np.tile(np.array([0.97, 0.98], f32), (n, 1))
        P["BETADS"], P["BETAIS"] = one(0.5), one(0.5)
        P["CO2"], P["O2"] = one(355.0e-6), one(0.209)
        P["TIMEAN"], P["FSATMX"], P["MFSNO"], P["Z0SNO"], P["SSI"], P["SWEMX"] = one(10.5), one(0.38), one(2.5), one(0.002), one(0.03), one(1.0)
        soil4 = lambda a: np.tile(a[isl][:, None], (1, 4)).astype(f32)
        P["BEXP"], P["SMCDRY"], P["SMCWLT"] = soil4(s["bb"]), soil4(s["drysmc"]), soil4(s["wltsmc"])
        P["SMCREF"], P["SMCMAX"], P["PSISAT"] = soil4(s["refsmc"]), soil4(s["maxsmc"]), soil4(s["satpsi"])
        P["DKSAT"], P["DWSAT"], P["QUARTZ"] = soil4(s["satdk"]), soil4(s["satdw"]), soil4(s["qtz"])
        P["F1"] = s["f11"][isl].astype(f32)
        P["REFDK"], P["REFKDT"] = one(g["refdk"]), one(g["refkdt"])
        P["CSOIL"], P["ZBOT"], P["CZIL"] = one(g["csoil"]), one(g["zbot"]), one(g["czil"])
        P["KDT"] = (P["REFKDT"] * P["DKSAT"][:, 0] / P["REFDK"]).astype(f32)
        P["SLOPE"] = one(g["slope_data"][0])
        u = urban[:, None]
        P["SMCMAX"] = np.where(u, f32(0.45), P["SMCMAX"]); P["SMCREF"] = np.where(u, f32(0.42), P["SMCREF"])
        P["SMCWLT"] = np.where(u, f32(0.40), P["SMCWLT"]); P["SMCDRY"] = np.where(u, f32(0.40), P["SMCDRY"])
        P["CSOIL"] = np.where(urban, f32(3.0e6), P["CSOIL"])
        frzfact = (P["SMCMAX"][:, 0] / P["SMCREF"][:, 0]) * f32(0.412 / 0.468)
        P["FRZX"] = np.where(self.soiltyp != 14, f32(g["frzk"]) * frzfact, f32(0.0)).astype(f32)
        self.par = {k: np.ascontiguousarray(a, f32) for k, a in P.items()}
        self.glacier_idx = np.zeros(0, np.int32)
        self.fvgmax = np.max(self.green, axis=0).astype(f32)
        self.zsoil = -np.cumsum(self.sldpth).astype(f32)

    def initial_state(self):
        """NoahMP36_coldstart (no initial snow). Arrays have a leading tile axis."""
        n, c = self.ntile, self.i36
        one = lambda x: np.full(n, x, f32)
        st = {}
        st["SMH2O"] = np.tile(np.array(c["sh2o"], f32), (n, 1)); st["SMC"] = np.tile(np.array(c["smc"], f32), (n, 1))
        st["STC"] = np.concatenate([np.zeros((n, 3), f32), np.tile(np.array(c["stc"], f32), (n, 1))], 1)
        st["ZSNSO"] = np.tile(np.concatenate([np.zeros(3, f32), self.zsoil]), (n, 1))
        st["SNICE"], st["SNLIQ"] = np.zeros((n, 3), f32), np.zeros((n, 3), f32)
        st["ISNOW"] = np.zeros(n, np.int32)
        for k, key in (("ALBOLD", "albold"), ("SNEQVO", "sneqvo"), ("TAH", "tah"), ("EAH", "eah"), ("FWET", "fwet"), ("CANLIQ", "canliq"),
                       ("CANICE", "canice"), ("TV", "tv"), ("TG", "tg"), ("QSNOW", "qsnow"), ("SNDPTH", "snowh"), ("SWE", "sneqv"),
                       ("ZWT", "zwt"), ("WA", "wa"), ("WT", "wt"), ("WSLAKE", "wslake"), ("LFMASS", "lfmass"), ("RTMASS", "rtmass"),
                       ("STMASS", "stmass"), ("WOOD", "wood"), ("STBLCP", "stblcp"), ("FASTCP", "fastcp"), ("PLAI", "lai"),
                       ("PSAI", "sai"), ("CM", "cm"), ("CH", "ch"), ("TAUSS", "tauss"), ("SMCWTD", "smcwtd"),
                       ("DEEPRECH", "deeprech"), ("RECH", "rech")):
            st[k] = one(c[key])
        for k in ("SFCRUNOFF UDRRUNOFF ACSNOM ACSNOW GDD GRAIN").split():
            st[k] = one(0.0)
        st["SMOISEQ"] = np.zeros((n, 4), f32)
        return st
