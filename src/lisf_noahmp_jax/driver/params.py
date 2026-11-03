"""Tile/parameter setup for Noah-3.9 from an LDT parameter file (LIS 'LDT' data sources), mirroring
noah39_setsoils / noah39_setvegparms / noah39_dynsetup / noah39_coldstart for the walkthrough-style configuration:
one tile per grid cell (dominant land cover), STATSGO/FAO soil texture, monthly greenness and albedo maps,
LAI/emissivity/roughness derived from the vegetation table, tbot and max-snow-albedo maps."""
import numpy as np
import netCDF4 as nc
from . import tables, timeutil as tu
from .config import Config

f32 = np.float32


def lutype4(lcscheme):
    return {"UMD": "UMD ", "USGS": "USGS", "IGBPNCEP": "MODI", "MODIS": "MODI", "ECOCLIMAP2": "ECM2"}[lcscheme]


class Setup:
    def __init__(self, cfg_path, base_dir="."):
        self.cfg = c = Config(cfg_path)
        self.base = base_dir
        P = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
        self.nc_path = P(c.s("LIS domain and parameter data file"))
        self.lcscheme = "IGBPNCEP"   # taken from the LDT file's classification (walkthrough: IGBPNCEP)
        self.model = c.s("Land surface model")               # "Noah.3.9" or "Noah.3.6"
        assert self.model in ("Noah.3.9", "Noah.3.6", "Noah.3.3", "Noah.3.2"), f"unsupported land surface model {self.model}"
        self.v33 = self.model in ("Noah.3.3", "Noah.3.2")          # 3.3 = 3.6 physics without UA snow, ice flag only for water tiles, czil/cmcmax/fxexp/sbeta from tables
        self.v36 = self.model in ("Noah.3.6", "Noah.3.3", "Noah.3.2")
        M = self.model
        self.veg = tables.read_veg(P(c.s(f"{M} vegetation parameter table")), lutype4(self.lcscheme))
        self.soil = tables.read_soil(P(c.s(f"{M} soil parameter table")), fxsb=self.v33)
        self.gen = tables.read_gen(P(c.s(f"{M} general parameter table")))
        self.nslay = c.i(f"{M} number of soil layers")
        self.sldpth = np.array(c.fl(f"{M} layer thicknesses"), f32)
        ts = c.s(f"{M} model timestep")
        self.dt = int(ts.rstrip("mnMN")) * 60 if ts.lower().endswith("mn") else (int(ts[:-2]) * 3600 if ts.lower().endswith("hr") else int(ts))
        self.zlvl = c.f(f"{M} reference height for forcing T and q")
        self.zlvl_wind = c.f(f"{M} reference height for forcing u and v")
        self.opt_thcnd = 0 if self.v36 else c.i(f"{M} thermal conductivity option")
        self.sfcdif_option = c.i(f"{M} sfcdif option")
        self.ua_phys = (not self.v33) and c.s(f"{M} Run UA snow-physics option").lower() in (".true.", "true", "t")
        assert self.nslay == 4 and self.sfcdif_option in (0, 1), "port covers 4 layers and SFCDIF options 0/1"
        self.isurban = 13 if self.lcscheme == "IGBPNCEP" else 1
        self.bare, self.snowc, self.water, self.glacier = 16, 15, 17, 15   # IGBP-NCEP classes
        self.build_tiles()

    def build_tiles(self):
        d = nc.Dataset(self.nc_path)
        lc = np.array(d["LANDCOVER"][:]); tx = np.array(d["TEXTURE"][:])
        nr, ncol = lc.shape[1:]
        self.nr, self.nc = nr, ncol
        lat = np.array(d["lat"][:]); lon = np.array(d["lon"][:])
        landmask = np.array(d["LANDMASK"][:]) > 0.5
        # dominant LAND class: water classes (17, and 21 = lakes/inland water in the MODIS-IGBP LDT file) are not LSM tiles
        lcl = lc.copy()
        for wc in (self.water, 21):
            if wc <= lcl.shape[0]:
                lcl[wc - 1] = -1.0
        dom = lcl.argmax(0) + 1
        land = landmask
        rows, cols = np.nonzero(land)          # row-major, column fastest (LIS tile order)
        self.rows, self.cols = rows, cols
        self.glat1d, self.glon1d = lat[:, 0].astype(f32), lon[0, :].astype(f32)
        self.lat, self.lon = lat[rows, cols].astype(f32), lon[rows, cols].astype(f32)
        self.ntile = len(rows)
        T = lambda name: np.array(d[name][:])[rows, cols].astype(f32)
        M = lambda name: np.array(d[name][:])[:, rows, cols].astype(f32)
        self.vegt = dom[rows, cols].astype(np.int32)
        soilt = (tx.argmax(0) + 1)[rows, cols].astype(np.int32)
        soilt = np.where((soilt == 14) & (self.vegt != self.water), 7, soilt)     # noah39_setsoils: water texture on land -> sandy clay loam
        self.soiltyp = soilt.astype(np.int32)
        self.slope, self.shdmin, self.shdmax = T("SLOPE"), T("SHDMIN"), T("SHDMAX")
        self.snoalb, self.tbot = T("MXSNALBEDO"), T("TBOT")
        self.green, self.albmap = M("GREENNESS"), M("ALBEDO")
        v, s, g = self.veg, self.soil, self.gen
        iv, isl = self.vegt - 1, self.soiltyp - 1
        self.glacier_idx = (np.zeros(0, np.int32) if self.v33 else np.nonzero(self.vegt == self.glacier)[0].astype(np.int32))   # Noah 3.3: ice only for water tiles / xice=1 (none here)
        self.par = {
            "vegt": self.vegt, "soiltyp": self.soiltyp,
            "nroot": v["nroot"][iv].astype(np.int32), "slope": self.slope, "shdmin": self.shdmin,
            "shdmax": self.shdmax, "snoalb": self.snoalb, "tbot": self.tbot,
            "hs": v["hs"][iv], "rgl": v["rgl"][iv], "snup": v["snup"][iv], "ztopv": v["ztopv"][iv] if "ztopv" in v else np.zeros(len(iv), f32),
            "zbotv": v["zbotv"][iv] if "zbotv" in v else np.zeros(len(iv), f32), "bexp": s["bb"][isl], "dksat": s["satdk"][isl], "dwsat": s["satdw"][isl],
            "f1": s["f11"][isl], "quartz": s["qtz"][isl], "psisat": s["satpsi"][isl],
            "sbeta": np.full(self.ntile, g.get("sbeta", 0.0), f32), "fxexp": np.full(self.ntile, g.get("fxexp", 0.0), f32),
            "csoil": np.full(self.ntile, g["csoil"], f32), "salp": np.full(self.ntile, g["salp"], f32),
            "czil": np.full(self.ntile, g.get("czil", 0.0), f32), "lvcoef": np.full(self.ntile, g["lvcoef"], f32),
            "zbot": np.full(self.ntile, g["zbot"], f32), "refkdt": np.full(self.ntile, g["refkdt"], f32),
            "topt": np.full(self.ntile, v["topt"], f32), "cmcmax": np.full(self.ntile, v.get("cmcmax", 0.0), f32),
            "cfactr": np.full(self.ntile, v["cfactr"], f32), "rsmax": np.full(self.ntile, v["rsmax"], f32),
        }
        if self.v33:
            if self.model == "Noah.3.3":
                self.par["czil"] = v["ztopv"][iv].astype(f32)        # table columns 16, 17 are CZIL and CMCMAX in the Noah 3.3 vegetation table
                self.par["cmcmax"] = v["zbotv"][iv].astype(f32)
            self.par["sbeta"] = s["sbeta"][isl].astype(f32)
            self.par["fxexp"] = s["fxexp"][isl].astype(f32)
            self.par["ztopv"] = np.zeros(self.ntile, f32); self.par["zbotv"] = np.zeros(self.ntile, f32)
        self.par["kdt"] = (self.par["refkdt"] * self.par["dksat"] / f32(g["refdk"])).astype(f32)
        self.frzk = f32(g["frzk"])
        # persistent soil parameters (SFLX overwrites them for urban tiles; carried in the state)
        self.soilp0 = {"smcmax": s["maxsmc"][isl], "smcref": s["refsmc"][isl], "smcwlt": s["wltsmc"][isl],
                       "smcdry": s["drysmc"][isl], "rsmin": v["rs"][iv]}
        self.table_veg = {k: v[k][iv] for k in ("emissmin", "emissmax", "laimin", "laimax", "albmin", "albmax",
                                                "z0min", "z0max")}

    # ------------------------------------------------------------------
    def dyn(self, y, m, d, newday=True):
        """noah39_dynsetup for the date of the step label (greenness+albedo maps, table-derived LAI/emiss/z0)."""
        vt = self.vegt
        shdfac = tu.month_d(self.green, y, m, d).astype(f32)
        shdfac = np.where((vt == self.bare) | (vt == self.snowc) | (vt == self.water), f32(0), shdfac)
        T = self.table_veg
        hi = shdfac >= self.shdmax
        lo = (~hi) & (shdfac <= self.shdmin)
        den = np.where(self.shdmax > self.shdmin, self.shdmax - self.shdmin, f32(1))
        fr = np.clip((shdfac - self.shdmin) / den, f32(0), f32(1)).astype(f32)
        mid_ok = self.shdmax > self.shdmin
        out = {}
        for key, tmin, tmax in (("embrd", "emissmin", "emissmax"), ("z0brd", "z0min", "z0max"), ("lai", "laimin", "laimax")):
            a, b = T[tmin], T[tmax]
            midv = np.where(mid_ok, (f32(1) - fr) * a + fr * b, f32(0.5) * a + f32(0.5) * b)
            out[key] = np.where(hi, b, np.where(lo, a, midv)).astype(f32)
        out["alb"] = tu.month_d(self.albmap, y, m, d).astype(f32)
        out["shdfac"] = shdfac
        out["newday"] = np.full(self.ntile, 1.0 if newday else 0.0, f32)
        return out

    def initial_state(self, y, m, d):
        c = self.cfg
        n = self.ntile
        one = lambda v: np.full(n, v, f32)
        z0 = tu.month_d(np.array(c.fl(f"{self.model} background roughness length"), f32), y, m, d)
        dyn = self.dyn(y, m, d)
        st = {
            "cmc": one(c.f(f"{self.model} initial canopy water")), "t1": one(c.f(f"{self.model} initial skin temperature")),
            "snowh": one(c.f(f"{self.model} initial snow depth")), "sneqv": one(c.f(f"{self.model} initial snow equivalent")),
            "albedo": dyn["alb"].copy(), "ch": one(1.0e-4), "cm": one(1.0e-4), "z0": one(z0),
            "snotime1": one(0.0), "emiss": one(0.96), "q1": one(-1.0), "xlai": dyn["lai"].copy(), "sca": one(0.0),
            "z0brd_old": one(z0), "z0_old": one(z0),
            "stc": np.stack([one(x) for x in c.fl(f"{self.model} initial soil temperatures")]),
            "smc": np.stack([one(x) for x in c.fl(f"{self.model} initial total soil moistures")]),
            "sh2o": np.stack([one(x) for x in c.fl(f"{self.model} initial liquid soil moistures")]),
        }
        st.update({k: v.astype(f32) for k, v in self.soilp0.items()})
        return st
