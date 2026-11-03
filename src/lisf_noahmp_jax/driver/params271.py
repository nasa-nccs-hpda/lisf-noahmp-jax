"""Tile/parameter setup for Noah 2.7.1 from an LDT parameter file, mirroring noah271_setsoils / noah271_setvegparms / noah271_main
(KDT/FRZX/DWSAT/F1/SMCREF/SMCWLT derived in the main routine, urban overrides) for the walkthrough-style configuration: one tile per grid
cell (dominant land cover, IGBP-NCEP classes), STATSGO texture, monthly greenness and albedo maps, TBOT and max-snow-albedo maps.
The vegetation/soil tables are the derived files written by Reports/noah271_20261022/make_noah271_tables.py (see its README)."""
import numpy as np
import netCDF4 as nc
from . import timeutil as tu
from .config import Config

f32 = np.float32
NVEG, NSOILCLS, NSOILP, NVEGP = 20, 19, 10, 7


def read_veg271(path):
    L = open(path).read().split("\n")
    v = {}
    for j in range(NVEGP):
        v[L[2 * j].strip().lower()] = np.array([float(x) for x in L[2 * j + 1].split()], f32)
    return v


def read_soil271(path):
    L = open(path).read().split("\n")
    names = ["maxsmc", "satpsi", "satdk", "bb", "qtz", "refsmc", "wltsmc", "drysmc", "satdw", "f11"]
    out = {}
    for j in range(NSOILP):
        assert L[2 + 2 * j].strip().lower() == names[j].replace("maxsmc", "maxsmc"), L[2 + 2 * j]
        out[names[j]] = np.array([float(x) for x in L[3 + 2 * j].split()], f32)
    out["czil"] = f32(float(L[3 + 2 * NSOILP].split()[0]))
    return out


class Setup271:
    def __init__(self, cfg_path, base_dir="."):
        self.cfg = c = Config(cfg_path)
        self.base = base_dir
        P = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
        M = self.model = c.s("Land surface model")
        assert M == "Noah.2.7.1", M
        self.v271 = True
        self.nc_path = P(c.s("LIS domain and parameter data file"))
        self.lcscheme = "IGBPNCEP"
        self.veg = read_veg271(P(c.s(f"{M} vegetation parameter table")))
        self.soil = read_soil271(P(c.s(f"{M} soil parameter table")))
        assert c.i(f"{M} number of soil layers") == 4 and c.i(f"{M} use PTF for mapping soil properties") == 0
        self.sldpth = np.array(c.fl(f"{M} layer thicknesses"), f32)
        ts = c.s(f"{M} model timestep")
        self.dt = int(ts.rstrip("mnMN")) * 60 if ts.lower().endswith("mn") else (int(ts[:-2]) * 3600 if ts.lower().endswith("hr") else int(ts))
        self.zlvl = c.f(f"{M} reference height for forcing T and q")
        self.zlvl_wind = c.f(f"{M} reference height for forcing u and v")
        self.isurban = 13
        self.bare, self.snowc, self.water = 16, 15, 17
        self.build_tiles()

    def build_tiles(self):
        d = nc.Dataset(self.nc_path)
        lc = np.array(d["LANDCOVER"][:]); tx = np.array(d["TEXTURE"][:])
        self.nr, self.nc = lc.shape[1:]
        lat = np.array(d["lat"][:]); lon = np.array(d["lon"][:])
        landmask = np.array(d["LANDMASK"][:]) > 0.5
        lcl = lc.copy()
        for wc in (self.water, 21):
            if wc <= lcl.shape[0]:
                lcl[wc - 1] = -1.0
        dom = lcl.argmax(0) + 1
        rows, cols = np.nonzero(landmask)
        self.rows, self.cols = rows, cols
        self.glat1d, self.glon1d = lat[:, 0].astype(f32), lon[0, :].astype(f32)
        self.lat, self.lon = lat[rows, cols].astype(f32), lon[rows, cols].astype(f32)
        self.ntile = n = len(rows)
        T = lambda name: np.array(d[name][:])[rows, cols].astype(f32)
        Mm = lambda name: np.array(d[name][:])[:, rows, cols].astype(f32)
        self.vegt = dom[rows, cols].astype(np.int32)
        soilt = (tx.argmax(0) + 1)[rows, cols].astype(np.int32)
        soilt = np.where((soilt == 14) | (soilt == 0), 1, soilt)           # noah271_setsoils: water/undefined texture on a land tile -> class 1 (STATSGO)
        self.soiltyp = soilt.astype(np.int32)
        self.snoalb, self.tbot = T("MXSNALBEDO"), T("TBOT")
        self.green, self.albmap = Mm("GREENNESS"), Mm("ALBEDO")
        v, s = self.veg, self.soil
        iv, isl = self.vegt - 1, self.soiltyp - 1
        smcmax, psisat, dksat = s["maxsmc"][isl], s["satpsi"][isl], s["satdk"][isl]
        bexp, quartz = s["bb"][isl], s["qtz"][isl]
        # noah271_main: derived soil parameters (float32 arithmetic)
        smlow, smhigh = f32(0.5), f32(6.0)
        dwsat = (bexp * dksat * (psisat / smcmax)).astype(f32)
        f1 = (np.log10(psisat) + bexp * np.log10(smcmax) + f32(2.0)).astype(f32)
        smcref1 = (smcmax * np.power(f32(5.79e-9) / dksat, f32(1.0) / (f32(2.0) * bexp + f32(3.0)))).astype(f32)
        smcref = (smcref1 + (smcmax - smcref1) / smhigh).astype(f32)
        smcwlt1 = (smcmax * np.power(f32(200.0) / psisat, f32(-1.0) / bexp)).astype(f32)
        smcwlt = (smcwlt1 - smlow * smcwlt1).astype(f32)
        smcdry = smcwlt.copy()
        kdt = (f32(3.0) * dksat / f32(2.0e-6)).astype(f32)
        frzx = (f32(0.15) * ((smcmax / smcref) * (f32(0.412) / f32(0.468)))).astype(f32)
        rsmin = v["rsmin"][iv].copy()
        urb = self.vegt == self.isurban          # noah271_main: urban tiles
        smcmax = np.where(urb, f32(0.45), smcmax); smcref = np.where(urb, f32(0.42), smcref)
        smcwlt = np.where(urb, f32(0.40), smcwlt); rsmin = np.where(urb, f32(400.0), rsmin)
        # noah271_f2t reads Wind_E/Wind_N (and tests precipitation validity) at index t instead of tile_id: the grid cell with
        # row-major index t (1-based) supplies tile t's wind (Fortran bug reproduced on purpose; see README of Reports/noah271_20261022)
        idx = np.arange(n)
        self.f2t_alt = (self.glat1d[idx // self.nc], self.glon1d[idx % self.nc])
        one = lambda x: np.full(n, x, f32)
        self.par = {
            "vegt": self.vegt, "soiltyp": self.soiltyp, "nroot": v["nroot"][iv].astype(np.int32), "rsmin": rsmin.astype(f32),
            "rgl": v["rgl"][iv], "hs": v["hs"][iv], "snup": v["snup"][iv], "z0": v["z0"][iv], "xlai": v["lai"][iv],
            "smcmax": smcmax.astype(f32), "psisat": psisat, "dksat": dksat, "bexp": bexp, "quartz": quartz, "dwsat": dwsat,
            "smcwlt": smcwlt.astype(f32), "smcref": smcref.astype(f32), "smcdry": smcdry, "f1": f1, "kdt": kdt, "frzx": frzx,
            "tbot": self.tbot, "snoalb": self.snoalb, "czil": one(s["czil"]), "sbeta": one(-2.0), "fxexp": one(2.0), "cfactr": one(0.5),
            "cmcmax": one(0.5e-3), "rsmax": one(5000.0), "topt": one(298.0), "csoil": one(2.0e6), "slope": one(1.0),
            "zbot": one(-8.0), "emissi": one(1.0), "emiss_out0": one(1.0),
        }

    def dyn(self, y, m, d, newday=True):
        vt = self.vegt
        shdfac = tu.month_d(self.green, y, m, d).astype(f32)
        shdfac = np.where((vt == self.bare) | (vt == self.snowc) | (vt == self.water), f32(0), shdfac)
        shdfac = np.where(vt == self.isurban, f32(0.05), shdfac)
        return {"shdfac": shdfac.astype(f32), "alb": tu.month_d(self.albmap, y, m, d).astype(f32)}

    def initial_state(self, y, m, d):
        c, M = self.cfg, self.model
        n = self.ntile
        one = lambda x: np.full(n, x, f32)
        return {
            "cmc": one(c.f(f"{M} initial canopy water")), "t1": one(c.f(f"{M} initial skin temperature")),
            "snowh": one(c.f(f"{M} initial snow depth")), "sneqv": one(c.f(f"{M} initial snow equivalent")),
            "ch": one(1.0e-4), "cm": one(1.0e-4),
            "stc": np.stack([one(x) for x in c.fl(f"{M} initial soil temperatures")]),
            "smc": np.stack([one(x) for x in c.fl(f"{M} initial total soil moistures")]),
            "sh2o": np.stack([one(x) for x in c.fl(f"{M} initial liquid soil moistures")]),
        }
