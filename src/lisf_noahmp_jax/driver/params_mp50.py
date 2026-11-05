"""Tile/parameter setup for Noah-MP 5.0 from an LDT parameter file, mirroring NoahMP50_readcrd, NoahmpReadTableMod (NoahmpTable.TBL ->
NoahmpIO%*_TABLE) and TRANSFER_MP_PARAMETERS_NEW (NoahMP50_setup.F90: LisNoahmpParam_type per tile). One tile per grid cell (dominant land
cover). Everything the 4.0.1 setup does for the dveg 4 style configuration applies; the 5.0 table merges MPTABLE, SOILPARM and GENPARM."""
import re
import numpy as np
import netCDF4 as nc
from . import mp_tables as T
from .params_mp import SetupMP, NL
from .config import Config

f32 = np.float32
LANDUSE = {"IGBPNCEP": "modis", "USGS": "usgs", "UMD": "modis"}
OPTKEYS = [("dveg", "dynamic vegetation option"), ("crs", "canopy stomatal resistance option"),
           ("btr", "soil moisture factor for stomatal resistance"), ("runsrf", "surface runoff option"),
           ("runsub", "subsurface runoff and groundwater option"), ("infdv", "dynamic VIC infiltration option"),
           ("sfc", "surface layer drag coefficient option"), ("frz", "supercooled liquid water option"),
           ("inf", "frozen soil permeability option"), ("rad", "canopy radiative transfer option"),
           ("alb", "snow surface albedo option"), ("snf", "rain-snow partition option"),
           ("tksno", "snow thermal conductivity option"), ("tbot", "lower boundary of soil temperature option"),
           ("stc", "snow&soil temperature time scheme option"), ("gla", "glacier ice option"),
           ("rsf", "surface resistance option"), ("soil", "soil configuration option"), ("pedo", "soil pedotransfer function option"),
           ("crop", "crop model option"), ("irr", "irrigation trigger option"), ("irrm", "irrigation method option"),
           ("tdrn", "tile drainage option"), ("urban", "urban physics option")]
VEG_SCALARS = ("CH2OP DLEAF Z0MVT HVT HVB DEN RC MFSNO SCFFAC CBIOM SLA DILEFC DILEFW FRAGR LTOVRC C3PSN KC25 AKC KO25 AKO VCMX25 AVCMX BP MP "
               "QE25 AQE RMF25 RMS25 RMR25 ARM FOLNMX TMIN XL MRP CWPVT WRRAT WDPOOL TDLEF NROOT RGL HS TOPT RSMAX RTOVRC RSWOODC BF WSTRC "
               "LAIMIN XSAMIN").split()
GLOBALS = ("CO2 O2 TIMEAN FSATMX ROUS CMIC Z0SNO SSI SNOW_RET_FAC SNOW_EMIS SWEMX TAU0 GRAIN_GROWTH EXTRA_GROWTH DIRT_SOOT BATS_COSZ BATS_VIS_NEW "
           "BATS_NIR_NEW BATS_VIS_AGE BATS_NIR_AGE BATS_VIS_DIR BATS_NIR_DIR RSURF_SNOW RSURF_EXP C2_SNOWCOMPACT C3_SNOWCOMPACT C4_SNOWCOMPACT "
           "C5_SNOWCOMPACT DM_SNOWCOMPACT ETA0_SNOWCOMPACT SNLIQMAXFRAC SWEMAXGLA WSLMAX SNOWDEN_MAX CLASS_ALB_REF CLASS_SNO_AGE CLASS_ALB_NEW "
           "PSIWLT Z0SOIL Z0LAKE").split()


CROP_SCALARS = {"BP": "BPI", "KC25": "KC25I", "KO25": "KO25I", "AKC": "AKCI", "AKO": "AKOI", "FOLNMX": "FOLNMXI", "QE25": "QE25I",
                "VCMX25": "VCMX25I", "AVCMX": "AVCMXI", "C3PSN": "C3PSNI", "MP": "MPI", "ARM": "Q10MR", "RMF25": "LFMR25", "RMS25": "STMR25",
                "RMR25": "RTMR25", "FRAGR": "FRA_GR", "TDLEF": "LEFREEZ", "BIO2LAI": "BIO2LAI", "GRAINMR25": "GRAINMR25", "GDDTBASE": "GDDTBASE",
                "GDDTCUT": "GDDTCUT"}
CROP_STAGES = {"LF_OVRC": "LF_OVRC", "ST_OVRC": "ST_OVRC", "RT_OVRC": "RT_OVRC", "DILE_FC": "DILE_FC", "DILE_FW": "DILE_FW", "LFPT": "LFPT",
               "STPT": "STPT", "RTPT": "RTPT", "GRAINPT": "GRAINPT", "LFCT": "LFCT", "STCT": "STCT", "RTCT": "RTCT"}
CROP_GDD = ("GDDS1", "GDDS2", "GDDS3", "GDDS4", "GDDS5")


def read_crop_table(g):
    """noahmp_crop_parameters -> dict: DEFAULT_CROP, PLTDAY, HSDAY, GDDS1-5, scalars (5,) and the eight-stage arrays (5, 8)."""
    c = g["noahmp_crop_parameters"]
    d = {"DEFAULT_CROP": int(c["DEFAULT_CROP"][0])}
    for k in ("PLTDAY", "HSDAY") + CROP_GDD + tuple(CROP_SCALARS.values()):
        d[k] = np.array(c[k][:5], f32)
    for k in CROP_STAGES.values():
        d[k] = np.stack([np.array(c[f"{k}_S{i}"][:5], f32) for i in range(1, 9)], 1)
    return d


def crop_overrides(crop, croptype, planting, harvest, season_gdd):
    """Per-tile parameters of the dynamic crop model (BiochemVarInTransfer, EnergyVarInTransfer) as a dict of arrays; entries are valid where
    croptype > 0 (CropType 1-5, others are filled with harmless defaults). The stage thresholds are scaled by SEASON_GDD / 1770 (crop option 1)."""
    n = len(croptype)
    ic = np.maximum(croptype, 1).astype(int) - 1
    out = {}
    for new, old in CROP_SCALARS.items():
        out[new] = crop[old][ic].astype(f32)
    for k in CROP_STAGES:
        out[k] = crop[CROP_STAGES[k]][ic].astype(f32)
    out["PLTDAY"] = np.trunc(np.asarray(planting, f32))          # DatePlanting and DateHarvest are integers in the Fortran (assignment truncates)
    out["HSDAY"] = np.trunc(np.asarray(harvest, f32))
    sc = (np.asarray(season_gdd, f32) / f32(1770.0)).astype(f32)
    for k in CROP_GDD:
        out[k] = (sc * crop[k][ic]).astype(f32)
    out["CROPTYPE"] = croptype.astype(f32)
    return out


def apply_crop(P, ov, n):
    """Overwrite the table parameters of the crop tiles in P (dict of per-tile arrays) with the crop-table values."""
    m = ov["CROPTYPE"] > 0.5
    for k, v in ov.items():
        if k in P and k != "CROPTYPE":
            mm = m.reshape((-1,) + (1,) * (v.ndim - 1))
            P[k] = np.where(mm, v, P[k]).astype(f32)
        else:
            P[k] = v
    return P


def read_table50(path, scheme):
    """NoahmpTable.TBL -> dict of numpy arrays: veg (per class), soil (per category), radiation, global scalars, general."""
    g = T.read_namelists(path)
    key = LANDUSE[scheme]
    cat, par = g[f"noahmp_{key}_veg_categories"], g[f"noahmp_{key}_parameters"]
    nveg = cat["NVEG"][0]
    d = {"NVEG": nveg}
    for k in "ISURBAN ISWATER ISBARREN ISICE ISCROP EBLFOREST NATURAL".split():
        d[k] = par[k][0]
    arr = lambda k: np.array(par[k][:nveg], f32)
    for k in VEG_SCALARS:
        d[k] = arr(k)
    d["RSMIN"] = arr("RS")
    months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    d["SAIM"] = np.stack([arr(f"SAI_{m}") for m in months], 1)
    d["LAIM"] = np.stack([arr(f"LAI_{m}") for m in months], 1)
    for a in ("RHOL", "RHOS", "TAUL", "TAUS"):
        d[a] = np.stack([arr(f"{a}_VIS"), arr(f"{a}_NIR")], 1)
    r = g["noahmp_rad_parameters"]
    d["ALBSAT"] = np.stack([np.array(r["ALBSAT_VIS"], f32), np.array(r["ALBSAT_NIR"], f32)], 1)
    d["ALBDRY"] = np.stack([np.array(r["ALBDRY_VIS"], f32), np.array(r["ALBDRY_NIR"], f32)], 1)
    for k in ("ALBICE", "ALBLAK", "OMEGAS", "EG"):
        d[k] = np.array(r[k], f32)
    for k in ("BETADS", "BETAIS", "EICE"):
        d[k] = f32(r[k][0])
    gl = g["noahmp_global_parameters"]
    for k in GLOBALS:
        d[k] = f32(gl[k][0])
    s = g["noahmp_soil_stas_parameters"]
    nsl = g["noahmp_stas_soil_categories"]["SLCATS"][0]
    for new, old in (("BEXP", "BB"), ("SMCDRY", "DRYSMC"), ("SMCMAX", "MAXSMC"), ("SMCREF", "REFSMC"), ("PSISAT", "SATPSI"),
                     ("DKSAT", "SATDK"), ("DWSAT", "SATDW"), ("SMCWLT", "WLTSMC"), ("QUARTZ", "QTZ"), ("BVIC", "BVIC"), ("AXAJ", "AXAJ"),
                     ("BXAJ", "BXAJ"), ("XXAJ", "XXAJ"), ("BDVIC", "BDVIC"), ("GDVIC", "GDVIC"), ("BBVIC", "BBVIC")):
        d[new] = np.array(s[old][:nsl], f32)
    td = g["noahmp_tiledrain_parameters"]
    d["DRAIN_LAYER_OPT"] = f32(td["DRAIN_LAYER_OPT"][0])
    for k in ("TDSMC_FAC", "TD_DEPTH", "TD_DC", "TD_DCOEF", "TD_D", "TD_ADEPTH", "TD_RADI", "TD_SPAC", "TD_DDRAIN"):
        d[k] = np.array(td[k][:nsl], f32)
    d["KLAT_FAC"] = np.array(td["KLAT_FAC"][:nsl], f32) if "KLAT_FAC" in td else np.ones(nsl, f32)
    d["CROP"] = read_crop_table(g)
    ge = g["noahmp_general_parameters"]
    d["SLOPE"] = np.array(ge["SLOPE_DATA"], f32)
    for k in ("CSOIL", "REFDK", "REFKDT", "FRZK", "ZBOT", "CZIL"):
        d[k] = f32(ge[f"{k}_DATA"][0])
    return d


class SetupMP50(SetupMP):
    def __init__(self, cfg_path, base_dir="."):
        self.cfg = c = Config(cfg_path)
        self.base = base_dir
        P = lambda p: p if p.startswith("/") else f"{base_dir}/{p}"
        self.nc_path = P(c.s("LIS domain and parameter data file"))
        self.model = c.s("Land surface model")
        if not (self.model == "Noah-MP.5.0"):
            raise ValueError(f"unsupported land surface model {self.model}")
        M = "Noah-MP.5.0"
        self.o = {k: c.i(f"{M} {t}") for k, t in OPTKEYS}
        o = self.o
        if not (o["runsrf"] == o["runsub"] or (o["runsrf"] in (6, 7, 8) and o["runsub"] in (3, 6, 7, 8))):
            raise ValueError("this combination of surface and subsurface runoff options is not ported")
        o["run"] = o["runsrf"]
        supported = dict(dveg=(1, 2, 3, 4, 5, 6, 7, 8, 9), crs=(1, 2), btr=(1, 2, 3), run=(1, 2, 3, 4, 6, 7, 8), sfc=(1, 2), frz=(1, 2), inf=(1, 2), rad=(1, 2, 3),
                         alb=(1, 2), snf=(1, 2, 3, 5), tksno=(1, 2, 3, 4, 5), tbot=(1, 2), stc=(1, 2, 3), rsf=(1, 2, 3, 4), soil=(1,),
                         crop=(0, 1), irr=(0,), tdrn=(0, 1, 2), urban=(0,))
        for k, ok in supported.items():
            if not (o[k] in ok):
                raise ValueError(f"Noah-MP 5.0 option {k}={o[k]} not ported (supported: {ok})")
        self.nslay = c.i(f"{M} number of soil layers")
        assert self.nslay == 4
        self.sldpth = np.array(c.fl(f"{M} thickness of soil layers"), f32)
        ts = c.s(f"{M} model timestep")
        self.dt = int(ts.rstrip("mnMN")) * 60 if ts.lower().endswith("mn") else (int(ts[:-2]) * 3600 if ts.lower().endswith("hr") else int(ts))
        tss = c.s(f"{M} soil timestep")
        if not ((int(tss.rstrip("mnMN")) * 60 if tss.lower().endswith("mn") else int(tss)) == self.dt):
            raise ValueError("soil timestep different from the model timestep is not ported")
        self.zref = c.f(f"{M} reference height of temperature and humidity")
        self.dz8w = f32(self.zref) * f32(2.0)
        d = nc.Dataset(self.nc_path)
        scheme = getattr(d, "LANDCOVER_SCHEME", "IGBPNCEP")
        d.close()
        self.lcscheme = scheme
        self.tab = T50 = read_table50(P(c.s(f"{M} parameter table")), scheme)
        # names used by the shared 4.0.1 code paths
        self.veg = {"ISURBAN": T50["ISURBAN"], "ISWATER": T50["ISWATER"], "ISBARREN": T50["ISBARREN"], "ISICE": T50["ISICE"],
                    "ISCROP": T50["ISCROP"], "EBLFOREST": T50["EBLFOREST"], "SLA": T50["SLA"],
                    "LOW_DENSITY_RESIDENTIAL": -1, "HIGH_DENSITY_RESIDENTIAL": -1, "HIGH_INTENSITY_INDUSTRIAL": -1}
        self.soil = {k: T50[k] for k in ("BEXP", "SMCDRY", "SMCMAX", "SMCREF", "PSISAT", "DKSAT", "DWSAT", "SMCWLT", "QUARTZ")}
        self.init = dict(tskin=c.f(f"{M} initial surface skin temperature"),
                         tslb=c.fl(f"{M} initial soil temperatures"), smc=c.fl(f"{M} initial total soil moistures"),
                         sneqv=c.f(f"{M} initial snow water equivalent"), snowh=c.f(f"{M} initial snow depth"),
                         canwat=c.f(f"{M} initial total canopy surface water"), lai=c.f(f"{M} initial leaf area index"),
                         zwt=c.f(f"{M} initial water table depth"), wa=c.f(f"{M} initial water in the aquifer"),
                         wt=c.f(f"{M} initial water in aquifer and saturated soil"))
        if not (self.init["sneqv"] == 0.0 and self.init["snowh"] == 0.0):
            raise ValueError("coldstart with initial snow not ported")
        self.dx = f32(c.f(f"{M} domain resolution dx"))
        self.crop_maps = None
        self.build_tiles()
        self.td_fraction = None
        if self.o.get("crop", 0) > 0:                                  # LDT variables PLANTING, HARVEST, SEASON_GDD (NoahMP50_setup)
            dd = nc.Dataset(self.nc_path)
            if not (getattr(dd, "CROPCLASS", 0) == 0):
                raise ValueError("LDT global attribute CROPCLASS > 0 (whole domain forced to one crop) is not ported")
            if (self.croptype > 0).any():
                self.crop_maps = tuple(np.array(dd[k][:])[self.rows, self.cols].astype(f32) for k in ("PLANTING", "HARVEST", "SEASON_GDD"))
                self.build_param()
            dd.close()
        if self.o.get("tdrn", 0) > 0:                                  # LDT variable TD_FRACTION (NoahMP50_setup)
            dd = nc.Dataset(self.nc_path)
            self.td_fraction = np.array(dd["TD_FRACTION"][:])[self.rows, self.cols].astype(f32)
            dd.close()

    def build_param(self):
        if self.o["dveg"] in (7, 8, 9):                 # input LAI: monthly LAI climatology of the LDT file ('LAI data source: LDT')
            d_ = nc.Dataset(self.nc_path)
            self.lai_mon = np.array(d_["LAI"][:])[:, self.rows, self.cols].astype(f32)        # (12, ntile)
            d_.close()
        t = self.tab
        n = self.ntile
        iv, isl = self.vegt - 1, self.soiltyp - 1
        one = lambda x: np.full(n, x, f32)
        urb = self.vegt == t["ISURBAN"]
        self.croptype = np.where((self.o.get("crop", 0) > 0) & (self.vegt == t["ISCROP"]), t["CROP"]["DEFAULT_CROP"], 0).astype(np.int32)
        P = {"ISWATER": one(t["ISWATER"]), "ISBARREN": one(t["ISBARREN"]), "ISICE": one(t["ISICE"]), "ISCROP": one(t["ISCROP"]),
             "EBLFOREST": one(t["EBLFOREST"]), "URBAN_FLAG": urb.astype(f32)}
        for k in VEG_SCALARS + ["RSMIN"]:
            P[k] = t[k][iv].astype(f32)
        for k in ("SAIM", "LAIM", "RHOL", "RHOS", "TAUL", "TAUS"):
            P[k] = t[k][iv].astype(f32)
        soilcolor = 4
        P["ALBSAT"] = np.tile(t["ALBSAT"][soilcolor - 1], (n, 1)); P["ALBDRY"] = np.tile(t["ALBDRY"][soilcolor - 1], (n, 1))
        for k in ("ALBICE", "ALBLAK", "OMEGAS", "EG"):
            P[k] = np.tile(t[k], (n, 1))
        for k in ("BETADS", "BETAIS", "EICE"):
            P[k] = one(t[k])
        for k in GLOBALS:
            P[k] = one(t[k])
        for k in ("BEXP", "SMCDRY", "SMCWLT", "SMCREF", "SMCMAX", "PSISAT", "DKSAT", "DWSAT", "QUARTZ"):
            P[k] = np.tile(t[k][isl][:, None], (1, 4)).astype(f32)
        for k in ("TDSMC_FAC", "TD_DEPTH", "TD_DC", "TD_DCOEF", "TD_D", "TD_ADEPTH", "TD_RADI", "TD_SPAC", "TD_DDRAIN", "KLAT_FAC"):
            P[k] = t[k][isl].astype(f32)
        P["DRAIN_LAYER_OPT"] = one(t["DRAIN_LAYER_OPT"])
        for k in ("BVIC", "AXAJ", "BXAJ", "XXAJ", "BDVIC", "GDVIC", "BBVIC"):
            P[k] = t[k][isl].astype(f32)
        for k in ("REFDK", "REFKDT", "FRZK", "CSOIL", "ZBOT", "CZIL"):
            P[k] = one(t[k])
        P["KDT"] = (P["REFKDT"] * P["DKSAT"][:, 0] / P["REFDK"]).astype(f32)
        P["SLOPE"] = one(t["SLOPE"][0])
        u = urb[:, None]
        P["SMCMAX"] = np.where(u, f32(0.45), P["SMCMAX"]); P["SMCREF"] = np.where(u, f32(0.42), P["SMCREF"])
        P["SMCWLT"] = np.where(u, f32(0.40), P["SMCWLT"]); P["SMCDRY"] = np.where(u, f32(0.40), P["SMCDRY"])
        P["CSOIL"] = np.where(urb, f32(3.0e6), P["CSOIL"])
        with np.errstate(divide="ignore", invalid="ignore"):   # water tiles have SMCREF = 0; FRZX is set to 0 for them below
            frzfact = (P["SMCMAX"][:, 0] / P["SMCREF"][:, 0]) * f32(0.412 / 0.468)
        P["FRZX"] = np.where(self.soiltyp != 14, P["FRZK"] * frzfact, f32(0.0)).astype(f32)
        P["MXSNALB"], P["MNSNALB"], P["sndecayexp"] = one(0.84), one(0.55), one(0.01)
        P["T_ulimit"], P["T_mlimit"], P["T_llimit"], P["snowf_scalef"] = one(2.5), one(2.0), one(0.5), one(1.0)
        if self.o.get("crop", 0) > 0:
            pl, hv, sg = self.crop_maps if self.crop_maps is not None else (np.zeros(n, f32), np.full(n, 366.0, f32), np.full(n, 1770.0, f32))
            apply_crop(P, crop_overrides(t["CROP"], self.croptype, pl, hv, sg), n)
        self.par = {k: np.ascontiguousarray(a, f32) for k, a in P.items()}
        self.glacier_idx = np.nonzero(self.vegt == t["ISICE"])[0].astype(np.int32)
        self.fvgmax = (np.max(self.green * f32(100.0), axis=0) / f32(100.0)).astype(f32)
        self.zsoil = -np.cumsum(self.sldpth).astype(f32)

    def initial_state(self):
        st = super().initial_state()
        bare = np.isin(self.vegt, [self.tab["ISBARREN"], self.tab["ISICE"], self.tab["ISURBAN"], self.tab["ISWATER"]])
        st["STMASS"] = np.where(bare, f32(0), (st["PSAI"] * f32(1000.0)) / f32(3.0)).astype(f32)      # NoahMP50 coldstart: XSAI*1000/3 (left to right)
        if self.o.get("crop", 0) > 0:                                                                  # NoahMP50_coldstart: grain, GDD and growth stage start at 0
            st["GRAIN"] = np.zeros(self.ntile, f32); st["GDD"] = np.zeros(self.ntile, f32); st["PGS"] = np.zeros(self.ntile, f32)
        return st
