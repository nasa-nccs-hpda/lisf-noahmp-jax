"""Adapter: Noah-MP 5.0 dump (lis_patches/make_noahmp50_hook.py, names of NoahmpIO / LisNoahmpParam_type) -> the 4.0.1 dump layout of lis/dumpio_mp.py
(names of the 4.0.1 driver arguments), so that lis/replay_mp.py can replay 5.0 calls through the JAX port and show where the physics differ.
Mappings follow ForcingVarInTransferMod, ConfigVarInTransferMod, EnergyVarInTransferMod, WaterVarInTransferMod and BiochemVarInTransferMod of the 5.0 LIS driver."""
import os
import numpy as np
import dumpio_mp
from lisf_noahmp_jax.driver import params_mp50

from lisf_noahmp_jax.driver import mp_tables as T_
f32 = np.float32
TABLE = os.path.join(os.environ["LISF_BUILD"], "lis/surfacemodels/land/noahmp.5.0/phys/noahmp/parameters/NoahmpTable.TBL")
_load_raw = dumpio_mp.load            # the real loader (replay_mp50.py replaces dumpio_mp.load by load50)


def load50(dirpath, max_records=None, start=0, idx=None):
    pre, post = _load_raw(dirpath, max_records, start, idx)
    pre, post = to401(pre, post, table_path(dirpath))
    if not os.environ.get("DUMP50_KEEP_ICE"):                          # land-ice points go through NoahmpMainGlacier in 5.0 (separate JAX module)
        keep = pre["VEGTYP"] != pre["p_ISICE"]
        pre = {k: v[keep] for k, v in pre.items()}; post = {k: v[keep] for k, v in post.items()}
    return pre, post


def table_path(dirpath):
    """parameter table of the run (label 'Noah-MP.5.0 parameter table:' of the LIS config in the dump directory)"""
    local = os.path.join(dirpath, "NoahmpTable.TBL")                    # test data: a table that holds the crop group only
    if os.path.exists(local):
        return local
    cfg = os.path.join(dirpath, "lis.config")
    if os.path.exists(cfg):
        for ln in open(cfg):
            if ln.startswith("Noah-MP.5.0 parameter table:"):
                return ln.split(":", 1)[1].split("#")[0].strip()
    return TABLE


def to401(a, b, table=TABLE):
    n = len(a["itimestep"])
    g = lambda k: a[k]
    z = lambda: np.zeros(n, f32)
    dt = g("dtbl")
    pre = {}
    # ---- scalars / forcing
    pre["ITIMESTEP"] = g("itimestep"); pre["LAT"] = g("xlat"); pre["YEARLEN"] = g("YEARLEN"); pre["JULIAN"] = g("JULIAN")
    pre["COSZ"] = g("COSZEN"); pre["DT"] = dt; pre["DX"] = g("DX"); pre["DZ8W1D"] = g("DZ8W")[:, 0]
    pre["FVEG"] = g("VEGFRA") / f32(100.0); pre["FVGMAX"] = g("GVFMAX") / f32(100.0)
    pre["VEGTYP"] = g("ivgtyp"); pre["ICE"] = g("ICE"); pre["IST"] = np.ones(n, f32); pre["CROPTYPE"] = z(); pre["NSOIL"] = g("nsoil")
    pre["T_ML"] = g("T_PHY")[:, 0]; pre["P_ML"] = (g("P8W")[:, 0] + g("P8W")[:, 1]) * f32(0.5); pre["PSFC"] = g("P8W")[:, 0]
    pre["U_ML"] = g("U_PHY")[:, 0]; pre["V_ML"] = g("V_PHY")[:, 0]
    qv = g("QV_CURR")[:, 0]; pre["Q_ML"] = qv / (f32(1.0) + qv); pre["QC"] = z()
    pre["SWDN"] = g("SWDOWN"); pre["LWDN"] = g("GLW")
    tot = g("RAINBL") / dt
    conv, nonc, shcv = g("MP_RAINC") / dt, g("MP_RAINNC") / dt, g("MP_SHCV") / dt
    other = np.maximum(f32(0.0), tot - conv - nonc - shcv)
    pre["PRCPCONV"] = conv; pre["PRCPNONC"] = nonc + other; pre["PRCPSHCV"] = shcv
    pre["PRCPSNOW"] = g("MP_SNOW") / dt + other * g("SR"); pre["PRCPGRPL"] = g("MP_GRAUP") / dt; pre["PRCPHAIL"] = g("MP_HAIL") / dt
    pre["TBOT"] = g("tmn")
    pre["CO2PP"] = g("p_CO2") * pre["P_ML"]; pre["O2PP"] = g("p_O2") * pre["P_ML"]; pre["FOLN"] = np.ones(n, f32)
    pre["Z_ML"] = g("DZ8W")[:, 0] * f32(0.5)
    # ---- states
    for new, old in (("ALBOLDXY", "ALBOLD"), ("SNEQVOXY", "SNEQVO"), ("TAHXY", "TAH"), ("EAHXY", "EAH"), ("FWETXY", "FWET"), ("CANLIQXY", "CANLIQ"),
                     ("CANICEXY", "CANICE"), ("TVXY", "TV"), ("TGXY", "TG"), ("QSFC", "QSFC1D"), ("QSNOWXY", "QSNOW"), ("ISNOWXY", "ISNOW"),
                     ("SNOWH", "SNDPTH"), ("SNOW", "SWE"), ("ZWTXY", "ZWT"), ("WAXY", "WA"), ("WTXY", "WT"), ("WSLAKEXY", "WSLAKE"),
                     ("LFMASSXY", "LFMASS"), ("RTMASSXY", "RTMASS"), ("STMASSXY", "STMASS"), ("WOODXY", "WOOD"), ("STBLCPXY", "STBLCP"),
                     ("FASTCPXY", "FASTCP"), ("LAI", "PLAI"), ("XSAIXY", "PSAI"), ("CMXY", "CM"), ("CHXY", "CH"), ("TAUSSXY", "TAUSS"),
                     ("SMCWTDXY", "SMCWTD"), ("rivsto", "rivsto"), ("fldsto", "fldsto"), ("fldfrc", "fldfrc")):
        pre[old] = g(new)
    # ---- options
    pre["IDVEG"] = g("IOPT_DVEG")
    if "GRAINXY" in a:                                                 # dynamic crop model states
        pre["GRAIN"] = g("GRAINXY"); pre["GDD"] = g("GDDXY"); pre["PGS"] = g("PGSXY")
    if "IRFRACT" in a:
        for k in ("IRFRACT", "SIFRACT", "MIFRACT", "FIFRACT", "IRWATSI", "IRWATMI", "IRWATFI", "IRNUMSI", "IRNUMMI", "IRNUMFI", "IRRSPLH"):
            pre[k] = g(k)
    if "TD_FRACTION" in a:
        pre["TD_FRACTION"] = g("TD_FRACTION"); pre["DX"] = g("DX")
    for k in ("IOPT_CRS", "IOPT_BTR", "IOPT_SFC", "IOPT_FRZ", "IOPT_INF", "IOPT_RAD", "IOPT_ALB", "IOPT_SNF", "IOPT_TBOT", "IOPT_STC", "IOPT_RSF", "IOPT_TKSNO",
              "IOPT_SOIL", "IOPT_PEDO", "IOPT_CROP", "IOPT_INFDV", "IOPT_TDRN", "IOPT_IRR", "IOPT_IRRM"):
        pre[k] = g(k)
    pre["IOPT_RUN"] = g("IOPT_RUNSRF")
    # ---- arrays
    pre["ZSOIL"] = g("ZSOIL"); pre["SMCEQ"] = g("SMOISEQ"); pre["SMH2O"] = g("SH2O"); pre["SMC"] = g("SMOIS")
    pre["SNICE"] = g("SNICEXY"); pre["SNLIQ"] = g("SNLIQXY"); pre["ZSNSO"] = g("ZSNSOXY")
    pre["STC"] = np.concatenate([g("TSNOXY"), g("TSLB")], axis=1)
    isn = g("ISNOWXY").astype(int)
    ice = (g("SNICEXY") / np.maximum(g("SNICEXY") + g("SNLIQXY"), 1e-30)).astype(f32)
    layer = np.arange(-2, 1)[None, :]                                  # snow layer index -2..0 -> columns 0..2 of the 3 snow layers
    pre["FICEOLD"] = np.where((layer[:, :] >= (isn[:, None] + 1)) , ice, f32(0.0)).astype(f32)
    # ---- parameters: LISparam fields keep their 4.0.1 names; soil properties come from the per-layer arrays
    for k, v in a.items():
        if k.startswith("p_"):
            pre[k] = v
    for k, v in a.items():                                      # global table scalars (NoahmpIO%<NAME>_TABLE) not already in LISparam
        if k.endswith("_TABLE") and ("p_" + k[:-6]) not in pre and v.ndim == 1:
            pre["p_" + k[:-6]] = v
    for old, new in (("MXSNALB", "p_mxsnalb"), ("MNSNALB", "p_mnsnalb"), ("T_ulimit", "p_t_ulimit"), ("T_llimit", "p_t_llimit"), ("T_mlimit", "p_t_mlimit")):
        pre["p_" + old] = g(new)
    pre["p_F1"] = z()
    if int(g("IOPT_IRR")[0]) == 2 and "PLANTING" in a:                  # irrigation trigger 2: DatePlanting / DateHarvest from the PLANTING / HARVEST maps (integers)
        pre["p_PLTDAY"] = np.trunc(g("PLANTING")).astype(f32); pre["p_HSDAY"] = np.trunc(g("HARVEST")).astype(f32)
    if int(g("IOPT_CROP")[0]) > 0:                                     # crop tiles: parameters of the crop table (BiochemVarInTransfer / EnergyVarInTransfer)
        tab = params_mp50.read_crop_table(T_.read_namelists(os.environ.get("NOAHMP50_TABLE", table)))
        ct = np.where(g("ivgtyp") == a["ISCROP_TABLE"], tab["DEFAULT_CROP"], 0)
        assert not (g("CROPCAT") > 0).any(), "CROPCAT > 0 is not ported"
        ov = params_mp50.crop_overrides(tab, ct, g("PLANTING"), g("HARVEST"), g("SEASON_GDD"))
        for k, v in ov.items():
            pre["p_" + k] = np.where((ct > 0).reshape((-1,) + (1,) * (v.ndim - 1)), v, pre["p_" + k]).astype(f32) if ("p_" + k) in pre else v
    post = {}
    for new, old in (("FSAXY", "FSA"), ("FIRAXY", "FIRA"), ("HFX", "FSH"), ("GRDFLX", "SSOIL"), ("ECANXY", "ECAN"), ("ETRANXY", "ETRAN"),
                     ("EDIRXY", "ESOIL"), ("TRADXY", "TRAD"), ("RUNSFXY", "RUNSF"), ("RUNSBXY", "RUNSB"), ("T2MVXY", "T2MV"), ("T2MBXY", "T2MB"),
                     ("TGVXY", "TGV"), ("TGBXY", "TGB"), ("TVXY", "TV"), ("TGXY", "TG"), ("EAHXY", "EAH"), ("TAHXY", "TAH"), ("CMXY", "CM"),
                     ("CHXY", "CH"), ("ALBOLDXY", "ALBOLD"), ("SNEQVOXY", "SNEQVO"), ("FWETXY", "FWET"), ("CANLIQXY", "CANLIQ"),
                     ("CANICEXY", "CANICE"), ("QSNOWXY", "QSNOW"), ("ISNOWXY", "ISNOW"), ("SNOWH", "SNDPTH"), ("SNOW", "SWE"), ("ZWTXY", "ZWT"),
                     ("WAXY", "WA"), ("WTXY", "WT"), ("WSLAKEXY", "WSLAKE"), ("TAUSSXY", "TAUSS"), ("LAI", "PLAI"), ("XSAIXY", "PSAI"),
                     ("APARXY", "APAR"), ("PSNXY", "PSN"), ("SAVXY", "SAV"), ("SAGXY", "SAG"), ("FVEGXY", "FVEGMP"), ("QSNBOTXY", "QSNBOT"),
                     ("PONDINGXY", "PONDING"), ("RSSUNXY", "RSSUN"), ("RSSHAXY", "RSSHA"), ("BGAPXY", "BGAP"), ("WGAPXY", "WGAP"),
                     ("CHVXY", "CHV"), ("CHBXY", "CHB"), ("SHGXY", "SHG"), ("SHCXY", "SHC"), ("SHBXY", "SHB"), ("EVGXY", "EVG"), ("EVBXY", "EVB"),
                     ("GHVXY", "GHV"), ("GHBXY", "GHB"), ("TRXY", "TR"), ("EVCXY", "EVC"), ("FPICEXY", "FPICE"), ("QSFC", "QSFC1D"),
                     ("SH2O", "SMH2O"), ("SMOIS", "SMC"), ("ZSNSOXY", "ZSNSO"), ("SNICEXY", "SNICE"), ("SNLIQXY", "SNLIQ"), ("RELSMC", "RELSMC"),
                     ("LFMASSXY", "LFMASS"), ("RTMASSXY", "RTMASS"), ("STMASSXY", "STMASS"), ("WOODXY", "WOOD"), ("STBLCPXY", "STBLCP"),
                     ("FASTCPXY", "FASTCP"), ("GRAINXY", "GRAIN"), ("GDDXY", "GDD"), ("PGSXY", "PGS"), ("GPPXY", "GPP"), ("NPPXY", "NPP"), ("NEEXY", "NEE"),
                     ("SMCWTDXY", "SMCWTD"), ("IRWATSI", "IRWATSI"), ("IRWATMI", "IRWATMI"), ("IRWATFI", "IRWATFI"),
                     ("IRNUMSI", "IRNUMSI"), ("IRNUMMI", "IRNUMMI"), ("IRNUMFI", "IRNUMFI"), ("IRRSPLH", "IRRSPLH"), ("FGEV_PET", "FGEV_PET"), ("FCEV_PET", "FCEV_PET"), ("FCTR_PET", "FCTR_PET")):
        if new in b:
            post[old] = b[new]
    post["RUNSF"] = b["RUNSFXY"] / dt                                  # 5.0 reports runoff per time step [mm], 4.0.1 [mm/s]
    post["RUNSB"] = b["RUNSBXY"] / dt
    post["STC"] = np.concatenate([b["TSNOXY"], b["TSLB"]], axis=1)
    post["LH"] = b["LH"]                                               # = FCEV + FGEV + FCTR
    post["FSNO"] = None
    post = {k: v for k, v in post.items() if v is not None}
    return pre, post
