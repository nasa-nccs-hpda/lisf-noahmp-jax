"""Dump hook for Noah-MP 3.6 (BUILD COPY only): around the NOAHMP_SFLX_36 call of noahmp_driver_36.F90 it records every input/state (pre) and
every state/output (post) with the self-describing layout of jaxdump_mp_mod.F90, using the names of the 4.0.1 hook where an equivalent exists.
usage: python3 make_noahmp36_hook.py <build>/lis/surfacemodels/land/noahmp.3.6"""
import sys
path = sys.argv[1] + "/noahmp_driver_36.F90"
s = open(path).read()
assert "jaxdump_mp_mod" not in s
SC = {   # python name : fortran expression
 "DT": "dt", "LAT": "lat", "YEARLEN": "yearlen", "JULIAN": "julian", "COSZ": "cosz", "FVGMAX": "shdmax", "SHDFAC": "shdfac", "VEGTYP": "vegetype",
 "ICE": "ice", "IST": "ist", "ISC": "isc", "NSOIL": "nsoil", "T_ML": "sfctmp", "P_ML": "sfcprs", "PSFC": "psfc", "U_ML": "uu", "V_ML": "vv", "Q_ML": "q2",
 "QC": "qc", "SWDN": "soldn", "LWDN": "lwdn", "PRCP": "prcp", "TBOT": "tbot", "CO2PP": "co2air", "O2PP": "o2air", "FOLN": "foln", "Z_ML": "zlvl",
 "ALBOLD": "albold", "SNEQVO": "sneqvo", "TAH": "tah", "EAH": "eah", "FWET": "fwet", "CANLIQ": "canliq", "CANICE": "canice", "TV": "tv", "TG": "tg",
 "QSFC1D": "qsfc", "QSNOW": "qsnow", "ISNOW": "isnow", "SNDPTH": "snowh", "SWE": "sneqv", "ZWT": "zwt", "WA": "wa", "WT": "wt", "WSLAKE": "wslake",
 "LFMASS": "lfmass", "RTMASS": "rtmass", "STMASS": "stmass", "WOOD": "wood", "STBLCP": "stblcp", "FASTCP": "fastcp", "PLAI": "lai", "PSAI": "sai",
 "CM": "cm", "CH": "ch", "TAUSS": "tauss", "SMCWTD": "smcwtd", "rivsto": "rivsto", "fldsto": "fldsto", "fldfrc": "fldfrc",
 "IDVEG": "dveg", "IOPT_CRS": "opt_crs", "IOPT_BTR": "opt_btr", "IOPT_RUN": "opt_run", "IOPT_SFC": "opt_sfc", "IOPT_FRZ": "opt_frz", "IOPT_INF": "opt_inf",
 "IOPT_RAD": "opt_rad", "IOPT_ALB": "opt_alb", "IOPT_SNF": "opt_snf", "IOPT_TBOT": "opt_tbot", "IOPT_STC": "opt_stc"}
AR = {"ZSOIL": "zsoil", "SMCEQ": "smceq", "FICEOLD": "ficeold", "STC": "stc", "SMH2O": "sh2o", "SMC": "smc", "ZSNSO": "zsnso", "SNICE": "snice", "SNLIQ": "snliq"}
# parameters (names of the 4.0.1 'parameters' struct)
V = "vegetype"
PV = {k: f"g_{k}({V})" for k in "CH2OP DLEAF Z0MVT HVT HVB RC XL CWPVT C3PSN KC25 AKC KO25 AKO AVCMX AQE LTOVRC DILEFC DILEFW RMF25 SLA FRAGR TMIN VCMX25 TDLEF BP MP QE25 RMS25 RMR25 ARM FOLNMX WDPOOL WRRAT MRP".split()}
PS = {"ISWATER": "g_ISWATER", "ISBARREN": "g_ISBARREN", "ISICE": "g_ISSNOW", "EBLFOREST": "g_EBLFOREST", "URBAN_FLAG": "merge(1.0,0.0,vegetype==isurban)",
      "NROOT": "real(g_NROOT)", "RGL": "g_RGL", "RSMIN": "g_RSMIN", "HS": "g_HS", "TOPT": "g_TOPT", "RSMAX": "g_RSMAX",
      "BETADS": "g_BETADS", "BETAIS": "g_BETAIS", "CO2": "355.e-6", "O2": "0.209", "TIMEAN": "g_TIMEAN", "FSATMX": "g_FSATMX", "MFSNO": "g_M", "Z0SNO": "g_Z0SNO",
      "SSI": "g_SSI", "SWEMX": "g_SWEMX", "F1": "g_F1", "SLOPE": "g_SLOPE", "CSOIL": "g_CSOIL", "ZBOT": "g_ZBOT", "CZIL": "g_CZIL", "KDT": "g_KDT", "FRZX": "g_FRZX",
      "REFDK": "p_refdk", "REFKDT": "p_refkdt"}
PA = {"RHOL": "g_RHOL(vegetype,1:2)", "RHOS": "g_RHOS(vegetype,1:2)", "TAUL": "g_TAUL(vegetype,1:2)", "TAUS": "g_TAUS(vegetype,1:2)",
      "SAIM": "g_SAIM(vegetype,1:12)", "LAIM": "g_LAIM(vegetype,1:12)", "ALBSAT": "g_ALBSAT(isc,1:2)", "ALBDRY": "g_ALBDRY(isc,1:2)", "ALBICE": "g_ALBICE(1:2)",
      "ALBLAK": "g_ALBLAK(1:2)", "OMEGAS": "g_OMEGAS(1:2)", "EG": "g_EG(1:2)"}
for k in "BEXP SMCDRY SMCWLT SMCREF SMCMAX PSISAT DKSAT DWSAT QUARTZ".split():
    PA[k] = f"(/g_{k},g_{k},g_{k},g_{k}/)"
POST_SC = {"ALBOLD": "albold", "SNEQVO": "sneqvo", "TAH": "tah", "EAH": "eah", "FWET": "fwet", "CANLIQ": "canliq", "CANICE": "canice", "TV": "tv", "TG": "tg", "QSFC1D": "qsfc",
 "QSNOW": "qsnow", "ISNOW": "isnow", "SNDPTH": "snowh", "SWE": "sneqv", "ZWT": "zwt", "WA": "wa", "WT": "wt", "WSLAKE": "wslake", "LFMASS": "lfmass", "RTMASS": "rtmass",
 "STMASS": "stmass", "WOOD": "wood", "STBLCP": "stblcp", "FASTCP": "fastcp", "PLAI": "lai", "PSAI": "sai", "CM": "cm", "CH": "ch", "TAUSS": "tauss", "SMCWTD": "smcwtd",
 "DEEPRECH": "deeprech", "RECH": "rech", "FSA": "fsa", "FSR": "fsr", "FIRA": "fira", "FSH": "fsh", "SSOIL": "ssoil", "FCEV": "fcev", "FGEV": "fgev", "FCTR": "fctr", "ECAN": "ecan",
 "ETRAN": "etran", "ESOIL": "edir", "TRAD": "trad", "SUBSNOW": "subsnow", "TGB": "tgb", "TGV": "tgv", "T2MV": "t2mv", "T2MB": "t2mb", "Q2MV": "q2v", "Q2MB": "q2b",
 "RUNSF": "runsrf", "RUNSB": "runsub", "APAR": "apar", "PSN": "psn", "SAV": "sav", "SAG": "sag", "FSNO": "fsno", "NEE": "nee", "GPP": "gpp", "NPP": "npp", "FVEGMP": "fveg",
 "SALB": "albedo", "QSNBOT": "qsnbot", "PONDING": "ponding", "PONDING1": "ponding1", "PONDING2": "ponding2", "RSSUN": "rssun", "RSSHA": "rssha", "BGAP": "bgap",
 "WGAP": "wgap", "CHV": "chv", "CHB": "chb", "EMISSI": "emissi", "SHG": "shg", "SHC": "shc", "SHB": "shb", "EVG": "evg", "EVB": "evb", "GHV": "ghv", "GHB": "ghb",
 "IRG": "irg", "IRC": "irc", "IRB": "irb", "TR": "tr", "EVC": "evc", "CHLEAF": "chleaf", "CHUC": "chuc", "CHV2": "chv2", "CHB2": "chb2", "FPICE": "fpice"}
POST_AR = {"STC": "stc", "SMH2O": "sh2o", "SMC": "smc", "ZSNSO": "zsnso", "SNICE": "snice", "SNLIQ": "snliq"}
pre = ["      call jdm_begin()"]
for k, v in SC.items(): pre.append(f"      call jdm_add('{k}', {v})")
for k, v in AR.items(): pre.append(f"      call jdm_add('{k}', {v})")
for d in (PV, PS, PA):
    for k, v in d.items(): pre.append(f"      call jdm_add('p_{k}', {v})")
pre.append("      call jdm_flush(1)")
post = ["      call jdm_begin()"]
for k, v in POST_SC.items(): post.append(f"      call jdm_add('{k}', {v})")
for k, v in POST_AR.items(): post.append(f"      call jdm_add('{k}', {v})")
post.append("      call jdm_flush(2)")
uses = """  use jaxdump_mp_mod
  use noahmp_globals_36, only: g_M => M, g_TIMEAN => TIMEAN, g_FSATMX => FSATMX, g_Z0SNO => Z0SNO, g_SSI => SSI, g_SWEMX => SWEMX, &
       g_NROOT => NROOT, g_RGL => RGL, g_RSMIN => RSMIN, g_HS => HS, g_RSMAX => RSMAX, g_TOPT => TOPT, &
       g_BEXP => BEXP, g_SMCDRY => SMCDRY, g_F1 => F1, g_SMCMAX => SMCMAX, g_SMCREF => SMCREF, g_PSISAT => PSISAT, g_DKSAT => DKSAT, &
       g_DWSAT => DWSAT, g_SMCWLT => SMCWLT, g_QUARTZ => QUARTZ, g_SLOPE => SLOPE, g_CSOIL => CSOIL, g_ZBOT => ZBOT, g_CZIL => CZIL, g_KDT => KDT, g_FRZX => FRZX, &
       g_CH2OP => CH2OP, g_DLEAF => DLEAF, g_Z0MVT => Z0MVT, g_HVT => HVT, g_HVB => HVB, g_RC => RC, g_XL => XL, g_CWPVT => CWPVT, g_C3PSN => C3PSN, &
       g_KC25 => KC25, g_AKC => AKC, g_KO25 => KO25, g_AKO => AKO, g_AVCMX => AVCMX, g_AQE => AQE, g_LTOVRC => LTOVRC, g_DILEFC => DILEFC, g_DILEFW => DILEFW, &
       g_RMF25 => RMF25, g_SLA => SLA, g_FRAGR => FRAGR, g_TMIN => TMIN, g_VCMX25 => VCMX25, g_TDLEF => TDLEF, g_BP => BP, g_MP => MP, g_QE25 => QE25, &
       g_RMS25 => RMS25, g_RMR25 => RMR25, g_ARM => ARM, g_FOLNMX => FOLNMX, g_WDPOOL => WDPOOL, g_WRRAT => WRRAT, g_MRP => MRP, &
       g_RHOL => RHOL, g_RHOS => RHOS, g_TAUL => TAUL, g_TAUS => TAUS
  use noahmp_veg_parameters_36, only: g_ISWATER => ISWATER, g_ISBARREN => ISBARREN, g_ISSNOW => ISSNOW, g_EBLFOREST => EBLFOREST, g_LAIM => LAIM, g_SAIM => SAIM
  use noahmp_rad_parameters_36, only: g_ALBSAT => ALBSAT, g_ALBDRY => ALBDRY, g_ALBICE => ALBICE, g_ALBLAK => ALBLAK, g_OMEGAS => OMEGAS, g_BETADS => BETADS, g_BETAIS => BETAIS, g_EG => EG
"""
k = s.index("  use noahmp_routines_36, only: noahmp_sflx_36, redprm")
k = s.index("\n", k) + 1
s = s[:k] + uses + s[k:]
mark = "  call  noahmp_sflx_36 ("
i = s.index(mark)
s = s[:i] + "\n".join(pre) + "\n" + s[i:]
m2 = "  qfx = fgev + fcev + fctr"
j = s.index(m2, i)
s = s[:j] + "\n".join(post) + "\n" + s[j:]
open(path, "w").write(s)
print("pre", len(pre), "post", len(post))
