"""Apply the Noah-MP 4.0.1 dump hook to the BUILD COPY of module_sf_noahmpdrv_401.F90 (never the repo)."""
import os
import sys, re
BUILD = os.path.join(os.environ["LISF_BUILD"], "lis/surfacemodels/land/noahmp.4.0.1/phys/")
path = BUILD + 'module_sf_noahmpdrv_401.F90'
src = open(path).read()
if 'jdm_begin' in src:
    sys.exit('already applied')
S = lambda s: s.split()
pre_sc = S("""ITIMESTEP LAT YEARLEN JULIAN COSZ DT DX DZ8W1D FVEG FVGMAX VEGTYP ICE IST CROPTYPE NSOIL T_ML P_ML PSFC U_ML V_ML Q_ML QC SWDN LWDN
PRCPCONV PRCPNONC PRCPSHCV PRCPSNOW PRCPGRPL PRCPHAIL TBOT CO2PP O2PP FOLN Z_ML ALBOLD SNEQVO TAH EAH FWET CANLIQ CANICE TV TG QSFC1D QSNOW
ISNOW SNDPTH SWE ZWT WA WT WSLAKE LFMASS RTMASS STMASS WOOD STBLCP FASTCP PLAI PSAI CM CH TAUSS SMCWTD rivsto fldsto fldfrc
IDVEG IOPT_CRS IOPT_BTR IOPT_RUN IOPT_SFC IOPT_FRZ IOPT_INF IOPT_RAD IOPT_ALB IOPT_SNF IOPT_TBOT IOPT_STC IOPT_RSF IOPT_SOIL IOPT_PEDO IOPT_CROP""")
pre_arr = S("ZSOIL SMCEQ FICEOLD STC SMH2O SMC ZSNSO SNICE SNLIQ")
par_sc = S("""ISWATER ISBARREN ISICE ISCROP EBLFOREST URBAN_FLAG CH2OP DLEAF Z0MVT HVT HVB DEN RC MFSNO SLA DILEFC DILEFW FRAGR LTOVRC C3PSN KC25 AKC KO25 AKO
VCMX25 AVCMX BP MP QE25 AQE RMF25 RMS25 RMR25 ARM FOLNMX TMIN XL MRP CWPVT WRRAT WDPOOL TDLEF NROOT RGL RSMIN HS TOPT RSMAX BETADS BETAIS CO2 O2
TIMEAN FSATMX Z0SNO SSI SWEMX RSURF_SNOW F1 SLOPE CSOIL ZBOT CZIL REFDK REFKDT KDT FRZX sndecayexp MXSNALB MNSNALB T_ulimit T_llimit T_mlimit snowf_scalef""")
par_arr = S("SAIM LAIM RHOL RHOS TAUL TAUS ALBSAT ALBDRY ALBICE ALBLAK OMEGAS EG BEXP SMCDRY SMCWLT SMCREF SMCMAX PSISAT DKSAT DWSAT QUARTZ")
post_sc = S("""ALBOLD SNEQVO TAH EAH FWET CANLIQ CANICE TV TG QSFC1D QSNOW ISNOW SNDPTH SWE ZWT WA WT WSLAKE LFMASS RTMASS STMASS WOOD STBLCP FASTCP PLAI PSAI
CM CH TAUSS SMCWTD DEEPRECH RECH Z0WRF FSA FSR FIRA FSH SSOIL FCEV FGEV FCTR ECAN ETRAN ESOIL TRAD SUBSNOW TGB TGV T2MV T2MB Q2MV Q2MB RUNSF RUNSB
APAR PSN SAV SAG FSNO NEE GPP NPP FVEGMP SALB QSNBOT PONDING PONDING1 PONDING2 RSSUN RSSHA BGAP WGAP CHV CHB EMISSI SHG SHC SHB EVG EVB GHV GHB
IRG IRC IRB TR EVC FGEV_PET FCEV_PET FCTR_PET CHLEAF CHUC CHV2 CHB2 FPICE PAHV PAHG PAHB PAH LAISUN LAISHA RB""")
post_arr = S("STC SMH2O SMC ZSNSO SNICE SNLIQ RELSMC")
def adds(sc, arr, pref=''):
    out = []
    for v in sc: out.append(f"            call jdm_add('{v}', {pref}{v})")
    for v in arr: out.append(f"            call jdm_add('{v}', {pref}{v})")
    return out
keep = ["            jdm_keep = .true.",
        "            if (jdm_snowonly()) jdm_keep = (SWE > 0.) .or. (SNDPTH > 0.) .or. (ISNOW < 0) .or. &",
        "                 (((PRCPCONV+PRCPNONC+PRCPSHCV) > 0.) .and. (T_ML < 278.))",
        "            if (jdm_noveg()) jdm_keep = .false."]
pre = keep + ["            call jdm_begin()"] + adds(pre_sc, pre_arr)
pre += [f"            call jdm_add('p_{v}', parameters%{v})" for v in par_sc + par_arr]
pre += ["            call jdm_flush(1)"]
post = ["            call jdm_begin()"] + adds(post_sc, post_arr) + ["            call jdm_flush(2)"]
marker = "         CALL NOAHMP_SFLX (parameters, &"
i = src.index(marker)
src = src[:i] + "\n".join(pre) + "\n" + src[i:]
# post: after the statement 'LH       (I,J)                = FCEV + FGEV + FCTR'
m2 = "            LH       (I,J)                = FCEV + FGEV + FCTR\n"
j = src.index(m2) + len(m2)
src = src[:j] + "\n".join(post) + "\n" + src[j:]
# use statement
k = src.index("    USE module_sf_noahmp_glacier_401")
src = src[:k] + "    USE jaxdump_mp_mod\n" + src[k:]
# ---- GLA_HOOK: land-ice branch (NOAHMP_GLACIER)
g_pre_sc = S("""ITIMESTEP COSZ DT NSNOW NSOIL T_ML P_ML U_ML V_ML Q_ML SWDN PRCP LWDN TBOT Z_ML IOPT_SNDPTH QSNOW SNEQVO ALBOLD CM CH ISNOW SWE SNDPTH TG QSFC1D TAUSS IOPT_ALB IOPT_SNF IOPT_TBOT IOPT_STC IOPT_GLA""")
g_pre_arr = S("FICEOLD ZSOIL SMC ZSNSO SNICE SNLIQ STC SMH2O")
g_post_sc = S("""QSNOW SNEQVO ALBOLD CM CH ISNOW SWE SNDPTH TG QSFC1D TAUSS FSA FSR FIRA FSH FGEV SSOIL TRAD ESOIL RUNSF RUNSB SAG SALB QSNBOT PONDING PONDING1 PONDING2 T2MB Q2MB EMISSI FPICE CHB2 SUBSNOW""")
g_post_arr = S("SMC ZSNSO SNICE SNLIQ STC SMH2O")
gp = ["            call jdg_begin()"] + [f"            call jdg_add('{v}', {v})" for v in g_pre_sc + g_pre_arr] + ["            call jdg_flush(1)"]
gq = ["            call jdg_begin()"] + [f"            call jdg_add('{v}', {v})" for v in g_post_sc + g_post_arr] + ["            call jdg_flush(2)"]
mk = "         CALL NOAHMP_GLACIER("
ii = src.index(mk)
src = src[:ii] + "\n".join(gp) + "\n" + src[ii:]
m3 = "         FSNO   = 1.0       \n"
jj = src.index(m3)
src = src[:jj] + "\n".join(gq) + "\n" + src[jj:]
k2 = src.index("    USE jaxdump_mp_mod")
src = src[:k2] + "    USE jaxdump_gla_mod\n" + src[k2:]
open(path, 'w').write(src)
print('pre', len(pre), 'post', len(post))
