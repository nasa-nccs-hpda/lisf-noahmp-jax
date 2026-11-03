"""Dump hook for Noah 2.7.1 (BUILD COPY only): writes jaxdump_noah39.bin records (hdr 9 int32, pre 100 f32, post 60 f32) around the
SFLX call of noah271_main.F90: pre = all SFLX inputs (+ ch/cm/t1v/th2v before SFCDIF), post = the updated states and outputs.
usage: python3 make_noah271_hook.py <build>/lis/surfacemodels/land/noah.2.7.1"""
import sys
p = sys.argv[1] + "/noah271_main.F90"
s = open(p).read()
assert "jaxdump_mod" not in s
S = "noah271_struc(n)%noah(t)%"
pre = ["sbeta", "fxexp", "czil", "cfactr", "cmcmax", "rsmax", "topt", "refdk", "refkdt", "csoil", "frzk", "ffrozp", "real(ice)", "dt", "zlvl",
       "lwdn", "soldn", "sfcprs", "prcp", "sfctmp", "q2", "sfcspd", "th2", "q2sat", "dqsdt2", "slope", "shdfac", "shdmin", "alb", "snoalb",
       "rsmin", "rgl", "hs", "snup", "z0", S + "emiss", "xlai", "real(nroot)", "psisat", "bexp", "dksat", "smcmax", "quartz", "dwsat", "smcwlt",
       "smcref", "smcdry", "f1", "kdt", "frzx", "tbot", S + "cmc", S + "t1",
       S + "stc(1)", S + "stc(2)", S + "stc(3)", S + "stc(4)", S + "smc(1)", S + "smc(2)", S + "smc(3)", S + "smc(4)",
       S + "sh2o(1)", S + "sh2o(2)", S + "sh2o(3)", S + "sh2o(4)", S + "snowh", S + "sneqv", "jd_ch0", "jd_cm0", "t1v", "th2v", "real(soiltyp)", "real(" + S + "vegt)",
       S + "uwind", S + "vwind", "real(noah271_struc(n)%forc_count)", "real(tid)", S + "tair", S + "psurf", S + "lwdown", S + "swdown"]
pre = [x.replace("sbeta", S + "sbeta", 1) if x == "sbeta" else x for x in pre]
for k in ("fxexp", "czil", "cfactr", "cmcmax", "rsmax", "topt", "refdk", "refkdt", "csoil", "frzk"):
    pre[pre.index(k)] = S + k
post = [S + "cmc", S + "t1", S + "stc(1)", S + "stc(2)", S + "stc(3)", S + "stc(4)", S + "smc(1)", S + "smc(2)", S + "smc(3)", S + "smc(4)",
        S + "sh2o(1)", S + "sh2o(2)", S + "sh2o(3)", S + "sh2o(4)", S + "snowh", S + "sneqv", "albedo", S + "ch", S + "cm", "evp", "eta", "shtflx",
        "eta_kinematic", "ec", "edir", "et(1)", "et(2)", "et(3)", "et(4)", "ett", "esnow", "drip", "dew", "beta", "etp", "gflx", "flx1", "flx2", "flx3",
        "snomlt", "sncovr", S + "runoff1", S + "runoff2", "runoff3", "rc", "pc", "rcs", "rct", "rcq", "rcsoil", "q1", "soilm", "etanrg", "tsoil",
        "emissi_out", "ustar", "mstavrz", "mstavtot"]
assert len(pre) == 81 and len(post) == 58, (len(pre), len(post))
def arr(name, items):
    return f"        {name}(1:{len(items)}) = (/ " + ", &\n     ".join(items) + " /)\n"
pre_blk = ("        jd_hdr = (/ LIS_rc%yr, LIS_rc%mo, LIS_rc%da, LIS_rc%hr, LIS_rc%mn, LIS_rc%ss, t, "
           "LIS_surface(n,LIS_rc%lsm_index)%tile(t)%col, LIS_surface(n,LIS_rc%lsm_index)%tile(t)%row /)\n        jd_pre = 0.0\n" + arr("jd_pre", pre))
post_blk = "        jd_post = 0.0\n" + arr("jd_post", post) + "        call jaxdump_write(jd_hdr, jd_pre, jd_post)\n"
s = s.replace("  use noah271_lsmMod \n", "  use noah271_lsmMod \n  use jaxdump_mod\n", 1)
assert "jaxdump_mod" in s
s = s.replace("  real :: ustar\n", "", 1) if False else s
k = s.index("  integer :: i\n")
s = s[:k] + "  real    :: jd_pre(JD_NPRE), jd_post(JD_NPOST), jd_ch0, jd_cm0\n  integer :: jd_hdr(JD_NHDR)\n" + s[k:]
k = s.index("        if ( noah271_struc(n)%forcing_ch == 0 ) then")
s = s[:k] + f"        jd_ch0 = {S}ch\n        jd_cm0 = {S}cm\n" + s[k:]
k = s.index("        CALL SFLX (T,")
s = s[:k] + pre_blk + s[k:]
k = s.index("MSTAVRZ,MSTAVTOT,SOILM,ETANRG,TSOIL,EMISSI_OUT)\n", k) + len("MSTAVRZ,MSTAVTOT,SOILM,ETANRG,TSOIL,EMISSI_OUT)\n")
s = s[:k] + post_blk + s[k:]
open(p, "w").write(s)
print("hook inserted")
