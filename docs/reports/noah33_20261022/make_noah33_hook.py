"""Dump hook for Noah 3.3 (BUILD COPY only): writes the same jaxdump_noah39.bin record layout as noah39_jaxdump.patch around the
land SFLX call of noah33_main.F90 (fields Noah 3.3 does not have are written as 0). usage: python3 make_noah33_hook.py <build>/lis/surfacemodels/land/noah.3.3"""
import re, sys
p = sys.argv[1] + "/noah33_main.F90"
s = open(p).read()
assert "jaxdump_mod" not in s
S = "noah33_struc(n)%noah(t)%"
pre1 = ["ffrozp", "dt", "zlvl", "lwdn", "soldn", "solnet", "sfcprs", "prcp", "sfctmp", "q2", "sfcspd", "th2", "q2sat", "dqsdt2",
        "real(" + S + "vegt)", "real(soiltyp)", "slope", S + "shdfac", S + "shdmin", S + "shdmax", S + "alb", "snoalb", "tbot", S + "z0brd", S + "z0",
        S + "emiss", S + "embrd", S + "cmc", S + "t1", S + "stc(1)", S + "stc(2)", S + "stc(3)", S + "stc(4)", S + "smc(1)", S + "smc(2)",
        S + "smc(3)", S + "smc(4)", S + "sh2o(1)", S + "sh2o(2)", S + "sh2o(3)", S + "sh2o(4)", S + "snowh", S + "sneqv", S + "albedo", S + "ch", S + "cm",
        "0.0", "0.0", S + "snotime1", "ribb", S + "lai", S + "rsmin", "jd_ch0", "jd_cm0", S + "z0_old", "t1v", "th2v", "sfcspd"]
pre2 = [S + "smcwlt", S + "smcdry", S + "smcref", S + "smcmax", "real(" + S + "nroot)", S + "hs", S + "rgl", S + "snup", S + "cmcmax", S + "rsmax", S + "topt",
        S + "bexp", S + "dksat", S + "dwsat", S + "f1", S + "quartz", S + "psisat", S + "czil", S + "sbeta", S + "fxexp", S + "csoil", S + "salp", S + "kdt",
        S + "cfactr", S + "zbot", S + "refkdt", S + "frzx", S + "sndens", S + "lvcoef", "0.0", "1.0", "real(isurban)", "real(nsoil)",
        S + "z0brd_old", S + "q1", "real(noah33_struc(n)%iz0tlnd)", "real(noah33_struc(n)%sfcdifoption)", "0.0", "0.0", "0.0", "0.0"]
post = [S + "cmc", S + "t1", S + "stc(1)", S + "stc(2)", S + "stc(3)", S + "stc(4)", S + "smc(1)", S + "smc(2)", S + "smc(3)", S + "smc(4)",
        S + "sh2o(1)", S + "sh2o(2)", S + "sh2o(3)", S + "sh2o(4)", S + "snowh", S + "sneqv", S + "albedo", S + "ch", S + "cm", "sncovr", S + "z0",
        S + "snotime1", S + "q1", S + "emiss", S + "lai", S + "shdfac", "eta", "shtflx", "gflx", "etp", "edir", "ec", "ett", "esnow", S + "runoff1", S + "runoff2",
        "runoff3", "snomlt", "drip", "dew", "beta", "fdown", "rc", "pc", "soilm", "soilw", "tsoil", "flx1", "flx2", "flx3", "ustar", "evp", "eta_kinematic",
        S + "rsmin", S + "smcmax", S + "smcwlt"]
def arr(name, lo, items):
    lines = [", &\n     ".join(items[i:i + 6]) for i in range(0, len(items), 6)]
    return f"           {name}({lo}:{lo+len(items)-1}) = (/ " + ", &\n     ".join(items) + " /)\n"
pre_blk = ("           jd_hdr = (/ LIS_rc%yr, LIS_rc%mo, LIS_rc%da, LIS_rc%hr, LIS_rc%mn, LIS_rc%ss, t, col, row /)\n           jd_pre = 0.0\n"
           + arr("jd_pre", 1, pre1) + arr("jd_pre", 59, pre2))
post_blk = "           jd_post = 0.0\n" + arr("jd_post", 1, post) + "           call jaxdump_write(jd_hdr, jd_pre, jd_post)\n"
assert len(pre1) == 58 and len(pre2) == 41 and len(post) == 56, (len(pre1), len(pre2), len(post))
s = s.replace("  use noah33_lsmMod\n", "  use noah33_lsmMod\n  use jaxdump_mod\n", 1)
s = s.replace("  real        :: ustar\n", "  real        :: ustar\n  real        :: jd_pre(JD_NPRE), jd_post(JD_NPOST), jd_ch0, jd_cm0\n  integer     :: jd_hdr(JD_NHDR)\n", 1)
k = s.index("        if ((noah33_struc(n)%forcing_ch.eq.0)")
s = s[:k] + f"        jd_ch0 = {S}ch\n        jd_cm0 = {S}cm\n" + s[k:]
k = s.index("           call sflx(")
s = s[:k] + pre_blk + s[k:]
k = s.index("                ,pcpdrp &\n#endif\n                )\n", k) + len("                ,pcpdrp &\n#endif\n                )\n")
s = s[:k] + post_blk + s[k:]
open(p, "w").write(s)
print("hook inserted")
