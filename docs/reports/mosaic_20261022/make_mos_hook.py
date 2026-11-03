"""Dump hook for Mosaic (BUILD COPY only): writes jaxdump_noah39.bin records (same container as the Noah hooks) around CALL MOSTILE in mos_main.F90.
pre (87 values): vegt, vegp(24), vegip(6), soilp(10), tmp2m humid swdn lwdwn wnd sfcprs tpcp cpcp, sunang_raw lat lon, ct sot qa sowet(3) ics snow,
  then the arguments computed by mos_main for MOSTILE: dedqa dedtc hsturb dhsdqa dhsdtc qm cd sunang pardir pardif swnet psur zlai green z2 sqscat rsoil1 rsoil2 rdc u2fac qsattc dqsdtc alwrad blwrad ra eturb tm
post (32 values): ct sot qa sowet(3) ics snow evap shflux runoff eint esoi eveg esno smelt hlatn hlwup gdrain runsrf fwsoil ghflux water1 water2 water3 por1 por2 por3 albave snwfrc tempac tempcc
usage: python3 make_mos_hook.py <build>/lis/surfacemodels/land/mosaic"""
import sys
p = sys.argv[1] + "/mos_main.F90"
s = open(p).read()
assert "jaxdump_mod" not in s
M = "MOS_STRUC(N)%MOS(M)%"
pre = ["real(" + M + "vegt)"] + [f"{M}vegp({i})" for i in range(1, 25)] + [f"{M}vegip({i})" for i in range(1, 7)] + [f"{M}soilp({i})" for i in range(1, 11)] + \
      ["tmp2m", "humid", "swdn", "lwdwn", "wnd", "sfcprs", "tpcp", "cpcp", "jd_sun0", "LIS_domain(n)%grid(index1)%lat", "LIS_domain(n)%grid(index1)%lon",
       M + "ct", M + "sot", M + "qa", M + "sowet(1)", M + "sowet(2)", M + "sowet(3)", M + "ics", M + "snow"] + \
      "dedqa dedtc hsturb dhsdqa dhsdtc qm cd sunang pardir pardif swnet psur zlai green z2 sqscat rsoil1 rsoil2 rdc u2fac qsattc dqsdtc alwrad blwrad ra eturb tmp2m".split()
post = [M + "ct", M + "sot", M + "qa", M + "sowet(1)", M + "sowet(2)", M + "sowet(3)", M + "ics", M + "snow"] + \
       "evap shflux runoff eint esoi eveg esno smelt hlatn hlwup gdrain runsrf fwsoil ghflux water1 water2 water3 por1 por2 por3 albave snwfrc tempac tempcc".split()
assert len(pre) == 87 and len(post) == 32, (len(pre), len(post))
def arr(name, items):
    return f"        {name}(1:{len(items)}) = (/ " + ", &\n     ".join(items) + " /)\n"
s = s.replace("  use mos_lsmMod     \n", "  use mos_lsmMod     \n  use jaxdump_mod\n", 1)
s = s.replace("  integer index1\n", "  integer index1\n  real :: jd_pre(JD_NPRE), jd_post(JD_NPOST), jd_sun0\n  integer :: jd_hdr(JD_NHDR)\n", 1)
# raw (unclipped) cosine of zenith angle = value returned by ASTRO: capture right after the call
k = s.index("        CALL ASTRO(")
k2 = s.index("\n", s.index("1,SUNANG,RA)", k)) + 1
s = s[:k2] + "        jd_sun0 = SUNANG\n" + s[k2:]
k = s.index("        CALL MOSTILE(")
pre_blk = ("        jd_hdr = (/ LIS_rc%yr, LIS_rc%mo, LIS_rc%da, LIS_rc%hr, LIS_rc%mn, LIS_rc%ss, m, LIS_surface(n,LIS_rc%lsm_index)%tile(m)%col, LIS_surface(n,LIS_rc%lsm_index)%tile(m)%row /)\n"
           "        jd_pre = 0.0\n" + arr("jd_pre", pre))
s = s[:k] + pre_blk + s[k:]
k = s.index("mos_struc(n)%mos(m)%vegt)\n", s.index("        CALL MOSTILE(")) + len("mos_struc(n)%mos(m)%vegt)\n")
post_blk = "        jd_post = 0.0\n" + arr("jd_post", post) + "        call jaxdump_write(jd_hdr, jd_pre, jd_post)\n"
s = s[:k] + post_blk + s[k:]
open(p, "w").write(s)
print("mos hook inserted")
