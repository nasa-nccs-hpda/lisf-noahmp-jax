"""Extra dump inside CLSMF25 (BUILD COPY of clsmf25_model.F90): after the three energy balances and EIRFRC, tag-2 records with
intermediate arrays. usage: python3 make_clsm_inter_hook.py <build>/lis/surfacemodels/land/clsm.f2.5"""
import sys
d = sys.argv[1]
p = f"{d}/clsmf25_model.F90"
s = open(p).read()
assert "jdc_begin(NCH, 2)" not in s
names = ["AR1", "AR2", "AR4", "RZEQOL", "RZI", "SWSRF1", "SWSRF2", "SWSRF4", "SRFMX", "SRFMN", "HFTDS1", "HFTDS2", "HFTDS4",
         "DHFT1", "DHFT2", "DHFT4", "RCUN", "EMAXRT", "POTFRC", "TC1SF", "TC2SF", "TC4SF", "QA1", "QA2", "QA4",
         "EVAP1", "EVAP2", "EVAP4", "SHFLUX1", "SHFLUX2", "SHFLUX4", "HLWUP1", "HLWUP2", "HLWUP4", "GHFLUX1", "GHFLUX2",
         "GHFLUX4", "RX11", "RX21", "RX12", "RX22", "RX14", "RX24", "EIRFRC", "CSOIL", "SATCAP", "CAPAC", "CATDEF", "RZEXC",
         "SRFEXC"]
blk = "      call jdc_begin(NCH, 2)\n" + "".join(f"      call jdc_v1({n})\n" for n in names)
marker = "WRITE(*,*) 'EIRFRC OK'"
i = s.index(marker)
j = s.index("ENDIF", i) + len("ENDIF")
s = s[:j] + "\n" + blk + s[j:]
k = s.index("IMPLICIT NONE", s.index("SUBROUTINE CLSMF25"))
s = s[:k] + "use jaxdump_clsm_mod\n      " + s[k:]
open(p, "w").write(s)
open(f"{d}/jaxdump_clsm_inter.layout", "w").write("\n".join(names))
print("inserted", len(names))
