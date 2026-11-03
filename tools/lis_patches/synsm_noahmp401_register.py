"""Register Noah-MP.4.0.1 with the 'Synthetic SM' DA observation set (upstream registers it for Noah 3.x/3.6/3.9 and
Noah-MP 3.6 only). Copies the SMOPS-ASCAT registration block of Noah-MP.4.0.1 (same state/obs-prediction/QC routines).
Applied to the BUILD COPY only: python3 synsm_noahmp401_register.py <build>/lis/plugins/LIS_lsmda_pluginMod.F90"""
import sys, re
p = sys.argv[1]
s = open(p).read()
if "Noah-MP.4.0.1 Synthetic SM" in s:
    sys.exit("already applied")
a = s.index("! Noah-MP.4.0.1 RT SMOPS ASCAT soil moisture")
b = s.index("! Noah-MP.4.0.1 ", a + 10)  # next block
blk = s[a:b]
new = blk.replace("RT SMOPS ASCAT soil moisture", "Synthetic SM").replace("LIS_SMOPS_ASCATsmobsId", "LIS_synsmId")
s = s[:b] + "! Noah-MP.4.0.1 Synthetic SM (added for the JAX port twin experiment)\n" + new.split("\n", 1)[1] + "\n" + s[b:]
open(p, "w").write(s)
print("ok", len(new.splitlines()), "lines added")
