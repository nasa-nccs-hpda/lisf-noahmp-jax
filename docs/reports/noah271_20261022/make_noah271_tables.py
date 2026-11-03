"""Build Noah 2.7.1 vegetation and soil parameter files (the formats read by noah271_setvegparms / noah271_setsoils)
from the public Noah 3.3 tables (MODIFIED_IGBP_MODIS_NOAH 20-class vegetation, STAS soil). The public LIS data set has no Noah 2.7.1 tables, so these are
DERIVED (documented in README.txt); the Fortran reference and the JAX port both read these same files.
usage: make_noah271_tables.py OUTDIR"""
import sys, re, os
src = "/css/lisdata-pub/data/PARAMETERS/noah33_parms/"
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
lines = open(src + "VEGPARM.TBL").read().split("\n")
i = [k for k, l in enumerate(lines) if l.strip() == "MODIFIED_IGBP_MODIS_NOAH"][0]
names = ["NROOT", "RSMIN", "RGL", "HS", "SNUP", "Z0", "LAI"]
def parse(l):
    t = re.split(r"[,\s]+", l.replace("'", " ").strip())
    t = [x for x in t if x != ""]
    return t
rows = []
for l in lines[i + 2:i + 22]:
    t = parse(l)
    num = []
    for x in t:
        try: num.append(float(x))
        except ValueError: break
    rows.append(num)
assert len(rows) == 20 and all(len(r) >= 17 for r in rows), [len(r) for r in rows]
with open(out + "/noah271_vegparms.txt", "w") as f:
    for n in names:
        f.write(f"{n}\n")
        v = []
        for r in rows:
            # r = [id, shdfac, nroot, rs, rgl, hs, snup, maxalb, laimin, laimax, emismin, emismax, albmin, albmax, z0min, z0max, czil, cmcmax]
            d = dict(NROOT=r[2], RSMIN=r[3], RGL=r[4], HS=r[5], SNUP=r[6], Z0=0.5 * (r[14] + r[15]), LAI=0.5 * (r[8] + r[9]))
            v.append(d[n])
        f.write(" ".join(f"{x:.6g}" for x in v) + "\n")
sl = open(src + "SOILPARM.TBL").read().split("\n")
j = [k for k, l in enumerate(sl) if l.strip() == "STAS"][0]
srows = []
for l in sl[j + 2:j + 21]:
    t = re.split(r"[,\s]+", l.split("'")[0].strip()); t = [x for x in t if x]
    srows.append([float(x) for x in t])
assert len(srows) == 19 and all(len(r) == 13 for r in srows)
# columns: id BB DRYSMC F11 MAXSMC REFSMC SATPSI SATDK SATDW WLTSMC QTZ FXEXP SBETA
cols = [("MAXSMC", 4), ("SATPSI", 6), ("SATDK", 7), ("BB", 1), ("QTZ", 10), ("REFSMC", 5), ("WLTSMC", 9),
        ("DRYSMC", 2), ("SATDW", 8), ("F11", 3)]
with open(out + "/noah271_soilparms.txt", "w") as f:
    f.write("soil depths (m) and parameters, derived from Noah 3.3 SOILPARM.TBL STAS\n")
    f.write("0.1 0.3 0.6 1.0\n")
    for n, c in cols:
        f.write(f"{n}\n")
        f.write(" ".join(f"{r[c]:.8g}" for r in srows) + "\n")
    f.write("CZIL\n0.1\n")
print("wrote", out)
