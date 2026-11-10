"""Config for the Intel cross-check: R's config, NetCDF NLDAS-2.0 reader on converted files, 5 days, given processor layout.
usage: python mkcfg_intel.py SRC DST FORCING_DIR OUTDIR NX NY"""
import sys
src, dst, fdir, out, nx, ny = sys.argv[1:7]
keys = {"Met forcing sources:": '"NLDAS2 netcdf"', "Ending year:": "2008", "Ending month:": "10", "Ending day:": "6", "Output directory:": '"%s"' % out,
        "Number of processors along x:": nx, "Number of processors along y:": ny, "Diagnostic output file:": '"%s/lislog"' % out}
done, lines = set(), []
for l in open(src):
    if l.startswith("NLDAS2 forcing directory:"):
        l = "NLDAS-2.0 FORA forcing directory:".ljust(40) + fdir + "\n"; done.add("fora")
    for k, v in keys.items():
        if l.startswith(k):
            l = k.ljust(40) + v + "\n"; done.add(k)
    lines.append(l)
if "fora" not in done:
    lines.append("NLDAS-2.0 FORA forcing directory:".ljust(40) + fdir + "\n")
open(dst, "w").writelines(lines)
print("wrote", dst, sorted(k for k in done if k != "fora"))
