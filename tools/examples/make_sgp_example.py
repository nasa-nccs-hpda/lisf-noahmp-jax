"""Build examples/sgp_5d from the cluster data (not needed to RUN the example; only to regenerate it).
Case: Noah-MP 5.0, Southern Great Plains 0.25 deg domain (28 x 22 cells), 2017-01-01 to 01-06, 15-min step, 6-hourly output, cold start.
Contents: lis.config (relative paths), LDT parameter file, NoahmpTable.TBL, output list, NLDAS-2 hourly forcing cut to the domain (full NLDAS grid, fill value
elsewhere, zlib), and the real Fortran LIS 7.8 history files for the same run (reference/). usage: python3 make_sgp_example.py"""
import glob, os, re, shutil
import numpy as np
import netCDF4 as nc

RUN = os.environ.get("LISF_RUN", "/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run")
FORA = "/css/nldas/NLDAS2/FORA"
TBL = "/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_build/lis/surfacemodels/land/noahmp.5.0/phys/noahmp/parameters/NoahmpTable.TBL"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples", "sgp_5d")
VARS = ("Tair", "Qair", "PSurf", "Wind_E", "Wind_N", "LWdown", "SWdown", "Rainf")
END_DAY = 6

os.makedirs(OUT, exist_ok=True)
for d in ("ldt", "forcing", "reference"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

# configuration: the one the Fortran year run used, with relative paths and a 5-day span
src = open(f"{RUN}/mp5_sgp_year/lis.config.fort").read()
src = re.sub(r"^Ending year:.*$", "Ending year: 2017", src, flags=re.M)
src = re.sub(r"^Ending month:.*$", "Ending month: 1", src, flags=re.M)
src = re.sub(r"^Ending day:.*$", f"Ending day: {END_DAY}", src, flags=re.M)
src = re.sub(r"^Output directory:.*$", "Output directory: ./out", src, flags=re.M)
src = re.sub(r"^Diagnostic output file:.*$", "Diagnostic output file: ./out/lislog", src, flags=re.M)
src = re.sub(r"^(LIS domain and parameter data file:\s+).*$", r"\1./ldt/lis_input.sgp.noahmp5.nc", src, flags=re.M)
src = re.sub(r"^(NLDAS-2.0 FORA forcing directory:\s+).*$", r"\1./forcing", src, flags=re.M)
src = re.sub(r"^(Noah-MP.5.0 parameter table:\s+).*$", r"\1./NoahmpTable.TBL", src, flags=re.M)
src = re.sub(r"^(Model output attributes file:\s*).*$", r"\1./MODEL_OUTPUT_LIST.TBL.noahmp5", src, flags=re.M)
src = re.sub(r"^(Forcing variables list file:\s+).*$", r"\1./forcing_variables.txt", src, flags=re.M)
open(os.path.join(OUT, "lis.config"), "w").write(src)
shutil.copy(f"{RUN}/mp5_sgp_year/ldt_mp5/out/lis_input.sgp.noahmp5.nc", f"{OUT}/ldt/")
shutil.copy(TBL, f"{OUT}/NoahmpTable.TBL")
shutil.copy(f"{RUN}/INPUT/MODEL_OUTPUT_LIST.TBL.noahmp5", OUT)
shutil.copy(f"{RUN}/INPUT/forcing_variables.txt", OUT)

# forcing: domain box + 0.5 deg, NLDAS grid kept
ld = nc.Dataset(f"{OUT}/ldt/lis_input.sgp.noahmp5.nc")
la, lo = np.asarray(ld["lat"][:]), np.asarray(ld["lon"][:])
box = (la.min() - 0.5, la.max() + 0.5, lo.min() - 0.5, lo.max() + 0.5)
n = 0
for h in range(24 * (END_DAY - 1) + 2):
    day, hr = divmod(h, 24)
    f = f"{FORA}/2017/{day + 1:03d}/NLDAS_FORA0125_H.A201701{day + 1:02d}.{hr:02d}00.020.nc"
    s = nc.Dataset(f)
    glat, glon = np.asarray(s["lat"][:]), np.asarray(s["lon"][:])
    jm = (glat >= box[0]) & (glat <= box[1]); im = (glon >= box[2]) & (glon <= box[3])
    dd = os.path.join(OUT, "forcing", "2017", f"{day + 1:03d}"); os.makedirs(dd, exist_ok=True)
    o = nc.Dataset(os.path.join(dd, os.path.basename(f)), "w", format="NETCDF4")
    for k in ("lat", "lon", "time"):
        o.createDimension(k, len(s.dimensions[k]) if k != "time" else 1)
    for k in ("lat", "lon"):
        v = o.createVariable(k, "f4", (k,)); v[:] = s[k][:]
    v = o.createVariable("time", "f8", ("time",)); v[:] = s["time"][:]; v.units = s["time"].units
    for k in VARS:
        a = np.full((1, len(glat), len(glon)), -9999.0, np.float32)
        a[0][np.ix_(jm, im)] = np.ma.filled(s[k][0], -9999.0)[np.ix_(jm, im)]
        v = o.createVariable(k, "f4", ("time", "lat", "lon"), zlib=True, complevel=6, fill_value=np.float32(-9999.0)); v[:] = a
    o.close(); s.close(); n += 1
print("forcing files", n)

# reference: Fortran history files of the same run (first 5 days of the Fortran year run), repacked with zlib
for f in sorted(glob.glob(f"{RUN}/mp5_sgp_year/fort/SURFACEMODEL/201701/LIS_HIST_2017010[1-5]*.nc")) + [f"{RUN}/mp5_sgp_year/fort/SURFACEMODEL/201701/LIS_HIST_201701060000.d01.nc"]:
    if not os.path.exists(f):
        continue
    d = os.path.join(OUT, "reference", "SURFACEMODEL", "201701"); os.makedirs(d, exist_ok=True)
    a = nc.Dataset(f); b = nc.Dataset(os.path.join(d, os.path.basename(f)), "w", format="NETCDF4")
    b.setncatts({k: a.getncattr(k) for k in a.ncattrs()})
    for k, v in a.dimensions.items():
        b.createDimension(k, len(v))
    for k, v in a.variables.items():
        fv = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
        x = b.createVariable(k, v.dtype, v.dimensions, zlib=True, complevel=6, fill_value=fv)
        x.setncatts({kk: v.getncattr(kk) for kk in v.ncattrs() if kk != "_FillValue"}); x[:] = v[:]
    a.close(); b.close()
print("reference files", len(glob.glob(f"{OUT}/reference/SURFACEMODEL/*/*.nc")))
