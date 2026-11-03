"""Do the worst Fortran-vs-JAX tiles coincide with the tiles that respond most to a tiny JAX perturbation?
usage: python3 -m lisf_noahmp_jax.driver.compare_tiles FORTRAN_DIR JAX_DIR JAX_ULP_DIR [N=200]
Per variable and tile, the maximum over all history files and layers of |Fortran-JAX| (A) and |JAX-JAXulp| (B); then the overlap of the N
worst tiles of A and B (random overlap would be N*N/ntiles), the Spearman rank correlation of A and B over the tiles where either is nonzero,
and the size of A and B on the other set's worst tiles."""
import sys, glob, os
import numpy as np
import netCDF4 as nc

ref, jx, ul = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 200
VARS = ["Qle_tavg", "Qh_tavg", "Qg_tavg", "AvgSurfT_tavg", "SWE_inst", "SoilMoist_inst", "SoilTemp_inst"]
A, B = {}, {}
files = sorted(glob.glob(f"{ref}/SURFACEMODEL/*/LIS_HIST_*.nc"))
n = 0
for f in files:
    g, h = f.replace(ref, jx), f.replace(ref, ul)
    if not (os.path.exists(g) and os.path.exists(h)):
        continue
    n += 1
    F, J, U = nc.Dataset(f), nc.Dataset(g), nc.Dataset(h)
    for v in VARS:
        if v not in F.variables or v not in J.variables or v not in U.variables:
            continue
        x, y, z = (np.ma.filled(d[v][:].astype(float), np.nan) for d in (F, J, U))
        a = np.abs(x - y); b = np.abs(y - z)
        if a.ndim > 2:
            a = np.nanmax(a.reshape((-1,) + a.shape[-2:]), axis=0); b = np.nanmax(b.reshape((-1,) + b.shape[-2:]), axis=0)
        a = np.nan_to_num(a, nan=0.0); b = np.nan_to_num(b, nan=0.0)
        A[v] = np.maximum(A.get(v, 0), a); B[v] = np.maximum(B.get(v, 0), b)
    for d in (F, J, U):
        d.close()
print(f"{n} history files; N = {N} worst tiles per variable")
print(f"{'variable':10s} {'tiles':>7s} {'overlap':>8s} {'random':>7s} {'spearman':>8s} {'A@Aworst':>9s} {'B@Aworst':>9s} {'B@Bworst':>9s} {'A@Bworst':>9s}  (A=|Fortran-JAX|, B=|JAX-JAX+1ulp|, max over time)")
for v in VARS:
    if v not in A:
        continue
    a, b = A[v].ravel(), B[v].ravel()
    act = (a > 0) | (b > 0); nt = int(act.sum()); a, b = a[act], b[act]
    k = min(N, nt); ia = np.argsort(-a)[:k]; ib = np.argsort(-b)[:k]
    ov = len(set(ia) & set(ib))
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    rho = np.corrcoef(ra, rb)[0, 1]
    print(f"{v:10s} {nt:7d} {ov:8d} {k*k/nt:7.1f} {rho:8.3f} {np.median(a[ia]):9.3g} {np.median(b[ia]):9.3g} {np.median(b[ib]):9.3g} {np.median(a[ib]):9.3g}")
