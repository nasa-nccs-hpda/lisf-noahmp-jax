"""First divergence of SWE_inst between two history trees, per 6-hourly file: counts of cells with |diff| > 1e-3, 1e-2, 0.1 mm and the max; saves the >1e-3 masks as .npy.
usage: python3 swe_divergence.py DIR_A DIR_B N_FILES OUT.npy      (DIR = history root containing SURFACEMODEL/; used for Fortran vs JAX and JAX vs JAX+1ulp)"""
import netCDF4 as nc, numpy as np, glob, os, sys
A,B=sys.argv[1]+"/SURFACEMODEL/",sys.argv[2]+"/SURFACEMODEL/"; N=int(sys.argv[3])
fs=sorted(os.path.basename(f) for f in glob.glob(A+"*/LIS_HIST*.nc"))[:N]
def rd(p): return np.ma.filled(nc.Dataset(p)["SWE_inst"][:].astype(float),np.nan).squeeze()
m=[]
for f in fs:
    ym=f[9:15]; d=np.abs(rd(A+ym+"/"+f)-rd(B+ym+"/"+f)); d[~np.isfinite(d)]=0
    m.append(d>1e-3)
    print(f[9:21],"d>1e-3 %d d>0.01 %d d>0.1 %d max %.3g"%((d>1e-3).sum(),(d>1e-2).sum(),(d>.1).sum(),d.max()))
np.save(sys.argv[4],np.array(m))
