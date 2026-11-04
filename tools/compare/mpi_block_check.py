import netCDF4 as nc, numpy as np, glob, sys, os
R="conus_mp5_60d/lisout"
for tag in sys.argv[1:]:
    M="conus_mp5_mpi/lisout_"+tag
    fs=sorted(glob.glob(M+"/SURFACEMODEL/*/LIS_HIST_*.nc"))
    if not fs: print(tag,"no output"); continue
    tot=0; box=[999,-1,999,-1]; nf=0; mx=0
    for g in fs:
        f=g.replace(M,R)
        if not os.path.exists(f): continue
        A=nc.Dataset(f); B=nc.Dataset(g); nf+=1
        x=np.ma.filled(A["Tair_f_inst"][:].astype(float),np.nan); y=np.ma.filled(B["Tair_f_inst"][:].astype(float),np.nan)
        d=np.abs(x-y); bad=(np.isfinite(d)&(d>0.01))|(np.isnan(x)!=np.isnan(y))
        tot+=int(bad.sum())
        if bad.any():
            r,c=np.where(bad); box=[min(box[0],r.min()),max(box[1],r.max()),min(box[2],c.min()),max(box[3],c.max())]
    print(tag,"files",nf,"Tair_f bad cells (sum over files)",tot,"bbox rows/cols",box if tot else None,flush=True)
