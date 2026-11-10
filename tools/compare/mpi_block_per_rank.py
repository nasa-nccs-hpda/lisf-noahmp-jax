import netCDF4 as nc, numpy as np, glob, sys, os
tag,X,Y=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
R="conus_mp5_60d/lisout"; M="conus_mp5_mpi/lisout_"+tag
fs=sorted(glob.glob(M+"/SURFACEMODEL/*/LIS_HIST_*.nc"))[:8]
cnt=np.zeros((len(fs),X*Y),int)
for i,g in enumerate(fs):
    f=g.replace(M,R)
    x=np.ma.filled(nc.Dataset(f)["Tair_f_inst"][:].astype(float),np.nan); y=np.ma.filled(nc.Dataset(g)["Tair_f_inst"][:].astype(float),np.nan)
    d=np.abs(x-y); bad=(np.isfinite(d)&(d>0.01))|(np.isnan(x)!=np.isnan(y))
    nr,nc_=bad.shape; bw=-(-nc_//X); bh=-(-nr//Y)
    for p in range(X*Y):
        px,py=p%X,p//X
        cnt[i,p]=bad[py*bh:(py+1)*bh,px*bw:(px+1)*bw].sum()
print(tag,"shape",bad.shape)
for i in range(len(fs)): print(" file",i,{p:int(c) for p,c in enumerate(cnt[i]) if c})
