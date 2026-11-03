import sys, datetime as dt; sys.path.insert(0,".")
from lisjax import run
base,out,hours=sys.argv[1],sys.argv[2],int(sys.argv[3])
s=dt.datetime.strptime(sys.argv[4],"%Y%m%d") if len(sys.argv)>4 else dt.datetime(2017,2,1); tm={}
run.run(f"{base}/lis.config",base,f"{base}/INPUT/NLDAS2.FORCING",out,s,s+dt.timedelta(hours=hours),timing=tm)
print("TIMING",{k:(round(v,1) if not isinstance(v,list) else [round(x,1) for x in v]) for k,v in tm.items()})
