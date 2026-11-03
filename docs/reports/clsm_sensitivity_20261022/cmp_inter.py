import sys, numpy as np
sys.path.insert(0,"."); sys.path.insert(0,"lis")
import jax, jax.numpy as jnp
import dumpio_clsm as D, clsm_jax as C
from replay_clsm import build_x
d=sys.argv[1]; step=int(sys.argv[2])
names=open(f"{d}/jaxdump_clsm_inter.layout").read().split() if False else None
import re
lay=[l.split("|") for l in open(f"{d}/jaxdump_clsm.layout").read().split("\n") if l]
dn=[x for x in D.dummy_names() if x not in ("nch","bug")]
inames=open(d+"/jaxdump_clsm_inter.layout").read().split()
raw=open(f"{d}/jaxdump_clsm.bin","rb").read(); pos=0
def rd(dt,n):
    global pos
    a=np.frombuffer(raw,dtype=dt,count=n,offset=pos); pos+=a.nbytes; return a
ns=int(rd(">i4",1)[0]); idx=rd(">i4",ns)
R={0:{},1:{},2:{}}
while pos<len(raw):
    tag,st=[int(v) for v in rd(">i4",2)]
    dd={}
    if tag==2:
        for n in inames: dd[n.lower()]=rd(">f4",ns).astype(np.float32)
    else:
        for (e,k),n in zip(lay,dn):
            if k=="s": dd[n]=float(rd(">f4",1)[0])
            elif k=="v": dd[n]=rd(">f4",ns).astype(np.float32)
            elif k=="k": kk=6 if n=="ghtcnt" else 3; dd[n]=rd(">f4",kk*ns).reshape(ns,kk).T.astype(np.float32)
            elif k=="t": dd[n]=rd(">f4",ns*3).reshape(3,ns).astype(np.float32)
    R[tag][st]=dd
x=build_x(R[0][step])
out=C.clsmf25(jnp.float32(R[0][step]["dtstep"]),jnp.float32(R[0][step]["sfrac"]),x)
rows=[]
for k,v in out["inter"].items():
    a=np.asarray(v,dtype=np.float64); b=R[2][step][k].astype(np.float64)
    e=np.abs(a-b); rows.append((np.nanmax(e),k,np.nanmax(np.abs(b)),np.percentile(e,99),int(np.argmax(e))))
for r in sorted(rows,reverse=True): print("%-8s max %.4g  p99 %.3g  refmax %.4g  worst %d"%(r[1],r[0],r[3],r[2],r[4]))
print("---- post")
from replay_clsm import compare
res=compare(out,R[1][step])
a,b=res["tc2"]; e=np.abs(a-b); w=np.argsort(-e)[:12]
wes=R[0][step]["wesnn"].sum(0)
for i in w: print(i,"tc2 err %.3g"%e[i],"swe_pre %.4g"%wes[i],"wesn_post %.4g"%R[1][step]["wesnn"].sum(0)[i],"asnow0 ref %.3g jax %.3g"%(R[1][step]["asnow0"][i],np.asarray(out["asnow0"])[i]),"tpsn1 ref %.2f jax %.2f"%(R[1][step]["tpsn1"][i],np.asarray(out["tpsn1"])[i]), "tc2 ref %.2f jax %.2f pre %.2f"%(b[i],a[i],R[0][step]["tc2"][i]))
m=wes>0
for k in ["tc1","tc2","tc4","shflux","esoi","catdef"]:
    a,b=res[k]; e=np.abs(a-b); print(k,"snow cells p99 %.3g max %.3g | nosnow p99 %.3g max %.3g"%(np.percentile(e[m],99),e[m].max(),np.percentile(e[~m],99),e[~m].max()))
