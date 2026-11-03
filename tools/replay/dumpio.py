"""Reader for jaxdump_noah39.bin written by the instrumented LIS (lis_patches/jaxdump_mod.F90).
Big-endian stream: int32 (NHDR, NPRE, NPOST), then records of NHDR int32, NPRE float32, NPOST float32."""
import numpy as np

PRE_NAMES = (
    "ffrozp dt zlvl lwdn soldn solnet sfcprs prcp sfctmp q2 sfcspd th2 q2sat dqsdt2 vegt soiltyp slope "
    "shdfac shdmin shdmax alb snoalb tbot z0brd z0 emiss embrd cmc t1 stc1 stc2 stc3 stc4 smc1 smc2 smc3 smc4 "
    "sh2o1 sh2o2 sh2o3 sh2o4 snowh sneqv albedo ch cm ztopv zbotv snotime1 ribb lai rsmin ch0 cm0 z0_old t1v "
    "th2v sfcspd_b smcwlt smcdry smcref smcmax nroot hs rgl snup cmcmax rsmax topt bexp dksat dwsat f1 quartz "
    "psisat czil sbeta fxexp csoil salp kdt cfactr zbot refkdt frzx sndens lvcoef opt_thcnd aoasis isurban nsoil "
    "z0brd_old q1_pre iz0tlnd sfcdifopt uaflag sca_pre"
).split()
POST_NAMES = (
    "cmc t1 stc1 stc2 stc3 stc4 smc1 smc2 smc3 smc4 sh2o1 sh2o2 sh2o3 sh2o4 snowh sneqv albedo ch cm sncovr z0 "
    "snotime1 q1 emissi xlai shdfac eta sheat ssoil etp edir ec ett esnow runoff1 runoff2 runoff3 snomlt drip "
    "dew beta fdown rc pc soilm soilw tsoil flx1 flx2 flx3 ustar evp eta_kin rsmin smcmax smcwlt"
).split()


def load(path, max_records=None):
    with open(path, "rb") as f:
        nh, npre, npost = np.fromfile(f, ">i4", 3)
        dt = np.dtype([("hdr", ">i4", int(nh)), ("pre", ">f4", int(npre)), ("post", ">f4", int(npost))])
        rec = np.fromfile(f, dt, count=-1 if max_records is None else max_records)
    pre = {n: rec["pre"][:, i].astype(np.float32) for i, n in enumerate(PRE_NAMES)}
    post = {n: rec["post"][:, i].astype(np.float32) for i, n in enumerate(POST_NAMES)}
    hdr = rec["hdr"].astype(np.int32)  # yr mo da hr mn ss tile col row
    return hdr, pre, post
