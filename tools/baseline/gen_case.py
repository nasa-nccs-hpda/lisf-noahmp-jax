"""Generate a synthetic but physically plausible one-day Noah 3.9 test case.

Parameters come from the real LISF tables (lis/configs/557WW-7.3-FOC/noah39_parms,
USGS vegetation, STAS soils, GENPARM). Forcing and initial states are SYNTHETIC
(seeded random), not taken from any observed or LIS-produced dataset.
Usage: python gen_case.py N [seed] [member] -> writes case_N.bin (member 0) or case_N_mK.bin.
member K>0 perturbs t1 and the soil temperature/moisture by +-1 float32 ulp (noise-floor test).
"""
import sys
import numpy as np
import spec

HERE = __file__.rsplit("/", 1)[0]
PARMS = HERE + "/../../lis/configs/557WW-7.3-FOC/noah39_parms"


def read_soil():
    rows = {}
    lines = open(PARMS + "/SOILPARM.TBL").read().splitlines()
    n = int(lines[2].split(",")[0])
    for ln in lines[3:3 + n]:
        p = [x.strip() for x in ln.split(",")]
        rows[int(p[0])] = dict(zip(
            ["bb", "drysmc", "f11", "maxsmc", "refsmc", "satpsi", "satdk",
             "satdw", "wltsmc", "qtz"], map(float, p[1:11])))
    return rows


def read_veg():
    lines = open(PARMS + "/VEGPARM.TBL").read().splitlines()
    n = int(lines[2].split(",")[0])
    rows = {}
    keys = ["shdfac", "nroot", "rs", "rgl", "hs", "snup", "maxalb", "laimin",
            "laimax", "emissmin", "emissmax", "albmin", "albmax", "z0min",
            "z0max", "ztopv", "zbotv"]
    for ln in lines[3:3 + n]:
        p = [x.strip() for x in ln.split(",")]
        rows[int(p[0])] = dict(zip(keys, map(float, p[1:18])))
    return rows


def calhum(t, p):
    a2, a3, a4, elwv, e0, rv, eps = 17.67, 273.15, 29.65, 2.501e6, 611.0, 461.0, 0.622
    es = e0 * np.exp(elwv / rv * (1.0 / a3 - 1.0 / t))
    q2sat = eps * es / (p - (1 - eps) * es)
    return q2sat


def main():
    N = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261021
    member = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = np.random.default_rng(seed)
    soil, veg = read_soil(), read_veg()
    NS, NT = spec.NSOIL, 48
    sldpth = np.array([0.1, 0.3, 0.6, 1.0])
    gen = dict(slope=[0.1, 0.6, 1.0], sbeta=-2.0, fxexp=2.0, csoil=2.0e6, salp=2.6,
               refdk=2.0e-6, refkdt=3.0, frzk=0.15, zbot=-8.0, czil=0.1, lvcoef=0.5,
               topt=298.0, cmcmax=0.5e-3, rsmax=5000.0)
    vegs = np.array([v for v in range(1, 24) if v != 16] + [24])
    soils = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12])
    vt = rng.choice(vegs, N)
    st = rng.choice(soils, N)
    slt = rng.integers(0, 3, N)
    # climate zone per column: mean temperature 258..302 K so some columns snow/freeze
    tmean = rng.uniform(258.0, 302.0, N)
    amp = rng.uniform(3.0, 10.0, N)
    I = {k: np.zeros(N, np.float32) for k in spec.STATIC + spec.INIT}
    ints = {"vegt": vt.astype(np.int32), "soiltyp": st.astype(np.int32)}
    nroot = np.zeros(N, np.int32)
    for i in range(N):
        s, v = soil[st[i]], veg[vt[i]]
        nroot[i] = int(v["nroot"])
        I["slope"][i] = gen["slope"][slt[i]]
        I["shdfac"][i] = v["shdfac"]
        I["shdmin"][i] = 0.0
        I["shdmax"][i] = min(1.0, v["shdfac"] * 1.2)
        I["z0brd"][i] = 0.5 * (v["z0min"] + v["z0max"])
        I["embrd"][i] = 0.5 * (v["emissmin"] + v["emissmax"])
        I["alb"][i] = 0.5 * (v["albmin"] + v["albmax"])
        I["snoalb"][i] = v["maxalb"] / 100.0
        I["tbot"][i] = tmean[i] - 1.0
        for k in ("czil", "sbeta", "fxexp", "csoil", "salp", "refkdt", "zbot", "lvcoef",
                  "topt", "cmcmax", "rsmax"):
            I[k][i] = gen[k]
        I["cfactr"][i] = 0.5
        I["ztopv"][i], I["zbotv"][i] = v["ztopv"], v["zbotv"]
        I["smcwlt"][i], I["smcdry"][i] = s["wltsmc"], s["drysmc"]
        I["smcref"][i], I["smcmax"][i] = s["refsmc"], s["maxsmc"]
        I["hs"][i], I["rgl"][i], I["snup"][i] = v["hs"], v["rgl"], v["snup"]
        I["bexp"][i], I["dksat"][i], I["dwsat"][i] = s["bb"], s["satdk"], s["satdw"]
        I["f1"][i], I["quartz"][i], I["psisat"][i] = s["f11"], s["qtz"], s["satpsi"]
        I["rsmin"][i] = v["rs"]
        I["kdt"][i] = gen["refkdt"] * s["satdk"] / gen["refdk"]
        I["frzx"][i] = gen["frzk"] * (s["maxsmc"] / s["refsmc"]) * (0.412 / 0.468)
        I["xlai"][i] = 0.5 * (v["laimin"] + v["laimax"])
    ints["nroot"] = nroot
    # initial state
    cold = tmean < 273.15
    I["t1"][:] = tmean + rng.normal(0, 1.0, N)
    I["cmc"][:] = rng.uniform(0.0, 1.0e-4, N)
    sneqv = np.where(cold & (rng.random(N) < 0.7), rng.uniform(0.005, 0.08, N), 0.0)
    I["sneqv"][:] = sneqv
    I["snowh"][:] = sneqv * rng.uniform(3.0, 8.0, N)   # density 125..333 kg/m3
    I["albedo"][:] = I["alb"]
    I["ch"][:] = 0.01
    I["cm"][:] = 0.01
    I["sncovr"][:] = 0.0
    I["z0"][:] = I["z0brd"]
    I["snotime1"][:] = 0.0
    I["emissi"][:] = I["embrd"]
    q0 = rng.uniform(0.002, 0.012, N)
    I["q1"][:] = q0
    L = {k: np.zeros((NS, N), np.float32) for k in spec.INIT_LAYER}
    for k in range(NS):
        frac = rng.uniform(0.3, 0.9, N)
        smc = I["smcwlt"] + frac * (I["smcmax"] - I["smcwlt"])
        L["smc"][k] = smc
        L["stc"][k] = tmean + rng.normal(0, 1.0, N) + (k * 0.3)
        L["sh2o"][k] = np.where(L["stc"][k] < 273.15, 0.7 * smc, smc)
    # forcing, one day at 30 min
    t = (np.arange(NT) + 0.5) / NT
    phase = np.sin(2 * np.pi * (t - 0.25))[:, None]
    sfctmp = tmean[None] + amp[None] * phase
    sfcprs = 101325.0 * np.exp(-rng.uniform(0, 3000, N)[None] / 8000.0) * np.ones((NT, 1))
    day = np.clip(np.sin(2 * np.pi * (t - 0.25)), 0, None)[:, None]
    soldn = day * rng.uniform(300, 900, N)[None]
    lwdn = 5.67e-8 * (sfctmp - rng.uniform(8, 25, N)[None]) ** 4
    q2 = calhum(sfctmp, sfcprs) * rng.uniform(0.4, 0.95, N)[None]
    sfcspd = rng.uniform(0.5, 7.0, N)[None] * (1 + 0.3 * rng.standard_normal((NT, N)))
    sfcspd = np.clip(sfcspd, 0.0, None)
    wet = rng.random(N) < 0.5
    ev_start = rng.integers(0, NT - 6, N)
    ev_len = rng.integers(2, 6, N)
    prcp = np.zeros((NT, N))
    for i in np.where(wet)[0]:
        prcp[ev_start[i]:ev_start[i] + ev_len[i], i] = rng.uniform(5e-5, 6e-4)
    F = dict(sfctmp=sfctmp, q2=q2, soldn=soldn, lwdn=lwdn, sfcspd=sfcspd,
             sfcprs=sfcprs, prcp=prcp)
    if member:
        prng = np.random.default_rng(1000 + member)
        sign = lambda shape: np.where(prng.random(shape) < 0.5, -np.inf, np.inf).astype(np.float32)
        I["t1"] = np.nextafter(I["t1"], sign(N))
        for k in spec.INIT_LAYER:
            L[k] = np.nextafter(L[k], sign((NS, N)))
    name = f"case_{N}.bin" if not member else f"case_{N}_m{member}.bin"
    with open(name, "wb") as f:
        np.array([N, NT, NS], np.int32).tofile(f)
        for k in spec.STATIC_INT:
            ints[k].tofile(f)
        for k in spec.STATIC:
            I[k].astype(np.float32).tofile(f)
        for k in spec.INIT:
            I[k].astype(np.float32).tofile(f)
        for k in spec.INIT_LAYER:
            L[k].astype(np.float32).tofile(f)
        for k in spec.FORC:
            F[k].astype(np.float32).tofile(f)
    print("wrote", name, "N", N, "steps", NT)


if __name__ == "__main__":
    main()
