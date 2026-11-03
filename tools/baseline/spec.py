"""Shared binary layout between gen_case.py, driver.F90 and the JAX port.

All floats are float32 (the Fortran default real in LIS builds), ints are int32.
If you change a list here, change the read/write order in driver.F90 as well.
"""
STATIC_INT = ["vegt", "soiltyp", "nroot"]
STATIC = [
    "slope", "shdfac", "shdmin", "shdmax", "z0brd", "embrd", "alb", "snoalb",
    "tbot", "czil", "sbeta", "fxexp", "csoil", "salp", "kdt", "cfactr", "zbot",
    "refkdt", "frzx", "lvcoef", "ztopv", "zbotv", "smcwlt", "smcdry", "smcref",
    "smcmax", "hs", "rgl", "snup", "cmcmax", "rsmax", "topt", "bexp", "dksat",
    "dwsat", "f1", "quartz", "psisat", "rsmin",
]
INIT = [
    "cmc", "t1", "snowh", "sneqv", "albedo", "ch", "cm", "sncovr", "z0",
    "snotime1", "q1", "emissi", "xlai",
]
INIT_LAYER = ["stc", "smc", "sh2o"]
FORC = ["sfctmp", "q2", "soldn", "lwdn", "sfcspd", "sfcprs", "prcp"]
NSOIL = 4
OUT = (
    ["cmc", "t1"]
    + [f"stc{k}" for k in range(1, NSOIL + 1)]
    + [f"smc{k}" for k in range(1, NSOIL + 1)]
    + [f"sh2o{k}" for k in range(1, NSOIL + 1)]
    + ["snowh", "sneqv", "albedo", "ch", "cm", "sncovr", "z0", "snotime1", "q1",
       "emissi", "xlai", "shdfac", "eta", "sheat", "ssoil", "etp", "edir", "ec",
       "ett", "esnow", "runoff1", "runoff2", "runoff3", "snomlt", "drip", "dew",
       "beta", "fdown", "rc", "pc", "soilm", "soilw", "tsoil"]
)
DT = 1800.0
ZLVL = 10.0
