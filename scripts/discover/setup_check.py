import sys, traceback
from lisf_noahmp_jax.driver.params_mp50 import SetupMP50
cfg, base = sys.argv[1], sys.argv[2]
try:
    S = SetupMP50(cfg, base)
    print("SETUP OK ntile", S.ntile, "lat", float(S.lat.min()), float(S.lat.max()), "lon", float(S.lon.min()), float(S.lon.max()))
except Exception:
    tb = traceback.format_exc().splitlines()
    print("SETUP FAILED"); print("\n".join(tb[-8:]))
