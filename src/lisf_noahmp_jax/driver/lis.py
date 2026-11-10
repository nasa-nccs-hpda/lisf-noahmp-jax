"""LIS-style front end of the Python/JAX port: reads a LIS configuration file (lis.config), picks the land surface model, and runs the
period given by the 'Starting ...' / 'Ending ...' keys, writing LIS-style history files under 'Output directory'.
usage: python -m lisf_noahmp_jax.driver.lis [-C CASE_DIR] lis.config
Supported 'Land surface model' values: Noah.3.9, Noah.3.6, Noah.3.3, Noah.3.2, Noah.2.7.1, Noah-MP.4.0.1, NoahMP.3.6, CLSM F2.5, Mosaic (see STATUS_20261022.md for the
validated configurations; other keys of the configuration are read only by the model that needs them). Forcing: NLDAS2 netcdf, MERRA2."""
import sys, os, datetime as dt, argparse
from lisf_noahmp_jax.driver.config import Config


# LIS restart-file names (LIS_RST_<tag>_YYYYMMDDHHMM.d01.nc) of the models whose runner can read and write restarts
RST_TAG = {"Noah-MP.4.0.1": "NOAHMP401", "Noah-MP.5.0": "NoahMP50", "NoahMP.3.6": "NOAHMP36", "Noah.3.9": "NOAH39", "Noah.3.6": "NOAH36"}


def period(c):
    g = lambda k: c.i(k)
    t0 = dt.datetime(g("Starting year"), g("Starting month"), g("Starting day"), g("Starting hour"), g("Starting minute"), g("Starting second"))
    t1 = dt.datetime(g("Ending year"), g("Ending month"), g("Ending day"), g("Ending hour"), g("Ending minute"), g("Ending second"))
    return t0, t1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("-C", "--case-dir", default=None, help="directory the relative paths of the configuration refer to (default: its directory)")
    ap.add_argument("--hours", type=int, default=6, help="history output interval in hours (LIS 'Surface model output interval')")
    a = ap.parse_args(argv)
    cfg = os.path.abspath(a.config)
    base = os.path.abspath(a.case_dir or os.path.dirname(cfg))
    c = Config(cfg)
    P = lambda p: p if p.startswith("/") else os.path.normpath(os.path.join(base, p))
    start, end = period(c)
    out = P(c.s("Output directory").strip('"'))
    model = c.s("Land surface model").strip('"')
    src = c.s("Met forcing sources").strip('"')
    if src.startswith("NLDAS2"):
        from lisf_noahmp_jax.driver.forcing import forcing_source
        fdir = P(forcing_source(c.s, src)[1])
    elif src == "MERRA2":
        fdir = P(c.s("MERRA2 forcing directory"))
    else:
        raise SystemExit(f"forcing source {src!r} is not ported")
    tm = {}
    if model.startswith("Noah-MP") or model == "NoahMP.3.6":
        from lisf_noahmp_jax.driver import run_mp as R
    elif model in ("Noah.3.9", "Noah.3.6", "Noah.3.3", "Noah.3.2", "Noah.2.7.1"):
        from lisf_noahmp_jax.driver import run as R
    elif model.startswith("CLSM"):
        from lisf_noahmp_jax.driver import run_clsm as R
    elif model == "Mosaic":
        from lisf_noahmp_jax.driver import run_mos as R
    else:
        raise SystemExit(f"land surface model {model!r} is not ported")
    kw = {}
    tag = RST_TAG.get(model)
    if tag:                                                   # restart handling (config-driven, as in LIS): models whose runner supports it
        rin = None
        if c.d.get("Start mode", "coldstart").strip().strip('"') == "restart":
            for key in (f"{model} restart file", model.replace("NoahMP", "Noah-MP") + " restart file"):
                if key in c.d and c.d[key].strip().strip('"').lower() != "none":
                    rin = P(c.d[key].strip().strip('"')); break
            if rin is None:
                raise SystemExit("Start mode is restart but no '<model> restart file' is set")
        rout = None
        if c.d.get("Output model restart files", "0").split()[0] == "1":
            rout = f"{out}/SURFACEMODEL/{end:%Y%m}/LIS_RST_{tag}_{end:%Y%m%d%H%M}.d01.nc"
            os.makedirs(os.path.dirname(rout), exist_ok=True)
        kw = dict(restart_in=rin, restart_out=rout)
    elif c.d.get("Start mode", "coldstart").strip().strip('"') == "restart":
        raise SystemExit(f"restart start is not ported for {model!r}")
    R.run(cfg, base, fdir, out, start, end, out_interval_h=a.hours, timing=tm, **kw)
    print("TIMING", {k: (round(v, 2) if not isinstance(v, list) else [round(x, 2) for x in v]) for k, v in tm.items()})


if __name__ == "__main__":
    main()
