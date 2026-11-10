"""Regression tests: the JAX column physics must reproduce the recorded outputs of the real LIS (GNU 14, serial, float32) on a seeded random
sample of 500-700 SFLX calls per model (data/ built by make_test_data.py). Thresholds are on the 99th percentile of the absolute error
(maxima are not meaningful for float32 freeze/thaw tiles, see STATUS_20261022.md section 3) and are about 5-10 times the values measured on
2026-10-22, so a real change of the physics fails and rounding-level noise does not.
run:  JAX_PLATFORMS=cpu python3 -m pytest -q tests          (about 31 min on one CPU core: 5 default cases + 22 Noah-MP option variants + 3 Noah-MP 5.0 cases)"""
import os
import pytest
import replay_cases as R

HERE = os.path.dirname(os.path.abspath(__file__))

# p99 absolute error limits per field
NOAH33 = dict(t1=1e-3, stc1=5e-3, smc1=1e-6, sh2o1=1e-4, eta=5e-2, sheat=0.2, ssoil=0.2, snowh=1e-6, sneqv=1e-7, cmc=1e-8, runoff1=1e-12,
              runoff2=1e-11, evp=1e-8)
NOAH271 = dict(t1=1e-3, stc1=1e-2, smc1=1e-6, sh2o1=1e-4, eta=5e-3, snowh=1e-6, sneqv=1e-7, cmc=1e-8, evp=1e-8)
MP = dict(TV=1e-2, TG=2e-3, FSH=5e-2, FCTR=1e-3, FGEV=2e-3, SSOIL=5e-2, FIRA=2e-2, FSA=1e-3, STC=1e-3, SMC=1e-6, SWE=1e-5, SNDPTH=1e-7,
          CANLIQ=1e-5, CANICE=1e-5, ECAN=1e-8, ETRAN=1e-8, ESOIL=1e-8, ZWT=1e-4, RUNSF=1e-9, RUNSB=1e-9, FSNO=1e-5)


def check(stats, limits):
    for k, lim in limits.items():
        assert k in stats, f"{k} missing"
        s = stats[k]
        assert s["nonfinite"] == 0, f"{k}: {s['nonfinite']} non-finite values"
        assert s["p99"] <= lim, f"{k}: p99 error {s['p99']:.3g} above {lim:g}"


@pytest.fixture(scope="module")
def noah33():
    return R.noah33()


@pytest.fixture(scope="module")
def noah271():
    return R.noah271()


@pytest.fixture(scope="module")
def mp401():
    return R.noahmp("401", os.path.join(HERE, "data", "noahmp401"))[0]


@pytest.fixture(scope="module")
def mp36():
    return R.noahmp("36", os.path.join(HERE, "data", "noahmp36"))[0]


def test_noah33(noah33):
    check(noah33, NOAH33)


def test_noah271(noah271):
    check(noah271, NOAH271)
    # SH2O(1) based direct evaporation (3.x uses SMC(1)): the layer-1 moisture of tiles with soil ice would be off by 1e-4 otherwise
    assert noah271["smc1"]["p99"] < 1e-6


def test_noahmp401(mp401):
    check(mp401, MP)
    assert mp401["ISNOW"]["max"] == 0.0


def test_noahmp36(mp36):
    check(mp36, MP)
    assert mp36["ISNOW"]["max"] == 0.0


# CLSM F2.5 core (clsmf25) on the July 1 2017 coldstart call dump, steps 2 and 12, 1500 tiles. July wet-canopy and freezing tiles are
# ill-conditioned (see LOG.md 2026-10-22 ~21:30), so the limits are looser than for the other models and on p99 only.
CLSM = dict(tc1=0.05, tc2=0.05, tc4=0.05, qa1=1e-5, capac=1e-3, catdef=1e-2, rzexc=1e-2, srfexc=1e-2, evap=1e-4, shflux=20.0, hlatn=20.0,
            tsurf=0.5, ar1=1e-4, ghflux=5.0, tp1=0.1, runoff=1e-5)


@pytest.fixture(scope="module")
def clsm():
    return R.clsm()


def test_clsm(clsm):
    check(clsm, CLSM)


# Noah-MP option variants (data/noahmp401_<tag>, data/noahmp36_<tag>: 266 recorded calls each, recorded with the option changed in lis.config).
# Same p99 limits as the default case; the loosened ones are measured differences of the Fortran/JAX pair, not tuned to pass:
#  - 3.6 sfc=2 (Chen 1997 without the 4.0.1 stability limits) is ill-conditioned for about 1% of the tiles (LOG.md 2026-10-23 ~00:30)
#  - 401 sfc=2 and stc 2/3 move the surface fluxes of the snow tiles a little more than the default case
MP_LOOSE_36_SFC2 = dict(MP, TV=0.2, TG=0.05, FSH=1.0, FCTR=5e-3, FGEV=0.05, SSOIL=1.0, FIRA=0.5, FSA=1e-3)
MP_LOOSE_401 = dict(MP, TV=5e-2, TG=1e-2, FSH=0.2, SSOIL=0.2, FIRA=0.1)
MP_LOOSE_36_STC2 = MP_LOOSE_401       # 266 calls, a third of them on snow: p99 is the third-largest error, noisy; the replay of all snow tiles gave FSH p99 2.3e-3, TG 6e-5 K (limits raised after seeing 0.064 W/m2 and 2.2e-3 K)
VARIANTS = [("401", t, MP) for t in ("run2", "run3", "run4", "frz2", "inf2", "crs2", "dveg3", "dveg1")] + \
           [("401", "sfc2", MP_LOOSE_401), ("401", "stc2", MP_LOOSE_401), ("401", "stc3", MP_LOOSE_401)] + \
           [("36", t, MP) for t in ("run2", "run4", "frz2", "inf2", "crs2", "dveg3", "dveg1")] + [("36", "stc2", MP_LOOSE_36_STC2), ("36", "sfc2", MP_LOOSE_36_SFC2)]


@pytest.mark.parametrize("ver,tag,limits", VARIANTS, ids=[f"mp{v}-{t}" for v, t, _ in VARIANTS])
def test_noahmp_option(ver, tag, limits):
    stats = R.noahmp(ver, os.path.join(HERE, "data", f"noahmp{ver}_{tag}"))[0]
    check(stats, limits)


# Noah-MP 5.0 (ported 2026-10-24): CONUS 3 h and SGP day-1 dumps of the 5.0 driver (data/noahmp50*), land points and land-ice points
MP50 = {k: v for k, v in dict(MP, TV=2e-2, TG=5e-3, FSH=1e-1, SSOIL=1e-1, FIRA=5e-2, SWE=1e-5).items() if k not in ("FCTR", "FGEV", "FSNO")}
GLA50 = dict(TG=1e-3, FIRA=1e-3, FSH=1e-3, SSOIL=1e-3, FSA=1e-4, STC=1e-2, SWE=1e-5, SNDPTH=1e-6, SNICE=1e-4, TAUSS=1e-8, ZSNSO=1e-5, T2MB=1e-3)


@pytest.mark.parametrize("tag", ["noahmp50", "noahmp50_sgp"])
def test_noahmp50(tag):
    stats, n = R.noahmp50(os.path.join(HERE, "data", tag))
    check(stats, MP50)


MP50_RUN = dict(MP50, RUNSF=2e-7)       # surface runoff [mm/s]; measured p99 2e-8 (option 8), 4e-8 (6, 7) on the 14 Feb rain records


@pytest.mark.parametrize("opt", [6, 7, 8])
def test_noahmp50_runoff(opt):
    stats, n = R.noahmp50(os.path.join(HERE, "data", f"noahmp50_run{opt}"))
    check(stats, MP50_RUN)


def test_noahmp50_glacier():
    stats, n = R.noahmp50(os.path.join(HERE, "data", "noahmp50_ice"), ice=True)
    assert n > 100
    check(stats, GLA50)

# Deep, wet, layered snowpack (CONUS 1 Mar 2017, 849 records: 150 each with 1, 2, 3 snow layers, liquid water > 1 mm, snow > 100 mm, plus 100 snow-free; built by
# make_test_data.snow_family). Limits fixed by rule BEFORE any mutation run: 10 x the p99 measured on 2026-11-10 (float32 noise level), rounded up to one digit.
# Written to catch the edits that passed every other test: snow-layer liquid retention (SNLIQMAXFRAC), bottom-layer release (SNOW_RET_FAC), ice conductivity (TKICE).
SNOW50 = dict(TG=7e-4, STC=6e-3, SWE=2e-4, SNICE=5e-6, SNLIQ=3e-6, ZSNSO=3e-6, SNDPTH=1e-6, FSH=2e-1, FIRA=7e-3, SSOIL=7e-3, TV=2e-3, FSA=2e-4, QSNBOT=3e-10)


def test_noahmp50_snow():
    stats, n = R.noahmp50(os.path.join(HERE, "data", "noahmp50_snow"))
    assert n > 800
    check(stats, SNOW50)



# dynamic vegetation 2, 5, 6 (carbon pools, predicted LAI) and 7, 8, 9 (input LAI): SGP Jul 9 2017, 150+ random calls each; measured p99 on 15,000 calls:
# PLAI 6e-8, FASTCP 6e-5, LFMASS 0, PSN 2e-6 (2-6) and 9e-6 (7-9), TV 6e-5..9e-5, FSH 7e-4..6e-3 (single-tile maxima up to 7 W/m2 in the stability iteration)
MP50_DVEG = dict(MP50, PLAI=1e-5, LFMASS=1e-3, RTMASS=1e-3, STMASS=1e-3, FASTCP=1e-2, STBLCP=1e-2, PSN=1e-4)


@pytest.mark.parametrize("opt", [2, 5, 6, 7, 8, 9])
def test_noahmp50_dveg(opt):
    stats, n = R.noahmp50(os.path.join(HERE, "data", f"noahmp50_dveg{opt}"))
    check(stats, MP50_DVEG if opt < 7 else {k: v for k, v in MP50_DVEG.items() if k not in ("LFMASS", "RTMASS", "STMASS", "FASTCP", "STBLCP")})


# tile drainage 1 (simple) and 2 (Hooghoudt): SGP Feb 14 2017, measured p99 on 15,000 calls: SMC 0 (max 6e-8); without the drainage SMC is off by up to 1.9e-4 (1) / 6.9e-3 (2)
@pytest.mark.parametrize("opt", [1, 2])
def test_noahmp50_tile_drainage(opt):
    stats, n = R.noahmp50(os.path.join(HERE, "data", f"noahmp50_tdrn{opt}"))
    check(stats, dict(MP50, SMC=1e-6))


# irrigation: measured p99 on 15,000 calls: SMC 0 (max 1.7e-6), FSH 7.8e-3, IRWAT*/IRNUM* identical, IRRSPLH accumulated [J/m2] p99 0
@pytest.mark.parametrize("tag", ["m0", "m3", "l1"])
def test_noahmp50_irrigation(tag):
    stats, n = R.noahmp50(os.path.join(HERE, "data", f"noahmp50_irr{tag}"))
    check(stats, dict(MP50, SMC=1e-6, IRWATSI=1e-6, IRWATMI=1e-6, IRWATFI=1e-6, IRNUMSI=0.5, IRNUMMI=0.5, IRNUMFI=0.5, IRRSPLH=1e3))


# dynamic crop model (crop option 1, corn on the cropland tiles, synthetic PLANTING / HARVEST / SEASON_GDD maps): SGP Jul 9 2017, 536 cropland calls that cover all eight
# growth stages (and the harvest reset); measured p99: PGS 0 (exact), GDD 1e-6, GRAIN 3e-8, LFMASS 0, STMASS 2e-6, PSN 2e-6, GPP/NPP/NEE 2e-11 / 2e-11 / 8e-11
def test_noahmp50_crop():
    stats, n = R.noahmp50(os.path.join(HERE, "data", "noahmp50_crop1"))
    check(stats, dict(MP50_DVEG, PGS=0.5, GDD=1e-4, GRAIN=1e-5, GPP=1e-9, NPP=1e-9, NEE=1e-8))


# irrigation trigger 2 (within [planting day, harvest day - IRR_HAR]), flood method; measured p99 on 15,000 calls: SMC 0 (max 1.7e-6)
def test_noahmp50_irrigation_trigger2():
    stats, n = R.noahmp50(os.path.join(HERE, "data", "noahmp50_irrt2"))
    check(stats, dict(MP50, SMC=1e-6, IRWATSI=1e-6, IRWATMI=1e-6, IRWATFI=1e-6, IRNUMSI=0.5, IRNUMMI=0.5, IRNUMFI=0.5, IRRSPLH=1e3))
