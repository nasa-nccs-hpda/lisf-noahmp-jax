# Reproducing one Noah-MP result from scratch (2026-10-24)

Owner: G. Tamkin. Written by a Claude Code session. Project-local. Status: DRAFT, not yet followed by a second person (that is the point of this page).
Purpose: let someone other than the author check one claim end to end: **Noah-MP 4.0.1, Southern Great Plains (27x22 cells, 0.25 deg), NLDAS-2 forcing, 2017-01-01 to 2018-01-01, JAX on a V100 is about 8-10x faster than the serial Fortran LIS and agrees with it** (STATUS_20261022.md, docs/reports/BENCHMARKS.md). Total effort: about 1.5 h of machine time, most of it the Fortran run.

## 0. What you need (all outside the repository; paths of the author's account, group `ilab`)
| Item | Where |
|---|---|
| This repository (`lisf-noahmp-jax/`) | `projects/LISF/lisf-noahmp-jax` |
| Case directory `$LISF_RUN` with `INPUT/` (NLDAS-2 netcdf 2017, parameter tables), `ldt_mp/` (LDT config + output), `LIS_mp_clean` (Fortran LIS, GNU 14.2 serial, Noah-MP built in) | `$LISF_RUN` |
| Fortran runtime environment | `source $LISF_DEPS/env_lis.sh` |
| Python with JAX 0.5.3 (conda env `graphcast-env`); on a GPU node `--constraint=v100 --gres=gpu:1`; CPU also works (`JAX_PLATFORMS=cpu`, slower) | Nexus conda module, see `Reports/ENVIRONMENT.md` |
| Raw LDT inputs (only for step 1) | `$LISF_RUN/ldt_mp/INPUT/LS_PARAMETERS` |

Anything that is not under these paths has not been tested; the code contains some absolute paths of the author (see `DATA_AND_PATHS.md`).

## 1. Parameter file (optional: the real LDT output is already in `$LISF_RUN/ldt_mp/out/`)
```
cd $LISF_RUN/ldt_mp
PYTHONPATH=<repo>/src python3 -m lisf_noahmp_jax.driver.ldt ldt.config out/lis_input.sgp.noahmp401.py.nc
PYTHONPATH=<repo>/src python3 -m lisf_noahmp_jax.driver.compare_ldt out/lis_input.sgp.noahmp401.nc out/lis_input.sgp.noahmp401.py.nc
```
Expected: every variable bitwise identical except SLOPE and ASPECT (max relative 4e-7). 25 s on one core.

## 2. Configuration (one year)
Take `$LISF_RUN/lis.config.mp` (Noah-MP 4.0.1, 557WW-style options) and set the period and output directory:
```
sed 's#^Output directory:.*#Output directory: ./fort_year#; s#^Diagnostic output file:.*#Diagnostic output file: ./fort_year/lislog#; \
     s#^Starting month:.*#Starting month: 1#; s#^Starting day:.*#Starting day: 1#; s#^Ending year:.*#Ending year: 2018#; \
     s#^Ending month:.*#Ending month: 1#; s#^Ending day:.*#Ending day: 1#' lis.config.mp > lis.config.fort_year
sed 's#fort_year#jax_year#g' lis.config.fort_year > lis.config.jax_year
```

## 3. Fortran reference (about 50 min on one Xeon Gold 6248 core)
```
cd $LISF_RUN && mkdir -p fort_year && source $LISF_ENV
/usr/bin/time -v ./LIS_mp_clean -f lis.config.fort_year > lis.fort_year.out 2> lis.fort_year.time
```
Expected: `LIS Run completed.` in `fort_year/lislog.0000`; wall time 2,983 s on the author's run (job 37946318).

## 4. JAX run (about 5 min on a V100 including compile; the first 2-3 min are XLA compile)
```
cd <repo>
PYTHONPATH=src python3 -u -m lisf_noahmp_jax.driver.lis -C $LISF_RUN $LISF_RUN/lis.config.jax_year
```
Prints a `TIMING` line (setup, forcing_host, device, output, total). Expected total about 290-370 s (286.5 s in job 37946338). Env var `LISJAX_FORCING_PROCS=1` disables the parallel forcing reader and background writer (slower, same numbers).

## 5. Compare
```
PYTHONPATH=<repo>/src python3 -m lisf_noahmp_jax.driver.compare_hist $LISF_RUN/fort_year $LISF_RUN/jax_year
```
Expected (author's run): 1,460 history files; p99 |diff| Qle 0.128, Qh 0.132, Qg 0.129 W/m2; SoilMoist 3.7e-4 m3/m3; SoilTemp 4.4e-3 K; SWE 5e-5 kg/m2; no 'NaN only in the new run' for model fields. Forcing diagnostics (Tair, Qair, Psurf, LWdown) bitwise identical.
The comparison reports p99 (not the maximum): a few freeze/thaw and wet-canopy tiles amplify 1-ulp float32 differences (maxima 2.6 / 5.3 / 6.6 W/m2 here).

## 6. What would count as failure
* Qle, Qh or Qg p99 above about 0.3 W/m2, soil moisture p99 above 2e-3 m3/m3, or NaN in JAX where the Fortran is finite.
* JAX speed-up below 4x on the same node type (the goal) - report the TIMING breakdown.
* Fortran run not finishing or a different wall time by more than 2x (different node type): compare only same-node pairs.

## 7. Shorter checks (minutes)
* `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu python3 -m pytest -q tests` (48 tests, 28-52 min, see tests/README.md; `-k "not option"` about 4 min) replays recorded Fortran calls; `tests/test_ldt.py` needs the raw LDT data and is skipped without it.
* Restart round trip (about 5 min CPU): LOG.md 2026-10-24 ~03:00.

## 8. The same check for Noah-MP 5.0 (added 2026-10-24; DRAFT, not followed by a second person)
Claim: **Noah-MP 5.0, SGP, 2017 full year, JAX on a V100 is about 10x faster than the Fortran LIS (2,945 s vs 277 s) and reproduces its history files** (SoilMoist p99 6.6e-4, SoilTemp p99 4.4e-2 K).
* Parameter file: the real LDT output is in `$LISF_RUN/ldt_mp5/out/lis_input.sgp.noahmp5.nc` (Noah-MP 5.0 uses the same LDT parameters as 4.0.1 plus the table `NoahmpTable.TBL`, which LIS reads itself).
* Configs: `$LISF_RUN/mp5_sgp_year/lis.config.fort` and `lis.config.jax` (from `lis.config.mp5`, period 2017-01-01 to 2018-01-01, output `./fort` / `./jax`, 6-hourly history).
* Run both with `sbatch scripts/slurm/gpu_bench_mp5_sgp_year.sbatch` (Fortran about 49 min on one core, JAX about 5 min), then `PYTHONPATH=<repo>/src python3 -m lisf_noahmp_jax.driver.compare_hist $LISF_RUN/mp5_sgp_year/fort $LISF_RUN/mp5_sgp_year/jax`.
* Expected: 1,460 files; SoilMoist p99 about 6.6e-4, SoilTemp p99 about 4.4e-2 K (larger than 4.0.1 because of snow-onset chaos in January, see `docs/reports/BENCHMARKS.md`), SWE p99 3.4e-5; no 'NaN only in the new run' for model fields. The 5.0 history list of the testcase holds states only (no Qle/Qh/Qg).
* CONUS 1 day: `sbatch scripts/slurm/gpu_bench_conus_mp5.sbatch` (Fortran 663 s, JAX 95 s), restart round trip: `sbatch scripts/slurm/gpu_rt_conus_mp5.sbatch`.
* Open-loop check of the physics: `MPVER=50 python3 tools/replay/replay_mp50.py <dump dir> 40000` (land) and `GLA=50 python3 tools/replay/replay_gla50.py <dump dir>` (land ice); the dump needs `LIS_mp50_dump` (built with `lis_patches/make_noahmp50_hook.py`).


## Notes from the independent review (2026-11-05)
- To match the stored hourly Fortran output of the 1-day CONUS case, run the JAX driver with `--hours 1`; the default is 6.
- `-C` must be the case directory that holds `INPUT/` and the LDT file.
- Expect a divide-by-zero warning from water tiles in older versions; it is harmless and now masked.
