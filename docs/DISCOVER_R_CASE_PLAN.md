# Discover test plan: LIS developer R's Noah-MP 5.0 NLDAS-2 case

Status: PLAN, recon done 2026-11-10, no JAX run yet. Owner: G. Tamkin. Replaces the generic `DISCOVER_TEST.md` for this case.

## 1. What R provided (read on Discover 2026-11-10, read-only)
Directory `/discover/nobackup/projects/eis_freshwater/rabolafi/V5_testing/NLDAS2-v5-testing-OL/` (group s2443, readable by gtamkin).
| Item | Value |
|---|---|
| Run | LIS 7.8 (module `lisf_7.8_intel_2025.3.2`), `Noah-MP.5.0`, retrospective, open loop |
| Domain | lat 25.0625-52.9375, lon -124.9375 to -67.0625, 0.125 deg = 464 x 224: the same NLDAS-2 grid as our CONUS case |
| Period in config | 2008-10-01 to 2010-10-01 (2 water years); Noah-MP and soil timestep 30 min |
| Forcing | `"NLDAS2 grib"`, hourly GRIB (`NLDAS_FORA0125_H.A*.grb`) in `/discover/nobackup/projects/lis/MET_FORCING/NLDAS2.FORCING/<yr>/<doy>/`; neighbor interpolation; model-level data off |
| Parameters | `lis_input.nldas2.noahmp401.OL.nc` (9.2 MB; file name says 4.0.1 although the LSM is 5.0: check it holds the 5.0 parameters); output list `input/MODEL_OUTPUT_LIST_noahmp5_NLDAS.TBL` |
| Job | 60 ranks (10 x 6), 1 node, 3 h limit, `--account=s2443 --constraint=mil --qos=serial` |
| Existing Fortran output | `OUTPUT_Basic_OL/netcdf/SURFACEMODEL/<yyyymm>/LIS_HIST_*.nc`, 11,404 hourly files, 96 GB, from 2008-10-01 01:00 to 2010-01-19 04:00 (the run did not reach 2010-10-01; the job log has no completion line; probably stopped at the time limit: check) |

## 2. Differences from what is validated so far
1. **Forcing is GRIB; the JAX driver reads NetCDF** (`NLDAS2 netcdf`, `NLDAS_FORA0125_H.A*.nc`). Needs a GRIB reader (eccodes / pygrib / cfgrib) or a NetCDF conversion. Biggest gate.
2. **Period and season**: 2008-2010, all seasons (our validation is Feb-Apr 2017 plus SGP). Summer, vegetation and irrigation behaviour are new to the 5.0 port.
3. **Parameter file** built for 4.0.1 naming; the JAX parameter builder expects the 5.0 LDT layout (`ldt_mp5c`).
4. **Timestep and output**: 30 min, hourly history; our runs use the config's step and `--hours 1` for hourly output.
5. **Fortran build**: Intel 2025.3.2 on Milan, 60 ranks; ours was GNU 14.2. An independent reference, and a different compiler for the 32+ rank question.

## 3. What Discover offers
- GPU partition `gpu_a100` (10 nodes, 4 x A100, 48 CPUs, 510 GB). First A100 timing for this port (we have V100 only), account `ilab` per the note from the Discover session; groups of gtamkin include s2443, s3278, ilab.
- Login node can reach PyPI and GitHub (HTTP 200). Python via `module load python/GEOSpyD/24.11.3-0/3.12` (3.12.9; JAX 0.5.x supports 3.10-3.12). Checked 2026-11-10 in that module: `pygrib`, `xarray`, `netCDF4`, `numpy` import; `eccodes`, `cfgrib` and `jax` do not (jax is pip-installable: PyPI is reachable).
- Compute nodes cannot see /panfs or home: work from `/discover/nobackup/gtamkin`.
- Access: gateway refuses commands over an SSH master. Commands go through the user's authenticated screen session (`screen -S 3822538.discover -X stuff ...`, log `/home/gtamkin/discover_screen.log`); that session is shared with another Claude session, so keep commands short, use markers, never use `^` or `#` in typed text (screen interprets them), and do not change the shell environment (use subshells).

## 3b. NetCDF NLDAS-2 forcing on Discover? No (checked 2026-11-10)
`/discover/nobackup/projects/lis/MET_FORCING/` has `NLDAS2.FORCING` (years 1979-2023, hourly `NLDAS_FORA0125_H.A*.002.grb` plus `NLDAS_FORB0125*.grb`, GRIB only, also for 2017), `NLDAS1.FORCING` and `NLDAS_ENS` (`conus_ens_*.nc`, an ensemble product, not NLDAS-2 hourly). No NetCDF copy of the hourly NLDAS-2 forcing found in that collection; other project directories were not searched. Our ADAPT NetCDF files cover Feb-Apr 2017 only, not 2008-2010. So the route is a GRIB reader (`pygrib` is available) or a one-off GRIB-to-NetCDF conversion with it; downloading NetCDF from GES-DISC needs an Earthdata login.

## 3c. Result: GRIB reader vs R's Fortran forcing (2008-10-01 01:00, 2026-11-10, Discover, `tools/forcing/check_grib_vs_hist.py`)
Venv on Discover: Python 3.12.9 (GEOSpyD module, system site packages for pygrib/netCDF4), jax 0.5.3, pygrib 2.1.6, netCDF4 1.7.2, numpy 2.0.2; repo at 14737c6. The driver's `read_grib` of `NLDAS_FORA0125_H.A20081001.0100.002.grb` against `Tair/Qair/Psurf/LWdown_f_inst` in R's `LIS_HIST_200810010100.d01.nc`: **bitwise identical on all 76,180 cells** for all four; wind speed within 7.7e-7 (float32 rounding of the speed); SWdown 75,592 of 76,180 bitwise (max 4.4 W m-2) and Rainf 64,148 of 76,180 (max 2.8e-3 kg m-2 s-1 differences): these two go through the zenith-angle and bracket time interpolation of the LIS forcing code, which `NLDAS2.at()` implements and which is not part of the reader; not yet separated out. The previous-hour file (00:00) does not match, as expected.

## 3d. Status 2026-11-10 (Discover)
- Parameters: `SetupMP50` builds from R's config and `lis_input.nldas2.noahmp401.OL.nc` (login node, CPU): 76,180 tiles, lat 25.0625-52.9375, lon -124.9375 to -67.0625, same tile count as our CONUS case (`scripts/discover/setup_check.py`).
- A100 jobs: `gpu_a100` accepts a job only with `-C rome` and a time limit above 1 h (nodes carry features `rome,edr,cssro,notshort`); without them `sbatch` fails with "Requested node configuration is not available" even though `--test-only` passes. Account `ilab`, `--gres=gpu:1 -c 16`. Queue estimate on the day: start about 18:58.
- 1-day JAX run (2008-10-01 to 10-02, hourly history): job 59223727, `scripts/discover/run1d.sbatch` (`sbatch -C rome`), output `/discover/nobackup/gtamkin/jaxtest/out_1d`, log `r1d_<jobid>.out`. Compare with R's history files using `driver/compare_hist.py`.
- First A100 job (59223727) ran 21 s and failed: JAX could not load its CUDA libraries ("Error loading CUDA libraries. GPU will not be used", then "DNN library initialization failed"). Cause: the venv was created with `--system-site-packages`, so pip found the NVIDIA wheels in the GEOSpyD base environment and did not install them into the venv; the JAX CUDA plugin looks for them inside its own site-packages. Fix: a clean `venv2` (no system packages: `pip install 'jax[cuda12]>=0.5,<0.6'` then `pip install -e '.[grib]'`; pygrib 2.1.8 ships a wheel with eccodes). Versions in venv2: jax 0.5.3, numpy 2.5.4, pygrib 2.1.8, netCDF4 1.7.5, cudnn 9.28. The scripts now set `JAX_PLATFORMS=cuda` so a missing GPU fails instead of falling back to CPU. Retry: job 59224273 (pending, priority).
- Typing into the shared screen session: commands over about 250 characters, `^`, `#`, a trailing `&` before the marker and `\n` in sed all break; scripts go over as base64 lines or via `git pull` (works without a prompt).

## 4. Stages
| # | Stage | Done when | Needs |
|---|---|---|---|
| 0 | Recon (done) | facts in section 1 | |
| 1 | Put the repo on Discover (`git clone` over HTTPS needs a GitHub credential entered by the user; or the user copies a tarball) | `git log` shows the commit; `pip install ".[test]"` into a venv under `/discover/nobackup/gtamkin/venvs` | user credential |
| 2 | Environment: venv with `jax[cuda12]>=0.5,<0.6`, netCDF4, a GRIB reader; test on a `gpu_a100` allocation (not the login node) | `python -c "import jax; jax.devices()"` shows an A100 | |
| 3 | Fortran reference check: confirm the existing output is complete for the window we compare (first 1-3 days, then 60 days from 2008-10-01), copy only the needed hourly files (or read in place) | file count and checksums noted; the run's actual stop reason found in `lis.59119005.out` / `noahmp.err` | |
| 4 | GRIB forcing reader in the JAX driver (done, synthetic-tested) and variable mapping to our 11 forcing fields; verify one hour against the Fortran forcing diagnostics in the history files (`Tair_f_inst` etc.): must match bitwise, as in the NetCDF case | forcing test passes for 24 h | decision: reader vs conversion |
| 5 | Parameters: build the JAX parameters from `lis_input.nldas2.noahmp401.OL.nc` and the config; list missing 5.0 parameters | 1-step replay or 1-hour run runs without NaN | R's LDT inputs (`LDT/`, `ldt.config.Basic.OL`) |
| 6 | JAX 1-day run on A100 vs Fortran output; then 60 days (2008-10-01 to 2008-11-30) | compare_hist: p99 per variable, bitwise %, flux variables included (we lack 5.0 fluxes in closed loop so far) | |
| 7 | Timing: JAX wall time (cold and steady) on A100; Fortran time from R's job log for the same days; report the 60-rank ratio with its caveats (Intel, Milan, 60 ranks, one run, no repeats) | table in `docs/reports/BENCHMARKS.md` | R's job timing |
| 8 | Report to R: what matched, what did not, and what we need (the exact LDT/forcing used) | report in `docs/reports/` | user approval to share |

## 5. Decisions needed from the user
1. Clone path for the repo on Discover (user enters a GitHub credential in the screen session, or provides another transfer).
2. GRIB route: decided, native `pygrib` reader implemented (see 3b); next is checking it on real files.
3. Whether to share the report and results with R, and the repo access that implies.
4. Slurm account for the A100 job (default `ilab`).

## 6. Risks
- The reference run may be partial or from different parameters than the case file (4.0.1 file name): confirm with R before drawing conclusions.
- 2008-2010 seasons not yet validated for 5.0 (summer vegetation, irrigation off in the config?): disagreements may be physics, not forcing.
- A100 results are not comparable to the V100 numbers until the same case runs on both.
- The shared screen session is also used by another agent session; keep to short commands and never interrupt its jobs.
