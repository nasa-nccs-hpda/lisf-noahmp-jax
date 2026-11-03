# LISF to JAX: result report (version 3, 2026-10-22)

Written by a Claude Code session working autonomously on the owner's instructions of 2026-10-21 ("build LIS with GNU 14, show a real LIS simulation of a real region can be reproduced, then port all pieces, keep going, then capture benchmarks"). Status: DRAFT for owner review; no person has reviewed any of it; nothing is committed to git.
Audience: the project owner (G. Tamkin), who said they know nothing about this software.
Companion files: `LOG.md` (every step with dates and numbers, including failures), `BENCHMARKS.md`, `GOAL.md` and `ACCEPTANCE_CRITERIA.md` (the rules agreed in advance for the first, physics-only phase), `ENVIRONMENT.md`, `lis_patches/` (all changes made to build LISF), and the code in `noah39_jax.py`, `lisjax/`, `lis/`, `baseline/`.
The first version of this report (physics routine only, synthetic forcing) is superseded; its findings still hold and are summarized in section 5.

## 1. In one paragraph

LISF (NASA's Land Information System Framework) is a large Fortran system for land-surface hydrology. I built it from source with the GNU 14 compiler (LDT, the parameter processor, and LIS, the model), ran NASA's public walkthrough over the Southern Great Plains with real NLDAS-2 weather data, and confirmed that my build reproduces NASA's published reference output. I then rewrote, in Python/JAX, a complete working LIS configuration for the Noah land model (versions 3.9 and 3.6): reading the configuration file, the LDT parameter file and the NLDAS-2 forcing, building the model grid and parameters, stepping the model with a JAX re-implementation of the Noah physics (including the snow physics option, the MYJ surface-exchange option and the land-ice branch), and writing LIS-format history and restart files. Run side by side with the real Fortran LIS on the same inputs, it reproduces the real model's output to float32 rounding level in the typical case (99th percentile of heat-flux differences below 1 W/m2) with isolated larger differences in snow-threshold cases, and it reproduces NASA's own published reference output as closely as my independent Fortran build does. On a CONUS-size domain (76,180 land tiles) one simulated day takes 443 s in the Fortran LIS on one core and 27 s (cold, including 17 s of compilation) or about 5 s (steady state) in Python/JAX on one V100.

## 2. What "a real LIS simulation of a real region reproduced" means here

- Region and period: Southern Great Plains (34.4 to 39.6 N, 102.9 to 96.1 W, 0.25 degree, 616 land cells), January and February 2017; and the continental US on the 0.125 degree NLDAS grid (76,180 land tiles), 1 to 2 days of February 2017.
- Real inputs: NASA's published parameter file for the walkthrough (and, for CONUS, a parameter file produced by my build of LDT from NASA's global source maps) and hourly NLDAS-2 meteorological forcing from NASA's data archive on this cluster.
- Reference results: (a) NASA's published walkthrough output (`target_step2`), (b) the real LISF Fortran code built by me with GNU 14.
- The comparison is made on LIS history files (six-hourly time averages of 19 variables), on restart files, and, for the physics, on every individual call of the Noah routine (instrumented real runs).

## 3. The Fortran build (what it took)

- Compiler and libraries: GNU 14.2 (gfortran/gcc), built here from source: HDF5 1.14.4, netCDF-C 4.9.2, netCDF-Fortran 4.6.1, ecCodes 2.38.3 (built but not used), ESMF 8.8.0 (no MPI), HDF4 4.2.16 and libtirpc (for one LDT input file). Serial (no MPI): the cluster's OpenMPI module needs a library (`libhwloc.so.15`) that is missing on the nodes. Scripts in `lis_patches/`.
- Source problems found and fixed in a build copy (never in the repository): a missing git submodule (Noah-MP 5.0), two serial-build bugs in LIS (`NoahMP401_setup.F90`, `HYMAP3_routing_writerst.F90`), a typo'd module name and an undefined subroutine in LDT (`USAFSI_ssmisMod.F90`, `SMAP_E_OPL/TOOLSUBS.F90`). Patches in `lis_patches/`.
- Validation of the build: LDT output for the walkthrough equals NASA's published parameter file bitwise in 19 of 22 variables (slope, aspect and bottom temperature differ by at most 4e-9, 7e-7, 9e-5). LIS (Noah 3.6, the walkthrough configuration, two months) versus NASA's published reference: forcing diagnostics identical, fluxes median 1e-3 W/m2, 99th percentile 0.14 to 0.6 W/m2.

## 4. What was ported to Python/JAX (and what was not)

Ported (files in `lisjax/` and `noah39_jax.py`):
| LISF component | Python/JAX | Status |
|---|---|---|
| Noah 3.9 and Noah 3.6 column physics (SFLX and ~25 routines), `thermal conductivity` options 0/1/2 | `noah39_jax.py` | validated on 414k to 1.4M real calls |
| UA snow physics, SFCDIF_OFF and SFCDIF_MYJ surface exchange | `noah39_jax.py` | validated on real calls |
| Land-ice branch (SFLX_GLACIAL) | `noah39_jax.py` | validated on 2,976 real calls |
| Config reading (the keys the Noah configurations use) | `lisjax/config.py` | |
| LDT parameter file, tile creation, soil/vegetation/general tables, coldstart state, daily and per-step parameter updates ("dynsetup" with monthly map interpolation) | `lisjax/params.py`, `tables.py`, `timeutil.py` | all 47 parameters/states identical to the real run |
| NLDAS-2 forcing path (neighbor interpolation, time and zenith-angle interpolation) | `lisjax/forcing.py` | 100% bitwise for 6 of 7 fields |
| Time stepping on device, time-averaged history output, restart write/read | `lisjax/model.py`, `run.py` | |
Not ported (still Fortran, or not done): LDT itself (parameter processing from raw global maps; the Fortran LDT is built and works), all other land surface models (Noah-MP, CLSM, Mosaic, JULES, VIC, CLM, SUMMA, ...), all other forcing readers (GDAS, GEOS, MERRA-2, GLDAS, ERA5, ...), data assimilation (EnKF, observation readers, perturbations), routing (HyMAP), irrigation, radiative-transfer models, parameter estimation/optimization (optUE), the verification toolkit (LVT), MPI domain decomposition, ensembles/multiple tiles per cell, the full list of ~200 output variables (19 done), 6-hourly output only (other intervals not coded). About 2,900 Fortran files make up LIS; the ported part is the Noah-3.x slice (physics ~6,300 lines, wrapper/setup ~3,600, one forcing reader).

## 5. Evidence

All numbers from `LOG.md` and the result files; "p99" is the 99th percentile over all tiles and times.
1. **Single physics calls (open loop)**: 414k real calls (Noah 3.9, sfcdif 0): 98 to 100% bitwise for soil moisture/snow/albedo; t1 p99 3e-5 K, max 1.1e-2 K; heat fluxes p99 < 2e-3 W/m2. With UA snow + MYJ (413,952 calls) and 1.4 million January calls with snow: same accuracy class. Land ice (2,976 calls): t1 97.7% bitwise.
2. **Free-running real tiles**: each of the 616 tiles run for a week from recorded forcing: worst-tile excursions 0.4 K in surface temperature on one day (near-freezing tiles), p99 0.05 K.
3. **End-to-end Python/JAX vs real LIS (SGP, one week)**: 28 history files: Rainfall, snowfall, air temperature, LW down bitwise; fluxes median 1e-3 W/m2, p99 0.1 to 1.1 W/m2; soil moisture p99 2e-4; soil temperature p99 0.035 K.
4. **Python/JAX vs NASA's published reference** (Noah 3.6, UA snow, MYJ, Jan to Feb 2017, 236 files): fluxes p99 0.15 to 0.68 W/m2 (max 2 to 9), SWE p99 0.0075 kg/m2, soil moisture p99 4.4e-4, soil temperature p99 0.035 K. My Fortran build against the same reference: fluxes p99 0.14 to 0.60 W/m2 (max 3 to 16). So the port is as close to NASA's numbers as an independent Fortran build.
5. **CONUS, 2 days, 76,180 tiles**: Python/JAX vs real LIS: fluxes p99 0.0003 to 0.95 W/m2 (max 0.05 to 10), SWE max 0.026 kg/m2, soil moisture p99 8e-5, soil temperature p99 0.022 K.
6. **Restart**: a run split through a restart file is bitwise identical to the one-shot run; the restart written at 2017-02-08 agrees with the real LIS restart (T1 max 3.3e-3 K).
7. **Original first-phase result (synthetic forcing, physics routine only)**: JAX V100 vs one Fortran core: 146x at 262,144 columns; the pre-registered agreement rule was not fully met (one field, `dew`, 5 to 23% over its limit); that rule has not been revisited.

Larger differences: single tiles can differ by several W/m2 in a six-hour mean (maximum 10 W/m2 in CONUS) because snow melt/freeze thresholds make the model sensitive to rounding; a different compiler would show the same effect (my own Fortran build against NASA's reference shows 16 W/m2 maxima).

Two bugs the real-data comparison caught, which synthetic tests could not: (a) LIS resets leaf area every step, so snow burial must not accumulate; (b) land-ice tiles use a roughness that LIS never updates.

## 6. Speed (one simulated CONUS day, same V100 node type; details in `BENCHMARKS.md`)
- Fortran LIS, one core: 443 s. Python/JAX on V100: 26.8 s the first time (17 s compile), about 5 s per day afterwards (0.32 s of that is the GPU; the rest is single-threaded numpy preparation of forcing, the next thing to optimize).
- Against a parallel (MPI) LIS the gap would be smaller; no parallel LIS was built.
- Plain CPU JAX is roughly competitive with Fortran only through its multi-threading (20.7 s with 8 threads).

## 7. Limits and risks
- One physical model family (Noah 3.x), one forcing dataset (NLDAS-2), two regions, February/January 2017. Other seasons, regions, climates (tropics, deserts, permafrost), longer periods and other LSMs are untested.
- The Python/JAX program covers a configuration subset: 4 soil layers, one tile per grid cell, LDT/IGBPNCEP land cover, monthly greenness/albedo maps, no ensembles, no irrigation, no routing, 6-hour output.
- float32 only; GPU results differ slightly from CPU and across JAX versions.
- The environment used JAX from an existing conda environment; no clean rebuild test.
- Nobody has line-by-line reviewed the port. Faithfulness rests on the comparisons in section 5.

## 8. Decisions for you
1. Which further LISF pieces matter most? Highest value per effort in my view: (a) more forcing readers (MERRA-2, GLDAS, ERA5) to run global cases, (b) Noah-MP and CLSM (large ports), (c) data assimilation (large, only statistically verifiable).
2. Accept the first-phase `dew` result (already accepted by you on 2026-10-21 for now).
3. Commit/push: nothing is committed. Files are under `projects/LISF/jax-port/` (untracked in the LISF clone) and, outside the repo, in `dev/lisf_deps`, `dev/lisf_build`, `dev/lisf_run`.

## 9. Update in version 3: Noah-MP, land ice, ensembles, restart, global run (all DRAFT, unreviewed)

Added since version 2 (details and numbers in `LOG.md`; speeds in `BENCHMARKS.md`):
| Piece | Files | Result vs the real Fortran LIS (GNU 14 build) |
|---|---|---|
| Noah-MP 4.0.1 (557WW option set: dveg 4, Ball-Berry, SIMGM runoff, M-O drag, CLASS snow albedo; 4 soil + 3 snow layers) | `noahmp401_jax.py`, `lisjax/params_mp.py`, `model_mp.py`, `run_mp.py` | 30k and 60k real-call replays: snow-layer count exact, STC max 1.3e-3 K, heat flux p99 <= 2.2e-3 W/m2. SGP 7-day end-to-end: Qle p99 0.017, Qh 0.26, Qg 0.32 W/m2. 59-day snow run: flux p99 <= 0.25 W/m2 |
| Noah-MP land-ice branch (veg class 15) | `noahmp401_glacier_jax.py` | 9,424 replayed calls at float32 level; CONUS day with 31 ice tiles included |
| MERRA-2 forcing reader, global Noah-MP (30,307 tiles, 86 ice) | `lisjax/forcing.py` | Qle p99 3.3e-3, Qh 6e-3, Qg 3.9e-3 W/m2 |
| CONUS Noah-MP, 1 day, 76,180 tiles | same | Qle p99 1.3e-3 (max 0.29), Qh p99 0.056 (max 2.4), Qg p99 0.061 (max 2.0) |
| Restart write/read in LIS format | `run_mp.py` | split 3d+4d run bitwise equal to continuous for 46 variables; float32-level match to Fortran restart |
| Ensembles (12 members) with GMAO forcing perturbation (L'Ecuyer RNG, AR(1), bias correction) | `lisjax/pert.py` | seeds identical, AR fields to 2e-6, ensemble-mean Qle p99 1.1e-4, Qh 7.9e-4 W/m2 |

Speed (Noah-MP, CONUS, one simulated day): Fortran 626.6 s on a gpu011 core; JAX on V100 89.2 s cold (83 s compile) = 7.0x, 95.8 s for 2 days = 13.1x, about 4.5-6.6 s/day steady state (about 95x; the GPU itself is 0.32 s/day, the rest is host-side forcing preparation). The 4x goal is met for runs of one day or more on this domain, with the compile cost included. Week-long projection (about 34x) is a projection, not a measurement, until the 7-day Fortran reference finishes and is compared.

Still not ported: data assimilation (EnKF; in progress), other LSMs (CLSM, Mosaic, JULES, VIC, CLM, SUMMA), other forcing readers (ERA5, GLDAS/FLDAS, IMERG), HyMAP routing, LVT, LDT, MPI, the full output list. Limits in section 7 still apply except "no ensembles" (now supported for Noah-MP with forcing perturbation) and "one LSM family".

### 9.1 Added later on 2026-10-22 (also DRAFT, unreviewed)
| Piece | Files | Result vs the real Fortran LIS |
|---|---|---|
| EnKF soil-moisture data assimilation (Noah-MP + "Synthetic SM", 12 members, forcing/state/observation perturbations) | `lisjax/da.py`, `lis_patches/synsm_noahmp401_register.py` | identical-twin test, SGP 3 days, 12 analyses: ensemble-mean soil moisture differs by 2.7e-7 (day 1) to 2.2e-3 (day 3, max; rms 6e-5) from the real LIS, against 2.2e-2 (rms 1.4e-3) for the same Fortran run compared with the JAX run without assimilation |
| HyMAP2 river routing (kinematic / local-inertia / diffusive, floodplain, adaptive step) | `lisjax/hymap2.py` | CONUS 2 days, 78,554 river cells: streamflow p99 5.5e-4 m3/s (range 197), river storage p99 12 m3 (range 7e6), flooded fraction bitwise in 99% of cells |
| Second use case: global winter (MERRA-2 forcing, Jan 1-5 2017, 30,307 tiles incl. snow, frozen soil, 86 ice tiles) | same code | Qle p99 2.5e-3, Qh 0.04, Qg 0.046 W/m2; SWE p99 3.2e-5 kg/m2 |
| CONUS 7 days (Noah-MP, 76,180 tiles) | same code | Qle p99 7.5e-3, Qh 0.155, Qg 0.183 W/m2; no NaN tiles |

Bugs the new comparisons found (and fixed): (1) after about 30 h a few Noah-MP tiles went NaN in JAX because `jnp.minimum/maximum` propagate NaN while gfortran's MIN/MAX return the non-NaN argument; fixed with `fmin/fmax`. The history comparison script ignored non-finite values, so it is now reported ("variables with NaN only in the new run"). (2) Three upstream serial-build bugs (no MPI) in HyMAP2 had to be patched in the build copy to run the Fortran reference (`lis_patches/serial_build_fixes_hymap2.diff`); this is a sign that the serial path of upstream LIS is little used.

Speed (details in `BENCHMARKS.md`): CONUS Noah-MP + routing, 2 days: Fortran 1273 s vs JAX 100.6 s on a V100 (12.7x); CONUS 7 days: JAX 123 s vs 4,386 s (projected one-core gpu011) or 6,390 s (measured on the shared login host): 36x to 52x. The compile (83 s) is paid once per process, so the ratio grows with run length. The 4x goal is met in all multi-day cases and for the one-day case with compile included only for the larger domains.

Still not ported: CLSM, Mosaic, JULES, VIC, Noah-MP 3.6 and the other LSMs; other DA observation operators and algorithms (EnKS, particle filters, snow/LAI/TWS assimilation, real satellite observation readers); routing extras (reservoirs, evaporation, DWI, urban drainage, 2-way coupling); ERA5/GDAS/GLDAS forcing readers; LDT and LVT; MPI.

### 9.2 Added on 2026-10-22 afternoon (also DRAFT, unreviewed; see `../STATUS_20261022.md` for the current table)
| Piece | Files | Result vs the real Fortran LIS (GNU 14 build) |
|---|---|---|
| Catchment LSM F2.5 (76,088 CONUS tiles) | `clsm_jax.py`, `lisjax/run_clsm.py`, `lisjax/clsm_tables.py` | full-domain call replay at float32 rounding; 1 day: Qle p99 0.058, Qh 0.11, Qg 0.069 W/m2; maxima on about 0.25% freeze/thaw-sensitive tiles (`clsm_sensitivity_20261022/`); 13.6x on a V100 |
| Noah 3.3 (76,180 CONUS tiles) | `noah39_jax.py` (v33 flag), `lisjax/params.py` | 2 days: Qle p99 0.19, Qh 0.55, Qg 0.82 W/m2 = the spread of a 1e-6 perturbation of the JAX initial state; open-loop replay of real calls at rounding level; 30.1x |
| Mosaic LSM (76,064 CONUS tiles) | `mosaic_jax.py`, `lisjax/run_mos.py`, `lisjax/mosaic_sibalb.npy` | parameters and time-dependent inputs identical to the real LIS; 2 days: Qle p99 2.2e-3, Qh 1.7e-3, Qg 6.7e-4 W/m2; 21.6x |
| 12-member Noah-MP ensemble (CONUS, 1 day) | `run_mp.py` | Fortran 2 h 5 min vs JAX 385 s (19.5x); Qle p99 9.7e-4 W/m2 (mean fields; the spread was not compared) |
| Parallel runner | `gpu/run_variants.sh`, `gpu/summarize_variants.py` | Fortran and JAX on the same exclusive V100 node per case, one summary table |

Still not ported: JULES (the external JULES library is not in the repository), VIC, CLM2, RUC 3.7, Noah 2.7.1, other forcing readers, MPI, LDT, LVT, CLSM and Mosaic restart and DA. Limits of section 7 still apply; tolerances were chosen by the assistant (`ACCEPTANCE_CRITERIA.md` amendment).
