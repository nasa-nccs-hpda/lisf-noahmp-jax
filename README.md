# lisf-noahmp-jax

Noah-MP 5.0 from NASA's Land Information System Framework (LISF 7.8, Fortran), rewritten in JAX and checked against the real Fortran LIS. A research prototype by one person with an AI assistant; it has not been reviewed by anyone who knows LIS or Noah-MP. Delivered as a tarball.

## Start here (5 minutes)
```bash
uv sync --locked --extra test            # environment pinned by uv.lock (CPU)
cd examples/sgp_5d
uv run --project ../.. lisf-noahmp-jax -C . $PWD/lis.config                       # about 1.5 min on CPU
uv run --project ../.. python ../../src/lisf_noahmp_jax/driver/compare_hist.py reference out
```
`examples/sgp_5d` is a complete 5-day case (Southern Great Plains, cropped NLDAS-2 forcing) with the real Fortran LIS 7.8 output beside it. `docs/REPRODUCE.md` covers the other checks.

## What was found (Noah-MP 5.0, JAX minus Fortran)
| | Result |
|---|---|
| Your Discover case (1 Oct to 30 Nov 2008, 76,180 tiles, 480 files, against the 60-rank Intel LIS) | weather input bitwise identical; p99 snow water equivalent 0.0007 mm, soil temperature 0.0037 K (CPU); 0.0016 mm and 0.0055 K (A100). Worst tiles: SWE 7.5 mm, soil T 9 K, not yet examined |
| Our CONUS case (60 days, Feb to Apr 2017) | p99 SWE 0.023 mm, soil T 0.021 K; fluxes p99 Qle 0.04, Qh 0.49, Qg 0.63 W/m2 against limits 5, 10, 5 fixed beforehand |
| Largest flux differences | 650 of 18 million cell-times over 50 W/m2, all on thin near-freezing snow; replayed from the Fortran state they match to 0.002 W/m2; a 3e-5 K input nudge or float64 gives the same drift |
| Balance checks | NaN, range, energy and water-balance residuals no worse than the Fortran's |
| Speed (CONUS, 60 days, one V100: 255 s) | 172x vs 1 Fortran core (43,900 s); **12.1x vs 20-core MPI** (3,092 s, output verified identical); 8.7x vs 40-core MPI (2,207 s, but that run has wrong output in 2.7% of tiles, a fault in our GNU build) |

Details and every table: `docs/reports/`. One run per configuration; baselines are our own Fortran builds. Your own Fortran wall time on your case is the missing baseline.

## What it is not
Not all of LIS: no LDT/LVT, two forcings (NLDAS-2, MERRA-2), no VIC, CLM2, RUC or JULES, single process. Seasons tested: Feb to Jul 2017, one SGP year, Oct to Nov 2008. 5.0 fluxes are compared on CONUS Feb to Apr only. Gate status and open items: `docs/PORT_ASSURANCE.md`. Earlier work on Noah, Mosaic, CLSM and Noah-MP 3.6 is in the git tag `archive-2026-11-16`, not in this tree.

## Layout
| Path | Purpose |
|---|---|
| `src/lisf_noahmp_jax/physics/noahmp50_jax.py` | the Noah-MP 5.0 column physics (the file to review against the Fortran); `noahmp401_jax.py` shares the driver |
| `src/lisf_noahmp_jax/driver/` | LIS-style runner: config, GRIB/netCDF forcing, parameters, history output, comparison and sanity checks |
| `examples/sgp_5d/` | self-contained case with Fortran reference output |
| `tests/` | replay tests against recorded Fortran calls, example test, balance checks (`tests/README.md`) |
| `configs/` | LIS configs of the headline runs |
| `scripts/discover/`, `scripts/slurm/` | job scripts: your case on Discover, CONUS runs and Fortran references (paths are environment variables, `docs/DATA_AND_PATHS.md`) |
| `tools/` | replay harness, patches for the Fortran build copy, comparison tools |
| `docs/` | reproduce guide, assurance gates, reports, owner slides, handoff record |

## Licence
Apache-2.0 for the project's code (`LICENSE`); the Noah-MP translation is also under the UCAR Noah-MP agreement (`licenses/UCAR-Noah-MP-LICENSE.txt`): no sale, keep `NOTICE` with every copy. Working rules for contributors: `AGENTS.md`, `CONTRIBUTING.md`.
