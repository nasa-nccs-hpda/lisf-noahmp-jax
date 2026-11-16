# lisf-noahmp-jax

JAX ports of land surface models from NASA's Land Information System Framework (LISF, Fortran), with the Noah-MP family (3.6, 4.0.1, 5.0) first. Each port is checked against the real Fortran LIS (GNU 14, serial) on the same inputs. A research prototype, not a replacement for LIS.

**What it is not:** not all of LIS (single-process JAX, no MPI; no LDT/LVT beyond what Noah-MP needs, two forcings, no VIC/CLM2/RUC/JULES). Not independently reproduced or reviewed by the model owners yet (see [docs/PORT_ASSURANCE.md](docs/PORT_ASSURANCE.md)). Also: Noah-MP 5.0 is compared with Fortran in closed loop on state variables only (fluxes only in one-step replay); 148 CONUS water-soil tiles are unstable in the warm season in both models and are masked; the MPI LIS used as speed baseline gives wrong output at 32 or more ranks (20 ranks is clean). The independent review (`docs/reports/independent_test_20261105.md`) rates the port Level 1 for the named cases only.

## Status (2026-11-03)
- Noah-MP 5.0 (the NCAR snapshot LIS 7.8 uses), 4.0.1, 3.6; Noah 3.x, Mosaic and CLSM F2.5 are ported. Details and open items: [docs/STATUS_20261022.md](docs/STATUS_20261022.md), [docs/ROADMAP.md](docs/ROADMAP.md) (running at-a-glance board).
- Speed (one case: Noah-MP 5.0, CONUS, 60 days, JAX 255 s on one V100; Fortran = LIS 7.8, GNU 14.2, our build, run once each). Quote all three baselines together:

  | Baseline | Wall time | JAX speed-up | Fortran output |
  |---|---|---|---|
  | 1 core | about 43,900 s | 172x | reference |
  | 20-core MPI | 3,092 s | **12.1x (the fair claim)** | bitwise identical to 1 core |
  | 40-core MPI | 2,207 s | 8.7x | wrong in 2.7% of tiles (fault in our build, open) |

  Tables: [docs/reports/BENCHMARKS.md](docs/reports/BENCHMARKS.md).
- Agreement (same run): p99 of JAX-minus-Fortran SWE 0.023 mm, soil temperature 0.021 K; single-tile maxima are larger (snow/freeze-thaw thresholds).

## Run
The validated environment is Python 3.10 with JAX 0.5.3, numpy and netCDF4 (the `graphcast-env` conda environment on the Nexus cluster). Try it first: `examples/sgp_5d/README.md` (a 5-day case that runs in about a minute without `/panfs`, with the Fortran reference). Reproducible install: `uv sync --locked --extra test` (uses `uv.lock`; add `--extra gpu` for CUDA), then `uv run lisf-noahmp-jax -C CASE_DIR CASE_DIR/lis.config.jax`. With pip: `pip install ".[test]"` then `lisf-noahmp-jax -C CASE_DIR CASE_DIR/lis.config.jax`. or `PYTHONPATH=src`.

```bash
PYTHONPATH=src python -m lisf_noahmp_jax.cli -C CASE_DIR CASE_DIR/lis.config.jax     # run from a LIS-style config; add --hours 1 to write hourly history files (default 6, the config's output interval is ignored)
PYTHONPATH=src python -m lisf_noahmp_jax.driver.compare_hist FORTRAN_OUT JAX_OUT      # per-variable p50/p99/max vs the Fortran output
PYTHONPATH=src python -m pytest tests -q                                              # 48 tests, 28-52 min on one CPU core (see tests/README.md)
```
Environment variables: `JAX_PLATFORMS=cpu` (force CPU), `LISJAX_FORCING_PROCS` (forcing reader processes), `LISJAX_ULP_TAIR=1` (move air temperature one float32 ulp, for the noise-floor test). Skipped tests print their reason with `pytest -rs`. Run data, forcing and the Fortran build are outside the repository: [docs/DATA_AND_PATHS.md](docs/DATA_AND_PATHS.md), [docs/REPRODUCE_NOAHMP.md](docs/REPRODUCE_NOAHMP.md).

## Layout
| Path | Purpose |
| --- | --- |
| `src/lisf_noahmp_jax/physics/` | One column-physics module per model/version (Noah-MP 5.0: `noahmp50_jax.py`) |
| `src/lisf_noahmp_jax/driver/` | LIS-style runner: config, forcing, parameters, model wrappers, history output, comparison tools |
| `src/lisf_noahmp_jax/cli.py` | `lisf-noahmp-jax` command |
| `tools/replay/` | Open-loop replay harness: reads state dumped by an instrumented Fortran LIS and re-runs one step in JAX |
| `tools/lis_patches/` | Patches and hook generators applied to a *build copy* of LISF (never to the upstream clone) |
| `tools/baseline/` | First-phase standalone Noah 3.9 baseline driver and noise-floor tools |
| `scripts/slurm/` | Cluster job scripts (GPU runs, Fortran references, MPI baseline) |
| `tests/` | Regression tests (replay cases with recorded Fortran data under `tests/data`) |
| `docs/` | Roadmap, assurance protocol, status, log, reports, slides, setup/handoff records |

## Upstream references
LISF: github.com/NASA-LIS/LISF (reference clone at commit 95ae9791a). Noah-MP physics: github.com/NCAR/noahmp, pinned by LISF as a submodule at 0cd33858 (Apr 2024); a 5.2.1 update is planned (separate plan in docs). Origin of this code: the `jax-port` repository (history frozen at its last commit; moved here without rewriting history).

See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for working rules. Licence: Apache-2.0 (LICENSE) for the project's own code; the Noah-MP translation is also under the UCAR Noah-MP agreement (licenses/UCAR-Noah-MP-LICENSE.txt: no sale, keep the notice); see NOTICE.
