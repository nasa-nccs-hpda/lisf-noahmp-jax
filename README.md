# lisf-noahmp-jax

JAX ports of land surface models from NASA's Land Information System Framework (LISF, Fortran), with the Noah-MP family (3.6, 4.0.1, 5.0) first. Each port is checked against the real Fortran LIS (GNU 14, serial) on the same inputs. A research prototype, not a replacement for LIS.

**What it is not:** not all of LIS (no MPI, no LDT/LVT beyond what Noah-MP needs, two forcings, no VIC/CLM2/RUC/JULES). Not independently reproduced or reviewed by the model owners yet (see [docs/PORT_ASSURANCE.md](docs/PORT_ASSURANCE.md)).

## Status (2026-11-03)
- Noah-MP 5.0 (the NCAR snapshot LIS 7.8 uses), 4.0.1, 3.6; Noah 3.x, Mosaic and CLSM F2.5 are ported. Details and open items: [docs/STATUS_20261022.md](docs/STATUS_20261022.md), [docs/ROADMAP.md](docs/ROADMAP.md) (running at-a-glance board).
- Speed: Noah-MP 5.0, CONUS, 60 days: 255 s on one V100 vs about 43,900 s on one Fortran core (172x). Against 20-core MPI LIS with bitwise-identical output (3,092 s) it is 12.1x; against 40 ranks (2,207 s, output not clean) 8.7x. Tables: [docs/reports/BENCHMARKS.md](docs/reports/BENCHMARKS.md).
- Agreement (same run): p99 of JAX-minus-Fortran SWE 0.023 mm, soil temperature 0.021 K; single-tile maxima are larger (snow/freeze-thaw thresholds).

## Run
The validated environment is Python 3.10 with JAX 0.5.3, numpy and netCDF4 (the `graphcast-env` conda environment on the NCCS cluster). With uv: `uv sync --locked` then `uv run --locked lisf-noahmp-jax -C CASE_DIR CASE_DIR/lis.config.jax`. Without uv: `pip install -e .` or `PYTHONPATH=src`.

```bash
PYTHONPATH=src python -m lisf_noahmp_jax.cli -C CASE_DIR CASE_DIR/lis.config.jax     # run from a LIS-style config
PYTHONPATH=src python -m lisf_noahmp_jax.driver.compare_hist FORTRAN_OUT JAX_OUT      # per-variable p50/p99/max vs the Fortran output
PYTHONPATH=src python -m pytest tests -q                                              # 48 tests, about an hour on CPU
```
Run data, forcing and the Fortran build are outside the repository: [docs/DATA_AND_PATHS.md](docs/DATA_AND_PATHS.md), [docs/REPRODUCE_NOAHMP.md](docs/REPRODUCE_NOAHMP.md).

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

See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for working rules. The Apache-2.0 license is the template's; the license for code derived from NCAR Noah-MP is an open question (docs/project-brief.md).
