# Documentation index

| Document | What it is |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | Running at-a-glance board of workstreams and answered questions (start here) |
| [PORT_ASSURANCE.md](PORT_ASSURANCE.md) | Gates and ratification levels for trusting a port; scorecard |
| [STATUS_20261022.md](STATUS_20261022.md) | What is ported and validated, with open items |
| [LOG.md](LOG.md) | Dated engineering log (historical; paths in it are the old `jax-port` layout) |
| [reports/](reports/) | Benchmark and comparison tables (`BENCHMARKS.md` is the main one) |
| [slides/](slides/) | Briefing decks, owner-meeting deck, port-status table (PDF + build scripts) |
| [DATA_AND_PATHS.md](DATA_AND_PATHS.md), [REPRODUCE_NOAHMP.md](REPRODUCE_NOAHMP.md) | Where run data lives (outside the repo) and how to reproduce |
| [README_jax-port_original.md](README_jax-port_original.md) | README of the original repository, kept for reference |
| project-brief, template-setup, deployment, handoff | DSG project records (template) |

## Path map from the original `jax-port` repository
Older documents (LOG, BENCHMARKS, STATUS, slides) cite old paths. Mapping:

| Old | New |
| --- | --- |
| `noahmp50_jax.py`, `noahmp401_jax.py`, ... (repo root) | `src/lisf_noahmp_jax/physics/` |
| `lisjax/` (`python -m lisjax.lis`) | `src/lisf_noahmp_jax/driver/` (`python -m lisf_noahmp_jax.cli`) |
| `lis/` (replay harness) | `tools/replay/` |
| `lis_patches/`, `baseline/`, `bench.py`, `compare.py`, `run_case.py` | `tools/...` |
| `gpu/`, `gpu_runs/` | `scripts/slurm/`, `docs/reports/gpu_runs/` (logs not committed) |
| `Reports/`, `status_slides/` | `docs/reports/`, `docs/slides/` |
| `run` (symlink to `lisf_run`) | removed; see DATA_AND_PATHS.md |
