# Data, environments and paths a new developer needs (2026-10-22)

Owner: G. Tamkin. Written by a Claude Code session from the paths used in the runs of 2026-10-21 and 2026-10-22. Project-local. Status: DRAFT. Absolute paths were replaced by environment variables on 2026-11-17 (list below).

## 1. What is outside the repository

| Variable (export before use) | What it is | Value |
|---|---|---|
| `LISF_RUN` | case directories (`conus_clsm1d`, `conus_hy`, `conus_mp7`, ...), input links, NLDAS-2 forcing, LDT outputs, and the Fortran binaries `LIS_clsm_dump2`, `LIS_mp_hy` | your case directory (the author's was a private directory) |
| `LISF_ENV` | script that sets up GNU 14.2 and the libraries (ESMF mpiuni, NetCDF, HDF) for running the Fortran binaries | `$LISF_DEPS/env_lis.sh`, with `LISF_DEPS` = directory of the GNU 14.2 dependency build |
| `CONDA_BIN`, `JAX_ENV` | conda executable and the JAX environment (`graphcast-env`: jax 0.5.3, Python 3.10) | path of your conda executable (on Nexus, the anaconda module) |
| (not read by the runner) build copy of LIS and LDT with patches and dump hooks | `LISF_BUILD` = directory holding `{lis,ldt}`; rebuild with `lis_patches/env_lis.sh` and `make` in `lis/make`; restore the hooks with `\cp -f` from the repo before rebuilding |
| public LIS data | parameters (`PARAMETERS/clsm25_parms`, `HYMAP_parms/NLDAS_12.5KM`, `mos_parms`, `VIC`, ...) | `/css/lisdata-pub/data` |

Built-in knowledge for each case: CLSM needs `conus_clsm/out/lis_input.conus.clsm.nc` (made by LDT with `conus_clsm/ldt.config`); HyMAP2 needs `conus_hy/out/lis_input.conus.hymap.nc`; `conus_clsm1d/INPUT` and `out` must be working links (a dangling link made a job fail silently once).

## 2. What each variant reads

* `pair_clsm_1d`: `conus_clsm1d` (links to `conus_clsm/INPUT` with NLDAS-2 FORA for Feb 2017, and `out/`), `LIS_clsm_dump2` with the dump hook disabled by `JAXDUMP_CLSM_MAXSTEP=0`.
* `pair_hymap_2d`: `conus_hy` and `LIS_mp_hy`.
* `jax_mp_7d`: `conus_mp7` and its stored Fortran output `lisout`.
* Sizes were not measured. The smallest subset that reproduces one case was not determined.

## 3. Cluster facts the runner assumes

* Nexus ADAPT: partition `compute`, `--constraint=v100`, Tesla V100 nodes (gpu011, gpu012 seen); the login host has no SLURM client, so submit with `SBATCH_HOST=gpulogin1` (ssh).
* Timing pairs need exclusive nodes (the runner asks for them). Node types seen: gpu012 Xeon Gold 6248. Fortran time on a login node is slower than on a compute node; compare only same-node pairs.
* Do not use `pkill -f` with a pattern that matches your own shell; `rm -rf` on globs may be blocked.

Other variables the scripts and tools read (none has a default; the repository holds no machine-specific path): `LISFJAX_REPO` (this repository's root), `LISF_DEV` (parent of your LISF checkouts, used by one CLSM tool), `INTEL_BIN` (Intel compiler `bin` for `tools/baseline`), `F64_SRC` (float64 copy of `src`, `scripts/slurm/gpu_conus_mp5_f64_2d.sbatch`), and on Discover `DISCOVER_WORK` (your scratch directory) and `R_CASE` (the directory of the developer's case). Files under `docs/reports/` and `docs/LOG.md` are run records and keep the paths of the day they were written.
