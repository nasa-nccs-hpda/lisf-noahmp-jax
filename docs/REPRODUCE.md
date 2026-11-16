# Reproducing the checks

Each section is independent. Set the environment variables of `DATA_AND_PATHS.md` where a script needs them.

## 1. The example (needs only this package; about 2 minutes on CPU)
See `examples/sgp_5d/README.md`. Fails if any variable's p99 error against the shipped Fortran output exceeds its limit, or if the balance checks (NaN, range, energy residual) are worse than the Fortran's: `uv run --locked --extra test python -m pytest tests/test_example_sgp.py -q`. Passed on Nexus (CPU) and on Discover (CPU suite, A100 example).

## 2. The test suite (about 30 minutes on one CPU core)
`uv run --locked --extra test python -m pytest tests -q -rs`. Replays recorded one-step calls of the real LIS (Noah-MP 5.0 options, land ice, crop, irrigation, snow cases; 4.0.1 options), per-step water balance, the example. Limits and what is not covered: `tests/README.md`. The GRIB reader test needs `--extra grib`; tests needing the author's raw LDT data skip.

## 3. Your case on Discover (GPU or CPU)
Needs your case directory (`R_CASE`: config, parameter file) and a work directory (`DISCOVER_WORK`). Set up the environment on the login node (compute nodes have no PyPI): `bash scripts/discover/tarball_check_setup.sh` is a template. Then `sbatch -C rome scripts/discover/run60d_a100.sbatch` (A100, 253 s) or `sbatch scripts/discover/run60d_cpu.sbatch` (16 cores, about 18 min); `scripts/discover/mkcfg.py` rewrites your config's end date and output directory. Compare with `python src/lisf_noahmp_jax/driver/compare_hist.py <your LIS output> <JAX output>`. Results: `reports/discover_r_case_60d_20261110.md`. Wall time of your own Fortran run is not measured; please send it.

## 4. CONUS 60 days (needs data and a Fortran build outside this package)
Configs: `configs/conus_mp5_60d/`. Jobs: `scripts/slurm/gpu_conus_mp5_60d_*.sbatch` (submit from the repository root, `mkdir gpu_runs` first). Needs the CONUS parameter file made by LDT, the NLDAS-2 forcing, and LIS 7.8 built with GNU 14.2 (`tools/lis_patches/` has the patches for the build copy). Compare with `compare_hist.py` and `tools/compare/flux_compare.py`. Fluxes, outliers: `reports/flux_compare_mp5_60d_20261110.md`, `reports/flux_outliers_mp5_60d_20261111.md`.

## 5. What would count as failure
Any p99 above the limits in `tests/test_example_sgp.py`; NaN in JAX where the Fortran is finite; a balance residual worse than the Fortran's; on your case, SWE or soil temperature p99 far above the values in the README table.
