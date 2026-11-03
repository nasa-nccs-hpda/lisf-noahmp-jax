# CLSM F2.5 run recipe (2026-10-22)

Owner: G. Tamkin. Written by a Claude Code session. Project-local. Status: DRAFT.

1. Parameters: LDT (my GNU 14 build) with `dev/lisf_run/conus_clsm/ldt.config` on the public set `/css/lisdata-pub/data/PARAMETERS/clsm25_parms/NLDAS_0.125-deg` gives `out/lis_input.conus.clsm.nc` (76,088 land cells). Needs Surface model types "LSM" only, read-in mask `CLSMF2.5`, landcover `MOSAIC`, albedo data source none.
2. Fortran reference: `LIS_clsm_dump2 -f lis.config` (GNU 14 serial, `source lis_patches/env_lis.sh` first); `JAXDUMP_CLSM_MAXSTEP=0` disables the dump hook.
3. JAX: `python -m lisjax.run_clsm BASE OUT lis.config 2017020100 2017020200` (CPU: `JAX_PLATFORMS=cpu`; 6 h takes 31 s including compile).
4. Compare: `python3 lisjax/compare_hist.py BASE/lisout OUT`; read p99, not max (`Reports/clsm_sensitivity_20261022/`).
5. Core replay: build the dump hooks (`lis_patches/make_clsm_hook.py`, `make_clsm_inter_hook.py`) in the build copy, run with `JAXDUMP_CLSM_STRIDE=1 JAXDUMP_CLSM_MAXSTEP=3`, then `python3 lis/replay_clsm.py DUMPDIR 1,2,3`.
6. Benchmark: `gpu/gpu_bench_clsm.sbatch` or `gpu/run_variants.sh`.
