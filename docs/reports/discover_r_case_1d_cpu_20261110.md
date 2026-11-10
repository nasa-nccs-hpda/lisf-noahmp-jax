# R's Noah-MP 5.0 NLDAS-2 case on Discover: 1-day JAX run (CPU) vs R's Fortran output

Date 2026-11-10. Job 59224494 (`scripts/discover/run1d_cpu.sbatch`): Discover `compute`, `mil` node, 16 CPUs, account ilab, `JAX_PLATFORMS=cpu`, venv2 (jax 0.5.3, numpy 2.5.4, pygrib 2.1.8), repo at 14737c6 plus the Discover docs commits. Input: R's `lis.config.nldas2.Basic.OL` and `lis_input.nldas2.noahmp401.OL.nc` unchanged except end date (2008-10-02) and output directory; forcing read natively from the NLDAS-2 GRIB files (`"NLDAS2 grib"`). Reference: R's LIS 7.8 Intel run, `OUTPUT_Basic_OL/netcdf` (60 ranks). Coldstart on both sides, 2008-10-01 00:00 to 2008-10-02 00:00, hourly history.

Wall time 1 min 35 s (includes JIT compilation, 16 CPUs), 3.4 GB resident. One run, no repeats.

`compare_hist.py`, 24 history files (hourly, 76,180 tiles):
| Variable | Range | p99 abs diff | Max abs diff | Bitwise % |
|---|---|---|---|---|
| SWE_inst | 10.3 | 1.1e-8 | 1.2e-4 | 95.74 |
| SnowDepth_inst | 0.099 | 9.3e-11 | 1.0e-6 | 95.75 |
| SoilMoist_inst | 0.852 | 3.3e-7 | 6.9e-3 | 85.23 |
| SoilTemp_inst | 49.3 | 6.1e-5 | 0.177 | 90.44 |
| SmLiqFrac_inst | 0.419 | 3.3e-7 | 6.9e-3 | 85.23 |
| CanopInt_inst | 0.757 | 2.5e-6 | 0.197 | 79.21 |
| Snowcover_inst | 1 | 6.8e-9 | 1.0e-4 | 96.03 |
| Wind_f_inst | 12.3 | 4.8e-7 | 9.5e-7 | 92.84 |
| Tair, Qair, Psurf, SWdown, LWdown (_f_inst) | | 0 | 0 | **100.00** |
| Landcover, Greenness | | 0 | 0 | 100.00 |
No cell is NaN only in the JAX run. p50 is 0 for every variable.

Reading: the JAX port, driven by the GRIB forcing and R's parameter file, reproduces the Intel-compiled Fortran at float32 level on the first day: all five forcing fields are bitwise identical in every hour (so the GRIB reader and the LIS time interpolation of the driver are right for this case), and state-variable p99 errors are at 1e-5 to 1e-8 of range. The maxima (SoilTemp 0.18 K, CanopInt 0.20 on a range of 0.76) are single tiles; their cause (threshold behaviour as in earlier runs, or a difference in a parameter or option) is not yet examined for this case.

Limits: one day, one season (October), coldstart; no flux variables in R's output list of this comparison; one run, so the timing is indicative only. Not yet done: 60 days, then the full window to 2010-01-19, and the A100 timing (job 59224273 pending).
