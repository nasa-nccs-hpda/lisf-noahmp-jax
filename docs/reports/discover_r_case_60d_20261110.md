# R's Noah-MP 5.0 NLDAS-2 case on Discover: 60-day JAX runs (CPU and A100) vs R's Fortran output

Date 2026-11-10. Case: R's `lis.config.nldas2.Basic.OL` and `lis_input.nldas2.noahmp401.OL.nc`, only the end date (2008-11-30) and output directory changed; 2008-10-01 to 2008-11-30 (60 days), coldstart, 3-hourly history (480 files), forcing read natively from the NLDAS-2 GRIB files. Reference: R's LIS 7.8 Intel run (60 ranks), `OUTPUT_Basic_OL/netcdf`. Repo at the commit of 2026-11-10 (GRIB reader 85e66f6 and later). venv2: jax 0.5.3, numpy 2.5.4, pygrib 2.1.8.

## Runs
| Run | Job | Hardware | Wall time | Breakdown |
|---|---|---|---|---|
| JAX CPU | 59224676 | Discover `mil` node, 16 CPUs, `JAX_PLATFORMS=cpu` | 17 min 55 s | setup 8.7 s, forcing read 171 s, device 880 s (first chunk with compile 61 s, then about 1.7 s per 3 h chunk), output 3.5 s |
| JAX A100 | 59224826 | 1 x A100-SXM4-40GB, 16 CPUs, `JAX_PLATFORMS=cuda` | 4 min 13 s | setup 12 s, forcing read (GRIB, 8 processes) 104 s, device 122 s (first chunk with compile 113 s, then 0.02 s per chunk: about 8.5 s for the other 479), output 6.9 s |
One run each, no repeats. R's Fortran wall time for the same days was not measured here (R's job log shows only CPU time), so no ratio against R's run is given. The A100 total is dominated by compilation and forcing reading, not by the model compute. The V100 CONUS run (255 s, NetCDF forcing, different case and season) is not a controlled comparison.

## Agreement with R's Fortran (`compare_hist.py`, 480 files, 76,180 tiles)
| Variable | Range | CPU p99 | CPU max | CPU bitwise % | A100 p99 | A100 max | A100 bitwise % |
|---|---|---|---|---|---|---|---|
| SWE_inst | 591 | 6.9e-4 | 7.5 | 82.7 | 1.6e-3 | 5.4 | 82.5 |
| SnowDepth_inst | 2.16 | 5.8e-6 | 3.3e-2 | 82.7 | 1.2e-5 | 3.3e-2 | 82.4 |
| SoilMoist_inst | 0.962 | 6.5e-6 | 5.2e-2 | 54.4 | 3.8e-5 | 3.2e-2 | 1.5 |
| SoilTemp_inst | 59.4 | 3.7e-3 | 9.04 | 64.4 | 5.5e-3 | 9.60 | 27.6 |
| SmLiqFrac_inst | 0.519 | 4.7e-5 | 5.5e-2 | 54.4 | 7.6e-5 | 5.5e-2 | 1.5 |
| CanopInt_inst | 9.8 | 8.8e-6 | 0.44 | 68.0 | 1.3e-5 | 0.46 | 67.8 |
| Snowcover_inst | 1 | 2.9e-5 | 1.0 | 86.6 | 6.1e-5 | 1.0 | 86.3 |
| Wind_f_inst | 23.6 | 4.8e-7 | 1.9e-6 | 92.5 | 9.5e-7 | 1.9e-6 | 83.3 |
| Tair, Qair, Psurf, SWdown, LWdown (_f_inst) | | 0 | 0 | **100** | 0 | 0 | **100** |
No cell is NaN only in the JAX runs (both). p50 is 0 for the CPU run for every state variable; on the A100 SoilMoist and SoilTemp p50 are 1.0e-5 and 3.1e-5 (GPU rounding differs from CPU and Fortran; bitwise share of SoilMoist 1.5% against 54% on CPU).

Reading: forcing identical for all 60 days; typical (p99) state errors are small (SWE 7e-4 mm on a range of 591, soil temperature 4e-3 K, soil moisture 6e-6 on CPU), and 2 to 6 times larger on the A100. The maxima are single tiles: SWE 7.5 mm, SoilTemp 9 K, Snowcover flipping 0 to 1, CanopInt 0.44 mm, consistent with threshold behaviour in snow and freeze-thaw tiles as in earlier runs; not examined tile by tile for this case. One CPU-run cell differs by 0.495 in Greenness_inst (p99 0, bitwise 100.00% rounded): not examined. For comparison, our own CONUS 60-day case (Feb-Apr 2017, GNU Fortran) had all-tile p99 SWE 0.023 mm and soil temperature 0.021 K; the cases differ in season (October-November has little snow), domain period and Fortran build, so the numbers are not a ranking.

Limits: one period of 60 days, one coldstart, October-November only; only the variables in R's output list (no fluxes); R's run reaches only 2010-01-19; one run per configuration.
