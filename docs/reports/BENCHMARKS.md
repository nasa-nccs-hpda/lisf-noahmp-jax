# Benchmarks and agreement, Noah-MP 5.0 (CONUS and SGP, real NLDAS-2 forcing)

Kept 2026-11-16: only the Noah-MP 5.0 sections. Results for Noah, Noah-MP 3.6 and 4.0.1, Mosaic and CLSM, and the one-day Noah 3.9 tables, are in the git tag `archive-2026-11-16`. Fortran = real LIS 7.8 (GNU 14.2, our build) unless stated. Scripts named in each section are in `scripts/`; the paths inside them are environment variables (docs/DATA_AND_PATHS.md). p99 = 99th percentile of the absolute JAX-minus-Fortran difference over tiles.

## Noah-MP 5.0, CONUS 0.125 deg (76,180 tiles), 1 simulated day, 2017-02-01 (job 37946423 Fortran + JAX, 37946425 JAX rerun after output fixes; gpu009, `gpu_runs/conus_mp5_*.out`, `gpu/gpu_bench_conus_mp5.sbatch`)
| run | wall time | notes |
|---|---|---|
| Fortran LIS (GNU 14, serial, `LIS_mp_clean`), 1 core Xeon Gold 6248 | **663.3 s** (11 min 3.3 s; user 656 s) | testcase option block (runoff 3, BATS snow albedo, dveg 4), 1-hour history |
| Python/JAX, V100, cold start | **95.2 s** (7.0x) | 90.1 s of the first 6 h chunk is JIT compilation (vegetated + land-ice branches), setup 2.1 s, forcing 2.3 s, output 0.4 s |
| Python/JAX, device compute after compilation | 0.08-0.10 s per 6 h chunk | same as 4.0.1 |

Agreement (24 hourly history files, `lisjax/compare_hist.py`; the 5.0 output list of the testcase holds only these variables): SWE p99 1.7e-5 kg/m2 (max 0.70 at one snow-threshold tile), SnowDepth p99 2.3e-7 m, SoilMoist p99 1.9e-6, SoilTemp p99 1.6e-3 K (max 1.04 K, one freeze/thaw tile), SmLiqFrac p99 1.2e-5, CanopInt p99 3.0e-6, Snowcover p99 7.2e-6 (max 0.16, snow threshold); forcing fields bitwise. The 4,259 open-water cells and 27,756 Landcover cells are NaN only in the JAX files (not written for those cells), as for 4.0.1.
Steady state for 5.0 is measured from the 60-day CONUS run (see below): 2.7 s per simulated day.

## Noah-MP 5.0, SGP 27x22 (0.25 deg, 616 tiles), 365 simulated days, 2017 (job 37946426, gpu009, `gpu/gpu_bench_mp5_sgp_year.sbatch`, `gpu_runs/mp5_sgp_year_37946426.out`)
| run | wall time | notes |
|---|---|---|
| Fortran LIS (Noah-MP 5.0, GNU 14 serial, 1 core) | **2,944.8 s** (49:04.8; user 2,162 s) | testcase option block, 6-hourly history, 1,460 files |
| Python/JAX, V100 (cold, parallel forcing reader, background writer) | **277.3 s** (sum of setup 1.7 + forcing 166.0 + device 101.4 incl. 60.8 compile + output 8.2) | **10.6x**, 90.6% less time; forcing reads are 60% of the JAX time |

Agreement over the year (1,460 history files): SoilMoist p99 6.6e-4 m3/m3, SoilTemp p99 4.4e-2 K (max 9.4 K at one freeze/thaw tile on one date), SWE p99 3.4e-5 kg/m2 (max 12.8 at a snow-threshold tile), SnowDepth p99 4.5e-7 m, CanopInt p99 2.0e-5, Snowcover p99 4.1e-6; forcing fields bitwise; no NaN only in the JAX run. The 4.0.1 year had SoilTemp p99 4.4e-3, i.e. the 5.0 year is about 10x less close on soil temperature. Where it comes from: the open-loop replay of the Fortran 5.0 calls of Jan 3-5 (100,000 calls, snow onset) is at noise level (STC max 2.4e-3 K, SWE p99 0), so it is not a physics difference but chaotic amplification at the first snow events (Jan 4: top layer up to 0.65 K at ~115 cells, the 4.0.1 year has the same pattern, 0.64 K at 151 cells) that then persists in the deep soil layers (Feb 20: layers 3-4 differ by 0.03-0.05 K at 212 of 616 cells, soil moisture by 1e-3) and decays by summer (p99 5e-3 K on Jun 25, 2e-3 K on Oct 3).

## Noah-MP 5.0, global MERRA-2 (30,307 land tiles incl. Greenland), 1 simulated day, 2017-07-01 (`lisf_run/glob_mp5`; LDT for 5.0 rerun with the real LDT, 3 min 47 s)
| run | wall time | notes |
|---|---|---|
| Fortran LIS (GNU 14, serial, 1 core, login host) | 446 s (7 min 26 s) | |
| Python/JAX on CPU only (12 threads, login host) | 182.7 s | compile 147 s of the first 6 h chunk; steady state about 6.9 s per 6 h chunk; GPU not used (2.4x, CPU only) |

Agreement (4 history files): SWE p99 5.6e-9 kg/m2 (max 0.65 at one tile), SnowDepth p99 4.4e-11 m, SoilMoist p99 1.5e-7, SoilTemp p99 6.1e-5 K (max 0.016), CanopInt p99 2.5e-5, Snowcover p99 9e-10; forcing fields bitwise. First 5.0 test of the land-ice branch on real ice sheets (Greenland) and of a non-NLDAS forcing and grid with Noah-MP 5.0.

## Noah-MP 5.0 option coverage added 2026-10-24 (open-loop replay of Fortran calls and short closed loops; no new speed measurement)
These options were validated for correctness only; wall times are those of the default-option 5.0 runs above. All on SGP 27x22 (616 tiles) unless noted; the reference is the real Fortran LIS (GNU 14) or, for irrigation, a LIS build patched to set the land-use name (stock 7.8 never irrigates).
| option | check | result |
|---|---|---|
| surface runoff 6 (VIC), 7 (Xinanjiang), 8 (dynamic VIC/Philip) | replay on a rainy day, closed loop 3 d | noise level |
| dynamic vegetation 2/5/6 (carbon) and 7/8/9 (input LAI) | replay SGP Jul 9, closed loop 3 d (2/5/6/7/8/9); dveg 8 closed loop Jul 14 - Aug 3 (20 d, across day 16 and the month change) | float32 level; 20 d SoilMoist p99 4.5e-6, SoilTemp p99 1.5e-4 K (max 1.7e-2) |
| tile drainage 1 and 2 | replay Feb 14, synthetic TD_FRACTION; closed loop 3 d | SMC max 6e-8 (1.9e-4 / 6.9e-3 without drainage); SoilMoist p99 5.4e-7 |
| irrigation triggers 1, 2, 3; methods map/sprinkler/micro/flood | replay Jul 9 vs patched build | SMC max 1.7e-6; amounts and counts identical |
| dynamic crop (crop option 1, corn) | replay of 536 cropland calls, closed loop 10 d | growth stage exact, GDD 1e-6, GRAIN 3e-8; SoilMoist p99 5.2e-7, SoilTemp 9e-5 K (crop off in JAX: 9.8e-3, 1.0 K) |
Crop types 2-5 (2026-10-25, DEFAULT_CROP=2..5 table copies, same synthetic maps, replay of 15,000 sampled calls each): TV p99 1.5e-4 K, FSH p99 7e-3 W/m2, SMC p99 0, PSN p99 1e-5, PGS exact, GDD p99 6e-7, GRAIN max 2e-8, LFMASS 0-5e-7, RTMASS 0-5e-7, STMASS 1.9e-6 for all four types.
Limits: synthetic parameter maps (TD_FRACTION, PLANTING/HARVEST/SEASON_GDD) and a table copy with DEFAULT_CROP=1 (the stock table makes the crop model inert); irrigation is not in the JAX runner; CROPCLASS>0 domains and GECROS untested; runoff 5 not ported. Regression tests: 46 pass (3,548 s).

## Noah-MP 5.0, CONUS 0.125 deg (76,180 tiles), 60 simulated days (2017-02-01 to 04-02, 15-min step, 6-hourly history, 240 files; Fortran job 37946608, JAX jobs 37946609 and 37946638 (one ulp), 2026-10-24/25; `lisf_run/conus_mp5_60d`, scripts `gpu/gpu_conus_mp5_60d_*.sbatch`, tables `Reports/cmp_conus_mp5_60d_20261025.txt`, `Reports/tile_check_conus_mp5_60d_20261025.txt`)
| run | wall time | notes |
|---|---|---|
| Fortran LIS (GNU 14, serial, 1 core, V100 node) | **43,944 s** (12 h 12 min 24 s; 732 s per simulated day) | measured in full, no extrapolation |
| Python/JAX, V100, cold | **277.1 s** (4 min 37 s) | **158.6x**, 99.4% less time; setup 9.4 + forcing 136.8 + device 110.9 (91.3 compile + first chunk, then 0.08 s per 6 h chunk) + output 4.5 |

**First comparison (before the fix below): agreement was NOT at the level of the 1-day, SGP-year or 4.0.1 results.** Fortran vs JAX over the 240 files (p99 / max): SWE 4.26 / 50 mm, SnowDepth 1.9e-2 / 0.19 m, SoilMoist 6.1e-3 / 0.11, SoilTemp 0.56 / 18.8 K, SmLiqFrac 8.6e-3, CanopInt 2.5e-3 / 2.0, Snowcover 0.175 / 1.0; forcing bitwise (Tair, Qair, Psurf, SWdown, LWdown). The 4.0.1 89-day run had SWE 2.8e-2 and SoilTemp 9.4e-3 at p99.
Where it starts: day 1 at 65 thin-snow cells (SWE < 1 mm, Tair 266-271 K), p99 SWE 1.7e-3 mm on Feb 4, 0.44 on Feb 7, about 4-5 mm from Feb 10 to the end (no growth after that). Net bias is about zero (mean JAX - Fortran SWE -0.5 to +0.2 mm over snow cells) but about 3,000 of 36,000 snow cells (8%) differ by more than 1 mm, mainly evergreen needleleaf (landcover 1: 1,747 of 2,984 on Feb 10), landcover 10 and 12.
One-ulp test (JAX vs JAX with Tair + 1 ulp, same 60 days): heat fluxes respond about as much as Fortran-vs-JAX did in 4.0.1 (Qle p99 3.8e-2, Qh 0.17, SoilMoist_tavg 9.7e-5), but on the 200 worst Fortran-vs-JAX tiles the one-ulp response is tiny: SWE median 0.097 mm against a Fortran-JAX difference of 24.9 mm, SoilMoist 5.6e-4 against 6.5e-2, SoilTemp 0.20 K against 13.9 K; overlap of the worst-200 sets 3 (SWE), 3 (SoilMoist), 6 (SoilTemp) of 200 (random 0.5-0.7). So, unlike in 4.0.1, the worst 5.0 tiles are NOT explained by float32 chaos of this kind; they point to a systematic snow-related difference between the 5.0 port and the Fortran that the earlier replays (Feb 1 snow tiles over 3 h, SGP year) did not exercise (deeper, multi-layer snow packs under forest). Under investigation: open-loop replay of Fortran calls from a deep-snow state (restart of Mar 1), see LOG.md. This was a real porting error, found by this run. Open-loop replay from a deep-snow state (JAX restart of Mar 1 read by the Fortran dump LIS, 4 h of snow tiles, 19,984 calls; `gpu/gpu_replay_conus_mp5_mar1.sbatch`, `lisf_run/mp50_rst_mar1`) gave SWE p99 0.26 mm and SNLIQ p99 0.16 mm per 15-min step, SMC p99 1.4e-3 (energy terms were fine: FSH p99 1e-2, TV 9e-5). Cause: the land snow-layer hydrology in `noahmp50_jax.py` (`snowh2o`) was still the 4.0.1 form; the 5.0 `SnowpackHydrologyMod` has a release-timescale factor SNOW_RET_FAC on the bottom layer, a maximum liquid fraction SNLIQMAXFRAC and a layer-thickness update (already in the land-ice module). Earlier replays missed it because thin or cold snow never exceeds the holding capacity. After the fix the same replay gives SWE p99 7.6e-6, SNLIQ 2.4e-7, SMC 0 (bitwise 99%).
**After the fix** (JAX rerun job 37946640, 255.4 s = 4:15; `docs/reports/cmp_conus_mp5_60d_20261025.txt` is the table before the fix): Fortran vs JAX over 240 files, p99 / max: SWE 2.3e-2 / 14.9 mm, SnowDepth 1.4e-4 / 0.14 m, SoilMoist 7.9e-5 / 4.4e-2, SoilTemp 2.1e-2 / 18.7 K, SmLiqFrac 2.1e-4, CanopInt 2.7e-5, Snowcover 3.3e-4 / 1.0 (was 4.3, 0.56, 6.1e-3, 0.175 before). This is the size of the 4.0.1 89-day run (SWE 2.8e-2, SoilTemp 9.4e-3, SoilMoist 1.3e-4). Speed with the fix: 43,944 / 255.4 = **172x** (99.4% less time; the first, unfixed run took 277.1 s = 158.6x). Fortran time is measured in full.
One-ulp tile check on the fixed code (job 37946641): worst-200 overlap of Fortran-vs-JAX and JAX-vs-JAX+1ulp tiles is 46 (SWE), 50 (SoilMoist), 48 (SoilTemp) of 200 (random 0.7, 0.5, 0.5); rank correlation 0.81, 0.27, 0.84; the one-ulp response on the worst Fortran-vs-JAX tiles is 30-50% of the difference (SWE 1.25 vs 3.88 mm, SoilTemp 1.19 vs 2.79 K). That is the 4.0.1 pattern (chaotic amplification at snow-threshold tiles), whereas before the fix the response was 0.1 mm against 25 mm. Caveats: 60 days (Feb-Mar) only; maxima remain single tiles (SWE 14.9 mm, SoilTemp 18.7 K); the other 5.0 snow paths in the same module (e.g. layered snow combine/divide with very deep packs, April melt-out) were not replayed separately; consistent with chaos, not proof.

## Noah-MP 5.0, CONUS 0.125 deg, Feb 1 - Jul 1 2017 (150 simulated days; JAX job 37946655, Fortran segment jobs 37946656/57/58, 2026-10-25/26; `lisf_run/conus_mp5_long`, scripts `gpu/gpu_conus_mp5_long_*.sbatch`, tables `Reports/cmp_conus_mp5_long_seg*_20261026.txt`)
Design: a continuous Fortran run of 150 d would take about 31 h, so the Fortran ran as three monthly segments in parallel, each started from the JAX restart of the segment start (Apr 1, May 1, Jun 1; JAX itself ran Feb-Apr, Apr, May, Jun chained by restarts). The comparison therefore shows the drift of Fortran from JAX over one month starting from an identical state; it is not an independent Fortran run from February (Feb-Mar is covered by the 60-day run above, which is independent).
| run | wall time |
|---|---|
| JAX V100, four chained segments (Feb-Apr 270 s, Apr 174, May 174, Jun 172) | **790 s** for 150 d (each segment recompiles, about 87 s) |
| Fortran segments, 1 core | Apr 1 - May 1: 22,244 s (6:10:44, 741 s/d); May 1 - Jun 1: 23,107 s (6:25:07, 745 s/d); Jun 1 - Jul 1: ABORTED at Jun 6 01:30 after 3,566 s (see below) |
Speed: Fortran measured for 121 simulated days (Feb 1 - Jun 1: 43,944 + 22,244 + 23,107 = 89,295 s, 738 s/d); against the 790 s JAX run of 150 d that is at least 113x; against Fortran extrapolated linearly to 150 d (110,700 s) it is about 140x. (The 255 s 60-day run is 172x because it recompiles once.)
**Unstable tiles (a finding about the reference as much as about the port).** 148 land cells have soil type 14 (water; LIS prints 'SOIL TYPE FOUND TO BE WATER AT A LAND-POINT' and resets them to sandy clay loam), spread over many land covers. In these tiles BOTH models go unphysical in the warm season: Fortran soil temperature up to 388 K in April and 599 K in May (min 17 K), then the Fortran LIS stopped at Jun 6 01:30 (ground temperature -437 K, 'emitted longwave <0', exit status 0, history ends Jun 6 00:00); JAX reaches 418 K in April and -13,544 K in late May (first cell 2017-05-28 18:00 at lat 29.94 lon -89.81, landcover 11), 1 to 6 cells by June. JAX does not crash but its values in those cells are meaningless; neither model is a valid reference there. Those 148 tiles are excluded in the comparison below (`lisjax/compare_hist_mask.py`); unmasked, SoilTemp max differences are 125 K (April) and 13,900 K (May).
Agreement excluding the 148 tiles (p99 / max): 
| segment (files) | SWE mm | SoilMoist | SoilTemp K | SnowDepth m | CanopInt | Snowcover |
|---|---|---|---|---|---|---|
| Apr 1 - May 1 (120) | 5.9e-3 / 11.4 | 4.4e-5 / 4.9e-2 | 9.8e-4 / 5.9 | 2.4e-5 / 4.8e-2 | 1.4e-5 / 0.42 | 5.4e-5 / 0.99 |
| May 1 - Jun 1 (124) | 2.7e-3 / 151 | 4.4e-5 / 8.8e-2 | 6.7e-4 / 3.4 | 8.4e-6 / 0.35 | 8.2e-6 / 0.37 | 6.1e-7 / 0.52 |
| Jun 1 - Jun 6 (20, Fortran aborted) | 0 / 2.7 | 9.7e-6 / 3.6e-3 | 2.8e-4 / 0.04 | 0 / 7e-3 | 9.6e-6 / 0.11 | 0 / 3.7e-3 |
So April-May melt-out agrees at p99 as well as the winter months; maxima are single tiles (one May snow tile differs by 151 mm). Limits: Fortran state identical to JAX at each segment start by construction; June covered only 5 days; the one-ulp tile check was not repeated for this run; the water-soil-type tiles should be treated as invalid in both models (an input-data problem of the LDT soil map, not a physics result).

**Noah-MP 5.0 CONUS steady state (from the 60-day JAX run of job 37946640, 240 chunks, `jax_fix1.out`).** Device time per 6 h chunk after the first (compile) chunk: median 0.080 s = 0.32 s per simulated day (first chunk 87.3 s). Per simulated day: forcing reads on the host 2.31 s (138.9 s / 60 d), device 0.32 s, output 0.08 s; total about **2.7 s per day** once compiled, against 732 s per day for the serial Fortran = **about 270x steady state** (the wall-clock ratio of the 60-day run is 172x because 87 s of compile and 2 s of setup are included). The forcing reads (NLDAS-2 netCDF on the host, 8 processes) are 85% of the steady-state time, so the GPU is not the limit. Caveat: one V100 node with 8 cores for the readers; Fortran is one core of the same node type, not MPI LIS.

## Speed, three-way comparison: Noah-MP 5.0, CONUS 0.125 deg (76,180 tiles), 60 simulated days (added 2026-11-03)
Same domain, forcing and physics. Fortran = real LIS 7.8, GNU 14.2 -O2. MPI = MPICH 4 + ESMF (mpich), one 40-core Xeon Gold 6248 node.
| Run | Hardware | Wall time (60 d) | Speed-up of JAX over it |
|---|---|---|---|
| Fortran LIS, serial | 1 CPU core | about 43,900 s (12.2 h; 732 s per simulated day, measured) | **172x** |
| Fortran LIS, MPI | 40 CPU cores (one node) | 2,207 s (36.8 min; job 37950501, `lisf_run/conus_mp5_mpi/lis.60d_n40.time`) | **8.7x** |
| JAX | 1 V100 GPU | 255 s (includes about 90 s compile; steady state 2.7 s per simulated day) | 1x |

MPI scaling (5-day case, same binary): 1 rank 3,331 s (55:31); 40 ranks 172 s (2:52); 19.4x faster than 1 rank, flattening above about 32 ranks (earlier 4/8/16/32 rank points in `lisf_run/conus_mp5_mpi/lis.n*.time`).
How to read it: **172x is the speed-up against one core. The fair claim is 12.1x against 20 cores, where the output is verified identical; 8.7x against 40 cores is the least favourable ratio, but that Fortran run has wrong output in 2.7% of tiles (open fault in our build).** The ratio against a larger MPI run (several nodes) would be smaller still; not measured. Each configuration was run once; I/O on shared storage adds noise. Output agreement of the MPI run with serial Fortran was not checked here (speed comparison only).

## MPI (40 ranks) vs serial Fortran, 60 days: agreement check (2026-11-04)
Serial `conus_mp5_60d/lisout` vs MPI `conus_mp5_mpi/lisout_60d_n40`, 240 history files, `compare_hist.py` (full table in the git tag `archive-2026-11-16`).
- **Outside one block the two runs are bitwise identical.** All flux variables (Swnet, Lwnet, Qle, Qh, Qg, rain, snow, runoff) are 100% bitwise equal; no differing cell lies outside the block below.
- **One 8x5-decomposition block disagrees** (columns 0-57, rows 90-134 of the 464x224 grid; x-block 0, y-block 2, i.e. one of 40 ranks, about 2,000 of 76,000 land tiles, 2.7%). It differs in every file from 2017-02-01 on, including the *forcing diagnostics* (Tair_f differs by up to 287 K, mean 25 K in the block; 68 more missing cells in MPI), so the MPI run is not feeding that rank the same forcing or tile mapping. Some states there blow up (soil layer 2 about 5,300 K on 02-01 18Z; max 1.8e32 in later files). The max column of the compare table (1.84e32) comes from this block only.
- Not a simple grid shift (tested +/-6 cells). Cause not found. Suspects: forcing read or tile ordering on that rank in LIS's parallel path, or the 148-tile water-soil instability interacting with that subdomain.
- **What this means for the speed table:** the 2,207 s timing is a valid wall-clock measurement of the MPI run, but that run is not a clean reference for 2.7% of the domain. It does not change the 172x (serial, clean) figure, and 8.7x stands as a timing comparison only until the block is explained. Serial vs JAX agreement results are unaffected.
- Next: rerun the 5-day case with a different decomposition (for example 5x8 or 32 ranks). If the bad block moves with the decomposition it is a parallel-path problem; if it stays at the same geography it is a data or tile issue.

### Decomposition test of the MPI disagreement (job 37950794 plus earlier 5-day runs, 2026-11-04)
Metric: cells where forcing diagnostic Tair_f differs from serial Fortran by more than 0.01 K, summed over the 20 history files of the 5-day case. Script `scripts/slurm/gpu_conus_mp5_mpi_decomp.sbatch`.
| Ranks, layout (x by y) | Bad cells | Where |
|---|---|---|
| 1, 4 (4x1), 8 (8x1), 16 (8x2), 20 (4x5) | 0 | bitwise clean |
| 32 (8x4) | 45,797 | three whole rank blocks: (x0,y1), (x2,y0), (x5,y3) |
| 40 (8x5) | 37,129 | one block (x0,y2); same block in the 60-day run, so repeatable for a layout |
| 40 (5x8) | 57,085 | different blocks |
| 40 (10x4) | 61,725 | different blocks |
**Finding:** the bad blocks move with the layout, and nearly every land cell inside a bad block is wrong (for example 1905 of 2083). All runs with 20 or fewer ranks are clean; all runs with 32 or 40 ranks have bad blocks. This is a parallel-path problem in the MPI LIS run (forcing or tile mapping on some ranks at high rank counts), not a data or physics issue, and not in JAX. Cause not yet found. Consequence: MPI timings at 32 and 40 ranks are valid wall-clock numbers, but those runs are not clean references; the clean 20-rank run (4x5) took 4:17 for 5 days vs 2:52 for the 40-rank run. Fair comparison on clean output: JAX vs MPI at 20 ranks (not yet run for 60 days). Next: run the 60-day case at 20 ranks (4x5) to get a clean MPI reference and timing.

### Clean MPI reference: 60 days at 20 ranks (4x5), job 37950803 (2026-11-04)
Same case as above. Wall time **51:32 (3,092 s)**, exit 0, all 240 history files. Compared with serial Fortran (`cmp_mpi20_vs_serial_60d_20261104.txt`): **all 42 variables are 100% bitwise identical, max difference 0, in all 240 files.** So the MPI LIS itself is correct at 20 ranks; the bad blocks at 32 and 40 ranks are a high-rank-count problem of this MPI build, not of the model.
| Run (Noah-MP 5.0 CONUS, 60 d) | Hardware | Wall time | Output vs serial | JAX speed-up |
|---|---|---|---|---|
| Fortran, serial | 1 core | about 43,900 s | reference | 172x |
| Fortran, MPI 20 ranks (4x5) | 20 cores | 3,092 s | bitwise identical | **12.1x** |
| Fortran, MPI 40 ranks (8x5) | 40 cores | 2,207 s | 2.7% of tiles wrong (one rank block) | 8.7x (timing only) |
| JAX | 1 V100 | 255 s | p99 at float32 level (see above) | 1x |
**Quote 12.1x as the fair speed claim against parallel Fortran with verified output; 172x only as against one core.** The 40-rank run is faster but its output is not trustworthy, so it should not be the baseline.

### Cause of the bad rank blocks: what is established so far (2026-11-04, cause not yet found)
- Deterministic for a layout (8x5 gives the same block in a 5-day and a 60-day run), different blocks for different layouts, none at 20 ranks or fewer. So not random and not one faulty core or node.
- Inside a bad block **every forcing diagnostic holds the same values** (Tair_f, Qair_f, Psurf_f, Wind_f, SWdown_f, LWdown_f, Rainf_f all show min 0, max 293.7, mean 254 K) and 9% of the cells are exactly 0. The model then sees surface pressure of about 280 Pa and blows up. The values are not a time shift (best match is the same time step, still wrong) and not a spatial shift (values do not appear anywhere in the serial field).
- The land mask also differs in the block (2,015 land cells in MPI vs 2,083 in serial: 8 extra, 76 missing), so the block's tile or grid mapping differs, not only the forcing. That points at domain decomposition or the parallel read of the LDT parameter file, rather than the NLDAS-2 file reader alone.
- Config uses "NLDAS2 netcdf" (plugin `metforcing/nldas-2.0`, `read_nldas20b.F90`), neighbor interpolation. The reader keeps `varfield` and `dummy` (about 400 KB) as automatic stack arrays; stack-size test submitted as `scripts/slurm/gpu_conus_mp5_mpi_stack.sbatch`.
- Not yet tested: other forcing plugin; non-optimised build (-O0 with NaN-initialised reals); the LDT parameter read on the bad ranks.

### Stack test result and correction (job 37950826, 2026-11-05)
- The stack limit was already `unlimited` before the script raised it, so this was **not** a test of the stack hypothesis. Hypothesis 1 is still open.
- The repeat runs give different results for the same layout, so the earlier statement "deterministic per layout" was too strong. Bad blocks at one time step (Tair_f, Feb 3 06Z): 8x5 original (0,2); 8x5 repeat (0,2) and (2,0); 8x4 original (0,1), (2,0), (5,3); 8x4 repeat (0,2). The same two areas recur: columns 0-57 / rows about 90-167 (northern California to Nevada area) and columns 116-173 / rows 0-55 (southern border area).
- Reading: an initialisation problem that varies between runs (uninitialised memory or a race in start-up arrays such as the neighbor-interpolation index), plus something data-dependent that favours some areas. The error then persists for the whole run.
- Next, decisive: build the MPI LIS with `-O0 -g -fcheck=all -finit-real=snan -ffpe-trap=invalid` and run 32 ranks for one day; it should stop at the line that uses an uninitialised value. Needs a rebuild of the MPI LIS (`dev/lisf_build_mpi`).

### Fairness caveats added after the independent review (2026-11-05)
- The serial Fortran 60-day job ran with `--cpus-per-task=2` and no `--exclusive`; the JAX job was exclusive on a V100 node and used 8 host cores for the forcing readers (139 of 255 s is forcing). Repeats: one run per configuration.
- The "1 core" baseline differs by about 9% between builds: serial LIS 732 s per simulated day, the 1-rank MPI build 666 s per day (5-day n1 run). Against the MPI-build speed the single-core ratio would be about 156x, not 172x.
- JAX time includes about 90 s compile in a cold 60-day run. Steady state per simulated day: 2.7 s (JAX) vs 51.5 s (20 ranks) = about 19x, vs 36.8 s (40 ranks) = about 14x.
- Not normalised for cost (a 20-rank job is half a node; the JAX job holds a whole exclusive GPU node), I/O contention, or compiler (gfortran -O2, not Intel).

### MPI bad blocks: compiler-flag experiments (2026-11-05)
All runs: Noah-MP 5.0 CONUS, 5 simulated days, 20 history files, 32 and 40 ranks; metric = Tair_f cells differing from serial Fortran by more than 0.01 K (sum over files). Same source and config in every build; only FFLAGS differ. Scripts `scripts/slurm/gpu_conus_mp5_mpi_{debug,debug_notrap,debug_5d,o2snan_5d,novec_5d}.sbatch`.
| Build | 32 ranks | 40 ranks | Note |
|---|---|---|---|
| -O2 (the production MPI build) | 45,797 | 37,129 | bad blocks |
| -O2, repeat runs of the same binary | 54,625 | 32,771 | different blocks each run: not deterministic |
| -O0 -g -finit-real=snan -finit-integer=-99999 -fcheck=bounds, no FP trap | **0** | **0** | clean for 1 and 5 days; no bounds error and no NaN anywhere |
| -O0 with -ffpe-trap=invalid | stops in 5 s | stops in 4 s | SIGFPE in Noah-MP `SurfaceEnergyFluxBareGround` on the ranks that print "SOIL TYPE FOUND TO BE WATER" (the known water-soil tiles); unrelated to the bad blocks |
| -O2 -finit-real=snan -finit-integer=-99999 | 41,384 | 36,182 | still bad: not an uninitialised stack variable |
| -O2 -fno-tree-vectorize -fno-tree-slp-vectorize | 38,910 | 37,662 | still bad: not vectorisation |
| -O1 | 33,451 | 26,570 | still bad: not -O2-only passes |
| -O2 -fno-strict-aliasing | 44,573 | 29,201 | still bad: not strict aliasing |
Per output step the number of bad cells changes (for example 3,248, 6,168, 1,402, ... 0 over the first 8 files), and the bad cells are present from the first history file. So whole rank blocks receive wrong forcing at some forcing reads and not others; this is not a fixed start-up mapping error.
Reading: the bug needs optimisation (-O1 and -O2 code or the timing it produces), is not deterministic, is not an uninitialised stack variable or vectorisation, and -fcheck=bounds finds no out-of-bounds access in the executed code at -O0. Remaining suspects: a data race or message-ordering problem in the parallel forcing path that only shows with fast ranks, an uninitialised heap (ALLOCATE) array, or an optimiser-exposed invalid-Fortran construct. Next probes: -O1; -O2 with `-fno-strict-aliasing`/`-fno-inline`; running the -O2 binary under valgrind or with a different MPICH progress setting; compare the forcing arrays on a bad rank directly (print `metdata2` checksums per forcing read).
Forcing-checksum probe (production -O2 build plus per-rank checksums in `read_nldas20a`, patch in `tools/mpi_debug/`; 32 and 40 ranks, 5 days, still bad: 36,732 and 28,721 cells). Findings, 32 ranks: (1) the raw NetCDF data are bitwise the same on every rank for all 122 reads, so the file read is not the fault; (2) the output block of rank 8 (8x4 layout) is wrong from the third history file on (about 1,400 cells, all its land cells), yet that rank's interpolated forcing sums look normal and change smoothly from read to read; (3) so the corruption happens after the read and spatial interpolation: time interpolation (`timeinterp_nldas20`), the forcing state hand-off, or later. Only the checksums of the read were instrumented; not yet checked: the values after `timeinterp_nldas20` and after the downscaling step.

Intel cross-check (2026-11-10, `intel_xcheck_20261110.md`): R's Intel-built LIS is bitwise identical at 20, 32 and 40 ranks (0 differing cells in 120 files each, NetCDF reader path): the fault is specific to our GNU 14.2 / MPICH build or its patches, not to LIS.

Practical consequence: the 20-rank -O2 baseline is clean in all tests (60 days bitwise identical); results at 32 or more ranks with the optimised build must not be used. An -O0 build is clean at 40 ranks but about 1.5x slower (5 days: 4:22 vs 2:57).

## 5.0 closed-loop flux comparison 2026-11-10 (CONUS, 60 d, 6-hourly instantaneous, JAX V100 job 37955747)

All six fluxes pass limits fixed before the run (`tools/compare/flux_compare.py`): p99 |diff| Qle 0.038, Qh 0.49, Qg 0.63, Swnet 0.011, Lwnet 0.12 W/m2, Evap 1.5e-8 kg/m2/s; snow and frozen-soil subsets also pass (frozen Qg 2.6, Qh 2.2). Energy-balance residual agrees with Fortran (p99 0.02 W/m2). Maximum differences are large in a few cells (Qh 425, Qg 432, Swnet 338 W/m2) and were not examined; state drift found earlier (SoilTemp p99 0.56 K) is the likely cause. No noise-floor run. Details: `docs/reports/flux_compare_mp5_60d_20261110.md`.
