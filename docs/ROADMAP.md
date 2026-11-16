# Roadmap at a glance (running; newest work first)

Last updated: 2026-11-05 (independent review done; MPI bug is optimisation-dependent). Owner: G. Tamkin. Everything below is work from 2026-11-02 onward; earlier history is in `LOG.md`, `STATUS_20261022.md`, `Reports/BENCHMARKS.md`.

## Status board
| # | Workstream | Status | Next step | Where |
|---|---|---|---|---|
| 1 | MPI baseline (fair speed comparison) | DONE | 60 d: serial 1 core 43,900 s; MPI 40 cores 2,207 s; JAX 1 V100 255 s. JAX is 172x vs 1 core, **8.7x vs 40 cores**. Customer told 172x on 2026-11-03: send the clarification | `lisf_run/conus_mp5_mpi/`, `gpu/gpu_conus_mp5_mpi_scaling.sbatch` |
| 2 | Max differences for the client | DONE (table below) | Add a tile map of where the maxima sit, if asked | `Reports/cmp_conus_mp5_60d_fix1_20261025.txt` |
| 3 | Independent reproduction | SIMULATED TESTER DONE 2026-11-05: accept with conditions (Level 1 for named cases) | Fix the conditions below; then a human run (owner/teammate) using the same task list; Discover run still to do | `docs/reports/independent_test_20261105.md` |
| 4 | NCAR/noahmp 5.2.1 (fixes that change results first) | STARTED | Add dump hook to the 5.2.1 build, replay, port the default-path fixes | `projects/LISF/noahmp-ncar-jax/PLAN.md`, `dev/lisf_build_552/` |
| 5 | Owner-meeting follow-up | DECK DONE | Decide repo access for the owner; chase the five asks | `status_slides/owner_meeting_20261102.pdf` |
| 6 | Standard confidence-building protocol | DRAFT v0.1 | Owner review of gates/thresholds; add balance checks (G5) and configs into repo (G8) | `PORT_ASSURANCE.md` |
| 7 | Refactor into `lisf-noahmp-jax` (dsg-template layout) | DONE; 48 tests pass in new layout; pushed (private) | Freeze `jax-port` (no new work there), continue only here; decide license and `uv.lock`; give the owner access when agreed | https://github.com/nasa-nccs-hpda/lisf-noahmp-jax |

## Next steps (prioritized; updated 2026-11-03)
| Pri | Step | Who | Done when |
|---|---|---|---|
| 1 | Send the customer the speed clarification (172x vs 1 core, 8.7x vs 40-core MPI); draft is in the chat history / item 1 | G. Tamkin | Sent |
| 2 | MPI vs serial agreement: 20 ranks bitwise identical over 60 d (240 files); bad rank blocks only at 32 and 40 ranks | DONE for the baseline | Optional: find cause of the high-rank bad blocks (check block properties, ESMF/forcing read) |
| 3 | Independent reproduction: owner or teammate runs the 48 tests and the 60 d CONUS case from `lisf-noahmp-jax` | Owner / teammate | One external run recorded (needs repo access, see 4) |
| 4 | Repo housekeeping: freeze `jax-port`; give owner access to the private repo; pick a license; `uv.lock` generated 2026-11-11 (uv 0.13.7); fix old-path references in historical docs and slurm scripts | G. Tamkin (decisions), assistant (edits) | CI green on `main`; no `jax-port` paths in live docs |
| 5 | NCAR/noahmp 5.2.1: apply dump hook to `dev/lisf_build_552`, run 5.2.1 LIS, replay, port the default-path fixes as a variant (MPVER=52) | Assistant | G1 replay and G3 closed loop pass for 5.2.1 |
| 6 | Port assurance v0.2: owner review of gates and thresholds; add balance checks (G5) and configs (G8) | Owner + assistant | Level 1 ratification of the 5.0 port |
| 7 | Known gap: 148 CONUS water-soil tiles unstable in warm season (input data); warm-season cases beyond Feb-Jul 2017 | Assistant | Fix or documented mask; one summer case validated |
| 8 | Larger MPI baseline (multi-node) if customers ask for a bigger-core comparison | Assistant | Time at 80+ ranks recorded |
| 9 | Close the test gaps found by the independent review: 4 of 9 mutations of `noahmp50_jax.py` pass all tests (snow-layer release, snow liquid fraction, ice conductivity +4.5%, latent heat -0.4%) | Assistant | Add a committed deep-snow / frozen-soil replay (Mar 1 restart case) and per-variable limits near float32 noise; the 4 mutations must then fail 2026-11-10: new `noahmp50_snow` replay test (849 deep/wet layered-snow records, limits fixed before mutation runs) catches M1 (marginal: TG 7.3e-4 vs 7e-4), M3 and M6; M8 (SNLIQMAXFRAC halved) was not caught by it; 2026-11-11: new `noahmp50_wetsnow` replay test (360 records from the 17/24 Mar dumps with layer liquid fraction 0.2-0.4 and at the cap; limits fixed before the mutation run) CATCHES M8 (SWE p99 error 6.4 mm vs limit 4e-5). All of M1, M3, M6, M8 now fail a test; M1 is still marginal (TG 7.3e-4 vs 7e-4). Other 5 of the original 9 mutations were not re-run. Runner: `tools/mutation/run_mutations.py`. |
| 10 | 5.0 closed-loop flux comparison (Qle/Qh/Qg, energy balance) against Fortran; pre-registered numeric thresholds; ensemble noise floor | Assistant | Fortran 5.0 reference with flux output; conditional p99 on snow/frozen tiles | 2026-11-10: DONE for 60 d CONUS (240 file pairs): all fluxes pass thresholds fixed beforehand (p99 Qle 0.04, Qh 0.49, Qg 0.63 W/m2); energy-balance residual matches Fortran (p99 0.02 W/m2); maxima up to 425 W/m2 examined 2026-11-11 (`flux_outliers_mp5_60d_20261111.md`: 650 of 18M cell-times, thin near-freezing snow, Qh/Qg split; replay of the Mar 6 18Z event from Fortran pre-state matches to 0.002 W/m2: sensitivity, not a physics error; 5 Swnet outlier cells replayed the same way, also match; restart-resumed Fortran differs from the original in 9-17% of cells at 18Z Mar 17/24, cause open); noise floor not done. `docs/reports/flux_compare_mp5_60d_20261110.md`.
| 11 | Provenance (G8): `uv.lock` (needs uv), configs and a small public dataset with reference outputs in or beside the repo, CI running a fast replay subset | G. Tamkin / assistant | 2026-11-11: `uv.lock` committed; `uv sync --locked --extra test` works and all 49 tests pass in that clean environment (27 min, CPU); CI rewritten (fast job ~2.5 min: CLI + LAI/LDT/snow/glacier replays; weekly full suite; container build) and Dockerfile uv pinned to 0.13.7, NOT yet run on GitHub (no Docker here), Python 3.12 matrix leg untested; example case `examples/sgp_5d` (5 d SGP, own forcing, Fortran reference, 41 MB, test + CI step) added 2026-11-11 |
| 12 | MPI bad blocks at 32+ ranks in OUR GNU 14.2/MPICH build: -O0 clean; -O1/-O2 bad; not deterministic. 2026-11-10: R's Intel build is bitwise clean at 20, 32 and 40 ranks on the same NetCDF reader path (reports/intel_xcheck_20261110.md), so the defect is in our build, not in LIS | Assistant | Optional: rebuild GNU + OpenMPI / Intel MPI and with unpatched sources to find the culprit (MPI library, ESMF or our patches). No report to the LIS team needed. Low priority: the 12.1x claim uses the clean 20-rank run |

## 1. MPI baseline
- Why: the 172x compares one V100 with one serial Fortran core. Real LIS runs on many cores; a fair claim is "N times faster than N cores".
- Plan: MPICH (GNU 14) -> ESMF with MPICH -> MPI copy of LIS -> run 5.0 CONUS, 5 days at several core counts on the same Xeon node type -> full 60 days at the best count -> report time and speed-up vs JAX.
- Expect the headline ratio to fall. That is the point. It fell from 172x to 12.1x (20 cores, verified output) and 8.7x (40 cores, output not clean).

| Run (Noah-MP 5.0 CONUS, 60 d) | Hardware | Wall time | JAX speed-up |
|---|---|---|---|
| Fortran, serial | 1 core | ~43,900 s (12.2 h) | 172x |
| Fortran, MPI | 40 cores, 1 node | 2,207 s (36.8 min); output wrong in 2.7% of tiles | 8.7x |
| Fortran, MPI | 20 cores (4x5) | 3,092 s (51.5 min); output bitwise identical to serial | **12.1x** |
| JAX | 1 V100 | 255 s | 1x |

MPI scaling, 5 days: 1 rank 55:31, 40 ranks 2:52 (19.4x), flattening above ~32 ranks. Details: `Reports/BENCHMARKS.md` (last section).
- Status 16:20: MPICH, ESMF-MPI and the MPI copy of LIS (`dev/lisf_build_mpi`, pinned 5.0, clean driver, `-DSPMD`) are built (first compile failed for lack of `-DSPMD`; fixed, rebuilt 14:44). Scaling job 37950488 started 16:16 on a 40-core Xeon Gold 6248 node (same type as the serial and JAX runs); first run (4 ranks) is in progress. Results go here and in `Reports/BENCHMARKS.md`.

## 2. Max differences, 5.0 CONUS 60 d, JAX vs real LIS (240 history files)
| Variable | Range | p99 | Max | Bitwise equal |
|---|---|---|---|---|
| SWE (mm) | 992 | 0.023 | 14.9 | 62% |
| Snow depth (m) | 3.6 | 0.00014 | 0.14 | 62% |
| Soil moisture | 0.997 | 0.00008 | 0.044 | 2% |
| Soil temperature (K) | 61 | 0.020 | 18.7 | 19% |
| Canopy interception (mm) | 17.9 | 0.00003 | 1.14 | 55% |
| Snow cover fraction | 1 | 0.0003 | 0.999 | 78% |
The maxima are single tiles at snow / freeze-thaw switches (float32 amplification), a handful of ~76,000 tiles. Long run (Apr-May Fortran from JAX restarts): SoilTemp max 5.9 K, SWE max 11 mm. 5.0 output has no flux variables; latent heat max 84 W/m2 (range 571) is from the 4.0.1 CONUS 89 d run.

## 3. Credibility of the speed numbers
- Fair points: one GPU vs one serial core; only one person has run it; best numbers are the long cases (range 2.4x to 172x).
- Safe wording: "172x vs one CPU core and 8.7x vs a 40-core MPI run, one 60-day CONUS case, one V100." Always quote both. Fair claim vs verified parallel LIS: 12.1x (20 ranks, bitwise-clean output); 8.7x vs 40 ranks (output not clean).
- Actions: item 1; ask the teammate or owner's side to reproduce the 60-day run and test suite (48 tests).

## 4. NCAR/noahmp 5.2.1
- Key finding: LIS 7.8's Noah-MP 5.0 is a submodule of the NCAR repo pinned at 0cd33858 (Apr 2024). Our `noahmp50_jax.py` already matches that pin. So only changes since then are new work.
- Delta: 50 files, +9,224 lines: SNICAR, AR24 compaction, AR25 snow cover, wetland (new options); bug fixes that change default results (snow thickness, canopy interception, snow layers, radiation/albedo).
- Decision: fixes that change existing results first; new options later.
- Done so far: separate build copy `dev/lisf_build_552` with 5.2.1 physics swapped in; it compiled (exit 0) and the binary contains the new SNICAR code. Not run yet.
- Next: apply the dump hook, run SGP/CONUS day, replay with JAX, port fixes as an MPVER=52 variant.

## 5. Owner meeting
- Two-slide deck: what was done / what we ask of them (review, reproduce, decide, provide data, name a maintainer). Footer key defines SWE, p99, tile.
- Open: the repo (nasa-nccs-hpda/lis) is private; decide how the owner gets access. Decks rebuilt with dveg 7-9 result.

## Answered questions (for reuse)
| Question | Short answer |
|---|---|
| Starting URL | github.com/NASA-LIS/LISF (clone at 95ae9791a); Noah-MP physics from github.com/NCAR/noahmp via submodule |
| CPUs / GPUs | V100 on gpu006/009/011/013; Xeon Gold 6248 host, Fortran on one core; global cases CPU-only |
| Single JAX file? | Physics yes, one file per Noah-MP version (5.0: `noahmp50_jax.py`, 2,768 lines); driver/params/tools separate |
| Benchmark from repo? | SGP year follows `docs/public_testcase_walkthrough/step_2.adoc`; 5.0 CONUS runs are our own, configs in `dev/lisf_run/` |
| SWE, p99 | snow water equivalent; 99th percentile of the JAX-minus-Fortran difference |
