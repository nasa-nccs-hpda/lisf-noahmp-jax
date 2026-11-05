# Independent test report: lisf-noahmp-jax (fresh clone, commit a2c98f0)

Reviewer: simulated skeptical LIS/Noah-MP/HPC engineer, CPU only, no Slurm, no GPU, 2026-11-05.
Environment: `/home/gtamkin/.conda/envs/graphcast-env` (Python 3.10.16, JAX 0.5.3, CPU). I did NOT build an environment from `pyproject.toml` (no uv, no lock file; I only did `pip install -e . --no-deps` into a venv with system site packages, which worked and gave a working `lisf-noahmp-jax` command). Whether the README alone suffices on a machine without this conda env is therefore untested.
Everything ran in `/panfs/ccds02/nobackup/people/gtamkin/dev/indep_test` (clone `lisf-noahmp-jax/`, scratch `t3/`, mutated copies `M*_*/`, logs `res_*.log`, `ext_*.log`, `t1_pytest.log`). The clone itself stayed clean.

## 1. Verdict

**Accept with conditions (as "Level 1, validated for the named cases", not as "reproduces LIS" in general).** What I could re-run (48 tests, a 1-day CONUS Noah-MP 5.0 case against the stored Fortran output, a one-ulp noise run) matches the documented numbers, and the headline speed arithmetic traces to the raw timing files, but the unit tests are too weak to protect the physics (3 of 9 mutations (M3, M6, M8), including the very class of bug the author found later, survive every Noah-MP 5.0 test) and the strongest evidence (60-day CONUS) was not reproducible by me in budget.

Conditions: (1) fix the no-op CLI command and the missing `uv.lock`/CI; (2) add Fortran flux output (Qle/Qh/Qg) to the Noah-MP 5.0 closed-loop reference, since 5.0 flux agreement is only evidenced by one-step replay; (3) add closed-loop snow-hydrology and thermal-constant regression tests (or tighten thresholds) so mutations M3/M6/M8 below are caught; (4) state the speed claim as "255 s incl. ~90 s compile + 8-core forcing reader on one V100 node vs 43,944 s on one core, 3,092 s on 20 ranks, one run each" and never quote 172x alone; (5) put run configs and a small shippable data subset into the repo/DOI so a non-author can run T3 without the author's `/panfs` directories.

## 2. Task table

| Task | Result | Commands (abridged) | Wall time |
|---|---|---|---|
| T0 install, `--help`, README | partial | `pip install -e . --no-deps` in a venv (conda python); `lisf-noahmp-jax --help` OK. But README's `python -m lisf_noahmp_jax.cli --help` prints NOTHING and exits 0 (no `__main__` guard). `python -m lisf_noahmp_jax.driver.lis` works. `uv sync --locked` cannot work: no `uv.lock` in repo | ~35 s install; ~10 min reading |
| T1 test suite | pass | `PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 taskset -c 0 python -m pytest tests -q --durations=30` -> **48 passed in 3114 s (52 min)** (README says ~30 min; host was shared, load 4-9 of 12 cores). `tests/README.md` is an EMPTY file, so "what each group checks" comes only from `test_regression.py` docstrings/comments. Two of the 48 are `test_ldt`, which silently depend on `/panfs/.../lisf_run` (default path in the test); they would be SKIPPED elsewhere, so "48 pass" is really "46 pass + 2 skip" for anyone else | 3130 s |
| T2 replay vs recorded Fortran | pass (understood) | read `tests/test_regression.py`, `replay_cases.py` | 20 min |
| T3 one small documented case | pass (with undocumented step) | see section 2a | 188 s JAX CPU + 10 s compare |
| T4 claim audit | mostly traced | section 4 | 40 min |
| T5 speed fairness | assessed | section 8 | - |
| T6 break it | partial | (a) 9 mutations: section 5; (b) one-ulp run, behaves as documented (2b); (c) unsupported runoff option rejected cleanly (2c) | ~100 min wall, 3 cores |
| T7 gates | done | section 6 | - |
| T8 limits findable | partial | README has "What it is not" and mentions 40-rank output "not clean"; the 148 water-soil tiles and the finding that 5.0 flux output was never compared in closed loop are NOT in the README (only BENCHMARKS/ROADMAP/STATUS/LOG) | 10 min |

### 2a. T3 detail: Noah-MP 5.0, CONUS, 1 day (2017-02-01), 76,180 tiles
Chosen because it is the smallest 5.0 case with a stored Fortran output that the docs point to (`REPRODUCE_NOAHMP.md` section 8 "CONUS 1 day"). Data: `$LISF_RUN/conus_mp5/{lis.config.jax, INPUT -> ../INPUT, lisout (24 hourly Fortran files)}` (read-only use; I copied the config and redirected output to my area).
```
sed 's#^Output directory:.*#Output directory: <mine>/t3/jaxout#' $LISF_RUN/conus_mp5/lis.config.jax > t3/lis.config.jax
PYTHONPATH=src JAX_PLATFORMS=cpu python -u -m lisf_noahmp_jax.driver.lis -C $LISF_RUN/conus_mp5 t3/lis.config.jax --hours 1
PYTHONPATH=src python -m lisf_noahmp_jax.driver.compare_hist $LISF_RUN/conus_mp5/lisout t3/jaxout
```
Result (CPU, 3 cores, 188 s total: 108 s first-chunk compile, then ~3 s per 1-h chunk; BENCHMARKS says V100 95 s): 24 files compared, p99 / max / bitwise%: SWE 2.5e-6 / 0.18 mm / 74%; SnowDepth 3.7e-8 / 2.6e-3 m; SoilMoist 5.2e-7 / 1.8e-3 / 81%; SoilTemp 4.3e-4 / 1.19 K / 90%; SmLiqFrac 2.6e-6; CanopInt 1.4e-6 / 3.3e-3; Snowcover 2.4e-6 / 0.28; forcing fields (Tair, Qair, Psurf, SWdown, LWdown) bitwise identical. BENCHMARKS (GPU run) quotes SWE p99 1.7e-5 (max 0.70), SoilTemp p99 1.6e-3 (max 1.04): my CPU numbers are of the same size or smaller; passes. The 4,259 open-water cells / 27,756 Landcover cells are NaN only in JAX (documented there).
Undocumented steps I had to discover: (i) `--hours 1` is required to produce hourly files matching the stored Fortran output (default 6; docs never say so; without it compare_hist matches nothing useful); (ii) `-C` must be the case dir whose `INPUT` link resolves; (iii) the config's parameter file path `../mp50_dump_conus/ldt_mp5c/out/...` and NoahmpTable path point into the author's directories; (iv) the doc's literal command `sbatch gpu/gpu_bench_conus_mp5.sbatch` points to a path that no longer exists (it is `scripts/slurm/`), and the sbatch writes to `gpu_runs/` which is not in the repo. I could not run the Fortran side (binary `LIS_mp_clean` exists, but running it is a 11 min job writing into the author's tree; not needed because the stored output exists).
Not reproduced: the headline 60-day case (JAX run needs 8 cores + GPU for 4 min; on CPU about 60 d x 4 chunks x 3 s device + 140 s forcing + compile is feasible in principle but needs a 60-day Fortran reference that only exists in the author's tree and a ~1 h CPU run; skipped for budget). I did however read the stored 60-day comparison files and spot-checked them (section 7).

### 2b. T6(b) one-ulp
`LISJAX_ULP_TAIR=1` (hook in `driver/forcing.py`, off by default, not mentioned in README) on the same 1 day (181 s CPU). JAX vs JAX+1ulp: SWE p99 1.8e-5 / max 0.18 mm; SoilTemp p99 4.6e-4 / max 1.07 K; Qh_tavg p99 0.066 / max 8.6 W/m2; bitwise only 0.4-1% for fluxes. JAX-vs-Fortran (2a) is of the same size or smaller (SWE max 0.18 vs 0.18; SoilTemp max 1.19 vs 1.07 K). So on this day the port-vs-Fortran difference is indistinguishable from a 3e-5 K forcing perturbation: behaves as documented. Note that the JAX run writes Qle/Qh/Qg for 5.0 but the stored Fortran output has none, so those are exactly the variables for which no Fortran comparison exists.

### 2c. T6(c) untested option
Setting `Noah-MP.5.0 surface runoff option: 5` -> immediate `AssertionError: this combination of surface and subsurface runoff options is not ported` (clear, early, ~10 s; but it is a bare `assert`, which disappears under `python -O`). Forcing source other than NLDAS-2/MERRA2 -> `SystemExit ... not ported`. Season/region not validated (CONUS summer for 5.0, other regions) is not rejected, only documented in prose (BENCHMARKS), no machine-readable "validated envelope".

## 3. Friction log

| # | Sev | What happened | Fix |
|---|---|---|---|
| 1 | slows | README `python -m lisf_noahmp_jax.cli ...` (and docs/README.md path map) silently does nothing: `cli.py` has no `if __name__ == "__main__"`; exit 0, no output. A newcomer thinks a run succeeded | add main guard, or point README to `...driver.lis` / the installed command |
| 2 | blocks (off-cluster) | Everything past unit tests needs `/panfs/...` data of the author; no data subset, no config files in repo (only `docs/reports/mosaic_*/lis.config`); DATA_AND_PATHS says "smallest subset not determined" | ship configs + a tiny domain (e.g. SGP 1 day, 616 tiles, forcing + reference output) as release asset/DOI |
| 3 | slows | `uv sync --locked` in README and CI, but no `uv.lock`; CI as written fails at first step; CI runs only 2 tests (`test_lai_tracker`), none of the physics replays | generate lock; run a fast replay subset in CI |
| 4 | slows | `tests/README.md` is empty although the T1 pass criterion and docs refer to it | write it (groups, data origin, tolerances, runtime) |
| 5 | slows | Test counts/times inconsistent across docs: 48 tests/28 min (plan), "about 30 min" (README), "30 tests, 31 min" and "46 pass (3,548 s)" (REPRODUCE, BENCHMARKS). Observed 48 / 52 min on a shared host | one number, with host and load stated |
| 6 | slows | `test_ldt` silently skips off the author's machine -> "48 pass" overstated | print a skip reason in the summary; document `LISF_RUN` |
| 7 | slows | `--hours 1` needed to match the stored hourly Fortran reference; not documented. Default 6 silently differs from config's `Surface model output interval: 1hr` printed in the config (the CLI ignores that key) | read the interval from the config or warn on mismatch |
| 8 | slows | REPRODUCE_NOAHMP.md still cites old layout: `gpu/...`, `lis/replay_mp50.py`, `Reports/BENCHMARKS.md`, `lisjax`, `gpu_runs/`; README says `docs/reports`. BENCHMARKS cites `gpu_runs/*.out` job logs (e.g. conus_bench_37944918.out) that are not in the repo (docs/README: "logs not committed") | fix paths; commit the small `.time`/TIMING lines used for the speed table |
| 9 | slows | REPRODUCE_NOAHMP.md section 8 says SGP/1-day recipes are "DRAFT, not followed by a second person"; sections 1-3 assume 1.5 h of machine time and the Fortran binary `LIS_mp_clean`, whose build recipe is in `tools/lis_patches` with absolute author paths | document build with a clean-room recipe |
| 10 | cosmetic | 5.0 SoilTemp p99 quoted as 0.021 K (README), 0.020 (ROADMAP), 2.05e-2 (cmp file) | quote from the file |
| 11 | cosmetic | `RuntimeWarning: divide by zero` in `params_mp50.py:231` on every run (water tiles, SMCREF=0); `numpy.ndarray size changed` warning in tests | mask before dividing |
| 12 | cosmetic | `ULP` hook env var `LISJAX_ULP_TAIR` and `LISJAX_FORCING_PROCS` undocumented in README | list env vars |
| 13 | slows | Mutation testing (my own work) is slow by design: every JAX replay test compiles ~1-3 min on CPU, so even the "fast" 3 tests take ~3 min and a whole 5.0 set 22 min | provide a `pytest -m fast` subset (one compile, shared across tests) |
| 14 | cosmetic | Config comment says option 3 = Schaake96; BENCHMARKS text for the 1-day case calls the option block "runoff 3" while other places say "SIMGM"; hard to know which option set the headline case used without reading the config | print effective option table at run start |

## 4. Claim audit (T4)

Raw files: `$LISF_RUN` = `/panfs/ccds02/nobackup/people/gtamkin/dev/lisf_run`.

| Claim (where) | Source found | Check | Status |
|---|---|---|---|
| Fortran serial, 60 d CONUS 5.0, 43,944 s (BENCH, README "about 43,900") | `$LISF_RUN/conus_mp5_60d/lis.fortran.time`: wall 12:12:24 = 43,944 s; user 43,525 s | exact | traced (file outside repo) |
| JAX 255 s (BENCH, README) | `conus_mp5_60d/jax_fix1.time` wall 4:15.42 = 255.4 s; `jax_fix1.out` TIMING total 252.47 s (setup 2.1, forcing 138.9, device 106.7 incl. 87.3 first chunk, out 4.8) | exact | traced |
| 172x vs 1 core | 43,944 / 255.4 = 172.06 | arithmetic correct. Earlier unfixed run: 277.1 s = 158.6x (BENCH) | traced; see fairness notes |
| 12.1x vs 20-rank MPI (3,092 s) | `conus_mp5_mpi/lis.60d_n20.time` wall 51:32.03 = 3,092.0 s, exit 0; 3,092 / 255 = 12.12 | arithmetic correct (255 s used, not 255.4: 12.11) | traced |
| 20 ranks bitwise identical to serial, 42 variables, 240 files | `docs/reports/cmp_mpi20_vs_serial_60d_20261104.txt` in repo: 42 rows all 100.00%, max 0, "NaN only in new run: none" | I counted 42 rows at 100.00 | traced (in repo) |
| 8.7x vs 40 ranks (2,207 s) | `lis.60d_n40.time` 36:47.37 = 2,207.4 s; 2,207/255 = 8.65 (8.64 with 255.4) | rounds to 8.7 (8.6 if 255.4 used); within rounding | traced |
| 40-rank output wrong in 2.7% of tiles | `cmp_mpi40_vs_serial_60d_20261104.txt`: bitwise 99.84-99.94%, max 1.8e32 / 318 K in a block | consistent | traced (in repo) |
| MPI scaling 5 d: 1 rank 3,331 s, 40 ranks 172 s, 19.4x | `lis.n1.time` 55:31.31 = 3,331 s; `lis.n40.time` 2:51.84 = 171.8 s; ratio 19.39 | exact. Also: 20 ranks 4:16.77 = 257 s (13.0x vs 1 rank, 65% efficiency) | traced |
| Agreement: SWE p99 0.023 mm, SoilTemp 0.021 K (README); table in ROADMAP section 2 | `docs/reports/cmp_conus_mp5_60d_fix1_20261025.txt` (in repo): SWE 2.33e-2 (max 14.9, bitwise 61.96%), SoilTemp 2.05e-2 (max 18.7, bitwise 18.8%), SoilMoist 7.9e-5 (max 4.35e-2, bitwise 2.1%), Snowcover 3.25e-4 (max 0.999) | matches ROADMAP table exactly; README rounds SoilTemp 2.05e-2 to 0.021 (ROADMAP 0.020) | traced; underlying 240-file data in author's dir, I re-derived spot checks (section 7) |
| 48 tests | reproduced: 48 passed | yes, but 2 are author-path dependent; time 52 min vs 28-30 | reproduced |
| "148 water-soil tiles" excluded from long run | BENCH text; masked compare files in repo (`cmp_conus_mp5_long_seg*_masked`) | I did not recompute | partly traced |
| JAX steady state 2.7 s/day (BENCH) | TIMING chunk list: 0.08 s per 6-h chunk device (0.32 s/day) + forcing 138.9/60 = 2.3 s/day + output | consistent: 2.3+0.32+0.08 = 2.7 | traced |
| 150-day result "113x to 140x" (BENCH) | Fortran 121 d = 43,944+22,244+23,107 s; Fortran segments started from JAX restarts, so not an independent run; 790 s = 4 chained JAX segments | arithmetic 89,295/790 = 113 OK; but 140x uses a linear extrapolation of Fortran | partly traced (extrapolated) |
| 1-day CONUS 5.0, Fortran 663.3 s, JAX 95.2 s, 7.0x | `$LISF_RUN/conus_mp5/lis.fortran.time`, `jax.time` exist; I did not parse; my CPU JAX 188 s | agreement numbers reproduced (2a); timing not re-measured | partly traced |
| Raw job logs `gpu_runs/*.out` cited in BENCH header | not in repo | - | not traceable inside repo |
| SGP-year "10x" (4.0.1/5.0), Noah 3.x 8-30x etc. | not audited beyond reading | - | not checked |

Weak spots in traceability: all raw timings live in the author's directories (not the repo); the Fortran serial run was submitted with `--cpus-per-task=2` and no `--exclusive` (sbatch `gpu_conus_mp5_60d_fortran.sbatch`) whereas the JAX job is exclusive on a V100 node; each timing is a single run (no repeats).

## 5. Mutation results (T6a)

Method: copy of the repo, one edit in `physics/noahmp50_jax.py` (or `noahmp50_glacier_jax.py`), `PYTHONPATH=src` of the copy. Fast set = `test_noahmp50` (2 cases) + `test_noahmp50_glacier` (3 tests, ~3 min, CPU, 1 core). Extended = all 19 `-k noahmp50` tests (22 min, 1 core). Baseline unmutated fast set: 3 pass in 180 s.

| # | Mutation | Kind | Fast set (3 tests, ~3 min) | Extended (19 tests, 22 min) | Time to know |
|---|---|---|---|---|---|
| M1 | `HVAP 2.5104e06 -> 2.5004e06` (-0.4% latent heat of vaporization) | physical constant | NOT caught | caught by `runoff[6]` only, marginally: TG p99 5.14e-3 vs limit 5.0e-3 | ~7 min (fast set + first 3 ext tests 222 s) |
| M2 | `SB 5.67e-8 -> 5.68e-8` (+0.18%) | constant | CAUGHT, all 3 (TG 0.0123 vs 0.005; glacier TG 0.060 vs 0.001) | - | 3 min |
| M3 | delete bottom-layer snow release (`SNOW_RET_FAC * dt * qo` in `snowh2o`) | snow hydrology logic | NOT caught | **NOT caught: 19/19 pass** | 22 min, never |
| M4 | canopy snow capacity coefficient `6.6 -> 6.0` (first of two occurrences, line 211) | parameter/logic | CAUGHT by CONUS case (FSH p99 0.49 vs 0.10) | - | 3 min |
| M5 | `VKC 0.40 -> 0.41` | constant | CAUGHT, all 3 (TG 0.26 vs 0.005) | - | 3 min |
| M6 | `TKICE 2.2 -> 2.3` (+4.5% ice thermal conductivity) | constant (snow/frozen soil) | NOT caught | **NOT caught: 19/19 pass** | 22 min, never |
| M7 | `CPAIR 1004.64 -> 1004.74` (+1e-4 relative) | tiny constant | caught only by glacier test (FIRA 1.47e-3 vs 1e-3); both land tests pass | - | 3 min |
| M8 | `SNLIQMAXFRAC` halved in `snowh2o` | snow logic/parameter | NOT caught | **NOT caught: 19/19 pass** | 20 min, never |
| M9 | glacier snow-aging exponent `5.E3 -> 5.1E3` | glacier constant | CAUGHT by glacier test only (TAUSS 5.7e-6 vs 1e-8) | - | 3 min |

Test gaps revealed:
1. The land-column tests use 3 h of CONUS (Feb 1, mostly thin/cold snow) and SGP day 1 (one-step replay from recorded pre-state, ~500 calls). They have almost no deep, wet, multi-layer snow, so snow-pack hydrology (M3, M8) and ice/frozen-soil conductivity (M6) are invisible. M3 is the same class as the real porting error the author found only through a 60-day closed-loop run (`snowh2o` was still the 4.0.1 form; BENCHMARKS: "Earlier replays missed it because thin or cold snow never exceeds the holding capacity"). The unit tests have therefore not been fixed against it: a regression of the fix would not be caught.
2. Thresholds are loose relative to constants: a 0.4% error in HVAP and 4.5% in TKICE pass; M1 is caught only by luck (5.14e-3 vs 5.0e-3 in one non-primary test). Limits are "about 5-10x the measured value", and several were raised after seeing results (test comments: "limits raised after seeing 0.064 W/m2 and 2.2e-3 K"; "measured differences ... not tuned"), which contradicts PORT_ASSURANCE principle 2 (thresholds before results).
3. Glacier tests are much tighter (1e-3) than land tests and caught constants the land tests did not (M7, M9). Tightening the land limits per variable to a few x float32 noise would be cheap.
4. No test exercises the closed-loop driver (`driver/lis.py`, forcing interpolation, output writer) beyond the 2 `lai_tracker` tests; the only closed-loop check is the manual benchmark runs.
5. Only the 5.0 physics module was mutated; the 3.6/4.0.1/CLSM/Mosaic tests were not probed.

## 6. Gate table (T7), PORT_ASSURANCE.md G0-G8

| Gate | Mark | Evidence checked |
|---|---|---|
| G0 Scope | partial (not "reproducible from the repo") | Reference pinned in prose (LISF 95ae9791a, NCAR noahmp 0cd33858, GNU 14.2 -O2) and dump hooks in `tools/lis_patches`; but run configs, LDT outputs, forcing and binaries are outside the repo; hooks have author absolute paths |
| G1 One-step replay | pass for the sample in repo | 48/48 tests pass; replay is on 150-700 sampled calls per case; "all tiles" larger replays are described in LOG/BENCH but I did not rerun them; thresholds are post hoc in part |
| G2 Option coverage | partial | Option x test list exists as test names (runoff 6/7/8, dveg 2,5-9, tdrn 1,2, irrigation, crop 1) and "not ported" assertions; no explicit matrix document; runoff 1-5 fail loudly |
| G3 Closed loop | pass with caveats (not reproduced by me at 60 d) | 60-d independent Fortran vs JAX comparison file in repo and consistent with BENCH; 1-day case reproduced; 148 tiles excluded from long run; 150-d Fortran segments started from JAX restarts; 5.0 reference has NO flux variables; p99 dominated by snow-free tiles (section 7) |
| G4 Noise floor | partial / pass | one-ulp (Tair only) run: 60 d overlap 46-50/200 vs random 0.5-0.7 (BENCH); my 1-day run shows JAX-vs-Fortran <= JAX-vs-JAX+1ulp. Single perturbation type, no ensemble |
| G5 Sanity | not evidenced | no water/energy balance tests, no NaN/range assertions in tests; only "no NaN only in new run" in compare output; BENCH admits 148 tiles go to -13,544 K (JAX) and Fortran aborted at Jun 6 |
| G6 Independent check | not evidenced as defined | this report is a partial independent (author-assisted-environment) run; no model-owner physics review exists (README says so) |
| G7 Performance | partial / fail on "3 repeats, steady vs total" | serial and 20/40-rank MPI measured, JAX total and steady-state stated, each config run once, Fortran serial not exclusive, no repeats, JAX uses 8 host cores + GPU |
| G8 Provenance | fail | no lock file, CI cannot run, configs/raw logs outside repo, stale paths in docs/scripts, CLI doc command broken, tests partly skip off-site |

At the doc's own rule (Level 1 = G0-G5 and G8, author-run) the port does not currently satisfy G5 and G8, so Level 1 is claimed more strongly than evidenced for 5.0; "Level 0.5-1" for the cases listed is my reading.

## 7. Independent assessment: metric and fairness

### Is p99 of |JAX - Fortran| + bitwise% + one-ulp a sound way to judge float32? Partly.
- Good: p99 not max for chaotic threshold models; forcing diagnostics checked bitwise; the 20-rank MPI run being 100% bitwise vs serial proves the Fortran reference is deterministic, so differences are real; the one-ulp comparison with tile-overlap is a discriminating test (it flagged the real snow bug: overlap 3/200 before the fix, 46/200 after).
- Weak: the 60-day p99 is computed over all tiles x all times, including tiles with zero snow. For SWE the difference is 0 on 62% of values (bitwise 61.96%), so the all-tile p99 understates snow-tile error. From the stored 60-day files (`conus_mp5_60d/lisout` vs `jaxout_fix1`, my script `snowp.py`): 2017-02-15 12Z: all-tile SWE p99 3.8e-3 mm; restricted to tiles with SWE>0 in either model (29,172 tiles) p99 1.3e-2, p99.9 0.16, max 3.9 mm. 2017-03-15 12Z: all-tile p99 0.050, snow tiles p99 0.136, p99.9 0.78, max 4.7; 2017-04-01 18Z: all-tile 0.0525, snow tiles p99 0.237, p99.9 0.91, max 4.3. So (a) the headline 0.023 mm is a pooled number; error is about 10x larger by mid-March than mid-February, i.e. it grows with time as expected for chaotic divergence and a single pooled p99 hides it; (b) snow-tile p99 is 2-5x the all-tile p99. Still small in absolute terms (sub-mm), and only about 0.1% of snow tiles differ by more than 1 mm. SoilTemp 2017-03-15: p99 0.045 K but max 9.3 K, p99.9 0.30 K.
- The pass/fail p99 ignores the worst 1% (about 760 tiles in each file). Maxima (SWE 14.9 mm, SoilTemp 18.7 K, Snowcover 0.999, i.e. a tile flipping between snow covered and not) are explained as threshold chaos, but the one-ulp tile test explains only about 25% overlap of the worst tiles, "consistent with chaos, not proof" (author's own words).
- 5.0 closed-loop comparison has state variables only (SWE, SnowDepth, SoilMoist, SoilTemp, SmLiqFrac, CanopInt, Snowcover); no fluxes, so surface energy balance of the 5.0 port in a free run is NOT compared with the Fortran, only one-step FSH/FIRA in replay (limits 0.1 W/m2 on FSH, p99). This is the largest evidential hole for land-surface users, since the JAX writes Qle/Qh/Qg.
- Thresholds: stated in test files, many raised after seeing results; the "hard" success criterion for closed loop (Level 1) is described in text ("p99 at float32 level"), not as a number fixed before the run.
- Bias check: author reports mean SWE bias ~0 and no drift over 2 months; I could see growth of p99 with time.

### Is the speed comparison fair? Reasonably, if worded fully; the headline alone is not.
- 172x compares one V100 (plus 8 CPU cores used by 8 forcing-reader processes; forcing read is 139 of 255 s) with one serial Fortran core. 12.1x vs a 20-core clean MPI run is the fair headline; 8.7x vs 40 ranks is a timing-only number because 2.7% of tiles are wrong in that run, but it is the fastest verified wall time on a node and is the baseline a user would try; it is fair to quote both, as the ROADMAP does.
- JAX time includes ~90 s compile in one 60-day run (cold), so it shortens with run length (steady 2.7 s/day vs Fortran 732 s/day: ~270x per day vs 1 core, ~19x vs 20 ranks (51.5 s/day) and ~14x vs 40 ranks (36.8 s/day) in steady state, by my arithmetic from their numbers). Both are legitimate but a reader should see them.
- Not measured: cost normalisation (energy, dollars, node count: a 20-rank job is half a node, the JAX job reserves a whole exclusive GPU node), I/O contention (shared panfs, one run each), variance (no repeats), MPI scaling flattening and the 5-day vs 60-day scaling differences (the 5-day 20-rank run extrapolates to 3,081 s vs the measured 3,092: consistent). Serial Fortran was not run exclusive.
- Serial binary vs MPI 1-rank binary: 732 s/day vs 666 s/day (5-day n1 = 3,331 s), i.e. the "1 core" baseline itself differs by ~9% between two builds; a 1-core baseline at the MPI-build speed (666 s/day x 60 = 39,960 s) would give ~156x.
- Also both Fortran and JAX are float32; gfortran -O2 (not Intel / -O3 / vectorised); a tuned Fortran build is not tried.
- JAX result depends on the post-hoc snow fix: the first run (unfixed) was 277.1 s and disagreed (SWE p99 4.26 mm). Documented honestly.

### What I would want before ratifying (Level 1 -> 2)
1. A reference Fortran 5.0 run with flux and energy-balance output, and a closed-loop flux comparison (p99 and conditional on snow/frozen tiles); water and energy closure per step.
2. Pre-registered numeric thresholds per variable and condition (snow-covered, frozen, vegetated) and conditional (not pooled) p99/p99.9 with time dependence, plus the exclusion list (148 tiles) fixed in the repo.
3. An ensemble noise floor: several one-ulp perturbations (Tair, Qair, SWdown, wind) in both models (perturbing Fortran too if possible) and a distribution test, not only one JAX-vs-JAX+1ulp.
4. Mutation-resistant tests: closed-loop deep-snow and frozen-soil replay (restart on Mar 1 as the author did, committed as a test), constants tested at ~1e-4 relative sensitivity; M3, M6, M8 must fail.
5. A runnable clean-room path: lock file, working CLI, small public dataset (SGP day + 60-d CONUS subset) with reference outputs and configs; CI running a fast subset.
6. Model-owner (NASA LIS / NCAR) review of `noahmp50_jax.py` against pinned 0cd33858 (README itself says none has happened), additional seasons (summer CONUS for 5.0, other regions), and explanation or fix of the 148 water-soil tiles and the 32/40-rank MPI bad blocks.
7. Speed: three repeats, exclusive nodes for all runs, the same I/O configuration, cost per simulated year, and a steady-state and cold-start pair for each baseline.

## 8. Things I could not verify, and what I would need
- A fresh dependency install from `pyproject.toml` (no network/uv test; conda env used). Need: a machine without the author's conda env.
- The 60-day CONUS result and the 150-day result: need a ~1 h CPU or 5 min GPU JAX run plus the author's Fortran 60-d output (exists in `$LISF_RUN` but I only re-derived spot checks from it, not a full rerun).
- Fortran timings (43,944 s, 3,092 s, 2,207 s): only read from the author's `.time` files; could not rerun (12 h / 50 min jobs on nodes I may not use; no Slurm).
- GPU timings and GPU vs CPU numerics (no GPU).
- Whether the Fortran build used for the reference equals the upstream LISF at the stated commit (I did not diff `tools/lis_patches` hooks against upstream; read-only plausibility only).
- Physics correctness against NCAR Noah-MP source (not reviewed line by line; 2,768-line module).
- The 148-tile claim, the MPI bad-block cause, the SGP-year, Noah 3.x/Mosaic/CLSM claims: not audited.

## 9. Wall-time ledger
Clone 8 s; install 35 s; T1 3,130 s (1 core); T3 JAX CPU 188 s (3 cores), compare 10 s; one-ulp run 181 s; each fast mutation set ~180 s; each extended set 22 min (M1 stopped at first failure after 222 s); reading and analysis ~1 h.

## 10. Inert-or-real check of the surviving mutations (code read)
- M3, M8: `snowh2o` runs for every tile with `isnow < 0` (layered snow); SNOW_RET_FAC / SNLIQMAXFRAC are read from the parameter table (`params_mp50.GLOBALS`), so the edits are live code. Survival means the recorded test calls have no snow layers holding enough liquid to hit the limit: a real test gap.
- M6: `TKICE` enters `thksat` (soil thermal conductivity, line 262) for every soil layer, scaled by saturation: live, but the effect is small for the recorded states: real gap (thresholds too loose for this term).
- M1: `HVAP` is used in canopy and ground latent-heat paths for all tiles; live. Caught only marginally.
