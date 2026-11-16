# Port assurance protocol: how a port earns ratification

Purpose: one repeatable way to say how far a JAX port of a Fortran model can be trusted, and what is still missing. Draft v0.1, 2026-11-03, written by the port author; it has not been reviewed by the model owners. Their review of the gates and thresholds is part of adopting it.

## Principles (learned on Noah-MP, CLSM, Mosaic)
1. **Fix the reference before testing**: model version/commit, compiler, options, inputs. Record them.
2. **Set pass/fail thresholds before looking at results**, per variable, written in the report.
3. **Judge float32 by p99, report max with an explanation.** Single tiles at snow/freeze-thaw thresholds amplify rounding; a max is not a verdict by itself.
4. **Know the noise floor.** Perturb the inputs by one float32 step; port-vs-reference differences should look like that.
5. **A port is validated only for what was run.** Untested options, seasons, regions, forcings are listed as NOT VALIDATED, not left implicit.
6. **Authorship is not verification.** Checks run by the author count toward Validated; Ratified needs an independent party.
7. **Speed claims need a fair baseline** (same hardware class, parallel reference, steady state vs including compile).

## The gates
| Gate | Question | Evidence required | Pass rule (set in advance) |
|---|---|---|---|
| G0 Scope | What exactly is the reference? | Version/commit pin, build flags, option list, input files, config files kept in the repo | Complete and reproducible from the repo |
| G1 One-step replay | Is the physics the same from identical state? | Fortran dump of state before/after each step, JAX replay on all tiles, per-variable error table | p99 at float32 level (e.g. <= 1e-5 of range) for every output; max attributed to named tiles |
| G2 Option coverage | Which options are checked? | Matrix: option x (replayed, closed-loop, not ported) | Every ported option replayed; every gap listed |
| G3 Closed loop | Does it hold over time? | Free run vs reference, days to months, 50+ outputs; masked tiles listed with cause | Drift p99 within threshold; no systematic bias; failures reproduced in both models or explained |
| G4 Noise floor | Is the remaining difference just chaos? | One-ulp perturbation run (and tile overlap of worst offenders) | Port-vs-reference p99 not larger than perturbation-vs-reference p99 |
| G5 Sanity | Are conservation and bounds respected? | Water/energy balance residuals, NaN/Inf and range checks over the run | Residuals no worse than the reference's |
| G6 Independent check | Would someone else get the same result? | Second person re-runs G1+G3 on their system; model owner reviews the physics diff | Reproduced within thresholds; review findings closed |
| G7 Performance | Is the speed claim fair? | Reference serial and parallel on same node type; JAX steady-state and total; 3 repeats; stated cost | Claim worded exactly as measured |
| G8 Provenance | Can it be rebuilt? | Tests in repo, scripts, data pointers, report tables, version of every tool | Clean checkout reproduces tables |

## Ratification levels
| Level | Meaning | Gates |
|---|---|---|
| 0 Experimental | Runs; some comparisons | none required |
| 1 Validated for named cases | Matches the reference for the cases and options listed | G0-G5, G8 (author-run) |
| 2 Reviewed | Owners reviewed physics and thresholds | Level 1 + part of G6 (review) |
| 3 Ratified | Independently reproduced; claims fair | Level 2 + G6 reproduction + G7 |

## Scorecard (as of 2026-11-03; Noah-MP 5.0 row updated 2026-11-17; honest, author-assessed)
| Port | G0 | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | Level |
|---|---|---|---|---|---|---|---|---|---|---|
| Noah-MP 5.0 (LIS 7.8 pin) | yes | yes | most options; gaps listed | 60 d independent, 150 d partly from JAX restarts | partial (one-ulp, 46-50/200 overlap) | yes (2026-11-17): NaN/range/energy on the SGP example, water balance on replay, both no worse than Fortran | no | serial and 20/40-core MPI (one run each) | partial: configs in `configs/`, example and replay tables reproduce from a clean checkout; CONUS tables need external data and our Fortran build | 1 (cases listed; G8 partial) |
| Noah-MP 4.0.1 | yes | yes | yes | CONUS 89 d | one-ulp done | not systematic | no | serial only | partial | 1 |
| CLSM F2.5 | yes | yes (76,088 tiles) | partial | 1 day | chaos shown | not systematic | no | serial only | partial | 0-1 |
| Noah-MP 5.2.1 deltas | pin chosen | no | no | no | no | no | no | no | no | not started |

## Per-port report template (copy into Reports/)
1. Reference pin and build; 2. options in/out of scope; 3. thresholds (fixed before runs); 4. G1 table; 5. G3 table with max and p99; 6. noise-floor result; 7. balance/sanity result; 8. who reproduced it; 9. speed claim with baseline; 10. NOT VALIDATED list.

## What would move the current ports up
- Level 2: owner reviews `noahmp50_jax.py` against the pinned NCAR code and agrees the thresholds.
- Level 3: a second person reproduces the 60 d CONUS run and tests; parallel LIS baseline (in progress); fix the water-soil tiles in the inputs.
- Closed 2026-11-17: balance/NaN/range checks (G5: `driver/sanity.py`, `tests/test_sanity_g5.py`, run in `tests/test_example_sgp.py`) and configs in the repo (G8: `configs/`; absolute paths replaced by environment variables, see `docs/DATA_AND_PATHS.md`). Still open for G8: the CONUS tables cannot be rebuilt from the repository alone (data and Fortran builds live outside). G5 limits: the JAX history has no TWS or instantaneous runoff, so water balance is checked per step on replay data, not over closed-loop runs.
