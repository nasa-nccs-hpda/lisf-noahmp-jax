# Plan: simulate an independent tester of this repo

Purpose: roadmap item 3 (independent reproduction). Before asking a real person (owner, teammate), run a simulated tester who knows nothing of how the port was built, to find what a newcomer cannot reproduce or cannot believe. Status: PLAN, not yet run. Owner: G. Tamkin.

## 1. The simulated tester (persona)
- A land-surface or HPC engineer who knows LIS and Noah-MP but has not seen this work. Skeptical, time-boxed (one working day), told to decide: "Would I accept this port as reproducing LIS, and are the speed claims fair?"
- Starts from a **fresh clone of `lisf-noahmp-jax` only**. No chat history, no memory notes, no `jax-port`, no access to the author. Allowed inputs: the repo, the public LISF/NCAR repos, and the data paths the repo documents.
- Must write down every point where it gets stuck, guesses, or has to read source to continue (the friction log).

## 2. How to run the simulation
- Fresh agent session (no inherited context) in an empty scratch directory; clone via `git clone`, create a new virtual environment from `pyproject.toml` only (Python 3.10, JAX 0.5.x).
- Prompt (verbatim intent): "You are an independent reviewer. Using only this repository, install it, run its tests, reproduce one documented result, audit the headline claims, and try to break it. Report what you could and could not do. Do not ask for help; record each blocker."
- Run once for a CPU-only tester (no V100), and once for a tester with the cluster data paths, so data-access gaps show up.
- Separately record what a tester with no access to `/panfs` could do at all. That is the realistic case for the owner's side.

## 3. Tasks, in order, with pass criteria
| # | Task | Pass if |
|---|---|---|
| T0 | Install, `lisf-noahmp-jax --help`, read README | Works from README alone in under 15 min; no undocumented step |
| T1 | Run the test suite (`pytest tests`, 48 tests, about 28 min) | 48 pass; tester can say what each group checks (tests/README) |
| T2 | Replay tests against recorded Fortran data in `tests/data` | Tester can state what is compared, with what tolerance, and why p99 not max |
| T3 | Reproduce one small documented case end to end (1-day CONUS 5.0 or SGP) and compare to the stored Fortran output | Numbers match `BENCHMARKS.md` within the stated tolerance; commands come from `docs/REPRODUCE_NOAHMP.md` unchanged |
| T4 | Claim audit: take every number in README, ROADMAP and BENCHMARKS speed and agreement tables; find its source file or job | Each number traces to a file; mismatches listed. Specifically: 172x vs 1 core, 12.1x vs 20-rank MPI, 8.7x vs 40 ranks, the 99% / p99 agreement table |
| T5 | Fairness check of the speed claim | Tester independently states what the baseline is, what is in the JAX time (compile), and what is not measured; can say why 40 ranks is not the baseline |
| T6 | Break it: (a) mutate physics (change one constant in `noahmp50_jax.py`) and see which test fails; (b) perturb forcing by one float32 ulp; (c) run an untested option or season | Mutation is caught by at least one test; (b) behaves as documented; (c) is either rejected clearly or listed as not validated |
| T7 | Review against `PORT_ASSURANCE.md` gates G0-G8 | Tester can mark each gate pass / fail / not evidenced and says what evidence is missing |
| T8 | Read the "What it is not" and known issues (148 water-soil tiles, MPI bad blocks at 32+ ranks) | Limits are findable without reading code |

## 4. What to record (deliverable)
`docs/reports/independent_test_<date>.md` with:
1. Verdict per task (T0-T8), with the exact commands and wall times.
2. Friction log: each blocker, guess, or missing document, with a severity (blocks / slows / cosmetic).
3. Claims that could not be traced or reproduced.
4. Test gaps found by mutation (T6a): mutations no test caught.
5. A one-paragraph answer to "would you ratify this port at Level 1?" (PORT_ASSURANCE levels).

## 5. Known traps to watch for (things a newcomer will hit)
- Data lives outside the repo (`lisf_run`, LDT outputs, forcing); `DATA_AND_PATHS.md` must be enough on its own.
- Some scripts still contain old `jax-port` or `lisjax` paths; `replay_mp50.py` needs `MPVER=50` in the environment.
- `uv.lock` does not exist yet; versions are pinned only by ranges.
- The full suite takes about 28 min on one core; the tester may skip it unless told.
- The speed table's MPI rows: 20 ranks is the clean baseline; 32 and 40 ranks have wrong output blocks.

## 6. Fixes to make before running
Fix the stale paths, add the missing README steps, and decide repo access (a real tester needs it, a simulated one needs a clone from the private repo). Anything the simulation finds should be fixed in the repo, not explained in chat.

## 7. After the simulation
Fix blockers, update README and docs, record results in ROADMAP item 3, then send the real owner or teammate the same task list (sections 3 and 4) so the human run is comparable.
