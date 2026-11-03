# Acceptance criteria for the LISF JAX port (DRAFT for owner decision)

Owner: G. Tamkin (to confirm). Drafted by a Claude Code session on 2026-10-21.
Status: **DRAFT, not approved.** These must be approved BEFORE any JAX-versus-Fortran comparison is run. After approval, any change is dated and recorded here, never applied silently after a result is known.
Structure source: `projects/imvi/rocke3d_jax/Reports/ACCEPTANCE_CRITERIA.md` (A/B/C/D categories, the split between port consistency and fidelity, and the "what a result must not claim" section). The numbers below are **placeholders for the owner to set**; none were measured on LISF.
Review by: when the Fortran baseline and its noise floor have been measured (phase P1).

Scope: the first target in `GOAL.md` section 2 (recommended: Noah 3.9, retrospective mode, one public test case). Replace "the model" below with whatever the owner confirms.

## 1. What "JAX-driven" means

A run counts as JAX-driven only if all of these hold; each is reported as a line in the result file:

1. **State updates:** every prognostic variable the model advances (for a land model: soil moisture and temperature layers, canopy water, snow state, and so on, as the model defines them) is updated by a JAX function, with arrays staying on the JAX device between time steps. A Python or NumPy loop that only calls JAX kernels and copies back does not qualify.
2. **Non-JAX stages:** any stage that runs as NumPy, Python, or Fortran is listed by name with its share of run time.
3. **Recorded inputs:** every input taken from a Fortran record instead of being computed (forcing, parameters, initial state) is listed with its source file and size. Forcing and parameters are expected to be recorded; anything else needs a justification.
4. **Third-party pieces:** any external library called from the model path (listed in phase P2) is stated, with how it is handled (ported, called through a callback, or recorded).

## 2. Two comparisons, reported separately

- **C1, port consistency (optional):** JAX against a NumPy or plain-Python transcription, if one is made as an intermediate step. Skipped if the port goes straight from Fortran to JAX.
- **C2, fidelity:** JAX against the original Fortran model built in phase P1, from the same protected start state and forcing.

Only C2 supports a statement about LISF.

## 3. Gate for one time step

Categories, applied per prognostic and per output field to the maximum absolute difference as a fraction of the field's scale:

- **A** bitwise equal.
- **B** at most 1e-12 of scale. (placeholder; owner to confirm)
- **C** at most 1e-6 of scale. (placeholder; owner to confirm)
- **D** worse.

Proposed rule: **met** if every field is in A or B, except a short named exception list (at most 10 cells per field, each at most 1e-9 of scale, listed with coordinates). **Partly met:** runs end to end with named exceptions in B or C and the first failing routine identified. **Not met:** otherwise. Reported per start state, not pooled.

Math library: bitwise results depend on the compiler's math library (`exp`, `pow`, and similar). The baseline records compiler, flags, and math library; if the JAX side cannot match them, the result is reported at B or C and says so.

### 3a. Proposed tolerance (written 2026-10-21 for the owner; owner said "a tolerance is fine, help me define it")

Plain-language idea: the JAX answer may differ from the Fortran answer by no more than the arithmetic noise that already exists between two correct builds of the same Fortran. Anything larger needs an explanation. The size of that noise is not known for LISF yet, so the rule is a procedure, not a guess.

1. **Precision.** Fortran `real` in LIS builds is most likely 32-bit (`GOAL.md` section 5c, not confirmed for a specific build). The JAX port must use the same precision as the Fortran baseline for the comparison (enable 64-bit in JAX only if the baseline is 64-bit). If the baseline is 32-bit, single-precision rounding is about 1e-7 of a value, and the thresholds below for B and C scale up accordingly; the 1e-12 figures in section 3 apply only to a 64-bit baseline.
2. **Noise floor, measured in P1.** Run the Fortran baseline several times from start states perturbed by one unit in the last place, and with a second compiler optimization level. The largest spread per field after one step and after the full day is the noise floor `N`.
3. **Pass level (target)** per field: JAX difference at most `max(10 x N, rounding level of the precision)`.
4. **Ceiling** per field: the largest difference the owner would accept without further explanation. Proposed ceilings, in the field's own units, chosen to be far below any scientific meaning (owner to adjust): soil and canopy temperature 1e-3 K; soil moisture 1e-6 m3/m3; snow water equivalent 1e-6 m; surface fluxes 1e-3 W/m2. A result between target and ceiling is "partly met" with the field named; above the ceiling it is "not met". These ceilings are proposals and were not derived from any LISF measurement.
5. **Branch-flip exceptions.** Freezing, snow and glacial branches can flip in a few cells from tiny differences. Allowed: at most 0.1% of columns per field may exceed the ceiling, each listed with coordinates and the branch that flipped, and none may exceed it by more than 100x. Larger counts mean "not met".
6. **Scale.** Differences are reported both in the field's units and as a fraction of the field's range over the test case.

### 3b. Amendment 2026-10-21 (written after the noise-floor measurement, BEFORE any JAX-versus-Fortran comparison)

Measured on the Fortran baseline (`baseline/noise_floor_report.txt`, `baseline/noise_floor.json`; 2,000 synthetic columns, 48 half-hour steps, float32, ifort 19.1.3 strict):

- Builds at `-O0`, `-O2`, `-O3` (all with `-fp-model strict`) gave **bitwise identical** output in all 2,000 columns, and a re-run was identical. Compiler optimization level adds no noise.
- Five members with the start state moved by one float32 unit in the last place (t1 and the soil temperature, total and liquid moisture) differ from the base run, with heavy tails. Examples for the worst member of five: the day-maximum difference is 0.53 K for surface temperature `t1`, 0.84 K for `stc1`, 5.2e-3 for liquid moisture `sh2o1`, 1.8 W/m2 for latent heat `eta`, 6.6 W/m2 for sensible heat `sheat`; the 99th percentile of cell-step differences is far smaller (for `t1`, 4.9e-4 K). Most of the spread comes from a small number of columns, consistent with threshold branches (freeze/thaw, snow, surface exchange iteration).
- So the fixed ceilings proposed in 3a item 4 (for example 1e-3 K) are **below the baseline's own noise in its worst columns** and cannot be used. They are withdrawn.

**Replacement rule (the ensemble-envelope gate).** For each output field, compute the distribution of |JAX - Fortran| over all cell-steps of the day, at percentiles 50, 90, 99, 99.9 and 100. The field passes if each percentile is at most `2 x` the same percentile of the largest of the five one-ulp members (stored in `baseline/noise_floor.json`, key `member_max`), or at most `1e-6 x` the field's range, whichever is larger. The factor 2 and the 1e-6 floor are proposals by this session, not measured values. A field with member spread exactly zero (for example the vegetation fraction and leaf area index) must match to within the 1e-6 range floor.
- **Met:** every field passes. **Partly met:** all fields pass at percentiles 50 to 99 and some fail only at 99.9 or 100; the failing fields and columns are listed. **Not met:** any field fails at 99 or below.
- Sanity check of the gate itself, required before use: running the gate with a Fortran one-ulp member as the "JAX" run must give "met".
- Items 1, 2, 3 and 5 of 3a stand. Item 4 (fixed ceilings) is withdrawn. Speed claims remain separate (section 6).

This amendment is the only change; it was made before any JAX run existed, which is why it is allowed here.

## 4. Gate for a multi-step run

Land columns do not feed back on a chaotic atmosphere in an offline run, so a pointwise comparison may be possible; that is untested. Proposed procedure:

1. In phase P1, measure the Fortran noise floor: run the baseline from perturbed (one-ulp) copies of the start state and record how far trajectories drift.
2. If the drift stays below a tolerance the owner sets, the multi-step gate is pointwise: JAX versus Fortran stays within that tolerance (owner to set; placeholder none) at every step of the window.
3. If the drift grows (chaotic behavior), use the statistical rule from the Rocke3D criteria: the JAX difference must stay within the spread of the Fortran members.

Window (placeholders): first one day, then one month, then one year of the public test case. Open loop first (forcing and parameters recorded), then free-running.

Always reported: the number of recorded inputs, the compiler and math library of the baseline, and the hardware.

## 5. What a result must NOT claim

- It must not claim to reproduce any scientific dataset or published LIS result unless that exact comparison was run.
- It must not claim a model other than the confirmed first target is ported.
- It must not call a run "end-to-end JAX" if any item of section 1 fails; it says which items fail.
- It must not claim agreement better than the measured noise floor justifies.

## 6. Performance report (separate from correctness)

Reported with every correctness result: wall time per step (first call with compilation, and steady state), device and core count, number of cells, and whether a compile cache was used (not allowed for validated runs). Speed is never a reason to loosen a section 3 or 4 criterion.

## 7. Owner decisions so far (2026-10-21, G. Tamkin, in chat; not yet a signed approval of this file)

- The multi-step gate window is **one simulated day**. Longer windows (month, year) are not required for "done".
- The purpose is speed; section 6 is therefore a primary report, but still never loosens sections 3 and 4.
- A reduced-scope port is an acceptable deliverable. A result must then state the scope (model, configuration, cells, day) and list what is not ported (add to section 5: it must not describe a reduced-scope port as the full model).
- Baseline compiler and hardware are meant to be those of the Rocke3D mentor session (details in `GOAL.md` section 5a: ifort 19.1.3, libimf, forest204 CPU, V100 GPU nodes). Whether LISF can actually be built with that compiler is unresolved (netCDF Fortran 90 module problem, `GOAL.md` section 5a).

- Speed goal: 4x over the current system, defined provisionally in `GOAL.md` section 7. It is a performance claim, reported under section 6, and passes only if the correctness gate in sections 3 and 4 passed for the same run.
- First model: Noah 3.9, chosen by this session at the owner's request (`GOAL.md` section 5c).

## 8. Open points for the owner

1. Approve or change the A/B/C/D thresholds (section 3).
2. Approve the pointwise-first approach for the one-day window, conditional on the measured noise floor (section 4), and set the tolerance.
3. Name the public test case and the day.
4. State whether recorded forcing and parameters are acceptable inputs (section 1 item 3) or whether the forcing readers must also be ported.
5. Name the speed target, if any (for example speedup over the Fortran baseline on the same hardware).

## 9. Results against these criteria (2026-10-21, recorded by the Claude Code session; criteria above unchanged)

Full detail in `REPORT.md`. Summary:
- Section 1 (JAX-driven): state updates are JAX functions on device arrays inside one jitted scan; no non-JAX stage in the day loop; recorded inputs are forcing, parameters and the initial state (all synthetic forcing/state, real table parameters); no third-party library on the model path. SFCDIF_OFF is ported, not recorded. Reduced scope: no LIS wrapper/I/O, no glacial or UA branch.
- Section 3b gate, free-running day, float32, V100: not met. 2,000 columns: 45 pass / 1 partial / 1 fail; 20,000: 43 / 3 / 1; 262,144: 45 / 1 / 1. The single failing field in every run is `dew` (q99 5% to 23% over the limit). One-step (open-loop) agreement is at float32 rounding level apart from a few near-freezing cells.
- Section 6 (speed): steady-state day on V100 is 146x faster than one Fortran core and 12.6x faster than twelve cores at 262,144 columns; 29x and 4.0x at 20,000 columns; 3.0x vs one core at 2,000 columns. Including compilation the single-day run is faster only at very large size (2.0x at 262,144 columns).
- Section 5 (must not claim): this does not show that LIS Noah 3.9 as a whole is reproduced; the reference is the standalone Fortran physics with synthetic forcing.

## Amendment 2026-10-22 (assistant-chosen, NOT approved by the owner)

In the autonomous phase the owner delegated tolerances to the assistant. Applied in hindsight to the results so far (not fixed before the runs): compare per output variable the median, p99 and maximum difference with the real LIS output; call a case reproduced when p99 is at rounding level or below 0.1% of the variable's range, forcing diagnostics are bitwise equal, and no cell is NaN only in the JAX run. Maxima are reported but not gated (see `STATUS_20261022.md` section 3 and `Reports/clsm_sensitivity_20261022/`). The placeholder numbers above remain unapproved.
