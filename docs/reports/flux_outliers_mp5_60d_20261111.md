# Flux outlier cells, Noah-MP 5.0 CONUS 60 d (2026-11-11)

Follows `flux_compare_mp5_60d_20261110.md`. Outlier = |JAX - Fortran| > 50 W/m2 in Qh, Qg or Swnet (6-hourly instantaneous, 240 file pairs).

| Finding | Value |
|---|---|
| Events | 650 cell-times of about 18 million (3.6e-5), in 412 distinct cells; median 1 per file, worst file 18 (2017-03-23 18Z) |
| Where | scattered across the northern and mountain CONUS and the Northeast; no single cell dominates (top 20 cells hold 24% of events); mostly cropland, grass/shrub landcover |
| Snow state | Fortran SWE is 0-20 mm in 89% of events (0 mm in 10%, >20 mm in 1%); Tair median 271.8 K, 60% within 268-278 K |
| Time of day | 18Z 57%, 12Z 15%, 06Z 15%, 00Z 13%; 65% have SWdown > 100 W/m2 |
| Type of error | 89% are Qh/Qg: they move in opposite directions (|dQh+dQg| median 31 vs |dQh| median 53 W/m2), i.e. the split between sensible and ground heat flux, not the total, is off. 26% involve Swnet (> 50 W/m2, either sign) |
| Next step in time | e.g. cell (192,215) 2017-03-06 18Z: Qh F55 J128, Qg F92 J14, Swnet F171 J172, SWE 2.7 vs 3.2 mm; the next output is back within ~10 W/m2 |

Reading: these are thin, near-freezing snow cells at the melt/no-melt edge, where the surface temperature and snow-cover fraction solution is sensitive, so a small state difference gives a large instantaneous flux difference that relaxes by the next output. Same cells fit the earlier SWE/SoilTemp drift. This is consistent with the data but not proven: I did not step the Fortran and JAX columns from identical state for these cells, and 86 of 650 events have a SWE difference > 5 mm or snow-cover difference > 0.5, the rest have similar snow state.

## Replay from the Fortran pre-state (2026-11-11)

Cell (192,215), 2017-03-06 18Z (Fortran Qh 55, JAX closed loop 128). Stage 1: real LIS, MPI 20 ranks, Feb 1 to Mar 6 with daily restarts (28.6 min, job 37956056). Stage 2: Fortran dump LIS from the Mar 6 00Z restart, 72 steps of 15 min, snow tiles (job 37956073; it reproduces the original 60-day Fortran exactly at this cell: SWE 2.68 mm, Qh 55.0, Swnet 170.9 at 18Z). Replay: `tools/replay/replay_event.py`, JAX `noahmp50_jax` stepped from the Fortran pre-state at every step.

| Result (all 72 steps, same cell) | max abs difference |
|---|---|
| FSH (Qh) | 0.002 W/m2 |
| SSOIL (Qg) | 0.0008 W/m2 |
| FSA, FIRA | 5e-5, 2e-4 W/m2 |
| TG, TV | 3e-5 K |
| SWE | 7e-9 mm |

At the event step (72): Qh 55.0 / 55.0, Qg 92.5 / 92.5, Swnet 170.9 / 170.9 (Fortran / JAX). So from identical state the JAX physics gives the Fortran answer, including at the outlier step: **not a port error in the physics for this event**. The 128 vs 55 in the free run comes from the state JAX carries into that step (SWE 3.2 vs 2.7 mm).

The Fortran model itself jumps between flux regimes on thin melting snow: steps 62 to 63 Qg 46 to 6, Qh 31 to 78; steps 67 to 68 to 69 Qh 110, 47, 125; step 72 Qh 55 against 127 one step earlier. A free run that sits in the other regime at an output instant differs by 70+ W/m2 with a tiny state difference. Output times are on the hour, and the low-Qh steps 68 and 72 are on the hour, which fits the 57% of outliers at 18Z, but I did not test why.

Limits: one cell, one event; I did not replay other outlier types (Swnet outliers: 26%, cells with SWE differences > 5 mm). The outliers are rare (about 1e-4 of snow-covered cell-times) but not zero, and a user working on thin spring snow would still see them.
