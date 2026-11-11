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

Limit: the outliers are rare (about 1e-4 of snow-covered cell-times) but not zero, and a user working on thin spring snow would see them. A replay of one event from the Fortran pre-state (dump hook, `replay_mp50.py`) would say whether it is a port error or sensitivity. Not done.
