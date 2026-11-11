# Tests

Run (CPU, one core; JAX compiles on every case, so it is slow):
```
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest tests -q -rs
```
48 tests. Measured 28 min on an idle node and 52 min on a shared one (independent run, 2026-11-05). Use `-rs` to see skip reasons.

| File | What it checks | Reference data | Limits |
|---|---|---|---|
| `test_regression.py` | One-step replay: the JAX column physics (Noah 2.7.1, 3.3, 3.9; Noah-MP 3.6, 4.0.1 and option variants, 5.0 incl. runoff/dveg/drainage/irrigation/crop variants, glacier; CLSM; Mosaic) reproduces 150-700 sampled calls recorded from the real Fortran LIS | `tests/data/` (recorded inputs and Fortran outputs, built by `make_test_data.py` with the dump hooks in `tools/lis_patches`) | p99 absolute error per field, about 5-10 times the value measured on 2026-10-22 (see the dicts at the top of the file) |
| `test_regression.py::test_noahmp50_snow` | Deep, wet, layered snowpack (849 records, 1 Mar 2017) replayed against the Fortran, limits 10x the float32-level error, fixed before mutation runs | `tests/data/noahmp50_snow` (built by `make_test_data.snow_family`) | catches 3 of the 4 edits that passed the older tests (M1 marginally, M3, M6); M8 is caught by the wet-snow test below; see `tools/mutation/run_mutations.py` |
| `test_regression.py::test_noahmp50_wetsnow` | Snow layers that end the step wet (360 records, 17/24 Mar 2017, layer liquid fraction 0.2-0.4 and at the 0.4 cap), limits 10x the unmutated error, fixed before the mutation run | catches M8 (`SNLIQMAXFRAC` halved: SWE error 6.4 mm) |
| `test_lai_tracker.py` | LAI tracking helper (fast; the only test CI runs) | none | exact |
| `test_ldt.py` | Python LDT parameter processing against the Fortran LDT output | needs `LISF_RUN` (the author's case directories, see `docs/DATA_AND_PATHS.md`) | skips when the data is missing; the skip reason prints with `-rs` |

Known limits of the suite (from the independent review, `docs/reports/independent_test_20261105.md`):
- Replays use sampled one-step calls (mostly thin, cold snow on 1 Feb and one SGP day). Changes to deep or wet multi-layer snow hydrology and to ice/frozen-soil conductivity can pass every Noah-MP 5.0 test: 4 of 9 mutations of `noahmp50_jax.py` survived.
- Limits are loose relative to physical constants: a 0.4% change in the latent heat of vaporization is caught only marginally.
- The closed-loop driver is not unit-tested; closed-loop evidence comes from the benchmark runs in `docs/reports/BENCHMARKS.md`.
- Closed-loop Noah-MP 5.0 comparison covers state variables; fluxes are compared only in replay.
