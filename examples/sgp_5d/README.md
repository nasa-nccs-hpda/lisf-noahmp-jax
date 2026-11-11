# Example: Noah-MP 5.0, Southern Great Plains, 5 days

A complete case that runs without `/panfs`, with the real Fortran LIS 7.8 result beside it.

| | |
|---|---|
| Model | Noah-MP 5.0 (`Noah-MP.5.0`), default options of the validation runs |
| Domain | 28 x 22 cells at 0.25 deg (34.4 N to 39.9 N, 102.9 W to 96.1 W), from the LDT parameter file in `ldt/` |
| Period | 2017-01-01 00:00 to 2017-01-06 00:00, 15-minute step, 6-hourly output, cold start |
| Forcing | NLDAS-2 hourly (`NLDAS_FORA0125_H.*.020.nc`), 122 files cut to the domain plus 0.5 deg (the NLDAS grid is kept, other cells hold the fill value) |
| Reference | `reference/`: the Fortran LIS 7.8 history files (20) of the same run, repacked with zlib |

## Run

```bash
uv sync --locked --extra test                       # once, from the repository root
cd examples/sgp_5d
uv run --project ../.. lisf-noahmp-jax -C . $PWD/lis.config     # about 1 min on CPU; writes ./out/SURFACEMODEL/201701/
uv run --project ../.. python ../../src/lisf_noahmp_jax/driver/compare_hist.py reference out
```

(`JAX_PLATFORMS=cpu` forces CPU. On a GPU machine install with `--extra gpu`.)

## What to expect (measured 2026-11-11, CPU, 20 history files)

The forcing fields come out identical to the reference. 99th-percentile differences: Qle 1.1e-2 and Qh 1.1 and Qg 1.4 W/m2 (of ranges 97, 217 and 195), Lwnet 0.26 W/m2, SWE 3e-6 mm, soil moisture 1e-6 m3/m3, soil temperature 0.016 K. Snow cover is thin in this case, so it says little about deep snow. `tests/test_example_sgp.py` runs this case and fails if any variable is above its limit.

## Sources and licences

- Forcing: NASA GES DISC NLDAS-2 (public data), cut to the domain by `tools/examples/make_sgp_example.py`.
- `NoahmpTable.TBL`, `MODEL_OUTPUT_LIST.TBL.noahmp5`, `forcing_variables.txt`: copied from the NASA LISF 7.8 distribution (Apache 2.0).
- `ldt/lis_input.sgp.noahmp5.nc`: produced by LDT 7.8 for this domain.
- Regenerate everything (needs the cluster data): `python tools/examples/make_sgp_example.py`.
