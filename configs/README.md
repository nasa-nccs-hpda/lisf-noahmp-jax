# Configurations of the headline runs

The LIS configuration files behind the CONUS tables in `docs/reports/BENCHMARKS.md` and the flux, restart and MPI reports (Gate G8: configs in the repository).

| Directory | Files | Used for |
|---|---|---|
| `conus_mp5_60d/` | `lis.config.fortran`, `lis.config.jax`, `.jax_flux`, `.jax_fix1`, `.jax_fix1_ulp`, `.jax_ulp`, `.jax_f64`; `MODEL_OUTPUT_LIST.TBL.noahmp5`, `forcing_variables.txt` | CONUS 60-day Noah-MP 5.0, Fortran serial vs JAX; flux comparison; one-ulp and float64 checks |
| `conus_mp5_mpi/` | `lis.config.base`, `.60d_n20`, `.60d_n40`, `.n32`, `.n40`, `.chk32`, `.chk40`, `.evt_rst2` | 20- and 40-core MPI baselines, the 32+ rank fault, restart chain |
| `restart_dumps/` | `lis.config.mp50_evt0201` ... `0324` (`0306b` uses rain-snow option 1) | Fortran state dumps for the first-divergence replays |

Run directories are relative (`./INPUT`, `./lisout`). Two absolute paths are written `@LISF_BUILD@` (the Noah-MP.5.0 parameter table inside your LISF build) and `@LISF_RUN@` (the case directory); substitute them with `sed`. The domain file `lis_input.conus.noahmp5.nc` is made by LDT from the LDT config in your case directory and the NLDAS-2 forcing is the GES DISC archive (`INPUT/wget_gesdisc_nldas2.sh` pattern); neither is in the repository (30 MB forcing for the 5-day example only). The developer's own case configuration stays with its owner on Discover (see `docs/DISCOVER_R_CASE_PLAN.md`).
