# LISF to JAX: goal (DRAFT for owner decision)

Owner: G. Tamkin (to confirm). Drafted by a Claude Code session on 2026-10-21.
Status: signed off in chat by the owner on 2026-10-21 (see section 8); work then ran autonomously overnight. Outcome: see `REPORT.md`.
Method source: `projects/imvi/rocke3d_jax/Reports/FORTRAN_TO_JAX_RECIPE_20261021.md`, itself an unreviewed draft. It is project-local to imvi; its structure is reused here, its numbers and Rocke-specific details are not.
Review by: when the owner answers the open decisions below.

## 1. Facts found in the repository (2026-10-21)

These come from reading `projects/LISF` at commit `95ae9791a`. Nothing was built or run.

- LISF is NASA's Land Information System Framework (upstream `NASA-LIS/LISF`). It has three programs: LDT (data toolkit), LIS (the land modeling system), LVT (verification toolkit).
- LIS has about 2,900 `.F90` files and 311 `.c`/`.cpp` files. Fortran is the main language.
- LIS contains about 28 land surface model directories under `lis/surfacemodels/land/` (for example `noah.3.9`, `noahmp.4.0.1`, `clsm.f2.5`, `clm2`, `jules.5.x`, `summa.1.0`, `ac.7.2`, `cable`, `mosaic`, `vic.4.1.2.l`, `ruc.3.7`, `hyssib`). Line counts per model directory range from about 750 (template) to about 55,000 (`noahmp.4.0.1`). `noah.3.9` is about 18,000 lines, of which the core physics file (`phys/module_sf_noah39lsm.F90`) is about 6,300 lines with its two sibling modules.
- Besides land models there are routing, irrigation, data assimilation, optimization/uncertainty (`optUE`), RTMs, many meteorological forcing readers (`metforcing`), and run modes (`retrospective`, `forecast`, `agrmetmode`, and coupled modes).
- External dependencies named in the module files under `env/discover/`: MPI, NetCDF, HDF4, HDF5, HDF-EOS, ESMF, ecCodes, OpenJPEG, MINPACK, CRTM, CMEM, LAPACK, PETSc. Reference toolchains are Intel (discover, narwhal) and GNU (recent commits add GNU 16 support).
- On the current login node: `mpif90` (anaconda) and `gcov` exist; `gfortran`, `ifort`, `nvfortran` and `nf-config` are not on the default PATH. Whether a module provides them has not been checked.
- `projects/LISF` is a clone of the upstream repository with its own git history. Files added there show as untracked in that clone.

## 2. Proposed goal (to be confirmed)

Port one LIS land surface model, in one fixed configuration and run mode, to JAX, and show it reproduces the original Fortran model's output to criteria fixed in advance, so that it can run on accelerators and be differentiated.

The first target is a recommendation, not a fact about the project: **Noah 3.9 (`lis/surfacemodels/land/noah.3.9`) in the retrospective run mode, with one public test case.** Reasons: it is a mid-size, column-independent physics core of about 6,300 lines; land columns do not interact within a step, which suits `vmap`; and the repository ships a public test case walkthrough (`docs/public_testcase_walkthrough/`). Alternatives the owner may prefer: Noah-MP 4.0.1 (the more widely used model, about 55,000 lines), Catchment (`clsm.f2.5`), or a different model entirely.

## 3. Direction rules carried over from Rocke3D (proposed)

1. An end-to-end JAX deliverable comes before per-routine perfection. A thin vertical slice (one model, one configuration, one short run) is built first, then deepened.
2. Reduced scope is allowed only if the owner names it as the deliverable. Otherwise it is a stepping stone and is labelled as such.
3. Any component that cannot be ported (for example an external library) is decided early (phase P8 of the recipe), not left to the end.
4. Environments and paths are pinned and portable from the first day.

## 4. Out of scope until the owner says otherwise

LDT and LVT; data assimilation; routing; optimization and uncertainty estimation; MPI domain decomposition and I/O plumbing (the JAX port reads a prepared input and writes a comparison file, not LIS's full I/O layer); coupled run modes; any LIS model other than the first target.

## 5. Owner decisions (recorded 2026-10-21, G. Tamkin, in chat)

- **Done means:** one simulated day (the "done day"). Window length and which test case are still to be named.
- **Compiler and hardware:** the same as the Rocke3D mentor session (`rocke3d-jax-8c`). The exact compiler, flags, math library and hardware have been asked of that session and are **not yet recorded here**; do not assume them.
- **Purpose:** speed. Differentiability is not a goal. Correctness criteria are still required; speed is reported separately and never loosens them.
- **Reduced scope is a deliverable.** A reduced-scope port may be the final product, if it is labelled as reduced and the omitted parts are listed.

## 5a. Toolchain and hardware of the Rocke3D mentor session (reported by `rocke3d-jax-8c` on 2026-10-21; not verified by this session)

All items come from that session's reading of files in `projects/imvi/rocke3d_jax`. Items it did not find are marked "not recorded".

- Compiler: Intel ifort 19.1.3.304, module `intel/2020Update4`. Flags used for ModelE: `-O2 -ftz -convert big_endian -assume protect_parens -fp-model strict`. Math library: Intel libimf from the same install.
- NetCDF: `netcdf4/4.9.3s`. Its Fortran 90 module (`use netcdf`) **failed with ifort** because the installed `netcdf.mod` was built by gfortran; only the F77 interface (`include 'netcdf.inc'`) worked.
- MPI, HDF5, ESMF with this toolchain: not recorded.
- Setup: `source fullfidelity/env_modele.sh` (load Lmod directly; the intel module does not put `ifort` on PATH in a non-login shell).
- CPU host: `forest204`, 12 cores, Xeon E5-2697 v3 at 2.60 GHz, no SLURM client (submit with `ssh gpulogin1 sbatch`).
- GPU: V100 nodes, `--constraint=v100 --gres=gpu:1`, partition `compute`. Typical request `--nodes=1 --cpus-per-task=6 --mem=64G --time=01:00:00`; exclusive for timing jobs. SLURM account: not recorded. Driver 580.x.
- Their caveat: copy this compiler only if LISF's own build supports it, and if bitwise comparison to an existing LISF build is a goal, use that build's compiler and math library.

**Conflict to resolve before P1 (found by this session in the LISF source, 2026-10-21):** 229 LIS `.F90` files use the `use netcdf` module, 3 use `include 'netcdf.inc'`, 1,437 use `use esmf`, and 11 use `use hdf5`. The Fortran 90 netCDF module failed with ifort 19.1.3 in the mentor's environment, so LISF very likely cannot build there as-is. LISF's own reference toolchains are Intel 2023.2.1 (discover) and GNU. Options for the owner: (a) build with that ifort but with a netCDF built for it; (b) use a different compiler on the same hardware; (c) build only a minimal Noah 3.9 driver outside LIS (a reduced-scope baseline), which fits the decision that reduced scope is a deliverable. This is a proposal, not a finding about a working build; nothing was compiled.

## 5b. Further owner decisions (2026-10-21, G. Tamkin, in chat)

- **Speed goal: 4x speedup over "the current system".** "Current system" is not yet defined; see section 7.
- **First model: delegated to this session. Chosen: Noah 3.9** (`lis/surfacemodels/land/noah.3.9`), retrospective mode. The owner said they know nothing about this software, so this is a recommendation they have not independently checked.
- **Agreement: a tolerance is fine** (bitwise is not required). This session defines it in `ACCEPTANCE_CRITERIA.md` section 3 for the owner to approve.

## 5c. Why Noah 3.9 (facts read from the source, 2026-10-21; nothing compiled)

- The physics entry point is one subroutine, `SFLX` in `lis/surfacemodels/land/noah.3.9/phys/module_sf_noah39lsm.F90` (starts at line 75), which calls about 30 helper subroutines in the same file (soil heat and water flow, snow, evaporation, transpiration, a tridiagonal solver, and others). The file imports only a small constants module.
- It works on one land column at a time with a handful of soil layers. Columns do not exchange data inside a step, so the same function can be applied to many columns at once (`vmap`). That is where a speedup comes from.
- The LIS wrapper `noah39_main.F90` (about 1,600 lines) gathers forcing and parameters and calls `SFLX`. The wrapper, the I/O and the forcing readers are the parts a reduced-scope port would leave out.
- Precision: none of the `lis/arch/configure.*` files set a real-size flag (no `-r8`, `-fdefault-real-8`, or similar), so default `real` is most likely 32-bit single precision in LIS builds. This was checked by grep only, in those files; a specific build may differ. It matters for the tolerance (acceptance criteria section 3).
- Risk: Noah has threshold logic (freezing, snow, glacial branch `SFLX_glacial`) where a tiny difference can flip a branch in some cells.

## 6. Still open

1. Which public test case, and which day (the owner said "one day").
2. Recorded forcing and parameters acceptable, or must the forcing readers be ported? (Recommended: recorded, as part of the reduced scope.)
3. Where the project's working files live: inside the `projects/LISF` clone (current draft location, untracked) or a separate project directory?
4. Compiler: options (a), (b), (c) in section 5a; recommended (c).

## 7. Definition of "4x speedup over the current system" (proposed, needs owner confirmation)

"Current system" is unknown to this session. Proposal: the original Fortran Noah 3.9 physics (the `SFLX` loop over columns) built in phase P1, run on **one core of `forest204`** with the baseline compiler, for the same columns and the same day. The JAX version is timed on the V100 and, separately, on CPU. Report both a single-core and a 12-core Fortran baseline so the claim cannot be read wrongly. Time covers physics only, not I/O, and includes one run with compilation and one steady-state run. Speedup is only meaningful above some number of columns, because one column on a GPU is slower than Fortran; the column count of the test case is therefore part of the claim. If "current system" means something else (for example the full LIS retrospective run, with I/O), say so and the measured scope changes.

## 8. Sign-off and default answers (2026-10-21, G. Tamkin, in chat)

The owner signed off on the drafts and told this session to choose defaults for ALL open questions and run autonomously overnight until the project is finished. Defaults chosen by this session (recorded before any comparison; changeable by the owner, never silently):

1. **"Current system" / speedup baseline:** as in section 7 (original Fortran Noah 3.9 `SFLX` column loop, one core of `forest204`, plus a 12-core number). Target 4x on the V100 for the test-case column count; CPU JAX reported separately.
2. **Recorded inputs:** allowed (forcing, parameters, initial state), all listed.
3. **Files stay in `projects/LISF/jax-port/`** (owner decision).
4. **Compiler:** option (c), a minimal Noah 3.9 driver outside LIS, built with the mentor toolchain (ifort 19.1.3) if it works, else the first compiler that works; the choice is recorded in `jax-port/LOG.md`.
5. **Test case and day:** no public test case is assumed. The driver is fed a synthetic but physically plausible forcing and parameter set unless a shipped test case's forcing can be read cheaply; either way the result says which. One simulated day.
6. **Bitwise:** not required; tolerance per `ACCEPTANCE_CRITERIA.md` section 3a.
7. **Commits:** none. The owner has not authorized committing or pushing; all work stays uncommitted in the working tree.
8. **Stop conditions:** the run stops and records the reason if a step needs something outside `projects/LISF` (installing system software, sharing data externally, spending an unusual amount of cluster time). Progress and decisions are logged in `jax-port/LOG.md`.
