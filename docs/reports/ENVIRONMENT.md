# Environment for the LISF Noah 3.9 JAX port

Written by a Claude Code session on 2026-10-21. Everything below was observed on 2026-10-21 on the machines named; nothing was tested on another machine.

## Fortran baseline (CPU host `forest204`, 12 cores, Xeon E5-2697 v3)

- Compiler: Intel ifort 19.1.3.304, `module load intel/2020Update4` via Lmod (`source /usr/share/lmod/lmod/init/bash`), then put the compiler directory on PATH:
  `/panfs/ccds02/app/modules/intel/platform/x86_64/rhel/8.6/2020Update4/compilers_and_libraries_2020.4.304/linux/bin/intel64`
- Flags: `-O2 -ftz -assume protect_parens -fp-model strict -fpp`; default 4-byte `real`. This is the Rocke3D toolchain except that `-convert big_endian` is not used (the driver reads little-endian stream files).
- Build: `baseline/build.sh` (optional suffix and flags, e.g. `./build.sh _O0 -O0`). It supplies an empty `LIS_misc.h` because LIS generates that file at configure time; the Noah 3.9 physics needs no macro from it. The source under `lis/surfacemodels/land/noah.3.9/phys/` is not modified.
- Not used: NetCDF, HDF5, ESMF, MPI. The mentor session reported that `use netcdf` failed with this ifort; this driver avoids the question by not linking LIS.
- Coverage and the cross-compiler check: `module load gcc/14.2.0` (gfortran), `--coverage -O0`, no `-fconvert=big-endian`.

## JAX

- Used: JAX 0.5.3 / jaxlib 0.5.3, Python 3.10.16, NumPy 2.2.4, from the existing conda env `graphcast-env` (created by the Rocke3D project; this session did not create or modify it). CPU runs also worked with the site python's JAX 0.4.30 (no GPU there). Outputs from the two JAX versions are not bitwise identical on CPU.
- GPU: partition `compute`, `--constraint=v100 --gres=gpu:1`, node `gpu012` (Tesla V100-SXM2-32GB, driver 580.178.04, CPU Xeon Gold 6248). Submit through `ssh gpulogin1 sbatch` (forest204 has no SLURM client). No SLURM account was given; the default worked.
- Run: `conda run -n graphcast-env python bench.py baseline/case_N.bin`. Set `JAX_PLATFORMS=cpu` on hosts without a GPU.
- Paths: `gpu_bench.sbatch` and `baseline/build.sh` contain this checkout's absolute paths and the cluster module path; edit before use elsewhere.

## Data

- `baseline/gen_case.py N [seed] [member]` writes `case_N.bin`: parameters from the real LISF tables in `lis/configs/557WW-7.3-FOC/noah39_parms`, SYNTHETIC forcing and initial state. Case files, reference outputs and executables are excluded from git locally through `.git/info/exclude` (about 4 GB in total); regenerate them with `gen_case.py` (seed 20261021) and `build.sh`.
