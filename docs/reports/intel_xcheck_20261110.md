# MPI cross-check: R's Intel-built LIS at 20, 32 and 40 ranks

Date 2026-11-10, Discover. Question: is the wrong-output-at-32+-ranks fault of our GNU 14.2 / MPICH build (BENCHMARKS.md, ROADMAP item 12) in LIS itself or specific to our build?

Setup (`scripts/discover/intel_xcheck/`, job 59225494): LIS 7.8 binary built by R with Intel ifx 2025.3.2 and Intel MPI 2021.17 (environment module of the LISF repo, `env/discover/sles15/lisf_7.8_intel_2025.3.2`, copy in the repo), R's Noah-MP 5.0 NLDAS-2 configuration and parameter file, **NetCDF NLDAS-2.0 forcing reader** (the same reader that failed in our runs, `metforcing/nldas-2.0`) on files converted from the Discover GRIB (`tools/forcing/grib_to_nc.py`, 144 files), 1 to 6 October 2008 (5 days, 120 hourly history files per run), one Milan node. Layouts 4x5 = 20, 8x4 = 32, 8x5 = 40 ranks, as in our failing runs. Wall time: 2:20 (20 ranks), 1:53 (32), 1:45 (40).

Result (`cmp_runs.py`, 32 and 40 ranks against the 20-rank run of the same build, 120 files each): **0 differing cells and max abs difference 0** for Tair_f_inst, SoilTemp_inst and SWE_inst. The Intel build gives bitwise identical output at 20, 32 and 40 ranks. Our GNU build gave 26,000 to 55,000 wrong cells (Tair_f, 20 files) at 32 and 40 ranks in every optimised variant.

Reading: LIS itself, including the NetCDF NLDAS-2.0 forcing path and the Noah-MP 5.0 code, is clean at 32 and 40 ranks. The defect is specific to our GNU 14.2 / MPICH / ESMF build or to the patches we applied for it (tools/lis_patches). It is not a LIS defect, so nothing needs to go to the LIS team. The 12.1x speed claim is unaffected (it uses the clean 20-rank GNU run); the 32- and 40-rank GNU timings stay timing-only.

Limits: different case and period from the failing runs (October 2008 against February 2017), different compiler, MPI library and node type, so the cross-check shows that LIS is fine, not which part of our build is at fault; 5 days; one run per layout; only three variables compared; the failure in our build was not deterministic, but the clean result covers 120 files in each of two runs.

Next, if the 32+ rank GNU baseline matters: rebuild our LIS with GNU plus Intel MPI or OpenMPI (module `lisf_7.5_intel_2023.2.1_openmpi_5.0.0` exists in the LISF env directory) and with the unpatched LISF sources, to separate the MPI library, ESMF and our patches. Not started.
