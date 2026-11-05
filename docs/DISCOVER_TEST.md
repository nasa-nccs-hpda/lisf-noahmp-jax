# Running the independent test on Discover (NCCS)

Status: NOT RUN. Blocked 2026-11-05: no authenticated SSH session from the work machine to `login.nccs.nasa.gov` (publickey / keyboard-interactive needs a person).

## To unblock (person, once)
In a normal terminal on the work machine (not inside the agent session):
```
ssh -O exit -o ControlPath=~/.ssh/cm-nccs gtamkin@login.nccs.nasa.gov    # removes the stale master from the failed -fNM attempt
ssh -M -o ControlPath=~/.ssh/cm-nccs -o ControlPersist=8h gtamkin@login.nccs.nasa.gov    # authenticate (password / 2FA), then log out; the master stays for 8 h
ssh -o ControlPath=~/.ssh/cm-nccs -O check gtamkin@login.nccs.nasa.gov   # expect "Master running"
```
Then the agent can run `ssh -o ControlPath=~/.ssh/cm-nccs gtamkin@login.nccs.nasa.gov '<command>'`.

## What the Discover test is for
Everything so far ran on one cluster (ADAPT/forest, Xeon Gold 6248, V100). A tester on Discover checks that the repo works from a clean clone on a different system with its own Python, MPI, module and file-system setup, which is the owner's likely situation (LIS reference toolchains are Intel on Discover). Same tasks as `docs/INDEPENDENT_TEST_PLAN.md` T0-T8, minus anything needing the author's data paths:
1. Clone from GitHub (needs a credential on Discover; the repo is private).
2. Create the Python environment from `pyproject.toml` (Python 3.10+, `jax>=0.5,<0.6`); note module names and whether PyPI is reachable from a login or compute node.
3. T0-T2 on a compute node via `salloc` or `sbatch` (do not run the 28 min test suite on a login node).
4. T3 only if the CONUS/SGP inputs are present on Discover; otherwise report exactly what is missing.
5. Optional: a Discover Fortran LIS (Intel) build exists? If so, compare against it as an independent reference.

## Facts to record on Discover
Hostnames and node types, Python and JAX versions, wall time of the test suite, anything that differed from the work machine (paths, modules, file-system speed, missing libraries).
