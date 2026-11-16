# Delivery and handoff record

Status: delivery to the LIS developer (R) in progress, 2026-11-16. Acceptance and maintenance are not decided.

| Item | Evidence / owner / status |
|---|---|
| Agreed deliverables and acceptance criteria | Deliverable: a tarball with the port, example, tests, configs, scripts and results (this repository at the tagged commit). Acceptance thresholds: set by the author before the runs (`docs/PORT_ASSURANCE.md`, limits in `tests/`); **not approved by the code owners** |
| Acceptance results and client decision | None yet. Requested from R: review of `physics/noahmp50_jax.py`, Fortran wall time of the 60-day run, further reference cases and tolerances, reproduction |
| Repository and maintainer access | GitHub repository `nasa-nccs-hpda/lisf-noahmp-jax` stays private; delivery is by tarball. **Receiving maintainer: not named** |
| Setup/run/test commands verified from a fresh environment | Extracted tarball and `uv sync --locked`: on Nexus (CPU) and on Discover (CPU suite 49 tests, A100 example), commit 12d535c, 2026-11-16. Re-run of the current tag on Discover: not done |
| Release commit and artifact version | Not yet tagged (testing of the cleaned tree in progress); planned `v0.2.0`, tarball made with `git archive` from the tag; the earlier `v0.1.0` tag is superseded (its Slurm output paths are broken); full pre-cleanup tree: tag `archive-2026-11-16` |
| Deployment target and smoke-check evidence | Not applicable (a research code run by the user; no service) |
| Operations and rollback runbook | Not applicable |
| Hosting, registry, domain, and billing ownership | Not applicable; GitHub CI (Actions) only |
| Secret locations and access transfer | None used |
| Data ownership, retention, and backup/restore | Example input is cropped NLDAS-2 (NASA GES DISC, public); CONUS and Discover run data stay in the author's and R's directories and are not in the package |
| Monitoring and support/security contacts | **None named** |
| License, notices, and third-party obligations | Apache-2.0 plus the UCAR Noah-MP terms; `LICENSE`, `NOTICE`, `licenses/` |
| Known limitations, open issues, and next steps | `README.md` (what it is not), `docs/PORT_ASSURANCE.md` (open gates G6 and G7, G8 partial): no expert review; worst tiles on R's case not examined; R's Fortran wall time unmeasured; our GNU-built MPI LIS gives wrong output at 32+ ranks (our build, not isolated) |
| Support period and responsibility after delivery | **Not decided** |
