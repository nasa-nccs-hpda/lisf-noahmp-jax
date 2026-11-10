# Audit: nasa-nccs-hpda/lis (main) against nasa-nccs-hpda/dsg-template

Date 2026-11-10. Auditor: Claude (assistant), read-only. Subject: shallow clone of `lis` `main` at 8b37caa (336 tracked files). Reference: `dsg-template` at 71521ca. Method: file-by-file comparison with what the template's README, AGENTS.md, setup checklist, release notes and CI require, plus a scan of the tracked files. Not done: full git-history scan (shallow clone), running the repo's tests, legal review of licences.

## Verdict
**Not ready for external delivery.** `lis` is the earlier flat layout of the JAX port (frozen on 2026-11-05). It has none of the template's packaging, CI, container, agent or project-record files. The active, template-derived repository is `lisf-noahmp-jax`; it meets most of the template, with the gaps in section 3. Recommendation: deliver from `lisf-noahmp-jax`, and keep `lis` internal or archive it. Do not deliver both.

## 1. Conformance table, `lis` (and `lisf-noahmp-jax` for comparison)
| Template item | `lis` main | `lisf-noahmp-jax` |
|---|---|---|
| `src/<package>/` installable package | MISSING: about 20 modules at the repo root plus `lisjax/`, `lis/` (not installable, no entry point; run as `python -m lisjax...`) | present (`src/lisf_noahmp_jax`) |
| `pyproject.toml`, `.python-version` | MISSING | present (requires Python >= 3.10; template CI matrix is 3.11-3.13) |
| `uv.lock`, `uv sync --locked` flow | MISSING | MISSING (uv not available on the work machines; CI uses pip) |
| `Dockerfile`, `.dockerignore`, container smoke test | MISSING | present (not built or run in this review) |
| `.github/workflows/ci.yml` | MISSING (no `.github/` at all) | present, deviates: setup-python + pip, 1 fast test, no container job |
| `.github/pull_request_template.md`, issue templates | MISSING | present |
| `AGENTS.md` and `.agents/skills/` (3 skills) | MISSING | present |
| `CONTRIBUTING.md` | MISSING | present |
| `LICENSE` (template keeps Apache-2.0) | MISSING: no licence at all | present (template Apache-2.0); project brief marks licence OPEN |
| `.env.example` | MISSING (code reads `LISF_RUN`, `JAX_PLATFORMS`, `MPVER`; documented only in `DATA_AND_PATHS.md`) | present |
| `docs/project-brief.md`, `template-setup.md`, `deployment.md`, `handoff.md`, `docs/README.md` | MISSING (equivalents exist as `STATUS_*.md`, `ROADMAP.md`, `DATA_AND_PATHS.md`, `REPRODUCE_NOAHMP.md`, `PORT_ASSURANCE.md`, `LOG.md`) | present, partly filled: client, owner, maintainer, hosting, data classification, support period all Unresolved |
| `examples/` small synthetic fixtures | MISSING | MISSING (tests use recorded real-LIS data, see 2) |
| `tests/` runnable with documented command | pytest tests exist (4 modules, 43 `.bin` data sets); no lockfile, no CI, `tests/README.md` is empty | present, 49 tests, README written; only `test_lai_tracker.py` runs in CI |
| `docs/template-release.md` removed from client copy | n/a | removed (not in `docs/`) |
| `SECURITY.md` / `CODEOWNERS` (only with real contacts) | MISSING | MISSING (optional; needs a real contact) |
| Starter identifiers renamed (`dsg-project`, `dsg_project`) | n/a | renamed (remaining mentions are in `template-setup.md`, `deployment.md`, skills: expected) |

## 2. Hygiene findings in `lis` that would block external delivery
1. **Absolute internal paths** in 60 tracked text files (`/panfs/...`, `/home/gtamkin`, `/discover/...`). Reveals the author's account and directory layout; the scripts are not portable.
2. **Internal and meeting material tracked**: 8 PDFs in `status_slides/` (management, science, developer briefings, an owner-meeting deck, status table) and their builders; `LOG.md` (dated record including failures and drafts), `Reports/` run outputs. Review for anything not meant for the receiving party before delivery.
3. **Test data provenance**: 43 `.bin` files and one `.npz` (about 40 MB) recorded from real LIS runs (NLDAS-2 forcing, LDT parameters, SGP data). The template wants small synthetic fixtures. Record the source, licence and sensitivity of this data in the project brief, or replace it with a synthetic set.
4. **Licence**: none in the repo. The ported physics derives from LISF (Apache-2.0) and NCAR Noah-MP (UCAR licence); attribution and the licence choice are undecided (project brief, "OPEN").
5. **Secrets**: no credential patterns found in tracked text (password, token, API key, private-key headers: none; one e-mail-like hit was binary noise in a PDF). **Not verified for git history** (shallow clone); run a full-history secret scan (for example gitleaks) before delivery.
6. **Naming**: the repo is called `lis` but is not LIS; the README calls it `jax-port`. Likely to confuse the receiving team.
7. **Run instructions** point to the author's environment scripts (`dev/lisf_deps/env_lis.sh`); `REPRODUCE_NOAHMP.md` says "not yet followed by anyone but the author" (the 2026-11-05 independent review since followed part of the `lisf-noahmp-jax` version).

## 3. Gaps in `lisf-noahmp-jax` (the repo to deliver)
1. `uv.lock` missing; CI uses pip, not the template's uv flow; Python floor 3.10 vs template 3.11+ (JAX 0.5.x is validated on 3.10).
2. CI runs one fast test only; the 49-test suite takes 28-52 min; no container job; no `examples/`.
3. Project brief fields unresolved: client, decision maker, receiving maintainer, repo owner and visibility, support and security contact, licence and attribution, hosting, data classification, delivery date. Handoff form is blank by design.
4. Same data-provenance and absolute-path items as section 2 in part (paths mostly fixed on 2026-11-05; re-scan before delivery).
5. Run a full-history secret scan; confirm third-party licences.
6. Container: Dockerfile present but not built or run here.

## 4. Suggested order
1. Decide which repo ships (recommend `lisf-noahmp-jax`); mark `lis` archived with a pointer.
2. Fill the project brief (owner, maintainer, licence, data classification) with the owner.
3. Add `uv.lock` (needs a machine with uv) and align CI with the template, including the container build and a fast replay subset.
4. Full-history secret scan; sweep for `/panfs`, `/home/gtamkin`, `/discover` strings; review tracked decks and logs.
5. Record test-data provenance or add a small synthetic `examples/` set.
6. Verify from a clean clone (template's own acceptance check): install, tests, one case, container smoke test.
