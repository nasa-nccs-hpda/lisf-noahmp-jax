# Audit: nasa-nccs-hpda/facts2-experiments (main) against nasa-nccs-hpda/dsg-template

Date 2026-11-10. Auditor: Claude (assistant), read-only. Subject: shallow clone at 723e5ec (63 tracked files, 5 MB). Reference: `dsg-template` 71521ca. Method: file-by-file diff against the template, static scan of tracked files. Not done: running the repo's tests (the Claude Code classifier blocked executing the cloned repo's code; left for the owner), a container build, a full-history secret scan (shallow clone), legal review of the patched third-party modules.

## Verdict
**Close to ready for external delivery; fix the items in section 3 first.** It was started from the template and kept its structure, tooling and agent skills. The remaining gaps are mostly owner decisions (project brief, handoff) plus a few small defects.

## 1. Conformance table
| Template item | Status |
|---|---|
| `src/<package>/`, `pyproject.toml`, `.python-version`, `uv.lock` | present; renamed (`facts2_experiments`, command `facts2`); Python >= 3.11, 3.12 default, as in the template |
| `Dockerfile`, `.dockerignore` | present; Dockerfile differs from the template only in the entrypoint (`facts2`); `.dockerignore` identical. Base image `python:3.12-slim` is not pinned by digest (same as the template) |
| CI `.github/workflows/ci.yml` | present, differs in 10 lines: smoke tests changed to `facts2 --help` (no sample data); same Python matrix, uv version and container job |
| PR template, issue templates | present, identical to the template |
| `AGENTS.md`, `.agents/skills/` (3 skills) | present; `dsg-deployment` and `dsg-handoff` identical, `dsg-project-kickoff` adapted; AGENTS.md and CONTRIBUTING.md adapted (4 to 12 changed lines) |
| `LICENSE` | present, identical (Apache-2.0); `pyproject.toml` declares Apache-2.0 |
| `.env.example`, `.gitignore` | present, adapted |
| `docs/project-brief.md`, `handoff.md`, `deployment.md`, `README.md` | present and filled; deployment marked not applicable with a reason (batch tool on a 12-core node) |
| `docs/template-setup.md`, `template-release.md` | removed, as the template instructs |
| `tests/` | present: 182 tests (42 + 126 + 14, matches the README claim), `tests/README.md` present; not run in this audit |
| `examples/` (template sample fixtures) | not present; the input data is 33 GB from Zenodo, fetched by `scripts/fetch_catalog_data.sh`; only site lists are tracked in `data/`. Acceptable, but the quickstart cannot be run end to end without the 33 GB download and Apptainer |
| `SECURITY.md`, `CODEOWNERS` | missing (optional; need real contacts) |

## 2. Hygiene
- Absolute internal paths: none in tracked files (the templates use `@WORKSPACE@`). Good.
- Secrets: none found in tracked text (the only hit is the `@WORKSPACE@` token constant). No e-mail addresses. Full-history scan not done.
- Size: 5 MB, largest file 376 KB. No binaries beyond three PNG figures.
- Third-party code: `patches/container/` holds modified copies of upstream module sources together with `*.orig.*` originals, and `patches/registry/` two patches. The upstream licences are "to be confirmed" in the project brief; confirm them before delivery.
- `docs/DRAFT_upstream_issues.md` is a draft deliberately held back ("NOT posted"). Decide whether it ships to the receiving party.

## 3. Findings to fix
1. **Stale paths in `patches/README.md`**: lines 9-10 apply `../patches/bamber19_pass_to_total.patch` and `../patches/deconto21_larmip_pass_to_total.patch`; the files are in `patches/registry/` (the main README has the correct paths). A user following the patches README gets a "file not found".
2. **Project brief**: client, decision maker, receiving maintainer, support and security contact, and delivery date and support period are `Unresolved` (5 fields, plus the open question on who maintains it after delivery); the handoff form has 2 unresolved rows (monitoring and support contacts, support period). These need the owner.
3. **Licences of the patched modules** (section 2) are unconfirmed.
4. **No end-to-end check that a recipient without the 33 GB data and Apptainer can verify the repo**: only the unit tests (no data needed) and the `--help` smoke tests. Consider a small synthetic fixture and say in the README which checks need what.
5. Optional: `SECURITY.md` and `CODEOWNERS` once real contacts exist; pin the base image by digest for production, as the template's deployment guide advises.
6. Not verified here: tests pass, container builds, CI green on `main`, repository visibility and access (brief says private, to be confirmed).
