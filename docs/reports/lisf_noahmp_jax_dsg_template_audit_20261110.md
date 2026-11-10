# Full audit: lisf-noahmp-jax (main) against nasa-nccs-hpda/dsg-template

Date 2026-11-10. Auditor: Claude (assistant). Subject: local clone of `nasa-nccs-hpda/lisf-noahmp-jax`, main, 24 commits, 382 tracked files, `.git` 49 MB. Reference: `dsg-template` 71521ca. Method: file-by-file diff, content read of every template-derived file, scans of tracked files and of the full git history (24 commits), `lisf-noahmp-jax --help` and the fast test (2 passed). Not done: container build (podman cannot run here: no writable storage), `uv` flow (uv not installed), the 49-test suite (28-52 min, run on 2026-11-05 by the independent tester), CI status on GitHub (not queried), legal review.

## Verdict
**Not ready for external delivery yet, but close in structure.** The template files are present, but three things break the template's own contract (a container build that cannot work, a uv workflow with no lockfile, project records that are still the template's blank text) and the licence and data questions are open. All are fixable; most need a person's decision, not engineering.

## 1. Conformance table
| Template item | Status | Note |
|---|---|---|
| `src/<pkg>/`, `pyproject.toml`, console script | OK | `src/lisf_noahmp_jax`, command `lisf-noahmp-jax` runs (`--help` verified); Python >= 3.10, the template says 3.11+ (JAX 0.5.3 validated on 3.10) |
| `.python-version` | DEVIATES | 3.10 (template 3.12) |
| `uv.lock` + `uv sync --locked` flow | **MISSING** | AGENTS.md says `uv sync --locked`; no lockfile exists; uv not available on the work machines |
| `Dockerfile` | **BROKEN (by reading; build not run)** | `COPY pyproject.toml uv.lock ...` and `uv sync --locked` need `uv.lock`, which is not in the repo, so the build must fail at the COPY step; `.dockerignore` also allow-lists `uv.lock` |
| `.dockerignore` | OK | identical to template |
| CI `ci.yml` | DEVIATES, and likely red | uses setup-python + pip and runs one fast test (`test_lai_tracker.py`); the container job builds the Dockerfile above, so it should fail; no `examples/` smoke test; CI result on GitHub not checked |
| PR template, issue templates | OK | identical |
| `AGENTS.md` | OK, adapted | project rules added; test command now `uv run --locked --extra test python -m pytest` (cannot run without the lockfile) |
| `.agents/skills/` (3 skills) | OK | identical to the template |
| `CONTRIBUTING.md`, `LICENSE` (Apache-2.0) | OK | identical; see licence finding |
| `.env.example` | OK, adapted | lists the real variables (`LISF_RUN`, `LIS_BIN`, `MPVER`, ...) |
| `.gitignore` | OK, adapted | |
| `docs/project-brief.md` | PARTLY | filled where known; 9 `Unresolved` entries (client, decision maker, receiving maintainer, owner and visibility, support and security contact, hosting, data classification, delivery date and support period; licence marked OPEN) |
| `docs/deployment.md` | **NOT FILLED, STALE** | byte-identical to the template: still says "The CSV example is a batch CLI" and `docker build -t dsg-project:local .` |
| `docs/handoff.md` | **NOT FILLED** | identical to the template (every row `Unresolved`) |
| `docs/template-setup.md` | NOT DONE | identical to the template, unticked; the template says to remove it once its decisions are recorded |
| `docs/README.md` | OK, adapted | index plus old-to-new path map |
| `docs/template-release.md` | OK (absent) | |
| `examples/` fixtures | MISSING | no synthetic example input; tests use recorded real-LIS data |
| `tests/` and `tests/README.md` | OK | 49 tests; README written 2026-11-05; README files still say 48 in places |
| `SECURITY.md`, `CODEOWNERS` | MISSING | optional; need real contacts |

## 2. Hygiene
- **Absolute internal paths**: 73 tracked files contain `/panfs`, `/home/gtamkin` or `/discover` (38 `scripts/slurm`, 16 `docs/reports`, 9 `tools/lis_patches`, plus tools, tests, DATA_AND_PATHS, LOG). None in `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `pyproject.toml`, `Dockerfile`, `src/` or `.github/`. Expected for cluster job scripts and logs; still reveals the author's account layout.
- **Secrets**: none in tracked text; none matching token, key or password patterns in the added lines of the 24-commit history.
- **Personal data**: tracked files include a login host with the author's username (4 times). Git history has 21 commits by the NASA address and 3 by a personal Gmail address: decide whether the history can ship as is.
- **Data provenance**: `tests/data` (45 case folders, 73 MB on disk, largest files 5-8 MB) holds records dumped from real LIS runs (NLDAS-2 forcing, LDT parameters, SGP). The template wants small synthetic fixtures. Source, licence and sensitivity are not recorded (project brief: data classification `Unresolved`).
- **Internal material tracked**: 18 files in `docs/slides` (briefing decks, owner-meeting deck), `docs/LOG.md`, 43 files in `docs/reports`, `docs/INDEPENDENT_TEST_PLAN.md`, `docs/DISCOVER_TEST.md`. Review before delivery to an outside party.
- **Licence and attribution**: only the template's Apache-2.0 `LICENSE`. No `NOTICE`; none of the 10 physics modules carries a licence or attribution header (they cite the Fortran source file in a docstring). The code is a derivative of LISF (Apache-2.0) and NCAR Noah-MP (UCAR licence), so attribution is required and the brief marks it OPEN.

## 3. Findings, in order of importance
1. **The container build cannot work as written** (Dockerfile, `.dockerignore`, CI container job all need `uv.lock`). Fix: create `uv.lock` on a machine with uv, or change the Dockerfile and CI to pip as the CI test job already does.
2. **No `uv.lock`**, so AGENTS.md, the Dockerfile and the template's workflow disagree with the repo (CI uses pip). Pick one flow and make AGENTS.md, README, CI and Dockerfile say the same thing. Pinning `jax>=0.5,<0.6` is a range, not a lock.
3. **Licence and attribution unresolved**; add NCAR/LISF notices and headers once decided.
4. **`deployment.md`, `handoff.md`, `template-setup.md` are the template's text** (stale, refers to `dsg-project` and a CSV example). Fill them (deployment: batch jobs on Slurm, no hosted service, mark not applicable with a reason as `facts2-experiments` did) or remove the template-only parts.
5. **Project brief**: 9 unresolved entries need the owner.
6. **CI covers one test (2 tests) of 49** and no replay subset; the container job probably fails. Consider running the new snow test (37 s) and a small replay subset in CI.
7. **Test data provenance and size** (73 MB, real-run data); no `examples/`.
8. **Paths and internal material** (section 2): decide what ships; the slurm scripts and reports are the bulk.
9. **Docs drift**: README and tests/README say 48 tests (now 49); `.python-version` and Python floor differ from the template without a note in the brief.
10. Not verified: CI status on GitHub, container build, `uv` flow, repository visibility and access.

## 4. What is fine
Package layout, console script, agent skills (3), issue and PR templates, `.dockerignore`, `.env.example` with real names, `docs/README.md` with a path map, a written `tests/README.md`, an independent-review report, an assurance protocol and roadmap, no secrets in history, and a repo size (49 MB) that is reasonable.
