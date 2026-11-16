# Project brief

Status: Partly configured (2026-11-03). Fields marked Unresolved are unknown, not invented.

| Decision | Value |
| --- | --- |
| Project name and purpose | lisf-noahmp-jax: JAX ports of LISF land surface models (Noah-MP first), validated against the real Fortran LIS, for speed on GPUs |
| Client / receiving organization | Unresolved |
| DSG delivery lead | G. Tamkin |
| Client decision maker | Unresolved |
| Receiving technical maintainer | Unresolved |
| Repository owner and visibility | Unresolved |
| Support and security reporting contact | Unresolved |
| License and attribution requirements | DECIDED 2026-11-16: keep the Apache-2.0 template for the project's own code; Noah-MP-derived files also carry the UCAR agreement (licenses/UCAR-Noah-MP-LICENSE.txt, no sale, notice kept) and are credited in NOTICE |
| Runtime / stack and rationale | Python 3.10 + JAX 0.5.3 (validated); Fortran LIS GNU 14 as reference |
| Hosting account, platform, region, and billing owner | Unresolved |
| Data classification, storage, retention, and access | Unresolved |
| Delivery date and support period | Unresolved |

## Scope

Users: LIS/Noah-MP owners and customers evaluating GPU land-surface modeling. Deliverables: validated JAX ports with reproducible comparisons. Exclusions: full LIS (MPI, LDT/LVT, other models) unless chosen by the owners.

## Acceptance criteria

Defined by the port assurance protocol (docs/PORT_ASSURANCE.md): gates G0-G8, ratification Level 1 reached for Noah-MP 5.0/4.0.1 on named cases; Level 2-3 need owner review and independent reproduction.
Distinguish example behavior from the client's required functionality.

## Decisions and dependencies

Record external integrations, approved services, constraints, and reasons for major
choices. Link to private records for sensitive details.

## Open questions

For each unresolved decision, record an owner, next action, and whether it blocks
implementation, deployment, or handoff.
