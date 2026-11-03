"""Noise floor of the Fortran baseline: compare reference outputs from (a) -O0/-O3 builds and
(b) one-ulp perturbed start states against the -O2 base run. Prints per-field statistics."""
import sys
import numpy as np
import spec

N = 2000
NT = 48
NV = len(spec.OUT)


def load(p):
    return np.fromfile(p, np.float32).reshape(NT, NV, N).astype(np.float64)


base = load("ref_2000_O2.bin")
others = {"O0": load("ref_2000_O0.bin"), "O3": load("ref_2000_O3.bin")}
for m in range(1, 6):
    others[f"ulp{m}"] = load(f"ref_2000_m{m}.bin")

rng_scale = {}
for i, n in enumerate(spec.OUT):
    a = base[:, i]
    rng_scale[n] = max(a.max() - a.min(), 1e-30)

print(f"{'field':9s} {'range':>10s} | " + " | ".join(f"{k:>17s}" for k in others))
print("(each cell: max|diff| over day / fraction of cell-steps with |diff|>1e-5*range)")
rows = []
for i, n in enumerate(spec.OUT):
    cells = []
    for k, o in others.items():
        d = np.abs(o[:, i] - base[:, i])
        cells.append((d.max(), (d > 1e-5 * rng_scale[n]).mean()))
    rows.append((n, rng_scale[n], cells))
    print(f"{n:9s} {rng_scale[n]:10.3g} | " + " | ".join(f"{c[0]:9.2e} {c[1]:7.4f}" for c in cells))

# column-level: how many columns differ at all, and the worst columns for the O0 comparison
for k in ("O0", "O3", "ulp1"):
    d = np.abs(others[k] - base)
    for n in ("t1", "smc1", "sneqv", "eta"):
        i = spec.OUT.index(n)
        col = d[:, i].max(0)
        print(f"{k} {n}: columns with any diff {int((col>0).sum())}/{N}; worst col {col.argmax()} diff {col.max():.3e}; 99th pct col diff {np.percentile(col,99):.3e}")

# ---- ensemble envelope used by the acceptance gate (see ACCEPTANCE_CRITERIA.md 3a) ----
import json
QS = (50.0, 90.0, 99.0, 99.9, 100.0)
env = {}
for i, n in enumerate(spec.OUT):
    per_member = []
    for m in range(1, 6):
        d = np.abs(others[f"ulp{m}"][:, i] - base[:, i]).ravel()
        per_member.append([float(np.percentile(d, q)) for q in QS])
    env[n] = dict(range=float(rng_scale[n]), quantiles=list(QS),
                  member_max=[float(max(col)) for col in zip(*per_member)])
json.dump(env, open("noise_floor.json", "w"), indent=1)
print("wrote noise_floor.json")
