"""Compare a JAX output file with the Fortran reference and apply the ensemble-envelope gate
(ACCEPTANCE_CRITERIA.md 3b). Usage: python compare.py ref.bin jax.bin N [gate [envelope.json]]
The envelope should come from Fortran one-ulp members run at the SAME N (tail percentiles grow with N)."""
import sys, json
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
import spec

ref_p, jax_p, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
gate = len(sys.argv) > 4
NT, NV = 48, len(spec.OUT)
a = np.fromfile(ref_p, np.float32).reshape(NT, NV, N).astype(np.float64)
b = np.fromfile(jax_p, np.float32).reshape(NT, NV, N).astype(np.float64)
QS = (50.0, 90.0, 99.0, 99.9, 100.0)
env = json.load(open(sys.argv[5] if len(sys.argv) > 5 else "baseline/noise_floor.json")) if gate else None
nfail = 0
print(f"{'field':9s} {'range':>9s} " + " ".join(f"{'q%g'%q:>10s}" for q in QS) + ("   gate" if gate else ""))
for i, n in enumerate(spec.OUT):
    d = np.abs(b[:, i] - a[:, i]).ravel()
    d = np.where(np.isnan(d), np.inf, d)
    rng = max(a[:, i].max() - a[:, i].min(), 1e-30)
    q = [np.percentile(d, x) for x in QS]
    line = f"{n:9s} {rng:9.3g} " + " ".join(f"{x:10.2e}" for x in q)
    if gate:
        lim = [max(2 * m, 1e-6 * env[n]["range"]) for m in env[n]["member_max"]]
        ok = [x <= l for x, l in zip(q, lim)]
        status = "PASS" if all(ok) else ("PART" if all(ok[:3]) else "FAIL")
        if status == "FAIL":
            nfail += 1
        line += "   " + status
    print(line)
if gate:
    print("fields failing at q99 or below:", nfail, "of", NV)
