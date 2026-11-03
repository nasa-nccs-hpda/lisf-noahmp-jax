"""Build the ensemble-envelope reference from M one-ulp members. Usage: python noise_floor_env.py M out.json [N]
Expects ref_{N}_O2.bin and ref_{N}_m{k}.bin (N defaults to 2000, whose base file is named ref_2000_O2.bin)."""
import sys, json
import numpy as np
import spec
M, outp = int(sys.argv[1]), sys.argv[2]
N, NT, NV = (int(sys.argv[3]) if len(sys.argv) > 3 else 2000), 48, len(spec.OUT)
ld = lambda p: np.fromfile(p, np.float32).reshape(NT, NV, N).astype(np.float64)
base = ld(f"ref_{N}_O2.bin")
QS = (50.0, 90.0, 99.0, 99.9, 100.0)
mem = [ld(f"ref_{N}_m{m}.bin") for m in range(1, M + 1)]
env = {}
for i, n in enumerate(spec.OUT):
    rng = max(base[:, i].max() - base[:, i].min(), 1e-30)
    q = [[float(np.percentile(np.abs(x[:, i] - base[:, i]).ravel(), p)) for p in QS] for x in mem]
    env[n] = dict(range=float(rng), quantiles=list(QS), members=M, member_max=[max(c) for c in zip(*q)])
json.dump(env, open(outp, "w"), indent=1)
print("wrote", outp, "members", M)
