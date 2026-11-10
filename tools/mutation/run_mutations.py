"""Mutation check of the Noah-MP 5.0 replay tests: apply one edit at a time to a scratch copy of the package, run the chosen pytest selection, report caught / NOT caught.
usage: python tools/mutation/run_mutations.py [-k EXPR] [M1 M3 ...]       (default selection: test_noahmp50_snow; CPU, JAX_PLATFORMS=cpu)
The original tree is never modified. A mutation is CAUGHT when pytest fails."""
import os, re, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOD = "src/lisf_noahmp_jax/physics/noahmp50_jax.py"
# id: (description, old text, new text); each old text must occur exactly once
MUTATIONS = {
    "M1": ("HVAP -0.4%", "HVAP = 2.5104e06", "HVAP = 2.5004e06"),
    "M2": ("SB +0.18%", "SB = 5.67e-8", "SB = 5.68e-8"),
    "M3": ("delete bottom-layer snow release", 'jnp.fmax((vol_liq - epore) * dzs, P["SNOW_RET_FAC"] * dt * qo)', 'jnp.fmax((vol_liq - epore) * dzs, 0.0 * qo)'),
    "M6": ("TKICE +4.5%", "TKICE = 2.2", "TKICE = 2.3"),
    "M8": ("SNLIQMAXFRAC halved", 'fmx = P["SNLIQMAXFRAC"]', 'fmx = 0.5 * P["SNLIQMAXFRAC"]'),
}


def run(mid, sel, py):
    desc, old, new = MUTATIONS[mid]
    tmp = tempfile.mkdtemp(prefix=f"mut_{mid}_")
    try:
        for d in ("src", "tests", "tools"):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(tmp, d), ignore=shutil.ignore_patterns("__pycache__"))
        p = os.path.join(tmp, MOD); s = open(p).read()
        if s.count(old) != 1:
            return mid, desc, f"SKIPPED (pattern found {s.count(old)} times)", 0
        open(p, "w").write(s.replace(old, new))
        env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=os.path.join(tmp, "src"))
        t = time.time()
        r = subprocess.run([py, "-m", "pytest", "tests/test_regression.py", "-k", sel, "-q", "-x", "--no-header", "-p", "no:cacheprovider"], cwd=tmp, env=env,
                           capture_output=True, text=True)
        msg = [l for l in r.stdout.splitlines() if l.startswith("E ") or "passed" in l or "failed" in l]
        return mid, desc, ("CAUGHT" if r.returncode else "NOT caught"), time.time() - t, " | ".join(msg[-3:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    a = sys.argv[1:]; sel = "test_noahmp50_snow"
    if "-k" in a:
        i = a.index("-k"); sel = a[i + 1]; del a[i:i + 2]
    ids = a or ["M1", "M3", "M6", "M8"]
    for m in ids:
        print(*run(m, sel, sys.executable), sep="  ", flush=True)
