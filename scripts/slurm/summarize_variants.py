"""Summary of one run_variants.sh run: wall times from the job logs and p99/max differences from lisjax/compare_hist.py.
usage: summarize_variants.py RUN_DIR LISF_RUN NAME     (RUN_DIR = gpu_runs/NAME)"""
import re, subprocess, sys, os
run, lisf, name = sys.argv[1:4]
CASES = {"pair_clsm_1d": ("conus_clsm1d", f"lisout_{name}", f"jax_{name}"), "pair_hymap_2d": ("conus_hy", f"lisout_{name}", f"jax_{name}"),
         "pair_n32_2d": ("conus_n32_2d", f"lisout_{name}", f"jax_{name}"), "pair_mp36_2d": ("conus_mp36_2d", f"lisout_{name}", f"jax_{name}"), "pair_n271_2d": ("conus_n271_2d", f"lisout_{name}", f"jax_{name}"), "pair_mos_2d": ("conus_mos_2d", f"lisout_{name}", f"jax_{name}"), "pair_n33_2d": ("conus_n33_2d", f"lisout_{name}", f"jax_{name}"), "jax_mp_7d": ("conus_mp7", "lisout", f"jax_{name}")}
print(f"# Summary of {name}\n\nNot bitwise repeatable in time; read p99, not equality. Compare lines show range, p50, p99, max, bitwise%.\n")
print("| case | Fortran wall | JAX total (s) | JAX compile+first chunk (s) | job log |\n|---|---|---|---|---|")
for k, (case, f, j) in CASES.items():
    log = f"{run}/{k}.out"
    if not os.path.exists(log):
        continue
    t = open(log).read()
    m = re.search(r"Elapsed \(wall clock\) time.*?: (\S+)", t); b = re.search(r"'total': ([\d.]+)", t) or re.search(r"wrote history\s+\(([\d.]+)s elapsed\)(?![\s\S]*wrote history)", t) or re.search(r'"total_s": ([\d.]+)', t)
    c = re.search(r"'chunk_device_s': \[([\d.]+)", t) or re.search(r'"first_chunk_device_s": ([\d.]+)', t)
    print(f"| {k} | {m.group(1) if m else 'n/a'} | {b.group(1) if b else 'FAILED?'} | {c.group(1) if c else 'n/a'} | {log} |")
for k, (case, f, j) in CASES.items():
    d = f"{lisf}/{case}"
    if not (os.path.isdir(f"{d}/{f}") and os.path.isdir(f"{d}/{j}")):
        print(f"\n## {k}: output directories missing, no comparison"); continue
    print(f"\n## {k}: JAX against Fortran\n```")
    r = subprocess.run([sys.executable, "lisjax/compare_hist.py", f"{d}/{f}", f"{d}/{j}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print(r.stdout.strip()); print("```")
