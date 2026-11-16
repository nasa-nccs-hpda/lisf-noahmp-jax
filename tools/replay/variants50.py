"""Run the 5.0 dump LIS and the JAX replay for Noah-MP 5.0 option variants and print a summary table.
usage: python lis/variants50.py TAG KEY=VALUE [KEY=VALUE ...] [--domain sgp|conus]
KEY is (@label for a raw config label) the option label of the LIS config without the 'Noah-MP.5.0 ' prefix and without the colon, e.g. 'surface layer drag coefficient option=2'.
SGP: Feb 1 2017 (env VAR_DAY=n: Feb n) (24 h, 616 tiles, mostly snow free); CONUS: Feb 1 0-3 h, snow tiles only (JAXDUMP_SNOWONLY, 12 steps)."""
import os, re, subprocess, sys
RUN = os.environ["LISF_RUN"]
ENV = os.path.join(os.environ["LISF_DEPS"], "env_lis.sh")
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
FIELDS = "TV TG FSH SSOIL SMC STC SWE PSN RUNSB TAH".split()


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    dom = "sgp"
    if "--domain" in sys.argv:
        dom = sys.argv[sys.argv.index("--domain") + 1]; a = [x for x in a if x != dom]
    tag, kv = a[0], dict(x.split("=", 1) for x in a[1:])
    d = f"{RUN}/mp50v_{dom}_{tag}"
    os.makedirs(d + "/lisout", exist_ok=True)
    for l in ("INPUT", "ldt_mp5", "ldt_mp5c"):
        if not os.path.exists(f"{d}/{l}"):
            os.symlink(f"{RUN}/{l}", f"{d}/{l}")
    txt = open(f"{RUN}/lis.config.mp5").read()
    txt = re.sub(r"^Output directory:.*$", "Output directory: ./lisout", txt, flags=re.M)
    txt = re.sub(r"^Diagnostic output file:.*$", "Diagnostic output file: ./lisout/lislog", txt, flags=re.M)
    txt = re.sub(r"^Ending day:.*$", "Ending day: " + ("2" if dom == "sgp" else "1"), txt, flags=re.M)
    txt = re.sub(r"^Output model restart files:.*$", "Output model restart files: 0", txt, flags=re.M)
    if dom == "conus":
        txt = re.sub(r"^Ending hour:.*$", "Ending hour: 3", txt, flags=re.M)
        txt = txt.replace("ldt_mp5/out/lis_input.sgp.noahmp5.nc", "ldt_mp5c/out/lis_input.conus.noahmp5.nc")
    if os.environ.get("VAR_DAY"):                           # run the SGP day VAR_DAY (Feb 2017) instead of Feb 1 (wetter days exercise the runoff schemes)
        dd = int(os.environ["VAR_DAY"])
        txt = re.sub(r"^Starting day:.*$", f"Starting day: {dd}", txt, flags=re.M)
        txt = re.sub(r"^Ending day:.*$", f"Ending day: {dd + 1}", txt, flags=re.M)
        if os.environ.get("VAR_MONTH"):                     # VAR_MONTH=m: month of the replay day (default Feb)
            for w in ("Starting", "Ending"):
                txt = re.sub(rf"^{w} month:.*$", f"{w} month: {int(os.environ['VAR_MONTH'])}", txt, flags=re.M)
    for k, v in kv.items():
        key = k[1:] + ":" if k.startswith("@") else f"Noah-MP.5.0 {k}:"       # a key starting with @ is a raw label of the LIS config
        pat = re.compile(r"^" + re.escape(key) + r"\s*\S+(.*)$", re.M)
        if k.startswith("@") and not pat.search(txt):                 # raw label absent from the base config: add it after the LAI source line
            txt = re.sub(r"^(LAI data source:.*)$", lambda m: m.group(1) + "\n" + key + " " + v, txt, count=1, flags=re.M)
            continue
        assert pat.search(txt), key
        txt = pat.sub(lambda m: f"{key} {v}" + m.group(1), txt, count=1)
    open(f"{d}/lis.config", "w").write(txt)
    env = dict(os.environ)
    if dom == "conus":
        env.update(JAXDUMP_SNOWONLY="1", JAXDUMP_MAXSTEP="12")
    if not os.path.exists(f"{d}/jaxdump_mp.bin"):
        subprocess.run(f"source {ENV}; cd {d}; rm -f jaxdump*; {RUN}/{os.environ.get('LIS_BIN', 'LIS_mp50_dump')} -f lis.config > lis.out 2>&1", shell=True, executable="/bin/bash", env=env, check=False)
    if "LIS Run completed" not in open(f"{d}/lisout/lislog.0000").read():
        print(tag, dom, "FORTRAN RUN FAILED:", open(f"{d}/lis.out").read()[-300:]); return
    env2 = dict(os.environ, MPVER="50", JAX_PLATFORMS="cpu")
    out = subprocess.run([sys.executable, f"{HERE}/replay_mp50.py", d, "300000", "0", "rand:15000"], capture_output=True, text=True, env=env2, cwd=PROJ)
    print(f"=== {tag} [{dom}] {kv}")
    lines = out.stdout.split("\n")
    print([l for l in lines if l.startswith("options")][0] if any(l.startswith("options") for l in lines) else out.stderr[-600:])
    print(f"{'field':8s} {'range':>9s} {'p50':>9s} {'p99':>9s} {'p99.9':>9s} {'max':>9s} {'bitwise%':>8s} nan")
    for l in lines:
        t = l.split()
        if t and t[0] in FIELDS:
            print(l)


if __name__ == "__main__":
    main()
