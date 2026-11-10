"""Noah-MP 5.0 closed-loop flux comparison, Fortran LIS vs JAX (roadmap item 10).

usage: python3 flux_compare.py <fortran lisout> <jax lisout> [out.md]

Compares the instantaneous 6-hourly fluxes in every history file pair, over tiles valid in both:
all cells, snow cells (Fortran SWE_inst > 1 kg/m2), frozen-soil cells (Fortran SoilTemp layer 1 < 273.15 K), and
the surface energy-balance residual R = Swnet + s*Lwnet - Qle - Qh - Qg, with the sign s (+1/-1) chosen as the one
that closes better in the Fortran output (the LIS Lwnet sign convention is not assumed).

THRESHOLDS were fixed on 2026-11-10 BEFORE any 5.0 closed-loop flux result had been looked at; do not tune them
after seeing results. A breach is reported as FAIL, not adjusted. Units: W m-2 (Evap: kg m-2 s-1).
"""
import glob, os, sys
import numpy as np
import netCDF4 as nc

VARS = ["Qle", "Qh", "Qg", "Swnet", "Lwnet", "Evap"]
# p99 of |JAX - Fortran| over all valid cells, and over snow / frozen cells (allowed to be 3x larger)
THRESH = {"Qle": 5.0, "Qh": 10.0, "Qg": 5.0, "Swnet": 1.0, "Lwnet": 2.0, "Evap": 2e-6}
COND_FACTOR = 3.0
P50_MAX = {"Qle": 0.1, "Qh": 0.1, "Qg": 0.1, "Swnet": 0.01, "Lwnet": 0.05, "Evap": 5e-8}   # median |diff|
CLOSURE_P99 = 1.0        # p99 of |R_jax - R_fortran| (W m-2)
BAD = -9000.0


def rd(f, v):
    d = nc.Dataset(f)
    a = np.ma.filled(d[v][:], np.nan).astype(np.float64)
    d.close()
    return np.where(a < BAD, np.nan, a)


def main():
    fo, jx = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    files = sorted(glob.glob(f"{fo}/SURFACEMODEL/*/LIS_HIST_*.nc"))
    acc = {k: [] for k in VARS}
    cond = {c: {k: [] for k in VARS} for c in ("snow", "frozen")}
    rf, rj = {1: [], -1: []}, {1: [], -1: []}
    n = 0
    for f in files:
        g = f.replace(fo, jx)
        if not os.path.exists(g):
            continue
        n += 1
        F = {v: rd(f, v + "_inst") for v in VARS}
        J = {v: rd(g, v + "_inst") for v in VARS}
        swe = rd(f, "SWE_inst")
        st1 = rd(f, "SoilTemp_inst")[0]
        ok = np.ones(swe.shape, bool)
        for v in VARS:
            ok &= np.isfinite(F[v]) & np.isfinite(J[v])
        masks = {"snow": ok & (swe > 1.0), "frozen": ok & (st1 < 273.15)}
        for v in VARS:
            d = np.abs(J[v] - F[v])
            acc[v].append(d[ok])
            for c, m in masks.items():
                cond[c][v].append(d[m])
        for s in (1, -1):
            rf[s].append((F["Swnet"] + s * F["Lwnet"] - F["Qle"] - F["Qh"] - F["Qg"])[ok])
            rj[s].append((J["Swnet"] + s * J["Lwnet"] - J["Qle"] - J["Qh"] - J["Qg"])[ok])
    cat = lambda l: np.concatenate(l) if l else np.array([])
    q = lambda a, p: float(np.percentile(a, p)) if a.size else float("nan")
    L = [f"# 5.0 closed-loop flux comparison ({n} history file pairs)", "",
         "| flux | p50 | p99 | max | p99 limit | snow p99 | frozen p99 | cond limit | result |", "|---|---|---|---|---|---|---|---|---|"]
    allok = True
    for v in VARS:
        a, s_, z = cat(acc[v]), cat(cond["snow"][v]), cat(cond["frozen"][v])
        lim = THRESH[v]
        ok_ = q(a, 99) <= lim and q(a, 50) <= P50_MAX[v] and (s_.size == 0 or q(s_, 99) <= COND_FACTOR * lim) \
            and (z.size == 0 or q(z, 99) <= COND_FACTOR * lim)
        allok &= ok_
        L.append(f"| {v} | {q(a,50):.3g} | {q(a,99):.3g} | {a.max() if a.size else float('nan'):.3g} | {lim:g} | "
                 f"{q(s_,99):.3g} (n={s_.size}) | {q(z,99):.3g} (n={z.size}) | {COND_FACTOR*lim:g} | {'pass' if ok_ else 'FAIL'} |")
    s = min((1, -1), key=lambda k: np.nanmedian(np.abs(cat(rf[k]))))
    resf, resj = cat(rf[s]), cat(rj[s])
    dres = np.abs(resj - resf)
    cl = q(dres, 99) <= CLOSURE_P99
    allok &= cl
    L += ["", f"Energy balance R = Swnet {'+' if s == 1 else '-'} Lwnet - Qle - Qh - Qg (sign chosen by Fortran closure).",
          f"Fortran |R| p50 {q(np.abs(resf),50):.3g}, p99 {q(np.abs(resf),99):.3g}; JAX |R| p50 {q(np.abs(resj),50):.3g}, p99 {q(np.abs(resj),99):.3g}; "
          f"|R_jax - R_fortran| p99 {q(dres,99):.3g} (limit {CLOSURE_P99:g}): {'pass' if cl else 'FAIL'}",
          "", f"Overall: {'PASS' if allok else 'FAIL'} against thresholds fixed before the run (see script header)."]
    txt = "\n".join(L)
    print(txt)
    if out:
        open(out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
