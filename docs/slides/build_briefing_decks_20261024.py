"""Build the three LISF-to-JAX briefing decks of 2026-10-24: management, science, developer (Noah-MP focus, with figures from make_figures.py).
Same layout layer and style as the ROCKE-3D/JAX decks of 2026-10-19 (rocke3d_jax/status_slides).

usage (reportlab and Pillow; matplotlib for the figures; no network):  python3 make_figures.py images; python3 build_briefing_decks_20261024.py
Every number is typed in from LOG.md, BENCHMARKS.md or gpu_runs/*.out of this directory (named in each slide footer).
Nothing is read from run files, so the decks do not change when the data does. A text block that does not fit raises.
"""
import os
from reportlab.pdfgen import canvas
import deck_layout as L
from deck_layout import (AMBER, BLUE, CARD2, GREEN, INK, INK2, MUTED, ORANGE, W, H, MARGIN, Page, banner, card, footer, header)

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-10-24"
IMG = os.path.join(HERE, "images")


def _page(title, subtitle, eyebrow, size=23):
    p = Page(title)
    return p, header(p, eyebrow, title, subtitle, size)


def _metric(p, x, y, w, h, value, label, detail, color=GREEN):
    card(p, x, y, w, h, color)
    p.text(x + 15, y + 6, w - 30, value, 20, INK, True, maxh=26)
    p.text(x + 15, y + 31, w - 30, label, 9.8, INK2, True, maxh=14)
    p.text(x + 15, y + 47, w - 30, detail, 8.8, MUTED, maxh=h - 50)


def _cards(p, x, y, w, h, title, items, accent=BLUE, size=10.6, gap=7):
    card(p, x, y, w, h, accent)
    p.text(x + 15, y + 11, w - 30, title, 12.5, INK, True, maxh=22)
    yy = y + 36
    for it in items:
        sub = it.startswith("> ")   # sub-bullet: indented under the item above
        if sub:
            it = it[2:]
        ind = 18 if sub else 0
        p.text(x + 16 + ind, yy, 12, "-" if sub else "+", size, accent, True)
        yy += p.text(x + 30 + ind, yy, w - 46 - ind, it, size - (0.6 if sub else 0), INK2) + gap
    if yy - gap > y + h:
        L.OVERFLOWS.append("%s: card '%s' needs %.0f pt, box %.0f pt" % (p.name, title, yy - gap - y, h))


def _save(path, title, slides):
    L.OVERFLOWS.clear()
    c = canvas.Canvas(path, pagesize=(W, H)); c.setTitle(title); c.setAuthor("LISF JAX port (G. Tamkin)")
    prev = os.path.join(HERE, "preview_" + os.path.basename(path).split("_")[0]); os.makedirs(prev, exist_ok=True)
    for i, make in enumerate(slides, 1):
        p = make(i); p.to_pdf(c); p.to_png(os.path.join(prev, "slide_%02d.png" % i))
    c.save()
    for q in L.OVERFLOWS:
        print("OVERFLOW:", q)
    if L.OVERFLOWS:
        raise RuntimeError("%d layout overflow(s) in %s" % (len(L.OVERFLOWS), path))
    print("wrote", path, len(slides), "slides")


def bars(p, x, y, w, rows):
    """Horizontal bars on a log scale: rows = (label, fortran_s, jax_s, note)."""
    import math
    lo, hi = math.log10(10.0), math.log10(30000.0)
    lw = 190
    for i, (lab, f, j, note) in enumerate(rows):
        yy = y + i * 42
        p.text(x, yy + 2, lw - 10, lab, 10.2, INK, True, maxh=30)
        for k, (v, col, nm) in enumerate(((f, ORANGE, "Fortran"), (j, GREEN, "JAX"))):
            bw = max(4, (math.log10(v) - lo) / (hi - lo) * (w - lw - 150))
            p.rect(x + lw, yy + k * 18, bw, 14, col, 2)
            p.text(x + lw + bw + 6, yy + k * 18 + 1, 140, "%s %s" % (nm, ("%.0f s" % v) if v < 1000 else ("%.0f s (%.0f min)" % (v, v / 60))), 8.6, INK2, maxh=14)
        p.text(x + lw, yy + 32, w - lw, note, 7.6, MUTED, maxh=10)


# ===================================================================================================================
def management():
    def m1(n):
        p, y = _page("LIS on JAX: where the project stands",
                     "Goal: run NASA's Land Information System (LISF, Fortran) on GPUs through a JAX rewrite that reproduces the real model, at least 4x faster.", "MANAGEMENT BRIEFING | %s" % DATE, 24)
        p.rect(MARGIN, y + 6, 540, 290, L.CARD, 8)
        p.image(MARGIN + 6, y + 10, 528, 250, os.path.join(IMG, "fig_speedup.png"))
        p.text(MARGIN + 12, y + 262, 516, "Noah-MP (versions 3.6, 4.0.1, 5.0); same simulation and output files in each pair. CONUS 89 d: Fortran ran 59 of the 89 days (10 h 35 min) and is scaled.", 8.4, MUTED, maxh=28)
        mx, mw = MARGIN + 540, W - 2 * MARGIN - 540
        _metric(p, mx, y + 6, mw, 70, "2.4x - 166x", "Noah-MP speed-up range", "Goal was 4x. Smallest: 5.0 global 1 d on CPU only (2.4x); CONUS 89 d is 166x.", GREEN)
        _metric(p, mx, y + 82, mw, 70, "Qle p99 0.038 W/m2", "Latent heat, CONUS 89 days", "Range 571 W/m2; maxima (84 W/m2) sit on single snow-melt tiles.", GREEN)
        _metric(p, mx, y + 158, mw, 70, "Reduced scope", "Not a full LIS", "Noah-MP, Noah, Mosaic, CLSM; two forcings; no VIC/CLM2/RUC.", ORANGE)
        banner(p, MARGIN, y + 290, W - 2 * MARGIN, 62,
               "Bottom line: Noah-MP (including the 5.0 version of primary interest) reproduces the real Fortran model closely and runs 2.4 to 172 times faster than one CPU core for the cases tested (8.7 times faster than 40-core MPI Fortran on the 60-day CONUS case) (5.0 on CONUS: 60 d 172x, 150 d about 140x). It is a research result for named cases; other land models, other forcings, MPI and LVT are not ported.", AMBER, 11)
        p.text(MARGIN, y + 362, W - 2 * MARGIN, "Fortran = real LIS 7.8 built with GNU 14.2, serial, one core. JAX = V100 unless stated. Every row compares the same simulation and the same output files.", 9.5, MUTED, maxh=30)
        footer(p, n, "BENCHMARKS.md; LOG.md entries 2026-10-22 (global Jan, HyMAP2 2-day, CONUS 7-day, CLSM 1-day); gpu_runs/clsm_bench_37944930.out")
        return p

    def m2(n):
        p, y = _page("How the project got here", "Autonomous run, 21-24 Oct: reference first, then one model family at a time, each checked against the real Fortran output.", "01 | JOURNEY", 23)
        steps = [
            ("1  PLAN", "Mentor-session plan, owner decisions: done = one simulated day, purpose = speed, goal 4x.", "Noah 3.9 first, dew gate accepted.", MUTED),
            ("2  REFERENCE", "LISF built with GNU 14.2 (serial) so every JAX result has a real run to match.", "Three serial-build bugs patched in the build copy only.", BLUE),
            ("3  LAND MODELS", "Noah 3.9/3.6, Noah-MP 4.0.1 + land ice, ensembles, perturbations, restart.", "Global and CONUS; CONUS 30 d: 82.6x.", GREEN),
            ("4  DA, ROUTING, CLSM", "Soil-moisture EnKF, HyMAP2 river routing, Catchment LSM F2.5.", "13.2x routing 2 d; CLSM 13.6x 1 d.", GREEN),
            ("5  NOAH-MP 5.0", "5.0 with land ice and ~25 option variants; CONUS 60 d, Feb-Jul and SGP year checked.", "10.6x SGP year; CONUS 60 d 172x vs 1 core (8.7x vs 40-core MPI), 150 d ~140x vs 1 core.", GREEN),
        ]
        cw, gap = 166, 14
        for i, (t_, txt, res, col) in enumerate(steps):
            x = MARGIN + i * (cw + gap)
            card(p, x, y + 12, cw, 210, col)
            p.text(x + 13, y + 24, cw - 24, t_, 10.5, col, True, maxh=16)
            p.text(x + 13, y + 48, cw - 24, txt, 10.8, INK2, maxh=96)
            p.rect(x + 13, y + 150, cw - 26, 1, L.RULE)
            p.text(x + 13, y + 158, cw - 24, res, 9.8, col, True, maxh=56)
        banner(p, MARGIN, y + 236, W - 2 * MARGIN, 58, "Owner decisions: files stay in the projects/LISF clone; nothing is committed or pushed; the assistant chooses models and tolerances; reduced scope is acceptable; keep going without waiting for review.", AMBER, 10.4)
        footer(p, n, "LOG.md (2026-10-21 to 2026-10-22); GOAL.md; ACCEPTANCE_CRITERIA.md; jax-port/README.md")
        return p

    def m3(n):
        p, y = _page("What the evidence says, and does not say", "Speed is not the only gate: scope, repeat runs and the float32 caveat decide how far to trust it.", "02 | EVIDENCE AND LIMITS", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 270, "SUPPORTED BY MEASUREMENTS", [
            "Noah-MP cases compared with real Fortran output file by file (CONUS 89 d: 236 files, Qle p99 3.8e-2 W/m2); forcing diagnostics are bitwise equal.",
            "One-ulp test: moving air temperature by 3e-5 K in JAX alone gives differences as large as Fortran vs JAX (Qle p99 5.7e-2, max 143 W/m2): the outliers behave like chaos, not a porting error (tile overlap not checked).",
            "CLSM core replayed on all 76,088 tiles from the exact Fortran state: agrees to float32 rounding.",
            "Noah-MP 5.0 land ice replayed on 372 Fortran calls at float32 level; about 25 option variants at noise.",
        ], GREEN, 11.2)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 270, "NOT ESTABLISHED", [
            "About 0.25% of CLSM tiles (cold, partly frozen soil) amplify a 1 J/m2 rounding difference to 3 K in one step; maxima there (Qle 82 W/m2) say little about the port.",
            "One Fortran build, one compiler. Speed-ups are vs one core; vs 40-core MPI the 60 d CONUS case is 8.7x.",
            "Noah-MP 5.0 on CONUS 60 days: 172x (1 core), SWE p99 0.023 mm; Feb-Jul 150 d: JAX 790 s, ~140x, Apr-May segments agree; other models over months NOT KNOWN. 5.0 options ported since (runoff 6-8, dveg 2/5/6, tile drainage, irrigation, crop 1-5) are checked on SGP days only; dveg 7-9 (input LAI) is in the runner and matches the real LIS in a 20 d closed loop.",
            "12-member ensemble CONUS 1 day: 19.5x; spread itself not compared. 5.0 assimilation has no Fortran reference.",
            "CLSM from a summer coldstart is NOT reproduced (Qle p99 31 W/m2 after 24 h); only NLDAS-2 and MERRA-2 forcing.",
        ], ORANGE, 11.2)
        banner(p, MARGIN, y + 290, W - 2 * MARGIN, 48, "Scope: winter 2017 cases (Feb CONUS, Jan global), V100 and Xeon Gold 6248, float32 as in LIS. Do not generalize to other seasons, regions or resolutions.", AMBER, 10.4)
        footer(p, n, "LOG.md 2026-10-22 ~08:30 and ~09:10; BENCHMARKS.md; gpu_runs/conus_ens_37944928.out")
        return p

    def m4(n):
        p, y = _page("Decisions needed from management", "The next work depends on what the speed-up is for and how much of LIS has to be covered.", "03 | DECISIONS", 23)
        cw, gap = (W - 2 * MARGIN - 15) / 2, 15
        _cards(p, MARGIN, y + 6, cw, 150, "1  TARGET USE", ["Which LIS configuration matters (land model, forcing, region, run length)? Start-up compile is 20 to 196 s; long runs hide it (CONUS 89 d: 345 s).", "GPU ensembles or long single runs? The 12-member ensemble is 19.5x."], BLUE, 11)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 150, "2  ACCURACY BAR", ["Accept p99-based tolerances because of the freeze/thaw-sensitive tiles, or require a stricter measure?", "Who reviews the tolerances? The assistant chose them."], ORANGE, 11)
        _cards(p, MARGIN, y + 170, cw, 150, "3  SCOPE", ["Continue with the remaining Noah-MP 5.0 options, then VIC, CLM2, RUC 3.7, other forcings, MPI? (JULES needs an external library not in the repository)", "Each needs its own reference run and parameter data."], GREEN, 11)
        _cards(p, MARGIN + cw + gap, y + 170, cw, 150, "4  RELEASE CONTROL", ["Nothing is committed or pushed. Decide where the jax-port code lives and who may run it.", "Only the assistant has run the cases; a second person has not."], BLUE, 11)
        banner(p, MARGIN, y + 334, W - 2 * MARGIN, 46, "Recommended posture: treat it as a validated prototype for named cases, and fund a short qualification before any wider claim.", AMBER, 11)
        footer(p, n, "LOG.md; BENCHMARKS.md; REPORT.md section 9; owner messages of 2026-10-21/22")
        return p

    def m5(n):
        p, y = _page("Proposed gates: from a validated prototype to a decision", "Small, ordered checks; stop or re-scope if the evidence does not support wider use.", "04 | NEXT STEPS", 23)
        rows = [("A", "REPRODUCE", "A second person runs gpu/run_variants.sh and gets the same table within noise; first full runner run finished (jobs 37944931-34).", BLUE),
                ("B", "FIDELITY", "Noah-MP 5.0 beyond Jul, another region, a clean soil map (148 water-soil tiles break both models); LIS scientist reviews the p99 tolerances (89 d Noah-MP 4.0.1 and the one-ulp test are done).", ORANGE),
                ("C", "ENSEMBLES", "Ensemble spread comparison (the 12-member timing is in: 19.5x).", BLUE),
                ("D", "SCOPE", "Choose what to port next: remaining Noah-MP 5.0 options, then VIC, CLM2, RUC 3.7, other forcings, MPI, with their reference runs.", ORANGE),
                ("E", "PROMOTE OR HOLD", "Compare the evidence with the target workload, then decide where the code lives and who may use it.", GREEN)]
        yy = y + 12
        for key, t_, d, col in rows:
            card(p, MARGIN, yy, W - 2 * MARGIN, 58, col)
            p.text(MARGIN + 14, yy + 11, 30, key, 20, col, True, maxh=28)
            p.text(MARGIN + 54, yy + 12, 150, t_, 12, INK, True, maxh=30)
            p.text(MARGIN + 214, yy + 9, W - 2 * MARGIN - 230, d, 12, INK2, maxh=46)
            yy += 66
        banner(p, MARGIN, yy + 2, W - 2 * MARGIN, 44, "No schedule is implied: the owner sets dates and acceptance limits once the target workload is chosen.", AMBER, 10.5)
        footer(p, n, "STATUS_20261022 s4-6; LOG.md; Reports/ACCEPTANCE_CRITERIA.md amendment 2026-10-22")
        return p

    return [m1, m2, m3, m4, m5]


# ===================================================================================================================
def science():
    def s1(n):
        p, y = _page("What is ported, and what is not", "LIS is a framework; the port covers selected land models, forcings, a router and one assimilation scheme.", "SCIENCE BRIEFING | %s" % DATE, 23)
        rows = [["Piece", "Status", "Validated case"],
                ["Noah 3.9 / 3.6 / 3.3 / 3.2 / 2.7.1", "ported (2.7.1: derived tables)", "CONUS 2 d (3.3, 3.2, 2.7.1)"],
                ["Mosaic LSM", "ported", "CONUS 2 d, 76,064 tiles"],
                ["Noah-MP 4.0.1, 5.0 (+ land ice), 3.6", "ported; ~25 option variants checked per version", "4.0.1: CONUS 89 d, SGP 365 d; 5.0: CONUS 150 d (Feb-Jul), SGP 365 d; 3.6: CONUS 2 d"],
                ["CLSM F2.5", "ported (no restart)", "CONUS 1 d, 76,088 tiles"],
                ["HyMAP2 routing", "ported (kinematic, inertial, floodplain)", "CONUS 2 d, 78,554 river cells"],
                ["Ensembles, GMAO forcing perturbation", "ported", "12-member CONUS 1 d (19.5x; mean fields compared)"],
                ["EnKF soil-moisture DA", "ported, Noah-MP", "identical twin"],
                ["Forcing: MERRA-2, NLDAS-2", "ported", "bitwise forcing diagnostics"],
                ["JULES (external library), VIC, CLM2, RUC 3.7", "NOT ported", "-"],
                ["LDT parameter processing (Noah-MP 4.0.1)", "ported", "SGP + CONUS: bitwise identical to LDT (slope/aspect 1e-6)"],
                ["Config-driven restarts (Noah-MP, Noah 3.9/3.6)", "ported", "split run bitwise = straight run; Fortran reads JAX restart"],
                ["Other forcings, MPI, LVT, LDT for other models", "NOT ported", "-"],
                ["Routing: reservoirs, evaporation, 2-way", "NOT ported", "-"]]
        p.table(MARGIN, y + 6, [330, 290, W - 2 * MARGIN - 620], rows, 10, maxh=300)
        banner(p, MARGIN, y + 322, W - 2 * MARGIN, 52, "The reference is the real LIS built from the repository with GNU 14.2, serial; setups follow the public LIS data portal (parameters in /css/lisdata-pub).", AMBER, 10)
        footer(p, n, "LOG.md; REPORT.md section 9; memory file lisf-jax-port-status")
        return p

    def s2(n):
        p, y = _page("How agreement is measured", "Three layers, from strict to practical. Plain wording, not reviewed by a domain scientist.", "01 | METHOD", 23)
        cw, gap = (W - 2 * MARGIN - 20) / 3, 10
        _cards(p, MARGIN, y + 6, cw, 250, "1  CALL-LEVEL REPLAY", ["Record every argument of the Fortran routine before and after the call (all tiles).", "Feed the same state to the JAX routine, compare intermediates and result.", "Isolates porting errors from drift."], BLUE, 10.6)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 250, "2  HISTORY FILES", ["Run both models over the same period; compare every output variable per tile.", "Report range, median, p99, maximum and bitwise share.", "NaNs found only in the new run are counted."], GREEN, 10.6)
        _cards(p, MARGIN + 2 * (cw + gap), y + 6, cw, 250, "3  SENSITIVITY TEST", ["Add rounding-size noise to the Fortran state and rerun the JAX routine.", "If that alone gives the observed differences, they are model sensitivity, not a porting error.", "Done for CLSM ground heat."], ORANGE, 10.6)
        banner(p, MARGIN, y + 272, W - 2 * MARGIN, 52, "Judge by p99 and fraction of tiles within tolerance. Maxima are dominated by a few tiles at freeze/thaw thresholds.", AMBER, 10.6)
        footer(p, n, "LOG.md 2026-10-22 ~08:30; lisjax/compare_hist.py; lis/replay_clsm.py")
        return p

    def s3(n):
        p, y = _page("Results against real Fortran LIS", "p99 absolute difference over all tiles and output times (range of the variable in brackets).", "02 | RESULTS", 23)
        rows = [["Case", "Qle W/m2", "Qh W/m2", "Qg W/m2", "Other"],
                ["Noah-MP CONUS 7 d", "0.0075", "0.155", "0.183", "SWE 2.6e-4 kg/m2; soil moisture 1.2e-5"],
                ["Noah-MP SGP 365 d", "0.128", "0.132", "0.129", "soil moisture 3.7e-4; SWE 5e-5; (5.0: states only: 6.6e-4; 3.4e-5)"],
                ["Noah 3.6 Step 2 365 d", "0.088", "0.139", "0.234", "Fortran vs NASA target: 0.075 / 0.112 / 0.191"],
                ["Noah-MP global Jan 5 d", "0.0025 (max 1.4)", "0.040 (max 3.3)", "0.046 (max 3.8)", "SWE 3.2e-5; soil temp 2.9e-3 K"],
                ["Noah-MP CONUS 89 d", "0.038 (max 84)", "0.092 (max 168)", "0.127 (max 228)", "soil moisture 1.3e-4; soil temp 9.4e-3 K; SWE 2.8e-2"],
                ["Noah-MP 5.0 CONUS 1 d / SGP 365 d", "-", "-", "-", "1 d: SWE 1.7e-5, soil temp 1.6e-3 K; year: soil moisture 6.6e-4, soil temp 4.4e-2 K"],
                ["CLSM F2.5 CONUS 1 d", "0.058 (max 82)", "0.11 (max 112)", "0.069 (max 18)", "AvgSurfT 2.9e-3 K; SWE 7e-7; soil temp 9e-5 K"]]
        p.table(MARGIN, y + 6, [200, 130, 130, 130, W - 2 * MARGIN - 590], rows, 9.2, maxh=190)
        _cards(p, MARGIN, y + 205, W - 2 * MARGIN, 136, "CLSM LARGE MAXIMA: MODEL SENSITIVITY", [
            "About 200 cold, partly frozen tiles (0.25%) turn 1 J/m2 of ground-heat rounding into up to 3 K in tc2 in one step.",
            "Exact Fortran state plus +-8 J/m2 noise on ground heat alone: latent heat max change 116 W/m2, 756 tiles above 2 W/m2.",
            "The real 6 h run shows max 112 W/m2 on 1,339 tiles: the same size, consistent with model sensitivity rather than a porting error (one step, CLSM only).",
            "July coldstart is different: about 1% of the wet Gulf/Florida tiles diverge (Qle p99 31 W/m2 after 24 h); a limit of the port, documented."], ORANGE, 10.6)
        footer(p, n, "BENCHMARKS.md; LOG.md entries 2026-10-22 (global Jan, HyMAP2, 7-day, CLSM)")
        return p

    def s4(n):
        p, y = _page("What this evidence cannot say", "Each statement has a scope; agreement in these cases does not imply broader validity.", "03 | LIMITS", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 250, "SCOPE", ["Winter 2017 only (Feb CONUS, Jan global).", "NLDAS-2 and MERRA-2 forcing only.", "CONUS 1 to 30 days (global 5 days); full year only on the SGP domain: CONUS Noah-MP 89 d done (Qle p99 3.8e-2); other models beyond a month NOT KNOWN.", "Float32 as in LIS; one compiler and one core for the reference."], BLUE, 11)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 250, "INTERPRETATION", ["Tolerances were chosen by the assistant, not reviewed by a LIS scientist.", "Identical-twin DA shows the filter is reproduced, not that it improves estimates.", "Routing uses the model's own runoff; no comparison with gauges.", "Snow, frozen soil: good in the cases run, thresholds amplify differences."], ORANGE, 11)
        footer(p, n, "LOG.md; ACCEPTANCE_CRITERIA.md")
        return p

    def s5(n):
        p, y = _page("Open science questions and the check for each", "Ordered by how much they would change the conclusions.", "04 | NEXT", 23)
        rows = [["Question", "Check"],
                ["Do differences stay bounded beyond 30 days?", "Noah-MP 4.0.1: yes over 89 d (Qle p99 3.8e-2). 5.0: yes, 60 d p99 SoilTemp 2e-2 K; Apr-May segments 1e-3 K"],
                ["Are the Noah-MP outliers chaos? One-ulp test says consistent", "Open: check that the largest-difference tiles coincide between the two pairs"],
                ["Does the ensemble spread match?", "Compare ensemble spread (the mean-field difference is checked)"],
                ["Does the port work in other regions and seasons?", "Global July; another regional domain"],
                ["Is VIC / CLM2 / RUC feasible?", "Each is 17k-40k lines; needs parameter data for a reference run"]]
        p.table(MARGIN, y + 6, [420, W - 2 * MARGIN - 420], rows, 11, maxh=230)
        footer(p, n, "LOG.md; memory file lisf-jax-port-status")
        return p

    def s6(n):
        p, y = _page("Routing and assimilation: what was reproduced", "Both ride on the Noah-MP runoff and states; checks are against the real LIS output.", "02b | ROUTING AND DA", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 250, "HYMAP2 ROUTING (CONUS, 2 DAYS)", ["78,554 river cells; kinematic and inertial flow, floodplain, linear reservoirs; sub-stepping as LIS.", "Streamflow p99 5.5e-4 m3/s (range 197); river storage p99 12 m3 (range 7e6).", "Flooded fraction bitwise in 99% of cells; threshold cells differ by up to 0.42.", "A gfortran MIN/MAX NaN bug in the land model reached the river network and was fixed."], BLUE, 10.6)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 250, "ENKF SOIL-MOISTURE DA", ["Ensemble Kalman filter as in LIS: observation QC, perturbations, bias correction, no localisation.", "Checked as an identical twin with synthetic observations (JAX against Fortran).", "Shows that the filter is reproduced, not that it improves soil moisture estimates.", "Only Noah-MP; other observation types NOT ported."], GREEN, 10.6)
        footer(p, n, "LOG.md (HyMAP2 and DA entries); Reports/REPORT.md section 9")
        return p

    def s7(n):
        p, y = _page("Noah-MP over 89 days: are the outliers a porting error?", "CONUS, Mar 3 - May 1 2017, 236 history files; air temperature moved by one float32 step (3e-5 K) in JAX alone for the second pair.", "02c | LONG RUN AND SENSITIVITY", 22)
        p.rect(MARGIN, y + 6, 540, 232, L.CARD, 8)
        p.image(MARGIN + 6, y + 10, 528, 224, os.path.join(IMG, "fig_agreement_89d.png"))
        _cards(p, MARGIN + 556, y + 6, W - 2 * MARGIN - 556, 232, "READING", ["Median differences are at rounding level; p99 stays below 1e-3 of the range for every field.", "The maxima (Qle 84 W/m2, soil temp 15 K) are single tiles at snow melt and freeze/thaw.", "JAX against itself, with a tiny forcing change, shows maxima of the same size."], GREEN, 9.8)
        banner(p, MARGIN, y + 250, W - 2 * MARGIN, 58, "Consistent with chaotic amplification of float32 rounding, not with a physics error. Not proven: one variable perturbed once, and the tiles of the largest differences were not compared between the two pairs.", AMBER, 10.4)
        footer(p, n, "Reports/BENCHMARKS.md (89-day and one-ulp sections); LOG.md 2026-10-24; jobs 37946353, 37946514, 37946517")
        return p

    return [s1, s2, s3, s7, s6, s4, s5]


# ===================================================================================================================
def developer():
    def d1(n):
        p, y = _page("Orientation: the jax-port directory", "Untracked clone of LISF; the Fortran build copy and run directories live outside the repository.", "DEVELOPER BRIEFING | %s" % DATE, 23)
        rows = [["Path", "Role"],
                ["jax-port/noahmp401_jax.py, noahmp50_jax.py (+ glacier), noah39_jax.py, clsm_jax.py", "single-tile or vectorised physics cores"],
                ["jax-port/lisjax/run_mp.py, run_clsm.py, run.py", "drivers: setup from the LDT file, forcing, scan over time, history"],
                ["lisjax/hymap2.py, da.py, pert.py, forcing.py", "routing, assimilation, perturbations, forcing readers"],
                ["lisjax/compare_hist.py", "history comparison: p50 / p99 / max / bitwise, NaN-only-in-new count"],
                ["lis/dumpio_clsm.py, replay_clsm.py", "reader of the Fortran call dumps; open-loop replay"],
                ["lis_patches/", "build fixes, dump hooks, sbatch files, env_lis.sh"],
                ["LOG.md, BENCHMARKS.md, REPORT.md", "dated log, results table, report"],
                ["dev/lisf_build/lis, dev/lisf_run/", "outside the repo: build copy and run directories"]]
        p.table(MARGIN, y + 6, [400, W - 2 * MARGIN - 400], rows, 10, maxh=270)
        footer(p, n, "jax-port/README.md; LOG.md")
        return p

    def d2(n):
        p, y = _page("Running it", "CPU run for a quick check; V100 through SLURM from the login node.", "01 | RUN", 23)
        p.rect(MARGIN, y + 6, W - 2 * MARGIN, 150, L.CARD, 8)
        cmds = ["# CLSM, 6 h, CPU (31 s):", "JAX_PLATFORMS=cpu python3 -m lisjax.run_clsm BASE OUT lis.config 2017020100 2017020106",
                "# compare with the Fortran history:", "python3 lisjax/compare_hist.py BASE/lisout OUT",
                "# 1-day benchmark, Fortran then JAX on a V100:", "ssh gpulogin1 'cd jax-port && sbatch gpu_bench_clsm.sbatch'"]
        for i, c in enumerate(cmds):
            p.text(MARGIN + 14, y + 14 + i * 22, W - 2 * MARGIN - 28, c, 9.6, GREEN if c.startswith("#") else INK2, False, True, maxh=16)
        _cards(p, MARGIN, y + 170, W - 2 * MARGIN, 150, "TRAPS", [
            "Source jax-port/lis_patches/env_lis.sh before running the Fortran binary (libesmf.so).",
            "Restore dump hooks with \\cp -f from the repo before rebuilding; hook copies are in the build copy only.",
            "Do not pkill -f with a pattern matching your own shell; rm -rf on globs is blocked.",
            "Run directories need working INPUT/ and out/ links (a dangling link made a job fail silently once)."], ORANGE, 10.2)
        footer(p, n, "gpu_bench_clsm.sbatch; LOG.md; lis_patches/env_lis.sh")
        return p

    def d3(n):
        p, y = _page("Bugs and lessons that cost the most time", "Each one cost a validation round; check these first when a new case disagrees.", "02 | LESSONS", 23)
        rows = [["Symptom", "Cause", "Fix"],
                ["NaN tiles after about 30 h (Noah-MP)", "gfortran MIN/MAX return the non-NaN argument; jnp.minimum propagates NaN", "jnp.fmin / jnp.fmax"],
                ["CLSM core mismatch 3 K at step 2", "stride-sampled dump was misleading", "full-domain dump; replay exact"],
                ["Albedo all NaN in JAX", "LIS keeps the last defined value of an instantaneous variable", "carry last non-UDEF value"],
                ["Rainf_f differs", "diagnosed before rain/snow split", "rainf + snowf"],
                ["Fortran DA segfault", "stale dump hook in the build copy", "restore from repo"],
                ["Serial HyMAP2 crashes", "arrays filled only under #if SPMD", "serial_build_fixes_hymap2.diff"],
                ["Large maxima in a float32 land model", "freeze/thaw thresholds amplify rounding", "noise test; report p99"]]
        p.table(MARGIN, y + 6, [270, 400, W - 2 * MARGIN - 670], rows, 9.4, maxh=320)
        footer(p, n, "LOG.md entries 2026-10-22")
        return p

    def d4(n):
        p, y = _page("Benchmark arithmetic", "Compare equal simulations; separate compile from steady state.", "03 | MEASUREMENT", 23)
        rows = [["Case", "Fortran (1 core)", "JAX", "Ratio / saved", "Note"],
                ["CLSM CONUS 1 d", "351 s", "25.8 s (compile 19.9)", "13.6x / 92.7%", "gpu012, V100; steady 0.02 s per 6 h chunk"],
                ["Noah-MP + HyMAP2 2 d", "1,273 s", "96.4 s (compile 82)", "13.2x / 92.4%", "gpu011"],
                ["Noah-MP SGP 365 d", "2,983 s", "286.5 s", "10.4x / 90.4%", "jobs 37946318/338; forcing 185 s, device 90 s"],
                ["Noah-MP 5.0 SGP 365 d", "2,945 s", "277 s", "10.6x / 90.6%", "job 37946426; forcing 166 s, device 101 s"],
                ["Noah-MP CONUS 89 d", "57,400 s*", "345 s", "166x / 99.4%", "*59 d measured 38,095 s, scaled; job 37946514"],
                ["Noah-MP 5.0 CONUS 60 d", "43,944 s", "255 s", "172x (1 core); 8.7x (40-core MPI) / 99.4%", "jobs 37946608/40; Fortran measured"],
                ["Noah-MP 5.0 CONUS 150 d", "110,700 s*", "790 s", "140x / 99.3%", "*121 d measured, scaled; chained segments"],
                ["Noah 3.6 Step 2 SGP 365 d", "2,247 s", "256.8 s", "8.75x / 88.6%", "jobs 37946294/338; forcing 186 s"],
                ["Noah-MP 3.6 CONUS 2 d", "1,263 s", "46.3 s", "27.3x / 96.3%", "job 37946095, gpu006"],
                ["Mosaic CONUS 2 d", "407 s", "18.9 s (V100)", "21.6x / 95.4%", "job 37945004, gpu011"],
                ["Noah 3.3 CONUS 2 d", "881 s", "29.3 s (V100)", "30.1x / 96.7%", "job 37944988, gpu011"],
                ["Noah-MP global 5 d", "1,643 s", "308 s (compile 196)", "5.3x / 81.3%", "JAX on CPU, 12 threads"],
                ["12 members CONUS 1 d", "7,527 s", "385 s", "19.5x / 94.9%", "forcing read 256 s; steady 3.4 s per day"]]
        p.table(MARGIN, y + 6, [200, 130, 180, 110, W - 2 * MARGIN - 620], rows, 8.6, maxh=262)
        banner(p, MARGIN, y + 282, W - 2 * MARGIN, 52, "Short runs are compile-dominated; steady-state numbers are far larger than the wall-clock ratios above. Report both.", AMBER, 10.6)
        footer(p, n, "BENCHMARKS.md; gpu_runs/clsm_bench_37944930.out; gpu_runs/conus_ens_37944928.out; LOG.md")
        return p

    def d5(n):
        p, y = _page("Tests, gates and open work", "What was checked, what was not, and what to do next.", "04 | QUALITY", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 270, "CHECKED", ["History comparison per case, with NaN-only-in-new count.", "Open-loop replays of Fortran calls (Noah-MP 5.0: land and land ice).", "One-ulp sensitivity test (CLSM and Noah-MP 89 d).", "30 regression tests (tests/, 31 min on CPU)."], GREEN, 11)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 270, "OPEN", ["Noah-MP 5.0: runoff 5, CROPCLASS domains, soil option 2+; irrigation not in the runner; Fortran reference only monthly segments after Mar; 148 water-soil tiles unstable in both models; June only 5 days.", "Ensemble spread comparison; 5.0 assimilation has no Fortran reference.", "Outlier tiles of the 89 d run not compared with the one-ulp run.", "JULES, VIC, CLM2, RUC 3.7, other forcings, MPI, LVT.", "Nobody but the author has reproduced a result (REPRODUCE_NOAHMP.md)."], ORANGE, 11)
        footer(p, n, "LOG.md; BENCHMARKS.md")
        return p

    def d6(n):
        p, y = _page("Anatomy of a run", "One jitted scan over time per output interval; physics cores are pure functions over all tiles.", "01b | ARCHITECTURE", 23)
        rows = [["Stage", "Where", "Note"],
                ["setup", "run_*.py SetupX", "reads the LDT NetCDF file; tiles, parameters, initial state"],
                ["forcing", "lisjax/forcing.py (host, NumPy)", "NLDAS-2 / MERRA-2 interpolation as LIS; 5 s of 26 s for CLSM 1 d"],
                ["step", "model_mp.py, make_step() in run_clsm.py", "state in, state and diagnostics out; float32"],
                ["scan", "jax.lax.scan over a 6 h chunk", "history averages accumulate on the device"],
                ["routing", "lisjax/hymap2.py (float64 where Fortran uses real*8)", "called every 30 min; segment_sum for upstream flow"],
                ["DA", "lisjax/da.py", "perturb, QC, analysis after the history write"],
                ["output", "CLSMHist.write / run_mp.py", "LIS history names; last-defined value for instantaneous variables"]]
        p.table(MARGIN, y + 6, [90, 330, W - 2 * MARGIN - 420], rows, 9.6, maxh=290)
        footer(p, n, "lisjax/run_clsm.py; lisjax/run_mp.py; lisjax/hymap2.py; LOG.md")
        return p

    def d7(n):
        p, y = _page("Environment, data and the runner", "Everything outside the repository, and what gpu/run_variants.sh does with it.", "02b | ENVIRONMENT", 23)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 290, "OUTSIDE THE REPOSITORY", ["LISF_RUN: case directories and Fortran binaries (dev/lisf_run).", "LISF_ENV: GNU 14.2 environment script (dev/lisf_deps/env_lis.sh).", "CONDA_BIN, JAX_ENV: graphcast-env, jax 0.5.3.", "Build copy of LIS/LDT with hooks: dev/lisf_build.", "Public data: /css/lisdata-pub/data."], BLUE, 10.6)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 290, "THE RUNNER", ["Two exclusive V100 pairs (Fortran, then JAX on the same node): CLSM 1 d, Noah-MP + HyMAP2 2 d.", "One shared-node JAX run: Noah-MP 7 d against the stored Fortran output.", "A dependent summary job (afterany) writes SUMMARY.md.", "First real run variants_20261022_a: all jobs finished.", "DRY_RUN=1 only writes the batch files."], GREEN, 10.6)
        footer(p, n, "DATA_AND_PATHS.md; gpu/run_variants.sh; STATUS_20261022 s5")
        return p

    def d8(n):
        p, y = _page("The porting recipe that worked (Noah 3.3, Noah 3.2, Mosaic, CLSM)", "Seven steps, in this order; each step has a script in this directory.", "05 | RECIPE", 22)
        rows = [["Step", "What", "Where"],
                ["1", "Diff the new model's source against a ported relative; list real physics differences (comments and #ifdefs removed)", "diff of module_sf_*lsm.F90 (Noah 3.2/3.3/3.6)"],
                ["2", "Fortran reference run with public parameter tables, 6 h first (2 min), then days", "dev/lisf_run/conus_*; LDT config per model"],
                ["3", "Port with flags in the existing code (v36, v33) or a new core (CLSM, Mosaic)", "noah39_jax.py, clsm_jax.py, mosaic_jax.py"],
                ["4", "Dump hook in a build copy, open-loop replay of the real calls on all tiles", "lis_patches/make_*_hook.py, lis/replay_*.py"],
                ["5", "Closed-loop comparison of the history files, p99 first", "lisjax/compare_hist.py"],
                ["6", "If maxima are large: noise test (perturb the Fortran state or the JAX initial state at rounding size)", "Reports/*sensitivity*"],
                ["7", "Add the case to the runner and benchmark the same-node pair", "gpu/run_variants.sh"]]
        p.table(MARGIN, y + 6, [60, 560, W - 2 * MARGIN - 620], rows, 9.6, maxh=290)
        banner(p, MARGIN, y + 312, W - 2 * MARGIN, 52, "Noah 3.2 failed at step 5 (Qle p99 57 W/m2) until the diff of its driver (noah32_main.F90) showed the wind-height scaling ZFAC: diff the driver as well as the physics module.", AMBER, 10)
        footer(p, n, "LOG.md 2026-10-22 (Noah 3.3, Noah 3.2, Mosaic entries); Reports/noah33_20261022; Reports/mosaic_20261022")
        return p

    def d9(n):
        p, y = _page("Where the JAX time goes", "Forcing reads on the host are the largest part of every long run; compile is a fixed cost of 60 to 90 s.", "03b | MEASUREMENT", 23)
        p.rect(MARGIN, y + 6, 560, 200, L.CARD, 8)
        p.image(MARGIN + 6, y + 10, 548, 192, os.path.join(IMG, "fig_jax_time.png"))
        _cards(p, MARGIN + 576, y + 6, W - 2 * MARGIN - 576, 200, "SPEED LEVERS (NOT DONE)", ["Overlap or cache the forcing reads (60% of JAX time).", "Cache compiled code (84 s in the 89 d run).", "Run the global cases on the GPU (both were CPU only).", "Steady-state GPU time for 5.0 on CONUS not measured."], ORANGE, 9.8)
        banner(p, MARGIN, y + 220, W - 2 * MARGIN, 58, "Device time per 6 h chunk is about 0.1 s on CONUS (76k tiles); the 89-day device time of 111 s is 84 s of compile plus 28 s of 356 chunks. Fortran needed 645 s per simulated day.", AMBER, 10.4)
        footer(p, n, "Reports/BENCHMARKS.md; gpu_runs/conus_mp90/jax90_timed_37946514.out; LOG.md 2026-10-24")
        return p

    return [d1, d6, d2, d7, d3, d4, d9, d5, d8]


if __name__ == "__main__":
    _save(os.path.join(HERE, "management_briefing_20261024.pdf"), "LISF on JAX: management briefing", management())
    _save(os.path.join(HERE, "science_briefing_20261024.pdf"), "LISF on JAX: science briefing", science())
    _save(os.path.join(HERE, "developer_briefing_20261024.pdf"), "LISF on JAX: developer briefing", developer())
