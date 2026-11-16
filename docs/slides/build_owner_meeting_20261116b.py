"""Four slides for the code-owner meeting, rewritten 2026-11-16 for a skeptical audience (replaces owner_meeting_20261116 in content; older decks untouched).
Structure: claim and its weak points / agreement evidence / speed evidence / what we ask. Same layout layer as the briefing decks.
Numbers are typed in from Reports/BENCHMARKS.md, reports/discover_r_case_*.md, reports/flux_compare_mp5_60d_20261110.md, reports/flux_outliers_mp5_60d_20261111.md, reports/independent_test_20261105.md.
usage: python3 build_owner_meeting_20261116b.py   (reportlab + Pillow, no network; a text block that does not fit raises)
"""
import math
import os
import deck_layout as L
from deck_layout import AMBER, BLUE, GREEN, INK, INK2, MUTED, ORANGE, W, MARGIN, banner, footer
from build_briefing_decks_20261024 import _page, _cards, _metric, _save

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-11-16"
L.FOOTER_DATE = DATE
EYE = "OWNER MEETING | %s" % DATE
CW, GAP = (W - 2 * MARGIN - 16) / 2, 16


def slides():
    def s1(n):
        p, y = _page("Noah-MP on a GPU matches LIS and runs 12 to 172 times faster",
                     "For the cases we tested. A prototype by one person with an AI assistant; no one who knows LIS or Noah-MP has reviewed it yet.", EYE, 25)
        mw = (W - 2 * MARGIN) / 4
        for i, (v, lab, det, col) in enumerate([
                ("172x", "faster than 1 Fortran core", "CONUS, 60 days, Noah-MP 5.0: 43,900 s to 255 s on one V100", GREEN),
                ("12.1x", "faster than 20-core MPI", "3,092 s, output verified identical. 8.7x vs 40 cores (output wrong in 2.7% of tiles)", GREEN),
                ("99%", "of tile-values at float32 noise", "CONUS p99: SWE 0.023 mm, soil T 0.021 K", GREEN),
                ("480 / 480", "input files bitwise identical", "Weather read from GRIB equals the Fortran output, on R's case", GREEN)]):
            _metric(p, MARGIN + i * mw, y + 4, mw - 8, 84, v, lab, det, col)
        cy = y + 98
        _cards(p, MARGIN, cy, CW, 206, "WHY IT SHOULD BE BELIEVED", [
            "Real LIS 7.8 (our GNU build and R's Intel build), same inputs, compared tile by tile.",
            "Flux limits were fixed before looking: p99 Qle 0.04, Qh 0.49, Qg 0.63 W/m2 against 5, 10, 5.",
            "R's own case, unchanged but for dates: 76,180 tiles, 60 days, 480 files, against R's 60-rank Intel LIS. Typical tiles agree at rounding level on CPU and A100.",
            "Reruns pass elsewhere: tarball on Nexus, on Discover (CPU 49 tests, A100 example) and in GitHub CI.",
        ], GREEN, 10.4, 7)
        _cards(p, MARGIN + CW + GAP, cy, CW, 206, "WHY YOU SHOULD STAY SKEPTICAL", [
            "No expert review. Most checks were run by the assistant that wrote the code.",
            "Worst single tiles differ far more than typical ones (R's case: SWE 7.5 mm, soil T 9 K) and are not yet examined there.",
            "Scope: Noah-MP, Noah, Mosaic, CLSM; two forcings. No LDT/LVT, VIC, CLM2, RUC, JULES.",
            "Seasons: Feb-Jul 2017, one SGP year, Oct-Nov 2008. 5.0 fluxes: CONUS Feb-Apr only.",
            "Speed baselines are our own builds; R's Fortran wall time is unmeasured.",
        ], ORANGE, 10.4, 7)
        banner(p, MARGIN, y + 312, W - 2 * MARGIN, 40, "What we ask: try to break it. The rest of this deck shows where to push.", AMBER, 11.5)
        footer(p, n, "Reports/BENCHMARKS.md; reports/discover_r_case_60d_20261110.md; flux_compare_mp5_60d_20261110.md; independent_test_20261105.md; tests/README.md")
        return p

    def s2(n):
        p, y = _page("Agreement: two systems, two compilers, two GPU types",
                     "Differences from Fortran, tile by tile, all hourly or 3-hourly output. p99: 99% of values differ by less than this.", EYE, 24)
        cols = [210, 150, 150, 150, 150]
        rows = [["JAX minus Fortran", "CONUS 60 d (V100)", "R's case, CPU p99", "R's case, A100 p99", "R's worst tile (CPU)"],
                ["Snow water equivalent (mm)", "0.023", "0.0007", "0.0016", "7.5"],
                ["Soil temperature (K)", "0.021", "0.0037", "0.0055", "9.0"],
                ["Soil moisture (m3/m3)", "0.0001", "0.0000065", "0.000038", "0.052"],
                ["Qle / Qh / Qg (W/m2)", "0.04 / 0.49 / 0.63", "not in R's output", "-", "-"]]
        sc = (W - 2 * MARGIN) / sum(cols)
        p.table(MARGIN, y + 4, [c * sc for c in cols], rows, size=10.5, maxh=130, bold_cols=(0,), pad=6)
        cy = y + 140
        _cards(p, MARGIN, cy, CW, 200, "THE WORST TILES: WHAT WE FOUND", [
            "CONUS 60 d: 650 of 18 million cell-times differ by over 50 W/m2, all on thin (under 1 mm) near-freezing melting snow.",
            "Started from the Fortran state, JAX matches to 0.002 W/m2.",
            "A 3e-5 K nudge to air temperature (one float32 step) gives the same drift, first seen 1 Feb 18Z.",
            "Float64 is no closer to Fortran than float32.",
        ], ORANGE, 10.2, 7)
        _cards(p, MARGIN + CW + GAP, cy, CW, 200, "OUR READING, AND ITS WEAK POINT", [
            "The snow state is sensitive to the last bit: an intrinsic property, not a port error.",
            "Weak point: we cannot prove the absence of a small physics difference inside that sensitivity.",
            "R's worst tiles (SWE 7.5 mm, soil T 9 K) are not yet looked at; a reviewer with the Fortran will see them fastest.",
        ], BLUE, 10.2, 7)
        banner(p, MARGIN, y + 348, W - 2 * MARGIN, 34, "Typical tiles agree at rounding level on both systems. Rare snow tiles do not, and that is where to look.", AMBER, 10.8)
        footer(p, n, "reports/discover_r_case_60d_20261110.md; flux_compare_mp5_60d_20261110.md; flux_outliers_mp5_60d_20261111.md (first divergence, float64 check)")
        return p

    def s3(n):
        p, y = _page("Speed: large on our case; not yet a fair race on R's",
                     "Same simulation and output files in every pair. Wall time of the whole run, including compile and file reading for JAX.", EYE, 24)
        lo, span, bx, bw = 2.0, 3.0, MARGIN + 190, 330

        def bar(yy, lab, v, col, txt):
            p.text(MARGIN, yy + 1, 180, lab, 10.2, INK, True, maxh=16)
            w = max(4, (math.log10(v) - lo) / span * bw)
            p.rect(bx, yy, w, 18, col, 2)
            p.text(bx + w + 8, yy + 3, 170, txt, 9.6, INK2, maxh=14)
        p.text(MARGIN, y + 2, 500, "CONUS, Noah-MP 5.0, 60 days, 0.125 deg (log scale)", 10.6, GREEN, True, maxh=16)
        bar(y + 30, "Fortran, 1 core", 43900, ORANGE, "43,900 s (12 h)")
        bar(y + 60, "Fortran, 20-core MPI", 3092, ORANGE, "3,092 s (52 min), output verified")
        bar(y + 90, "Fortran, 40-core MPI", 2207, ORANGE, "2,207 s (37 min), output not clean")
        bar(y + 120, "JAX, 1 V100", 255, GREEN, "255 s: 172x / 12.1x / 8.7x")
        p.text(MARGIN, y + 150, 520, "Fortran = LIS 7.8, GNU 14.2, our build. 20-core output is bitwise identical to serial; the 40-core run is wrong in 2.7% of tiles (our build).", 9, MUTED, maxh=24)
        _cards(p, MARGIN + 570, y + 4, W - 2 * MARGIN - 570, 160, "R'S CASE (DISCOVER)", [
            "A100: 253 s = 113 s compile + 104 s GRIB read + 8.5 s model.",
            "16 CPU cores: 17 min 55 s.",
            "Ratio to R's Fortran: unknown, its wall time was not measured.",
        ], BLUE, 10.2, 7)
        _cards(p, MARGIN, y + 182, CW, 118, "WHAT THIS DOES NOT SHOW", [
            "One run per configuration; no repeats, no variance.",
            "Different cases and GPUs (V100, A100) are not a controlled comparison.",
            "Small cases gain less: 2.4x to 10x vs 1 core.",
        ], ORANGE, 10.2, 7)
        _cards(p, MARGIN + CW + GAP, y + 182, CW, 118, "OPEN FAULT, PROBABLY OURS", [
            "Our GNU-built LIS gives different output at 32+ cores than at 20.",
            "R's Intel LIS is bitwise identical at 20, 32 and 40 ranks, so the fault looks to be in our build or patches, not in LIS. Not yet isolated.",
        ], ORANGE, 10.2, 7)
        banner(p, MARGIN, y + 312, W - 2 * MARGIN, 34, "A fair claim needs R's Fortran wall time on R's case. That is ask 2.", AMBER, 11)
        footer(p, n, "Reports/BENCHMARKS.md; reports/discover_r_case_60d_20261110.md; DISCOVER_R_CASE_PLAN.md")
        return p

    def s4(n):
        p, y = _page("What we ask: find the errors, then decide",
                     "One tarball with code, run scripts and every check. We need reviewers and decisions; we are not asking you to adopt it.", EYE, 24)
        rows = [
            ("1 REVIEW", "Read src/.../physics/noahmp50_jax.py against your Fortran; name what looks wrong or missing.", "Noah-MP scientist, 1-2 days"),
            ("2 TIME", "Wall time of R's Fortran 60-day run, to make the speed claim fair.", "R, one number"),
            ("3 REFERENCE", "A summer or other-region case, and the tolerances you accept.", "Owner and data contact"),
            ("4 REPRODUCE", "Run examples/sgp_5d (about 1 minute on CPU, Fortran output included, environment pinned by uv.lock), or R's case. Tell us where our numbers break.", "Engineer, GPU optional"),
            ("5 DECIDE", "Which models and domains first; speed or differentiable/ML use; a named maintainer.", "Owner, one meeting"),
        ]
        yy = y + 4
        for lab, what, who in rows:
            L.card(p, MARGIN, yy, W - 2 * MARGIN, 46, BLUE)
            p.text(MARGIN + 16, yy + 8, 100, lab, 10.5, BLUE, True, maxh=16)
            p.text(MARGIN + 122, yy + 7, 470, what, 10, INK2, maxh=36)
            p.text(MARGIN + 612, yy + 8, W - 2 * MARGIN - 626, who, 9.6, MUTED, True, maxh=32)
            yy += 52
        _cards(p, MARGIN, yy + 6, CW, 104, "DELIVERY AND TERMS", [
            "Tarball (25 MB); the GitHub repo stays private.",
            "Apache-2.0, plus the UCAR Noah-MP terms: no sale, keep NOTICE with every copy.",
        ], GREEN, 10, 6)
        _cards(p, MARGIN + CW + GAP, yy + 6, CW, 104, "NOT PORTED (YOUR CALL)", [
            "Irrigation in the runner, runoff option 5, urban, GECROS, JULES, VIC, CLM2, RUC, LDT/LVT.",
            "Water tiles of soil type 14 (148 in CONUS) are unstable in Fortran too: an input issue, masked.",
        ], ORANGE, 10, 6)
        footer(p, n, "STATUS_20261022.md; Reports/BENCHMARKS.md; NOTICE; deliver/email_to_R_20261116.txt")
        return p

    return [s1, s2, s3, s4]


if __name__ == "__main__":
    _save(os.path.join(HERE, "owner_meeting_20261116b.pdf"), "LISF on JAX: owner meeting", slides())
