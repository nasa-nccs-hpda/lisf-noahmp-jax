"""Three slides for the code-owner follow-up of 2026-11-16 (revision of owner_meeting_20261111, which revised owner_meeting_20261110 and 20261102): what was done, the second-system result on LIS developer R's case, and what the owner's side can take on.
Same layout layer as the briefing decks. Numbers are typed in from BENCHMARKS.md, reports/discover_r_case_*.md, reports/independent_test_20261105.md, reports/flux_compare_mp5_60d_20261110.md, reports/flux_outliers_mp5_60d_20261111.md (incl. first-divergence section).
usage: python3 build_owner_meeting_20261111.py   (reportlab + Pillow, no network; a text block that does not fit raises)
"""
import os
import deck_layout as L
from deck_layout import AMBER, BLUE, GREEN, INK2, MUTED, ORANGE, W, MARGIN, banner, footer
from build_briefing_decks_20261024 import _page, _cards, _save

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-11-16"
L.FOOTER_DATE = DATE


def slides():
    def s1(n):
        p, y = _page("What was started: Noah-MP on a GPU, checked against the real LIS",
                     "A research prototype by one person with an AI assistant over about three weeks. It shows the approach works for named cases; it is not a product.", "OWNER MEETING FOLLOW-UP | %s" % DATE, 22)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 296, "WHAT EXISTS", [
            "Noah-MP 3.6, 4.0.1 and 5.0 (plus Noah 3.x, Mosaic, CLSM) rewritten in JAX, with a small Python driver. Replay, unit, GRIB and 5-day example tests; all 4 deliberate code edits that once passed every test now fail one.",
            "Reference: real LIS 7.8, built by us with GNU 14.2 (serial and 20-core MPI), and R's own Intel build (60 ranks) on Discover. Same inputs, same output files, compared tile by tile.",
            "Noah-MP 5.0, CONUS, 60 days, one V100 (255 s): 172x faster than 1 core (43,900 s; about 156x against a second, faster build), 12.1x faster than 20-core MPI (3,092 s, output bitwise identical to serial). Smaller cases 2.4x to 10x vs 1 core.",
            "Agreement, our CONUS case: 99% of tile-values differ from Fortran by under 0.023 mm SWE, 0.021 K soil temperature, 0.0001 soil moisture (p99): float32 rounding level. R's case, 60 days: weather input bitwise identical, p99 SWE 0.0007 mm, soil temperature 0.004 K. Worst single tiles differ more (snow, freeze-thaw).",
            "Noah-MP 5.0 fluxes, CONUS 60 days: p99 Qle 0.04, Qh 0.49, Qg 0.63 W/m2, inside limits fixed beforehand (5, 10, 5); energy balance matches Fortran to 0.02 W/m2. 650 of 18 million cell-times differ by over 50 W/m2, all on thin near-freezing snow. From the Fortran state they match to 0.002 W/m2; a 3e-5 K input nudge gives the same drift (first seen 1 Feb 18Z): sensitivity, not a physics error.",
            "> A first 5.0 comparison failed (SWE p99 4.3 mm): snow hydrology was in the 4.0.1 form. Fixed; a reviewer should look for such errors.",
        ], GREEN, 9.4, 6)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 296, "WHAT IT IS NOT", [
            "Not the full LIS: no LDT/LVT, only two forcings, no VIC, CLM2, RUC or JULES.",
            "Not yet reviewed by a person who knows LIS or Noah-MP. A simulated independent tester rated it acceptable for the named cases, with conditions. Most checks were run by the assistant that wrote the code.",
            "Seasons: Feb-Jul 2017, one SGP year and Oct-Nov 2008 (R's case). 5.0 fluxes are compared on CONUS Feb-Apr only, not on R's case or in summer. GitHub CI passes.",
            "148 CONUS water-soil tiles (soil type 14) go unstable in warm season in BOTH Fortran and JAX: an input-data issue, masked.",
            "Fair speed claim is 12.1x (vs 20 MPI cores, output verified identical); 172x is vs one core. MPI at 32+ ranks gives wrong output in our GNU build only: R's Intel-built LIS is bitwise identical at 20, 32 and 40 ranks, so the fault is in our build or patches, not LIS. Culprit not yet isolated.",
        ], ORANGE, 9.6, 6)
        banner(p, MARGIN, y + 312, W - 2 * MARGIN, 42, "Message: the hard question (can a JAX Noah-MP reproduce LIS?) now has an answer from two systems and two Fortran builds. The next questions need people who know LIS and Noah-MP.", AMBER, 10.6)
        p.text(MARGIN, 484, W - 2 * MARGIN, "Key: SWE = snow water equivalent (mm). p99 = 99th percentile of the JAX-minus-Fortran difference over all tiles: 99% of values differ by less than this. Tile = one land point of the CONUS grid.", 8.6, MUTED, maxh=24)
        footer(p, n, "Reports/BENCHMARKS.md; reports/discover_r_case_60d_20261110.md; flux_compare_mp5_60d_20261110.md; flux_outliers_mp5_60d_20261111.md; tests/README.md")
        return p

    def s2(n):
        p, y = _page("A second system: LIS developer R's case on Discover",
                     "R supplied a Noah-MP 5.0 NLDAS-2 test case and an Intel-built LIS reference. We ran it unchanged (dates only) on Discover, 1 Oct to 30 Nov 2008.", "OWNER MEETING FOLLOW-UP | %s" % DATE, 22)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 170, "WHAT WE RAN", [
            "R's configuration and parameter file, coldstart, 60 days, 76,180 tiles, history every 3 h (480 files). Reference: R's LIS 7.8 Intel build, 60 ranks.",
            "The weather input is GRIB, as LIS reads it. The JAX driver now reads GRIB natively; no conversion.",
            "Weather input: all five checked fields bitwise identical to R's output in all 480 files.",
        ], GREEN, 10.0, 5)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 170, "SPEED (ONE RUN EACH)", [
            "One A100: 253 s for 60 days: 113 s compile, 104 s reading GRIB, about 8.5 s model compute.",
            "16 CPU cores: 17 min 55 s.",
            "No ratio against R's Fortran: its wall time for these days was not measured. The V100 CONUS run (255 s) is a different case; not a controlled comparison.",
        ], BLUE, 10.0, 5)
        cols = [150, 86, 100, 100, 100, 100]
        rows = [["Variable (all tiles, 480 files)", "Range", "CPU p99", "CPU worst tile", "A100 p99", "A100 worst tile"],
                ["Snow water equivalent (mm)", "591", "0.0007", "7.5", "0.0016", "5.4"],
                ["Soil temperature (K)", "59.4", "0.0037", "9.0", "0.0055", "9.6"],
                ["Soil moisture (m3/m3)", "0.96", "0.0000065", "0.052", "0.000038", "0.032"],
                ["Canopy interception (mm)", "9.8", "0.0000088", "0.44", "0.000013", "0.46"],
                ["Air temperature, humidity, pressure, sw, lw", "", "identical", "identical", "identical", "identical"]]
        scale = (W - 2 * MARGIN) / sum(cols)
        p.table(MARGIN, y + 186, [c * scale for c in cols], rows, size=9.0, maxh=130, bold_cols=(0,))
        banner(p, MARGIN, y + 326, W - 2 * MARGIN, 40, "Message: two compilers, two systems and two GPUs agree with the port at float32 level on typical tiles. Worst single tiles (snow, freeze-thaw) are larger and not yet examined; the GPU differs more from Fortran than the CPU does.", AMBER, 10.0)
        p.text(MARGIN, 484, W - 2 * MARGIN, "Limits: one 60-day period (October-November), only the variables in R's output list (no surface fluxes), R's run reaches only 19 Jan 2010, one run per configuration. p99 = 99% of tile-values differ by less than this.", 8.6, MUTED, maxh=24)
        footer(p, n, "reports/discover_r_case_60d_20261110.md; reports/discover_r_case_1d_cpu_20261110.md; DISCOVER_R_CASE_PLAN.md")
        return p

    def s3(n):
        p, y = _page("How you can take it on: what we ask of your side",
                     "The code and every check are in one tarball with run scripts. We need owners, reviewers and decisions; we are not asking you to adopt it.", "OWNER MEETING FOLLOW-UP | %s" % DATE, 22)
        rows = [
            ("REVIEW", "Read the Noah-MP 5.0 port (noahmp50_jax.py) against your Fortran; name what looks wrong or is missing.", "A Noah-MP scientist, 1-2 days"),
            ("REPRODUCE", "Re-run R's case, the CONUS 60-day comparison or the 5-day example (examples/sgp_5d, about 1 minute on CPU, Fortran output included; environment pinned by uv.lock); confirm or break our numbers.", "An engineer with GPU access, 1 week"),
            ("DECIDE", "Which models, forcings and domains matter first (we have Noah-MP, Noah, Mosaic, CLSM only). Is speed the goal, or differentiable/ML use?", "Owner decision, one meeting"),
            ("PROVIDE", "More reference cases like R's (summer, other regions), the tolerances you accept, and a fix or a mask for the soil type 14 water tiles.", "Owner plus LDT/data contact"),
            ("OWN", "A named maintainer and a home for the code. Delivered as a tarball (the repo stays private). Apache-2.0, plus the UCAR Noah-MP terms and NCAR/LISF credit (file NOTICE).", "Owner decision"),
        ]
        yy = y + 6
        for lab, what, who in rows:
            L.card(p, MARGIN, yy, W - 2 * MARGIN, 50, BLUE)
            p.text(MARGIN + 16, yy + 8, 90, lab, 10.5, BLUE, True, maxh=16)
            p.text(MARGIN + 112, yy + 7, 440, what, 9.6, INK2, maxh=40)
            p.text(MARGIN + 572, yy + 8, W - 2 * MARGIN - 586, who, 9.2, MUTED, True, maxh=36)
            yy += 56
        _cards(p, MARGIN, yy + 4, (W - 2 * MARGIN - 16) / 2, 100, "WHAT WE KEEP DOING", [
            "Maintain the port, fix what review finds, extend only to what you pick.",
            "Keep every number tied to a script and a report.",
        ], GREEN, 9.8, 5)
        _cards(p, MARGIN + (W - 2 * MARGIN) / 2 + 8, yy + 4, (W - 2 * MARGIN - 16) / 2, 100, "STILL UNPORTED (YOUR CALL)", [
            "Irrigation in the runner, runoff option 5, soil option 2+, urban, GECROS.",
            "JULES, VIC, CLM2, RUC, MPI, LDT and LVT beyond Noah-MP.",
        ], ORANGE, 9.8, 5)
        footer(p, n, "STATUS_20261022.md; Reports/BENCHMARKS.md; status_slides/port_status_table_20261026.pdf (full ported / open table)")
        return p

    return [s1, s2, s3]


if __name__ == "__main__":
    _save(os.path.join(HERE, "owner_meeting_20261116.pdf"), "LISF on JAX: owner meeting follow-up", slides())
