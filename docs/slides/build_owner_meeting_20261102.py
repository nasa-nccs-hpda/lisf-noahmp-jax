"""Two slides for the code-owner meeting of 2026-11-03: what was done, and what the owner's side can take on.
Same layout layer as the briefing decks. Numbers are typed in from BENCHMARKS.md / LOG.md / STATUS_20261022.md.
usage: python3 build_owner_meeting_20261102.py   (reportlab + Pillow, no network; a text block that does not fit raises)
"""
import os
import deck_layout as L
from deck_layout import AMBER, BLUE, GREEN, INK2, MUTED, ORANGE, W, MARGIN, banner, footer
from build_briefing_decks_20261024 import _page, _cards, _save

HERE = os.path.dirname(os.path.abspath(__file__))
DATE = "2026-11-02"
L.FOOTER_DATE = DATE


def slides():
    def s1(n):
        p, y = _page("What was started: Noah-MP on a GPU, checked against the real LIS",
                     "A research prototype by one person with an AI assistant over about two weeks. It shows the approach works for named cases; it is not a product.", "OWNER MEETING | %s" % DATE, 22)
        cw, gap = (W - 2 * MARGIN - 16) / 2, 16
        _cards(p, MARGIN, y + 6, cw, 290, "WHAT EXISTS", [
            "Noah-MP 3.6, 4.0.1 and 5.0 (plus Noah 3.x, Mosaic, CLSM) rewritten in JAX, with a small Python driver. Not all of LIS. 48 automated tests pass.",
            "Reference: real LIS 7.8 built with GNU 14.2, serial (one core) and MPI (20 cores). Same inputs, same output files, compared tile by tile.",
            "Noah-MP 5.0, CONUS, 60 days, one V100 (255 s) vs Fortran: 172x faster than 1 core (43,900 s), 12.1x faster than 20-core MPI (3,092 s, output bitwise identical to serial). 150 days: about 140x vs 1 core. Smaller cases 2.4x to 10x vs 1 core.",
            "Agreement, 60 days, 76k tiles, same inputs: 99% of tile-values differ from Fortran by under 0.023 mm SWE, 0.021 K soil temperature, 0.0001 soil moisture (p99). Float32 rounding level; worst single tiles differ more (snow and freeze-thaw thresholds).",
            "> A first comparison failed (SWE p99 4.3 mm): the snow-layer hydrology was ported in the 4.0.1 form. Fixed to 5.0; this is the kind of error a reviewer should look for.",
            "> Not just rounding noise? Nudging JAX air temperature by one float32 step gives differences similar to JAX vs Fortran (overlap in worst tiles 46-50 of 200), consistent with chaos, not proof.",
        ], GREEN, 10.6)
        _cards(p, MARGIN + cw + gap, y + 6, cw, 290, "WHAT IT IS NOT", [
            "Not the full LIS: no LDT/LVT, only two forcings, no VIC, CLM2, RUC or JULES.",
            "Not independently reproduced and not reviewed by anyone who knows LIS or Noah-MP. All checks were run by the same assistant that wrote the code.",
            "Winter and spring cases only (CONUS Feb-Jul 2017, one SGP year). Float32 differences on snow and freeze-thaw tiles are expected.",
            "148 CONUS water-soil tiles (soil type 14) go unstable in warm season in BOTH Fortran and JAX: an input-data issue, masked.",
            "Fair speed claim is 12.1x (vs 20 MPI cores, output verified identical); 172x is vs one core. MPI at 32+ ranks gives wrong output in our build.",
        ], ORANGE, 10.6)
        banner(p, MARGIN, y + 306, W - 2 * MARGIN, 46, "Message: the hard question (can a JAX Noah-MP reproduce LIS?) has a promising first answer. The next questions need people who know LIS and Noah-MP.", AMBER, 11)
        p.text(MARGIN, 484, W - 2 * MARGIN, "Key: SWE = snow water equivalent (water in the snowpack, mm). p99 = 99th percentile of the JAX-minus-Fortran difference over all tiles: 99% of values differ by less than this. Tile = one land point of the CONUS grid.", 8.6, MUTED, maxh=24)
        footer(p, n, "Reports/BENCHMARKS.md; STATUS_20261022.md; LOG.md; tests/README.md")
        return p

    def s2(n):
        p, y = _page("How you can take it on: what we ask of your side",
                     "The code and every check are in one repository with run scripts. We need owners, reviewers and decisions; we are not asking you to adopt it.", "OWNER MEETING | %s" % DATE, 22)
        rows = [
            ("REVIEW", "Read the Noah-MP 5.0 port (noahmp50_jax.py) against your Fortran; name what looks wrong or is missing.", "A Noah-MP scientist, 1-2 days"),
            ("REPRODUCE", "Re-run the 60-day CONUS comparison and the test suite on your own system; confirm or break our numbers.", "An engineer with GPU access, 1 week"),
            ("DECIDE", "Which models, forcings and domains matter first (we have Noah-MP, Noah, Mosaic, CLSM only). Is speed the goal, or differentiable/ML use?", "Owner decision, one meeting"),
            ("PROVIDE", "Reference cases you trust, tolerances you accept, and a fix or a mask for the soil type 14 water tiles.", "Owner plus LDT/data contact"),
            ("OWN", "A named maintainer for the repo (now private, on one personal account) and agreement on where it lives.", "Owner decision"),
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

    return [s1, s2]


if __name__ == "__main__":
    _save(os.path.join(HERE, "owner_meeting_20261102.pdf"), "LISF on JAX: owner meeting", slides())
