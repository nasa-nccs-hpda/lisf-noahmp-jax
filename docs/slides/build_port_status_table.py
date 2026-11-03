"""One-page-set PDF table of what is ported, validated and still open for the LISF-to-JAX work (2026-10-26).
usage: python3 build_port_status_table.py   (reportlab; no network). Every entry is typed in from STATUS_20261022.md and Reports/BENCHMARKS.md."""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "port_status_table_20261026.pdf")
ROWS = [
 ("Land models", "Noah 3.9, 3.6, 3.3, 3.2 and 2.7.1", "Ported, validated", "Compared with the real Fortran LIS. CONUS 2 days: 3.3 is 30x, 3.2 is 32x, 2.7.1 is 28x. Noah 3.6, SGP full year: 8.75x"),
 ("", "Noah-MP 3.6", "Ported, validated", "CONUS 2 days: 27x. Runoff options 2 and 4, dynamic vegetation 1 and 3 and several other options pass"),
 ("", "Noah-MP 4.0.1 (plus land ice)", "Ported, validated", "CONUS 89 days: about 166x (Fortran extrapolated from 59 days). SGP full year: 10x"),
 ("", "Noah-MP 5.0 (plus land ice)", "Ported, validated", "CONUS 60 days: 172x. CONUS 150 days: about 140x (Fortran extrapolated from 121 days). SGP full year: 10.6x"),
 ("", "CLSM F2.5", "Ported, validated", "13.6x on CONUS 1 day. Limit: a summer cold start is not reproduced"),
 ("", "Mosaic", "Ported, validated", "21.6x on CONUS 2 days"),
 ("Noah-MP 5.0 options", "Surface runoff 6, 7, 8 (VIC, Xinanjiang, dynamic VIC)", "Ported, validated", "Checked on a rainy SGP day, replay and 3-day closed loop"),
 ("", "Dynamic vegetation 1-9", "Ported, validated", "7-9 (input LAI) are in the runner; 8 checked across a month boundary (20 days)"),
 ("", "Tile drainage 1 and 2", "Ported, validated", "Synthetic drainage map"),
 ("", "Irrigation: triggers 1, 2, 3; four methods", "Physics validated", "Checked against a patched build only (stock LIS never irrigates); the runner keeps irrigation off"),
 ("", "Dynamic crop model, types 1-5", "Ported, validated", "Corn: 10-day closed loop. Types 2-5: replay only. Synthetic planting and harvest maps"),
 ("", "About 25 other options (stomatal resistance, snow, soil temperature, albedo, radiation, ...)", "Ported, validated", "At float32 noise level"),
 ("Noah-MP 3.6 and 4.0.1 options", "Runoff 2-4, freezing, infiltration, stomatal resistance 2, soil temperature, surface layer, dynamic vegetation 1 and 3", "Ported, validated", "3.6: runoff 2 and 4 only"),
 ("LIS framework", "NLDAS-2 and MERRA-2 weather inputs", "Ported, validated", "Global and CONUS"),
 ("", "Restarts (Noah-MP, round trip with the Fortran)", "Ported, validated", ""),
 ("", "Ensembles with forcing perturbation", "Ported, validated", "12 members on CONUS: 19.5x"),
 ("", "Soil-moisture ensemble Kalman filter", "Ported", "Identical-twin test; no Fortran reference for Noah-MP 5.0"),
 ("", "HyMAP2 river routing", "Ported, validated", "Runs on the Noah-MP runoff"),
 ("", "LDT parameter preparation", "Partly ported", "Noah-MP 4.0.1 configuration only; bitwise identical to the real LDT except slope and aspect"),
 ("", "History output files", "Ported", "Background writer; same variable names as the Fortran"),
 ("Gaps: Noah-MP 5.0", "Runoff option 5 (Miguez-Macho/Fan groundwater)", "Not ported", "Needs groundwater inputs from LDT and extra state"),
 ("", "Soil option 2 and above", "Not ported", ""),
 ("", "Crop domains set by LDT's CROPCLASS; GECROS crop model; urban", "Not ported", ""),
 ("", "Irrigation in the JAX runner", "Not ported", ""),
 ("Gaps: Noah-MP 3.6 and 4.0.1", "Dynamic vegetation 2 and 5-9; 3.6 surface-layer options 3 and 4", "Not ported", "3.6 runoff option 3 is also unvalidated (the Fortran aborts)"),
 ("Gaps: other land models", "JULES", "Not ported", "Needs an external library"),
 ("", "VIC", "Not ported", "About 40,000 lines of C"),
 ("", "CLM2 and RUC 3.7", "Not ported", ""),
 ("Gaps: LIS framework", "MPI-parallel LIS", "Not ported", "All speedups are against one serial Fortran core"),
 ("", "LDT beyond Noah-MP 4.0.1; LVT", "Not ported", ""),
 ("", "Weather inputs other than NLDAS-2 and MERRA-2", "Not ported", "For example ERA5"),
 ("", "Other assimilation methods; CLSM and Mosaic restarts", "Not ported", ""),
 ("Validation", "Independent reproduction", "Not done", "Only the authors have reproduced the results"),
 ("", "Review by a LIS scientist", "Not done", "Tolerances were chosen by the authors"),
 ("", "Noah-MP 5.0 over a full year on CONUS; other seasons and regions", "Not done", "Longest CONUS run so far is 150 days (Feb-Jul 2017)"),
 ("Data issue", "148 land cells with a 'water' soil type", "Open", "Unstable in both the Fortran and JAX models; an input-data problem"),
]
COL = {"Ported, validated": "#DCFCE7", "Ported": "#DCFCE7", "Physics validated": "#FEF9C3", "Partly ported": "#FEF9C3",
       "Not ported": "#FEE2E2", "Not done": "#FEE2E2", "Open": "#FEE2E2"}


def main():
    cell = ParagraphStyle("c", fontName="Helvetica", fontSize=8, leading=9.6)
    bold = ParagraphStyle("b", parent=cell, fontName="Helvetica-Bold")
    head = ParagraphStyle("h", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=3)
    sub = ParagraphStyle("s", fontName="Helvetica", fontSize=8.6, leading=11, textColor=colors.HexColor("#334155"), spaceAfter=7)
    doc = SimpleDocTemplate(OUT, pagesize=landscape(letter), leftMargin=0.5 * inch, rightMargin=0.5 * inch, topMargin=0.5 * inch, bottomMargin=0.55 * inch,
                            title="LISF to JAX: port status", author="G. Tamkin (written with Claude Code)")
    data = [[Paragraph(h, head) for h in ("Area", "Item", "Status", "Note")]]
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for i, (a, it, st, nt) in enumerate(ROWS, 1):
        data.append([Paragraph(a, bold), Paragraph(it, cell), Paragraph(st, bold), Paragraph(nt, cell)])
        style.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(COL[st])))
        if a:
            style.append(("LINEABOVE", (0, i), (-1, i), 1.0, colors.HexColor("#64748B")))
    t = Table(data, colWidths=[1.35 * inch, 3.15 * inch, 1.2 * inch, 4.3 * inch], repeatRows=1)
    t.setStyle(TableStyle(style))
    note = ParagraphStyle("n", parent=cell, fontSize=7.6, leading=9.4, textColor=colors.HexColor("#475569"))
    story = [Paragraph("LISF to JAX: what is ported, validated and still open", title),
             Paragraph("LIS (the Land Information System framework) calls the land models; this work rewrites the land models, and the parts of LIS that drive them, in JAX and checks them against the real Fortran LIS (GNU 14, serial). It is a port of the Noah-MP-centred part of LIS, not of all of LIS.", sub),
             t, Spacer(1, 6),
             Paragraph("Speedups are one Fortran core against one V100 GPU, from single runs; the 89-day and 150-day Fortran times are extrapolated from measured days. 'Validated' means agreement with the real LIS output at float32 level for most variables (single-tile differences at snow and freeze/thaw thresholds remain); tolerances were chosen by the authors and not reviewed by a LIS scientist. "
                       "Status as of 2026-10-26. Draft, for discussion.", note)]
    doc.build(story, onFirstPage=lambda c, d: (c.setFont("Helvetica", 7.5), c.setFillColor(colors.HexColor("#64748B")), c.drawRightString(10.5 * inch, 0.32 * inch, "LISF to JAX port status, 2026-10-26")),
              onLaterPages=lambda c, d: (c.setFont("Helvetica", 7.5), c.setFillColor(colors.HexColor("#64748B")), c.drawRightString(10.5 * inch, 0.32 * inch, "LISF to JAX port status, 2026-10-26   page %d" % d.page)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
