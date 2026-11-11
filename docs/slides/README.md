# Status briefings (2026-10-24; earlier set of 2026-10-22 kept beside it)

Three concise PDF briefings, one per audience, 16:9, source-cited footers, same layout layer and style as `rocke3d_jax/status_slides` of the minimal worktree. Focus: Noah-MP (4.0.1, 5.0, 3.6), with figures.

| File | Audience | Slides | Content |
|---|---|---|---|
| `management_briefing_20261024.pdf` | management | 5 | speed-up figure and three numbers, the journey, evidence and limits, decisions, proposed gates |
| `science_briefing_20261024.pdf` | scientists | 7 | what is ported, how agreement is measured, results table, 89-day run and one-ulp sensitivity figure, routing and DA, limits, open questions |
| `developer_briefing_20261024.pdf` | the next developer | 9 | orientation, anatomy of a run, running it, environment and runner, lessons, benchmark arithmetic, where the JAX time goes (figure), tests and open work, porting recipe |

Figures (`images/`, made by `make_figures.py`; all numbers typed in from `STATUS_20261022.md`, `Reports/BENCHMARKS.md`, `LOG.md`): `fig_speedup.png` (Fortran vs JAX wall time, 9 Noah-MP cases), `fig_agreement_89d.png` (89-day CONUS difference to Fortran next to the JAX one-ulp response), `fig_jax_time.png` (JAX wall time by stage).

Owner: G. Tamkin. Written by a Claude Code session, 2026-10-24. Status: DRAFT, not reviewed by the owner, a LIS scientist or an external reviewer. Numbers are typed into the builder and figure script, not re-measured for the decks; the 89-day Fortran time is scaled from 59 measured days. The slide date in the footer comes from `FOOTER_DATE` in `deck_layout.py`. Review by: before any speed or fidelity statement is made from these slides.

Rebuild (reportlab, Pillow, matplotlib; no network):
```
cd jax-port/status_slides
python3 make_figures.py images
python3 build_briefing_decks_20261024.py     # three PDFs here, PNG previews in preview_*/
```
The builder raises if a text block does not fit its box. Only WinAnsi characters are used. `build_briefing_decks_20261022.py` and its PDFs are the previous set (no figures, footer date now shows 2026-10-24 if rebuilt).

- `owner_meeting_20261110.pdf` (builder `build_owner_meeting_20261110.py`): revision of the 2026-11-02 owner-meeting slides with the independent review, the snow test, R's case on Discover (60-day CPU and A100 results) and the narrowed MPI finding. The 2026-11-02 deck stays as presented.

- `build_owner_meeting_20261111.py` -> `owner_meeting_20261111.pdf` (2026-11-11): the 20261110 deck with the 5.0 flux comparison and outlier replays, the closed test gap, the example case and uv.lock added; slide 2 unchanged apart from the date. Older decks untouched.
