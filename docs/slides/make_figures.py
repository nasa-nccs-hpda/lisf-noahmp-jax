"""Figures for the 2026-10-24 LISF-to-JAX briefing decks. Every number is typed in from STATUS_20261022.md section 2, Reports/BENCHMARKS.md and LOG.md
(entries of 2026-10-24); nothing is read from a data file.
usage: python3 make_figures.py images"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "images"
os.makedirs(OUT, exist_ok=True)
NAVY, CARD, GREEN, ORANGE, AMBER, BLUE = "#0F172A", "#1E293B", "#4ADE80", "#F97316", "#FBBF24", "#38BDF8"
INK, MUTED, RULE = "#F8FAFC", "#94A3B8", "#334155"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": RULE, "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD})


def clean(ax, axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=axis, color=RULE, lw=0.6)
    ax.set_axisbelow(True)


# 1. wall time, Fortran (one core) against JAX (V100 unless noted), Noah-MP cases, log scale
cases = [("4.0.1 CONUS 89 d", 57400, 345, False), ("4.0.1 CONUS 30 d", 20398, 247, False), ("4.0.1 CONUS 7 d", 4386, 123, False),
         ("5.0 SGP 365 d", 2945, 277, False), ("4.0.1 SGP 365 d", 2983, 286.5, False), ("5.0 CONUS 150 d", 110700, 790, False), ("5.0 CONUS 60 d", 43944, 255.4, False), ("5.0 CONUS 1 d", 663, 95.2, False),
         ("3.6 CONUS 2 d", 1263, 46.3, False), ("4.0.1 global 5 d (CPU)", 1643, 308, True), ("5.0 global 1 d (CPU)", 446, 183, True)]
fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=200)
for i, (name, f, j, cpu) in enumerate(cases[::-1]):
    ax.barh(i + 0.19, f, height=0.34, color=ORANGE)
    ax.barh(i - 0.19, j, height=0.34, color=BLUE if cpu else GREEN)
    ax.text(f * 1.12, i + 0.19, "%s s" % format(int(round(f)), ","), va="center", fontsize=7, color=INK)
    ax.text(j * 1.12, i - 0.19, "%s s   %.1fx" % (format(j, ",.0f") if j >= 100 else "%.1f" % j, f / j), va="center", fontsize=7, color=INK)
ax.set_xscale("log"); ax.set_xlim(30, 600000)
ax.set_yticks(range(len(cases))); ax.set_yticklabels([c[0] for c in cases[::-1]], fontsize=8)
ax.set_xlabel("wall time, seconds (log scale); 89 d and 150 d Fortran extrapolated from measured days", fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="Fortran LIS, 1 core"), Patch(color=GREEN, label="JAX, V100"), Patch(color=BLUE, label="JAX, CPU only")],
          frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
clean(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_speedup.png")); plt.close(fig)

# 2. agreement on the 89-day CONUS run: Fortran vs JAX next to JAX vs JAX with air temperature moved by one float32 ulp; relative to the variable range
names = ["Qle", "Qh", "Qg", "AvgSurfT", "SoilMoist", "SoilTemp"]
rng = np.array([571, 868, 548, 73.9, 1.01, 63.1])
fj_p99 = np.array([3.80e-2, 9.16e-2, 1.27e-1, 4.79e-3, 1.31e-4, 9.43e-3]); ulp_p99 = np.array([5.67e-2, 1.63e-1, 2.23e-1, 8.76e-3, 1.31e-4, 1.56e-2])
fj_max = np.array([83.5, 168, 228, 9.36, 3.45e-2, 15.4]); ulp_max = np.array([143, 172, 209, 14.4, 2.95e-2, 17.6])
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=200, sharey=True)
for ax, (a, b, ttl) in zip(axs, ((fj_p99, ulp_p99, "p99 of |difference| / range"), (fj_max, ulp_max, "maximum of |difference| / range"))):
    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.19, a / rng, height=0.34, color=ORANGE, label="Fortran vs JAX")
    ax.barh(y - 0.19, b / rng, height=0.34, color=GREEN, label="JAX vs JAX + 1 ulp in Tair")
    ax.set_xscale("log"); ax.set_xlim(1e-6, 1)
    ax.set_title(ttl, fontsize=8.5, color=INK); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    clean(ax)
axs[0].legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_agreement_89d.png")); plt.close(fig)

# 3. where the JAX wall time goes (seconds): setup, forcing reads (host), device incl. compile, output
runs = [("4.0.1 SGP 365 d", 2.5, 185, 90, 9, 286.5), ("5.0 SGP 365 d", 1.7, 166.0, 101.4, 8.2, 277.3), ("4.0.1 CONUS 89 d", 7.0, 206.6, 111.4, 6.3, 331.3)]
fig, ax = plt.subplots(figsize=(7.6, 2.9), dpi=200)
cols = [MUTED, BLUE, GREEN, AMBER]; labs = ["setup", "forcing reads (host)", "device, incl. compile", "output"]
for i, r in enumerate(runs[::-1]):
    left = 0
    for v, c in zip(r[1:5], cols):
        ax.barh(i, v, left=left, color=c, height=0.55)
        if v > 25:
            ax.text(left + v / 2, i, "%.0f" % v, ha="center", va="center", fontsize=7.5, color=NAVY)
        left += v
    ax.text(left + 6, i, "%.0f s" % r[5], va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(runs))); ax.set_yticklabels([r[0] for r in runs[::-1]], fontsize=8.5)
ax.set_xlim(0, 380); ax.set_xlabel("seconds of JAX wall time (V100, parallel forcing reader, background writer)", fontsize=8)
ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labs)], frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, labelcolor=INK)
clean(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_jax_time.png")); plt.close(fig)
print("wrote 3 figures to", OUT)
