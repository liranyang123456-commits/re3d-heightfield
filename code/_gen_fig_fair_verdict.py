#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fair protocol-matched comparison figure.

Numbers are medians or means recomputed on the same rows of the 17-sequence
table (results/ALL_17_SEQUENCES_BENCHMARK.md) and the headline tables already
in the manuscript. Full-gauge ATE uses the visible chessboard; it is not a
marker-free comparison with VGGT.
"""
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\figures")
PAPER = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\paper")
STAGE = PAPER / "_cmig_clean_stage"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})

# (protocol, ours, vggt_or_best_other_name, other, orb_or_none)
# Panel A: ATE, lower better. Three matched protocols.
protocols = [
    ("13 real\nwith board", 4.49, 11.88, 41.98),
    ("17 seq.\nno board", 29.88, 16.86, 40.29),
    ("4 sim\nmean", 32.71, 28.26, 35.75),
]
# Panel B: non-ATE tasks. value is ours minus best other, signed so
# positive = ours better. We plot the raw pair instead.

fig = plt.figure(figsize=(11.4, 3.35))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.38)

ax = fig.add_subplot(gs[0])
x = np.arange(len(protocols))
w = 0.24
ours = [p[1] for p in protocols]
vggt = [p[2] for p in protocols]
orb = [p[3] for p in protocols]
b1 = ax.bar(x - w, ours, w, color="#2ca02c", label="Ours", zorder=2)
b2 = ax.bar(x, vggt, w, color="#4c78a8", label="VGGT-1B", zorder=2)
b3 = ax.bar(x + w, orb, w, color="#d62728", label="ORB-SfM", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels([p[0] for p in protocols], fontsize=7.2)
ax.set_ylabel("ATE (mm), lower is better")
ax.set_ylim(0, 52)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
ax.set_title("(a) Same-protocol trajectory error", fontsize=8.5, loc="left")
for bars in (b1, b2, b3):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.6, f"{h:.1f}",
                ha="center", va="bottom", fontsize=5.6)

ax2 = fig.add_subplot(gs[1])
# rows bottom to top
tasks = [
    ("EndoNeRF AbsRel\n(stereo, lower)", 0.0285, 0.0321, True),
    ("Chessboard scale\n(closer to 1)", 1.11, 91.0, True),
    ("Shape $\\lambda_3/\\lambda_1$\n(higher)", 0.42, 0.31, True),
    ("Board corner AbsRel\n(lower)", 0.247, 0.009, False),
    ("Board ATE, mm\n(lower)", 3.89, 3.02, False),
]
# Draw as paired dots on a per-row normalized axis is misleading.
# Use a verdict strip: ours value and other value as text, color by win.
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5, len(tasks) - 0.5)
ax2.axis("off")
ax2.set_title("(b) Other tasks, matched metric", fontsize=8.5, loc="left")
for i, (name, a, b, ours_better) in enumerate(tasks):
    y = i
    color = "#2ca02c" if ours_better else "#d62728"
    tag = "Ours better" if ours_better else "Baseline better"
    ax2.add_patch(plt.Rectangle((0.46, y - 0.32), 0.52, 0.64, facecolor=color, alpha=0.15, lw=0))
    ax2.text(0.0, y, name, va="center", ha="left", fontsize=6.6)
    ax2.text(0.48, y, f"Ours {a:g}", va="center", ha="left", fontsize=6.6, color="#1b7a1b")
    ax2.text(0.72, y, f"best other {b:g}", va="center", ha="left", fontsize=6.6)
    ax2.text(0.98, y + 0.18, tag, va="center", ha="right", fontsize=6.0, color=color, fontweight="bold")

fig.savefig(FIG / "fig_fair_verdict.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig_fair_verdict.png", bbox_inches="tight", dpi=140)
plt.close()
for dest in (PAPER, STAGE):
    shutil.copy(FIG / "fig_fair_verdict.pdf", dest / "fig_fair_verdict.pdf")
print("wrote", FIG / "fig_fair_verdict.pdf")
