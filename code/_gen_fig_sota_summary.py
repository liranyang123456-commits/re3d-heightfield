#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fair, protocol-labelled summary figure for the CMIG paper.

All numbers are read from existing evaluation JSONs. Panels answer
different questions and are NOT comparable across panels.
"""
from __future__ import annotations
import os, json
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = r"E:\MIS_TMI_Re_3D"
PNP = os.path.join(BASE, "re3d_cmpb_results", "results", "evaluation_vs_pnp.json")
EVAL = os.path.join(BASE, "benchmark_results", "evaluation.json")
OUT = os.path.join(BASE, "re3d_cmpb_results", "figures", "fig_sota_summary.pdf")
OUT_PNG = os.path.join(BASE, "re3d_cmpb_results", "figures", "fig_sota_summary.png")

with open(PNP, encoding="utf-8") as f:
    pnp = json.load(f)
with open(EVAL, encoding="utf-8") as f:
    ev = {r["method"]: r for r in json.load(f)}

# Chessboard shape (evaluation.json)
def lam(name, fallback=None):
    r = ev.get(name) or ev.get("Traditional" if name == "Stereo SGM" else name)
    if r and "pca_ratios" in r:
        return float(r["pca_ratios"][2])
    return fallback

SHAPE = [
    ("Ours", lam("Ours"), True),
    ("SGM", lam("Traditional"), True),
    ("MASt3R", lam("MASt3R"), False),
    ("VGGT", lam("VGGT"), False),
    ("COLMAP", lam("COLMAP"), False),
    ("DUSt3R", lam("DUSt3R"), False),
    ("MUSt3R", lam("MUSt3R"), False),
    ("ORB", lam("ORB-SfM"), False),
]

SCALE = [
    ("Ours", pnp["Ours"]["sim3_scale"]),
    ("ORB", pnp["ORB-SfM"]["sim3_scale"]),
    ("COLMAP", pnp["COLMAP"]["sim3_scale"]),
    ("MUSt3R", pnp["MUSt3R"]["sim3_scale"]),
    ("VGGT", pnp["VGGT"]["sim3_scale"]),
    ("Reloc3R", pnp["Reloc3R"]["sim3_scale"]),
    ("MASt3R", pnp["MASt3R"]["sim3_scale"]),
    ("DUSt3R", pnp["DUSt3R"]["sim3_scale"]),
]

# EndoNeRF 100-frame median-aligned AbsRel (paper table; source metrics.json)
ENDO = [
    ("Ours-E2E", 0.0285, True),
    ("FS", 0.0321, False),
    ("SGM", 0.0323, False),
    ("DA-V2-S", 0.2038, False),
    ("Mono", 0.4856, False),
]

# New-capture scale (eval JSON, already verified)
NEW = [
    ("V1 Ours", 0.47),
    ("V1 ORB", 2.06),
    ("V1 VGGT", 193.0),
    ("V2 Ours", 0.63),
    ("V2 ORB", 9.86),
    ("V2 VGGT", 916.0),
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 10.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=300)
fig.subplots_adjust(hspace=0.48, wspace=0.32, left=0.08, right=0.98,
                    top=0.90, bottom=0.10)

# ── (a) shape ──
ax = axes[0, 0]
names = [x[0] for x in SHAPE]
vals = [x[1] for x in SHAPE]
cols = ["#1b7f4e" if x[2] else "#7a8494" for x in SHAPE]
bars = ax.bar(range(len(names)), vals, color=cols, width=0.72, edgecolor="none")
ax.axhline(0.25, color="#1b7f4e", ls="--", lw=1.0, label=r"Normal 3D $(\geq 0.25)$")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=28, ha="right")
ax.set_ylabel(r"$\lambda_3/\lambda_1$")
ax.set_ylim(0, 0.52)
ax.set_title("(a) Chessboard shape  (scale-invariant)")
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, color="#1a1a1a")
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── (b) scale log ──
ax = axes[0, 1]
names = [x[0] for x in SCALE]
vals = [x[1] for x in SCALE]
cols = ["#1b7f4e" if n == "Ours" else "#7a8494" for n in names]
ax.bar(range(len(names)), vals, color=cols, width=0.72, edgecolor="none")
ax.set_yscale("log")
ax.axhline(1.0, color="#1b7f4e", ls="--", lw=1.0, label="ideal $1\\times$")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=28, ha="right")
ax.set_ylabel("Sim3 scale vs solvePnP")
ax.set_title("(b) Chessboard metric scale  (log)")
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center",
            fontsize=7.2)
ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── (c) EndoNeRF depth ──
ax = axes[1, 0]
names = [x[0] for x in ENDO]
vals = [x[1] for x in ENDO]
cols = ["#1b7f4e" if x[2] else "#7a8494" for x in ENDO]
ax.bar(range(len(names)), vals, color=cols, width=0.62, edgecolor="none")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_ylabel("AbsRel  (median-aligned)")
ax.set_title("(c) EndoNeRF depth, 100 frames")
ax.set_ylim(0, 0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.98, 0.92, "lower is better", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5, color="#555")

# ── (d) new-capture scale ──
ax = axes[1, 1]
names = [x[0] for x in NEW]
# plot |log10(s)| so 0.47 and 193 are both visible as distance from 1
dist = [abs(np.log10(v)) for _, v in NEW]
cols = ["#1b7f4e" if "Ours" in n else "#c0392b" if "VGGT" in n else "#7a8494"
        for n in names]
ax.bar(range(len(names)), dist, color=cols, width=0.65, edgecolor="none")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=28, ha="right")
ax.set_ylabel(r"$|\log_{10}(\mathrm{scale})|$  (0 = ideal)")
ax.set_title("(d) New-capture scale error  (lower is better)")
for i, ((_, v), d) in enumerate(zip(NEW, dist)):
    lab = f"{v:.2f}" if v < 10 else f"{v:.0f}"
    ax.text(i, d + 0.04, lab + r"$\times$", ha="center", fontsize=7.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Protocol-aware comparison.  Panels are not interchangeable: "
    "(a) shape, (b) metric scale, (c) aligned depth, (d) cross-hardware scale.",
    fontsize=10, y=0.98,
)

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight")
print("wrote", OUT)
