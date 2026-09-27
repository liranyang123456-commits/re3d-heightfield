#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate high-impact 17-sequence benchmark & deep training convergence figure.
Saves PDF and PNG to figures/ and paper/_cmig_clean_stage/.
"""
from __future__ import annotations
import os, json, shutil
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"E:\MIS_TMI_Re_3D"
RES = os.path.join(BASE, "re3d_cmpb_results", "results")
BENCH_17 = os.path.join(RES, "ALL_17_SEQUENCES_BENCHMARK.json")
DEEP_SUM = os.path.join(RES, "deep_pose_fusion_summary.json")

FIG_DIR = os.path.join(BASE, "re3d_cmpb_results", "figures")
STAGE_DIR = os.path.join(BASE, "re3d_cmpb_results", "paper", "_cmig_clean_stage")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(STAGE_DIR, exist_ok=True)

OUT_PDF = os.path.join(FIG_DIR, "fig_17seq_benchmark.pdf")
OUT_PNG = os.path.join(FIG_DIR, "fig_17seq_benchmark.png")

with open(BENCH_17, encoding="utf-8") as f:
    bench = json.load(f)
with open(DEEP_SUM, encoding="utf-8") as f:
    deep = json.load(f)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.labelsize": 9.5, "axes.titlesize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), dpi=300)
fig.subplots_adjust(hspace=0.38, wspace=0.26, left=0.07, right=0.97, top=0.92, bottom=0.09)

# -------------------------------------------------------------
# Panel (a): Representative Sequences Sim3 ATE (mm)
# -------------------------------------------------------------
ax = axes[0, 0]
rep_seqs = [
    ("traj_023422\n(Real Test)", "traj_20260923_023422"),
    ("traj_144019\n(Real Val)", "traj_20260924_144019"),
    ("traj_023241\n(Real Val)", "traj_20260923_023241"),
    ("traj_015629\n(Real Train)", "traj_20260923_015629"),
    ("s10097\n(Sim Cavity)", "sim_20260924_173049_s10097"),
    ("s10098\n(Sim Cavity)", "sim_20260924_173116_s10098"),
]
x = np.arange(len(rep_seqs))
width = 0.20

vggt_vals = [bench[s[1]]["VGGT"]["ate_mm"] for s in rep_seqs]
orb_vals = [bench[s[1]]["ORB-SfM"]["ate_mm"] for s in rep_seqs]
ba_vals = [bench[s[1]]["Ours-BA"]["ate_mm"] for s in rep_seqs]
gauge_vals = [bench[s[1]].get("Ours-FullGauge", {}).get("ate_mm", np.nan) for s in rep_seqs]

r1 = ax.bar(x - 1.5 * width, vggt_vals, width, label="VGGT-1B", color="#4e79a7")
r2 = ax.bar(x - 0.5 * width, orb_vals, width, label="ORB-SfM", color="#e15759")
r3 = ax.bar(x + 0.5 * width, ba_vals, width, label="Ours-BA (Ours)", color="#f28e2b")
r4 = ax.bar(x + 1.5 * width, gauge_vals, width, label="Ours-FullGauge", color="#2ca02c")

ax.set_ylabel("Sim3 ATE (mm) ↓")
ax.set_title("(a) Reconstruction Error on Representative Sequences")
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in rep_seqs], fontsize=8)
ax.legend(frameon=True, fontsize=8, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# -------------------------------------------------------------
# Panel (b): Overall All-17 Sequences Benchmark Summary
# -------------------------------------------------------------
ax = axes[0, 1]
methods = ["VGGT-1B", "ORB-SfM", "Ours-BA", "Ours-SparseG", "Ours-FullG"]

all_vggt = [r["VGGT"]["ate_mm"] for r in bench.values() if "VGGT" in r and "ate_mm" in r["VGGT"]]
all_orb = [r["ORB-SfM"]["ate_mm"] for r in bench.values() if "ORB-SfM" in r and "ate_mm" in r["ORB-SfM"]]
all_ba = [r["Ours-BA"]["ate_mm"] for r in bench.values() if "Ours-BA" in r and "ate_mm" in r["Ours-BA"]]
all_sg = [r["Ours-SparseGauge"]["ate_mm"] for r in bench.values() if "Ours-SparseGauge" in r and "ate_mm" in r["Ours-SparseGauge"]]
all_fg = [r["Ours-FullGauge"]["ate_mm"] for r in bench.values() if "Ours-FullGauge" in r and "ate_mm" in r["Ours-FullGauge"]]

means = [np.mean(all_vggt), np.mean(all_orb), np.mean(all_ba), np.mean(all_sg), np.mean(all_fg)]
medians = [np.median(all_vggt), np.median(all_orb), np.median(all_ba), np.median(all_sg), np.median(all_fg)]

x_b = np.arange(len(methods))
w_b = 0.35

ax.bar(x_b - w_b / 2, means, w_b, label="Mean ATE (mm)", color="#72b7b2")
ax.bar(x_b + w_b / 2, medians, w_b, label="Median ATE (mm)", color="#1b7f4e")

ax.set_ylabel("ATE Error (mm) ↓")
ax.set_title("(b) 17-Sequence Global Benchmark (Mean vs Median)")
ax.set_xticks(x_b)
ax.set_xticklabels(methods, rotation=15, ha="right")
ax.legend(frameon=True, fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i in range(len(methods)):
    ax.text(x_b[i] - w_b / 2, means[i] + 0.8, f"{means[i]:.1f}", ha="center", fontsize=7.2)
    ax.text(x_b[i] + w_b / 2, medians[i] + 0.8, f"{medians[i]:.1f}", ha="center", fontsize=7.2, fontweight="bold")

# -------------------------------------------------------------
# Panel (c): Deep Pose Fusion 120-Epoch Training Curve
# -------------------------------------------------------------
ax = axes[1, 0]
hist = deep["history"]
eps = [h["epoch"] for h in hist]
losses = [h["loss"] for h in hist]
val_ates = [h["heldout_val_ate"] for h in hist]
test_ates = [h["heldout_test_ate"] for h in hist]

ax_twin = ax.twinx()
p1 = ax.plot(eps, losses, "k-", lw=1.5, label="Training Loss (L_se3)")
p2 = ax_twin.plot(eps, val_ates, "g--", lw=1.5, marker="o", ms=4, label="Held-out Val ATE (mm)")
p3 = ax_twin.plot(eps, test_ates, "b-.", lw=1.3, marker="s", ms=3.5, label="Held-out Test ATE (mm)")

ax.set_xlabel("Training Epochs (ResNet-50 + AttnPool)")
ax.set_ylabel("Training Loss", color="k")
ax_twin.set_ylabel("Held-out ATE Error (mm) ↓", color="g")
ax.set_title("(c) Deep Pose Fusion Convergence & Generalization")

lines = p1 + p2 + p3
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc="upper right", frameon=True, fontsize=7.8)
ax.grid(True, linestyle="--", alpha=0.4)

best_ep = [h["epoch"] for h in hist if h.get("is_best")][-1]
best_v = [h["heldout_val_ate"] for h in hist if h.get("is_best")][-1]
ax_twin.annotate(f"Best Val: {best_v:.2f}mm", xy=(best_ep, best_v), xytext=(best_ep + 8, best_v + 6),
                 arrowprops=dict(facecolor="green", shrink=0.08, width=1, headwidth=4),
                 fontsize=8, color="green", fontweight="bold")

# -------------------------------------------------------------
# Panel (d): Scale Gauge vs Free Unconstrained Scales
# -------------------------------------------------------------
ax = axes[1, 1]
# Real sequences comparison of scale factors:
# VGGT vs Ours-BA vs Ours-FullGauge
real_names = [k for k, v in bench.items() if v["kind"] == "real"]
vggt_scales = [bench[k]["VGGT"]["scale"] for k in real_names]
ba_scales = [bench[k]["Ours-BA"]["scale"] for k in real_names]
fg_scales = [bench[k]["Ours-FullGauge"]["scale"] for k in real_names]

box_data = [fg_scales, ba_scales, vggt_scales]
bp = ax.boxplot(box_data, patch_artist=True, labels=["Ours-FullG\n(Metric Gauge)", "Ours-BA\n(Free Scale)", "VGGT-1B\n(Free Scale)"])
colors = ["#2ca02c", "#f28e2b", "#4e79a7"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_yscale("log")
ax.axhline(1.0, color="green", ls="--", lw=1.2, label="Ideal Metric Scale (1.0x)")
ax.set_ylabel("Reconstructed Sim3 Scale Factor (Log Scale)")
ax.set_title("(d) Metric Scale Fidelity across 13 Real Sequences")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(frameon=True, fontsize=8, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.savefig(OUT_PDF)
plt.savefig(OUT_PNG)
plt.close()

# Copy to stage dir
shutil.copy(OUT_PDF, os.path.join(STAGE_DIR, "fig_17seq_benchmark.pdf"))
print("Generated and saved:", OUT_PDF)
print("Generated and saved:", OUT_PNG)
