#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Stereo-Lap figures from the real merged results.

Fixes two problems in the previous versions:
  - fig_stereo_lap_perseq.pdf: win/loss panel counted ties as SGM wins;
    replaced by an aggregate-means panel (ties are reported as ties).
  - fig_stereo_lap_delta.pdf: dashed mean lines were hardcoded to stale
    values (0.974 / 0.880); now computed from the data.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6.5, "figure.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
COLORS = {"Ours": "#1A6B9E", "Stereo SGM": "#C05A1E", "Mono": "#9CA3AF"}

DATA = Path(r"E:\MIS_TMI_Re_3D\benchmark_stereo_lap\stereo_lap_merged.json")
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\figures")

with open(DATA) as f:
    data = json.load(f)
seqs = sorted(data["sequences"].keys(), key=int)
S = data["sequences"]

ours_a = [S[s]["ours"]["AbsRel"] for s in seqs]
sgm_a = [S[s]["sgm"]["AbsRel"] for s in seqs]
mono_a = [S[s]["mono"]["AbsRel"] for s in seqs]
ours_d = [S[s]["ours"]["d1_25"] for s in seqs]
sgm_d = [S[s]["sgm"]["d1_25"] for s in seqs]
mono_d = [S[s]["mono"]["d1_25"] for s in seqs]

# ── Figure 1: per-sequence AbsRel + aggregate means ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                               gridspec_kw={"width_ratios": [3, 1]})
x = np.arange(len(seqs))
w = 0.25
ax1.bar(x, mono_a, w, label="Mono", color=COLORS["Mono"], edgecolor="white", linewidth=0.3)
ax1.bar(x + w, sgm_a, w, label="Stereo SGM", color=COLORS["Stereo SGM"], edgecolor="white", linewidth=0.3)
ax1.bar(x + 2 * w, ours_a, w, label="Ours", color=COLORS["Ours"], edgecolor="white", linewidth=0.3)
ax1.set_xticks(x + w)
ax1.set_xticklabels(seqs, rotation=45, ha="right", fontsize=5.5)
ax1.set_ylabel("AbsRel")
ax1.set_xlabel("Sequence")
ax1.set_title("(a) Per-sequence AbsRel (identical for Ours and SGM)")
ax1.legend(loc="upper left", ncol=3, frameon=True, fontsize=6)
ax1.set_yscale("log")
ax1.set_ylim(0.005, 5)
ax1.grid(axis="y", alpha=0.3, linewidth=0.3)

means = [np.mean(ours_a), np.mean(sgm_a), np.mean(mono_a)]
labels = ["Ours", "Stereo SGM", "Mono"]
bars = ax2.bar(labels, means, color=[COLORS[l] for l in labels],
               edgecolor="white", linewidth=0.3)
for b, v in zip(bars, means):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.3f}",
             ha="center", va="bottom", fontsize=7, fontweight="bold")
ax2.set_yscale("log")
ax2.set_ylim(0.005, 5)
ax2.set_ylabel("Mean AbsRel")
ax2.set_title("(b) Aggregate (22 ties)")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(axis="y", alpha=0.3, linewidth=0.3)
plt.tight_layout()
fig.savefig(OUT / "fig_stereo_lap_perseq.pdf", dpi=300)
fig.savefig(OUT / "fig_stereo_lap_perseq.png", dpi=300)
plt.close()
print("[OK] fig_stereo_lap_perseq")

# ── Figure 2: per-sequence delta1 ──
m_o, m_s, m_m = np.mean(ours_d), np.mean(sgm_d), np.mean(mono_d)
fig, ax = plt.subplots(figsize=(7.0, 1.8))
ax.bar(x, ours_d, w, label="Ours", color=COLORS["Ours"], edgecolor="white", linewidth=0.2)
ax.bar(x + w, sgm_d, w, label="Stereo SGM", color=COLORS["Stereo SGM"], edgecolor="white", linewidth=0.2)
ax.bar(x + 2 * w, mono_d, w, label="Mono", color=COLORS["Mono"], edgecolor="white", linewidth=0.2)
ax.set_xticks(x + w)
ax.set_xticklabels(seqs, rotation=45, ha="right", fontsize=5.5)
ax.set_ylabel(r"$\delta_1$ accuracy")
ax.set_xlabel("Sequence")
ax.set_title(r"Per-sequence $\delta_1$ accuracy on the Stereo Laparoscopic Dataset", fontsize=9)
ax.legend(loc="lower left", ncol=3, fontsize=6.5)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3, linewidth=0.3)
ax.axhline(y=m_o, color=COLORS["Ours"], linestyle="--", linewidth=0.6, alpha=0.6)
ax.axhline(y=m_m, color=COLORS["Mono"], linestyle="--", linewidth=0.6, alpha=0.6)
plt.tight_layout()
fig.savefig(OUT / "fig_stereo_lap_delta.pdf", dpi=300)
fig.savefig(OUT / "fig_stereo_lap_delta.png", dpi=300)
plt.close()
print(f"[OK] fig_stereo_lap_delta (means: Ours {m_o:.3f}, SGM {m_s:.3f}, Mono {m_m:.3f})")
