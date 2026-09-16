#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate fig_quantitative_comparison with the INDEPENDENT solvePnP reference.

Pose metrics (ATE, Sim3 scale, RPE translation) come from
results/evaluation_vs_pnp.json (all methods aligned to solvePnP GT).
PCA shape ratios come from benchmark_results/evaluation.json (unchanged;
the ratio is scale-invariant and reference-free).

Panels:
  (a) PCA shape ratio lambda3/lambda1
  (b) ATE vs solvePnP (mm)            -- Ours included
  (c) Sim3 scale factor (log)         -- Ours included (1.11x)
  (d) RPE translation per pair (mm)   -- rotation dropped: ill-conditioned
      on this translation-dominant sequence
"""
from __future__ import annotations
import os, json
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE_DIR = r"E:\MIS_TMI_Re_3D"
EVAL_JSON = os.path.join(BASE_DIR, "benchmark_results", "evaluation.json")
PNP_JSON = os.path.join(BASE_DIR, "re3d_cmpb_results", "results", "evaluation_vs_pnp.json")
OUT_PDF = os.path.join(BASE_DIR, "re3d_cmpb_results", "figures", "fig_quantitative_comparison.pdf")
OUT_PNG = os.path.join(BASE_DIR, "re3d_cmpb_results", "figures", "fig_quantitative_comparison.png")

with open(EVAL_JSON) as f:
    results = json.load(f)
with open(PNP_JSON) as f:
    pnp = json.load(f)
res = {r["method"]: r for r in results}

ORDER = ["ORB-SfM", "COLMAP", "VGGT", "DUSt3R", "MASt3R", "MUSt3R", "Reloc3R", "Ours"]
COLORS = {
    "ORB-SfM": "#e74c3c", "COLMAP": "#3498db", "VGGT": "#9b59b6",
    "DUSt3R": "#1abc9c", "MASt3R": "#f39c12", "MUSt3R": "#16a085",
    "Reloc3R": "#e67e22", "Ours": "#27ae60",
}

def pca_ratio(name):
    r = res[name]
    if "pca_ratios" in r and isinstance(r["pca_ratios"], list):
        return float(r["pca_ratios"][2])
    if r.get("pca_str", "N/A") != "N/A":
        return float(r["pca_str"].split(":")[2])
    return 0.0

names_all = ORDER
colors_all = [COLORS[n] for n in names_all]
pca_all = [pca_ratio(n) for n in names_all]
ate_all = [pnp[n]["ate"] for n in names_all]
scale_all = [pnp[n]["sim3_scale"] for n in names_all]
rpet_all = [pnp[n]["rpe_trans_mean"] for n in names_all]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
fig.subplots_adjust(hspace=0.42, wspace=0.32, left=0.08, right=0.96,
                    top=0.92, bottom=0.10)


def add_value_label(ax, x, y, val, fmt='%.2f', fontsize=7.5, color='#2c3e50',
                    va='bottom', offset=0.02):
    ax.text(x, y + offset, fmt % val, ha='center', va=va,
            fontsize=fontsize, color=color, fontweight='bold')


def style_ax(ax):
    ax.grid(axis='y', alpha=0.3, linestyle='-', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def set_xticks(ax, names):
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=30, ha='right')


# ── (a) PCA shape ratio ──
ax = axes[0, 0]
bars = ax.bar(np.arange(len(names_all)), pca_all, color=colors_all,
              edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
ax.axhline(y=0.25, color='#27ae60', ls='--', lw=1.5, alpha=0.8, zorder=2)
ax.axhline(y=0.15, color='#e67e22', ls='--', lw=1.2, alpha=0.6, zorder=2)
ax.text(len(names_all) - 0.5, 0.26, 'Normal 3D', fontsize=7.5, color='#27ae60',
        ha='right', fontweight='bold')
ax.text(len(names_all) - 0.5, 0.155, 'Flat', fontsize=7.5, color='#e67e22',
        ha='right', fontweight='bold')
for i, v in enumerate(pca_all):
    if v > 0:
        add_value_label(ax, i, v, v, fmt='%.2f')
bars[-1].set_edgecolor('#145a32')
bars[-1].set_linewidth(2.5)
set_xticks(ax, names_all)
ax.set_ylabel('$\\lambda_3 / \\lambda_1$', fontsize=12)
ax.set_title('(a) PCA Shape Ratio', fontsize=12, fontweight='bold', color='#2c3e50')
ax.set_ylim(0, 0.55)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
style_ax(ax)

# ── (b) ATE vs solvePnP ──
ax = axes[0, 1]
bars = ax.bar(np.arange(len(names_all)), ate_all, color=colors_all,
              edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
best_idx = int(np.argmin(ate_all))
bars[best_idx].set_edgecolor('#2c3e50')
bars[best_idx].set_linewidth(2.0)
for i, v in enumerate(ate_all):
    add_value_label(ax, i, v, v, fmt='%.1f')
set_xticks(ax, names_all)
ax.set_ylabel('ATE (mm)', fontsize=12)
ax.set_title('(b) Absolute Trajectory Error vs solvePnP', fontsize=12,
             fontweight='bold', color='#2c3e50')
ax.set_ylim(0, max(ate_all) * 1.25)
style_ax(ax)

# ── (c) Sim3 scale (log) ──
ax = axes[1, 0]
bars = ax.bar(np.arange(len(names_all)), scale_all, color=colors_all,
              edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
best_idx = int(np.argmin([abs(s - 1) for s in scale_all]))
bars[best_idx].set_edgecolor('#145a32')
bars[best_idx].set_linewidth(2.5)
for i, v in enumerate(scale_all):
    add_value_label(ax, i, v, v, fmt='%.2f' if v < 10 else '%.0f')
ax.set_yscale('log')
set_xticks(ax, names_all)
ax.set_ylabel('Scale Factor ($\\times$)', fontsize=12)
ax.set_title('(c) Sim3 Scale Factor vs solvePnP', fontsize=12, fontweight='bold',
             color='#2c3e50')
ax.axhline(y=1.0, color='#27ae60', ls='--', lw=1.5, alpha=0.8, zorder=2)
ax.text(len(names_all) - 0.5, 1.15, 'Ideal = 1$\\times$', fontsize=7.5,
        color='#27ae60', ha='right', fontweight='bold')
style_ax(ax)

# ── (d) RPE translation ──
ax = axes[1, 1]
bars = ax.bar(np.arange(len(names_all)), rpet_all, color=colors_all,
              edgecolor='white', linewidth=0.8, width=0.7, zorder=3)
for i, v in enumerate(rpet_all):
    add_value_label(ax, i, v, v, fmt='%.2f')
set_xticks(ax, names_all)
ax.set_ylabel('RPE translation (mm)', fontsize=12)
ax.set_title('(d) Relative Translation Error per Frame Pair', fontsize=12,
             fontweight='bold', color='#2c3e50')
ax.set_ylim(0, max(rpet_all) * 1.25)
style_ax(ax)

fig.savefig(OUT_PDF, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PDF}")
print(f"Saved: {OUT_PNG}")
plt.close()
