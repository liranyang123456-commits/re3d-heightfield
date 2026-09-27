#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-labelled six-panel summary. Panels are not interchangeable."""
from __future__ import annotations
import os, json
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"E:\MIS_TMI_Re_3D"
PNP = os.path.join(BASE, "re3d_cmpb_results", "results", "evaluation_vs_pnp.json")
EVAL = os.path.join(BASE, "benchmark_results", "evaluation.json")
NEW = os.path.join(BASE, "re3d_cmpb_results", "results", "new_capture")
OUT = os.path.join(BASE, "re3d_cmpb_results", "figures", "fig_sota_summary.pdf")
OUT_PNG = os.path.join(BASE, "re3d_cmpb_results", "figures", "fig_sota_summary.png")

with open(PNP, encoding="utf-8") as f:
    pnp = json.load(f)
with open(EVAL, encoding="utf-8") as f:
    ev = {r["method"]: r for r in json.load(f)}


def lam(name):
    r = ev.get(name) or ev.get("Traditional" if name == "Stereo SGM" else name)
    return float(r["pca_ratios"][2]) if r and "pca_ratios" in r else 0.0


def load_new(alias, kind):
    return json.load(open(os.path.join(NEW, f"{alias}_eval_{kind}.json"), encoding="utf-8"))


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
ENDO = [
    ("Ours-E2E", 0.0285, True),
    ("FS", 0.0321, False),
    ("SGM", 0.0323, False),
    ("DA-V2-S", 0.2038, False),
    ("Mono", 0.4856, False),
]

# New-capture shape / scale / corner from eval JSON
SEQ = ["V1", "V2", "P1", "V3"]
ours = {a: load_new(a, "ours") for a in SEQ}
orb = {a: load_new(a, "orb_sfm") for a in SEQ}
vggt = {a: load_new(a, "vggt") for a in SEQ}
corner = {a: load_new(a, "corner_depth") for a in SEQ}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.labelsize": 9.5, "axes.titlesize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), dpi=300)
fig.subplots_adjust(hspace=0.46, wspace=0.32, left=0.06, right=0.99,
                    top=0.90, bottom=0.10)

# (a) chessboard shape
ax = axes[0, 0]
names = [x[0] for x in SHAPE]
vals = [x[1] for x in SHAPE]
cols = ["#1b7f4e" if x[2] else "#7a8494" for x in SHAPE]
ax.bar(range(len(names)), vals, color=cols, width=0.72, edgecolor="none")
ax.axhline(0.25, color="#1b7f4e", ls="--", lw=1.0, label=r"Normal 3D $(\geq 0.25)$")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=28, ha="right")
ax.set_ylabel(r"$\lambda_3/\lambda_1$")
ax.set_ylim(0, 0.52)
ax.set_title("(a) Chessboard shape")
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
ax.legend(frameon=False, loc="upper right", fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (b) chessboard scale
ax = axes[0, 1]
names = [x[0] for x in SCALE]
vals = [x[1] for x in SCALE]
cols = ["#1b7f4e" if n == "Ours" else "#7a8494" for n in names]
ax.bar(range(len(names)), vals, color=cols, width=0.72, edgecolor="none")
ax.set_yscale("log")
ax.axhline(1.0, color="#1b7f4e", ls="--", lw=1.0, label=r"ideal $1\times$")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=28, ha="right")
ax.set_ylabel("Sim3 scale vs solvePnP")
ax.set_title("(b) Chessboard metric scale")
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", fontsize=6.8)
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (c) EndoNeRF
ax = axes[0, 2]
names = [x[0] for x in ENDO]
vals = [x[1] for x in ENDO]
cols = ["#1b7f4e" if x[2] else "#7a8494" for x in ENDO]
ax.bar(range(len(names)), vals, color=cols, width=0.62, edgecolor="none")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names)
ax.set_ylabel("AbsRel (median-aligned)")
ax.set_title("(c) EndoNeRF-100 depth")
ax.set_ylim(0, 0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.98, 0.92, "lower is better", transform=ax.transAxes,
        ha="right", va="top", fontsize=7, color="#555")

# (d) new-capture shape grouped
ax = axes[1, 0]
x = np.arange(len(SEQ))
w = 0.26
y_o = [ours[a]["lambda3_lambda1"] for a in SEQ]
y_r = [orb[a]["lambda3_lambda1"] for a in SEQ]
y_v = [vggt[a]["lambda3_lambda1"] for a in SEQ]
ax.bar(x - w, y_o, w, color="#1b7f4e", label="Ours", edgecolor="none")
ax.bar(x, y_r, w, color="#7a8494", label="ORB", edgecolor="none")
ax.bar(x + w, y_v, w, color="#c0392b", label="VGGT", edgecolor="none")
ax.axhline(0.25, color="#1b7f4e", ls="--", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels(SEQ)
ax.set_ylabel(r"$\lambda_3/\lambda_1$")
ax.set_ylim(0, 0.78)
ax.set_title("(d) New-capture shape")
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
for i, v in enumerate(y_o):
    ax.text(x[i] - w, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (e) new-capture scale error
ax = axes[1, 1]
# |log10(scale)| for Ours / ORB / VGGT
pairs = []
for a in SEQ:
    pairs.append((f"{a} O", ours[a]["sim3_scale"], "#1b7f4e"))
    pairs.append((f"{a} R", orb[a]["sim3_scale"], "#7a8494"))
    pairs.append((f"{a} V", vggt[a]["sim3_scale"], "#c0392b"))
dist = [abs(np.log10(max(v, 1e-6))) for _, v, _ in pairs]
ax.bar(range(len(pairs)), dist, color=[c for *_, c in pairs], width=0.72, edgecolor="none")
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels([n for n, *_ in pairs], rotation=40, ha="right", fontsize=7)
ax.set_ylabel(r"$|\log_{10}(\mathrm{scale})|$")
ax.set_title("(e) New-capture scale error")
for i, ((_, v, _), d) in enumerate(zip(pairs, dist)):
    lab = f"{v:.2f}" if v < 10 else f"{v:.0f}"
    ax.text(i, d + 0.04, lab, ha="center", fontsize=5.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (f) 17-sequence benchmark ATE (headline trajectory result)
ax = axes[1, 2]
B17 = json.load(open(os.path.join(BASE, "re3d_cmpb_results", "results", "ALL_17_SEQUENCES_BENCHMARK.json"), encoding="utf-8"))
def _ates(m):
    return [r[m]["ate_mm"] for r in B17.values() if isinstance(r, dict) and m in r and "ate_mm" in r[m]]
grp = [
    ("VGGT-1B", np.median(_ates("VGGT")), "#4e79a7"),
    ("ORB-SfM", np.median(_ates("ORB-SfM")), "#e15759"),
    ("Ours-BA\n(pattern-free)", np.median(_ates("Ours-BA")), "#f28e2b"),
    ("Ours-FullG\n(gauge-ref.)", np.median(_ates("Ours-FullGauge")), "#1b7f4e"),
]
names = [g[0] for g in grp]
vals = [g[1] for g in grp]
cols = [g[2] for g in grp]
ax.bar(range(len(names)), vals, color=cols, width=0.62, edgecolor="none")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel("Median Sim3 ATE (mm)")
ax.set_title("(f) 17-sequence benchmark")
for i, v in enumerate(vals):
    ax.text(i, v + 0.6, f"{v:.2f}", ha="center", fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.98, 0.92, "lower is better", transform=ax.transAxes,
        ha="right", va="top", fontsize=7, color="#555")

fig.suptitle(
    "Protocol-aware results.  Panels are not interchangeable: "
    "(a,d) shape, (b,e) metric scale, (c) aligned depth, (f) trajectory ATE.",
    fontsize=10, y=0.98,
)
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT_PNG, bbox_inches="tight")
import shutil
shutil.copy(OUT, os.path.join(BASE, "re3d_cmpb_results", "paper", "_cmig_clean_stage", "fig_sota_summary.pdf"))
shutil.copy(OUT, os.path.join(BASE, "re3d_cmpb_results", "paper", "fig_sota_summary.pdf"))
print("wrote", OUT)
