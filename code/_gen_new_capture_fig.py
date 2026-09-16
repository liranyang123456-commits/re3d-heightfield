#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-hardware comparison figure on the 4 newly captured sequences.
Panels: (a) PCA shape ratio, (b) Sim3 scale (log), (c) ATE.
Methods: Ours, ORB-SfM, VGGT (COLMAP added if its eval exists).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.labelsize": 9, "axes.titlesize": 10,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                     "legend.fontsize": 7.5, "figure.dpi": 300})

NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
OUT = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\figures\fig_new_capture.pdf"
OUTP = OUT.replace(".pdf", ".png")

SESS = ["pnp_seq_20260911_011115", "vio_seq_20260911_010902",
        "vio_seq_20260911_011013", "vio_seq_20260912_104835"]
SESS_SHORT = ["P1", "V1", "V2", "V3"]
METHODS = [("ours", "Ours", "#27ae60"),
           ("orb_sfm", "ORB-SfM", "#e74c3c"),
           ("vggt", "VGGT", "#9b59b6"),
           ("colmap", "COLMAP", "#3498db")]


def load(sess, method):
    f = os.path.join(NEW_ROOT, sess, f"eval_{method}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))


data = {m: {"shape": [], "scale": [], "ate": []} for _, m, _ in METHODS}
for si, s in enumerate(SESS):
    for key, mname, _ in METHODS:
        j = load(s, key)
        data[mname]["shape"].append(j.get("lambda3_lambda1", np.nan) if j else np.nan)
        data[mname]["scale"].append(j.get("sim3_scale", np.nan) if j else np.nan)
        data[mname]["ate"].append(j.get("ate", np.nan) if j else np.nan)

present = [(k, n, c) for k, n, c in METHODS if not all(np.isnan(data[n]["shape"]))]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
x = np.arange(len(SESS))
nw = len(present)
width = 0.8 / nw

for mi, (key, mname, color) in enumerate(present):
    off = (mi - (nw - 1) / 2) * width
    # (a) shape
    axes[0].bar(x + off, data[mname]["shape"], width, label=mname,
                color=color, edgecolor="white", linewidth=0.4)
    # (b) scale
    axes[1].bar(x + off, data[mname]["scale"], width, label=mname,
                color=color, edgecolor="white", linewidth=0.4)
    # (c) ATE
    axes[2].bar(x + off, data[mname]["ate"], width, label=mname,
                color=color, edgecolor="white", linewidth=0.4)

axes[0].axhline(0.25, color="green", ls="--", lw=1.2)
axes[0].text(len(SESS) - 0.5, 0.27, "Normal 3D", color="green", fontsize=7,
             ha="right", fontweight="bold")
axes[0].axhline(0.15, color="orange", ls="--", lw=1.0)
axes[0].text(len(SESS) - 0.5, 0.16, "Flat", color="orange", fontsize=7, ha="right")
axes[0].set_ylabel(r"$\lambda_3/\lambda_1$")
axes[0].set_title("(a) 3D shape ratio")
axes[0].set_ylim(0, 0.75)

axes[1].set_yscale("log")
axes[1].axhline(1.0, color="green", ls="--", lw=1.2)
axes[1].text(len(SESS) - 0.5, 1.15, "Ideal = 1x", color="green", fontsize=7,
             ha="right", fontweight="bold")
axes[1].set_ylabel("Sim3 scale (x)")
axes[1].set_title("(b) Metric scale fidelity (log)")

axes[2].set_ylabel("ATE (mm)")
axes[2].set_title("(c) Absolute trajectory error")

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(SESS_SHORT)
    ax.grid(axis="y", alpha=0.3, linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].legend(loc="upper left", frameon=True, ncol=1)
axes[2].set_xlabel("Sequence (P1/V1/V2/V3 = newly captured endoscope)")

fig.suptitle("Cross-hardware evaluation on newly captured endoscopic sequences",
             fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUTP, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT)
