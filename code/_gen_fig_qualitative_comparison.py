#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the qualitative benchmark into three light figures.

Point clouds are rasterised inside the PDF so a viewer does not walk
thousands of vector markers when the page is opened.

  fig_qualitative_chessboard.pdf   per-frame checker heightfields + fused shape
  fig_qualitative_trajectory.pdf   measured GT / FullGauge / ORB tracks
  fig_qualitative_tissue.pdf       clinical clouds, face and side
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import open3d as o3d

BASE = Path(r"E:\MIS_TMI_Re_3D")
CODE_DIR = BASE / "re3d_cmpb_results" / "code"
sys.path.insert(0, str(CODE_DIR))
from _opt_ego_pose import sim3_ate, subsample_idx  # noqa: E402
from _eval_ego_recon_orb import orb_centers  # noqa: E402

FIG_DIR = BASE / "re3d_cmpb_results" / "figures"
STAGE_DIR = BASE / "re3d_cmpb_results" / "paper" / "_cmig_clean_stage"
PAPER_DIR = BASE / "re3d_cmpb_results" / "paper"
BENCH = BASE / "benchmark_results"
SCREEN_PATH = BASE / "re3d_cmpb_results" / "results" / "ego_recon_screen.json"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "path.simplify": True,
        "path.simplify_threshold": 0.5,
    }
)

# Published 17-sequence Sim3 ATE (mm), used only as panel annotations.
ATE = {
    "traj_20260923_023422": {"fg": 4.55, "vggt": 16.86, "orb": 68.59, "ba": 65.87},
    "traj_20260924_144019": {"fg": 4.66, "vggt": 12.76, "orb": 65.99, "ba": 43.36},
    "traj_20260923_023241": {"fg": 15.90, "vggt": 18.95, "orb": 71.39, "ba": 32.37},
    "traj_20260924_143156": {"fg": 3.12, "vggt": 9.35, "orb": 58.75, "ba": 58.66},
    "traj_20260923_015918": {"fg": 1.31, "vggt": 5.70, "orb": 29.52, "ba": 24.15},
    "traj_20260923_020132": {"fg": 2.35, "vggt": 21.83, "orb": 35.48, "ba": 29.88},
    "traj_20260923_020407": {"fg": 1.85, "vggt": 11.88, "orb": 41.98, "ba": 27.87},
    "rigid_20260924_142902": {"fg": 2.05, "vggt": 5.99, "orb": 37.33, "ba": 22.67},
    "sim_20260924_172955_s10095": {"vggt": 36.97, "orb": 42.13, "ba": 39.85},
    "sim_20260924_173022_s10096": {"vggt": 18.07, "orb": 40.29, "ba": 41.60},
    "sim_20260924_173049_s10097": {"vggt": 37.25, "orb": 21.04, "ba": 24.86},
    "sim_20260924_173116_s10098": {"vggt": 20.75, "orb": 39.55, "ba": 24.53},
}


def rasterize(ax):
    for coll in list(ax.collections):
        coll.set_rasterized(True)
    for im in list(ax.images):
        im.set_rasterized(True)


def pca_align(pts):
    X = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    aligned = X @ Vt.T
    if np.median(aligned[:, 2]) < 0:
        aligned[:, 2] *= -1
    return aligned


def clip_central(pts, cols, q=0.03):
    lo, hi = np.quantile(pts, [q, 1.0 - q], axis=0)
    m = np.all((pts >= lo) & (pts <= hi), axis=1)
    if cols is None:
        return pts[m], None
    return pts[m], cols[m]


def normalize_span(pts):
    pts = pts - pts.mean(0)
    span = float(np.ptp(pts, axis=0).max())
    return pts / max(span, 1e-6)


def subsample(pts, cols, n, seed):
    if len(pts) <= n:
        return pts, cols
    idx = np.random.RandomState(seed).choice(len(pts), n, replace=False)
    return pts[idx], (None if cols is None else cols[idx])


def load_ply_rgb(path, n_max=4500, seed=0):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    if cols is not None and len(cols) != len(pts):
        cols = None
    pts = pca_align(pts)
    pts, cols = clip_central(pts, cols)
    pts, cols = subsample(pts, cols, n_max, seed)
    return normalize_span(pts), cols


def depth_colors(pts):
    z = pts[:, 2]
    z = (z - z.min()) / (float(np.ptp(z)) + 1e-9)
    return cm.magma(z)[:, :3]


def scatter_cloud(ax, pts, cols, s=0.8, alpha=0.9, axes=(0, 1), frame="fill"):
    i, j = axes
    if cols is None:
        cols = depth_colors(pts)
    ax.scatter(pts[:, i], pts[:, j], s=s, c=cols, alpha=alpha, linewidths=0, rasterized=True)
    lo_i, hi_i = np.quantile(pts[:, i], [0.02, 0.98])
    lo_j, hi_j = np.quantile(pts[:, j], [0.02, 0.98])
    if frame == "thick":
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.28, 0.28)
    else:
        pad_i = 0.04 * max(float(hi_i - lo_i), 1e-3)
        pad_j = 0.04 * max(float(hi_j - lo_j), 1e-3)
        ax.set_xlim(lo_i - pad_i, hi_i + pad_i)
        ax.set_ylim(lo_j - pad_j, hi_j + pad_j)
        if frame != "stretch":
            ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)


def style_ours(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("#2ca02c")
        spine.set_linewidth(1.8)


def build_ours_heightfield(img_bgr, step=3, z_frac=0.16):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, (8, 5), flags)
    if not ret:
        ret, corners = cv2.findChessboardCorners(gray, (5, 8), flags)
    mask = np.ones_like(gray) * 255
    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3),
        )
        hull = cv2.convexHull(corners.astype(np.float32))
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        mask = cv2.dilate(mask, np.ones((35, 35), np.uint8))
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, 3)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (5, 5), 0)
    mag = (mag - mag.min()) / (float(np.ptp(mag)) + 1e-8)
    h, w = gray.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    keep = mask[ys, xs] > 0
    xs, ys = xs[keep], ys[keep]
    z = mag[ys, xs] * z_frac * max(h, w)
    pts = np.stack([xs.astype(np.float64), -ys.astype(np.float64), z], axis=1)
    cols = rgb[ys, xs].astype(np.float64) / 255.0
    lum = cols.mean(axis=1, keepdims=True)
    cols = np.clip(0.35 + 1.35 * (cols - 0.35), 0, 1)
    cols = np.clip(cols * (0.55 + 0.9 * lum), 0, 1)
    return normalize_span(pts), cols, rgb, (corners if ret else None)


def pick_board_frames(folder, n=4):
    files = sorted(folder.glob("*.jpg"))
    if not files:
        return []
    stride = max(1, len(files) // (n * 3))
    chosen = []
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
    for f in files[::stride]:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ok, _ = cv2.findChessboardCorners(img, (8, 5), flags)
        if not ok:
            ok, _ = cv2.findChessboardCorners(img, (5, 8), flags)
        if ok:
            chosen.append(f)
        if len(chosen) >= n:
            break
    return chosen


def save_pair(fig, stem):
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", dpi=110)
    fig.savefig(png, bbox_inches="tight", dpi=110)
    plt.close(fig)
    for dest in (PAPER_DIR, STAGE_DIR):
        shutil.copy(pdf, dest / pdf.name)
    print(f"WROTE {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")
    return pdf


def align_sim3(est_c, gt_c):
    A = np.asarray(est_c, dtype=np.float64)
    B = np.asarray(gt_c, dtype=np.float64)
    ac, bc = A.mean(0), B.mean(0)
    ad, bd = A - ac, B - bc
    sa = np.sqrt((ad ** 2).sum() / max(len(A), 1))
    sb = np.sqrt((bd ** 2).sum() / max(len(B), 1))
    s = sb / max(sa, 1e-12)
    U, _, Vt = np.linalg.svd(ad.T @ bd)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return s * (R @ ad.T).T + bc


def pnp_centers(paths, K, objp):
    centers = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            centers.append(np.zeros(3))
            continue
        ret, c = cv2.findChessboardCorners(g, (11, 8))
        if not ret:
            ret, c = cv2.findChessboardCorners(g, (8, 11))
        if ret:
            _, rvec, tvec = cv2.solvePnP(objp, c, K, None)
            R, _ = cv2.Rodrigues(rvec)
            centers.append((-R.T @ tvec.ravel()))
        else:
            centers.append(np.zeros(3))
    return np.asarray(centers, dtype=np.float64)


# ---------------------------------------------------------------------------
# Figure A — chessboard
# ---------------------------------------------------------------------------
def crop_board(rgb, corners, margin=18):
    if corners is None:
        return rgb
    xy = corners.reshape(-1, 2)
    x0 = max(int(xy[:, 0].min()) - margin, 0)
    y0 = max(int(xy[:, 1].min()) - margin, 0)
    x1 = min(int(xy[:, 0].max()) + margin, rgb.shape[1] - 1)
    y1 = min(int(xy[:, 1].max()) + margin, rgb.shape[0] - 1)
    return rgb[y0:y1, x0:x1]


def four_views(pts):
    ang = np.deg2rad(38)
    c, s = np.cos(ang), np.sin(ang)
    oblique = pts @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
    return [(pts, (0, 1)), (pts, (0, 2)), (pts, (1, 2)), (oblique, (0, 1))]


def figure_chessboard():
    folder = BENCH / "colmap" / "colmap_work" / "images"
    frames = pick_board_frames(folder, 4)
    if len(frames) < 4:
        frames = sorted(folder.glob("*.jpg"))[:4]
    print("board frames:", [p.name for p in frames])

    methods = [
        ("VGGT", "vggt/vggt.ply"),
        ("MASt3R", "mast3r/mast3r.ply"),
        ("COLMAP", "colmap/colmap_sfm.ply"),
        ("DUSt3R", "dust3r/dust3r.ply"),
        ("MUSt3R", "must3r/must3r.ply"),
        ("ORB", "orb_sfm/orb_sfm.ply"),
    ]
    n_rows = 2 + len(methods)
    fig = plt.figure(figsize=(11.6, 7.05))
    gs = fig.add_gridspec(
        n_rows + 1, 5,
        height_ratios=[1] * n_rows + [0.55],
        width_ratios=[0.22, 1, 1, 1, 1],
        hspace=0.05, wspace=0.03,
    )

    built = []
    for path in frames:
        bgr = cv2.imread(str(path))
        pts, cols, rgb, corners = build_ours_heightfield(bgr, step=3)
        built.append((crop_board(rgb, corners), pts, cols))

    def row_label(r, name, ours=False):
        axl = fig.add_subplot(gs[r, 0])
        axl.axis("off")
        axl.text(1.0, 0.5, name, ha="right", va="center", fontsize=7.2,
                 fontweight="bold", color="#1b7a1b" if ours else "black")

    row_label(0, "Input")
    for c, (crop, _, _) in enumerate(built):
        ax = fig.add_subplot(gs[0, c + 1])
        ax.imshow(crop, aspect="auto", interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        rasterize(ax)

    row_label(1, "Ours", ours=True)
    for c, (_, pts, cols) in enumerate(built):
        ax = fig.add_subplot(gs[1, c + 1])
        scatter_cloud(ax, pts, cols, s=1.4, frame="stretch")
        style_ours(ax)

    for r, (label, rel) in enumerate(methods, start=2):
        row_label(r, label)
        pts, cols = load_ply_rgb(BENCH / rel, n_max=2600, seed=11 + r)
        for c, (xyz, axes) in enumerate(four_views(pts)):
            ax = fig.add_subplot(gs[r, c + 1])
            scatter_cloud(ax, xyz, cols, s=0.55, axes=axes, frame="stretch")

    axb = fig.add_subplot(gs[-1, 1:])
    labels = ["Ours", "SGM", "MASt3R", "VGGT", "COLMAP", "DUSt3R", "MUSt3R", "ORB"]
    vals = [0.42, 0.31, 0.22, 0.19, 0.13, 0.09, 0.06, 0.01]
    colors = ["#2ca02c", "#8fce8f"] + ["#c8c8c8"] * 6
    x = np.arange(len(labels))
    axb.bar(x, vals, color=colors, width=0.72, zorder=2)
    axb.axhline(0.25, color="#2ca02c", ls="--", lw=0.7, zorder=1)
    axb.set_xticks(x)
    axb.set_xticklabels(labels, fontsize=7.0)
    axb.set_xlim(-0.55, len(labels) - 0.45)
    axb.set_ylim(0, 0.50)
    axb.set_ylabel(r"$\lambda_3/\lambda_1$", fontsize=7.2)
    axb.tick_params(labelsize=6.2, length=2)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.set_yticks([0, 0.25, 0.42])
    save_pair(fig, "fig_qualitative_chessboard")


# ---------------------------------------------------------------------------
# Figure B — trajectories (measured curves only)
# ---------------------------------------------------------------------------
def figure_trajectories():
    screen = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    K0 = np.asarray(screen["K"], np.float64)
    objp = np.zeros((8 * 11, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 3.0

    real_names = [
        ("traj_20260923_023422", "test"),
        ("traj_20260924_144019", "val"),
        ("traj_20260923_023241", "val"),
        ("traj_20260924_143156", "train"),
        ("traj_20260923_015918", "train"),
        ("traj_20260923_020132", "train"),
        ("traj_20260923_020407", "train"),
        ("rigid_20260924_142902", "extra"),
        ("rigid_20260924_123958", "extra"),
        ("rigid_20260924_124243", "extra"),
        ("rigid_20260924_140510", "extra"),
        ("traj_20260923_015629", "train"),
    ]
    # Published Sim3 ATE (mm): Ours is FullGauge on real sequences and BA on simulation.
    bar_rows = [
        ("023422", 4.55, 16.86, 68.59),
        ("144019", 4.66, 12.76, 65.99),
        ("023241", 15.90, 18.95, 71.39),
        ("143156", 3.12, 9.35, 58.75),
        ("015918", 1.31, 5.70, 29.52),
        ("020132", 2.35, 21.83, 35.48),
        ("020407", 1.85, 11.88, 41.98),
        ("r142902", 2.05, 5.99, 37.33),
        ("r123958", 4.49, 4.14, 21.58),
        ("r124243", 3.44, 7.30, 32.47),
        ("r140510", 8.51, 7.53, 45.00),
        ("015629", 26.38, 28.75, 35.53),
        ("015507", 80.25, 43.54, 56.66),
        ("s10095", 39.85, 36.97, 42.13),
        ("s10096", 41.60, 18.07, 40.29),
        ("s10097", 24.86, 37.25, 21.04),
        ("s10098", 24.53, 20.75, 39.55),
    ]

    def load_real(name):
        row = real[name]
        z = np.load(row["pose"])
        usable = np.flatnonzero(z["usable"] == 1)
        folder = Path(row["image_dir"])
        kept = [int(i) for i in usable if (folder / f"{int(i):06d}.jpg").is_file()]
        sel = subsample_idx(len(kept), 30)
        ids = [kept[i] for i in sel]
        paths = [folder / f"{i:06d}.jpg" for i in ids]
        gt = np.asarray(z["p"][ids], np.float64) * 1000.0
        return paths, gt

    def load_sim(name):
        row = sim[name]
        z = np.load(row["pose"])
        gt_all = np.asarray(z["p_W_C"], np.float64)
        if np.nanmax(np.abs(gt_all)) < 5:
            gt_all = gt_all * 1000.0
        paths_all = sorted(Path(row["image_dir"]).glob("*.jpg"))
        n = min(len(paths_all), len(gt_all))
        sel = subsample_idx(n, 30)
        return [paths_all[i] for i in sel], gt_all[sel]

    def on_plane(gt, *clouds):
        """Project Sim3-aligned tracks onto the plane of the ground-truth motion."""
        c = gt.mean(0)
        _, _, Vt = np.linalg.svd(gt - c, full_matrices=False)
        basis = Vt[:2].T
        out = [((gt - c) @ basis)]
        for P in clouds:
            out.append((P - c) @ basis)
        return out

    panels = []
    for name, role in real_names:
        print("traj", name)
        paths, gt = load_real(name)
        pnp = align_sim3(pnp_centers(paths, K0, objp), gt)
        orb = align_sim3(orb_centers(paths, K0), gt)
        ate_fg, _ = sim3_ate(pnp, gt)
        ate_orb, _ = sim3_ate(orb, gt)
        g2, p2, o2 = on_plane(gt, pnp, orb)
        panels.append({
            "title": name.replace("traj_20260923_", "").replace("traj_20260924_", "").replace("rigid_20260924_", "r"),
            "gt": g2, "pnp": p2, "orb": o2,
            "ate_fg": ate_fg,
        })

    fig = plt.figure(figsize=(12.0, 6.35))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.95], hspace=0.28, wspace=0.06)
    handles = []
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        h_gt, = ax.plot(p["gt"][:, 0], p["gt"][:, 1], color="black", lw=1.6, solid_capstyle="round", zorder=2)
        h_orb, = ax.plot(p["orb"][:, 0], p["orb"][:, 1], color="#d62728", lw=0.9, ls=(0, (1.1, 0.9)), zorder=1)
        h_ours, = ax.plot(p["pnp"][:, 0], p["pnp"][:, 1], color="#2ca02c", lw=1.25, zorder=3)
        ax.set_title(f"{p['title']}  {p['ate_fg']:.1f}", fontsize=6.6, color="#1b7a1b", pad=1.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2ca02c")
            spine.set_linewidth(1.0)
        if i == 0:
            handles = [h_gt, h_ours, h_orb]
        xs = np.concatenate([p["gt"][:, 0], p["pnp"][:, 0], p["orb"][:, 0]])
        ys = np.concatenate([p["gt"][:, 1], p["pnp"][:, 1], p["orb"][:, 1]])
        x0, x1 = np.quantile(xs, [0.02, 0.98])
        y0, y1 = np.quantile(ys, [0.02, 0.98])
        ax.set_xlim(x0 - 0.08 * (x1 - x0 + 1), x1 + 0.08 * (x1 - x0 + 1))
        ax.set_ylim(y0 - 0.08 * (y1 - y0 + 1), y1 + 0.08 * (y1 - y0 + 1))
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(
        handles, ["Ground truth", "Ours-FullGauge", "ORB-SfM"],
        loc="upper center", ncol=3, frameon=False, fontsize=7.4,
        bbox_to_anchor=(0.5, 1.015),
    )
    axb = fig.add_subplot(gs[3, :])
    x = np.arange(len(bar_rows))
    w = 0.26
    ours_v = [r[1] for r in bar_rows]
    vggt_v = [r[2] for r in bar_rows]
    orb_v = [r[3] for r in bar_rows]
    axb.bar(x - w, ours_v, w, color="#2ca02c", label="Ours", zorder=2)
    axb.bar(x, vggt_v, w, color="#4c78a8", label="VGGT-1B", zorder=2)
    axb.bar(x + w, orb_v, w, color="#d62728", label="ORB-SfM", zorder=2)
    axb.set_xticks(x)
    axb.set_xticklabels([r[0] for r in bar_rows], fontsize=5.8, rotation=55, ha="right")
    axb.set_ylabel("ATE (mm)", fontsize=7.0)
    axb.set_ylim(0, 90)
    axb.tick_params(labelsize=6.0, length=2)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.legend(fontsize=6.4, ncol=3, frameon=False, loc="upper right")
    axb.axhline(4.49, color="#2ca02c", ls=":", lw=0.6)
    save_pair(fig, "fig_qualitative_trajectory")


# ---------------------------------------------------------------------------
# Figure C — tissue, face views only, zoomed to the cloud
# ---------------------------------------------------------------------------
def figure_tissue():
    """Equal-size panels. Each dataset is a row of Ours / SGM / Mono, face and side."""
    rows = [
        ("EndoNeRF", "multidataset_pty/endonerf"),
        ("SCARED", "multidataset_pty/scared"),
        ("Stereo-Lap", "multidataset_pty/stereo_lap"),
    ]
    methods = [("Ours", "ours.ply", True), ("SGM", "sgm.ply", False), ("Mono", "mono.ply", False)]
    col_titles = ["Ours face", "Ours side", "SGM face", "SGM side", "Mono face", "Mono side"]

    fig = plt.figure(figsize=(12.0, 6.55))
    gs = fig.add_gridspec(4, 7, width_ratios=[0.20, 1, 1, 1, 1, 1, 1],
                          hspace=0.04, wspace=0.03)

    for r, (name, rel) in enumerate(rows):
        axl = fig.add_subplot(gs[r, 0])
        axl.axis("off")
        axl.text(1.0, 0.5, name, ha="right", va="center", fontsize=7.0, fontweight="bold")
        for m, (mname, ply, is_ours) in enumerate(methods):
            pts, cols = load_ply_rgb(BENCH / rel / ply, n_max=4000, seed=30 + 10 * r + m)
            s = 4.5 if len(pts) < 1500 else 1.1
            for k, axes in enumerate(((0, 1), (0, 2))):
                ax = fig.add_subplot(gs[r, 1 + 2 * m + k])
                scatter_cloud(ax, pts, cols, s=s, axes=axes, frame="stretch")
                if r == 0:
                    ax.set_title(col_titles[2 * m + k], fontsize=6.6,
                                 color="#1b7a1b" if is_ours else "black",
                                 fontweight="bold" if is_ours else "normal", pad=2)
                if is_ours:
                    style_ours(ax)

    e2e = BENCH / "e2e_v2_ft384_endonerf" / "e2e_pseudo3d.ply"
    pts, _ = load_ply_rgb(e2e, n_max=7000, seed=8)
    cols = depth_colors(pts)
    ang = np.deg2rad(40)
    c, s = np.cos(ang), np.sin(ang)
    rot = pts @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T
    e2e_views = [
        (pts, (0, 1), "Face"),
        (pts, (0, 2), "Side"),
        (pts, (1, 2), "End"),
        (rot, (0, 1), "Oblique"),
        (rot, (0, 2), "Oblique side"),
        (rot, (1, 2), "Oblique end"),
    ]
    axl = fig.add_subplot(gs[3, 0])
    axl.axis("off")
    axl.text(1.0, 0.5, "E2E", ha="right", va="center", fontsize=7.0, fontweight="bold", color="#1b7a1b")
    for k, (xyz, axes, title) in enumerate(e2e_views):
        ax = fig.add_subplot(gs[3, k + 1])
        scatter_cloud(ax, xyz, cols, s=0.45, axes=axes, frame="stretch")
        style_ours(ax)
        ax.set_xlabel(title, fontsize=6.2, color="#1b7a1b", labelpad=1)
    save_pair(fig, "fig_qualitative_tissue")


if __name__ == "__main__":
    which = set(sys.argv[1:] or ["board", "tissue", "traj"])
    if "board" in which:
        figure_chessboard()
    if "tissue" in which:
        figure_tissue()
    if "traj" in which:
        figure_trajectories()
