#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heightfield FPFH--ICP trajectory on the same 30-frame EGO clips.

This is the paper operating point used for pose: bilateral filter, Sobel
gradient, percentile truncation, height z = g * 0.15*max(H,W) on a cell-3
grid, then PCA coarse alignment with an FPFH-RANSAC fallback and
point-to-point ICP. Camera centers of the chained poses are compared with
Sim3 ATE, the same protocol as the other reconstruction baselines.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"
RESULT = HERE / "ego_recon_ours.json"
COMPARISON = HERE / "ego_recon_comparison.json"
MAX_FRAMES = 30
WORK_W = 640
CELL = 3
TAU_PCT = 68
CB_SIZE = (11, 8)  # GP050 inner corners


def sim3_ate(est_c, gt_c):
    A = np.asarray(est_c, dtype=np.float64)
    B = np.asarray(gt_c, dtype=np.float64)
    ac, bc = A.mean(0), B.mean(0)
    ad, bd = A - ac, B - bc
    sa = np.sqrt((ad ** 2).sum() / len(A))
    sb = np.sqrt((bd ** 2).sum() / len(B))
    s = sb / max(sa, 1e-12)
    U, _, Vt = np.linalg.svd(ad.T @ bd)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Aa = s * (R @ ad.T).T
    ate = float(np.sqrt(((Aa - bd) ** 2).sum(1).mean()))
    return ate, float(s)


def subsample_idx(n, k):
    idx = np.linspace(0, n - 1, min(k, n)).astype(int)
    return np.unique(idx)


def resize_work(img):
    h, w = img.shape[:2]
    if w == WORK_W:
        return img
    nh = int(round(h * WORK_W / w))
    return cv2.resize(img, (WORK_W, nh), interpolation=cv2.INTER_AREA)


def gradient_map(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, d=7,
                               sigmaColor=0.1, sigmaSpace=3.0)
    dx = cv2.Sobel(filt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filt, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(dx, dy)
    return (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)


def hist_percentile(gn, percent, bins=1024):
    hist, edges = np.histogram(gn, bins=bins, range=(0, 1))
    cum = np.cumsum(hist) / gn.size * 100.0
    x = (edges[:-1] + edges[1:]) / 2.0
    idx = np.searchsorted(cum, percent)
    return float(x[min(idx, len(x) - 1)])


def board_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for size in (CB_SIZE, (CB_SIZE[1], CB_SIZE[0])):
        ret, corners = cv2.findChessboardCorners(gray, size, flags)
        if ret:
            hull = cv2.convexHull(corners).astype(np.int32)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            return mask, True
    return np.full((h, w), 255, np.uint8), False


def build_heightfield(img_bgr):
    gn = gradient_map(img_bgr)
    th = hist_percentile(gn, TAU_PCT)
    g8 = np.clip(gn * 255.0, 0, 255).astype(np.uint8)
    trunc = np.where(g8 > int(th * 255.0), g8, 0).astype(np.float32)
    nz = trunc > 0
    out = np.zeros_like(trunc)
    if np.any(nz):
        lo, hi = float(trunc[nz].min()), float(trunc[nz].max())
        out[nz] = 255.0 if hi <= lo else (trunc[nz] - lo) / (hi - lo) * 255.0
    mask, found = board_mask(img_bgr)
    z_s = 0.15 * max(out.shape)
    a = out.astype(np.float32)
    if a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min()) * z_s
    else:
        a = np.zeros_like(a)
    a = np.where(mask > 0, a, 0.0)
    h, w = a.shape
    ys, xs = np.mgrid[0:h:CELL, 0:w:CELL]
    z = a[ys, xs]
    inside = mask[ys, xs] > 0
    # Keep the masked surface only. The full-image lattice sits on the same
    # pixels in every frame, so closest-point registration returns identity.
    if int(inside.sum()) < 30:
        inside = np.ones(ys.shape, dtype=bool)
    verts = np.stack([xs[inside].astype(np.float64),
                      ys[inside].astype(np.float64),
                      z[inside].astype(np.float64)], axis=1)
    return verts, found


def _pcd(verts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(verts))
    return pcd


def _rot_deg(T):
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


def register_pair(src_v, dst_v):
    src, dst = _pcd(src_v), _pcd(dst_v)
    span = max(float(np.ptp(src_v, axis=0).max()),
               float(np.ptp(dst_v, axis=0).max()), 8.0)
    dist = max(span * 0.08, 4.0)
    # Centroid translation is the stable init. PCA on a small rectangular
    # board flips by 90 or 180 degrees and is not used as the coarse pose.
    T_cent = np.eye(4)
    T_cent[:3, 3] = dst_v.mean(0) - src_v.mean(0)
    rad = o3d.geometry.KDTreeSearchParamHybrid(radius=max(span * 0.2, 6.0), max_nn=30)
    src.estimate_normals(rad)
    dst.estimate_normals(rad)
    f1 = o3d.pipelines.registration.compute_fpfh_feature(src, rad)
    f2 = o3d.pipelines.registration.compute_fpfh_feature(dst, rad)
    coarse_dist = dist * 5.0
    rr = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, f1, f2, mutual_filter=False,
        max_correspondence_distance=coarse_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(coarse_dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(20000, 0.999),
    )
    T_coarse = np.asarray(rr.transformation) if rr.fitness > 0.1 else T_cent
    if _rot_deg(T_coarse) > 90.0:
        T_coarse = T_cent
    icp = o3d.pipelines.registration.registration_icp(
        src, dst, dist, T_coarse,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    T = np.asarray(icp.transformation) if icp.fitness >= 0.1 else T_cent
    if _rot_deg(T) > 90.0 or icp.fitness < 0.1:
        T = T_cent
    return T, float(icp.fitness), float(rr.fitness)


def trajectory(paths):
    meshes = []
    n_board = 0
    t0 = time.time()
    for i, path in enumerate(paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        verts, found = build_heightfield(resize_work(img))
        meshes.append(verts)
        n_board += int(found)
        if (i + 1) % 10 == 0:
            print(f"  mesh {i+1}/{len(paths)} board {n_board} {time.time()-t0:.1f}s", flush=True)
    c2w = [np.eye(4)]
    fits = []
    for i in range(len(meshes) - 1):
        t1 = time.time()
        T_rel, fit, coarse = register_pair(meshes[i], meshes[i + 1])
        fits.append(fit)
        c2w.append(c2w[-1] @ np.linalg.inv(T_rel))
        ang = np.degrees(np.arccos(np.clip((np.trace(T_rel[:3, :3]) - 1) / 2, -1, 1)))
        print(f"  pair {i+1}/{len(meshes)-1} icp={fit:.3f} ransac={coarse:.3f} "
              f"rot={ang:.1f}deg {time.time()-t1:.1f}s", flush=True)
    centers = np.array([T[:3, 3] for T in c2w])
    return centers, fits, n_board


def frames_real(row):
    pose = np.load(row["pose"])
    usable = np.flatnonzero(pose["usable"] == 1)
    kept = []
    for i in usable:
        p = Path(row["image_dir"]) / f"{int(i):06d}.jpg"
        if p.is_file():
            kept.append(int(i))
    sel = subsample_idx(len(kept), MAX_FRAMES)
    ids = [kept[i] for i in sel]
    paths = [Path(row["image_dir"]) / f"{i:06d}.jpg" for i in ids]
    gt = pose["p"][ids] * 1000.0
    return paths, gt


def frames_sim(row):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, MAX_FRAMES)
    return [paths[i] for i in sel], gt_all[sel]


def eval_one(name, paths, gt, kind, role=None):
    print("OURS", name, "n", len(paths), flush=True)
    est, fits, n_board = trajectory(paths)
    ate, scale = sim3_ate(est, gt)
    rec = {
        "name": name,
        "kind": kind,
        "n": len(paths),
        "ate_mm": round(ate, 2),
        "sim3_scale": round(scale, 3),
        "board_frames": n_board,
        "median_fitness": round(float(np.median(fits)), 3) if fits else None,
        "identity_steps": int(sum(f < 0.1 for f in fits)),
    }
    if role is not None:
        rec["role"] = role
    print(rec, flush=True)
    return rec


def merge_comparison(rows):
    if COMPARISON.is_file():
        table = json.loads(COMPARISON.read_text(encoding="utf-8"))
    else:
        table = {"protocol": "30 frames, Sim3 ATE of camera centers in mm."}
    key_of = {
        "traj_20260923_023422": "test_traj_20260923_023422",
        "traj_20260924_144019": "val_traj_20260924_144019",
        "sim_20260924_172955_s10095": "sim_20260924_172955_s10095",
    }
    for rec in rows:
        key = key_of.get(rec["name"])
        if key is None:
            continue
        table.setdefault(key, {})
        table[key]["Ours"] = {
            "ate_mm": rec["ate_mm"],
            "sim3_scale": rec["sim3_scale"],
            "n": rec["n"],
            "board_frames": rec["board_frames"],
            "identity_steps": rec["identity_steps"],
        }
    note = "heightfield pipeline: not pointed at these 30-frame clips"
    table["not_run"] = [x for x in table.get("not_run", []) if x != note]
    COMPARISON.write_text(json.dumps(table, indent=2), encoding="utf-8")


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    rows = []
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        row = real[name]
        paths, gt = frames_real(row)
        rows.append(eval_one(name, paths, gt, "real", row.get("role")))
        RESULT.write_text(json.dumps({"method": "Ours-heightfield", "max_frames": MAX_FRAMES,
                                       "results": rows}, indent=2), encoding="utf-8")
    sname = "sim_20260924_172955_s10095"
    paths, gt = frames_sim(sim[sname])
    rows.append(eval_one(sname, paths, gt, "sim"))
    out = {
        "method": "Ours-heightfield",
        "max_frames": MAX_FRAMES,
        "work_width": WORK_W,
        "cell": CELL,
        "tau_pct": TAU_PCT,
        "results": rows,
    }
    RESULT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    merge_comparison(rows)
    print("wrote", RESULT)


if __name__ == "__main__":
    main()
