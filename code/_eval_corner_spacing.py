#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Superseded. Use _eval_corner_spacing_official.py.

This draft projected clouds with a different sampler and wrote a Sim3 scale
that does not match the published ATE. It must not overwrite
results/corner_spacing.json.
"""
import sys
sys.exit("Use code/_eval_corner_spacing_official.py")

"""
Square-size measurement from reconstructed corner spacing.

Protocol (no per-corner affine):
  Detected inner corners define pixel rays. Reconstructed depth at each
  corner is the median Z of fused-cloud points projecting within 25 px.
  Adjacent grid neighbours are back-projected and their Euclidean length
  is compared with the nominal square size.

  native: method camera pose, method cloud units.
  sim3:    one global Umeyama scale/rotation/translation from camera
           centres onto the PnP trajectory (the same alignment as ATE).
           Corner depths are not used to fit this scale.

Chessboard sanity: the same native samples, after an affine fit
gt = a*pred+b, must land near the published AbsRel 0.247.
"""
from __future__ import annotations
import json
import os
import glob
import numpy as np

BENCH = r"E:\MIS_TMI_Re_3D\benchmark_results"
GT_JSON = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\chessboard_depth_gt.json"
OUT_JSON = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\corner_spacing.json"
SEARCH_R = 25.0
CB = (8, 5)
SQ = 5.0


def load_ply(path):
    try:
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(path).points, dtype=np.float64)
    except Exception:
        pts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = True
            for ln in f:
                if header:
                    if ln.strip() == "end_header":
                        header = False
                    continue
                p = ln.split()
                if len(p) >= 3:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
        return np.asarray(pts, dtype=np.float64)


def load_poses(pose_dir, prefix):
    poses = {}
    for fn in os.listdir(pose_dir):
        if fn.startswith(prefix) and fn.endswith(".txt"):
            idx = int(fn[len(prefix):-4])
            poses[idx] = np.loadtxt(os.path.join(pose_dir, fn))
    return poses


def umeyama(A, B):
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
    t = bc - s * R @ ac
    return float(s), R, t


def edges_from_grid(pts, cols, rows):
    """pts: (N,3) in row-major order, x fastest."""
    d = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                d.append(np.linalg.norm(pts[i] - pts[i + 1]))
            if r + 1 < rows:
                d.append(np.linalg.norm(pts[i] - pts[i + cols]))
    return np.asarray(d, dtype=np.float64)


def summarise(dist, sq):
    err = dist - sq
    return {
        "n_edges": int(len(dist)),
        "mean_mm": float(np.mean(dist)),
        "median_mm": float(np.median(dist)),
        "mae_mm": float(np.mean(np.abs(err))),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "median_abs_mm": float(np.median(np.abs(err))),
        "rel": float(np.mean(np.abs(err) / sq)),
    }


def project_sample(cloud, T_c2w, K, corners, radius):
    """Return median camera-frame Z and a 3D point per corner (NaN if empty)."""
    R, t = T_c2w[:3, :3], T_c2w[:3, 3]
    Xc = (R.T @ (cloud - t).T).T
    z = Xc[:, 2]
    ok = z > 1.0
    proj = np.full((len(cloud), 2), np.nan)
    proj[ok, 0] = K[0, 0] * Xc[ok, 0] / z[ok] + K[0, 2]
    proj[ok, 1] = K[1, 1] * Xc[ok, 1] / z[ok] + K[1, 2]
    pred_z = np.full(len(corners), np.nan)
    pred_x = np.full((len(corners), 3), np.nan)
    for k, (u, v) in enumerate(corners):
        d = np.hypot(proj[:, 0] - u, proj[:, 1] - v)
        m = d < radius
        if not np.any(m):
            continue
        pred_z[k] = np.median(z[m])
        # back-project along the detected corner ray (measurement, not a cloud vertex)
        ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
        pred_x[k] = ray * pred_z[k]
    return pred_z, pred_x


def gt_spacing_check(frames, K):
    cols, rows = CB
    dists = []
    for fr in frames:
        z = np.asarray(fr["depths_mm"], dtype=np.float64)
        uv = np.asarray(fr["corners_xy"], dtype=np.float64)
        if len(z) != cols * rows:
            continue
        pts = []
        Kinv = np.linalg.inv(K)
        for (u, v), zz in zip(uv, z):
            pts.append(Kinv @ np.array([u, v, 1.0]) * zz)
        dists.append(edges_from_grid(np.asarray(pts), cols, rows))
    d = np.concatenate(dists)
    return summarise(d, SQ)


def eval_method(name, ply, pose_dir, prefix, frames, K_proj, gt_c2w, frame_index):
    cloud = load_ply(ply)
    est = load_poses(pose_dir, prefix)
    common = [i for i in frame_index if i in est]
    if len(cloud) == 0 or len(common) < 3:
        return {"method": name, "error": "missing cloud or poses"}

    E = np.array([est[i][:3, 3] for i in common])
    G = np.array([gt_c2w[i][:3, 3] for i in common])
    s, R, t = umeyama(E, G)

    def run(mode):
        preds, gts, edges = [], [], []
        n_frames = 0
        for i in common:
            fr = frames[i]
            uv = np.asarray(fr["corners_xy"], dtype=np.float64)
            gz = np.asarray(fr["depths_mm"], dtype=np.float64)
            if mode == "native":
                T = est[i]
                cloud_use = cloud
            else:
                # map method world into the PnP world, then use the PnP camera
                cloud_use = (s * (R @ cloud.T).T + t)
                T = gt_c2w[i]
            pz, px = project_sample(cloud_use, T, K_proj, uv, SEARCH_R)
            good = np.isfinite(pz)
            if good.sum() < 10:
                continue
            n_frames += 1
            preds.append(pz[good])
            gts.append(gz[good])
            # spacing only where both endpoints of an edge were sampled
            cols, rows = CB
            for r in range(rows):
                for c in range(cols):
                    a = r * cols + c
                    for b in ((a + 1) if c + 1 < cols else None,
                              (a + cols) if r + 1 < rows else None):
                        if b is None:
                            continue
                        if np.isfinite(px[a, 0]) and np.isfinite(px[b, 0]):
                            edges.append(np.linalg.norm(px[a] - px[b]))
        if not preds:
            return None
        pred = np.concatenate(preds)
        gt = np.concatenate(gts)
        A = np.vstack([pred, np.ones_like(pred)]).T
        a, b = np.linalg.lstsq(A, gt, rcond=None)[0]
        pa = a * pred + b
        dist = np.asarray(edges, dtype=np.float64)
        return {
            "n_frames": n_frames,
            "n_corners": int(len(pred)),
            "absrel_affine_check": float(np.mean(np.abs(pa - gt) / gt)),
            "affine_a": float(a),
            "affine_b": float(b),
            "pred_depth_median_mm": float(np.median(pred)),
            "gt_depth_median_mm": float(np.median(gt)),
            "spacing": summarise(dist, SQ) if len(dist) else None,
        }

    return {
        "method": name,
        "n_points": int(len(cloud)),
        "n_common_poses": len(common),
        "sim3_scale": s,
        "native": run("native"),
        "sim3": run("sim3"),
    }


def make_gt_c2w(frames, K):
    """PnP camera-to-world, normalised to frame 0, matching _recompute_ate_vs_pnp."""
    import cv2
    objp = np.zeros((CB[0] * CB[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CB[0], 0:CB[1]].T.reshape(-1, 2) * SQ
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    # poses in the depth-eval K are not required; use the published pipeline K
    # only to build the trajectory that ATE already uses.
    c2w = {}
    for i, fr in enumerate(frames):
        img = cv2.imread(fr["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # pipeline reference is 640x358; depth pixels are 1280x720
        g = cv2.resize(img, (640, 358), interpolation=cv2.INTER_AREA)
        ret, corners = cv2.findChessboardCorners(g, CB, flags)
        if not ret:
            continue
        corners = cv2.cornerSubPix(g, corners, (5, 5), (-1, -1), crit)
        fx = float(np.sqrt(640 ** 2 + 358 ** 2))
        Kp = np.array([[fx, 0, 320.0], [0, fx, 179.0], [0, 0, 1.0]])
        ok, rvec, tvec = cv2.solvePnP(objp, corners, Kp, None)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec.ravel()
        c2w[i] = np.linalg.inv(w2c)
    if 0 in c2w:
        T0i = np.linalg.inv(c2w[0])
        c2w = {i: T0i @ c2w[i] for i in c2w}
    return c2w


def main():
    with open(GT_JSON, encoding="utf-8") as f:
        gt = json.load(f)
    frames = gt["frames"]
    Kgt = gt["intrinsics"]
    K_json = np.array([[Kgt["fx"], 0, Kgt["cx"]],
                       [0, Kgt["fy"], Kgt["cy"]],
                       [0, 0, 1.0]], dtype=np.float64)
    # pipeline K at 640x358, mapped onto the 1280x720 corner pixels
    fx = float(np.sqrt(640 ** 2 + 358 ** 2))
    sx, sy = 1280 / 640.0, 720 / 358.0
    K_pipe = np.array([[fx * sx, 0, 320.0 * sx],
                       [0, fx * sy, 179.0 * sy],
                       [0, 0, 1.0]], dtype=np.float64)

    print("GT spacing with json K:", gt_spacing_check(frames[:50], K_json))
    print("GT spacing with pipe K:", gt_spacing_check(frames[:50], K_pipe))

    # Use the K that recovers ~5 mm on the ground-truth depths.
    gt_json_sum = gt_spacing_check(frames[:50], K_json)
    gt_pipe_sum = gt_spacing_check(frames[:50], K_pipe)
    K = K_json if abs(gt_json_sum["mean_mm"] - SQ) < abs(gt_pipe_sum["mean_mm"] - SQ) else K_pipe
    which = "json" if K is K_json else "pipe"
    print("using K:", which)

    gt_c2w = make_gt_c2w(frames, K)
    print("PnP frames", len(gt_c2w))
    frame_index = list(range(min(50, len(frames))))

    methods = [
        ("Ours", os.path.join(BENCH, "ours", "ours.ply"),
         os.path.join(BENCH, "ours", "ours_poses"), "pose_"),
        ("ORB-SfM", os.path.join(BENCH, "orb_sfm", "orb_sfm.ply"),
         os.path.join(BENCH, "orb_sfm", "orb_poses"), "c2w_"),
        ("COLMAP", os.path.join(BENCH, "colmap", "colmap_sfm.ply"),
         os.path.join(BENCH, "colmap", "colmap_poses"), "c2w_"),
        ("VGGT", os.path.join(BENCH, "vggt", "vggt.ply"),
         os.path.join(BENCH, "vggt", "vggt_poses"), "c2w_"),
        ("DUSt3R", os.path.join(BENCH, "dust3r", "dust3r.ply"),
         os.path.join(BENCH, "dust3r", "dust3r_poses"), "c2w_"),
        ("MASt3R", os.path.join(BENCH, "mast3r", "mast3r.ply"),
         os.path.join(BENCH, "mast3r", "mast3r_poses"), "c2w_"),
        ("MUSt3R", os.path.join(BENCH, "must3r", "must3r.ply"),
         os.path.join(BENCH, "must3r", "must3r_poses"), "c2w_"),
    ]
    out = {
        "protocol": "back-project median cloud depth along corner rays; no per-corner affine",
        "square_mm": SQ,
        "search_radius_px": SEARCH_R,
        "K": which,
        "gt_spacing_check": gt_spacing_check(frames[:50], K),
        "methods": [],
    }
    for name, ply, pdir, pre in methods:
        if not os.path.exists(ply):
            print(name, "missing ply")
            continue
        print("eval", name)
        rec = eval_method(name, ply, pdir, pre, frames, K, gt_c2w, frame_index)
        out["methods"].append(rec)
        nat = rec.get("native") or {}
        sim = rec.get("sim3") or {}
        print(f"  native spacing {nat.get('spacing')}  affine-check {nat.get('absrel_affine_check')}")
        print(f"  sim3   spacing {sim.get('spacing')}  scale {rec.get('sim3_scale')}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved", OUT_JSON)


if __name__ == "__main__":
    main()
