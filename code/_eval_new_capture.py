#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a pipeline run on the newly captured sequences against the
per-frame PnP ground truth (12x9-3mm calibration board, metric mm).

Metrics per session:
  - PCA shape ratio of the fused cloud
  - ATE (position RMSE after Sim3), Sim3 scale, RPE-t, on common GT frames
  - GT relative-rotation stats (N4: rotation-richness of the sequence)
  - GT outlier note: PnP on planar boards can flip; frames whose relative
    rotation to the median-neighbour exceeds 60 deg are excluded from ATE.

Usage:
  python _eval_new_capture.py --run_dir <sweep run dir> --session <name> [--cloud <ply>]
"""
from __future__ import annotations
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import load_poses, sim3_metrics

NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"


def pca_ratio(pts, max_sample=5000, seed=42):
    rng = np.random.RandomState(seed)
    if len(pts) > max_sample:
        pts = pts[rng.choice(len(pts), max_sample, replace=False)]
    d = pts - pts.mean(0)
    w = np.maximum(np.linalg.eigvalsh((d.T @ d) / len(pts))[::-1], 0)
    return float(w[2] / w[0]) if w[0] > 1e-12 else 0.0


def load_ply_xyz(path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)


def gt_trajectory(session):
    gt_dir = os.path.join(NEW_ROOT, session, "gt_poses")
    return load_poses(gt_dir, "pose_")


def filter_gt_outliers(gt):
    """Drop PnP flip outliers: frames whose rotation vs the temporal median
    neighbour exceeds 60 deg."""
    idx = sorted(gt)
    if len(idx) < 5:
        return gt, []
    Rs = {i: gt[i][:3, :3] for i in idx}
    drops = []
    for k, i in enumerate(idx):
        neigh = [j for j in idx if abs(j - i) <= 6 and j != i]
        if not neigh:
            continue
        angs = []
        for j in neigh:
            Rd = Rs[j].T @ Rs[i]
            tr = np.clip(np.trace(Rd), -1, 3)
            angs.append(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
        if np.median(angs) > 60:
            drops.append(i)
    clean = {i: gt[i] for i in idx if i not in drops}
    return clean, drops


def gt_motion_stats(gt):
    idx = sorted(gt)
    rel_r, rel_t = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        Ta, Tb = gt[a], gt[b]
        Rr = Ta[:3, :3].T @ Tb[:3, :3]
        tr = np.clip(np.trace(Rr), -1, 3)
        rel_r.append(float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))))
        rel_t.append(float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3])))
    return {"rel_rot_deg_mean": float(np.mean(rel_r)),
            "rel_rot_deg_p95": float(np.percentile(rel_r, 95)),
            "rel_rot_deg_max": float(np.max(rel_r)),
            "rel_trans_mm_mean": float(np.mean(rel_t)),
            "n": len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--cloud", default=None)
    ap.add_argument("--pose_subdir", default="estimated_poses_txt")
    ap.add_argument("--pose_prefix", default="pose_")
    ap.add_argument("--out_name", default="eval_ours.json")
    args = ap.parse_args()

    est = load_poses(os.path.join(args.run_dir, args.pose_subdir), args.pose_prefix)
    gt_all = gt_trajectory(args.session)
    gt, drops = filter_gt_outliers(gt_all)

    out = {"session": args.session, "run_dir": args.run_dir,
           "n_est": len(est), "n_gt": len(gt_all), "n_gt_clean": len(gt),
           "n_gt_outliers_dropped": len(drops)}
    out["gt_motion"] = gt_motion_stats(gt)

    m = sim3_metrics(est, gt)
    if m:
        out.update(m)
        print(f"[{args.session}] ATE={m['ate']:.2f}mm scale={m['sim3_scale']:.3f} "
              f"RPEt={m['rpe_trans_mean']:.2f}mm common={m['n_common']} "
              f"(GT outliers dropped: {len(drops)})")
    else:
        print(f"[{args.session}] <3 common frames between est and GT")

    cloud = args.cloud
    if cloud and os.path.exists(cloud):
        pts = load_ply_xyz(cloud)
        out["cloud_pts"] = int(len(pts))
        out["lambda3_lambda1"] = pca_ratio(pts)
        print(f"  cloud: {len(pts)} pts, lambda3/lambda1={out['lambda3_lambda1']:.4f}")

    out_path = os.path.join(NEW_ROOT, args.session, args.out_name)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
