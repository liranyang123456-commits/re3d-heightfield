#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the D:-drive ablation runs (Ablation_Full / wo_*) by fusing their
per-pair clouds into world coordinates with the provided c2w poses and
computing the PCA shape ratio lambda3/lambda1 of the fused cloud.

Compares against the paper's ablation table:
  Full pipeline 0.42 / no ROI mask 0.18 / no epipolar 0.29 / no multi-view 0.38
NOTE: the D: ablation set is wo_multiscale_icp / wo_posegraph / wo_quality_gate
(different ablation axes) plus Full.
"""
from __future__ import annotations
import os, re, glob, json
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')

ABL_ROOT = r"D:\reloc3r\Data_IMU_Camera_Pose_5"
DIRS = ["Ablation_Full", "Ablation_wo_multiscale_icp",
        "Ablation_wo_posegraph", "Ablation_wo_quality_gate"]
MAX_PAIRS = 60          # cap for speed
SUBSAMPLE = 200000      # points per pair cloud after loading


def load_poses(path):
    poses = {}
    txt = open(path).read()
    for m in re.finditer(r"frame_idx=(\d+)\s*\n((?:[-0-9.eE\s]+\n){4})", txt):
        idx = int(m.group(1))
        M = np.loadtxt(m.group(2).strip().split("\n"))
        poses[idx] = M
    return poses


def pca_ratio(pts, max_sample=20000, seed=42):
    rng = np.random.RandomState(seed)
    if len(pts) > max_sample:
        pts = pts[rng.choice(len(pts), max_sample, replace=False)]
    d = pts - pts.mean(0)
    w = np.maximum(np.linalg.eigvalsh((d.T @ d) / len(pts))[::-1], 0)
    return float(w[2] / w[0]) if w[0] > 1e-12 else 0.0, len(pts)


def main():
    import open3d as o3d
    out = {}
    for d in DIRS:
        ddir = os.path.join(ABL_ROOT, d)
        pose_file = os.path.join(ddir, "camera_poses_c2w_4x4.txt")
        if not os.path.exists(pose_file):
            print(f"{d}: no pose file"); continue
        poses = load_poses(pose_file)
        pairs = sorted(glob.glob(os.path.join(ddir, "vis",
                                              "pair_prev_*_cur_*_cur_in_prev.ply")))
        pairs = pairs[:MAX_PAIRS]
        all_pts = []
        used = 0
        for p in pairs:
            m = re.search(r"pair_prev_(\d+)_cur_(\d+)_cur_in_prev\.ply", p)
            if not m:
                continue
            prev_idx = int(m.group(1))
            if prev_idx not in poses:
                continue
            pcd = o3d.io.read_point_cloud(p)
            pts = np.asarray(pcd.points)
            if len(pts) == 0:
                continue
            if len(pts) > SUBSAMPLE:
                pts = pts[np.random.RandomState(0).choice(len(pts), SUBSAMPLE,
                                                          replace=False)]
            T = poses[prev_idx]
            pts_w = (T[:3, :3] @ pts.T).T + T[:3, 3]
            all_pts.append(pts_w)
            used += 1
        if not all_pts:
            print(f"{d}: no pair clouds fused (poses={len(poses)}, pairs={len(pairs)})")
            continue
        P = np.vstack(all_pts)
        # light dedup
        vs = 0.5
        vid = np.floor(P / vs).astype(np.int64)
        fl = vid[:, 0] * 100000000 + vid[:, 1] * 10000 + vid[:, 2]
        _, fi = np.unique(fl, return_index=True)
        P = P[fi]
        r, n = pca_ratio(P)
        out[d] = {"lambda3_lambda1": r, "n_pts_fused": int(len(P)),
                  "n_pairs": used, "n_poses": len(poses)}
        print(f"{d}: lambda3/lambda1={r:.4f}, fused pts={len(P)}, pairs={used}",
              flush=True)

    os.makedirs(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results", exist_ok=True)
    with open(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ablation_dirs_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved ablation_dirs_eval.json")


if __name__ == "__main__":
    main()
