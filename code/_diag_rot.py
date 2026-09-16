#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose per-frame rotation error vs GT-PnP (constant offset vs drift)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import make_K, make_gt_pnp, load_poses, BENCH

K = make_K(640, 358)
gt = make_gt_pnp(50, K)
ours = load_poses(os.path.join(BENCH, "ours", "ours_poses"), "pose_")
vggt = load_poses(os.path.join(BENCH, "vggt", "vggt_poses"), "c2w_")


def rot_series(est, gt, n=10):
    out = []
    for i in sorted(set(est) & set(gt))[:n]:
        Rd = gt[i][:3, :3].T @ est[i][:3, :3]
        tr = np.clip(np.trace(Rd), -1, 3)
        ang = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
        out.append((i, round(ang, 2)))
    return out


print("Ours per-frame rot err (first 10):", rot_series(ours, gt))
print("VGGT per-frame rot err (first 10):", rot_series(vggt, gt))

i = 1
print("\nOurs[1] R:\n", np.round(ours[i][:3, :3], 4))
print("GT[1] R:\n", np.round(gt[i][:3, :3], 4))
print("\nOurs[1] t:", np.round(ours[i][:3, 3], 3))
print("GT[1] t:", np.round(gt[i][:3, 3], 3))

# Check if a constant rotation C explains Ours' offset: C = gtR @ estR^T per frame
Cs = []
for i in sorted(set(ours) & set(gt))[:10]:
    Cs.append(gt[i][:3, :3] @ ours[i][:3, :3].T)
C0 = Cs[0]
dev = [float(np.degrees(np.arccos(np.clip((np.trace(C0.T @ C) - 1) / 2, -1, 1)))) for C in Cs]
print("\nConstant-offset hypothesis: C=gtR*estR^T deviation from C0 per frame (deg):",
      [round(d, 2) for d in dev])
print("C0 angle:", round(float(np.degrees(np.arccos(np.clip((np.trace(C0) - 1) / 2, -1, 1)))), 2))
