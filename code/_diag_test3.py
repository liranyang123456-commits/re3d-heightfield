#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare test3_50 poses vs results2 and vs solvePnP GT."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import load_poses, sim3_metrics, make_K, make_gt_pnp

t3 = load_poses(r"E:\MIS_TMI_Re_3D\sweep_runs\test3_50\estimated_poses_txt", "pose_")
ref = load_poses(r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2\estimated_poses_txt", "pose_")

print("test3 poses:", len(t3))
common = sorted(set(t3) & set(ref))
m = sim3_metrics({i: t3[i] for i in common}, {i: ref[i] for i in common})
print(f"test3 vs results2: ATE={m['ate']:.3f} scale={m['sim3_scale']:.3f} "
      f"RPEt={m['rpe_trans_mean']:.3f} RPEr={m['rpe_rot_mean']:.2f}deg")

K = make_K(640, 358)
gt = make_gt_pnp(50, K)
m2 = sim3_metrics({i: t3[i] for i in sorted(t3)}, gt)
print(f"test3 vs solvePnP: ATE={m2['ate']:.3f}mm scale={m2['sim3_scale']:.3f} "
      f"RPEt={m2['rpe_trans_mean']:.3f} RPEr={m2['rpe_rot_mean']:.2f}deg")

for i in range(5):
    a, b = common[i], common[i + 1]
    for name, P in [("t3 ", t3), ("ref", ref)]:
        Rr = P[a][:3, :3].T @ P[b][:3, :3]
        tr = np.clip(np.trace(Rr), -1, 3)
        ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
        tt = np.linalg.norm(P[b][:3, 3] - P[a][:3, 3])
        print(f"pair {a}->{b} {name}: rot={ang:.2f}deg trans={tt:.2f}mm", end="  ")
    print()
