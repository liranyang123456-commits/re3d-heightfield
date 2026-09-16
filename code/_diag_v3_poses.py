#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare default50_v3 poses vs results2 poses (same frames)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import load_poses, sim3_metrics

v3 = load_poses(r"E:\MIS_TMI_Re_3D\sweep_runs\default50_v3\estimated_poses_txt", "pose_")
ref = load_poses(r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2\estimated_poses_txt", "pose_")

print("v3 poses:", len(v3), "ref poses:", len(ref))
common = sorted(set(v3) & set(ref))
print("common:", len(common))

if len(common) >= 3:
    # position RMSE after Sim3 (v3 aligned to ref)
    m = sim3_metrics({i: v3[i] for i in common}, {i: ref[i] for i in common})
    print(f"v3 vs results2: ATE={m['ate']:.3f} scale={m['sim3_scale']:.3f} "
          f"RPEt={m['rpe_trans_mean']:.3f} RPEr={m['rpe_rot_mean']:.2f}deg")

# per-frame relative rotation magnitude comparison for first 10 pairs
for i in range(min(10, len(common) - 1)):
    a, b = common[i], common[i + 1]
    for name, P in [("v3", v3), ("ref", ref)]:
        Rr = P[a][:3, :3].T @ P[b][:3, :3]
        tr = np.clip(np.trace(Rr), -1, 3)
        ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
        tt = np.linalg.norm(P[b][:3, 3] - P[a][:3, 3])
        if name == "v3":
            print(f"pair {a}->{b}: v3 rot={ang:.2f}deg trans={tt:.2f}", end="  ")
        else:
            print(f"ref rot={ang:.2f}deg trans={tt:.2f}")
