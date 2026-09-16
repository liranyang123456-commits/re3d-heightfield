#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import load_poses, sim3_metrics, make_K, make_gt_pnp

est = load_poses(r"E:\MIS_TMI_Re_3D\sweep_runs\dec14_50\estimated_poses_txt", "pose_")
gt = make_gt_pnp(50, make_K(640, 358))
m = sim3_metrics(est, gt)
print(f"dec14 vs solvePnP: ATE={m['ate']:.3f} mm, scale={m['sim3_scale']:.3f}, "
      f"RPEt={m['rpe_trans_mean']:.3f} mm  (paper: 3.89 mm / 1.11)")
