#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check position/rotation ranges of GT-PnP vs Ours trajectories."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import make_K, make_gt_pnp, load_poses, BENCH

K = make_K(640, 358)
gt = make_gt_pnp(50, K)
ours = load_poses(os.path.join(BENCH, "ours", "ours_poses"), "pose_")
ours = {i: ours[i] for i in sorted(ours) if i < 50}


def stats(name, poses):
    idx = sorted(poses)
    t = np.array([poses[i][:3, 3] for i in idx])
    angs = []
    for i in idx:
        tr = np.clip(np.trace(poses[i][:3, :3]), -1, 3)
        angs.append(float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))))
    print(f"{name}: n={len(idx)}")
    print(f"  position range per axis (mm): {np.ptp(t, axis=0).round(2)}")
    print(f"  position std  per axis (mm): {t.std(axis=0).round(3)}")
    print(f"  |R| angle vs frame0: min={min(angs):.2f} max={max(angs):.2f} last={angs[-1]:.2f}")
    print(f"  first 5 t:\n{np.round(t[:5], 2)}")
    print(f"  last 3 t:\n{np.round(t[-3:], 2)}")


stats("GT-PnP", gt)
stats("Ours ", ours)

# relative rotation between consecutive frames
def rel_rot_stats(name, poses):
    idx = sorted(poses)
    rr = []
    for a, b in zip(idx[:-1], idx[1:]):
        R = poses[a][:3, :3].T @ poses[b][:3, :3]
        tr = np.clip(np.trace(R), -1, 3)
        rr.append(float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))))
    rr = np.array(rr)
    print(f"{name}: consecutive-frame rotation deg: mean={rr.mean():.3f} max={rr.max():.3f}")


rel_rot_stats("GT-PnP", gt)
rel_rot_stats("Ours ", ours)
