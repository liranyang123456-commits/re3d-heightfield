#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4: rotation-stratified pose-error analysis on the newly captured sequences.

For each completed session, align our estimated trajectory to the cleaned
solvePnP GT (Sim3 on positions), compute per-frame position error, and bin
frames by the GT relative rotation magnitude (deg per valid-frame step).
Reports mean position error per rotation bin -- tests the paper's limitation
claim that fast rotation degrades the heightfield pose.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import load_poses
from _eval_new_capture import gt_trajectory, filter_gt_outliers

RUN_ROOT = r"E:\MIS_TMI_Re_3D\new_capture_runs"
SESSIONS = ["pnp_seq_20260911_011115", "vio_seq_20260911_010902",
            "vio_seq_20260911_011013", "vio_seq_20260912_104835"]
BINS = [(0, 2), (2, 5), (5, 10), (10, 180)]   # deg per valid-frame step


def sim3_align(src, tgt):
    sc, tc = src.mean(0), tgt.mean(0)
    sd, td = src - sc, tgt - tc
    s = np.sqrt((td ** 2).sum() / len(td)) / max(np.sqrt((sd ** 2).sum() / len(sd)), 1e-8)
    U, _, Vt = np.linalg.svd(sd.T @ td)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return s, R, tc - s * R @ sc


def main():
    all_rows = []
    for sess in SESSIONS:
        est = load_poses(os.path.join(RUN_ROOT, sess, "estimated_poses_txt"), "pose_")
        gt_raw = gt_trajectory(sess)
        gt, drops = filter_gt_outliers(gt_raw)
        common = sorted(set(est) & set(gt))
        if len(common) < 5:
            print(f"{sess}: too few common frames"); continue

        # Sim3 align est->gt on positions
        E = np.array([est[i][:3, 3] for i in common])
        G = np.array([gt[i][:3, 3] for i in common])
        s, R, t = sim3_align(E, G)
        Ea = (s * (R @ E.T).T + t[None, :])
        pos_err = np.sqrt(((Ea - G) ** 2).sum(1))   # per common frame (mm)

        # GT relative rotation per step (between consecutive valid frames)
        rot_at = {}
        for a, b in zip(common[:-1], common[1:]):
            Rr = gt[a][:3, :3].T @ gt[b][:3, :3]
            tr = np.clip(np.trace(Rr), -1, 3)
            rot_at[b] = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))

        # bin frames (use the error at frame b, rotation of step a->b)
        print(f"\n[{sess}] common={len(common)}, GT outliers dropped={len(drops)}, "
              f"overall pos RMSE={np.sqrt((pos_err**2).mean()):.2f} mm")
        for lo, hi in BINS:
            sel = [i for i, fi in enumerate(common) if fi in rot_at
                   and lo <= rot_at[fi] < hi]
            if sel:
                e = pos_err[sel]
                print(f"  rot {lo:>3}-{hi:<3} deg: n={len(sel):>3}  "
                      f"pos err mean={e.mean():6.2f} mm  p95={np.percentile(e,95):6.2f} mm")
                all_rows.append({"session": sess, "bin": f"{lo}-{hi}",
                                 "n": len(sel), "mean_mm": float(e.mean()),
                                 "p95_mm": float(np.percentile(e, 95))})

    out = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\rotation_stratified.json"
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
