#!/usr/bin/env python3
"""Honest GT rotation / pose-valid stats for the four new sequences."""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _eval_new_capture import gt_trajectory, filter_gt_outliers

SESS = {
    "V1": "vio_seq_20260911_010902",
    "V2": "vio_seq_20260911_011013",
    "P1": "pnp_seq_20260911_011115",
    "V3": "vio_seq_20260912_104835",
}
META = {
    "V1": (535, 130, 127),
    "V2": (496, 141, 135),
    "P1": (496, 149, 129),
    "V3": (839, 99, 98),
}


def steps(gt, max_deg=None):
    idx = sorted(gt)
    rs = []
    for a, b in zip(idx[:-1], idx[1:]):
        Rr = gt[a][:3, :3].T @ gt[b][:3, :3]
        tr = np.clip(np.trace(Rr), -1, 3)
        r = float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))
        if max_deg is None or r < max_deg:
            rs.append(r)
    return np.array(rs) if rs else np.array([np.nan])


def main():
    for name, sess in SESS.items():
        raw = gt_trajectory(sess)
        clean, drops = filter_gt_outliers(raw)
        r_raw = steps(raw)
        r_cl = steps(clean)
        r_ok = steps(clean, 60)
        n, nv, nt = META[name]
        print(f"{name} {sess}")
        print(f"  frames={n} pose_valid={nv} ({100*nv/n:.1f}%) "
              f"traj={nt} ({100*nt/n:.1f}%) dropped={len(drops)}")
        print(f"  UNFILTERED mean/p95/max = {r_raw.mean():.2f}/"
              f"{np.percentile(r_raw,95):.1f}/{r_raw.max():.1f}  n={len(r_raw)}")
        print(f"  CLEANED    mean/p95/max = {r_cl.mean():.2f}/"
              f"{np.percentile(r_cl,95):.1f}/{r_cl.max():.1f}  n={len(r_cl)}")
        print(f"  CLEAN<60   mean/p95/max = {r_ok.mean():.2f}/"
              f"{np.percentile(r_ok,95):.1f}/{r_ok.max():.1f}  n={len(r_ok)}")


if __name__ == "__main__":
    main()
