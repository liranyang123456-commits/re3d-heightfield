#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sparse baseline gauge on chessboard corner tracks.

Triangulate consecutive frames with the stored heightfield poses and the
published K. Every 10th pair sets one scale so its median edge equals 5 mm.
Intermediate pairs receive the interpolated scale and are held out.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

GT = Path(r"E:\MIS_TMI_Re_3D\benchmark_results\chessboard_depth_gt.json")
POSE = Path(r"E:\MIS_TMI_Re_3D\benchmark_results\ours\ours_poses")
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\baseline_gauge_spacing.json")
K = np.array([[2304.0, 0, 640.0], [0, 2304.0, 359.0], [0, 0, 1.0]])
SQ = 5.0
COLS, ROWS = 8, 5
STRIDE = 10


def edges(a, b, frames, poses):
    Trel = np.linalg.inv(poses[b]) @ poses[a]
    R, t = Trel[:3, :3], Trel[:3, 3]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    uv1 = np.asarray(frames[a]["corners_xy"], np.float64).T
    uv2 = np.asarray(frames[b]["corners_xy"], np.float64).T
    Xh = cv2.triangulatePoints(P1, P2, uv1, uv2)
    X = (Xh[:3] / Xh[3]).T
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            i = r * COLS + c
            nbrs = []
            if c + 1 < COLS:
                nbrs.append(i + 1)
            if r + 1 < ROWS:
                nbrs.append(i + COLS)
            for j in nbrs:
                if X[i, 2] > 1 and X[j, 2] > 1:
                    out.append(float(np.linalg.norm(X[i] - X[j])))
    return out


def main():
    gt = json.loads(GT.read_text(encoding="utf-8"))
    frames = {int(fr["photo_id"].split("_")[1]) - 1: fr for fr in gt["frames"]}
    poses = {int(f.stem.split("_")[-1]): np.loadtxt(f) for f in POSE.glob("pose_*.txt")}
    ids = sorted(set(frames) & set(poses))
    pairs = []
    for a, b in zip(ids, ids[1:]):
        if b != a + 1:
            continue
        es = edges(a, b, frames, poses)
        if len(es) < 10:
            continue
        pairs.append((a, float(np.median(es)), np.asarray(es)))
    idx = np.arange(len(pairs))
    gauge = idx[::STRIDE]
    scales = np.interp(idx, gauge, [SQ / pairs[i][1] for i in gauge])
    maes, meds = [], []
    for i, (a, _, es) in enumerate(pairs):
        if i in set(gauge.tolist()):
            continue
        ee = es * scales[i]
        maes.append(float(np.mean(np.abs(ee - SQ))))
        meds.append(float(np.median(ee)))
    maes = np.asarray(maes)
    out = {
        "gauge_stride": STRIDE,
        "square_mm": SQ,
        "n_pairs": len(pairs),
        "n_holdout": int(len(maes)),
        "holdout_median_of_medians": float(np.median(meds)),
        "holdout_median_mae": float(np.median(maes)),
        "holdout_mean_mae": float(np.mean(maes)),
        "n_mae_lt_3": int((maes < 3).sum()),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
