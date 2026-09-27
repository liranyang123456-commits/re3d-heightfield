#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pose-only check on V3: replace chained 4-frame relatives by a direct Kabsch.

Per-frame heightfields are in the camera frame (pixel x,y and gradient z).
7-D match lists were not archived. This script subsamples high-gradient
vertices, estimates a direct relative pose every 4 frames by robust Kabsch,
re-integrates the chain, and compares Sim3 scale / ATE with the stored chain.
It does not rewrite the published fused cloud.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _recompute_ate_vs_pnp import sim3_metrics
from _eval_new_capture import filter_gt_outliers, gt_trajectory

RUN = r"E:\MIS_TMI_Re_3D\new_capture_runs\vio_seq_20260912_104835"
SESSION = "vio_seq_20260912_104835"
OUT = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\v3_ba_skip.json"
STEP = 4
NPTS = 200
STRIDE_BYTES = 51


def load_high_z(idx, rng):
    path = os.path.join(RUN, "mesh", f"debug_mesh_{idx:04d}.ply")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        n = None
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line.strip() == b"end_header":
                break
        raw = f.read(n * STRIDE_BYTES)
    xyz = np.frombuffer(raw, dtype=np.uint8).reshape(n, STRIDE_BYTES)[:, :24].copy().view("<f8").reshape(n, 3)
    z = xyz[:, 2]
    thr = max(float(np.percentile(z, 90)), 1.0)
    sel = xyz[z >= thr]
    if len(sel) < 30:
        sel = xyz[z > 1.0]
    if len(sel) < 30:
        return None
    if len(sel) > NPTS:
        sel = sel[rng.choice(len(sel), NPTS, replace=False)]
    return sel


def load_pose(idx):
    return np.loadtxt(os.path.join(RUN, "estimated_poses_txt", f"pose_{idx:04d}.txt"))


def kabsch(A, B):
    ac, bc = A.mean(0), B.mean(0)
    H = (A - ac).T @ (B - bc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = bc - R @ ac
    return R, t


def nn_kabsch(Xi, Xj, T_i, T_j, iters=3):
    """Map camera-i points into camera j. Return T_rel with Xj ≈ R Xi + t, and median residual."""
    Ri, ti = T_i[:3, :3], T_i[:3, 3]
    Rj, tj = T_j[:3, :3], T_j[:3, 3]
    R = Rj.T @ Ri
    t = Rj.T @ (ti - tj)
    A, B = Xi, Xj
    med = None
    for _ in range(iters):
        pred = (R @ A.T).T + t
        d2 = ((pred[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(A)), j])
        med = float(np.median(dist))
        keep = dist < max(3.0 * med, 5.0)
        if keep.sum() < 12:
            break
        R, t = kabsch(A[keep], B[j[keep]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, med


def main():
    rng = np.random.RandomState(0)
    gt_all = gt_trajectory(SESSION)
    gt, drops = filter_gt_outliers(gt_all)
    idxs = list(range(0, 839, STEP))
    clouds, poses = {}, {}
    for i in idxs:
        X = load_high_z(i, rng)
        if X is None:
            continue
        clouds[i] = X
        poses[i] = load_pose(i)
    common = [i for i in idxs if i in clouds and i + STEP in clouds]
    print(f"frames {len(clouds)} links {len(common)}")

    est = {i: poses[i] for i in poses}
    base = sim3_metrics(est, gt)
    print("stored", base)

    rels = {}
    meds = []
    for i in common:
        T, med = nn_kabsch(clouds[i], clouds[i + STEP], poses[i], poses[i + STEP])
        rels[i] = T
        if med is not None:
            meds.append(med)
    print(f"kabsch median residual px-or-z: median {np.median(meds):.2f}  p90 {np.percentile(meds, 90):.2f}")

    # reintegrate. T_rel maps cam i -> cam j, and T_i = T_j @ T_rel, so T_j = T_i @ inv(T_rel)
    new = {idxs[0]: poses[idxs[0]].copy()}
    for i in common:
        Trel = rels[i]
        new[i + STEP] = new[i] @ np.linalg.inv(Trel)
    # frames that were not linked keep stored poses (should not happen)
    for i in poses:
        if i not in new:
            new[i] = poses[i]
    after = sim3_metrics(new, gt)
    print("reintegrated", after)

    out = {
        "session": SESSION,
        "step": STEP,
        "n_links": len(common),
        "kabsch_residual_median": float(np.median(meds)) if meds else None,
        "stored": base,
        "skip_kabsch": after,
        "n_gt_outliers_dropped": len(drops),
        "note": "High-gradient vertices only. Not the published fused cloud. 7-D matches were not stored.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
