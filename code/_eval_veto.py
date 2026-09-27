#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Empirical verification of the heightfield normal veto (Eq. normal_veto).

On chessboard frame pairs with significant camera displacement:
  * decompose the RANSAC homography into the two physical branches,
  * select the branch whose plane normal aligns with the operative
    heightfield normal prior (rig-dominant normal used by the pipeline),
  * measure |t_sel . t_gt| (unit vectors) against the GT relative motion,
  * and compare with the standard max-cheirality heuristic.
Writes results/veto_eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

BASE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results")
IMG_DIR = BASE / "data" / "chessboard_images"
POSE_DIR = BASE / "data" / "chessboard_gt_poses"
OUT = BASE / "results" / "veto_eval.json"

# Calibrated intrinsics at native 1280x720 (paper, Section 4.1)
K = np.array([[2304.0, 0.0, 640.0], [0.0, 2304.0, 360.0], [0.0, 0.0, 1.0]])

# Operative heightfield normal prior used by the pipeline of record
# (rig-dominant board normal in camera coordinates; code/_ba_ours.py).
N_HF = np.array([0.0, 0.92, 0.38])
N_HF = N_HF / np.linalg.norm(N_HF)

N_FRAMES = 50          # same input protocol as the benchmark
MIN_DISP_MM = 1.0      # "significant camera displacement"


def load_pose(i: int) -> np.ndarray:
    return np.loadtxt(POSE_DIR / f"pose_{i:04d}.txt")


def gt_relative(T_wc_i: np.ndarray, T_wc_j: np.ndarray):
    """Relative motion mapping points from camera i to camera j."""
    T_cw_i = np.linalg.inv(T_wc_i)
    T_ji = T_wc_j @ T_cw_i
    return T_ji[:3, :3], T_ji[:3, 3]


def as_wc(T: np.ndarray) -> np.ndarray:
    """The pose files store camera-to-world; convert to world-to-camera."""
    return np.linalg.inv(T)


def sift_matches(g1, g2):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m[0] for m in raw if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 12:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def physical_branches(p1, p2):
    """Return the cheirality-satisfying branches of the homography."""
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 2.5)
    if H is None or mask is None or int(mask.sum()) < 20:
        return None
    inl = mask.ravel().astype(bool)
    p1i, p2i = p1[inl], p2[inl]
    try:
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None
    P0 = K @ np.eye(3, 4)
    branches = []
    for R, t, n in zip(Rs, ts, ns):
        t = np.asarray(t, np.float64).reshape(3)
        n = np.asarray(n, np.float64).reshape(3)
        if np.linalg.norm(t) < 1e-9:
            continue
        P1 = K @ np.hstack([R, t.reshape(3, 1)])
        Xh = cv2.triangulatePoints(P0, P1, p1i.T, p2i.T)
        X = (Xh[:3] / np.clip(Xh[3], 1e-9, None)).T
        X2 = (R @ X.T + t.reshape(3, 1)).T
        front = int(((X[:, 2] > 0) & (X2[:, 2] > 0)).sum())
        branches.append({"R": R, "t": t, "n": n, "front": front})
    # keep the two physical (max-cheirality) branches
    branches.sort(key=lambda b: -b["front"])
    return branches[:2]


def board_normal_cam(i: int) -> np.ndarray:
    """True board plane normal in camera frame i via solvePnP (5.0 mm board)."""
    img = cv2.imread(str(IMG_DIR / f"photo_{i + 1:03d}.jpg"), cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img, (8, 5))
    if not ret:
        ret, corners = cv2.findChessboardCorners(img, (5, 8))
    if not ret:
        return None
    objp = np.zeros((8 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2) * 5.0
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]  # board +z in camera frame
    return n / np.linalg.norm(n)


def main():
    poses = [as_wc(load_pose(i)) for i in range(N_FRAMES)]
    normals = [board_normal_cam(i) for i in range(N_FRAMES)]
    all_rows = []
    for stride in (1, 2, 4, 8):
        rows = []
        for i in range(0, N_FRAMES - stride):
            j = i + stride
            R_gt, t_gt = gt_relative(poses[i], poses[j])
            disp = float(np.linalg.norm(t_gt))
            if disp < MIN_DISP_MM:
                continue
            im1 = cv2.imread(str(IMG_DIR / f"photo_{i + 1:03d}.jpg"), cv2.IMREAD_GRAYSCALE)
            im2 = cv2.imread(str(IMG_DIR / f"photo_{j + 1:03d}.jpg"), cv2.IMREAD_GRAYSCALE)
            if im1 is None or im2 is None:
                continue
            got = sift_matches(im1, im2)
            if got is None:
                continue
            br = physical_branches(*got)
            if len(br) < 2:
                continue
            t_dir_gt = t_gt / max(np.linalg.norm(t_gt), 1e-12)

            # The two physical branches already carry definite signs from
            # cheirality; the veto chooses between them as returned.
            b1, b2 = br[0], br[1]

            def veto_pick(n_ref):
                s1 = abs(float((b1["n"] / max(np.linalg.norm(b1["n"]), 1e-12)) @ n_ref))
                s2 = abs(float((b2["n"] / max(np.linalg.norm(b2["n"]), 1e-12)) @ n_ref))
                b = b1 if s1 >= s2 else b2
                return b["t"] / max(np.linalg.norm(b["t"]), 1e-12)

            t_true = veto_pick(normals[i]) if normals[i] is not None else None
            t_rig = veto_pick(N_HF)
            b_heur = max(br, key=lambda b: b["front"])
            t_heur = b_heur["t"] / max(np.linalg.norm(b_heur["t"]), 1e-12)

            # Branch-selection accuracy: which branch is GT-consistent, and
            # how often does each rule pick it?
            a1 = abs(float((b1["t"] / max(np.linalg.norm(b1["t"]), 1e-12)) @ t_dir_gt))
            a2 = abs(float((b2["t"] / max(np.linalg.norm(b2["t"]), 1e-12)) @ t_dir_gt))
            gt_branch = 0 if a1 >= a2 else 1
            branch_gap = abs(a1 - a2)  # how separable the two branches are

            def veto_branch(n_ref):
                s1 = abs(float((b1["n"] / max(np.linalg.norm(b1["n"]), 1e-12)) @ n_ref))
                s2 = abs(float((b2["n"] / max(np.linalg.norm(b2["n"]), 1e-12)) @ n_ref))
                return 0 if s1 >= s2 else 1

            heur_branch = 0 if b_heur is b1 else 1

            rows.append({
                "pair": [i, j],
                "stride": stride,
                "disp": disp,
                "veto_true_agree": None if t_true is None else abs(float(t_true @ t_dir_gt)),
                "veto_rig_agree": abs(float(t_rig @ t_dir_gt)),
                "heur_agree": abs(float(t_heur @ t_dir_gt)),
                "branch_gap": branch_gap,
                "veto_true_correct": None if normals[i] is None else int(veto_branch(normals[i]) == gt_branch),
                "veto_rig_correct": int(veto_branch(N_HF) == gt_branch),
                "heur_correct": int(heur_branch == gt_branch),
            })
        all_rows.extend(rows)
        for tag, key in (("true-normal veto", "veto_true_agree"),
                         ("rig-prior  veto", "veto_rig_agree"),
                         ("cheirality heur", "heur_agree")):
            vals = np.array([r[key] for r in rows if r[key] is not None])
            if len(vals):
                print(f"stride {stride}  {tag}: n={len(vals)}  median {np.median(vals):.4f}  "
                      f"mean {vals.mean():.4f}  flip(<0.5) {(vals < 0.5).mean() * 100:.0f}%  "
                      f"misselect(<0.7) {(vals < 0.7).mean() * 100:.0f}%")
        for tag, key in (("true-normal veto", "veto_true_correct"),
                         ("rig-prior  veto", "veto_rig_correct"),
                         ("cheirality heur", "heur_correct")):
            vals = np.array([r[key] for r in rows if r[key] is not None])
            if len(vals):
                print(f"           {tag} branch-accuracy: {vals.mean() * 100:.1f}%  (n={len(vals)})")

    def agg(key, strides):
        vals = np.array([r[key] for r in all_rows if r[key] is not None and r["stride"] in strides])
        return {
            "n": int(len(vals)),
            "median": float(np.median(vals)),
            "mean": float(vals.mean()),
            "flip_rate_lt_0.5": float((vals < 0.5).mean()),
            "misselect_rate_lt_0.7": float((vals < 0.7).mean()),
        }

    def agg_sel(key, strides):
        vals = np.array([r[key] for r in all_rows if r[key] is not None and r["stride"] in strides])
        return {"n": int(len(vals)), "branch_accuracy": float(vals.mean())}

    summary = {
        "protocol": "SIFT matches, RANSAC homography, two physical branches; agreement = |t_sel . t_gt| on unit vectors; branch accuracy = fraction of pairs where the rule selects the GT-consistent branch; GT from PnP-referenced pose files; K = calibrated 2304px at 1280x720.",
        "strides_4_8": {
            "veto_true_normal": agg("veto_true_agree", (4, 8)),
            "veto_rig_prior": agg("veto_rig_agree", (4, 8)),
            "cheirality_heuristic": agg("heur_agree", (4, 8)),
            "branch_accuracy": {
                "veto_true_normal": agg_sel("veto_true_correct", (4, 8)),
                "veto_rig_prior": agg_sel("veto_rig_correct", (4, 8)),
                "cheirality_heuristic": agg_sel("heur_correct", (4, 8)),
            },
        },
        "strides_1_8": {
            "veto_true_normal": agg("veto_true_agree", (1, 2, 4, 8)),
            "veto_rig_prior": agg("veto_rig_agree", (1, 2, 4, 8)),
            "cheirality_heuristic": agg("heur_agree", (1, 2, 4, 8)),
            "branch_accuracy": {
                "veto_true_normal": agg_sel("veto_true_correct", (1, 2, 4, 8)),
                "veto_rig_prior": agg_sel("veto_rig_correct", (1, 2, 4, 8)),
                "cheirality_heuristic": agg_sel("heur_correct", (1, 2, 4, 8)),
            },
        },
        "pairs": all_rows,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
