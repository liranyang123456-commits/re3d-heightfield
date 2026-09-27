#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ORB-SfM reconstruction on screened EGO-Mo image sequences.

Compares the chained essential-matrix trajectory with chessboard or
simulator camera centers. This is a reconstruction metric (Sim3 ATE),
not an IMU displacement score. Frames are subsampled so the first pass
finishes on CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from _screen_ego_recon import screen_real, screen_sim, CALIB, SPLIT

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
RESULT = HERE / "ego_recon_orb.json"
MAX_FRAMES = 30


def sim3_ate(est_c, gt_c):
    A = np.asarray(est_c, dtype=np.float64)
    B = np.asarray(gt_c, dtype=np.float64)
    ac, bc = A.mean(0), B.mean(0)
    ad, bd = A - ac, B - bc
    sa = np.sqrt((ad ** 2).sum() / len(A))
    sb = np.sqrt((bd ** 2).sum() / len(B))
    s = sb / max(sa, 1e-12)
    U, _, Vt = np.linalg.svd(ad.T @ bd)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Aa = s * (R @ ad.T).T
    ate = float(np.sqrt(((Aa - bd) ** 2).sum(1).mean()))
    return ate, float(s)


def orb_centers(image_paths, K):
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    poses = {0: np.eye(4)}
    prev_kp = prev_des = None
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, des = orb.detectAndCompute(img, None)
        if i == 0 or prev_des is None or des is None or len(kp) < 8:
            prev_kp, prev_des = kp, des
            poses.setdefault(i, poses.get(i - 1, np.eye(4)))
            continue
        matches = bf.knnMatch(prev_des, des, k=2)
        good = [m for m in matches if len(m) >= 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            poses[i] = poses[i - 1]
            prev_kp, prev_des = kp, des
            continue
        p1 = np.float32([prev_kp[m[0].queryIdx].pt for m in good])
        p2 = np.float32([kp[m[0].trainIdx].pt for m in good])
        E, _ = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            poses[i] = poses[i - 1]
            prev_kp, prev_des = kp, des
            continue
        _, R, t, _ = cv2.recoverPose(E, p1, p2, K)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        poses[i] = poses[i - 1] @ T
        prev_kp, prev_des = kp, des
    return np.array([poses[i][:3, 3] for i in range(len(image_paths))])


def subsample(paths, gt, max_frames):
    n = min(len(paths), len(gt))
    idx = np.linspace(0, n - 1, min(max_frames, n)).astype(int)
    idx = np.unique(idx)
    return [paths[i] for i in idx], gt[idx]


def eval_real(row, K):
    pose = np.load(row["pose"])
    usable = np.flatnonzero(pose["usable"] == 1)
    paths = [Path(row["image_dir"]) / f"{i:06d}.jpg" for i in usable]
    paths = [p for p in paths if p.is_file()]
    # usable indices may not match a filtered path list if files are missing;
    # keep only files that exist and the same pose rows.
    kept_i, kept_p = [], []
    for i in usable:
        p = Path(row["image_dir"]) / f"{int(i):06d}.jpg"
        if p.is_file():
            kept_i.append(int(i))
            kept_p.append(p)
    gt = pose["p"][kept_i] * 1000.0  # m -> mm
    imgs, gt = subsample(kept_p, gt, MAX_FRAMES)
    est = orb_centers(imgs, K)
    ate, scale = sim3_ate(est, gt)
    return {"n": len(imgs), "ate_mm": round(ate, 2), "sim3_scale": round(scale, 3)}


def eval_sim(row, K):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    imgs, gt = subsample(paths[:n], gt_all[:n], MAX_FRAMES)
    est = orb_centers(imgs, K)
    ate, scale = sim3_ate(est, gt)
    return {"n": len(imgs), "ate_mm": round(ate, 2), "sim3_scale": round(scale, 3)}


def main():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    K = np.asarray(calib["camera_matrix"], dtype=np.float64)
    real = [r for r in screen_real(split) if r["usable"]]
    sim = screen_sim()
    want = ["traj_20260923_023422", "traj_20260924_144019"]
    chosen = [r for r in real if r["name"] in want]
    if not chosen:
        chosen = real[:2]
    sim_one = sorted(sim, key=lambda r: -r["cam0"])[:1]
    rows = []
    for r in chosen:
        print("ORB", r["name"], flush=True)
        rec = eval_real(r, K)
        rec.update({"name": r["name"], "kind": "real", "role": r["role"]})
        rows.append(rec)
        print(rec, flush=True)
    # simulator intrinsics are not the real endoscope K; report the trajectory
    # shape only, and label the K as the real calibration.
    for r in sim_one:
        print("ORB", r["name"], flush=True)
        rec = eval_sim(r, K)
        rec.update({"name": r["name"], "kind": "sim", "note": "real endoscope K used; sim K may differ"})
        rows.append(rec)
        print(rec, flush=True)
    out = {"method": "ORB-SfM", "max_frames": MAX_FRAMES, "results": rows}
    RESULT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", RESULT)


if __name__ == "__main__":
    main()
