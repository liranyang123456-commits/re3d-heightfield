#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyframe translation from rotationally compensated anchor flow.

Per-frame baselines are a few pixels, so translation direction is noise.
Every `stride` frames the gradient anchors are tracked, rotation comes from
the essential matrix, and translation is the least-squares fit of the
residual flow. Four discrete compositions are scored; the one that matches
the camera-center chain is kept.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from _opt_ego_pose import prep, scale_K, sim3_ate, so3, subsample_idx, track_lk

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"


def flow_pose(p1, p2, K):
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is not None and mask is not None and int(mask.sum()) >= 25:
        inl = mask.ravel().astype(bool)
        p1, p2 = p1[inl], p2[inl]
    E, em = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.5)
    if E is None or E.shape != (3, 3):
        return None
    n_in, R, _, m2 = cv2.recoverPose(E, p1, p2, K, mask=em)
    if n_in < 20:
        return None
    try:
        R = so3(R)
    except np.linalg.LinAlgError:
        return None
    if m2 is not None and int((m2.ravel() > 0).sum()) >= 15:
        p1, p2 = p1[m2.ravel() > 0], p2[m2.ravel() > 0]
    Kinv = np.linalg.inv(K)
    def normed(pts):
        x = (Kinv @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
        return x[:, :2] / np.clip(x[:, 2:3], 1e-8, None)
    q1, q2 = normed(p1), normed(p2)
    rays = np.hstack([q1, np.ones((len(q1), 1))])
    r2 = (R @ rays.T).T
    qrot = r2[:, :2] / np.clip(r2[:, 2:3], 1e-8, None)
    flow = q2 - qrot
    A = np.zeros((2 * len(qrot), 3))
    b = np.zeros(2 * len(qrot))
    A[0::2, 0] = 1.0
    A[0::2, 2] = -qrot[:, 0]
    b[0::2] = flow[:, 0]
    A[1::2, 1] = 1.0
    A[1::2, 2] = -qrot[:, 1]
    b[1::2] = flow[:, 1]
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    if not np.isfinite(t).all() or np.linalg.norm(t) < 1e-10:
        return None
    return R, t


def chain(frames, K, stride, gt_of, flip_t, transpose_R):
    c = np.eye(4)
    pose_at = {frames[0][0]: c[:3, 3].copy()}
    i = 0
    steps = 0
    while i < len(frames) - 1:
        j = min(i + stride, len(frames) - 1)
        im1 = cv2.imread(str(frames[i][1]), cv2.IMREAD_COLOR)
        im2 = cv2.imread(str(frames[j][1]), cv2.IMREAD_COLOR)
        got = None
        if im1 is not None and im2 is not None:
            g1, mag, fov = prep(im1)
            g2, _, _ = prep(im2)
            tr = track_lk(g1, mag, fov, g2)
            if tr is not None and float(np.median(np.linalg.norm(tr[1] - tr[0], axis=1))) >= 1.0:
                got = flow_pose(tr[0], tr[1], K)
        T = np.eye(4)
        if got is not None:
            R, t = got
            if transpose_R:
                R = R.T
            if flip_t:
                t = -t
            if not transpose_R:
                t = t
            T[:3, :3] = R
            T[:3, 3] = t
            steps += 1
        c = c @ T
        for k in range(i + 1, j + 1):
            pose_at[frames[k][0]] = c[:3, 3].copy()
        i = j
    ids = [i for i in gt_of if i in pose_at]
    est = np.array([pose_at[i] for i in ids])
    gt = np.array([gt_of[i] for i in ids])
    ate, scale = sim3_ate(est, gt)
    return round(ate, 2), round(scale, 3), steps


def sequences(screen, K0):
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    jobs = []
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        row = real[name]
        pose = np.load(row["pose"])
        folder = Path(row["image_dir"])
        usable = [int(i) for i in np.flatnonzero(pose["usable"] == 1)
                  if (folder / f"{int(i):06d}.jpg").is_file()]
        sel = subsample_idx(len(usable), 30)
        ids = [usable[k] for k in sel]
        frames = [(i, folder / f"{i:06d}.jpg") for i in range(ids[0], ids[-1] + 1)
                  if (folder / f"{i:06d}.jpg").is_file()]
        gt = {i: pose["p"][i] * 1000.0 for i in ids}
        jobs.append((name, frames, gt))
    row = sim["sim_20260924_172955_s10095"]
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, 30)
    frames = [(i, paths[i]) for i in range(n)]
    gt = {int(i): gt_all[int(i)] for i in sel}
    jobs.append(("sim_20260924_172955_s10095", frames, gt))
    return jobs, K0


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    K0 = np.asarray(screen["K"], np.float64)
    summary = {}
    jobs, _ = sequences(screen, K0)
    for name, frames, gt in jobs:
        im = cv2.imread(str(frames[0][1]))
        K = scale_K(K0, (im.shape[1], im.shape[0]))
        print(f"\n{name} n={len(frames)}", flush=True)
        summary[name] = []
        for stride in (8, 12):
            for flip in (False, True):
                for transp in (False, True):
                    ate, scale, steps = chain(frames, K, stride, gt, flip, transp)
                    tag = f"s{stride}_flip{int(flip)}_RT{int(transp)}"
                    rec = {"tag": tag, "ate_mm": ate, "scale": scale, "steps": steps}
                    summary[name].append(rec)
                    print(f"  {tag:22} ate {ate:8} scale {scale} steps {steps}", flush=True)
        (HERE / "ego_flow_chain.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", HERE / "ego_flow_chain.json")


if __name__ == "__main__":
    main()
