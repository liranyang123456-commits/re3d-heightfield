#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyframe homography VO.

Single-frame baselines on this rig are only a few pixels, so the recovered
translation direction is noise. A keyframe is declared once the gradient
anchors have moved about 20 px. The calibrated homography of that pair is
decomposed, and the chained camera centers are scored on the same 30 timestamps.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from _opt_ego_pose import (
    CALIB_SIZE, prep, scale_K, sim3_ate, so3, subsample_idx, track_lk,
)

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"


def decompose_all(p1, p2, K):
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or mask is None or int(mask.sum()) < 30:
        return []
    try:
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return []
    inl = mask.ravel().astype(bool)
    P0 = K @ np.eye(3, 4)
    out = []
    for i, (R, t, n) in enumerate(zip(Rs, ts, ns)):
        R = np.asarray(R, np.float64)
        if not np.isfinite(R).all():
            continue
        try:
            R = so3(R)
        except np.linalg.LinAlgError:
            continue
        t = np.asarray(t, np.float64).reshape(3)
        n = np.asarray(n, np.float64).reshape(3)
        if not np.isfinite(t).all() or np.linalg.norm(t) < 1e-8:
            continue
        if n[2] < 0:
            n = -n
            t = -t
        P1 = K @ np.hstack([R, t.reshape(3, 1)])
        Xh = cv2.triangulatePoints(P0, P1, p1[inl].T, p2[inl].T)
        X = (Xh[:3] / np.clip(Xh[3], 1e-8, None)).T
        front = int((X[:, 2] > 0).sum())
        out.append({
            "R": R, "t": t, "n": n, "front": front,
            "visible": True,
            "rot": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
        })
    return out


def pick(cands, rule):
    pool = [c for c in cands if c["visible"]] or cands
    pool = [c for c in pool if c["rot"] < 35]
    if not pool:
        return None
    if rule == "visible_minrot":
        return min(pool, key=lambda c: c["rot"])
    if rule == "max_front":
        return max(pool, key=lambda c: c["front"])
    if rule == "max_nz":
        return max(pool, key=lambda c: float(c["n"][2]))
    if rule == "front_minrot":
        return min(pool, key=lambda c: ( -c["front"], c["rot"]))
    return None


def chain_ate(frames, K, rule, stride, gt_of):
    """frames: list of (index, path). gt_of: dict index -> xyz mm."""
    c = np.eye(4)
    pose_at = {frames[0][0]: c.copy()}
    i = 0
    steps = 0
    while i < len(frames) - 1:
        j = min(i + stride, len(frames) - 1)
        im1 = cv2.imread(str(frames[i][1]), cv2.IMREAD_COLOR)
        im2 = cv2.imread(str(frames[j][1]), cv2.IMREAD_COLOR)
        chosen = None
        if im1 is not None and im2 is not None:
            g1, mag, fov = prep(im1)
            g2, _, _ = prep(im2)
            tr = track_lk(g1, mag, fov, g2)
            if tr is not None:
                cands = decompose_all(tr[0], tr[1], K)
                chosen = pick(cands, rule)
        if chosen is None and j > i + 1:
            j = i + 1
            im2 = cv2.imread(str(frames[j][1]), cv2.IMREAD_COLOR)
            if im1 is not None and im2 is not None:
                g1, mag, fov = prep(im1)
                g2, _, _ = prep(im2)
                tr = track_lk(g1, mag, fov, g2)
                if tr is not None:
                    chosen = pick(decompose_all(tr[0], tr[1], K), rule)
        T = np.eye(4)
        if chosen is not None:
            T[:3, :3] = chosen["R"]
            T[:3, 3] = chosen["t"]
            steps += 1
        c = c @ T
        for k in range(i + 1, j + 1):
            pose_at[frames[k][0]] = c.copy()
        i = j
    ids = [i for i in gt_of if i in pose_at]
    est = np.array([pose_at[i][:3, 3] for i in ids])
    gt = np.array([gt_of[i] for i in ids])
    ate, scale = sim3_ate(est, gt)
    return ate, scale, steps, len(ids)


def jobs(screen, K0):
    real = {r["name"]: r for r in screen["real_usable"]}
    out = []
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        row = real[name]
        pose = np.load(row["pose"])
        folder = Path(row["image_dir"])
        usable = [int(i) for i in np.flatnonzero(pose["usable"] == 1)
                  if (folder / f"{int(i):06d}.jpg").is_file()]
        sel = subsample_idx(len(usable), 30)
        ids = [usable[i] for i in sel]
        frames = []
        for i in range(ids[0], ids[-1] + 1):
            p = folder / f"{i:06d}.jpg"
            if p.is_file():
                frames.append((i, p))
        gt = {i: pose["p"][i] * 1000.0 for i in ids}
        out.append((name, frames, gt))
    return out, K0


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    K0 = np.asarray(screen["K"], dtype=np.float64)
    seqs, _ = jobs(screen, K0)
    rules = ["visible_minrot", "max_front", "max_nz", "front_minrot"]
    summary = {}
    for name, frames, gt in seqs:
        sample = cv2.imread(str(frames[0][1]))
        h, w = sample.shape[:2]
        K = scale_K(K0, (w, h))
        print(f"\n{name} frames {len(frames)}", flush=True)
        summary[name] = []
        for stride in (8, 12):
            for rule in rules:
                ate, scale, steps, n = chain_ate(frames, K, rule, stride, gt)
                rec = {"stride": stride, "rule": rule, "ate_mm": round(ate, 2),
                       "scale": round(scale, 3), "steps": steps, "n": n}
                summary[name].append(rec)
                print(f"  stride {stride:2} {rule:16} ate {rec['ate_mm']:8} scale {rec['scale']} steps {steps}", flush=True)
        (HERE / "ego_keyframe_opt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", HERE / "ego_keyframe_opt.json")


if __name__ == "__main__":
    main()
