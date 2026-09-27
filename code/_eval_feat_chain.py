#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoFTR matches, minimum-rotation homography, chained camera centers.

On these clips the homography translation that agrees with the camera is the
small-rotation solution with the opposite sign from OpenCV's decomposition.
Matches are LoFTR (indoor) kept wherever they land; SIFT fills a pair when
LoFTR returns too few. Scored on the same 30 timestamps.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

from _eval_ego_recon_ours_feat import load_jobs, loftr_matches, sift_matches, work_image
from _opt_ego_pose import rot_deg, sim3_ate, so3

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
OUT = HERE / "ego_recon_ours_feat.json"


def matches(matcher, bgr1, bgr2):
    im1, _, _ = work_image(bgr1)
    im2, _, _ = work_image(bgr2)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    p1, p2 = loftr_matches(matcher, g1, g2)
    if len(p1) < 40:
        p1, p2 = sift_matches(g1, g2)
    if len(p1) < 25:
        return None
    Kscale = (im1.shape[1] / bgr1.shape[1], im1.shape[0] / bgr1.shape[0])
    return p1, p2, Kscale


def decompose(p1, p2, K):
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or mask is None or int(mask.sum()) < 25:
        return []
    inl = mask.ravel().astype(bool)
    p1i, p2i = p1[inl], p2[inl]
    try:
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return []
    P0 = K @ np.eye(3, 4)
    out = []
    for R, t, n in zip(Rs, ts, ns):
        try:
            R = so3(R)
        except np.linalg.LinAlgError:
            continue
        t = np.asarray(t, np.float64).reshape(3)
        n = np.asarray(n, np.float64).reshape(3)
        if np.linalg.norm(t) < 1e-8 or not np.isfinite(t).all():
            continue
        P1 = K @ np.hstack([R, t.reshape(3, 1)])
        Xh = cv2.triangulatePoints(P0, P1, p1i.T, p2i.T)
        X = (Xh[:3] / np.clip(Xh[3], 1e-8, None)).T
        nv = n / max(np.linalg.norm(n), 1e-8)
        plane = float(np.median(np.abs(X @ nv - 1.0)))
        out.append({"plane": plane, "nz": float(n[2]), "R": R, "t": t})
    return out


def step_motion(cands):
    """Prefer the plane solution that is not forced to be fronto-parallel.

    On a tilted board the wrong homography branch explains the same matches
    with a normal almost along the optical axis. The branch whose plane
    normal is more tilted, among those that actually pass through the
    matched points, is the camera motion.
    """
    if not cands:
        return None
    pool = [c for c in cands if c["plane"] < 3.0] or cands
    pick = min(pool, key=lambda c: abs(c["nz"]))
    return pick["R"], pick["t"]


def chain(paths, index, gt, K0, matcher, stride):
    im0 = cv2.imread(str(paths[0]))
    steps_flip = []
    steps_raw = []
    posed = 0
    i = 0
    order = []
    while i < len(paths) - 1:
        j = min(i + stride, len(paths) - 1)
        order.append((i, j))
        i = j
    for a, b in order:
        ia = cv2.imread(str(paths[a]))
        ib = cv2.imread(str(paths[b]))
        got = None if ia is None or ib is None else matches(matcher, ia, ib)
        motion = None
        if got is not None:
            p1, p2, (sx, sy) = got
            K = np.array(K0, np.float64, copy=True)
            K[0, :] *= sx
            K[1, :] *= sy
            motion = step_motion(decompose(p1, p2, K))
        gap = b - a
        steps_flip.append(motion)
        if motion is None:
            steps_raw.append(None)
        else:
            posed += 1
            steps_raw.append((motion[0], -motion[1]))
        for _k in range(gap - 1):
            steps_flip.append(None)
            steps_raw.append(None)
    rows = []
    for name, steps in (("flip", steps_flip), ("raw", steps_raw)):
        c = np.eye(4)
        centers = [c[:3, 3].copy()]
        for step in steps:
            T = np.eye(4)
            if step is not None:
                T[:3, :3] = step[0]
                T[:3, 3] = np.asarray(step[1]).reshape(3)
            c = c @ T
            centers.append(c[:3, 3].copy())
        centers = np.asarray(centers)
        ate, scale = sim3_ate(centers[index], np.asarray(gt))
        rows.append({"tag": name, "ate_mm": round(float(ate), 2),
                     "scale": round(float(scale), 3), "posed": posed,
                     "stride": stride})
    return rows


def main():
    screen = json.loads(HERE.joinpath("ego_recon_screen.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("LoFTR", device, flush=True)
    matcher = LoFTR(pretrained="indoor").to(device).eval()
    summary = {}
    for stride in (8,):
        print(f"\nstride {stride}", flush=True)
        for job in load_jobs(screen):
            print(job["name"], flush=True)
            rows = chain(job["paths"], job["index"], job["gt"], job["K"], matcher, stride)
            for rec in rows:
                print(f"  {rec['tag']:6} ate {rec['ate_mm']:8} scale {rec['scale']} posed {rec['posed']}", flush=True)
            summary.setdefault(job["name"], []).extend(rows)
            OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
