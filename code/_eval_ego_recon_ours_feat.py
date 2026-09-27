#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse gradient anchors with SIFT and LoFTR for the EGO pose comparison.

The heightfield grid is fixed to the pixel lattice, so closest-point ICP does
not follow the camera. SIFT and LoFTR supply correspondences. Matches that
do not sit on a strong gradient are dropped. The kept pairs are lifted with
the same height z = g * 0.15*max(H, W) and aligned by a rigid RANSAC, and
they are also passed to a calibrated homography.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

from _opt_ego_pose import scale_K, sim3_ate, so3, subsample_idx

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"
RESULT = HERE / "ego_recon_ours_feat.json"
CALIB_WH = (1280, 720)


def work_image(bgr):
    h, w = bgr.shape[:2]
    nw = 640 if w >= 640 else w // 8 * 8
    nh = int(round(h * (nw / w)))
    nh = max(8, nh // 8 * 8)
    nw = max(8, nw // 8 * 8)
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return img, nw / w, nh / h


def gradient(gray):
    filt = cv2.bilateralFilter(gray, 7, 25, 7)
    dx = cv2.Sobel(filt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filt, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(dx, dy)
    return mag


def height_of(mag, pts):
    h, w = mag.shape
    zs = 0.15 * max(h, w)
    g = mag / (mag.max() + 1e-6)
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    z = zs * g[ys, xs]
    return np.stack([pts[:, 0], pts[:, 1], z], axis=1)


def gate(p1, p2, mag1, mag2, pct=50):
    if len(p1) == 0:
        return p1, p2
    t1 = np.percentile(mag1, pct)
    t2 = np.percentile(mag2, pct)
    h, w = mag1.shape
    x1 = np.clip(np.round(p1[:, 0]).astype(int), 0, w - 1)
    y1 = np.clip(np.round(p1[:, 1]).astype(int), 0, h - 1)
    x2 = np.clip(np.round(p2[:, 0]).astype(int), 0, w - 1)
    y2 = np.clip(np.round(p2[:, 1]).astype(int), 0, h - 1)
    ok = (mag1[y1, x1] >= t1) & (mag2[y2, x2] >= t2)
    if int(ok.sum()) >= 30:
        return p1[ok], p2[ok]
    return p1, p2


def sift_matches(g1, g2):
    sift = cv2.SIFT_create(nfeatures=2500)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    p1, p2 = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            p1.append(k1[a.queryIdx].pt)
            p2.append(k2[a.trainIdx].pt)
    if not p1:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(p1, np.float64), np.asarray(p2, np.float64)


def loftr_matches(matcher, g1, g2):
    t1 = torch.from_numpy(g1).float()[None, None] / 255.0
    t2 = torch.from_numpy(g2).float()[None, None] / 255.0
    t1 = t1.to(next(matcher.parameters()).device)
    t2 = t2.to(t1.device)
    out = matcher({"image0": t1, "image1": t2})
    k0 = out["keypoints0"].detach().cpu().numpy()
    k1 = out["keypoints1"].detach().cpu().numpy()
    conf = out["confidence"].detach().cpu().numpy()
    ok = conf >= 0.4
    if int(ok.sum()) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return k0[ok].astype(np.float64), k1[ok].astype(np.float64)


def fuse(matcher, bgr1, bgr2):
    im1, _, _ = work_image(bgr1)
    im2, _, _ = work_image(bgr2)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    m1, m2 = gradient(g1), gradient(g2)
    parts = []
    if matcher is not None:
        try:
            a, b = loftr_matches(matcher, g1, g2)
            if len(a):
                parts.append((a, b))
        except RuntimeError:
            pass
    a, b = sift_matches(g1, g2)
    if len(a):
        parts.append((a, b))
    if not parts:
        return None
    p1 = np.vstack([a for a, _ in parts])
    p2 = np.vstack([b for _, b in parts])
    p1, p2 = gate(p1, p2, m1, m2)
    if len(p1) < 20:
        return None
    return p1, p2, m1, m2, g1.shape[1], g1.shape[0]


def rigid_ransac(P, Q, iters=80, thresh=5.0):
    n = len(P)
    if n < 8:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        R, t = _kabsch(P[idx], Q[idx])
        if R is None:
            continue
        err = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        inl = err < thresh
        count = int(inl.sum())
        if best is None or count > best[0]:
            best = (count, inl)
    if best is None or best[0] < 12:
        return None
    R, t = _kabsch(P[best[1]], Q[best[1]])
    return R, t, int(best[0])


def _kabsch(A, B):
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None, None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - R @ cA
    return R, t


def front_count(R, t, p1, p2, K):
    P0 = K @ np.eye(3, 4)
    P1 = K @ np.hstack([R, np.asarray(t, np.float64).reshape(3, 1)])
    try:
        Xh = cv2.triangulatePoints(P0, P1, p1.T.astype(np.float64), p2.T.astype(np.float64))
    except cv2.error:
        return 0
    X = (Xh[:3] / np.clip(Xh[3], 1e-8, None)).T
    X2 = (R @ X.T + np.asarray(t).reshape(3, 1)).T
    return int(((X[:, 2] > 0) & (X2[:, 2] > 0)).sum())


def homo_step(p1, p2, K):
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or mask is None or int(mask.sum()) < 20:
        return None
    inl = mask.ravel().astype(bool)
    p1i, p2i = p1[inl], p2[inl]
    E, em = cv2.findEssentialMat(p1i, p2i, K, cv2.RANSAC, 0.999, 1.5)
    if E is None or getattr(E, "shape", None) != (3, 3):
        return None
    n_in, R, _, m2 = cv2.recoverPose(E, p1i, p2i, K, mask=em)
    if n_in < 15:
        return None
    try:
        R = so3(R)
    except np.linalg.LinAlgError:
        return None
    if m2 is not None and int((m2.ravel() > 0).sum()) >= 12:
        p1i, p2i = p1i[m2.ravel() > 0], p2i[m2.ravel() > 0]
    Kinv = np.linalg.inv(K)
    def normed(pts):
        x = (Kinv @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
        return x[:, :2] / np.clip(x[:, 2:3], 1e-8, None)
    q1, q2 = normed(p1i), normed(p2i)
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
    options = []
    for sgn in (1.0, -1.0):
        tt = sgn * t
        options.append((R, tt, front_count(R, tt, p1i, p2i, K)))
        Rc = R.T
        tc = -Rc @ tt
        options.append((Rc, tc, front_count(Rc, tc, p1i, p2i, K)))
    R_best, t_best, score = max(options, key=lambda o: o[2])
    if score < 12:
        return None
    return R_best, t_best, score


def apply_chain(steps, invert):
    c = np.eye(4)
    centers = [c[:3, 3].copy()]
    used = 0
    for step in steps:
        T = np.eye(4)
        if step is not None:
            R, t = step
            T[:3, :3] = R
            T[:3, 3] = np.asarray(t, np.float64).reshape(3)
            used += 1
            if invert:
                T = np.linalg.inv(T)
        c = c @ T
        centers.append(c[:3, 3].copy())
    return np.asarray(centers), used


def pair_list(paths, stride):
    if stride <= 1:
        return [(i, i + 1) for i in range(len(paths) - 1)]
    pairs = []
    i = 0
    while i < len(paths) - 1:
        j = min(i + stride, len(paths) - 1)
        pairs.append((i, j))
        i = j
    return pairs


def run_pass(paths, index, gt, K0, matcher, stride, tag):
    im0 = cv2.imread(str(paths[0]))
    work0, _, _ = work_image(im0)
    K = scale_K(K0, (work0.shape[1], work0.shape[0]))
    # K was scaled from 1280x720. work_image already resized from the file,
    # so scale from the file size, not from CALIB, if the file differs.
    K = np.array(K0, np.float64, copy=True)
    K[0, :] *= work0.shape[1] / im0.shape[1]
    K[1, :] *= work0.shape[0] / im0.shape[0]
    rigid_steps = []
    homo_steps = []
    n_match = 0
    for a, b in pair_list(paths, stride):
        ia = cv2.imread(str(paths[a]))
        ib = cv2.imread(str(paths[b]))
        got = None if ia is None or ib is None else fuse(matcher, ia, ib)
        if got is None:
            # hold for every frame in between by repeating identity steps
            for _ in range(b - a):
                rigid_steps.append(None)
                homo_steps.append(None)
            continue
        p1, p2, m1, m2, _, _ = got
        n_match += 1
        P, Q = height_of(m1, p1), height_of(m2, p2)
        rig = rigid_ransac(P, Q)
        homo = homo_step(p1, p2, K)
        # one motion for the whole gap, then identity for the skipped frames
        rigid_steps.append(None if rig is None else (rig[0], rig[1]))
        homo_steps.append(None if homo is None else (homo[0], homo[1]))
        for _ in range(b - a - 1):
            rigid_steps.append(None)
            homo_steps.append(None)
    rows = []
    gt = np.asarray(gt)
    for name, steps in (("rigid", rigid_steps), ("homo", homo_steps)):
        for invert in (False, True):
            centers, used = apply_chain(steps, invert)
            if len(centers) != len(paths):
                # chain length follows steps+1; paths length should match
                n = min(len(centers), len(paths))
                centers = centers[:n]
            ate, scale = sim3_ate(centers[index], gt[:len(index)])
            rec = {
                "tag": f"{tag}:{name}{'_inv' if invert else ''}",
                "ate_mm": round(float(ate), 2),
                "scale": round(float(scale), 3),
                "posed": used,
                "matched_pairs": n_match,
            }
            rows.append(rec)
            print(f"  {rec['tag']:28} ate {rec['ate_mm']:8} scale {rec['scale']} posed {used}", flush=True)
    return rows


def load_jobs(screen):
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    K0 = np.asarray(screen["K"], np.float64)
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
        id_to_k = {i: k for k, (i, _) in enumerate(frames)}
        jobs.append({
            "name": name,
            "paths": [p for _, p in frames],
            "index": [id_to_k[i] for i in ids],
            "proto_paths": [folder / f"{i:06d}.jpg" for i in ids],
            "gt": pose["p"][ids] * 1000.0,
            "K": K0,
        })
    row = sim["sim_20260924_172955_s10095"]
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, 30)
    jobs.append({
        "name": "sim_20260924_172955_s10095",
        "paths": paths[:n],
        "index": sel.tolist(),
        "proto_paths": [paths[i] for i in sel],
        "gt": gt_all[sel],
        "K": K0,
    })
    return jobs


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading LoFTR indoor on", device, flush=True)
    matcher = LoFTR(pretrained="indoor").to(device).eval()
    summary = {}
    for job in load_jobs(screen):
        print(f"\n{job['name']}", flush=True)
        rows = []
        print("  30-frame pairs", flush=True)
        rows += run_pass(job["proto_paths"], list(range(30)), job["gt"], job["K"],
                         matcher, 1, "f30")
        print("  video stride 8", flush=True)
        rows += run_pass(job["paths"], job["index"], job["gt"], job["K"],
                         matcher, 8, "video8")
        summary[job["name"]] = rows
        RESULT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", RESULT, flush=True)


if __name__ == "__main__":
    main()
