#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoFTR correspondences with a known-rotation camera-center solve.

Frame-to-frame translation on the board is ambiguous. LoFTR matches give a
reliable essential-matrix rotation. With those rotations held fixed, every
match is a linear constraint on the camera centers, and one least-squares
solve replaces the drifting translation chain. The 30 evaluation timestamps
are included as keyframes.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

from _eval_ego_recon_ours_feat import load_jobs, loftr_matches, work_image
from _opt_ego_pose import sim3_ate, so3

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
OUT = HERE / "ego_recon_ours_registered.json"
STRIDE = 8


def k_of(orig_w, orig_h, work_w, work_h, K0):
    K = np.array(K0, np.float64, copy=True)
    K[0, :] *= work_w / orig_w
    K[1, :] *= work_h / orig_h
    return K


def rays(pts, K):
    x = np.hstack([pts, np.ones((len(pts), 1))])
    r = (np.linalg.inv(K) @ x.T).T
    r /= np.clip(np.linalg.norm(r, axis=1, keepdims=True), 1e-8, None)
    return r


def essential_rotation(p1, p2, K):
    if len(p1) < 30:
        return None, p1, p2
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.5)
    if E is None or getattr(E, "shape", None) != (3, 3):
        return None, p1, p2
    try:
        n_in, R, _, m2 = cv2.recoverPose(E, p1, p2, K, mask=mask)
    except cv2.error:
        return None, p1, p2
    if n_in < 20:
        return None, p1, p2
    try:
        R = so3(R)
    except np.linalg.LinAlgError:
        return None, p1, p2
    if m2 is not None and int((m2.ravel() > 0).sum()) >= 20:
        ok = m2.ravel() > 0
        p1, p2 = p1[ok], p2[ok]
    if len(p1) > 120:
        sel = np.linspace(0, len(p1) - 1, 120).astype(int)
        p1, p2 = p1[sel], p2[sel]
    return R, p1, p2


def add_pair(AtA, ri, rj, R_i, R_j, ii, jj):
    bi = (R_i @ ri.T).T
    bj = (R_j @ rj.T).T
    n = np.cross(bj, bi)
    w = np.linalg.norm(n, axis=1)
    ok = w > 1e-4
    if not np.any(ok):
        return 0
    n = n[ok] / w[ok, None]
    weight = np.clip(w[ok], 0, 0.5)
    ni, nj = ii * 3, jj * 3
    for vec, wt in zip(n, weight):
        outer = np.outer(vec, vec) * float(wt)
        AtA[ni:ni + 3, ni:ni + 3] += outer
        AtA[nj:nj + 3, nj:nj + 3] += outer
        AtA[ni:ni + 3, nj:nj + 3] -= outer
        AtA[nj:nj + 3, ni:ni + 3] -= outer
    return int(ok.sum())


def solve_centers(AtA):
    # Pin the first camera at the origin. The remaining null vector is scale.
    M = AtA[3:, 3:]
    M = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(M)
    v = evecs[:, 0]
    centers = np.zeros((AtA.shape[0] // 3, 3))
    centers[1:] = v.reshape(-1, 3)
    return centers, evals[:5]


def keyframe_ids(n_frames, protocol_index):
    ids = set(range(0, n_frames, STRIDE))
    ids.update(int(i) for i in protocol_index)
    ids.add(0)
    ids.add(n_frames - 1)
    return sorted(ids)


def evaluate(job, matcher):
    paths = job["paths"]
    protocol = [int(i) for i in job["index"]]
    kf = keyframe_ids(len(paths), protocol)
    kf_pos = {frame: k for k, frame in enumerate(kf)}
    print(f"  keyframes {len(kf)}", flush=True)
    # Precompute grayscale, matches and rotations for edges (k, k+1) and (k, k+2).
    grays = []
    Ks = None
    for frame in kf:
        bgr = cv2.imread(str(paths[frame]))
        im, _, _ = work_image(bgr)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        if Ks is None:
            Ks = k_of(bgr.shape[1], bgr.shape[0], im.shape[1], im.shape[0], job["K"])
    edges = []
    for gap in (1, 2, 3):
        for a in range(len(kf) - gap):
            edges.append((a, a + gap))
    rel_R = {}
    match_cache = {}
    with torch.inference_mode():
        for s, (a, b) in enumerate(edges):
            p1, p2 = loftr_matches(matcher, grays[a], grays[b])
            R, p1, p2 = essential_rotation(p1, p2, Ks)
            match_cache[(a, b)] = (p1, p2, R)
            if R is not None and b == a + 1:
                rel_R[a] = R
            if (s + 1) % 80 == 0:
                print(f"    matches {s+1}/{len(edges)}", flush=True)
    results = []
    for name, right_multiply in (("R_right", True), ("R_left", False)):
        Rs = [np.eye(3)]
        for a in range(len(kf) - 1):
            Rrel = rel_R.get(a, np.eye(3))
            if right_multiply:
                Rs.append(Rs[-1] @ Rrel)
            else:
                Rs.append(Rrel @ Rs[-1])
        n = len(kf)
        AtA = np.zeros((3 * n, 3 * n))
        n_con = 0
        for a, b in edges:
            p1, p2, R_edge = match_cache[(a, b)]
            if R_edge is None or len(p1) < 15:
                continue
            ri, rj = rays(p1, Ks), rays(p2, Ks)
            n_con += add_pair(AtA, ri, rj, Rs[a], Rs[b], a, b)
        centers, evals = solve_centers(AtA)
        # protocol frames are keyframes
        est, gt = [], []
        for frame, g in zip(protocol, job["gt"]):
            if frame not in kf_pos:
                continue
            est.append(centers[kf_pos[frame]])
            gt.append(g)
        ate, scale = sim3_ate(np.asarray(est), np.asarray(gt))
        rec = {
            "tag": name,
            "ate_mm": round(float(ate), 2),
            "scale": round(float(scale), 3),
            "constraints": n_con,
            "evals": [round(float(x), 6) for x in evals],
            "n": len(est),
        }
        results.append(rec)
        print(f"  {name:10} ate {rec['ate_mm']:8} scale {rec['scale']} constraints {n_con} evals {rec['evals']}", flush=True)
    return results


def main():
    screen = json.loads((HERE / "ego_recon_screen.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("LoFTR", device, flush=True)
    matcher = LoFTR(pretrained="indoor").to(device).eval()
    summary = {}
    for job in load_jobs(screen):
        print(job["name"], "frames", len(job["paths"]), flush=True)
        summary[job["name"]] = evaluate(job, matcher)
        OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
