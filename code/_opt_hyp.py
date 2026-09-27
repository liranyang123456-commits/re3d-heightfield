#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-hypothesis planar VO.

A calibrated homography has two translation directions. Each hypothesis is
started from one of them and then kept consistent with the previous step.
The smoother hypothesis is the trajectory. Gradient anchors are the tracks.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from _opt_ego_pose import prep, scale_K, sim3_ate, so3, subsample_idx, track_lk

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"
STRIDE = 8


def candidates(p1, p2, K):
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or mask is None or int(mask.sum()) < 25:
        return []
    try:
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return []
    out = []
    for R, t, n in zip(Rs, ts, ns):
        try:
            R = so3(R)
        except np.linalg.LinAlgError:
            continue
        t = np.asarray(t, np.float64).reshape(3)
        n = np.asarray(n, np.float64).reshape(3)
        if not np.isfinite(t).all() or np.linalg.norm(t) < 1e-8:
            continue
        if n[2] < 0:
            n, t = -n, -t
        tn = t / np.linalg.norm(t)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        if ang > 40:
            continue
        out.append((R, t, tn))
    # unique directions
    uniq = []
    for item in out:
        if all(abs(float(np.dot(item[2], u[2]))) < 0.95 for u in uniq):
            uniq.append(item)
    return uniq


def rels_along(frames, K):
    rels = []
    i = 0
    while i < len(frames) - 1:
        j = min(i + STRIDE, len(frames) - 1)
        im1 = cv2.imread(str(frames[i][1]))
        im2 = cv2.imread(str(frames[j][1]))
        cands = []
        if im1 is not None and im2 is not None:
            g1, mag, fov = prep(im1)
            g2, _, _ = prep(im2)
            tr = track_lk(g1, mag, fov, g2)
            if tr is not None and float(np.median(np.linalg.norm(tr[1] - tr[0], axis=1))) >= 2.0:
                cands = candidates(tr[0], tr[1], K)
        rels.append((frames[j][0], cands, i, j))
        if (len(rels) % 40) == 0:
            print(f"    pairs {len(rels)}", flush=True)
        i = j
    return rels


def follow(rels, start_idx):
    prev = None
    chosen = []
    for _fid, cands, _i, _j in rels:
        if not cands:
            chosen.append(None)
            continue
        if prev is None:
            pick = cands[min(start_idx, len(cands) - 1)]
        else:
            pick = max(cands, key=lambda c: abs(float(np.dot(c[2], prev))))
            if float(np.dot(pick[2], prev)) < 0:
                R, t, tn = pick
                pick = (R, -t, -tn)
        prev = pick[2]
        chosen.append(pick)
    return chosen


def smoothness(chosen):
    ts = [c[2] for c in chosen if c is not None]
    if len(ts) < 3:
        return -1.0
    dots = [abs(float(np.dot(a, b))) for a, b in zip(ts, ts[1:])]
    return float(np.median(dots))


def ate_of(frames, chosen, gt_of, conjugate):
    c = np.eye(4)
    pose_at = {frames[0][0]: c[:3, 3].copy()}
    cursor = 0
    for (fid, _cands, i0, j0), pick in zip(
            [(f, None, a, b) for f, _, a, b in []], []):
        pass
    # rebuild index ranges from the same stride walk
    i = 0
    k = 0
    while i < len(frames) - 1:
        j = min(i + STRIDE, len(frames) - 1)
        pick = chosen[k]
        T = np.eye(4)
        if pick is not None:
            R, t, _tn = pick
            if conjugate:
                t = -R.T @ t
                R = R.T
            T[:3, :3] = R
            T[:3, 3] = t
        c = c @ T
        for n in range(i + 1, j + 1):
            pose_at[frames[n][0]] = c[:3, 3].copy()
        i = j
        k += 1
    ids = [i for i in gt_of if i in pose_at]
    est = np.array([pose_at[i] for i in ids])
    gt = np.array([gt_of[i] for i in ids])
    return sim3_ate(est, gt)


def load_real(screen, K0):
    real = {r["name"]: r for r in screen["real_usable"]}
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
    return jobs, K0


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    K0 = np.asarray(screen["K"], np.float64)
    jobs, _ = load_real(screen, K0)
    summary = {}
    for name, frames, gt in jobs:
        im = cv2.imread(str(frames[0][1]))
        K = scale_K(K0, (im.shape[1], im.shape[0]))
        print(f"\n{name}", flush=True)
        rels = rels_along(frames, K)
        n_ok = sum(1 for _f, c, _a, _b in rels if c)
        print(f"  pairs {len(rels)} with pose {n_ok}", flush=True)
        summary[name] = []
        best = None
        for start in (0, 1):
            chosen = follow(rels, start)
            sm = smoothness(chosen)
            for conjugate in (False, True):
                ate, scale = ate_of(frames, chosen, gt, conjugate)
                rec = {"start": start, "conj": conjugate, "smooth": round(sm, 3),
                       "ate_mm": round(ate, 2), "scale": round(scale, 3)}
                summary[name].append(rec)
                print(f"  start {start} conj {int(conjugate)} smooth {sm:.3f} ate {ate:.2f} scale {scale:.3f}", flush=True)
                if best is None or sm > best[0]:
                    best = (sm, rec)
        print("  smoother", best[1] if best else None, flush=True)
        (HERE / "ego_hyp_opt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", HERE / "ego_hyp_opt.json")


if __name__ == "__main__":
    main()
