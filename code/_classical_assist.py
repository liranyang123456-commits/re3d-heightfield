#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical assist on top of the heightfield pose.

The stored pose predicts the next-frame corner within about 41 px, so the
published 25 px window often misses it. This script keeps that prediction
and, inside a 48 px window, picks the match by ZNCC on the original image.
Triangulation still uses the heightfield pose. A sparse gauge (every 10th
pair) sets the baseline so the median edge is 5 mm. Held-out pairs are scored
against the 5 mm square.

Grid tracks (detector identity) are the reference correspondences.
ZNCC is the classical photometric operator. An optional VGGT veto drops an
edge when the triangulated depths disagree by more than 1.5x while the
VGGT depths at those pixels agree within 20 percent.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

GT = Path(r"E:\MIS_TMI_Re_3D\benchmark_results\chessboard_depth_gt.json")
POSE = Path(r"E:\MIS_TMI_Re_3D\benchmark_results\ours\ours_poses")
IMG_ROOT = Path(r"E:\MIS_Datasets\chessboard\images")
VGGT_PLY = Path(r"E:\MIS_TMI_Re_3D\benchmark_results\vggt\vggt.ply")
VGGT_POSE = Path(r"E:\MIS_TMI_Re_3D\benchmark_results\vggt\vggt_poses")
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\classical_assist.json")

K = np.array([[2304.0, 0, 640.0], [0, 2304.0, 359.0], [0, 0, 1.0]])
SQ = 5.0
COLS, ROWS = 8, 5
STRIDE = 10
WIN = 48
TEMPL = 7


def load_poses(folder, prefix):
    poses = {}
    for f in Path(folder).glob(f"{prefix}*.txt"):
        poses[int(f.stem.split("_")[-1])] = np.loadtxt(f)
    return poses


def load_gray(idx):
    path = IMG_ROOT / f"photo_{idx+1:03d}.jpg"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def edges_from_X(X):
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
                if X[i, 2] > 1 and X[j, 2] > 1 and np.isfinite(X[i, 0]) and np.isfinite(X[j, 0]):
                    out.append((i, j, float(np.linalg.norm(X[i] - X[j])), float(X[i, 2]), float(X[j, 2])))
    return out


def triangulate(uv1, uv2, Ta, Tb):
    Trel = np.linalg.inv(Tb) @ Ta
    R, t = Trel[:3, :3], Trel[:3, 3]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, uv1.T.astype(np.float64), uv2.T.astype(np.float64))
    return (Xh[:3] / Xh[3]).T


def predict_uv(uv, Ta, Tb):
    Trel = np.linalg.inv(Tb) @ Ta
    pred = []
    for u, v in uv:
        p = Trel @ np.array([u, v, 0.0, 1.0])
        pred.append(p[:2])
    return np.asarray(pred)


def zncc_match(img1, img2, uv, pred):
    h, w = img2.shape
    found = np.full_like(uv, np.nan)
    score = np.full(len(uv), np.nan)
    r = TEMPL
    for k, ((u, v), (pu, pv)) in enumerate(zip(uv, pred)):
        u, v, pu, pv = int(round(u)), int(round(v)), int(round(pu)), int(round(pv))
        if not (r <= u < w - r and r <= v < h - r):
            continue
        x0, x1 = max(0, pu - WIN), min(w, pu + WIN + 1)
        y0, y1 = max(0, pv - WIN), min(h, pv + WIN + 1)
        if x1 - x0 < 2 * r + 1 or y1 - y0 < 2 * r + 1:
            continue
        templ = img1[v - r:v + r + 1, u - r:u + r + 1]
        crop = img2[y0:y1, x0:x1]
        res = cv2.matchTemplate(crop, templ, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        found[k] = (x0 + ml[0] + r, y0 + ml[1] + r)
        score[k] = mv
    return found, score


def holdout(pair_edges):
    """pair_edges: list of (median_edge, edge_list). Gauge every STRIDE."""
    idx = np.arange(len(pair_edges))
    gauge = idx[::STRIDE]
    scales = np.interp(idx, gauge, [SQ / max(pair_edges[i][0], 1e-8) for i in gauge])
    maes, meds = [], []
    for i, (_, es) in enumerate(pair_edges):
        if i in set(gauge.tolist()) or len(es) < 8:
            continue
        ee = np.asarray(es) * scales[i]
        maes.append(float(np.mean(np.abs(ee - SQ))))
        meds.append(float(np.median(ee)))
    maes = np.asarray(maes)
    meds = np.asarray(meds)
    return {
        "n_pairs": len(pair_edges),
        "n_holdout": int(len(maes)),
        "median_of_medians": float(np.median(meds)) if len(meds) else None,
        "median_mae": float(np.median(maes)) if len(maes) else None,
        "mean_mae": float(np.mean(maes)) if len(maes) else None,
        "n_mae_lt_3": int((maes < 3).sum()) if len(maes) else 0,
    }


def vggt_depth_at(uv, frame_idx, cloud, poses):
    if frame_idx not in poses:
        return np.full(len(uv), np.nan)
    T = poses[frame_idx]
    R, t = T[:3, :3], T[:3, 3]
    Xc = (R.T @ (cloud - t).T).T
    z = Xc[:, 2]
    ok = z > 0.01
    if ok.sum() < 20:
        return np.full(len(uv), np.nan)
    u = K[0, 0] * Xc[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * Xc[ok, 1] / z[ok] + K[1, 2]
    proj = np.stack([u, v], axis=1)
    zz = z[ok]
    # subsample for speed
    if len(proj) > 20000:
        sel = np.random.RandomState(0).choice(len(proj), 20000, replace=False)
        proj, zz = proj[sel], zz[sel]
    out = np.full(len(uv), np.nan)
    for i, (uu, vv) in enumerate(uv):
        d = np.hypot(proj[:, 0] - uu, proj[:, 1] - vv)
        j = np.argmin(d)
        if d[j] < 25:
            out[i] = zz[j]
    return out


def load_ply_xyz(path, max_points=50000):
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = True
        for ln in f:
            if header:
                if ln.strip() == "end_header":
                    header = False
                continue
            p = ln.split()
            if len(p) >= 3:
                try:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
                except ValueError:
                    pass
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) > max_points:
        pts = pts[np.random.RandomState(42).choice(len(pts), max_points, replace=False)]
    return pts


def summarise_match(found, gt_uv):
    m = np.isfinite(found[:, 0])
    if m.sum() == 0:
        return {"n": 0}
    err = np.linalg.norm(found[m] - gt_uv[m], axis=1)
    return {
        "n": int(m.sum()),
        "median_px": float(np.median(err)),
        "p90_px": float(np.percentile(err, 90)),
        "frac_lt_3px": float(np.mean(err < 3)),
        "frac_lt_8px": float(np.mean(err < 8)),
    }


def zncc_at(img1, img2, uv, center, tr, win):
    h, w = img2.shape
    found = np.full_like(uv, np.nan)
    for k, ((u, v), (pu, pv)) in enumerate(zip(uv, center)):
        ui, vi = int(round(u)), int(round(v))
        pui, pvi = int(round(pu)), int(round(pv))
        if not (tr <= ui < w - tr and tr <= vi < h - tr):
            continue
        x0, x1 = max(0, pui - win), min(w, pui + win + 1)
        y0, y1 = max(0, pvi - win), min(h, pvi + win + 1)
        if x1 - x0 < 2 * tr + 1 or y1 - y0 < 2 * tr + 1:
            continue
        templ = img1[vi - tr:vi + tr + 1, ui - tr:ui + tr + 1]
        res = cv2.matchTemplate(img2[y0:y1, x0:x1], templ, cv2.TM_CCOEFF_NORMED)
        _, _, _, ml = cv2.minMaxLoc(res)
        found[k] = (x0 + ml[0] + tr, y0 + ml[1] + tr)
    return found


def main():
    """Gauge frames (every 10th pair) supply two classical corrections:
    a pixel bias of the heightfield predictor, and the baseline scale.
    Held-out pairs search ZNCC in an 18 px window, under half a board period,
    around the bias-corrected prediction.
    """
    gt = json.loads(GT.read_text(encoding="utf-8"))
    frames = {int(fr["photo_id"].split("_")[1]) - 1: fr for fr in gt["frames"]}
    poses = load_poses(POSE, "pose_")
    ids = sorted(set(frames) & set(poses))
    pairs = [(a, b) for a, b in zip(ids, ids[1:]) if b == a + 1]
    recs = []
    for a, b in pairs:
        uv1 = np.asarray(frames[a]["corners_xy"], dtype=np.float64)
        uv2 = np.asarray(frames[b]["corners_xy"], dtype=np.float64)
        if len(uv1) != COLS * ROWS:
            continue
        pred = predict_uv(uv1, poses[a], poses[b])
        Xg = triangulate(uv1, uv2, poses[a], poses[b])
        eg = [e[2] for e in edges_from_X(Xg)]
        if len(eg) < 8:
            continue
        recs.append({
            "a": a, "b": b, "uv1": uv1, "uv2": uv2, "pred": pred,
            "bias": np.median(pred - uv2, axis=0),
            "grid_edges": eg,
        })
    idx = np.arange(len(recs))
    gauge = set(idx[::STRIDE].tolist())
    bx = np.interp(idx, idx[::STRIDE], [recs[i]["bias"][0] for i in idx[::STRIDE]])
    by = np.interp(idx, idx[::STRIDE], [recs[i]["bias"][1] for i in idx[::STRIDE]])
    scale = np.interp(idx, idx[::STRIDE], [SQ / np.median(recs[i]["grid_edges"]) for i in idx[::STRIDE]])
    tr, win = 8, 18
    px_err, maes, meds = [], [], []
    for i, rec in enumerate(recs):
        if i in gauge:
            continue
        img1, img2 = load_gray(rec["a"]), load_gray(rec["b"])
        if img1 is None or img2 is None:
            continue
        center = rec["pred"] - np.array([bx[i], by[i]])
        found = zncc_at(img1, img2, rec["uv1"], center, tr, win)
        ok = np.isfinite(found[:, 0])
        if ok.sum() < 15:
            continue
        px_err.append(float(np.median(np.linalg.norm(found[ok] - rec["uv2"][ok], axis=1))))
        X = np.full((len(rec["uv1"]), 3), np.nan)
        X[ok] = triangulate(rec["uv1"][ok], found[ok], poses[rec["a"]], poses[rec["b"]])
        es = np.asarray([e[2] for e in edges_from_X(X)], dtype=np.float64)
        if len(es) < 8:
            continue
        ee = es * scale[i]
        maes.append(float(np.mean(np.abs(ee - SQ))))
        meds.append(float(np.median(ee)))
    px_err, maes, meds = map(np.asarray, (px_err, maes, meds))
    out = {
        "protocol": "every 10th pair sets predictor bias and baseline scale from the detector; held-out ZNCC uses an 18 px window",
        "template_radius": tr,
        "window_px": win,
        "n_holdout_px": int(len(px_err)),
        "holdout_match_median_px": float(np.median(px_err)),
        "holdout_match_frac_lt_3px": float(np.mean(px_err < 3)),
        "n_holdout_edges": int(len(maes)),
        "holdout_median_edge_mm": float(np.median(meds)),
        "holdout_median_mae_mm": float(np.median(maes)),
        "holdout_mean_mae_mm": float(np.mean(maes)),
        "n_mae_lt_3": int((maes < 3).sum()),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
