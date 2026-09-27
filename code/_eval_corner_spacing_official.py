#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native square-size error on the chessboard, using the published depth matcher.

Same projection, intrinsics, pose convention and 25 px nearest-neighbour
match as _eval_chessboard_depth_v3.py. Depth is NOT affine-aligned.
Each matched corner is back-projected along its detected ray; adjacent
grid edges are compared with the 5.0 mm square. Ground-truth depths
under the same rays recover the square (sanity check).
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"E:\MIS_TMI_Re_3D")
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\corner_spacing.json")
CB = (8, 5)
SQ = 5.0

spec = importlib.util.spec_from_file_location("ev", ROOT / "_eval_chessboard_depth_v3.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def stats(dist):
    dist = np.asarray(dist, dtype=np.float64)
    err = dist - SQ
    return {
        "n_edges": int(len(dist)),
        "mean_mm": float(np.mean(dist)),
        "median_mm": float(np.median(dist)),
        "mae_mm": float(np.mean(np.abs(err))),
        "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "p25_mm": float(np.percentile(dist, 25)),
        "p75_mm": float(np.percentile(dist, 75)),
    }


def gt_edges(gt):
    Kinv = np.linalg.inv(ev.K)
    cols, rows = CB
    edges = []
    for frame in gt["frames"]:
        uv = np.asarray(frame["corners_xy"], dtype=np.float64)
        z = np.asarray(frame["depths_mm"], dtype=np.float64)
        if len(z) != cols * rows:
            continue
        for r in range(rows):
            for c in range(cols):
                a = r * cols + c
                nbrs = []
                if c + 1 < cols:
                    nbrs.append(a + 1)
                if r + 1 < rows:
                    nbrs.append(a + cols)
                for b in nbrs:
                    pa = Kinv @ np.array([uv[a, 0], uv[a, 1], 1.0]) * z[a]
                    pb = Kinv @ np.array([uv[b, 0], uv[b, 1], 1.0]) * z[b]
                    edges.append(np.linalg.norm(pa - pb))
    return stats(edges)


def method_edges(name, ply, pose_dir, gt):
    pts = ev.load_ply(Path(ply), max_points=ev.MAX_POINTS)
    poses = ev.load_poses(Path(pose_dir))
    Kinv = np.linalg.inv(ev.K)
    cols, rows = CB
    best = None
    for conv in ("c2w", "w2c"):
        edges, preds, gts = [], [], []
        for frame in gt["frames"]:
            fidx = int(frame["photo_id"].split("_")[1]) - 1
            if fidx not in poses:
                continue
            uv = np.asarray(frame["corners_xy"], dtype=np.float64)
            gz = np.asarray(frame["depths_mm"], dtype=np.float64)
            uvp, depths = ev.project_with_convention(pts, poses[fidx], conv)
            if len(uvp) == 0:
                continue
            pred_d, gt_idx = ev.match_corners_kdtree(uvp, depths, uv, ev.SEARCH_RADIUS)
            if len(pred_d) < 8:
                continue
            z = np.full(len(uv), np.nan)
            for i, d in zip(gt_idx, pred_d):
                z[i] = d
                preds.append(d)
                gts.append(gz[i])
            for r in range(rows):
                for c in range(cols):
                    a = r * cols + c
                    nbrs = []
                    if c + 1 < cols:
                        nbrs.append(a + 1)
                    if r + 1 < rows:
                        nbrs.append(a + cols)
                    for b in nbrs:
                        if np.isfinite(z[a]) and np.isfinite(z[b]):
                            pa = Kinv @ np.array([uv[a, 0], uv[a, 1], 1.0]) * z[a]
                            pb = Kinv @ np.array([uv[b, 0], uv[b, 1], 1.0]) * z[b]
                            edges.append(np.linalg.norm(pa - pb))
        rec = (conv, len(preds), np.asarray(edges), np.asarray(preds), np.asarray(gts))
        if best is None or rec[1] > best[1]:
            best = rec
    conv, ncorn, edges, preds, gts = best
    A = np.vstack([preds, np.ones_like(preds)]).T
    a, b = np.linalg.lstsq(A, gts, rcond=None)[0]
    pa = a * preds + b
    return {
        "method": name,
        "convention": conv,
        "n_corners": int(ncorn),
        "absrel_affine_check": float(np.mean(np.abs(pa - gts) / gts)),
        "spacing": stats(edges),
    }


def main():
    np.random.seed(42)
    gt = json.loads((ev.GT_PATH).read_text(encoding="utf-8"))
    bench = ev.BENCH
    jobs = [
        ("Ours", bench / "ours" / "ours.ply", bench / "ours" / "ours_poses"),
        ("ORB-SfM", bench / "orb_sfm" / "orb_sfm.ply", bench / "orb_sfm" / "orb_poses"),
        ("COLMAP", bench / "colmap" / "colmap_sfm.ply", bench / "colmap" / "colmap_poses"),
        ("VGGT", bench / "vggt" / "vggt.ply", bench / "vggt" / "vggt_poses"),
    ]
    out = {
        "protocol": (
            "Nearest cloud point within 25 px, published K and pose convention "
            "from chessboard_depth_eval_v3. Back-project that depth along the "
            "corner ray. No affine and no Sim3. Square size 5.0 mm."
        ),
        "gt_ray_spacing": gt_edges(gt),
        "methods": [method_edges(*j, gt) for j in jobs],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
