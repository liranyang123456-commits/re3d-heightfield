#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pycolmap on the same 30-frame subsets used for ORB and VGGT."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pycolmap

SCREEN = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_screen.json")
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_colmap.json")
WORK = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_colmap_work")
MAX_FRAMES = 30
REAL_NAMES = ("traj_20260923_023422", "traj_20260924_144019")


def sim3_ate(est_c, gt_c):
    A = np.asarray(est_c, dtype=np.float64)
    B = np.asarray(gt_c, dtype=np.float64)
    if len(A) < 3:
        return None, None
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


def subsample_idx(n, k):
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))


def real_paths(row):
    pose = np.load(row["pose"])
    usable = np.flatnonzero(pose["usable"] == 1)
    kept = [int(i) for i in usable if (Path(row["image_dir"]) / f"{int(i):06d}.jpg").is_file()]
    sel = subsample_idx(len(kept), MAX_FRAMES)
    ids = [kept[i] for i in sel]
    paths = [Path(row["image_dir"]) / f"{i:06d}.jpg" for i in ids]
    gt = pose["p"][ids] * 1000.0
    return paths, gt


def sim_paths(row):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, MAX_FRAMES)
    return [paths[i] for i in sel], gt_all[sel]


def run_colmap(name, paths, gt):
    work = WORK / name
    img_dir = work / "images"
    if work.exists():
        shutil.rmtree(work)
    img_dir.mkdir(parents=True)
    for i, src in enumerate(paths):
        shutil.copy2(src, img_dir / f"{i:06d}.jpg")
    db = str(work / "database.db")
    print(f"[{name}] extract {len(paths)}", flush=True)
    pycolmap.extract_features(db, str(img_dir))
    try:
        pycolmap.match_sequential(db)
    except AttributeError:
        pycolmap.match_exhaustive(db)
    recs = pycolmap.incremental_mapping(db, str(img_dir), str(work))
    if not recs:
        return {"name": name, "n": len(paths), "registered": 0, "ate_mm": None}
    recon = max(recs.values(), key=lambda r: len(r.images))
    centers = {}
    for image in recon.images.values():
        w2c = image.cam_from_world()
        R = w2c.rotation.matrix()
        t = np.asarray(w2c.translation, dtype=np.float64)
        c = -R.T @ t
        m = re.search(r"(\d+)", image.name)
        centers[int(m.group(1))] = c
    common = sorted(centers)
    if len(common) < 3:
        return {"name": name, "n": len(paths), "registered": len(common), "ate_mm": None}
    est = np.array([centers[i] for i in common])
    g = gt[common]
    ate, scale = sim3_ate(est, g)
    return {
        "name": name,
        "n": len(paths),
        "registered": len(common),
        "ate_mm": None if ate is None else round(ate, 2),
        "sim3_scale": None if scale is None else round(scale, 3),
    }


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    jobs = []
    for name in REAL_NAMES:
        if name in real:
            paths, gt = real_paths(real[name])
            jobs.append((name, "real", real[name].get("role"), paths, gt))
    if screen["sim_with_images"]:
        row = screen["sim_with_images"][0]
        paths, gt = sim_paths(row)
        jobs.append((row["name"], "sim", None, paths, gt))
    rows = []
    for name, kind, role, paths, gt in jobs:
        rec = run_colmap(name, paths, gt)
        rec["kind"] = kind
        rec["role"] = role
        rows.append(rec)
        print(rec, flush=True)
        OUT.write_text(json.dumps({"method": "COLMAP", "max_frames": MAX_FRAMES, "results": rows}, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
