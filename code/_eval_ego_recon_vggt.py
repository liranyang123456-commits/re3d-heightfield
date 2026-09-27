#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VGGT reconstruction on the screened EGO-Mo sequences.

Run with the vggt conda env:
  C:\\Users\\lry\\.conda\\envs\\vggt\\python.exe code\\_eval_ego_recon_vggt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\vggt-main\vggt-main")
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

SCREEN = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_screen.json")
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_vggt.json")
MAX_FRAMES = 30
REAL_NAMES = ("traj_20260923_023422", "traj_20260924_144019")


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


def subsample_idx(n, k):
    idx = np.linspace(0, n - 1, min(k, n)).astype(int)
    return np.unique(idx)


def centers_from_images(paths, model, device):
    images = load_and_preprocess_images([str(p) for p in paths]).to(device)
    with torch.no_grad():
        preds = model(images)
    h, w = images.shape[-2], images.shape[-1]
    extri, _ = pose_encoding_to_extri_intri(preds["pose_enc"], image_size_hw=(h, w))
    extri = extri.cpu().numpy()
    centers = []
    for k in range(len(paths)):
        w2c = np.eye(4)
        w2c[:3, :] = extri[0, k]
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])
    del images, preds
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(centers)


def eval_real(row, model, device):
    pose = np.load(row["pose"])
    usable = np.flatnonzero(pose["usable"] == 1)
    kept = []
    for i in usable:
        p = Path(row["image_dir"]) / f"{int(i):06d}.jpg"
        if p.is_file():
            kept.append(int(i))
    sel = subsample_idx(len(kept), MAX_FRAMES)
    ids = [kept[i] for i in sel]
    paths = [Path(row["image_dir"]) / f"{i:06d}.jpg" for i in ids]
    gt = pose["p"][ids] * 1000.0
    est = centers_from_images(paths, model, device)
    ate, scale = sim3_ate(est, gt)
    return {"name": row["name"], "kind": "real", "role": row["role"],
            "n": len(paths), "ate_mm": round(ate, 2), "sim3_scale": round(scale, 3)}


def eval_sim(row, model, device):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, MAX_FRAMES)
    paths = [paths[i] for i in sel]
    gt = gt_all[sel]
    est = centers_from_images(paths, model, device)
    ate, scale = sim3_ate(est, gt)
    return {"name": row["name"], "kind": "sim", "n": len(paths),
            "ate_mm": round(ate, 2), "sim3_scale": round(scale, 3)}


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading VGGT on", device, flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    rows = []
    real = {r["name"]: r for r in screen["real_usable"]}
    for name in REAL_NAMES:
        if name not in real:
            print("missing", name, flush=True)
            continue
        print("VGGT", name, flush=True)
        rec = eval_real(real[name], model, device)
        rows.append(rec)
        print(rec, flush=True)
        OUT.write_text(json.dumps({"method": "VGGT", "max_frames": MAX_FRAMES, "results": rows}, indent=2), encoding="utf-8")
    if screen["sim_with_images"]:
        row = screen["sim_with_images"][0]
        print("VGGT", row["name"], flush=True)
        rec = eval_sim(row, model, device)
        rows.append(rec)
        print(rec, flush=True)
    OUT.write_text(json.dumps({"method": "VGGT", "max_frames": MAX_FRAMES, "results": rows}, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
