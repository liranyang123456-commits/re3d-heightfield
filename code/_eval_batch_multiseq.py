#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch evaluation across additional real and simulation sequences from EGO_Mo.

Evaluates VGGT-1B, Ours-BA, and Ours-Gauge on:
- Additional real sequences: traj_20260923_015507 (train), traj_20260923_023241 (val), traj_20260924_143156 (train)
- Additional sim sequence: sim_20260924_173022_s10096
"""
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import roma
from kornia.feature import LoFTR

import sys
sys.path.insert(0, r"D:\vggt-main\vggt-main")
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from _opt_ego_pose import sim3_ate, scale_K, subsample_idx
from _ba_ours import build_feature_tracks, run_bundle_adjustment

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"
OUT = HERE / "ego_recon_multiseq_validation.json"

SQ = 3.0
objp = np.zeros((8 * 11, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * SQ


def vggt_centers(paths, model, device):
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
    torch.cuda.empty_cache()
    return np.asarray(centers)


def eval_real_seq(name, row, K0, vggt_model, matcher, device):
    pose = np.load(row["pose"])
    usable = np.flatnonzero(pose["usable"] == 1)
    folder = Path(row["image_dir"])
    kept = [int(i) for i in usable if (folder / f"{int(i):06d}.jpg").is_file()]
    sel = subsample_idx(len(kept), 30)
    ids = [kept[i] for i in sel]
    paths = [folder / f"{i:06d}.jpg" for i in ids]
    gt = pose["p"][ids] * 1000.0

    print(f"\nEvaluating real sequence: {name} ({len(paths)} frames)...")
    res = {"name": name, "kind": "real", "n": len(paths)}

    # 1. VGGT-1B
    try:
        t0 = time.time()
        c_vggt = vggt_centers(paths, vggt_model, device)
        ate_vggt, s_vggt = sim3_ate(c_vggt, gt)
        res["VGGT"] = {"ate_mm": round(float(ate_vggt), 2), "scale": round(float(s_vggt), 3), "time_s": round(time.time() - t0, 1)}
        print(f"  VGGT-1B: ATE = {res['VGGT']['ate_mm']} mm, scale = {res['VGGT']['scale']}")
    except Exception as e:
        res["VGGT"] = {"error": str(e)}

    # 2. Ours-BA (Multi-view Bundle Adjustment)
    try:
        t0 = time.time()
        K = scale_K(K0, (640, 360))
        pairs = build_feature_tracks(paths, matcher, K, strides=(1, 2, 3))
        c_ba = run_bundle_adjustment(len(paths), pairs, K, n_iters=400)
        ate_ba, s_ba = sim3_ate(c_ba, gt)
        res["Ours-BA"] = {"ate_mm": round(float(ate_ba), 2), "scale": round(float(s_ba), 3), "time_s": round(time.time() - t0, 1)}
        print(f"  Ours-BA: ATE = {res['Ours-BA']['ate_mm']} mm, scale = {res['Ours-BA']['scale']}")
    except Exception as e:
        res["Ours-BA"] = {"error": str(e)}

    # 3. Ours-Gauge (Metric Scale Gauge)
    try:
        gauge_frames = [0, 10, 20, 29]
        gauge_w2c = {}
        for gf in gauge_frames:
            g = cv2.imread(str(paths[gf]), cv2.IMREAD_GRAYSCALE)
            ret, c = cv2.findChessboardCorners(g, (11, 8))
            if not ret:
                ret, c = cv2.findChessboardCorners(g, (8, 11))
            if ret:
                _, rvec, tvec = cv2.solvePnP(objp, c, K0, None)
                R, _ = cv2.Rodrigues(rvec)
                gauge_w2c[gf] = (R, tvec.ravel())
        if len(gauge_w2c) >= 3:
            from scipy.spatial.transform import Slerp, Rotation as Rot
            R_interp, t_interp = [], []
            for i in range(30):
                prev_gf = max([gf for gf in gauge_w2c if gf <= i])
                next_gf = min([gf for gf in gauge_w2c if gf >= i])
                if prev_gf == next_gf:
                    R_interp.append(gauge_w2c[prev_gf][0])
                    t_interp.append(gauge_w2c[prev_gf][1])
                else:
                    alpha = (i - prev_gf) / (next_gf - prev_gf)
                    slerp = Slerp([0, 1], Rot.from_matrix([gauge_w2c[prev_gf][0], gauge_w2c[next_gf][0]]))
                    R_interp.append(slerp([alpha]).as_matrix()[0])
                    t_interp.append((1 - alpha) * gauge_w2c[prev_gf][1] + alpha * gauge_w2c[next_gf][1])
            c_gauge = [-R_interp[i].T @ t_interp[i] for i in range(30)]
            ate_gauge, s_gauge = sim3_ate(c_gauge, gt)
            res["Ours-SparseGauge"] = {"ate_mm": round(float(ate_gauge), 2), "scale": round(float(s_gauge), 3)}
            print(f"  Ours-SparseGauge: ATE = {res['Ours-SparseGauge']['ate_mm']} mm, scale = {res['Ours-SparseGauge']['scale']}")
    except Exception as e:
        res["Ours-SparseGauge"] = {"error": str(e)}

    return res


def eval_sim_seq(name, row, K0, vggt_model, matcher, device):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    sel = subsample_idx(min(len(paths), len(gt_all)), 30)
    paths_30 = [paths[i] for i in sel]
    gt_30 = gt_all[sel]

    print(f"\nEvaluating sim sequence: {name} ({len(paths_30)} frames)...")
    res = {"name": name, "kind": "sim", "n": 30}

    # 1. VGGT-1B
    try:
        t0 = time.time()
        c_vggt = vggt_centers(paths_30, vggt_model, device)
        ate_vggt, s_vggt = sim3_ate(c_vggt, gt_30)
        res["VGGT"] = {"ate_mm": round(float(ate_vggt), 2), "scale": round(float(s_vggt), 3), "time_s": round(time.time() - t0, 1)}
        print(f"  VGGT-1B: ATE = {res['VGGT']['ate_mm']} mm, scale = {res['VGGT']['scale']}")
    except Exception as e:
        res["VGGT"] = {"error": str(e)}

    # 2. Ours-BA
    try:
        t0 = time.time()
        K = scale_K(K0, (640, 360))
        pairs = build_feature_tracks(paths_30, matcher, K, strides=(1, 2, 3))
        c_ba = run_bundle_adjustment(len(paths_30), pairs, K, n_iters=400)
        ate_ba, s_ba = sim3_ate(c_ba, gt_30)
        res["Ours-BA"] = {"ate_mm": round(float(ate_ba), 2), "scale": round(float(s_ba), 3), "time_s": round(time.time() - t0, 1)}
        print(f"  Ours-BA: ATE = {res['Ours-BA']['ate_mm']} mm, scale = {res['Ours-BA']['scale']}")
    except Exception as e:
        res["Ours-BA"] = {"error": str(e)}

    return res


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    K0 = np.asarray(screen["K"], np.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading models on", device, "...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    matcher = LoFTR(pretrained="indoor").to(device).eval()

    test_real = ["traj_20260923_015507", "traj_20260923_023241", "traj_20260924_143156"]
    test_sim = ["sim_20260924_173022_s10096"]

    all_results = {}
    for name in test_real:
        if name in real:
            all_results[name] = eval_real_seq(name, real[name], K0, vggt_model, matcher, device)
            OUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    for name in test_sim:
        if name in sim:
            all_results[name] = eval_sim_seq(name, sim[name], K0, vggt_model, matcher, device)
            OUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\nAll completed! Wrote summary to {OUT}")


if __name__ == "__main__":
    main()
