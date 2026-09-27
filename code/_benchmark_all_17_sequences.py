#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive 17-Sequence Reconstruction Benchmark across EGO-Mo Dataset.

Evaluates on:
- 13 Real Usable Endoscopic Sequences (train, val, test, extra_train)
- 4 Rendered Surgical Simulation Sequences (s10095, s10096, s10097, s10098)

Compares:
- VGGT-1B (SOTA Transformer Baseline)
- ORB-SfM (Classical Monocular SfM)
- Ours-BA (Multi-view Bundle Adjustment)
- Ours-SparseGauge (Sparse Metric Gauge, 4 anchors)
- Ours-FullGauge (Full Metric Scale Gauge)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

import sys
sys.path.insert(0, r"D:\vggt-main\vggt-main")
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.load_fn import load_and_preprocess_images

from _opt_ego_pose import sim3_ate, scale_K, subsample_idx
from _ba_ours import build_feature_tracks, run_bundle_adjustment
from _eval_ego_recon_orb import orb_centers

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"
JSON_OUT = HERE / "ALL_17_SEQUENCES_BENCHMARK.json"
MD_OUT = HERE / "ALL_17_SEQUENCES_BENCHMARK.md"

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


def eval_one_sequence(seq_info, K0, vggt_model, matcher, device):
    name = seq_info["name"]
    kind = seq_info["kind"]
    paths = seq_info["paths"]
    gt = seq_info["gt"]
    role = seq_info.get("role", "N/A")
    
    print(f"\n=======================================================")
    print(f"  Benchmarking Sequence: {name} ({kind}, {role}) - 30 frames")
    print(f"=======================================================")
    
    res = {
        "name": name,
        "kind": kind,
        "role": role,
        "n_frames": len(paths),
    }
    
    # 1. VGGT-1B
    try:
        t0 = time.time()
        c_vggt = vggt_centers(paths, vggt_model, device)
        ate, sc = sim3_ate(c_vggt, gt)
        res["VGGT"] = {"ate_mm": round(float(ate), 2), "scale": round(float(sc), 3), "time_s": round(time.time() - t0, 1)}
        print(f"  [VGGT-1B]        ATE: {res['VGGT']['ate_mm']:6.2f} mm | Scale: {res['VGGT']['scale']:8.2f} | Time: {res['VGGT']['time_s']}s")
    except Exception as e:
        res["VGGT"] = {"error": str(e)}
        print(f"  [VGGT-1B] Error: {e}")
        
    # 2. ORB-SfM
    try:
        t0 = time.time()
        c_orb = orb_centers(paths, K0)
        ate, sc = sim3_ate(c_orb, gt)
        res["ORB-SfM"] = {"ate_mm": round(float(ate), 2), "scale": round(float(sc), 3), "time_s": round(time.time() - t0, 1)}
        print(f"  [ORB-SfM]        ATE: {res['ORB-SfM']['ate_mm']:6.2f} mm | Scale: {res['ORB-SfM']['scale']:8.2f} | Time: {res['ORB-SfM']['time_s']}s")
    except Exception as e:
        res["ORB-SfM"] = {"error": str(e)}
        
    # 3. Ours-BA (Bundle Adjustment on Multi-view tracks)
    try:
        t0 = time.time()
        K_work = scale_K(K0, (640, 360))
        pairs = build_feature_tracks(paths, matcher, K_work, strides=(1, 2, 3))
        c_ba = run_bundle_adjustment(len(paths), pairs, K_work, n_iters=400)
        ate, sc = sim3_ate(c_ba, gt)
        res["Ours-BA"] = {"ate_mm": round(float(ate), 2), "scale": round(float(sc), 3), "pairs": len(pairs), "time_s": round(time.time() - t0, 1)}
        print(f"  [Ours-BA]        ATE: {res['Ours-BA']['ate_mm']:6.2f} mm | Scale: {res['Ours-BA']['scale']:8.2f} | Pairs: {len(pairs)} | Time: {res['Ours-BA']['time_s']}s")
    except Exception as e:
        res["Ours-BA"] = {"error": str(e)}
        print(f"  [Ours-BA] Error: {e}")
        
    # 4. Gauge evaluations (only for real sequences with calibration pattern)
    if kind == "real":
        # Sparse gauge (4 frames: 0, 10, 20, 29)
        try:
            gauge_frames = [0, 10, 20, 29]
            gauge_w2c = {}
            for gf in gauge_frames:
                g = cv2.imread(str(paths[gf]), cv2.IMREAD_GRAYSCALE)
                ret, c = cv2.findChessboardCorners(g, (11, 8))
                if not ret: ret, c = cv2.findChessboardCorners(g, (8, 11))
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
                ate, sc = sim3_ate(c_gauge, gt)
                res["Ours-SparseGauge"] = {"ate_mm": round(float(ate), 2), "scale": round(float(sc), 3)}
                print(f"  [Ours-SparseG]   ATE: {res['Ours-SparseGauge']['ate_mm']:6.2f} mm | Scale: {res['Ours-SparseGauge']['scale']:8.2f}")
        except Exception as e:
            res["Ours-SparseGauge"] = {"error": str(e)}
            
        # Full gauge (all frames)
        try:
            c_pnp = []
            for p in paths:
                g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                ret, c = cv2.findChessboardCorners(g, (11, 8))
                if not ret: ret, c = cv2.findChessboardCorners(g, (8, 11))
                if ret:
                    _, rvec, tvec = cv2.solvePnP(objp, c, K0, None)
                    R, _ = cv2.Rodrigues(rvec)
                    c_pnp.append(-R.T @ tvec.ravel())
                else:
                    c_pnp.append(np.zeros(3))
            c_pnp = np.array(c_pnp)
            ate, sc = sim3_ate(c_pnp, gt)
            res["Ours-FullGauge"] = {"ate_mm": round(float(ate), 2), "scale": round(float(sc), 3)}
            print(f"  [Ours-FullG]     ATE: {res['Ours-FullGauge']['ate_mm']:6.2f} mm | Scale: {res['Ours-FullGauge']['scale']:8.2f}")
        except Exception as e:
            res["Ours-FullGauge"] = {"error": str(e)}
            
    return res


def update_markdown_table(all_results):
    lines = [
        "# 全序列 17 段综合三维重建基准对比 (Sim3 ATE, 单位: mm)",
        "",
        "| 序列名称 | 属性 | 分割角色 | VGGT-1B | ORB-SfM | **Ours-BA** | **Ours-SparseGauge** | **Ours-FullGauge** |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    
    vggt_list, orb_list, ba_list, sg_list, fg_list = [], [], [], [], []
    
    for name, r in all_results.items():
        kind = r.get("kind", "")
        role = r.get("role", "")
        vggt_str = f"{r['VGGT']['ate_mm']:.2f}" if "VGGT" in r and "ate_mm" in r["VGGT"] else "—"
        orb_str = f"{r['ORB-SfM']['ate_mm']:.2f}" if "ORB-SfM" in r and "ate_mm" in r["ORB-SfM"] else "—"
        ba_str = f"**{r['Ours-BA']['ate_mm']:.2f}**" if "Ours-BA" in r and "ate_mm" in r["Ours-BA"] else "—"
        sg_str = f"{r['Ours-SparseGauge']['ate_mm']:.2f}" if "Ours-SparseGauge" in r and "ate_mm" in r["Ours-SparseGauge"] else "—"
        fg_str = f"**{r['Ours-FullGauge']['ate_mm']:.2f}**" if "Ours-FullGauge" in r and "ate_mm" in r["Ours-FullGauge"] else "—"
        
        if "VGGT" in r and "ate_mm" in r["VGGT"]: vggt_list.append(r["VGGT"]["ate_mm"])
        if "ORB-SfM" in r and "ate_mm" in r["ORB-SfM"]: orb_list.append(r["ORB-SfM"]["ate_mm"])
        if "Ours-BA" in r and "ate_mm" in r["Ours-BA"]: ba_list.append(r["Ours-BA"]["ate_mm"])
        if "Ours-SparseGauge" in r and "ate_mm" in r["Ours-SparseGauge"]: sg_list.append(r["Ours-SparseGauge"]["ate_mm"])
        if "Ours-FullGauge" in r and "ate_mm" in r["Ours-FullGauge"]: fg_list.append(r["Ours-FullGauge"]["ate_mm"])
        
        lines.append(f"| `{name}` | {kind} | {role} | {vggt_str} | {orb_str} | {ba_str} | {sg_str} | {fg_str} |")
        
    # Summary row
    lines.append("|---|---|---|---|---|---|---|---|")
    m_vggt = f"{np.mean(vggt_list):.2f}" if vggt_list else "—"
    m_orb = f"{np.mean(orb_list):.2f}" if orb_list else "—"
    m_ba = f"**{np.mean(ba_list):.2f}**" if ba_list else "—"
    m_sg = f"{np.mean(sg_list):.2f}" if sg_list else "—"
    m_fg = f"**{np.mean(fg_list):.2f}**" if fg_list else "—"
    lines.append(f"| **全集平均 (Mean)** | — | — | {m_vggt} | {m_orb} | {m_ba} | {m_sg} | {m_fg} |")
    
    med_vggt = f"{np.median(vggt_list):.2f}" if vggt_list else "—"
    med_orb = f"{np.median(orb_list):.2f}" if orb_list else "—"
    med_ba = f"**{np.median(ba_list):.2f}**" if ba_list else "—"
    med_sg = f"{np.median(sg_list):.2f}" if sg_list else "—"
    med_fg = f"**{np.median(fg_list):.2f}**" if fg_list else "—"
    lines.append(f"| **中位数 (Median)** | — | — | {med_vggt} | {med_orb} | {med_ba} | {med_sg} | {med_fg} |")
    
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    K0 = np.asarray(screen["K"], np.float64)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Models: VGGT-1B and LoFTR on {device}...")
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    matcher = LoFTR(pretrained="indoor").to(device).eval()
    
    # Load existing progress if any
    all_results = {}
    if JSON_OUT.is_file():
        try:
            all_results = json.loads(JSON_OUT.read_text(encoding="utf-8"))
            print(f"Loaded {len(all_results)} previously completed sequence results.")
        except Exception:
            all_results = {}
            
    # Prepare all real sequences
    real_order = [
        "traj_20260923_023422",  # test
        "traj_20260924_144019",  # val
        "traj_20260923_023241",  # val
        "traj_20260923_015507",  # train
        "traj_20260923_015629",  # train
        "traj_20260923_015918",  # train
        "traj_20260923_020132",  # train
        "traj_20260923_020407",  # train
        "traj_20260924_143156",  # train
        "rigid_20260924_123958",  # extra_train
        "rigid_20260924_124243",  # extra_train
        "rigid_20260924_140510",  # extra_train
        "rigid_20260924_142902",  # extra_train
    ]
    
    sim_order = [
        "sim_20260924_172955_s10095",
        "sim_20260924_173022_s10096",
        "sim_20260924_173048_s10097",
        "sim_20260924_173114_s10098",
    ]
    
    t_start = time.time()
    for name in real_order:
        if name in all_results and "Ours-BA" in all_results[name]:
            print(f"Skipping already completed real sequence: {name}")
            continue
        if name not in real:
            continue
            
        row = real[name]
        pose = np.load(row["pose"])
        usable = np.flatnonzero(pose["usable"] == 1)
        folder = Path(row["image_dir"])
        kept = [int(i) for i in usable if (folder / f"{int(i):06d}.jpg").is_file()]
        sel = subsample_idx(len(kept), 30)
        ids = [kept[i] for i in sel]
        paths = [folder / f"{i:06d}.jpg" for i in ids]
        gt = pose["p"][ids] * 1000.0
        
        seq_info = {
            "name": name,
            "kind": "real",
            "role": row.get("role", "train"),
            "paths": paths,
            "gt": gt,
        }
        res = eval_one_sequence(seq_info, K0, vggt_model, matcher, device)
        all_results[name] = res
        JSON_OUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        update_markdown_table(all_results)
        
    for name in sim_order:
        if name in all_results and "Ours-BA" in all_results[name]:
            print(f"Skipping already completed sim sequence: {name}")
            continue
        if name not in sim:
            continue
            
        row = sim[name]
        z = np.load(row["pose"])
        key = "p_W_C" if "p_W_C" in z.files else "p"
        gt_all = np.asarray(z[key], np.float64)
        if np.nanmax(np.abs(gt_all)) < 5:
            gt_all = gt_all * 1000.0
        paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
        sel = subsample_idx(min(len(paths), len(gt_all)), 30)
        paths_30 = [paths[i] for i in sel]
        gt_30 = gt_all[sel]
        
        seq_info = {
            "name": name,
            "kind": "sim",
            "role": "sim",
            "paths": paths_30,
            "gt": gt_30,
        }
        res = eval_one_sequence(seq_info, K0, vggt_model, matcher, device)
        all_results[name] = res
        JSON_OUT.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        update_markdown_table(all_results)
        
    print(f"\n=======================================================")
    print(f"  All 17 Sequences Benchmark Complete in {(time.time() - t_start) / 60:.1f} mins!")
    print(f"  Output JSON: {JSON_OUT}")
    print(f"  Output Markdown Table: {MD_OUT}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
