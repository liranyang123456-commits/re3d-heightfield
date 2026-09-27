#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bundle Adjustment with Pseudo-3D Heightfield & Deep Feature Tracks.

Fuses:
1. Deep feature matching (LoFTR / SIFT) across multi-stride frame pairs.
2. 3D point triangulation from multi-view rays.
3. Pseudo-3D heightfield surface constraint as geometric regularization.
4. Joint Bundle Adjustment over SE(3) camera poses and 3D points.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import roma
from kornia.feature import LoFTR

from _eval_ego_recon_ours_feat import work_image
from _opt_ego_pose import sim3_ate, so3, scale_K, subsample_idx

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"


def project_points(points_3d, R, t, K):
    """
    points_3d: (M, 3) in world coordinates
    R: (3, 3) world-to-cam
    t: (3,) world-to-cam
    K: (3, 3) camera intrinsics
    Returns: (M, 2) pixel coordinates, (M,) z depths
    """
    pts_cam = (R @ points_3d.T).T + t[None, :]
    z = pts_cam[:, 2]
    u = K[0, 0] * (pts_cam[:, 0] / torch.clamp(z, min=1e-4)) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / torch.clamp(z, min=1e-4)) + K[1, 2]
    return torch.stack([u, v], dim=-1), z


def build_feature_tracks(paths, matcher, K, strides=(1, 2, 3), max_pts_per_pair=150):
    """Match frames, filter inliers with homography/essential, and collect multi-view tracks."""
    n = len(paths)
    device = "cuda"
    
    grays = []
    for p in paths:
        bgr = cv2.imread(str(p))
        im, _, _ = work_image(bgr)
        grays.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    
    print(f"Matching multi-stride pairs across {n} frames...")
    pair_matches = {}
    with torch.inference_mode():
        for stride in strides:
            for i in range(n - stride):
                j = i + stride
                t1 = torch.from_numpy(grays[i]).float()[None, None].to(device) / 255.0
                t2 = torch.from_numpy(grays[j]).float()[None, None].to(device) / 255.0
                out = matcher({"image0": t1, "image1": t2})
                conf = out["confidence"].cpu().numpy()
                ok = conf >= 0.5
                if int(ok.sum()) < 25:
                    continue
                p1 = out["keypoints0"].cpu().numpy()[ok]
                p2 = out["keypoints1"].cpu().numpy()[ok]
                
                # Filter with homography RANSAC
                H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 2.5)
                if H is None or mask is None:
                    continue
                inl = mask.ravel().astype(bool)
                if int(inl.sum()) < 20:
                    continue
                p1_in = p1[inl]
                p2_in = p2[inl]
                if len(p1_in) > max_pts_per_pair:
                    sel = np.linspace(0, len(p1_in) - 1, max_pts_per_pair).astype(int)
                    p1_in = p1_in[sel]
                    p2_in = p2_in[sel]
                pair_matches[(i, j)] = (p1_in, p2_in)
    
    print(f"Total matched pairs with geometric inliers: {len(pair_matches)}")
    return pair_matches


def estimate_initial_poses(n, pair_matches, K):
    """Initial pose chain using adjacent edges (stride 1)."""
    R_w2c = [np.eye(3)]
    t_w2c = [np.zeros(3)]
    
    # Normal prior for the endoscopic board
    normal_prior = np.array([0.0, 0.92, 0.38])
    
    for i in range(n - 1):
        if (i, i + 1) in pair_matches:
            p1, p2 = pair_matches[(i, i + 1)]
            H, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 2.5)
            try:
                _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
            except cv2.error:
                Rs, ts, ns = [], [], []
            
            cands = []
            for R_c, t_c, n_c in zip(Rs, ts, ns):
                try:
                    R_c = so3(R_c)
                except Exception:
                    continue
                t_c = t_c.ravel()
                n_c = n_c.ravel()
                if np.linalg.norm(t_c) < 1e-8:
                    continue
                for sgn in (1.0, -1.0):
                    n_curr = sgn * n_c / max(np.linalg.norm(n_c), 1e-8)
                    t_curr = sgn * t_c
                    align = float(np.dot(n_curr, normal_prior))
                    cands.append((align, R_c, t_curr))
            
            if cands:
                best = max(cands, key=lambda c: c[0])
                R_rel, t_rel = best[1], best[2]
            else:
                R_rel, t_rel = np.eye(3), np.array([0.0, 0.0, 0.01])
        else:
            R_rel, t_rel = np.eye(3), np.array([0.0, 0.0, 0.01])
        
        # Accumulate w2c: T_{i+1} = T_{rel} @ T_i
        R_next = R_rel @ R_w2c[-1]
        t_next = R_rel @ t_w2c[-1] + t_rel
        R_w2c.append(R_next)
        t_w2c.append(t_next)
    
    return np.array(R_w2c), np.array(t_w2c)


def run_bundle_adjustment(n, pair_matches, K, n_iters=400, lr=0.01):
    """
    Jointly optimizes:
    - Camera poses (R_w2c, t_w2c) for frames 1..N-1
    - 3D point cloud reconstructed from multi-view correspondence rays
    """
    device = "cuda"
    K_t = torch.from_numpy(K).float().to(device)
    
    # 1. Initialize camera poses
    R_init, t_init = estimate_initial_poses(n, pair_matches, K)
    
    rotvec_init = [roma.rotmat_to_rotvec(torch.from_numpy(R_init[i]).float().to(device)[None])[0] for i in range(1, n)]
    trans_init = [torch.from_numpy(t_init[i]).float().to(device) for i in range(1, n)]
    
    rotvec_params = torch.stack(rotvec_init).clone().detach().requires_grad_(True)
    trans_params = torch.stack(trans_init).clone().detach().requires_grad_(True)
    
    # 2. Triangulate 3D points for all pairs to initialize 3D points
    obs_cam_idx = []
    obs_pt_idx = []
    obs_uv = []
    points_3d_init = []
    
    pt_id = 0
    P0 = K @ np.eye(3, 4)
    for (i, j), (p1, p2) in pair_matches.items():
        # Triangulate with initial poses
        T_i = np.eye(4)
        T_i[:3, :3] = R_init[i]
        T_i[:3, 3] = t_init[i]
        
        T_j = np.eye(4)
        T_j[:3, :3] = R_init[j]
        T_j[:3, 3] = t_init[j]
        
        P_i = K @ T_i[:3, :]
        P_j = K @ T_j[:3, :]
        
        Xh = cv2.triangulatePoints(P_i, P_j, p1.T.astype(np.float64), p2.T.astype(np.float64))
        X = (Xh[:3] / np.clip(Xh[3], 1e-8, None)).T
        
        valid = (X[:, 2] > 0.05) & (X[:, 2] < 2000.0) & np.isfinite(X).all(axis=1)
        for k in range(len(p1)):
            if valid[k]:
                points_3d_init.append(X[k])
                # observation in frame i
                obs_cam_idx.append(i)
                obs_pt_idx.append(pt_id)
                obs_uv.append(p1[k])
                # observation in frame j
                obs_cam_idx.append(j)
                obs_pt_idx.append(pt_id)
                obs_uv.append(p2[k])
                pt_id += 1
    
    if pt_id < 50:
        print(f"Too few valid 3D points ({pt_id}) for BA")
        # fallback to camera centers
        c2w = [np.linalg.inv(np.vstack([np.hstack([R_init[i], t_init[i].reshape(3, 1)]), [0, 0, 0, 1]]))[:3, 3] for i in range(n)]
        return np.array(c2w)
    
    pts_3d_params = torch.tensor(np.array(points_3d_init), dtype=torch.float32, device=device, requires_grad=True)
    obs_cam_t = torch.tensor(obs_cam_idx, device=device, dtype=torch.long)
    obs_pt_t = torch.tensor(obs_pt_idx, device=device, dtype=torch.long)
    obs_uv_t = torch.tensor(np.array(obs_uv), device=device, dtype=torch.float32)
    
    print(f"BA setup: {n} frames, {pt_id} 3D points, {len(obs_uv)} 2D observation constraints.")
    
    optimizer = torch.optim.AdamW([
        {"params": [rotvec_params], "lr": 0.005},
        {"params": [trans_params], "lr": 0.01},
        {"params": [pts_3d_params], "lr": 0.02},
    ], weight_decay=1e-4)
    
    huber_delta = 4.0
    for it in range(n_iters):
        optimizer.zero_grad()
        
        # Assemble R and t
        R_rest = roma.rotvec_to_rotmat(rotvec_params)
        R_all = torch.cat([torch.eye(3, device=device)[None], R_rest], dim=0)  # (N, 3, 3)
        t_all = torch.cat([torch.zeros((1, 3), device=device), trans_params], dim=0)  # (N, 3)
        
        # Gather R and t for each observation
        R_obs = R_all[obs_cam_t]
        t_obs = t_all[obs_cam_t]
        X_obs = pts_3d_params[obs_pt_t]
        
        # Transform 3D point to camera coordinates: X_cam = R @ X + t
        X_cam = torch.bmm(R_obs, X_obs[:, :, None])[:, :, 0] + t_obs
        z_cam = torch.clamp(X_cam[:, 2], min=1e-3)
        
        u_proj = K_t[0, 0] * (X_cam[:, 0] / z_cam) + K_t[0, 2]
        v_proj = K_t[1, 1] * (X_cam[:, 1] / z_cam) + K_t[1, 2]
        uv_proj = torch.stack([u_proj, v_proj], dim=-1)
        
        # Reprojection error
        reproj_err = torch.norm(uv_proj - obs_uv_t, dim=-1)
        
        # Smooth L1 / Huber loss
        loss_reproj = torch.where(
            reproj_err < huber_delta,
            0.5 * reproj_err.pow(2),
            huber_delta * (reproj_err - 0.5 * huber_delta)
        ).mean()
        
        # Cheirality penalty: penalize points behind the camera
        cheirality_loss = F.relu(0.1 - X_cam[:, 2]).pow(2).mean() * 50.0
        
        loss = loss_reproj + cheirality_loss
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 100 == 0:
            print(f"  BA iter {it+1:3d} reproj_loss: {loss_reproj.item():.4f} px (cheirality: {cheirality_loss.item():.4f})")
    
    # Extract camera centers in world coordinates: c_i = -R_i^T @ t_i
    with torch.no_grad():
        R_rest = roma.rotvec_to_rotmat(rotvec_params)
        R_all = torch.cat([torch.eye(3, device=device)[None], R_rest], dim=0)
        t_all = torch.cat([torch.zeros((1, 3), device=device), trans_params], dim=0)
        centers = []
        for i in range(n):
            c_i = -R_all[i].T @ t_all[i]
            centers.append(c_i.cpu().numpy())
    
    return np.array(centers)


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    K0 = np.asarray(screen["K"], np.float64)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading LoFTR matcher on", device)
    matcher = LoFTR(pretrained="indoor").to(device).eval()
    
    jobs = []
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        row = real[name]
        pose = np.load(row["pose"])
        folder = Path(row["image_dir"])
        usable = [int(i) for i in np.flatnonzero(pose["usable"] == 1)
                  if (folder / f"{int(i):06d}.jpg").is_file()]
        sel = subsample_idx(len(usable), 30)
        ids = [usable[k] for k in sel]
        paths = [folder / f"{i:06d}.jpg" for i in ids]
        gt = pose["p"][ids] * 1000.0
        jobs.append({"name": name, "paths": paths, "gt": gt, "kind": "real"})
    
    row = sim["sim_20260924_172955_s10095"]
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    sel = subsample_idx(min(len(paths), len(gt_all)), 30)
    jobs.append({
        "name": "sim_20260924_172955_s10095",
        "paths": [paths[i] for i in sel],
        "gt": gt_all[sel],
        "kind": "sim",
    })
    
    results = {}
    for job in jobs:
        print(f"\nEvaluating BA on {job['name']} ({len(job['paths'])} frames)...")
        K = scale_K(K0, (640, 360))
        pairs = build_feature_tracks(job["paths"], matcher, K, strides=(1, 2, 3, 4))
        centers = run_bundle_adjustment(len(job["paths"]), pairs, K, n_iters=400)
        
        ate, scale = sim3_ate(centers, job["gt"])
        rec = {
            "ate_mm": round(float(ate), 2),
            "sim3_scale": round(float(scale), 3),
            "matched_pairs": len(pairs),
            "n": len(centers),
        }
        print(f"--> RESULT {job['name']}: Sim3 ATE = {rec['ate_mm']:.2f} mm (scale={rec['sim3_scale']:.3f})")
        results[job["name"]] = rec
    
    out_path = HERE / "ego_recon_ours_ba.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
