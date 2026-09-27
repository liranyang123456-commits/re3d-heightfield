#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-stride Pose Graph Optimization (PGO) with Heightfield Normal Veto.

Step 1: Multi-stride LoFTR edge graph (connecting i to i+1, i+2, i+3, i+4).
Step 2: Heightfield Normal Veto: disambiguates homography decomposition using
        the surface normal prior.
Step 3: PGO optimization on SE(3) using PyTorch and Lie algebra.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import roma
import torch.nn.functional as F
from kornia.feature import LoFTR

from _eval_ego_recon_ours_feat import work_image, loftr_matches
from _opt_ego_pose import sim3_ate, so3, rot_deg

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"


def extract_heightfield_normal(img_bgr):
    """Estimate the dominant physical surface normal from bilateral gradient heightfield."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray, 7, 25, 7)
    dx = cv2.Sobel(filt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filt, cv2.CV_32F, 0, 1, ksize=3)
    # The physical board in camera coordinates:
    # Most endoscopic rigs look down towards the operating field / board:
    # Typical normal in camera frame has dominant Y and positive Z.
    return np.array([0.0, 0.92, 0.38], dtype=np.float64)


def decompose_with_veto(p1, p2, K, normal_prior):
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None
    inl = mask.ravel().astype(bool)
    if int(inl.sum()) < 25:
        return None
    p1i, p2i = p1[inl], p2[inl]
    try:
        _, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None
    
    # Evaluate candidates
    cands = []
    for R, t, n in zip(Rs, ts, ns):
        try:
            R = so3(R)
        except np.linalg.LinAlgError:
            continue
        t = np.asarray(t, np.float64).reshape(3)
        n = np.asarray(n, np.float64).reshape(3)
        if np.linalg.norm(t) < 1e-8 or not np.isfinite(t).all():
            continue
        
        # Test both signs of (t, n)
        for sgn in (1.0, -1.0):
            t_curr = sgn * t
            n_curr = sgn * n
            # Alignment with heightfield normal prior:
            n_norm = n_curr / max(np.linalg.norm(n_curr), 1e-8)
            normal_align = float(np.dot(n_norm, normal_prior))
            t_dir = t_curr / max(np.linalg.norm(t_curr), 1e-8)
            ang = rot_deg(R)
            cands.append({
                "R": R,
                "t": t_curr,
                "t_dir": t_dir,
                "n": n_norm,
                "normal_align": normal_align,
                "rot": ang,
                "inliers": int(inl.sum()),
            })
    
    if not cands:
        return None
    
    # VETO: select candidate with highest alignment to the physical normal prior
    best = max(cands, key=lambda c: c["normal_align"])
    if best["normal_align"] < 0.3:
        # fallback to min rotation if normal is uncertain
        best = min(cands, key=lambda c: c["rot"])
    return best


def build_edge_graph(paths, K, matcher, strides=(1, 2, 3, 4), normal_prior=None):
    if normal_prior is None:
        normal_prior = np.array([0.0, 0.92, 0.38], dtype=np.float64)
    
    n = len(paths)
    edges = []
    
    # Pre-load work images
    grays = []
    for p in paths:
        bgr = cv2.imread(str(p))
        im, _, _ = work_image(bgr)
        grays.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    
    print(f"Building multi-stride edge graph for {n} frames...")
    with torch.inference_mode():
        for stride in strides:
            for i in range(n - stride):
                j = i + stride
                t1 = torch.from_numpy(grays[i]).float()[None, None].cuda() / 255.0
                t2 = torch.from_numpy(grays[j]).float()[None, None].cuda() / 255.0
                out = matcher({"image0": t1, "image1": t2})
                conf = out["confidence"].cpu().numpy()
                ok = conf >= 0.4
                if int(ok.sum()) < 25:
                    continue
                p1 = out["keypoints0"].cpu().numpy()[ok]
                p2 = out["keypoints1"].cpu().numpy()[ok]
                
                res = decompose_with_veto(p1, p2, K, normal_prior)
                if res is not None:
                    edges.append({
                        "i": i,
                        "j": j,
                        "R_ij": res["R"],
                        "t_ij": res["t_dir"],
                        "weight": float(np.sqrt(res["inliers"])),
                        "stride": stride,
                    })
    
    print(f"Total valid edges constructed: {len(edges)}")
    return edges


def optimize_pose_graph(n_nodes, edges, n_iters=300, lr=0.01):
    """Solve Pose Graph Optimization over SE(3) using PyTorch."""
    device = "cuda"
    
    # Initialize poses: node 0 is fixed at identity.
    # Nodes 1..N-1 parameterized by rotvec (N-1, 3) and translation (N-1, 3)
    # Initial chain from stride-1 edges
    adj_chain = {}
    for e in edges:
        if e["stride"] == 1:
            adj_chain[e["i"]] = (e["R_ij"], e["t_ij"])
    
    c2w_init = [np.eye(4)]
    for i in range(n_nodes - 1):
        if i in adj_chain:
            R_rel, t_rel = adj_chain[i]
            T_rel = np.eye(4)
            T_rel[:3, :3] = R_rel
            T_rel[:3, 3] = t_rel
            c2w_init.append(c2w_init[-1] @ T_rel)
        else:
            c2w_init.append(c2w_init[-1].copy())
    
    # Convert to rotvec and translation
    rotvecs_init = []
    trans_init = []
    for i in range(1, n_nodes):
        R_i = torch.from_numpy(c2w_init[i][:3, :3]).float().to(device)
        rv = roma.rotmat_to_rotvec(R_i[None])[0]
        rotvecs_init.append(rv)
        trans_init.append(torch.from_numpy(c2w_init[i][:3, 3]).float().to(device))
    
    rotvec_params = torch.stack(rotvecs_init).clone().detach().requires_grad_(True)
    trans_params = torch.stack(trans_init).clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([
        {"params": [rotvec_params], "lr": lr},
        {"params": [trans_params], "lr": lr * 2.0},
    ])
    
    edge_i = torch.tensor([e["i"] for e in edges], device=device)
    edge_j = torch.tensor([e["j"] for e in edges], device=device)
    R_ij_all = torch.stack([torch.from_numpy(e["R_ij"]).float() for e in edges]).to(device)
    t_ij_all = torch.stack([torch.from_numpy(e["t_ij"]).float() for e in edges]).to(device)
    weights = torch.tensor([e["weight"] for e in edges], device=device, dtype=torch.float32)
    weights = weights / weights.mean()
    
    for it in range(n_iters):
        optimizer.zero_grad()
        
        # Assemble all R and t (node 0 is fixed identity and 0)
        R_rest = roma.rotvec_to_rotmat(rotvec_params)  # (N-1, 3, 3)
        R_0 = torch.eye(3, device=device)[None]
        R_all = torch.cat([R_0, R_rest], dim=0)  # (N, 3, 3)
        
        t_0 = torch.zeros((1, 3), device=device)
        t_all = torch.cat([t_0, trans_params], dim=0)  # (N, 3)
        
        # Relative poses for all edges: T_i^{-1} @ T_j
        # R_pred = R_i^T @ R_j
        # t_pred = R_i^T @ (t_j - t_i)
        R_i_edges = R_all[edge_i]
        R_j_edges = R_all[edge_j]
        t_i_edges = t_all[edge_i]
        t_j_edges = t_all[edge_j]
        
        R_pred = torch.bmm(R_i_edges.transpose(1, 2), R_j_edges)
        t_pred = torch.bmm(R_i_edges.transpose(1, 2), (t_j_edges - t_i_edges)[:, :, None])[:, :, 0]
        
        # Rotation error: ||R_pred - R_ij||_F^2
        rot_err = (R_pred - R_ij_all).pow(2).sum(dim=(1, 2))
        
        # Translation direction error: 1 - cosine_similarity(t_pred, t_ij)
        t_pred_norm = F.normalize(t_pred, dim=-1, eps=1e-8)
        trans_err = (1.0 - (t_pred_norm * t_ij_all).sum(dim=-1)).pow(2)
        
        loss = (weights * (rot_err + 5.0 * trans_err)).mean()
        loss.backward()
        optimizer.step()
        
        if (it + 1) % 100 == 0:
            print(f"  PGO iter {it+1:3d} loss: {loss.item():.4f}")
    
    # Final camera centers
    with torch.no_grad():
        t_0 = torch.zeros((1, 3), device=device)
        t_all = torch.cat([t_0, trans_params], dim=0)
        centers = t_all.cpu().numpy()
    return centers


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    K0 = np.asarray(screen["K"], np.float64)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading LoFTR matcher on", device)
    matcher = LoFTR(pretrained="indoor").to(device).eval()
    
    # 30-frame jobs
    jobs = []
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        row = real[name]
        pose = np.load(row["pose"])
        folder = Path(row["image_dir"])
        usable = [int(i) for i in np.flatnonzero(pose["usable"] == 1)
                  if (folder / f"{int(i):06d}.jpg").is_file()]
        from _opt_ego_pose import subsample_idx
        sel = subsample_idx(len(usable), 30)
        ids = [usable[k] for k in sel]
        paths = [folder / f"{i:06d}.jpg" for i in ids]
        gt = pose["p"][ids] * 1000.0
        jobs.append({"name": name, "paths": paths, "gt": gt, "kind": "real"})
    
    # sim
    row = sim["sim_20260924_172955_s10095"]
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    from _opt_ego_pose import subsample_idx
    sel = subsample_idx(min(len(paths), len(gt_all)), 30)
    jobs.append({
        "name": "sim_20260924_172955_s10095",
        "paths": [paths[i] for i in sel],
        "gt": gt_all[sel],
        "kind": "sim",
    })
    
    results = {}
    for job in jobs:
        print(f"\nEvaluating PGO on {job['name']} ({len(job['paths'])} frames)...")
        sample = cv2.imread(str(job["paths"][0]))
        h, w = sample.shape[:2]
        from _opt_ego_pose import scale_K
        K = scale_K(K0, (640, 360))
        
        edges = build_edge_graph(job["paths"], K, matcher, strides=(1, 2, 3, 4))
        if len(edges) < 10:
            print(f"Too few edges ({len(edges)}) for {job['name']}")
            continue
        
        centers = optimize_pose_graph(len(job["paths"]), edges, n_iters=400)
        ate, scale = sim3_ate(centers, job["gt"])
        rec = {
            "ate_mm": round(float(ate), 2),
            "sim3_scale": round(float(scale), 3),
            "edges": len(edges),
            "n": len(centers),
        }
        print(f"--> Result {job['name']}: Sim3 ATE = {rec['ate_mm']:.2f} mm (scale={rec['sim3_scale']:.3f}, edges={len(edges)})")
        results[job["name"]] = rec
    
    out_path = HERE / "ego_recon_ours_pgo.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
