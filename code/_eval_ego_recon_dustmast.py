#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DUSt3R or MASt3R on the same 30-frame subsets. Use the vggt conda env."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\mast3r\dust3r")
sys.path.insert(0, r"D:\mast3r")

import torch
_torch_load = torch.load
def _load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)
torch.load = _load
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

SCREEN = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_screen.json")
MAX_FRAMES = 30
REAL_NAMES = ("traj_20260923_023422", "traj_20260924_144019")
CKPT = {
    "dust3r": Path(r"D:\dust3r_checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"),
    "mast3r": Path(r"D:\mast3r_checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
}


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
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))


def real_paths(row):
    pose = np.load(row["pose"])
    usable = np.flatnonzero(pose["usable"] == 1)
    kept = [int(i) for i in usable if (Path(row["image_dir"]) / f"{int(i):06d}.jpg").is_file()]
    sel = subsample_idx(len(kept), MAX_FRAMES)
    ids = [kept[i] for i in sel]
    paths = [str(Path(row["image_dir"]) / f"{i:06d}.jpg") for i in ids]
    return paths, pose["p"][ids] * 1000.0


def sim_paths(row):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, MAX_FRAMES)
    return [str(paths[i]) for i in sel], gt_all[sel]


def reconstruct(paths, model, device):
    imgs = load_images(paths, size=512, verbose=False, patch_size=model.patch_size)
    pairs = make_pairs(imgs, scene_graph="swin-2", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
    scene.compute_global_alignment(init="mst", niter=60, schedule="linear", lr=0.01)
    poses = scene.get_im_poses().detach().cpu().numpy()
    return poses[:, :3, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("dust3r", "mast3r"), required=True)
    args = ap.parse_args()
    if args.model == "dust3r":
        from dust3r.model import load_model
    else:
        from mast3r.model import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading", args.model, "on", device, flush=True)
    model = load_model(str(CKPT[args.model]), device)
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    jobs = []
    for name in REAL_NAMES:
        if name in real:
            jobs.append((name, "real", real[name].get("role"), *real_paths(real[name])))
    if screen["sim_with_images"]:
        row = screen["sim_with_images"][0]
        jobs.append((row["name"], "sim", None, *sim_paths(row)))
    out_path = Path(rf"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_{args.model}.json")
    rows = []
    for name, kind, role, paths, gt in jobs:
        print(args.model, name, "n", len(paths), flush=True)
        est = reconstruct(paths, model, device)
        ate, scale = sim3_ate(est, gt)
        rec = {"name": name, "kind": kind, "role": role, "n": len(paths),
               "ate_mm": round(ate, 2), "sim3_scale": round(scale, 3)}
        rows.append(rec)
        print(rec, flush=True)
        out_path.write_text(json.dumps({"method": args.model, "max_frames": MAX_FRAMES, "results": rows}, indent=2), encoding="utf-8")
        if device == "cuda":
            torch.cuda.empty_cache()
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
