#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MUSt3R camera-center Sim3 ATE on the same 30-frame EGO clips."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from must3r.model import get_dtype, get_pointmaps_activation, load_model
from must3r.engine.inference import inference_multi_ar, postprocess
from must3r.tools.image import get_resize_function
import must3r.tools.path_to_dust3r  # noqa: F401
from dust3r.datasets import ImgNorm

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"
RESULT = HERE / "ego_recon_must3r.json"
COMPARISON = HERE / "ego_recon_comparison.json"
CKPT = Path(r"E:\MIS_TMI_Re_3D\benchmark_results\must3r\MUSt3R_224_cvpr.pth")
MAX_FRAMES = 30
IMAGE_SIZE = 224
NUM_MEM = 10


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


def frames_real(row):
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
    return paths, pose["p"][ids] * 1000.0


def frames_sim(row):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, MAX_FRAMES)
    return [paths[i] for i in sel], gt_all[sel]


def load_views(paths, patch_size):
    views = []
    for path in paths:
        rgb = Image.open(path).convert("RGB")
        rgb.load()
        w, h = rgb.size
        resize_func, _, _ = get_resize_function(IMAGE_SIZE, patch_size, h, w)
        tensor = resize_func(ImgNorm(rgb))
        views.append({
            "img": tensor,
            "true_shape": np.int32([tensor.shape[-2], tensor.shape[-1]]),
        })
    return views


def centers_from_images(paths, encoder, decoder, device):
    views = load_views(paths, encoder.patch_size)
    n = len(views)
    keyframes = np.linspace(0, n - 1, min(NUM_MEM, n), dtype=int).tolist()
    rest = sorted(set(range(n)).difference(keyframes))
    order = keyframes + rest
    views_re = [views[i] for i in order]
    imgs = [b["img"].to(device) for b in views_re]
    true_shape = [torch.from_numpy(b["true_shape"]).to(device) for b in views_re]
    img_ids = [torch.tensor(v) for v in order]
    mem_batches = [min(NUM_MEM, n)]
    activation = get_pointmaps_activation(decoder, verbose=False)

    def post_fn(x):
        return postprocess(x, pointmaps_activation=activation, compute_cam=True)

    dtype = get_dtype("fp16")
    with torch.autocast("cuda", dtype=dtype):
        _, x_out = inference_multi_ar(
            encoder, decoder, imgs, img_ids, true_shape, mem_batches,
            max_bs=1, verbose=False, to_render=None,
            device=device, preserve_gpu_mem=True,
            post_process_function=post_fn,
        )
    centers = [None] * n
    for i, orig in enumerate(order):
        c2w = x_out[i]["c2w"].detach().cpu().numpy()
        centers[orig] = c2w[:3, 3]
    return np.asarray(centers, dtype=np.float64)


def merge(rows):
    table = json.loads(COMPARISON.read_text(encoding="utf-8"))
    key_of = {
        "traj_20260923_023422": "test_traj_20260923_023422",
        "traj_20260924_144019": "val_traj_20260924_144019",
        "sim_20260924_172955_s10095": "sim_20260924_172955_s10095",
    }
    for rec in rows:
        key = key_of[rec["name"]]
        table.setdefault(key, {})
        table[key]["MUSt3R"] = {
            "ate_mm": rec["ate_mm"],
            "sim3_scale": rec["sim3_scale"],
            "n": rec["n"],
        }
    table["not_run"] = [
        x for x in table.get("not_run", [])
        if not x.startswith("MUSt3R")
    ]
    COMPARISON.write_text(json.dumps(table, indent=2), encoding="utf-8")


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading MUSt3R on", device, flush=True)
    encoder, decoder = load_model(str(CKPT), device=device, img_size=IMAGE_SIZE, verbose=False)
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    jobs = []
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        paths, gt = frames_real(real[name])
        jobs.append((name, "real", real[name].get("role"), paths, gt))
    paths, gt = frames_sim(sim["sim_20260924_172955_s10095"])
    jobs.append(("sim_20260924_172955_s10095", "sim", None, paths, gt))
    rows = []
    for name, kind, role, paths, gt in jobs:
        print("MUSt3R", name, "n", len(paths), flush=True)
        est = centers_from_images(paths, encoder, decoder, device)
        ate, scale = sim3_ate(est, gt)
        rec = {
            "name": name, "kind": kind, "n": len(paths),
            "ate_mm": round(ate, 2), "sim3_scale": round(scale, 3),
        }
        if role:
            rec["role"] = role
        rows.append(rec)
        print(rec, flush=True)
        RESULT.write_text(json.dumps({
            "method": "MUSt3R", "max_frames": MAX_FRAMES,
            "image_size": IMAGE_SIZE, "results": rows,
        }, indent=2), encoding="utf-8")
    merge(rows)
    print("wrote", RESULT, flush=True)


if __name__ == "__main__":
    main()
