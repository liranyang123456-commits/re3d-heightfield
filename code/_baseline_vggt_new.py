#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VGGT baseline on the 4 newly captured sequences (GPU, vggt env).
Single-pass inference on a stride-sampled frame subset (<=60 frames/session)
so poses and world_points share one consistent coordinate frame.
Outputs per session under E:\\MIS_TMI_Re_3D\\benchmark_new\\vggt\\<session>\\
  estimated_poses_txt/pose_XXXXX.txt  (c2w, keyed by ORIGINAL frame index)
  vggt_pointcloud.ply
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys, time
import numpy as np

sys.path.insert(0, r"D:\vggt-main\vggt-main")
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
OUT_ROOT = r"E:\MIS_TMI_Re_3D\benchmark_new\vggt"
SESSIONS = ["pnp_seq_20260911_011115", "vio_seq_20260911_010902",
            "vio_seq_20260911_011013", "vio_seq_20260912_104835"]
MAX_FRAMES = 60


def write_ply(path, pts, cols):
    n = len(pts)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for j in range(n):
            f.write(f"{pts[j,0]:.6f} {pts[j,1]:.6f} {pts[j,2]:.6f} "
                    f"{int(cols[j,0])} {int(cols[j,1])} {int(cols[j,2])}\n")


def run_session(sess, model, device):
    img_dir = os.path.join(NEW_ROOT, sess, "images")
    out_dir = os.path.join(OUT_ROOT, sess)
    pose_dir = os.path.join(out_dir, "estimated_poses_txt")
    os.makedirs(pose_dir, exist_ok=True)

    all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    stride = max(1, len(all_imgs) // MAX_FRAMES)
    sel = all_imgs[::stride][:MAX_FRAMES]
    # original frame indices from img_XXXXX.jpg
    orig_idx = [int(f.replace("img_", "").replace(".jpg", "")) for f in sel]
    print(f"\n[{sess}] {len(all_imgs)} frames -> {len(sel)} sampled (stride {stride})", flush=True)

    paths = [os.path.join(img_dir, f) for f in sel]
    images = load_and_preprocess_images(paths).to(device)
    t0 = time.time()
    with torch.no_grad():
        preds = model(images)
    dt = time.time() - t0

    pose_enc = preds["pose_enc"]
    h, w = images.shape[-2], images.shape[-1]
    extri, intri = pose_encoding_to_extri_intri(pose_enc, image_size_hw=(h, w))
    extri = extri.cpu().numpy()
    world_pts = preds["world_points"].cpu().numpy()   # (1, N, H, W, 3)
    depth = preds["depth"].squeeze(-1).cpu().numpy()  # (1, N, H, W)

    # poses (c2w), keyed by original frame index
    for k in range(len(sel)):
        w2c = np.eye(4); w2c[:3, :] = extri[0, k]
        c2w = np.linalg.inv(w2c)
        np.savetxt(os.path.join(pose_dir, f"pose_{orig_idx[k]:05d}.txt"), c2w, fmt="%.8f")

    # fused cloud from world_points (already in one frame)
    pts_list, cols_list = [], []
    import cv2
    for k in range(len(sel)):
        wp = world_pts[0, k]           # (H, W, 3)
        dm = depth[0, k]               # (H, W)
        valid = dm > 0.01
        pts = wp[valid]
        if len(pts) == 0:
            continue
        img = cv2.imread(paths[k])
        ys, xs = np.where(valid)
        H, W = dm.shape
        xs_o = np.clip((xs * img.shape[1] / W).astype(int), 0, img.shape[1] - 1)
        ys_o = np.clip((ys * img.shape[0] / H).astype(int), 0, img.shape[0] - 1)
        cols = img[ys_o, xs_o, ::-1]
        pts_list.append(pts.astype(np.float32))
        cols_list.append(cols.astype(np.uint8))
    if pts_list:
        P = np.vstack(pts_list); C = np.vstack(cols_list)
        if len(P) > 50000:
            idx = np.random.RandomState(0).choice(len(P), 50000, replace=False)
            P = P[idx]; C = C[idx]
        write_ply(os.path.join(out_dir, "vggt_pointcloud.ply"), P, C)
    print(f"[{sess}] poses={len(sel)} cloud_pts={sum(len(p) for p in pts_list)} "
          f"infer={dt:.1f}s", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading VGGT...", flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    for s in SESSIONS:
        run_session(s, model, device)
    print("\nALL VGGT DONE")


if __name__ == "__main__":
    main()
