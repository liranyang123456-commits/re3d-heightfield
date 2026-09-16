#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COLMAP SfM baseline on the 4 newly captured sequences (base env, pycolmap).
Sequential matching for video. Outputs per session under
E:\\MIS_TMI_Re_3D\\benchmark_new\\colmap\\<session>\\
  estimated_poses_txt/pose_XXXXX.txt (c2w, original frame index)
  colmap_sfm.ply
"""
import os, sys, time, re, shutil
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '4')
import pycolmap

NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
OUT_ROOT = r"E:\MIS_TMI_Re_3D\benchmark_new\colmap"
SESSIONS = ["pnp_seq_20260911_011115", "vio_seq_20260911_010902",
            "vio_seq_20260911_011013", "vio_seq_20260912_104835"]


def run_session(sess):
    img_dir = os.path.join(NEW_ROOT, sess, "images")
    out_dir = os.path.join(OUT_ROOT, sess)
    work = os.path.join(out_dir, "work")
    colmap_imgs = os.path.join(work, "images")
    db = os.path.join(work, "database.db")
    os.makedirs(colmap_imgs, exist_ok=True)
    pose_dir = os.path.join(out_dir, "estimated_poses_txt")
    os.makedirs(pose_dir, exist_ok=True)

    fns = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    for f in fns:
        dst = os.path.join(colmap_imgs, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(img_dir, f), dst)
    print(f"\n[{sess}] {len(fns)} images", flush=True)

    if os.path.exists(db):
        os.remove(db)
    t0 = time.time()
    print("  extract features...", flush=True)
    pycolmap.extract_features(db, colmap_imgs)
    print("  sequential match...", flush=True)
    try:
        pycolmap.match_sequential(db)
    except AttributeError:
        print("  (match_sequential unavailable, exhaustive)", flush=True)
        pycolmap.match_exhaustive(db)
    print("  incremental mapping...", flush=True)
    recs = pycolmap.incremental_mapping(db, colmap_imgs, work)
    dt = time.time() - t0
    if not recs:
        print(f"[{sess}] no reconstruction", flush=True)
        return
    recon = max(recs.values(), key=lambda r: len(r.points3D))
    print(f"  points3D={len(recon.points3D)} images={len(recon.images)}", flush=True)

    # points
    pts = np.array([p.xyz for p in recon.points3D.values()]) if recon.points3D else np.zeros((0, 3))
    if len(pts):
        with open(os.path.join(out_dir, "colmap_sfm.ply"), 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p in pts:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128\n")

    # poses: img_XXXXX.jpg -> frame index XXXXX (0-based, no offset)
    n_poses = 0
    for iid, image in recon.images.items():
        w2c = image.cam_from_world()
        R = w2c.rotation.matrix(); t = w2c.translation
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ t
        m = re.search(r'(\d+)', image.name)
        idx = int(m.group(1)) if m else 0
        np.savetxt(os.path.join(pose_dir, f"pose_{idx:05d}.txt"), c2w, fmt="%.8f")
        n_poses += 1
    print(f"[{sess}] poses={n_poses} time={dt:.0f}s", flush=True)


for s in SESSIONS:
    run_session(s)
print("\nALL COLMAP DONE")
