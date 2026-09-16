#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ORB-SfM baseline on the 4 newly captured sequences (CPU, py3d env).
Outputs per session: orb_poses/c2w_XXXX.txt + orb_sfm.ply, under
E:\\MIS_TMI_Re_3D\\benchmark_new\\orb_sfm\\<session>\\"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import time, numpy as np, cv2
cv2.ocl.setUseOpenCL(False)

NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
OUT_ROOT = r"E:\MIS_TMI_Re_3D\benchmark_new\orb_sfm"
SESSIONS = ["pnp_seq_20260911_011115", "vio_seq_20260911_010902",
            "vio_seq_20260911_011013", "vio_seq_20260912_104835"]
# real calibrated intrinsics at 1280x720 (undistorted frames)
K = np.array([[1210.6664, 0, 726.3980],
              [0, 1194.5669, 391.9404],
              [0, 0, 1.0]], dtype=np.float64)


def run_session(sess):
    img_dir = os.path.join(NEW_ROOT, sess, "images")
    out_dir = os.path.join(OUT_ROOT, sess)
    os.makedirs(out_dir, exist_ok=True)
    fns = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    print(f"\n[{sess}] {len(fns)} images", flush=True)

    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    all_poses = {0: np.eye(4)}
    pts_all, cols_all = [], []
    prev_kp = prev_des = None
    t0 = time.time()
    for i, fn in enumerate(fns):
        img = cv2.imread(os.path.join(img_dir, fn))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if i == 0 or prev_des is None or des is None:
            prev_kp, prev_des = kp, des
            continue
        matches = bf.knnMatch(prev_des, des, k=2)
        good = [m for m in matches if len(m) >= 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            prev_kp, prev_des = kp, des
            continue
        pts_prev = np.float32([prev_kp[m[0].queryIdx].pt for m in good])
        pts_curr = np.float32([kp[m[0].trainIdx].pt for m in good])
        E, mask_e = cv2.findEssentialMat(pts_prev, pts_curr, K, cv2.RANSAC, 0.999, 1.0)
        # note: knnMatch(k=2) returns 2-tuples; best match is m[0]
        if E is None or E.shape != (3, 3):
            prev_kp, prev_des = kp, des
            continue
        _, R_rel, t_rel, mask_r = cv2.recoverPose(E, pts_prev, pts_curr, K)
        T_prev = all_poses.get(i - 1, np.eye(4))
        T_rel = np.eye(4); T_rel[:3, :3] = R_rel; T_rel[:3, 3] = t_rel.flatten()
        T_curr = T_prev @ T_rel
        all_poses[i] = T_curr
        # triangulate
        R_p, t_p = T_prev[:3, :3], T_prev[:3, 3].reshape(3, 1)
        R_c, t_c = T_curr[:3, :3], T_curr[:3, 3].reshape(3, 1)
        P_prev = K @ np.hstack([R_p, t_p]); P_curr = K @ np.hstack([R_c, t_c])
        inlier = mask_r.ravel() > 0
        pts1 = pts_prev[inlier].T; pts2 = pts_curr[inlier].T
        if pts1.shape[1] >= 5:
            pts4d = cv2.triangulatePoints(P_prev, P_curr, pts1, pts2)
            pts3d = (pts4d[:3] / pts4d[3:4]).T
            valid = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 5000) & np.isfinite(pts3d).all(axis=1)
            pts3d = pts3d[valid]
            if len(pts3d) > 0:
                p2 = pts2.T[valid].astype(int)
                p2[:, 0] = np.clip(p2[:, 0], 0, img.shape[1] - 1)
                p2[:, 1] = np.clip(p2[:, 1], 0, img.shape[0] - 1)
                cols_all.append(img[p2[:, 1], p2[:, 0], ::-1].astype(np.uint8))
                pts_all.append(pts3d.astype(np.float32))
        prev_kp, prev_des = kp, des
        if i % 50 == 0:
            print(f"  [{i}] poses={len(all_poses)}", flush=True)

    dt = time.time() - t0
    if pts_all:
        pts = np.vstack(pts_all); cols = np.vstack(cols_all)
        if len(pts) > 50000:
            vs = 0.5
            vidx = np.floor(pts / vs).astype(np.int64)
            flat = vidx[:, 0] * 100000000 + vidx[:, 1] * 10000 + vidx[:, 2]
            _, first = np.unique(flat, return_index=True)
            pts = pts[first]; cols = cols[first]
        ply = os.path.join(out_dir, "orb_sfm.ply")
        with open(ply, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for j in range(len(pts)):
                f.write(f"{pts[j,0]:.6f} {pts[j,1]:.6f} {pts[j,2]:.6f} "
                        f"{int(cols[j,0])} {int(cols[j,1])} {int(cols[j,2])}\n")
    pose_dir = os.path.join(out_dir, "orb_poses")
    os.makedirs(pose_dir, exist_ok=True)
    for i, T in all_poses.items():
        np.savetxt(os.path.join(pose_dir, f"c2w_{i:04d}.txt"), T, fmt="%.8f")
    print(f"[{sess}] poses={len(all_poses)} pts={sum(len(p) for p in pts_all)} "
          f"time={dt:.0f}s", flush=True)


for s in SESSIONS:
    run_session(s)
print("\nALL ORB-SfM DONE")
