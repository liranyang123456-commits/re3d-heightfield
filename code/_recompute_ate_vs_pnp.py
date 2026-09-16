#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute chessboard pose metrics (ATE / Sim3 scale / rotation / RPE) for ALL
methods against the INDEPENDENT solvePnP chessboard reference (GT-PnP),
instead of the heightfield-ICP trajectory used in evaluation.json.

Protocol identical to _experiment_three_pose_sources.py:
  - GT-PnP: solvePnP per frame on the first 50 images (photo_001..050),
    8x5 inner corners, 5.0 mm squares, K = diag estimate at 640x358,
    normalised to the frame-0 camera system.
  - Sim3 alignment (same Umeyama variant as _eval_all.py / ate_sim3).
  - Sanity check: Ours must reproduce ATE ~= 3.8864, scale ~= 1.107.

Output: re3d_cmpb_results/results/evaluation_vs_pnp.json
"""
from __future__ import annotations
import os, glob, json
import numpy as np
import cv2

IMG_DIR = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2"
BENCH = r"E:\MIS_TMI_Re_3D\benchmark_results"
OUT_JSON = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\evaluation_vs_pnp.json"

TARGET = (640, 358)
N_FRAMES = 50
CB_SIZE = (8, 5)      # inner corners (cols, rows)
CB_SQUARE = 5.0       # mm


def make_K(W, H):
    fx = float(np.sqrt(W * W + H * H))
    return np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]])


def make_gt_pnp(n, K):
    objp = np.zeros((CB_SIZE[0] * CB_SIZE[1], 3))
    objp[:, :2] = np.mgrid[0:CB_SIZE[0], 0:CB_SIZE[1]].T.reshape(-1, 2) * CB_SQUARE
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "photo_*.jpg")))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    c2w = {}
    for i in range(min(n, len(imgs))):
        g = cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(g, CB_SIZE, flags)
        if not ret:
            continue
        corners2 = cv2.cornerSubPix(g, corners, (5, 5), (-1, -1), crit)
        _, rvec, tvec = cv2.solvePnP(objp.astype(np.float32), corners2, K, None)
        R, _ = cv2.Rodrigues(rvec)
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = tvec.ravel()
        c2w[i] = np.linalg.inv(w2c)
    if 0 in c2w:
        T0i = np.linalg.inv(c2w[0])
        c2w = {i: T0i @ c2w[i] for i in c2w}
    print(f"[GT-PnP] {len(c2w)}/{n} frames")
    return c2w


def load_poses(pose_dir, prefix):
    poses = {}
    for fn in sorted(os.listdir(pose_dir)):
        if fn.endswith(".txt"):
            idx = int(fn.replace(prefix, "").replace(".txt", ""))
            poses[idx] = np.loadtxt(os.path.join(pose_dir, fn))
    return poses


def sim3_metrics(est, gt):
    common = sorted(set(est) & set(gt))
    if len(common) < 3:
        return None
    A = np.array([est[i][:3, 3] for i in common])
    B = np.array([gt[i][:3, 3] for i in common])
    ac, bc = A.mean(0), B.mean(0)
    ad, bd = A - ac, B - bc
    sa = np.sqrt((ad ** 2).sum() / len(A))
    sb = np.sqrt((bd ** 2).sum() / len(B))
    s = sb / max(sa, 1e-8)
    U, _, Vt = np.linalg.svd(ad.T @ bd)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Aa = s * (R @ ad.T).T
    ate = float(np.sqrt(((Aa - bd) ** 2).sum(1).mean()))

    # Rotation error AFTER applying the Sim3 rotation to the estimated
    # orientations (standard ATE-rotation counterpart); without this the
    # comparison is only valid when both trajectories share a world frame.
    rot_errs = []
    for i in common[:20]:                      # same subsample as _eval_all.py
        R_d = gt[i][:3, :3].T @ (R @ est[i][:3, :3])
        tr = np.clip(np.trace(R_d), -1, 3)
        rot_errs.append(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))

    rpe_t, rpe_r = [], []
    for k in range(len(common) - 1):
        i0, i1 = common[k], common[k + 1]
        Tg = np.linalg.inv(gt[i0]) @ gt[i1]
        Te = np.linalg.inv(est[i0]) @ est[i1]
        Terr = np.linalg.inv(Tg) @ Te
        rpe_t.append(np.linalg.norm(Terr[:3, 3]))
        tr = np.clip(np.trace(Terr[:3, :3]), -1, 3)
        rpe_r.append(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))

    return {
        "ate": ate,
        "sim3_scale": float(s),
        "rot_err_mean": float(np.mean(rot_errs)),
        "rot_err_max": float(np.max(rot_errs)),
        "rpe_trans_mean": float(np.mean(rpe_t)),
        "rpe_rot_mean": float(np.mean(rpe_r)),
        "n_common": len(common),
    }


def main():
    np.random.seed(42)
    K = make_K(*TARGET)
    print(f"K fx={K[0,0]:.2f}")
    gt = make_gt_pnp(N_FRAMES, K)

    methods = {
        "Ours":     (os.path.join(BENCH, "ours", "ours_poses"), "pose_"),
        "ORB-SfM":  (os.path.join(BENCH, "orb_sfm", "orb_poses"), "c2w_"),
        "COLMAP":   (os.path.join(BENCH, "colmap", "colmap_poses"), "c2w_"),
        "VGGT":     (os.path.join(BENCH, "vggt", "vggt_poses"), "c2w_"),
        "DUSt3R":   (os.path.join(BENCH, "dust3r", "dust3r_poses"), "c2w_"),
        "MASt3R":   (os.path.join(BENCH, "mast3r", "mast3r_poses"), "c2w_"),
        "MUSt3R":   (os.path.join(BENCH, "must3r", "must3r_poses"), "c2w_"),
        "Reloc3R":  (os.path.join(BENCH, "reloc3r", "reloc3r_poses"), "c2w_"),
    }

    out = {}
    print(f"\n{'Method':<10}{'n':>5}{'ATE(mm)':>10}{'Scale':>9}{'RotMean':>9}{'RPEt':>8}{'RPEr':>8}")
    print("-" * 60)
    for name, (d, pre) in methods.items():
        est = load_poses(d, pre)
        m = sim3_metrics(est, gt)
        if m is None:
            print(f"{name:<10}  <3 common frames")
            continue
        out[name] = m
        print(f"{name:<10}{m['n_common']:>5}{m['ate']:>10.4f}{m['sim3_scale']:>9.3f}"
              f"{m['rot_err_mean']:>9.2f}{m['rpe_trans_mean']:>8.3f}{m['rpe_rot_mean']:>8.2f}")

    # sanity check vs the three-pose-source experiment
    if "Ours" in out:
        a, s = out["Ours"]["ate"], out["Ours"]["sim3_scale"]
        ok = abs(a - 3.8864) < 0.2 and abs(s - 1.107) < 0.05
        print(f"\nSanity check Ours vs GT-PnP: ATE={a:.4f} (expect ~3.8864), "
              f"scale={s:.3f} (expect ~1.107) -> {'OK' if ok else 'MISMATCH'}")

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
