#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corner-depth accuracy on the newly captured sequences (calibrated intrinsics).

For each session: on frames with a valid 12x9-3mm board, detect the 11x8 inner
corners in the undistorted frame, take GT corner depth from the PnP pose
(board frame -> camera, mm), project our fused cloud into the frame with our
estimated pose, sample our depth near each corner pixel, and compare after
affine alignment (gt = a*pred + b) -- the same protocol as the chessboard
benchmark depth evaluation in the main paper.
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
RUN_ROOT = r"E:\MIS_TMI_Re_3D\new_capture_runs"
CLOUD_ROOT = r"E:\MIS_TMI_Re_3D\compare_pose_sources"
SESSIONS = ["pnp_seq_20260911_011115", "vio_seq_20260911_010902",
            "vio_seq_20260911_011013"]
CB = (11, 8)          # inner corners of the 12x9-3mm board
SQ = 3.0              # mm
SEARCH_R = 25.0       # px
MAX_CLOUD = 200000

# calibrated K at 1280x720, frames are undistorted; pipeline runs at 640x358
K1280 = np.array([[1210.6664, 0, 726.3980],
                  [0, 1194.5669, 391.9404],
                  [0, 0, 1.0]])
# our frames on disk are undistorted 1280x720; evaluation at native resolution
K = K1280


def load_cloud(session):
    import open3d as o3d
    p = os.path.join(CLOUD_ROOT, f"cloud_{session}.ply")
    if not os.path.exists(p):
        return None
    pts = np.asarray(o3d.io.read_point_cloud(p).points)
    if len(pts) > MAX_CLOUD:
        pts = pts[np.random.RandomState(0).choice(len(pts), MAX_CLOUD, replace=False)]
    return pts


def load_est_poses(session):
    d = os.path.join(RUN_ROOT, session, "estimated_poses_txt")
    out = {}
    for f in glob.glob(os.path.join(d, "pose_*.txt")):
        idx = int(os.path.basename(f).replace("pose_", "").replace(".txt", ""))
        out[idx] = np.loadtxt(f)
    return out


def gt_frames(session):
    """frame_idx -> (w2c 4x4) for valid 12x9 frames, from frames_labeled.csv."""
    import pandas as pd
    df = pd.read_csv(os.path.join(r"D:\reloc3r\Data_IMU_Camera_Pose_Sequences",
                                  "stereo_endoscope", session, "frames_labeled.csv"))
    v = df[(df.pose_valid == 1) & (df.chessboard_name == "12x9-3mm")]
    out = {}
    for _, r in v.iterrows():
        R = np.array([[r.R_00, r.R_01, r.R_02],
                      [r.R_10, r.R_11, r.R_12],
                      [r.R_20, r.R_21, r.R_22]])
        t = np.array([r.tvec_x, r.tvec_y, r.tvec_z])
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        out[int(r.frame_idx)] = w2c
    return out


def main():
    # board corner 3D (board frame, mm)
    objp = np.zeros((CB[0] * CB[1], 3))
    objp[:, :2] = np.mgrid[0:CB[0], 0:CB[1]].T.reshape(-1, 2) * SQ

    for sess in SESSIONS:
        cloud = load_cloud(sess)
        est = load_est_poses(sess)
        gt = gt_frames(sess)
        if cloud is None or not est or not gt:
            print(f"[{sess}] missing cloud/est/gt, skip")
            continue

        # Sim3 align our trajectory to GT (positions) to get world->gt frame
        common = sorted(set(est) & set(gt))
        if len(common) < 5:
            print(f"[{sess}] too few common frames"); continue
        E = np.array([est[i][:3, 3] for i in common])
        G = np.array([np.linalg.inv(gt[i])[:3, 3] for i in common])  # cam centers
        sc, tc = E.mean(0), G.mean(0)
        sd, td = E - sc, G - tc
        s = np.sqrt((td ** 2).sum() / len(td)) / max(np.sqrt((sd ** 2).sum() / len(sd)), 1e-8)
        U, _, Vt = np.linalg.svd(sd.T @ td)
        Rg = Vt.T @ U.T
        if np.linalg.det(Rg) < 0:
            Vt[-1, :] *= -1; Rg = Vt.T @ U.T
        tg = tc - s * Rg @ sc
        # our world point -> gt-board world:  X' = s*Rg@X + tg
        cloud_g = (s * (Rg @ cloud.T).T + tg[None, :])

        preds, gts = [], []
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        for fi in common[:: max(1, len(common) // 40)]:   # up to ~40 frames
            img_path = os.path.join(NEW_ROOT, sess, "images", f"img_{fi:05d}.jpg")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ret, corners = cv2.findChessboardCorners(img, CB, flags)
            if not ret:
                continue
            # GT corner depth (camera frame)
            w2c = gt[fi]
            Xc = (w2c[:3, :3] @ objp.T).T + w2c[:3, 3]
            gt_depth = Xc[:, 2]
            # our cloud in this frame's camera: need our c2w for frame fi,
            # mapped into gt world first, then into GT camera of frame fi
            T_ours = est[fi]                      # our c2w (our world)
            # our point -> our camera: Xc = inv(T_ours) applied in OUR world.
            # But cloud_g is in GT world; use GT camera w2c directly:
            Xc_ours = (w2c[:3, :3] @ cloud_g.T).T + w2c[:3, 3]
            z = Xc_ours[:, 2]
            fr = z > 1e-3
            proj = np.full((len(Xc_ours), 2), np.nan)
            proj[fr, 0] = K[0, 0] * Xc_ours[fr, 0] / z[fr] + K[0, 2]
            proj[fr, 1] = K[1, 1] * Xc_ours[fr, 1] / z[fr] + K[1, 2]
            for k in range(len(objp)):
                u, v = corners[k, 0]
                d = np.hypot(proj[:, 0] - u, proj[:, 1] - v)
                m = d < SEARCH_R
                if m.any():
                    preds.append(float(np.median(z[m])))
                    gts.append(float(gt_depth[k]))
        if len(preds) < 20:
            print(f"[{sess}] too few matched corners ({len(preds)})"); continue
        preds = np.array(preds); gts = np.array(gts)
        A = np.vstack([preds, np.ones_like(preds)]).T
        a, b = np.linalg.lstsq(A, gts, rcond=None)[0]
        pa = a * preds + b
        absrel = np.mean(np.abs(pa - gts) / gts)
        rmse = np.sqrt(np.mean((pa - gts) ** 2))
        d1 = np.mean(np.maximum(pa / gts, gts / pa) < 1.25)
        print(f"[{sess}] corners={len(preds)}  AbsRel={absrel:.4f}  "
              f"RMSE={rmse:.2f}mm  d1.25={d1:.3f}  (affine a={a:.3f})")
        out = {"corners": len(preds), "AbsRel": float(absrel),
               "RMSE_mm": float(rmse), "d1_25": float(d1),
               "affine_a": float(a), "affine_b": float(b)}
        with open(os.path.join(NEW_ROOT, sess, "eval_corner_depth.json"), "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
