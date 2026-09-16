#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the newly captured endoscope sequences (2026-09-11/12) for the
heightfield reconstruction benchmark.

Per session:
  1. Extract cam0/video.avi frames, undistort with the calibrated intrinsics
     (calib_intrinsics_20260911_005642/camera_calibration_cam0.json).
  2. Save frames as img_%05d.jpg (0-based, matching frames_labeled.csv).
  3. Parse frames_labeled.csv -> GT c2w poses (mm) for pose_valid frames,
     restricted to the calibration board (12x9-3mm) for a consistent
     trajectory reference; also keep a per-board pose breakdown for the
     multi-scale analysis (N2).
Outputs under E:\\MIS_TMI_Re_3D\\new_capture\\<session>\\
  images/img_%05d.jpg, gt_poses/pose_%05d.txt, gt_meta.json
"""
from __future__ import annotations
import os, json
import numpy as np
import cv2
import pandas as pd

SRC_ROOT = r"D:\reloc3r\Data_IMU_Camera_Pose_Sequences\stereo_endoscope"
CALIB = os.path.join(SRC_ROOT, "calib_intrinsics_20260911_005642",
                     "camera_calibration_cam0.json")
OUT_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
SESSIONS = [
    "pnp_seq_20260911_011115",
    "vio_seq_20260911_010902",
    "vio_seq_20260911_011013",
    "vio_seq_20260912_104835",
]
TRAJ_BOARD = "12x9-3mm"   # calibration board: most valid frames, single physical target


def main():
    calib = json.load(open(CALIB))
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["distortion_coefficients"], dtype=np.float64)
    print(f"K fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    for sess in SESSIONS:
        sdir = os.path.join(SRC_ROOT, sess)
        csv_path = os.path.join(sdir, "frames_labeled.csv")
        video_path = os.path.join(sdir, "cam0", "video.avi")
        out_dir = os.path.join(OUT_ROOT, sess)
        img_dir = os.path.join(out_dir, "images")
        gt_dir = os.path.join(out_dir, "gt_poses")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        n_csv = len(df)

        cap = cv2.VideoCapture(video_path)
        n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"\n[{sess}] csv rows={n_csv} video frames={n_vid} fps={fps:.1f}")

        # extract + undistort
        idx = 0
        n_saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            und = cv2.undistort(frame, K, dist)
            cv2.imwrite(os.path.join(img_dir, f"img_{idx:05d}.jpg"), und,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_saved += 1
            idx += 1
        cap.release()
        print(f"  saved {n_saved} undistorted frames")

        # GT poses
        dfv = df[df["pose_valid"] == 1]
        boards = {}
        n_traj = 0
        for _, row in dfv.iterrows():
            fi = int(row["frame_idx"])
            R = np.array([[row["R_00"], row["R_01"], row["R_02"]],
                          [row["R_10"], row["R_11"], row["R_12"]],
                          [row["R_20"], row["R_21"], row["R_22"]]], dtype=np.float64)
            t = np.array([row["tvec_x"], row["tvec_y"], row["tvec_z"]], dtype=np.float64)
            w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
            c2w = np.linalg.inv(w2c)
            bname = str(row["chessboard_name"])
            boards.setdefault(bname, []).append(fi)
            if bname == TRAJ_BOARD:
                np.savetxt(os.path.join(gt_dir, f"pose_{fi:05d}.txt"), c2w, fmt="%.8f")
                n_traj += 1

        meta = {
            "session": sess,
            "n_frames": n_saved,
            "n_pose_valid": int(len(dfv)),
            "boards": {k: len(v) for k, v in boards.items()},
            "traj_board": TRAJ_BOARD,
            "n_traj_poses": n_traj,
            "K_1280x720": K.tolist(),
            "dist": dist.tolist(),
            "fps": fps,
        }
        json.dump(meta, open(os.path.join(out_dir, "gt_meta.json"), "w"), indent=2)
        print(f"  pose_valid={len(dfv)} boards={meta['boards']} traj({TRAJ_BOARD})={n_traj}")

        # per-frame rotation rate of the GT trajectory (for N4)
        if n_traj > 5:
            idxs = sorted(int(os.path.basename(f)[5:-4])
                          for f in os.listdir(gt_dir) if f.startswith("pose_"))
            rel_rots = []
            for a, b in zip(idxs[:-1], idxs[1:]):
                Ta = np.loadtxt(os.path.join(gt_dir, f"pose_{a:05d}.txt"))
                Tb = np.loadtxt(os.path.join(gt_dir, f"pose_{b:05d}.txt"))
                Rrel = Ta[:3, :3].T @ Tb[:3, :3]
                tr = np.clip(np.trace(Rrel), -1, 3)
                rel_rots.append(float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))))
            rel_rots = np.array(rel_rots)
            print(f"  GT relative rotation per valid-frame step: mean={rel_rots.mean():.2f}deg "
                  f"max={rel_rots.max():.2f}deg")


if __name__ == "__main__":
    main()
