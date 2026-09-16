#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithful parameter sweeps by driving the ACTUAL Op3 pipeline as a library.

Per configuration:
  1. Run Op3 ImageProcessor over the first 50 frames with the swept
     attribute overridden (threshold_percent / o3d_step).
  2. Accumulate and save the heightfield-ICP poses (same calls as Op3 main).
  3. Run the paper's dense reconstruction (recover_true3d, unmodified) on the
     sweep output dir and record lambda3/lambda1, points, shape, runtime.

Usage:
  python _sweep_op3_driver.py --param threshold --value 68 [--frames 50]
  python _sweep_op3_driver.py --param cell --value 3 [--frames 50]
  python _sweep_op3_driver.py --param default --value 0 --frames 3  (smoke test)
"""
from __future__ import annotations
import os, sys, glob, time, json, argparse

# ---- env MUST be set before importing the Op3 module ----
def _setup_env(out_dir):
    os.environ["MATPLOTLIB_INTERACTIVE"] = "0"
    os.environ["USE_CPU"] = "1"
    os.environ["MATCH_VISUALIZE"] = "0"
    os.environ["O3D_VISUALIZE"] = "0"
    os.environ["OUTPUT_DIR"] = out_dir
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import cv2

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

OP3_PATH = os.environ.get("OP3_PATH", os.path.join(
    CODE_DIR, "ChessBoard_Cal_RT_CMP_3DICP_2DFFT_Fea_Matching_MIS_Op3.py"))
IMG_DIR = os.environ.get("OP3_IMG_DIR", r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2")
TARGET = (640, 358)
SWEEP_ROOT = os.environ.get("OP3_SWEEP_ROOT", r"E:\MIS_TMI_Re_3D\sweep_runs")
SUMMARY_JSON = os.path.join(SWEEP_ROOT, "sweep_summary.json")


def load_op3():
    import importlib.util
    spec = importlib.util.spec_from_file_location("op3_pipeline", OP3_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_op3(tag, param, value, n_frames):
    """Run the real Op3 pipeline over the first n_frames images."""
    out_dir = os.path.join(SWEEP_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    _setup_env(out_dir)

    mod = load_op3()
    processor = mod.ImageProcessor(sigma=4.0, sigma_spatial=3.0,
                                   sigma_range=0.1, kernel_radius=3)
    processor.start_frame_idx = 0
    if param == "threshold":
        processor.threshold_percent = int(value)
        processor.threshold_mode = f"th_{int(value)}"
    elif param == "cell":
        processor.o3d_step = int(value)          # heightfield grid step
        processor.mesh_cell_size = int(value)    # JSON grid (consistency)

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
                  glob.glob(os.path.join(IMG_DIR, "*.png")))[:n_frames]
    print(f"[{tag}] {len(imgs)} frames, param {param}={value}")
    t0 = time.time()
    for idx, p in enumerate(imgs):
        img = cv2.imread(p)
        processor.process_image(img, frame_idx=idx, original_image_path=p)
        if idx < 2 or (idx + 1) % 10 == 0:
            print(f"[{tag}] frame {idx} done ({time.time()-t0:.0f}s)", flush=True)
        # incremental pose checkpoint every 50 frames: a GPU crash must not
        # lose the whole chain (session-4 crash at frame 378 cost 3.7 h)
        if (idx + 1) % 50 == 0:
            try:
                acc = processor._accumulate_camera_poses_from_rt_matrices(
                    reference_frame_idx=0)
                if acc is not None:
                    processor._save_accumulated_poses(
                        acc, os.path.join(out_dir, "accumulated_camera_poses.json"))
                    print(f"[{tag}] pose checkpoint at frame {idx}", flush=True)
            except Exception as e:
                print(f"[{tag}] pose checkpoint failed at {idx}: {e}", flush=True)

    # accumulate and save poses (same as Op3 main)
    acc = processor._accumulate_camera_poses_from_rt_matrices(reference_frame_idx=0)
    if acc is not None:
        processor._save_accumulated_poses(acc, os.path.join(out_dir, "accumulated_camera_poses.json"))
    dt = time.time() - t0
    print(f"[{tag}] Op3 done in {dt:.0f}s -> {out_dir}", flush=True)
    return out_dir, dt


def run_dense(tag, out_dir, n_frames):
    from recover_true3d_sift_pseudo3d_v3 import (
        DatasetLoader, TraditionalPoseEstimator, Pseudo3DDenseReconstructor,
        _make_camera_intrinsics)
    import _experiment_three_pose_sources as tps

    ds = DatasetLoader(IMG_DIR, out_dir, TARGET)
    K = _make_camera_intrinsics(TARGET[0], TARGET[1], fx=0.0)
    pose_est = TraditionalPoseEstimator(ds, K)
    dense_rec = Pseudo3DDenseReconstructor(ds, K, reproj_thresh=4.0)
    # only frames with an actual saved pose file (get_rt defaults to identity
    # for missing frames, which would corrupt the fusion)
    pose_dir = os.path.join(out_dir, "estimated_poses_txt")
    c2w = {}
    for f in glob.glob(os.path.join(pose_dir, "pose_*.txt")):
        idx = int(os.path.basename(f).replace("pose_", "").replace(".txt", ""))
        c2w[idx] = np.loadtxt(f)
    n_use = min(ds.num_frames(), max(c2w) + 1 if c2w else 0)
    print(f"[{tag}] dense stage: {len(c2w)} poses, {n_use} frames")
    t0 = time.time()
    res = tps.build_cloud(c2w, n_use, ds, K, pose_est, dense_rec, tag)
    dt = time.time() - t0
    if res is None:
        return None
    res["dense_time_s"] = round(dt, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", default="default",
                    choices=["default", "threshold", "cell"])
    ap.add_argument("--value", default="0")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    val = float(args.value)
    tag = args.tag or (f"{args.param}{args.value}" if args.param != "default"
                       else f"default_smoke{args.frames}")

    out_dir, op3_time = run_op3(tag, args.param, val, args.frames)

    row = {"tag": tag, "param": args.param, "value": val,
           "frames": args.frames, "op3_time_s": round(op3_time, 1)}

    pose_dir = os.path.join(out_dir, "estimated_poses_txt")
    n_poses = len(glob.glob(os.path.join(pose_dir, "pose_*.txt")))
    row["n_poses"] = n_poses
    print(f"[{tag}] poses saved: {n_poses}")

    if args.frames >= 10 and n_poses >= 10:
        res = run_dense(tag, out_dir, args.frames)
        if res:
            row.update({"ratio": res["ratio"], "shape": res["shape"],
                        "pts": res["pts"], "dense_time_s": res["dense_time_s"]})
            print(f"[{tag}] lambda3/lambda1={res['ratio']:.4f} "
                  f"shape={res['shape']} pts={res['pts']}", flush=True)

    os.makedirs(SWEEP_ROOT, exist_ok=True)
    rows = []
    if os.path.exists(SUMMARY_JSON):
        rows = json.load(open(SUMMARY_JSON))
    rows = [r for r in rows if r["tag"] != tag]
    rows.append(row)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    print("summary updated:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
