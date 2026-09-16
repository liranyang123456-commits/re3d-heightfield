#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search-radius sweep (Supplement F.2) — FULLY faithful:
uses the paper's evaluation code (recover_true3d) with the original
results2 meshes and the real heightfield-ICP trajectory; only the
dense-matching search radius varies.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover_true3d_sift_pseudo3d_v3 import (
    DatasetLoader, TraditionalPoseEstimator, Pseudo3DDenseReconstructor,
    _make_camera_intrinsics,
)
import _experiment_three_pose_sources as tps

IMG_DIR = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2"
RESULTS_DIR = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2"
TARGET = (640, 358)
N = 50
RADII = [10, 15, 20, 25, 30, 40]
OUT_JSON = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\sweep_search_radius.json"


def main():
    ds = DatasetLoader(IMG_DIR, RESULTS_DIR, TARGET)
    K = _make_camera_intrinsics(TARGET[0], TARGET[1], fx=0.0)
    pose_est = TraditionalPoseEstimator(ds, K)
    c2w = {i: ds.get_rt(i) for i in range(N)}

    rows = []
    for r in RADII:
        dense_rec = Pseudo3DDenseReconstructor(ds, K, reproj_thresh=4.0)
        dense_rec.search_radius = float(r)
        t0 = time.time()
        res = tps.build_cloud(c2w, N, ds, K, pose_est, dense_rec, f"radius{r}")
        dt = time.time() - t0
        if res is None:
            print(f"r={r}: FAILED")
            continue
        row = {"r": r, "ratio": res["ratio"], "shape": res["shape"],
               "pts": res["pts"], "time_s": round(dt, 1)}
        rows.append(row)
        print(f"r={r}: lambda3/lambda1={res['ratio']:.4f} shape={res['shape']} "
              f"pts={res['pts']} time={dt:.1f}s", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    print("saved:", OUT_JSON)


if __name__ == "__main__":
    main()
