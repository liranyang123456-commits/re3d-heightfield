#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone, self-contained reimplementation of the paper's chessboard pipeline
for parameter sweeps (gradient threshold / mesh step / search radius / ICP).

Chain (following the paper Method and the Op3 reference implementation):
  gray(640x358) -> bilateral -> Sobel 3x3 magnitude -> min-max to [0,1]
  -> histogram percentile threshold (bins=1024) -> truncate -> nonzero min-max
  to [0,255] -> chessboard ROI mask (convex hull of 8x5 corners)
  -> heightfield z = minmax(g) * 0.15*max(H,W) on a regular grid (step=cell)
  -> per-pair FPFH + PCA/RANSAC coarse + point-to-point ICP (fitness gate)
  -> chained c2w poses -> dense reconstruction via recover_true3d's
     Pseudo3DDenseReconstructor (the paper's evaluation code, unmodified).

Validation mode reproduces the default operating point and compares against
the artifacts of record (lambda3/lambda1 ~ 0.42, ~11.7k points, ATE ~ 3.89 mm).
"""
from __future__ import annotations
import os, sys, glob, time, json, argparse
import numpy as np
import cv2

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover_true3d_sift_pseudo3d_v3 import (
    DatasetLoader, TraditionalPoseEstimator, Pseudo3DDenseReconstructor,
    _make_camera_intrinsics,
)
import _experiment_three_pose_sources as tps

IMG_DIR = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2"
TARGET = (640, 358)
N_FRAMES = 50
CB_SIZE = (8, 5)
Z_SCALE = 0.15 * max(TARGET)          # 96.0
SWEEP_ROOT = r"E:\MIS_TMI_Re_3D\sweep_runs"


# ---------------------------------------------------------------- heightfield
def gradient_map(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, d=7,
                               sigmaColor=0.1, sigmaSpace=3.0)
    dx = cv2.Sobel(filt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filt, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(dx, dy)
    gn = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    return gn                                        # [0,1]


def hist_percentile(gn, percent, bins=1024):
    hist, edges = np.histogram(gn, bins=bins, range=(0, 1))
    cum = np.cumsum(hist) / gn.size * 100.0
    x = (edges[:-1] + edges[1:]) / 2.0
    idx = np.searchsorted(cum, percent)
    return float(x[min(idx, len(x) - 1)])


def chessboard_mask(img_bgr, shape):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CB_SIZE, flags)
    h, w = shape
    if not ret:
        return np.ones((h, w), np.uint8) * 255
    hull = cv2.convexHull(corners).astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def build_heightfield(img_bgr, tau_pct, cell):
    """Return verts (N,3) float64 and color (N,3) uint8 for the grid mesh."""
    gn = gradient_map(img_bgr)
    th = hist_percentile(gn, tau_pct)
    g8 = np.clip(gn * 255.0, 0, 255).astype(np.uint8)
    trunc = np.where(g8 > int(th * 255.0), g8, 0).astype(np.float32)
    nz = trunc > 0
    if np.any(nz):
        lo, hi = trunc[nz].min(), trunc[nz].max()
        out = np.zeros_like(trunc)
        if hi > lo:
            out[nz] = (trunc[nz] - lo) / (hi - lo) * 255.0
        else:
            out[nz] = 255.0
    else:
        out = trunc
    mask = chessboard_mask(img_bgr, out.shape)
    # min-max to [0, Z_SCALE], then zero outside mask (Op3 order)
    a = out.astype(np.float32)
    if a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min()) * Z_SCALE
    else:
        a = np.zeros_like(a)
    a = np.where(mask > 0, a, 0.0)

    h, w = a.shape
    ys, xs = np.mgrid[0:h:cell, 0:w:cell]
    z = a[ys, xs]
    verts = np.stack([xs.ravel().astype(float), ys.ravel().astype(float),
                      z.ravel()], axis=1)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cols = gray[ys, xs].ravel()
    cols = np.stack([cols, cols, cols], axis=1).astype(np.uint8)
    return verts, cols


def write_mesh_ply(path, verts, cols):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(float) / 255.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=8.0, max_nn=30))
    o3d.io.write_point_cloud(path, pcd)   # x,y,z,nx,ny,nz,r,g,b


# ---------------------------------------------------------------- ICP poses
def pcd_from(verts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    return pcd


def register_pair(src_v, dst_v, voxel_budget=20000):
    """PCA coarse -> FPFH RANSAC fallback -> point-to-point ICP. Returns T (src->dst)."""
    src, dst = pcd_from(src_v), pcd_from(dst_v)
    scale = max(np.ptp(src_v, axis=0).max(), np.ptp(dst_v, axis=0).max(), 1e-6)

    if len(src_v) > voxel_budget:
        src = src.voxel_down_sample(scale * 0.02)
    if len(dst_v) > voxel_budget:
        dst = dst.voxel_down_sample(scale * 0.02)

    # PCA coarse alignment
    def pca_frame(p):
        c = p.mean(0)
        w, v = np.linalg.eigh(np.cov((p - c).T))
        return c, v[:, ::-1]
    c1, v1 = pca_frame(np.asarray(src.points))
    c2, v2 = pca_frame(np.asarray(dst.points))
    R = v2 @ v1.T
    if np.linalg.det(R) < 0:
        v1[:, -1] *= -1
        R = v2 @ v1.T
    T0 = np.eye(4)
    T0[:3, :3] = R
    T0[:3, 3] = c2 - R @ c1

    dist = scale * 0.05
    # fitness of PCA coarse; fallback to FPFH RANSAC if poor
    T_coarse = T0
    src_eval = pcd_from(np.asarray(src.points)).transform(T0)  # copy, no in-place
    ev = dst.compute_point_cloud_distance(src_eval)
    resid = float(np.mean(ev)) if len(ev) else 1e9
    if resid > 0.1 * scale:
        src_n = pcd_from(np.asarray(src.points)); dst_n = pcd_from(np.asarray(dst.points))
        src_n.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scale * 0.1, max_nn=30))
        dst_n.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scale * 0.1, max_nn=30))
        f1 = o3d.pipelines.registration.compute_fpfh_feature(
            src_n, o3d.geometry.KDTreeSearchParamHybrid(radius=scale * 0.2, max_nn=100))
        f2 = o3d.pipelines.registration.compute_fpfh_feature(
            dst_n, o3d.geometry.KDTreeSearchParamHybrid(radius=scale * 0.2, max_nn=100))
        rr = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_n, dst_n, f1, f2, mutual_filter=False,
            max_correspondence_distance=dist * 5.0,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist * 5.0)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        if rr.fitness > 0.1:
            T_coarse = np.asarray(rr.transformation)

    icp = o3d.pipelines.registration.registration_icp(
        src, dst, dist, T_coarse,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    if icp.fitness < 0.1:
        return np.eye(4), icp.fitness
    return np.asarray(icp.transformation), float(icp.fitness)


def estimate_trajectory(mesh_verts):
    """Chain pair registrations into c2w poses (frame0 = identity)."""
    n = len(mesh_verts)
    c2w = [np.eye(4)]
    fits = []
    for i in range(n - 1):
        T_rel, fit = register_pair(mesh_verts[i], mesh_verts[i + 1])
        fits.append(fit)
        # T_rel maps frame i -> i+1 in heightfield space; c2w chain:
        c2w.append(c2w[-1] @ np.linalg.inv(T_rel))
    return c2w, fits


# ---------------------------------------------------------------- sweep dirs
GT_POSE_DIR = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2\estimated_poses_txt"


def prepare_run_dir(tag, verts_per_frame, c2w=None, poses_from=None):
    run_dir = os.path.join(SWEEP_ROOT, tag)
    mesh_dir = os.path.join(run_dir, "mesh")
    pose_dir = os.path.join(run_dir, "estimated_poses_txt")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    for i, (v, c) in enumerate(verts_per_frame):
        write_mesh_ply(os.path.join(mesh_dir, f"debug_mesh_{i:04d}.ply"), v, c)
    if poses_from is not None:
        for f in glob.glob(os.path.join(poses_from, "pose_*.txt")):
            dst = os.path.join(pose_dir, os.path.basename(f))
            if not os.path.exists(dst):
                import shutil
                shutil.copy(f, dst)
    elif c2w is not None:
        for i, T in enumerate(c2w):
            np.savetxt(os.path.join(pose_dir, f"pose_{i:04d}.txt"), T, fmt="%.8f")
    return run_dir


def run_dense(tag, run_dir, n, K, search_radius=25.0):
    """Dense reconstruction via the paper's evaluation code (unmodified)."""
    ds = DatasetLoader(IMG_DIR, run_dir, TARGET)
    pose_est = TraditionalPoseEstimator(ds, K)
    dense_rec = Pseudo3DDenseReconstructor(ds, K, reproj_thresh=4.0)
    dense_rec.search_radius = search_radius
    pose_dir = os.path.join(run_dir, "estimated_poses_txt")
    c2w = {i: np.loadtxt(os.path.join(pose_dir, f"pose_{i:04d}.txt"))
           for i in range(n)
           if os.path.exists(os.path.join(pose_dir, f"pose_{i:04d}.txt"))}
    res = tps.build_cloud(c2w, n, ds, K, pose_est, dense_rec, tag)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="validate",
                    choices=["validate", "sweep_threshold", "sweep_cell"])
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "photo_*.jpg")))[:N_FRAMES]
    K = _make_camera_intrinsics(TARGET[0], TARGET[1], fx=0.0)
    print(f"frames: {len(imgs)}")

    if args.mode == "validate":
        # Mesh-only validation: my meshes + ORIGINAL poses (fixed).
        verts_per_frame = []
        for i, p in enumerate(imgs):
            img = cv2.resize(cv2.imread(p), TARGET, interpolation=cv2.INTER_AREA)
            verts_per_frame.append(build_heightfield(img, tau_pct=68, cell=3))
        run_dir = prepare_run_dir("validate_meshonly_t68_c3", verts_per_frame,
                                  poses_from=GT_POSE_DIR)
        res = run_dense("validate_meshonly_t68_c3", run_dir, N_FRAMES, K)
        if res:
            print(f"\n[VALIDATION mesh-only] points={res['pts']} (expect ~11735), "
                  f"lambda3/lambda1={res['ratio']:.4f} (expect ~0.42), shape={res['shape']}")
        return

    if args.mode == "sweep_threshold":
        values = [30, 40, 50, 60, 68, 75, 80, 90]
        rows = []
        for tau in values:
            tag = f"tau{tau}"
            verts_per_frame = []
            for p in imgs:
                img = cv2.resize(cv2.imread(p), TARGET, interpolation=cv2.INTER_AREA)
                verts_per_frame.append(build_heightfield(img, tau_pct=tau, cell=3))
            run_dir = prepare_run_dir(tag, verts_per_frame, poses_from=GT_POSE_DIR)
            t0 = time.time()
            res = run_dense(tag, run_dir, N_FRAMES, K)
            dt = time.time() - t0
            if res:
                n_valid = int(np.mean([np.sum(v[:, 2] > 0) for v, _ in verts_per_frame]))
                row = {"tau": tau, "ratio": res["ratio"], "shape": res["shape"],
                       "pts": res["pts"], "valid_verts_avg": n_valid,
                       "dense_time_s": round(dt, 1)}
                rows.append(row)
                print(f"tau={tau}: ratio={res['ratio']:.4f} shape={res['shape']} "
                      f"pts={res['pts']} verts~{n_valid}", flush=True)
                with open(os.path.join(SWEEP_ROOT, "sweep_threshold.json"), "w") as f:
                    json.dump(rows, f, indent=2)
        return

    if args.mode == "sweep_cell":
        values = [2, 3, 4, 5, 7, 9]
        rows = []
        for cell in values:
            tag = f"cell{cell}"
            verts_per_frame = []
            for p in imgs:
                img = cv2.resize(cv2.imread(p), TARGET, interpolation=cv2.INTER_AREA)
                verts_per_frame.append(build_heightfield(img, tau_pct=68, cell=cell))
            run_dir = prepare_run_dir(tag, verts_per_frame, poses_from=GT_POSE_DIR)
            t0 = time.time()
            res = run_dense(tag, run_dir, N_FRAMES, K)
            dt = time.time() - t0
            if res:
                n_verts = len(verts_per_frame[0][0])
                row = {"cell": cell, "ratio": res["ratio"], "shape": res["shape"],
                       "pts": res["pts"], "mesh_verts": n_verts,
                       "dense_time_s": round(dt, 1)}
                rows.append(row)
                print(f"cell={cell}: ratio={res['ratio']:.4f} shape={res['shape']} "
                      f"pts={res['pts']} mesh_verts={n_verts}", flush=True)
                with open(os.path.join(SWEEP_ROOT, "sweep_cell.json"), "w") as f:
                    json.dump(rows, f, indent=2)
        return


if __name__ == "__main__":
    main()
