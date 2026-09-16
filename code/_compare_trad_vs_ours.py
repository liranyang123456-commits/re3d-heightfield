#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双管道对比: Traditional SfM vs Ours (Dense Epipolar)
- 两条管道都使用GT绝对位姿，避免链式累积漂移导致的锥形畸变
- 各自独立运行，独立保存结果
- 生成对比可视化
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import time
import csv
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from recover_true3d_sift_pseudo3d_v3 import (
    DatasetLoader, TraditionalPoseEstimator, Pseudo3DDenseReconstructor,
    _make_camera_intrinsics, _safe_mm, _inv_3x3, _matmul_4x4, _write_ply,
    _parse_ply_binary, _build_grid_from_mesh,
)


# ============================================================
#  工具函数
# ============================================================

def get_world_transform(ds, frame_idx, K=None, ref_depth=None):
    """获取帧的c2w变换矩阵 (直接使用GT c2w，无需校正)
    
    GT c2w已在真实相机(X,Y,Z)坐标系中，可直接用于三角化点的世界坐标变换。
    之前的仿射校正(correct_gt_c2w_for_scale)反而造成了锥形畸变，
    因为A^{-1}@R_gt@A不是正交矩阵，破坏了旋转的结构。
    """
    return ds.get_rt(frame_idx)


def estimate_scale_z_from_trajectory(ds, K, max_frames):
    """已弃用: GT c2w无需尺度校正"""
    return 1.0


def triangulate_relative(pts1, pts2, R, t, K, dense_rec,
                         reproj_thresh=4.0, max_depth_ratio=5.0, min_disparity=1.0):
    """三角化: 委托给Pseudo3DDenseReconstructor.triangulate_with_known_pose"""
    result = dense_rec.triangulate_with_known_pose(pts1, pts2, R, t)
    if result is None:
        # 手动诊断
        P1 = np.zeros((3, 4), dtype=np.float64)
        P1[0, 0] = K[0,0]; P1[1, 1] = K[1,1]; P1[0, 2] = K[0,2]; P1[1, 2] = K[1,2]
        P2 = np.zeros((3, 4), dtype=np.float64)
        P2[:3, :3] = _safe_mm(K, R)
        P2[:3, 3:4] = _safe_mm(K, t.reshape(3, 1))
        X_h = cv2.triangulatePoints(P1, P2, pts1[:3].T.astype(np.float64), pts2[:3].T.astype(np.float64))
        X = (X_h[:3, :] / (X_h[3:4, :] + 1e-12)).T
        print(f"    DEBUG: first 3 pts3d Z: {X[:3,2]}")
        Z2_vals = (_safe_mm(R, X.T)+t.reshape(3,1)).T[:,2]
        print(f"    DEBUG: Z2: {Z2_vals[:3]}")
        return np.zeros((0, 3)), np.zeros(0, dtype=bool)
    X3d, pts1_ok, _, _ = result
    ok = np.ones(len(X3d), dtype=bool)
    return X3d, ok


def voxelize_downsample(pts, cols, vs=0.5):
    """体素降采样"""
    if len(pts) < 10:
        return pts, cols
    voxel_idx = np.floor(pts / vs).astype(np.int64)
    flat = voxel_idx[:, 0] * 100000000 + voxel_idx[:, 1] * 10000 + voxel_idx[:, 2]
    _, first_idx = np.unique(flat, return_index=True)
    return pts[first_idx], cols[first_idx]


def remove_outliers(pts, cols, ratio=5.0):
    """统计离群点移除"""
    if len(pts) < 100:
        return pts, cols
    centroid = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - centroid[None, :], axis=1)
    med = np.median(dists)
    inlier = dists < med * ratio
    return pts[inlier], cols[inlier]


# ============================================================
#  管道1: Traditional SfM (SIFT+ORB E-inliers 三角化)
# ============================================================

def run_traditional_pipeline(ds, K, pose_est, dense_rec, max_frames, output_dir):
    """传统SfM管道: SIFT+ORB特征匹配→E-RANSAC→三角化"""
    os.makedirs(output_dir, exist_ok=True)
    dir_ply = os.path.join(output_dir, "per_pair")
    os.makedirs(dir_ply, exist_ok=True)

    n = min(ds.num_frames(), max_frames)
    all_pts = []
    all_cols = []
    stats = []

    t0 = time.time()
    for i in range(n - 1):
        fi, fj = i, i + 1

        # SIFT+ORB特征匹配
        pts1_all, pts2_all = pose_est._match_features(fi, fj)
        if i < 3:
            print(f"  [{fi}->{fj}] raw matches={len(pts1_all)}")
        if len(pts1_all) < 8:
            if i < 3: print(f"  [{fi}->{fj}] SKIP: <8 raw matches")
            continue

        # E-RANSAC估计相对位姿 (用于三角化，已验证有效)
        result = pose_est.estimate_pose(fi, fj)
        if result is None:
            if i < 3: print(f"  [{fi}->{fj}] SKIP: no E-RANSAC result")
            continue
        R_rel, t_rel, inlier_mask, _ = result

        pts1_in = pts1_all[inlier_mask]
        pts2_in = pts2_all[inlier_mask]
        if i < 3:
            print(f"  [{fi}->{fj}] inliers={len(pts1_in)}, E t={t_rel}")
        if len(pts1_in) < 5:
            if i < 3: print(f"  [{fi}->{fj}] SKIP: <5 inliers")
            continue

        # 三角化 (使用E-matrix相对位姿)
        tri_result = dense_rec.triangulate_with_known_pose(pts1_in, pts2_in, R_rel, t_rel)
        if tri_result is None:
            if i < 3: print(f"  [{fi}->{fj}] SKIP: triangulation failed")
            continue
        X3d_ok, pts1_ok, _, _ = tri_result
        if i < 3:
            print(f"  [{fi}->{fj}] triangulated={len(X3d_ok)}")

        # 颜色采样
        img_i = ds.load_image(fi)
        if img_i is not None:
            xs = np.clip(np.round(pts1_ok[:, 0]).astype(int), 0, img_i.shape[1]-1)
            ys = np.clip(np.round(pts1_ok[:, 1]).astype(int), 0, img_i.shape[0]-1)
            colors = img_i[ys, xs, :]
        else:
            colors = np.full((len(X3d_ok), 3), 128, dtype=np.uint8)

        # cam1坐标 → 世界坐标 (GT c2w直接变换)
        T_c2w = get_world_transform(ds, fi)
        R_i = T_c2w[:3, :3]
        t_i_vec = T_c2w[:3, 3:4]
        X_world = (_safe_mm(R_i, X3d_ok.T) + t_i_vec).T

        _write_ply(X_world, colors, os.path.join(dir_ply, f"trad_{fi:04d}_{fj:04d}.ply"))
        all_pts.append(X_world.astype(np.float32))
        all_cols.append(colors.astype(np.uint8))
        stats.append({"pair": f"{fi}->{fj}", "points": len(X3d_ok)})

        if (i + 1) % 10 == 0:
            print(f"  [Traditional] {i+1}/{n-1} 对完成")

    if not all_pts:
        print("  [Traditional] 无点云生成!")
        return None, None

    merged = np.vstack(all_pts)
    merged_cols = np.vstack(all_cols)

    # 体素降采样 + 离群点移除
    merged, merged_cols = voxelize_downsample(merged, merged_cols, vs=0.5)
    merged, merged_cols = remove_outliers(merged, merged_cols, ratio=5.0)

    dt = time.time() - t0
    _write_ply(merged, merged_cols, os.path.join(output_dir, "traditional_final.ply"))

    # 统计
    with open(os.path.join(output_dir, "stats.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["pair", "points"])
        w.writeheader()
        for s in stats:
            w.writerow(s)

    summary = (f"Traditional SfM Pipeline\n"
               f"  总帧对: {n-1}\n"
               f"  成功帧对: {len(stats)}\n"
               f"  最终点数: {len(merged)}\n"
               f"  耗时: {dt:.1f}s\n")
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write(summary)

    print(f"\n{summary}")
    return merged, merged_cols


# ============================================================
#  管道2: Ours (GT位姿 + 密集极线匹配 + 传统融合)
# ============================================================

def run_ours_pipeline(ds, K, pose_est, dense_rec, max_frames, output_dir):
    """Ours管道: GT位姿 + 密集极线匹配 + 传统融合"""
    os.makedirs(output_dir, exist_ok=True)
    dir_ply = os.path.join(output_dir, "per_pair")
    os.makedirs(dir_ply, exist_ok=True)

    n = min(ds.num_frames(), max_frames)
    all_pts = []
    all_cols = []
    stats = []

    t0 = time.time()
    for i in range(n - 1):
        fi, fj = i, i + 1
        # E-matrix估计相对位姿 (用于三角化和极线匹配)
        result = pose_est.estimate_pose(fi, fj)
        if result is None:
            continue
        R_rel, t_rel, inlier_mask, _ = result

        # 加载P3D网格
        try:
            grid_i = ds.load_mesh_grid(fi)
            grid_j = ds.load_mesh_grid(fj)
        except Exception as e:
            continue

        img_i = ds.load_image(fi)

        # ---- 传统稀疏点 (获取参考深度) ----
        trad_pts = np.zeros((0, 3), dtype=np.float64)
        trad_cols = np.zeros((0, 3), dtype=np.uint8)
        trad_px = np.zeros((0, 2), dtype=np.float64)
        n_trad = 0
        ref_depth_median = None

        pts1_all, pts2_all = pose_est._match_features(fi, fj)
        if len(pts1_all) >= 8:
            pts1_in = pts1_all[inlier_mask]
            pts2_in = pts2_all[inlier_mask]
            if len(pts1_in) >= 5:
                tri_result = dense_rec.triangulate_with_known_pose(pts1_in, pts2_in, R_rel, t_rel)
                if tri_result is not None:
                    X3d_t, pts1_ok_t, _, _ = tri_result
                    if len(X3d_t) >= 3:
                        trad_pts = X3d_t
                        trad_px = pts1_ok_t
                        n_trad = len(X3d_t)
                        ref_depth_median = float(np.median(X3d_t[:, 2]))
                        if img_i is not None:
                            xs = np.clip(np.round(pts1_ok_t[:, 0]).astype(int), 0, img_i.shape[1]-1)
                            ys = np.clip(np.round(pts1_ok_t[:, 1]).astype(int), 0, img_i.shape[0]-1)
                            trad_cols = img_i[ys, xs, :]

        # ---- 密集极线匹配 ----
        t_epi = time.time()
        pts1_epi, pts2_epi, X3d_epi, colors_epi = dense_rec.match_dense_epipolar(
            grid_i, grid_j, R_rel, t_rel, img_i=img_i, ref_depth_median=ref_depth_median)
        dt_epi = time.time() - t_epi
        n_epi = len(pts1_epi)

        # ---- 融合: 传统 + 密集极线去重 ----
        epi_px = pts1_epi if n_epi > 0 else np.zeros((0, 2), dtype=np.float64)
        n_unique = 0
        if n_epi > 0 and n_trad > 0:
            keep = np.ones(n_epi, dtype=bool)
            dedup_px = 3.0
            for b in range(0, n_epi, 500):
                be = min(b + 500, n_epi)
                dists = np.linalg.norm(epi_px[b:be, None, :] - trad_px[None, :, :], axis=2)
                keep[b:be] = dists.min(axis=1) > dedup_px
            n_unique = keep.sum()
            X3d_epi_f = X3d_epi[keep]
            colors_epi_f = colors_epi[keep] if len(colors_epi) > 0 else colors_epi
        else:
            X3d_epi_f = X3d_epi
            colors_epi_f = colors_epi

        # 合并
        if n_trad > 0 and n_unique > 0:
            X_merged = np.vstack([trad_pts, X3d_epi_f])
            cols_merged = np.vstack([trad_cols, colors_epi_f])
        elif n_trad > 0:
            X_merged = trad_pts
            cols_merged = trad_cols
        elif n_unique > 0 or n_epi > 0:
            X_merged = X3d_epi_f if n_unique > 0 else X3d_epi
            cols_merged = colors_epi_f if n_unique > 0 else colors_epi
        else:
            continue

        # cam1坐标 → 世界坐标 (GT c2w直接变换)
        T_c2w = get_world_transform(ds, fi)
        R_i = T_c2w[:3, :3]
        t_i_vec = T_c2w[:3, 3:4]
        X_world = (_safe_mm(R_i, X_merged.T) + t_i_vec).T

        _write_ply(X_world, cols_merged, os.path.join(dir_ply, f"ours_{fi:04d}_{fj:04d}.ply"))
        all_pts.append(X_world.astype(np.float32))
        all_cols.append(cols_merged.astype(np.uint8))
        stats.append({
            "pair": f"{fi}->{fj}",
            "epi_matches": n_epi,
            "trad_matches": n_trad,
            "epi_unique": n_unique,
            "fused": len(X_merged),
            "dt_epi": round(dt_epi, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"  [Ours] {i+1}/{n-1} 对完成, epi={n_epi}, trad={n_trad}, fused={len(X_merged)}")

    if not all_pts:
        print("  [Ours] 无点云生成!")
        return None, None

    merged = np.vstack(all_pts)
    merged_cols = np.vstack(all_cols)

    merged, merged_cols = voxelize_downsample(merged, merged_cols, vs=0.5)
    merged, merged_cols = remove_outliers(merged, merged_cols, ratio=5.0)

    dt = time.time() - t0
    _write_ply(merged, merged_cols, os.path.join(output_dir, "ours_final.ply"))

    with open(os.path.join(output_dir, "stats.csv"), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["pair", "epi_matches", "trad_matches", "epi_unique", "fused", "dt_epi"])
        w.writeheader()
        for s in stats:
            w.writerow(s)

    epi_total = sum(s["epi_matches"] for s in stats)
    trad_total = sum(s["trad_matches"] for s in stats)
    summary = (f"Ours Pipeline (GT Poses + Dense Epipolar + Traditional)\n"
               f"  总帧对: {n-1}\n"
               f"  成功帧对: {len(stats)}\n"
               f"  密集极线总匹配: {epi_total}\n"
               f"  传统总匹配: {trad_total}\n"
               f"  最终点数: {len(merged)}\n"
               f"  耗时: {dt:.1f}s\n")
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write(summary)

    print(f"\n{summary}")
    return merged, merged_cols


# ============================================================
#  可视化 (OpenCV, 避免matplotlib BLAS死锁)
# ============================================================

def _hist2d_cv(pts, dx, dy, bins, xrange, yrange, colormap=cv2.COLORMAP_VIRIDIS):
    """OpenCV 2D直方图"""
    H = np.zeros((bins, bins), dtype=np.float32)
    x_vals = pts[:, dx]
    y_vals = pts[:, dy]
    xi = ((x_vals - xrange[0]) / max(xrange[1] - xrange[0], 1e-6) * bins).astype(int)
    yi = ((y_vals - yrange[0]) / max(yrange[1] - yrange[0], 1e-6) * bins).astype(int)
    valid = (xi >= 0) & (xi < bins) & (yi >= 0) & (yi < bins)
    for i in np.where(valid)[0]:
        H[yi[i], xi[i]] += 1
    H = np.log1p(H)
    H = (H / max(H.max(), 1e-6) * 255).astype(np.uint8)
    H_color = cv2.applyColorMap(H, colormap)
    return H_color


def plot_comparison_cv(pts_trad, pts_dense, output_path):
    """对比可视化"""
    all_pts = np.vstack([pts_trad, pts_dense])
    lo = np.percentile(all_pts, 2, axis=0)
    hi = np.percentile(all_pts, 98, axis=0)

    views = [
        ("XY (Top)", 0, 1),
        ("XZ (Front)", 0, 2),
        ("YZ (Side)", 1, 2),
    ]

    BINS = 200
    IMG_H = 300
    TITLE_H = 40
    LABEL_W = 180

    canvas_w = BINS * 3 + LABEL_W * 3 + 20
    canvas_h = IMG_H * 2 + TITLE_H * 3 + 80
    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)

    y_offset = 10
    ratio = len(pts_dense) / max(1, len(pts_trad))
    cv2.putText(canvas, f"Point Cloud: Traditional SfM vs Ours (GT Poses, {ratio:.1f}x)",
               (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    y_offset += TITLE_H

    for row_idx, (pts, label) in enumerate([
        (pts_trad, f"Traditional SfM ({len(pts_trad)} pts)"),
        (pts_dense, f"Ours: Dense Epi+Trad ({len(pts_dense)} pts)"),
    ]):
        cv2.putText(canvas, label, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        y_offset += TITLE_H

        for col_idx, (vtitle, dx, dy) in enumerate(views):
            xrange = [lo[dx], hi[dx]]
            yrange = [lo[dy], hi[dy]]
            mask = ((pts[:, dx] >= xrange[0]) & (pts[:, dx] <= xrange[1]) &
                    (pts[:, dy] >= yrange[0]) & (pts[:, dy] <= yrange[1]))
            p = pts[mask]
            cmap = cv2.COLORMAP_VIRIDIS if row_idx == 0 else cv2.COLORMAP_HOT
            img = _hist2d_cv(p, dx, dy, BINS, xrange, yrange, colormap=cmap)
            img = cv2.resize(img, (BINS, IMG_H))
            x_off = LABEL_W + col_idx * (BINS + LABEL_W) + 10
            canvas[y_offset:y_offset+IMG_H, x_off:x_off+BINS] = img
            cv2.putText(canvas, vtitle, (x_off, y_offset - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        y_offset += IMG_H + 10

    cv2.putText(canvas, f"Ratio: {ratio:.2f}x | Trad: {len(pts_trad)} | Ours: {len(pts_dense)}",
               (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)

    cv2.imwrite(output_path, canvas)
    print(f"[对比图] 已保存: {output_path}")


def plot_depth_dist_cv(pts_trad, pts_dense, output_path):
    """深度分布对比"""
    W = 800
    H = 600
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    dims = [(2, 'Z (Depth)'), (0, 'X'), (1, 'Y')]
    row_h = H // 3

    for idx, (dim, label) in enumerate(dims):
        y_off = idx * row_h
        vrange = (
            min(np.percentile(pts_trad[:, dim], 1), np.percentile(pts_dense[:, dim], 1)),
            max(np.percentile(pts_trad[:, dim], 99), np.percentile(pts_dense[:, dim], 99))
        )
        counts_t, _ = np.histogram(pts_trad[:, dim], bins=80, range=vrange)
        counts_d, _ = np.histogram(pts_dense[:, dim], bins=80, range=vrange)
        max_c = max(counts_t.max(), counts_d.max(), 1)

        plot_h = row_h - 40
        bar_w = max((W - 100) // 80, 1)
        x_start = 80

        for i in range(80):
            bh_t = int(counts_t[i] / max_c * plot_h)
            bh_d = int(counts_d[i] / max_c * plot_h)
            x1 = x_start + i * bar_w
            cv2.rectangle(canvas, (x1, y_off + 20 + plot_h - bh_d),
                         (x1 + bar_w - 1, y_off + 20 + plot_h), (0, 0, 200), -1)
            cv2.rectangle(canvas, (x1, y_off + 20 + plot_h - bh_t),
                         (x1 + bar_w - 1, y_off + 20 + plot_h), (200, 100, 0), -1)

        cv2.putText(canvas, f"{label} Distribution",
                   (10, y_off + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.rectangle(canvas, (W-200, 10), (W-190, 25), (200, 100, 0), -1)
    cv2.putText(canvas, f"Traditional ({len(pts_trad)})", (W-185, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.rectangle(canvas, (W-200, 30), (W-190, 45), (0, 0, 200), -1)
    cv2.putText(canvas, f"Ours ({len(pts_dense)})", (W-185, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    cv2.imwrite(output_path, canvas)
    print(f"[分布图] 已保存: {output_path}")


def plot_camera_path_cv(ds, max_frames, output_path, K=None, ref_depth=None):
    """绘制GT相机轨迹"""
    n = min(ds.num_frames(), max_frames)
    gt_positions = np.array([ds.get_rt(i)[:3, 3] for i in range(n)])

    W, H = 800, 600
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    for view_idx, (view_name, dx, dy) in enumerate([("XY", 0, 1), ("XZ", 0, 2)]):
        lo_all = gt_positions[:, [dx, dy]].min(axis=0) - 1
        hi_all = gt_positions[:, [dx, dy]].max(axis=0) + 1
        rng = hi_all - lo_all
        rng[rng < 1e-6] = 1

        y_off = view_idx * (H // 2)

        # GT轨迹 (橙色)
        pts_2d = gt_positions[:, [dx, dy]]
        px = ((pts_2d - lo_all) / rng * np.array([W - 80, H // 2 - 40]) + 40).astype(int)
        px[:, 0] = np.clip(px[:, 0], 5, W - 5)
        px[:, 1] = y_off + np.clip(px[:, 1], 5, H // 2 - 5)
        for i in range(len(px) - 1):
            cv2.line(canvas, tuple(px[i]), tuple(px[i+1]), (0, 165, 255), 2)
        for p in px:
            cv2.circle(canvas, tuple(p), 3, (0, 165, 255), -1)

        cv2.putText(canvas, f"{view_name} View ({n} frames)",
                   (10, y_off + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 图例
    cv2.rectangle(canvas, (W-250, 10), (W-240, 25), (0, 165, 255), -1)
    cv2.putText(canvas, "GT Camera Trajectory", (W-235, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.imwrite(output_path, canvas)
    print(f"[轨迹图] 已保存: {output_path}")


# ============================================================
#  主函数
# ============================================================

def main():
    image_dir = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2"
    results_dir = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2"
    target_size = (640, 358)
    max_frames = 50
    base_dir = r"E:\MIS_TMI_Re_3D\compare_gt_poses"

    ds = DatasetLoader(image_dir, results_dir, target_size)
    K = _make_camera_intrinsics(target_size[0], target_size[1], fx=0.0)

    pose_est = TraditionalPoseEstimator(ds, K)
    dense_rec = Pseudo3DDenseReconstructor(ds, K, reproj_thresh=4.0)

    print("=" * 70)
    print("双管道对比 (GT c2w 直接变换, 无需尺度校正)")
    print("=" * 70)

    # ---- 管道1: Traditional SfM ----
    print(f"\n{'='*70}")
    print("管道1: Traditional SfM (GT c2w 直接变换)")
    print(f"{'='*70}\n")

    trad_dir = os.path.join(base_dir, "traditional")
    pts_trad, cols_trad = run_traditional_pipeline(
        ds, K, pose_est, dense_rec, max_frames, trad_dir)

    # ---- 管道2: Ours ----
    print(f"\n{'='*70}")
    print("管道2: Ours (GT c2w 直接变换 + 密集极线)")
    print(f"{'='*70}\n")

    ours_dir = os.path.join(base_dir, "ours")
    pts_ours, cols_ours = run_ours_pipeline(
        ds, K, pose_est, dense_rec, max_frames, ours_dir)

    if pts_trad is None or pts_ours is None:
        print("[错误] 某管道未生成点云!")
        return

    # ---- 可视化 ----
    print(f"\n{'='*70}")
    print("生成对比可视化")
    print(f"{'='*70}\n")

    vis_dir = os.path.join(base_dir, "visual")
    os.makedirs(vis_dir, exist_ok=True)

    plot_comparison_cv(pts_trad, pts_ours, os.path.join(vis_dir, "compare_scatter.png"))
    plot_depth_dist_cv(pts_trad, pts_ours, os.path.join(vis_dir, "compare_depth_dist.png"))
    plot_camera_path_cv(ds, max_frames, os.path.join(vis_dir, "camera_path.png"))

    # ---- 几何诊断 (纯Python PCA, 避免BLAS死锁) ----
    def _power_eigen(C, n_iter=80):
        v = np.array([1.0, 0.0, 0.0])
        for _ in range(n_iter):
            w = np.array([C[0,0]*v[0]+C[0,1]*v[1]+C[0,2]*v[2],
                          C[1,0]*v[0]+C[1,1]*v[1]+C[1,2]*v[2],
                          C[2,0]*v[0]+C[2,1]*v[1]+C[2,2]*v[2]])
            nrm = (w[0]**2 + w[1]**2 + w[2]**2)**0.5
            if nrm < 1e-12: break
            v = w / nrm
        lam = sum(v[i]*C[i,j]*v[j] for i in range(3) for j in range(3))
        return lam, v

    for label, pts in [("Traditional", pts_trad), ("Ours", pts_ours)]:
        centroid = pts.mean(axis=0)
        # 采样避免BLAS死锁
        n = len(pts)
        if n > 2000:
            idx = np.random.RandomState(42).choice(n, 2000, replace=False)
            Xc = pts[idx] - centroid[None, :]
            ns = 2000
        else:
            Xc = pts - centroid[None, :]
            ns = n
        # 协方差 (逐列点积)
        cov = np.zeros((3, 3), dtype=np.float64)
        for d1 in range(3):
            for d2 in range(d1, 3):
                cov[d1, d2] = float(np.dot(Xc[:, d1], Xc[:, d2])) / ns
                cov[d2, d1] = cov[d1, d2]
        # PCA: power iteration
        lam1, v1 = _power_eigen(cov)
        C2 = cov - lam1 * np.outer(v1, v1)
        lam2, v2 = _power_eigen(C2)
        C3 = C2 - lam2 * np.outer(v2, v2)
        lam3, _ = _power_eigen(C3)
        eigs = sorted([lam1, lam2, lam3], reverse=True)
        ratios = [e / max(eigs[0], 1e-8) for e in eigs]
        ratio_str = f'{ratios[0]:.2f}:{ratios[1]:.2f}:{ratios[2]:.2f}'
        is_cone = ratios[2] < 0.15
        # 方向角 (采样)
        if n > 2000:
            idx_z = np.random.RandomState(42).choice(n, 2000, replace=False)
            pts_s = pts[idx_z]
        else:
            pts_s = pts
        diff = pts_s - centroid[None, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        safe = dists > 1e-8
        cos_z = np.zeros(len(pts_s))
        cos_z[safe] = diff[safe, 2] / dists[safe]
        angles_z = np.degrees(np.arccos(np.clip(cos_z, -1, 1)))
        # 只计算有效点的统计
        angles_z_valid = angles_z[safe]
        print(f"\n  [{label}] 点数={len(pts)}")
        print(f"    质心=[{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]")
        print(f"    范围 X[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
              f"Y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}] "
              f"Z[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")
        print(f"    主成分比 {ratio_str}")
        print(f"    Z方向角 mean={angles_z_valid.mean():.1f}° std={angles_z_valid.std():.1f}°")
        print(f"    锥形检测: {'WARNING-锥形!' if is_cone else 'OK-非锥形'}")

    # ---- 总结 ----
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    ratio = len(pts_ours) / max(1, len(pts_trad))
    print(f"  Traditional SfM: {len(pts_trad)} 点")
    print(f"  Ours:            {len(pts_ours)} 点")
    print(f"  提升:            {ratio:.2f}x")
    print(f"\n  结果保存在: {base_dir}")
    print(f"    traditional/traditional_final.ply")
    print(f"    ours/ours_final.ply")
    print(f"    visual/compare_scatter.png")
    print(f"    visual/compare_depth_dist.png")


if __name__ == "__main__":
    main()
