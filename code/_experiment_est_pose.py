#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4关键实验：用"自估位姿"替换 GT pose，重跑棋盘格 Ours pipeline，报告 λ₃/λ₁。

动机：论文 Table 3 棋盘格实验中 Ours/Traditional 用 GT pose，而 7 个 baseline 自估 pose，
审稿人会质疑不公平。本实验用本 pipeline 的 E-matrix + recoverPose 链式累积得到自估 c2w，
替换 GT c2w 做世界坐标变换，看 λ₃/λ₁ 是否仍能保持 Normal 3D (>=0.25)。

注意：recoverPose 给出的 t 只有方向（单位向量），无绝对尺度。
本实验采用两种尺度恢复策略并都报告：
  (a) 单位尺度 (|t|=1)：检验纯几何形状（旋转结构是否保持）
  (b) 用 chessboard GT 轨迹的中位段长做尺度对齐（仅借尺度，不借旋转/方向）
这样既诚实又公平——baseline 的 Sim3 对齐也允许借尺度。

输出：compare_est_poses/ 下 ours_est_{mode}.ply + summary + λ₃/λ₁ 打印。
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover_true3d_sift_pseudo3d_v3 import (
    DatasetLoader, TraditionalPoseEstimator, Pseudo3DDenseReconstructor,
    _make_camera_intrinsics, _safe_mm, _write_ply,
)


# ---------- PCA λ₃/λ₁（与 _compare_chessboard_3way.py 一致）----------
def pca_ratio(pts, max_sample=5000, seed=42):
    rng = np.random.RandomState(seed)
    if len(pts) > max_sample:
        idx = rng.choice(len(pts), max_sample, replace=False)
        pts = pts[idx]
    centroid = pts.mean(axis=0)
    diff = pts - centroid[None, :]
    C = (diff.T @ diff) / max(len(pts), 1)
    # 对称矩阵 → eigh
    w = np.linalg.eigvalsh(C)[::-1]  # 降序
    w = np.maximum(w, 0)
    if w[0] < 1e-12:
        return 0.0, w
    return float(w[2] / w[0]), w


def shape_class(r):
    return "Cone" if r < 0.15 else ("Flat" if r < 0.25 else "Normal3D")


# ---------- 链式累积自估 c2w ----------
def accumulate_poses(rel_poses, scale_mode="unit", gt_step_lens=None):
    """
    rel_poses: list of (R 3x3, t 3,) for pairs (0->1, 1->2, ...), camera frame
               semantics: x_j = R @ x_i + t  (frame_i -> frame_j)
    返回 c2w_abs: list of 4x4, c2w_abs[k] 把 frame_k 相机坐标变到世界(frame0相机)坐标。

    相邻相对位姿 ΔT (frame_k -> frame_{k+1}) 作为 4x4:
        ΔT = [[R, t],[0,1]]  (x_{k+1}^{cam} = R x_k^{cam} + t 是错的；
        recoverPose 的 R,t 满足: x_j = R x_i + t  其中 x 是相机系点坐标。
        即 frame_j 相机系下点 = R (frame_i 相机系下同一点) + t。
        这意味着 c2w 累积: T_{w<-cam_j} = T_{w<-cam_i} @ ΔT^{-1} ... 需谨慎。)

    recoverPose 文档: points1 in cam1, points2 in cam2; 返回 R,t 使
        x2 = R x1 + t   (x1: cam1 系, x2: cam2 系, 同一 3D 点)
    所以 cam2 相对于 cam1 的位姿: 若 cam1=c2w1, cam2=c2w2, 则
        x_world = c2w1 @ x1 = c2w2 @ x2 = c2w2 @ (R x1 + t)
        => c2w1 = c2w2 @ R  且 0 = c2w2 @ t   =>  c2w2 = c2w1 @ R^{-1}, 但平移需补。
    标准做法 (cam_i -> cam_j 的 R,t 视作 T_{j<-i}):
        T_{j<-i} = [[R, t],[0,1]];  则 T_{w<-j} = T_{w<-i} @ inv(T_{j<-i}) ... 仍乱。

    采用最稳健约定：把 R,t 组装成 T_{j<-i}（j 系到 i 系的变换作用在齐次点上），
    即 [x_j;1] = T_{j<-i} [x_i;1]。则世界系下 cam_i 的位姿 c2w_i 满足
        c2w_j = c2w_i @ inv(T_{j<-i})^(-1) ... 此处直接用增量反向链：
    我们想要 cam 坐标 -> world 坐标 (c2w)。
    x_world = c2w_i @ x_i  ;  x_j = R x_i + t
    x_world = c2w_j @ x_j = c2w_j @ (R x_i + t) = c2w_j @ R @ x_i + c2w_j @ t
    与 c2w_i @ x_i 比较 => c2w_i = c2w_j @ R  (旋转), 0 = c2w_j @ t
    => c2w_j = c2w_i @ R^{-1},  且 c2w_j 的平移 = -c2w_j[:3,:3] @ (c2w_j 平移?) ...
    简化：cam_j 在世界系的位置/姿态。设 c2w_i = [R_i|c_i]。
      cam_j 相对 cam_i 的 R,t (recoverPose): R (cam_i->cam_j 旋转), t (cam_j 原点在 cam_i 系)
      实际 recoverPose 的 t 是 cam2 原点在 cam1 系的坐标*方向。
    工程上直接用:  c2w_{i+1} = c2w_i @ se3(R.T, -R.T @ t)
      因为 R,t 描述 cam2 在 cam1 系:  c2w2 = c2w1 @ T_{cam1<-cam2} = c2w1 @ [[R.T, -R.T@t],[0,1]]
    """
    n = len(rel_poses) + 1
    c2w = [np.eye(4, dtype=np.float64) for _ in range(n)]

    # 尺度
    if scale_mode == "unit":
        scales = np.ones(len(rel_poses))
    elif scale_mode == "gt_median" and gt_step_lens is not None:
        # 用 GT 相邻帧平移长度的中位数作为单位 t 的尺度
        med = float(np.median(gt_step_lens)) if len(gt_step_lens) else 1.0
        scales = np.full(len(rel_poses), med)
    else:
        scales = np.ones(len(rel_poses))

    for k, (R, t) in enumerate(rel_poses):
        t = np.asarray(t, dtype=np.float64).ravel()
        tn = np.linalg.norm(t)
        if tn < 1e-12:
            t_s = t.copy()
        else:
            t_s = t / tn * scales[k]  # 应用尺度
        # cam_{k+1} 在 cam_k 系: 方向 t_s (单位化后乘尺度)
        # T_{camk <- cam{k+1}} 的旋转 = R.T, 平移 = t_s
        T_k_j = np.eye(4, dtype=np.float64)
        T_k_j[:3, :3] = R.T
        T_k_j[:3, 3] = t_s
        c2w[k + 1] = c2w[k] @ T_k_j
    return c2w


def main():
    image_dir = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2"
    results_dir = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2"
    target_size = (640, 358)
    max_frames = 50
    base_dir = r"E:\MIS_TMI_Re_3D\compare_est_poses"
    os.makedirs(base_dir, exist_ok=True)

    ds = DatasetLoader(image_dir, results_dir, target_size)
    K = _make_camera_intrinsics(target_size[0], target_size[1], fx=0.0)
    pose_est = TraditionalPoseEstimator(ds, K)
    dense_rec = Pseudo3DDenseReconstructor(ds, K, reproj_thresh=4.0)
    n = min(ds.num_frames(), max_frames)

    print("=" * 70)
    print(f"问题4实验: 自估位姿 ({n} 帧) -> Ours pipeline -> λ₃/λ₁")
    print("=" * 70)

    # ---- Step 1: 估计所有相邻相对位姿 ----
    print("\n[Step1] 估计相邻相对位姿 (E-matrix + recoverPose)...")
    rel_poses = []
    gt_step_lens = []
    failed = []
    t0 = time.time()
    for i in range(n - 1):
        result = pose_est.estimate_pose(i, i + 1)
        if result is None:
            failed.append(i)
            # 失败帧：用单位相对位姿占位（避免链断裂）
            rel_poses.append((np.eye(3, dtype=np.float64), np.array([0., 0., 1e-6])))
        else:
            R, t, mask, stats = result
            rel_poses.append((R, t))
        # GT 相邻段长（仅用于尺度对齐策略 b）
        try:
            gt_i = ds.get_rt(i)
            gt_j = ds.get_rt(i + 1)
            gt_step_lens.append(float(np.linalg.norm(gt_j[:3, 3] - gt_i[:3, 3])))
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n-1} 对完成, 失败 {len(failed)}")
    print(f"  完成. 成功 {len(rel_poses)-len(failed)}/{n-1}, 失败帧={failed}, 耗时 {time.time()-t0:.1f}s")

    # ---- Step 2: 两种尺度模式累积 c2w ----
    results = {}
    for mode in ["unit", "gt_median"]:
        print(f"\n[Step2] 累积自估 c2w (scale_mode={mode})...")
        c2w_abs = accumulate_poses(rel_poses, scale_mode=mode,
                                   gt_step_lens=gt_step_lens if mode == "gt_median" else None)

        # ---- Step 3: Ours pipeline 世界坐标变换 (用自估 c2w) ----
        print(f"[Step3] Ours pipeline 点云生成 (自估 c2w={mode})...")
        all_pts, all_cols = [], []
        t0 = time.time()
        for i in range(n - 1):
            result = pose_est.estimate_pose(i, i + 1)
            if result is None:
                continue
            R_rel, t_rel, inlier_mask, _ = result
            try:
                grid_i = ds.load_mesh_grid(i)
                grid_j = ds.load_mesh_grid(i + 1)
            except Exception:
                continue
            img_i = ds.load_image(i)

            # 稀疏参考深度
            pts1_all, pts2_all = pose_est._match_features(i, i + 1)
            trad_pts = np.zeros((0, 3)); trad_cols = np.zeros((0, 3), dtype=np.uint8)
            trad_px = np.zeros((0, 2)); ref_depth = None
            if len(pts1_all) >= 8:
                pts1_in = pts1_all[inlier_mask]; pts2_in = pts2_all[inlier_mask]
                if len(pts1_in) >= 5:
                    tri = dense_rec.triangulate_with_known_pose(pts1_in, pts2_in, R_rel, t_rel)
                    if tri is not None and len(tri[0]) >= 3:
                        X3d_t, pts1_ok_t, _, _ = tri
                        trad_pts = X3d_t; trad_px = pts1_ok_t
                        ref_depth = float(np.median(X3d_t[:, 2]))
                        if img_i is not None:
                            xs = np.clip(np.round(pts1_ok_t[:, 0]).astype(int), 0, img_i.shape[1]-1)
                            ys = np.clip(np.round(pts1_ok_t[:, 1]).astype(int), 0, img_i.shape[0]-1)
                            trad_cols = img_i[ys, xs, :]

            # 密集极线
            pts1_e, pts2_e, X3d_e, cols_e = dense_rec.match_dense_epipolar(
                grid_i, grid_j, R_rel, t_rel, img_i=img_i, ref_depth_median=ref_depth)
            n_e = len(pts1_e)

            # 去重
            if n_e > 0 and len(trad_pts) > 0:
                keep = np.ones(n_e, dtype=bool)
                for b in range(0, n_e, 500):
                    be = min(b + 500, n_e)
                    d = np.linalg.norm(pts1_e[b:be, None, :] - trad_px[None, :, :], axis=2)
                    keep[b:be] = d.min(axis=1) > 3.0
                X3d_f = X3d_e[keep]; cols_f = cols_e[keep] if len(cols_e) > 0 else cols_e
            else:
                X3d_f = X3d_e; cols_f = cols_e

            if len(trad_pts) > 0 and len(X3d_f) > 0:
                X_m = np.vstack([trad_pts, X3d_f]); C_m = np.vstack([trad_cols, cols_f])
            elif len(trad_pts) > 0:
                X_m = trad_pts; C_m = trad_cols
            elif len(X3d_f) > 0:
                X_m = X3d_f; C_m = cols_f
            else:
                continue

            # cam_i 坐标 -> 世界 (用自估 c2w 替换 GT)
            T_c2w = c2w_abs[i]
            R_i = T_c2w[:3, :3]; t_i = T_c2w[:3, 3:4]
            X_world = (_safe_mm(R_i, X_m.T) + t_i).T
            all_pts.append(X_world.astype(np.float32))
            all_cols.append(C_m.astype(np.uint8))

        if not all_pts:
            print(f"  [{mode}] 无点云!")
            results[mode] = None
            continue

        merged = np.vstack(all_pts)
        merged_cols = np.vstack(all_cols)
        # 体素降采样 + 离群
        if len(merged) > 10:
            vs = 0.5
            vid = np.floor(merged / vs).astype(np.int64)
            flat = vid[:, 0]*100000000 + vid[:, 1]*10000 + vid[:, 2]
            _, fi = np.unique(flat, return_index=True)
            merged, merged_cols = merged[fi], merged_cols[fi]
        if len(merged) > 100:
            cen = np.median(merged, axis=0)
            d = np.linalg.norm(merged - cen[None], axis=1)
            inl = d < np.median(d) * 5.0
            merged, merged_cols = merged[inl], merged_cols[inl]

        ply = os.path.join(base_dir, f"ours_est_{mode}.ply")
        _write_ply(merged, merged_cols, ply)
        r, w = pca_ratio(merged)
        results[mode] = (r, len(merged), shape_class(r), w)
        print(f"  [{mode}] 点数={len(merged)}, λ₃/λ₁={r:.4f}, shape={shape_class(r)}, 耗时={time.time()-t0:.1f}s")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("问题4实验汇总: 自估位姿下的 3D 形状质量")
    print("=" * 70)
    print(f"{'模式':<18}{'点数':>8}{'λ₃/λ₁':>10}{'形状':>12}")
    print("-" * 50)
    for mode in ["unit", "gt_median"]:
        if results.get(mode):
            r, np_, sh, _ = results[mode]
            print(f"{mode:<18}{np_:>8}{r:>10.4f}{sh:>12}")
        else:
            print(f"{mode:<18}{'FAIL':>8}")
    print("-" * 50)
    print("参考: GT pose 下 Ours λ₃/λ₁ ≈ 0.42 (Normal3D); 阈值 0.25=Normal3D")
    print("结论: 若自估 pose 下 λ₃/λ₁ >= 0.25 => heightfield pipeline 在自估 pose 下成立")

    # 保存汇总
    with open(os.path.join(base_dir, "summary_est_pose.txt"), 'w') as f:
        f.write("问题4实验: 自估位姿 (E-matrix+recoverPose 链式累积) 下棋盘格 Ours pipeline λ₃/λ₁\n")
        f.write("=" * 60 + "\n")
        f.write(f"帧数: {n}, 相邻对: {n-1}, 失败帧: {failed}\n\n")
        for mode in ["unit", "gt_median"]:
            if results.get(mode):
                r, np_, sh, w = results[mode]
                f.write(f"scale_mode={mode}: 点数={np_}, λ₃/λ₁={r:.4f}, shape={sh}\n")
                f.write(f"  eigenvalues: {w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}\n")
        f.write(f"\n参考 GT pose 下: λ₃/λ₁≈0.42 (Normal3D)\n")
    print(f"\n已保存: {os.path.join(base_dir, 'summary_est_pose.txt')}")


if __name__ == "__main__":
    main()
