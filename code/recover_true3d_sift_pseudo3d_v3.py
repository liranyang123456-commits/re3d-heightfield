#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_true3d_sift_pseudo3d.py  —  传统SfM + 伪3D几何特征融合 真实3D稠密重建 v3
================================================================================

v3 核心改进（相比v2）：
  问题诊断：
    - v2将P3D和传统匹配混合做E-RANSAC，P3D匹配被E约束错误过滤，损失57.6%
    - 21/49帧对E内点率<20%，最低3.9%
    - grid_step=8导致P3D匹配本身就稀疏

  解决方案（两阶段分离架构）：
    Stage 1 ─ 传统SIFT+ORB精确位姿恢复 (POSE ONLY)
      1. 仅使用SIFT+ORB传统匹配做E-matrix RANSAC
      2. findEssentialMat + recoverPose → 精确R, t
      3. 链式累积E-pose → 世界坐标相机轨迹
      → 保证位姿精度，不受伪3D匹配干扰

    Stage 2 ─ 伪3D稠密匹配 + 已知位姿三角化 (DENSE RECONSTRUCTION)
      1. 更密的网格采样 (grid_step=2, v2=8, 密度提升16倍)
      2. 伪3D RT引导的7D几何特征匹配 → 稠密像素对应关系
      3. 用Stage 1的R, t直接三角化（无需E-RANSAC过滤）
      4. 仅做cheirality + reprojection几何验证
      → P3D匹配不再被E-RANSAC错误过滤，大幅提升点云密度

    Stage 3 ─ 多帧融合 + 世界坐标系对齐
      1. E-accumulated世界坐标融合
      2. Voxel降采样 + 离群点移除
      3. 相机轨迹可视化

输入：
  原始RGB:  Lines_Photo _2/photo_XXX.jpg  (1280×720 → 缩放到640×358)
  PLY网格:  Line_Photo_2_results2/mesh/debug_mesh_XXXX.ply
  RT矩阵:   Line_Photo_2_results2/estimated_poses_txt/pose_XXXX.txt
  位姿格式:  cam2world (c2w), 即 X_world = T @ X_cam, 相机位置 = T[:3,3]

输出：
  <output_dir>/
  ├── 01_pose_recovery/       ← Stage 1 位姿恢复统计
  ├── 02_dense_p3d/           ← Stage 2 稠密P3D三角化
  ├── 03_fused/               ← Stage 3 融合点云+轨迹
  └── per_pair_stats.csv      ← 逐对统计
"""

from __future__ import annotations

# !!! 必须在import numpy之前设置，防止BLAS线程死锁 !!!
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import re
import sys
import struct
import argparse
import traceback
import csv
import time
from typing import Optional

import numpy as np
import cv2


def _matmul_4x4(A, B):
    """安全的4x4矩阵乘法，避免numpy BLAS死锁"""
    C = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            C[i, j] = A[i, 0]*B[0, j] + A[i, 1]*B[1, j] + A[i, 2]*B[2, j] + A[i, 3]*B[3, j]
    return C


def _safe_mm(A, B):
    """安全矩阵乘法(A@B)，避免Windows numpy BLAS崩溃。
    仅用于小矩阵(3x3, 3x4, 3x1等)。大矩阵使用cv2.gemm。"""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        # 降级到np.dot
        return np.dot(A, B)
    m, k1 = A.shape
    k2, n = B.shape
    k = min(k1, k2)
    # 小矩阵用手动循环
    if m <= 4 and n <= 4 and k <= 4:
        C = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for kk in range(k):
                    s += A[i, kk] * B[kk, j]
                C[i, j] = s
        return C
    # 大矩阵用cv2.gemm (不依赖numpy BLAS)
    try:
        return cv2.gemm(A, B, 1.0, None, 0.0)
    except Exception:
        return np.dot(A, B)


def _inv_3x3(M):
    """手动计算3x3矩阵逆，避免np.linalg.inv的BLAS崩溃"""
    a, b, c = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
    d, e, f = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
    g, h, i = float(M[2, 0]), float(M[2, 1]), float(M[2, 2])
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    if abs(det) < 1e-12:
        raise np.linalg.LinAlgError("Singular matrix")
    inv_det = 1.0 / det
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = (e*i - f*h) * inv_det
    R[0, 1] = (c*h - b*i) * inv_det
    R[0, 2] = (b*f - c*e) * inv_det
    R[1, 0] = (f*g - d*i) * inv_det
    R[1, 1] = (a*i - c*g) * inv_det
    R[1, 2] = (c*d - a*f) * inv_det
    R[2, 0] = (d*h - e*g) * inv_det
    R[2, 1] = (b*g - a*h) * inv_det
    R[2, 2] = (a*e - b*d) * inv_det
    return R

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
#  二进制PLY解析器 (向量化)
# ============================================================

def _parse_ply_binary(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """解析 binary_little_endian PLY。返回 vertices(N,9), faces(M,3)。"""
    NP_DTYPE = {'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2',
                'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8'}
    COL_MAP = {'x': 0, 'y': 1, 'z': 2, 'nx': 3, 'ny': 4, 'nz': 5,
               'red': 6, 'green': 7, 'blue': 8}

    with open(filepath, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line.decode('utf-8', errors='replace').rstrip('\n'))
            if line.strip() == b'end_header':
                break

        n_verts = 0; n_faces = 0
        props = []
        in_vertex = False
        for line in header_lines:
            line = line.strip()
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1]); in_vertex = True
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1]); in_vertex = False
            elif line.startswith('property') and in_vertex:
                parts = line.split()
                props.append((parts[2], parts[1]))

        dt_fields = [(name, NP_DTYPE.get(dtype, 'f4')) for name, dtype in props]
        vert_dt = np.dtype(dt_fields)
        raw = np.frombuffer(f.read(n_verts * vert_dt.itemsize), dtype=vert_dt, count=n_verts)

        vertices = np.zeros((n_verts, 9), dtype=np.float64)
        for name, _ in props:
            col = COL_MAP.get(name)
            if col is not None:
                vertices[:, col] = raw[name].astype(np.float64)

        faces = np.zeros((n_faces, 3), dtype=np.int64)
        if n_faces > 0:
            face_bytes = f.read()
            for i in range(n_faces):
                offset = i * 13
                if offset + 13 > len(face_bytes):
                    break
                if face_bytes[offset] == 3:
                    faces[i] = struct.unpack_from('<III', face_bytes, offset + 1)

        return vertices, faces


def _build_grid_from_mesh(vertices: np.ndarray, H: int, W: int) -> dict:
    """从mesh顶点构建规则网格查找表。向量化版本。"""
    u = vertices[:, 0].astype(np.int64)
    v = vertices[:, 1].astype(np.int64)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid]; v = v[valid]
    verts = vertices[valid]

    height_map = np.full((H, W), np.nan, dtype=np.float64)
    normal_map = np.full((H, W, 3), np.nan, dtype=np.float64)
    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    flat_idx = v * W + u
    _, first_idx = np.unique(flat_idx, return_index=True)
    first_idx = first_idx[np.argsort(first_idx)]

    fi_v = v[first_idx]
    fi_u = u[first_idx]
    height_map[fi_v, fi_u] = verts[first_idx, 2]
    normal_map[fi_v, fi_u] = verts[first_idx, 3:6]
    color_map[fi_v, fi_u] = verts[first_idx, 6:9].astype(np.uint8)

    return {
        'height': height_map,
        'normal': normal_map,
        'color': color_map,
        'valid_mask': ~np.isnan(height_map),
        'verts': verts,
        'u': u, 'v': v,
    }


# ============================================================
#  工具函数
# ============================================================

def _make_camera_intrinsics(W: int, H: int, fx: float = 0.0) -> np.ndarray:
    if fx <= 0:
        fx = float(np.sqrt(W * W + H * H))
    return np.array([
        [fx, 0.0, float(W) / 2.0],
        [0.0, fx, float(H) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _parse_rt_per_frame_txt(txt_dir: str) -> dict[int, np.ndarray]:
    """解析 estimated_poses_txt/pose_XXXX.txt → {frame_idx: 4x4}"""
    result: dict[int, np.ndarray] = {}
    if not os.path.isdir(txt_dir):
        return result
    for fn in sorted(os.listdir(txt_dir)):
        m = re.match(r'pose_(\d+)\.txt', fn)
        if not m:
            continue
        frame_idx = int(m.group(1))
        full = os.path.join(txt_dir, fn)
        try:
            mat = np.loadtxt(full, dtype=np.float64)
            if mat.shape == (4, 4):
                result[frame_idx] = mat
        except Exception:
            continue
    return result


def _write_ply(pts: np.ndarray, cols: np.ndarray, path: str):
    """写入PLY文件。向量化版本。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    has_c = cols is not None and len(cols) == len(pts) and cols.shape[1] >= 3
    n = len(pts)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_c:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_c:
            c = np.asarray(cols, dtype=np.uint8).reshape(-1, 3)
            lines = (np.char.mod("%.6f", pts[:, 0]) + " " +
                     np.char.mod("%.6f", pts[:, 1]) + " " +
                     np.char.mod("%.6f", pts[:, 2]) + " " +
                     c[:, 0].astype(str) + " " +
                     c[:, 1].astype(str) + " " +
                     c[:, 2].astype(str))
        else:
            lines = (np.char.mod("%.6f", pts[:, 0]) + " " +
                     np.char.mod("%.6f", pts[:, 1]) + " " +
                     np.char.mod("%.6f", pts[:, 2]))
        f.write("\n".join(lines) + "\n")


# ============================================================
#  数据加载器
# ============================================================

class DatasetLoader:
    """加载原始RGB + PLY网格 + RT矩阵"""

    def __init__(self, image_dir: str, results_dir: str, target_size: tuple = (640, 358)):
        self.image_dir = image_dir
        self.results_dir = results_dir
        self.target_W, self.target_H = target_size
        self._image_files: list[str] = []
        self._mesh_files: list[str] = []
        self._rt_matrices: dict[int, np.ndarray] = {}
        self._mesh_cache: dict[int, dict] = {}
        self._detect()

    def _detect(self):
        img_dir = self.image_dir
        if os.path.isdir(img_dir):
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self._image_files.append(os.path.join(img_dir, fn))

        mesh_dir = os.path.join(self.results_dir, "mesh")
        if os.path.isdir(mesh_dir):
            for fn in sorted(os.listdir(mesh_dir)):
                if fn.endswith('.ply'):
                    self._mesh_files.append(os.path.join(mesh_dir, fn))

        poses_dir = os.path.join(self.results_dir, "estimated_poses_txt")
        if os.path.isdir(poses_dir):
            self._rt_matrices = _parse_rt_per_frame_txt(poses_dir)

        print(f"[数据] RGB图像:    {len(self._image_files)} 帧")
        print(f"[数据] PLY网格:    {len(self._mesh_files)} 个")
        print(f"[数据] RT矩阵:     {len(self._rt_matrices)} 帧")
        print(f"[数据] 目标尺寸:   {self.target_W}x{self.target_H}")

    def load_image(self, frame_idx: int) -> Optional[np.ndarray]:
        img_idx = frame_idx + 1
        if img_idx < 1 or img_idx > len(self._image_files):
            return None
        path = self._image_files[img_idx - 1]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.shape[1] != self.target_W or img.shape[0] != self.target_H:
            img = cv2.resize(img, (self.target_W, self.target_H), interpolation=cv2.INTER_AREA)
        return img

    def load_mesh_grid(self, frame_idx: int) -> dict:
        if frame_idx in self._mesh_cache:
            return self._mesh_cache[frame_idx]
        ply_path = ""
        for f in self._mesh_files:
            nums = re.findall(r'\d+', os.path.basename(f))
            if nums and int(nums[-1]) == frame_idx:
                ply_path = f
                break
        if not ply_path:
            raise FileNotFoundError(f"未找到帧 {frame_idx} 的PLY文件")
        verts, _ = _parse_ply_binary(ply_path)
        grid = _build_grid_from_mesh(verts, self.target_H, self.target_W)
        self._mesh_cache[frame_idx] = grid
        return grid

    def get_rt(self, frame_idx: int) -> np.ndarray:
        return self._rt_matrices.get(frame_idx, np.eye(4, dtype=np.float64))

    def num_frames(self) -> int:
        return len(self._image_files)


# ============================================================
#  Stage 1: 传统SIFT+ORB精确位姿恢复 (POSE ONLY)
# ============================================================

class TraditionalPoseEstimator:
    """
    仅使用传统SIFT+ORB特征匹配恢复相机精确位姿。

    与v2的区别：
    - 不混入伪3D匹配，避免E-RANSAC被污染
    - 更高的RANSAC置信度
    - 返回(R, t)供Stage 2使用
    """

    def __init__(
        self,
        dataset: DatasetLoader,
        K: np.ndarray,
        sift_nfeatures: int = 2000,
        orb_nfeatures: int = 2000,
        ratio_thresh: float = 0.75,
        ransac_thresh: float = 2.0,
        ransac_prob: float = 0.9999,
        min_inlier_ratio: float = 0.15,
    ):
        self.ds = dataset
        self.K = K
        self.ratio_thresh = ratio_thresh
        self.ransac_thresh = ransac_thresh
        self.ransac_prob = ransac_prob
        self.min_inlier_ratio = min_inlier_ratio

        self._sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
        self._orb = cv2.ORB_create(nfeatures=orb_nfeatures,
                                    scaleFactor=1.2, nlevels=8,
                                    edgeThreshold=15, patchSize=31,
                                    fastThreshold=10, WTA_K=2)
        self._bf_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._cache: dict[int, dict] = {}

    def _load_gradient_gray(self, frame_idx: int) -> Optional[np.ndarray]:
        grad_path = os.path.join(self.ds.results_dir,
                                 f"gradient_before_threshold_{frame_idx:04d}.jpg")
        if os.path.exists(grad_path):
            gray = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
            if gray is not None and gray.shape[:2] == (self.ds.target_H, self.ds.target_W):
                return gray
        img = self.ds.load_image(frame_idx)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return None

    def _extract_all(self, frame_idx: int) -> dict:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        gray = self._load_gradient_gray(frame_idx)
        if gray is None:
            self._cache[frame_idx] = {"sift_kp": [], "sift_desc": None,
                                      "orb_kp": [], "orb_desc": None}
            return self._cache[frame_idx]
        sift_kp, sift_desc = self._sift.detectAndCompute(gray, None)
        orb_kp, orb_desc = self._orb.detectAndCompute(gray, None)
        self._cache[frame_idx] = {
            "sift_kp": sift_kp if sift_kp else [],
            "sift_desc": sift_desc,
            "orb_kp": orb_kp if orb_kp else [],
            "orb_desc": orb_desc,
        }
        return self._cache[frame_idx]

    def _match_features(self, frame_i: int, frame_j: int
                        ) -> tuple[np.ndarray, np.ndarray]:
        """SIFT + ORB 联合匹配，返回像素对应关系。"""
        feat_i = self._extract_all(frame_i)
        feat_j = self._extract_all(frame_j)

        all_pts1, all_pts2 = [], []

        # SIFT匹配 (Lowe's ratio test)
        if feat_i["sift_desc"] is not None and feat_j["sift_desc"] is not None:
            if len(feat_i["sift_desc"]) >= 2 and len(feat_j["sift_desc"]) >= 2:
                try:
                    matches = self._bf_l2.knnMatch(feat_i["sift_desc"], feat_j["sift_desc"], k=2)
                    for m, n in matches:
                        if m.distance < self.ratio_thresh * n.distance:
                            all_pts1.append(feat_i["sift_kp"][m.queryIdx].pt)
                            all_pts2.append(feat_j["sift_kp"][m.trainIdx].pt)
                except Exception:
                    pass

        # ORB匹配 (Hamming + ratio test)
        if feat_i["orb_desc"] is not None and feat_j["orb_desc"] is not None:
            if len(feat_i["orb_desc"]) >= 2 and len(feat_j["orb_desc"]) >= 2:
                try:
                    matches = self._bf_hamming.knnMatch(feat_i["orb_desc"], feat_j["orb_desc"], k=2)
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            all_pts1.append(feat_i["orb_kp"][m.queryIdx].pt)
                            all_pts2.append(feat_j["orb_kp"][m.trainIdx].pt)
                except Exception:
                    pass

        if len(all_pts1) < 8:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.array(all_pts1, dtype=np.float64), np.array(all_pts2, dtype=np.float64)

    def estimate_pose(self, frame_i: int, frame_j: int
                      ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
        """
        传统特征匹配 → E-matrix RANSAC → 精确位姿。

        Returns:
            R, t, inlier_mask, stats  or None
        """
        pts1, pts2 = self._match_features(frame_i, frame_j)
        n_matches = len(pts1)
        if n_matches < 15:
            return None

        K = self.K

        # findEssentialMat (更高置信度)
        try:
            E, mask_e = cv2.findEssentialMat(
                pts1, pts2, cameraMatrix=K,
                method=cv2.RANSAC, prob=self.ransac_prob,
                threshold=self.ransac_thresh,
            )
        except Exception:
            return None
        if E is None or E.shape != (3, 3):
            return None

        # recoverPose
        try:
            n_in, R, t, mask_pose = cv2.recoverPose(
                E, pts1, pts2, cameraMatrix=K, mask=mask_e)
        except Exception:
            return None

        # 最低内点数检查
        if n_in < max(8, int(n_matches * self.min_inlier_ratio)):
            return None

        inlier_mask = mask_pose.ravel().astype(bool) if mask_pose is not None else np.ones(n_matches, dtype=bool)
        t = t.ravel()

        e_ratio = n_in / n_matches
        stats = {
            "n_matches": n_matches,
            "E_inliers": int(n_in),
            "E_inlier_ratio": round(e_ratio, 3),
        }

        return R, t, inlier_mask, stats


# ============================================================
#  Stage 2: 伪3D稠密匹配 + 已知位姿三角化
# ============================================================

class Pseudo3DDenseReconstructor:
    """
    伪3D稠密几何特征匹配 + 已知位姿三角化。

    支持两种匹配模式:
    - 7D特征匹配 (旧方法): match_frames_dense()
    - 密集极线匹配 (新方法): match_dense_epipolar()
    
    密集极线匹配是核心改进:
    - 用SfM的RT计算基础矩阵F和极线约束
    - 沿极线带宽搜索候选匹配点
    - P3D法线+颜色作为匹配代价
    - 向量化实现，每帧对~0.6秒
    """

    def __init__(
        self,
        dataset: DatasetLoader,
        K: np.ndarray,
        grid_step: int = 6,
        search_radius: float = 25.0,
        feature_match_thresh: float = 1.0,
        reproj_thresh: float = 5.0,
        cross_check_px: float = 3.0,
        use_epipolar: bool = True,
        epipolar_width: float = 5.0,
        # 密集极线匹配参数
        epipolar_band: int = 2,
        normal_thresh: float = 0.15,
        color_weight: float = 0.2,
        # 深度过滤参数
        max_depth_ratio: float = 5.0,
        min_disparity: float = 1.0,
    ):
        self.ds = dataset
        self.K = K
        self.grid_step = grid_step
        self.search_radius = search_radius
        self._feat_thresh = feature_match_thresh
        self.reproj_thresh = reproj_thresh
        self._cross_check_px = cross_check_px
        self.use_epipolar = use_epipolar
        self._epipolar_width = epipolar_width
        self._epipolar_band = epipolar_band
        self._normal_thresh = normal_thresh
        self._color_weight = color_weight
        self._max_depth_ratio = max_depth_ratio
        self._min_disparity = min_disparity

    @staticmethod
    def _extract_geometric_features(grid: dict) -> np.ndarray:
        """从伪3D网格提取7D几何特征。[h_norm, nx, ny, nz, r, g, b]"""
        H, W = grid['height'].shape
        features = np.full((H, W, 7), np.nan, dtype=np.float32)
        mask = grid['valid_mask']
        h = grid['height'][mask]
        h_range = np.nanmax(h) - np.nanmin(h)
        h_norm = (h - np.nanmin(h)) / max(h_range, 1e-6)
        features[mask, 0] = h_norm
        features[mask, 1:4] = grid['normal'][mask].astype(np.float32)
        features[mask, 4:7] = grid['color'][mask].astype(np.float32) / 255.0
        return features

    @staticmethod
    def _relative_pose_from_abs(T1: np.ndarray, T2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从伪3D绝对位姿(cam2world)计算相对位姿

        GT位姿为cam2world(c2w)格式: X_world = T @ X_cam
        推导:
            X_world = R1 @ X_cam1 + t1
            X_world = R2 @ X_cam2 + t2
            => X_cam2 = R2^T @ R1 @ X_cam1 + R2^T @ (t1 - t2)
        返回 (R, t) 使得 X_cam2 = R @ X_cam1 + t
        """
        R1, t1 = T1[:3, :3], T1[:3, 3:4]
        R2, t2 = T2[:3, :3], T2[:3, 3:4]
        R = _safe_mm(R2.T, R1)
        t = _safe_mm(R2.T, (t1 - t2)).ravel()
        return R, t

    @staticmethod
    def _compute_fundamental(R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
        """从已知R, t, K计算基础矩阵F = K^(-T) @ [t]_x @ R @ K^(-1)"""
        K_inv = _inv_3x3(K)
        # [t]_x 叉积矩阵
        t_skew = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0],
        ], dtype=np.float64)
        E = _safe_mm(t_skew, R)
        F = _safe_mm(_safe_mm(K_inv.T, E), K_inv)
        return F

    def match_dense_epipolar(
        self, grid_i: dict, grid_j: dict,
        R: np.ndarray, t: np.ndarray,
        img_i: Optional[np.ndarray] = None,
        ref_depth_median: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化密集极线匹配 + P3D法线/颜色验证。

        核心思路: 用SfM的RT计算基础矩阵F和极线约束，
        沿极线带宽搜索候选匹配点，P3D法线+颜色作为匹配代价。

        Args:
            grid_i, grid_j: P3D mesh网格
            R, t: SfM相对位姿 (帧i到帧j)
            img_i: 帧i原始RGB图像 (用于颜色采样, 优先于P3D网格颜色)
            ref_depth_median: 参考深度中值 (来自传统匹配, 用于深度过滤)

        Returns:
            pts1: (M,2), pts2: (M,2), X3d: (M,3), colors: (M,3) uint8
        """
        H, W = grid_i['height'].shape
        K_inv = _inv_3x3(self.K)
        grid_step = self.grid_step
        normal_thresh = self._normal_thresh
        color_weight = self._color_weight
        reproj_thresh = self.reproj_thresh
        epipolar_band = self._epipolar_band

        # ---- 基础矩阵 F = K^{-T} [t]_x R K^{-1} ----
        t_skew = np.array([
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0],
        ], dtype=np.float64)
        E = _safe_mm(t_skew, R)
        F = _safe_mm(_safe_mm(K_inv.T, E), K_inv)

        # ---- 投影矩阵 ----
        P1 = _safe_mm(self.K, np.hstack([np.eye(3), np.zeros((3, 1))]))
        P2 = _safe_mm(self.K, np.hstack([R, t.reshape(3, 1)]))

        # ---- 帧i采样像素 ----
        mask_i = grid_i['valid_mask']
        mask_j = grid_j['valid_mask']
        normal_i = grid_i['normal']
        normal_j = grid_j['normal']
        color_i = grid_i['color']
        color_j = grid_j['color']

        ys_i, xs_i = np.where(mask_i)
        sample = (ys_i % grid_step == 0) & (xs_i % grid_step == 0)
        ys_s = ys_i[sample]
        xs_s = xs_i[sample]
        N = len(ys_s)

        if N < 8:
            return (np.empty((0, 2)), np.empty((0, 2)),
                    np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))

        # ---- 帧i的法线 (变换到帧j坐标系) ----
        ni = normal_i[ys_s, xs_s]  # (N, 3)
        ni_valid = ~np.any(np.isnan(ni), axis=1)
        ni_norm = np.zeros_like(ni)
        ni_norm[ni_valid] = ni[ni_valid] / (np.linalg.norm(ni[ni_valid], axis=1, keepdims=True) + 1e-12)
        ni_in_j = np.zeros_like(ni_norm)
        for idx in range(N):
            if ni_valid[idx]:
                ni_in_j[idx] = _safe_mm(R, ni_norm[idx].reshape(3, 1)).ravel()

        # ---- 帧i的颜色 ----
        ci = color_i[ys_s, xs_s].astype(np.float32) / 255.0

        # ---- 计算极线 ----
        pts_i_h = np.stack([xs_s.astype(np.float64),
                            ys_s.astype(np.float64),
                            np.ones(N, dtype=np.float64)], axis=0)  # (3, N)
        epi_lines = cv2.gemm(F, pts_i_h, 1.0, None, 0.0)  # (3, N)

        a_epi = epi_lines[0]  # (N,)
        b_epi = epi_lines[1]  # (N,)
        c_epi = epi_lines[2]  # (N,)

        # ---- 帧j法线预处理 ----
        nj_valid_map = mask_j & ~np.any(np.isnan(normal_j), axis=2)
        nj_norm_map = np.zeros_like(normal_j)
        valid_flat = nj_valid_map.ravel()
        nj_flat = normal_j.reshape(-1, 3)
        nj_norm_flat = nj_flat.copy()
        valid_idx = np.where(valid_flat)[0]
        if len(valid_idx) > 0:
            norms = np.linalg.norm(nj_flat[valid_idx], axis=1, keepdims=True)
            nj_norm_flat[valid_idx] = nj_flat[valid_idx] / (norms + 1e-12)
        nj_norm_map = nj_norm_flat.reshape(H, W, 3)

        # 帧j颜色归一化
        cj_norm = color_j.astype(np.float32) / 255.0

        # ---- 沿极线搜索 ----
        best_xj = np.full(N, -1, dtype=np.int32)
        best_yj = np.full(N, -1, dtype=np.int32)
        best_score = np.full(N, -1.0, dtype=np.float64)

        for vj in range(H):
            u_on_epi = -(b_epi * vj + c_epi) / (a_epi + 1e-12)
            u_center = np.round(u_on_epi).astype(np.int32)

            for du in range(-epipolar_band, epipolar_band + 1):
                u_int = u_center + du

                valid_bounds = (u_int >= 0) & (u_int < W)
                u_clipped = np.clip(u_int, 0, W - 1)
                valid_j = nj_valid_map[vj, u_clipped]

                valid = valid_bounds & valid_j & ni_valid
                if not valid.any():
                    continue

                nj_cand = nj_norm_map[vj, u_clipped]
                cj_cand = cj_norm[vj, u_clipped]

                cos_sim = np.abs(np.sum(ni_in_j * nj_cand, axis=1))
                color_diff = np.linalg.norm((ci - cj_cand).astype(np.float32), axis=1)
                color_sim = 1.0 / (1.0 + color_diff)

                score = cos_sim * (1.0 - color_weight) + color_sim * color_weight

                better = valid & (score > best_score) & (cos_sim >= normal_thresh)
                best_xj[better] = u_int[better]
                best_yj[better] = vj
                best_score[better] = score[better]

        # ---- 收集有效匹配 ----
        valid_match = best_xj >= 0
        if valid_match.sum() < 4:
            return (np.empty((0, 2)), np.empty((0, 2)),
                    np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))

        pt1_x = xs_s[valid_match].astype(np.float64)
        pt1_y = ys_s[valid_match].astype(np.float64)
        pt2_x = best_xj[valid_match].astype(np.float64)
        pt2_y = best_yj[valid_match].astype(np.float64)

        pts1 = np.stack([pt1_x, pt1_y], axis=1)
        pts2 = np.stack([pt2_x, pt2_y], axis=1)

        # ---- 批量三角化 ----
        X_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        X = (X_h[:3, :] / (X_h[3:4, :] + 1e-12)).T

        # Cheirality + 重投影
        Z1 = X[:, 2]
        X2 = (_safe_mm(R, X.T) + t.reshape(3, 1)).T
        Z2 = X2[:, 2]
        cheirality = (Z1 > 1e-3) & (Z2 > 1e-3)

        Xh = np.hstack([X, np.ones((len(X), 1), dtype=np.float64)])
        r1 = cv2.gemm(P1, Xh.T, 1.0, None, 0.0).T
        r1 = r1[:, :2] / (r1[:, 2:3] + 1e-12)
        r2 = cv2.gemm(P2, Xh.T, 1.0, None, 0.0).T
        r2 = r2[:, :2] / (r2[:, 2:3] + 1e-12)
        err1 = np.linalg.norm(r1 - pts1, axis=1)
        err2 = np.linalg.norm(r2 - pts2, axis=1)

        ok = cheirality & (err1 < reproj_thresh) & (err2 < reproj_thresh)

        # ---- 深度过滤 ----
        # 极线上近距离误匹配会导致深度趋于无穷远
        # 优先使用传统匹配的参考深度中值，更可靠
        if ok.sum() > 10:
            if ref_depth_median is not None and ref_depth_median > 0:
                med_z = ref_depth_median
            else:
                Z1_ok = Z1[ok]
                med_z = np.median(Z1_ok)
            max_z = med_z * self._max_depth_ratio
            depth_ok = (Z1 <= max_z) & (Z2 <= max_z)
            ok = ok & depth_ok

        # ---- 最小视差过滤 ----
        # 视差过小的点三角化不可靠 (深度不确定度极高)
        if ok.sum() > 4:
            disparity = np.abs(pts1[:, 0] - pts2[:, 0])  # 水平视差
            disp_ok = disparity >= self._min_disparity
            ok = ok & disp_ok

        pts1 = pts1[ok]
        pts2 = pts2[ok]
        X = X[ok]

        if len(pts1) < 3:
            return (np.empty((0, 2)), np.empty((0, 2)),
                    np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8))

        # 颜色 (优先使用原始RGB图像，避免P3D网格颜色黑点)
        p1x = np.clip(np.round(pts1[:, 0]).astype(int), 0, W - 1)
        p1y = np.clip(np.round(pts1[:, 1]).astype(int), 0, H - 1)
        if img_i is not None:
            colors = img_i[p1y, p1x].astype(np.uint8)
        else:
            colors = color_i[p1y, p1x].astype(np.uint8)

        return pts1, pts2, X, colors

    def match_frames_dense(
        self, grid_i: dict, grid_j: dict, frame_i: int, frame_j: int,
        R_known: Optional[np.ndarray] = None, t_known: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        伪3D稠密匹配。

        如果提供了R_known, t_known（来自Stage 1），使用epipolar约束加速搜索。

        Returns:
            pts1: (M,2), pts2: (M,2), colors: (M,3)
        """
        H, W = grid_i['height'].shape

        # 提取7D特征
        if 'features' not in grid_i:
            grid_i['features'] = self._extract_geometric_features(grid_i)
        if 'features' not in grid_j:
            grid_j['features'] = self._extract_geometric_features(grid_j)
        feat_i = grid_i['features']
        feat_j = grid_j['features']

        # 获取伪3D RT用于搜索窗口预测
        T_i = self.ds.get_rt(frame_i)
        T_j = self.ds.get_rt(frame_j)
        R_uvh, t_uvh = self._relative_pose_from_abs(T_i, T_j)

        # 计算epipolar约束（如果有已知位姿）
        F_known = None
        if self.use_epipolar and R_known is not None and t_known is not None:
            F_known = self._compute_fundamental(R_known, t_known, self.K)

        # 采样网格点
        mask_i = grid_i['valid_mask']
        ys, xs = np.where(mask_i)
        sample_mask = (ys % self.grid_step == 0) & (xs % self.grid_step == 0)
        ys_s = ys[sample_mask]
        xs_s = xs[sample_mask]

        if len(ys_s) < 8:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 3))

        # (u, v, h) → RT变换 → 预测在frame_j中的位置
        height_i = grid_i['height']
        pts_i_3d = np.column_stack([
            xs_s.astype(np.float64),
            ys_s.astype(np.float64),
            height_i[ys_s, xs_s],
        ])
        pts_i_in_j = (_safe_mm(R_uvh, pts_i_3d.T) + t_uvh.reshape(3, 1)).T

        matched_pts1 = []
        matched_pts2 = []
        matched_colors = []

        win = int(max(5, self.search_radius))
        invalid_mask_j = ~grid_j['valid_mask']

        for idx in range(len(ys_s)):
            x_i = xs_s[idx]
            y_i = ys_s[idx]
            pred_u = int(round(pts_i_in_j[idx, 0]))
            pred_v = int(round(pts_i_in_j[idx, 1]))

            # 搜索窗口
            u_lo = max(0, pred_u - win)
            u_hi = min(W - 1, pred_u + win)
            v_lo = max(0, pred_v - win)
            v_hi = min(H - 1, pred_v + win)
            if u_hi < u_lo or v_hi < v_lo:
                continue

            # frame_i查询特征
            fq = feat_i[y_i, x_i]
            if np.any(np.isnan(fq)):
                continue

            # 窗口内候选
            window_feat = feat_j[v_lo:v_hi + 1, u_lo:u_hi + 1]
            window_invalid = invalid_mask_j[v_lo:v_hi + 1, u_lo:u_hi + 1]
            if window_feat.size == 0:
                continue

            # 特征距离
            diff = window_feat - fq[None, None, :]
            nan_mask = np.any(np.isnan(diff), axis=2) | window_invalid
            feat_dist = np.sqrt(np.nansum(diff ** 2, axis=2))
            feat_dist[nan_mask] = np.inf
            if np.all(np.isinf(feat_dist)):
                continue

            # Epipolar约束过滤（如果有已知位姿）
            if F_known is not None:
                pt_i_h = np.array([x_i, y_i, 1.0], dtype=np.float64)
                epipolar_line = _safe_mm(F_known, pt_i_h.reshape(3, 1)).ravel()
                # 计算窗口内每个点到epipolar线的距离
                wv, wu = np.mgrid[v_lo:v_hi + 1, u_lo:u_hi + 1]
                epi_dist = np.abs(epipolar_line[0] * wu + epipolar_line[1] * wv + epipolar_line[2])
                epi_dist /= (np.sqrt(epipolar_line[0]**2 + epipolar_line[1]**2) + 1e-12)
                # 距离epipolar线太远的候选过滤掉
                feat_dist[epi_dist > self._epipolar_width] = np.inf

            # 最佳匹配
            min_j = np.argmin(feat_dist.ravel())
            min_dist = feat_dist.ravel()[min_j]
            if min_dist > self._feat_thresh:
                continue
            dj = min_j % feat_dist.shape[1]
            di = min_j // feat_dist.shape[1]
            x_j = u_lo + dj
            y_j = v_lo + di

            # 双向交叉验证
            fq_j = feat_j[y_j, x_j]
            if np.any(np.isnan(fq_j)):
                continue
            rev_u_lo = max(0, x_j - win)
            rev_u_hi = min(W - 1, x_j + win)
            rev_v_lo = max(0, y_j - win)
            rev_v_hi = min(H - 1, y_j + win)
            rev_feat = feat_i[rev_v_lo:rev_v_hi + 1, rev_u_lo:rev_u_hi + 1]
            rev_diff = rev_feat - fq_j[None, None, :]
            rev_nan = np.any(np.isnan(rev_diff), axis=2)
            rev_dist = np.sqrt(np.nansum(rev_diff ** 2, axis=2))
            rev_dist[rev_nan] = np.inf
            if np.all(np.isinf(rev_dist)):
                continue
            rev_min_j = np.argmin(rev_dist.ravel())
            rev_dj = rev_min_j % rev_dist.shape[1]
            rev_di = rev_min_j // rev_dist.shape[1]
            rev_x = rev_u_lo + rev_dj
            rev_y = rev_v_lo + rev_di
            if abs(rev_x - x_i) > self._cross_check_px or abs(rev_y - y_i) > self._cross_check_px:
                continue

            matched_pts1.append([float(x_i), float(y_i)])
            matched_pts2.append([float(x_j), float(y_j)])
            matched_colors.append(grid_i['color'][y_i, x_i])

        if len(matched_pts1) < 4:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 3))

        pts1 = np.array(matched_pts1, dtype=np.float64)
        pts2 = np.array(matched_pts2, dtype=np.float64)
        colors = np.array(matched_colors, dtype=np.float64)
        return pts1, pts2, colors

    def triangulate_with_known_pose(
        self, pts1: np.ndarray, pts2: np.ndarray,
        R: np.ndarray, t: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
        """
        用已知位姿(R, t)直接三角化 + 几何验证。

        不经过E-RANSAC，仅做cheirality + reprojection检查。
        """
        K = self.K
        P1 = _safe_mm(K, np.hstack([np.eye(3), np.zeros((3, 1))]))
        P2 = _safe_mm(K, np.hstack([R, t.reshape(3, 1)]))

        X_h = cv2.triangulatePoints(P1, P2,
                                     pts1.T.astype(np.float64),
                                     pts2.T.astype(np.float64))
        X = (X_h[:3, :] / (X_h[3:4, :] + 1e-12)).T

        # Cheirality
        Z1 = X[:, 2]
        X2 = (_safe_mm(R, X.T) + t.reshape(3, 1)).T
        Z2 = X2[:, 2]
        cheirality = (Z1 > 1e-6) & (Z2 > 1e-6)

        # Reprojection
        Xh = np.hstack([X, np.ones((len(X), 1), dtype=np.float64)])
        r1 = _safe_mm(P1, Xh.T).T; r1 = r1[:, :2] / (r1[:, 2:3] + 1e-12)
        r2 = _safe_mm(P2, Xh.T).T; r2 = r2[:, :2] / (r2[:, 2:3] + 1e-12)
        err1 = np.linalg.norm(r1 - pts1, axis=1)
        err2 = np.linalg.norm(r2 - pts2, axis=1)
        ok = cheirality & (err1 < self.reproj_thresh) & (err2 < self.reproj_thresh)

        # 深度过滤 (与极线匹配相同策略)
        if ok.sum() > 10:
            Z1_ok = Z1[ok]
            med_z = np.median(Z1_ok)
            max_z = med_z * self._max_depth_ratio
            depth_ok = (Z1 <= max_z) & (Z2 <= max_z)
            ok = ok & depth_ok

        if ok.sum() < 4:
            return None

        stats = {
            "total": len(pts1),
            "cheirality_ok": int(cheirality.sum()),
            "final": int(ok.sum()),
            "mean_depth": float(np.mean(Z1[ok])),
            "mean_reproj1": float(np.mean(err1[ok])),
            "mean_reproj2": float(np.mean(err2[ok])),
        }
        return X[ok], pts1[ok], pts2[ok], stats


# ============================================================
#  可视化
# ============================================================

def _save_projection_views(pts: np.ndarray, path: str, title: str = "Reconstruction"):
    if not HAS_MPL or len(pts) < 10:
        return
    try:
        MAX_PLOT = 20000
        if len(pts) > MAX_PLOT:
            rng = np.random.default_rng(42)
            pts = pts[rng.choice(len(pts), MAX_PLOT, replace=False)]
        for dim in range(3):
            col = pts[:, dim]
            lo, hi = np.quantile(col, [0.02, 0.98])
            mask = (col >= lo) & (col <= hi)
            pts = pts[mask]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        views = [
            ("XY (Top View)", 0, 1, "X (world)", "Y (world)"),
            ("XZ (Front View)", 0, 2, "X (world)", "Z (world)"),
            ("YZ (Side View)", 1, 2, "Y (world)", "Z (world)"),
        ]
        for ax, (vtitle, dx, dy, xlbl, ylbl) in zip(axes, views):
            ax.hist2d(pts[:, dx], pts[:, dy], bins=100, cmap='viridis')
            ax.set_title(vtitle, fontsize=13)
            ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
            ax.set_aspect('auto')
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [可视化] 已保存: {path}")
    except Exception as e:
        print(f"  [警告] 可视化失败: {e}")


def _save_camera_trajectory(poses: np.ndarray, path: str):
    """保存相机轨迹(仅XY平面图，避免3D图的np.linalg.inv BLAS死锁)"""
    if not HAS_MPL or len(poses) < 2:
        return
    try:
        fig = plt.figure(figsize=(12, 5))
        # 仅画XY平面图，避免3D图的numpy BLAS死锁
        ax = fig.add_subplot(111)
        ax.plot(poses[:, 0], poses[:, 1], 'b-o', markersize=3, linewidth=1)
        ax.scatter(poses[0, 0], poses[0, 1], c='green', s=50, label='Start')
        ax.scatter(poses[-1, 0], poses[-1, 1], c='red', s=50, label='End')
        ax.set_xlabel('X (world)'); ax.set_ylabel('Y (world)')
        ax.set_title('Camera Trajectory (XY)')
        ax.legend()
        ax.set_aspect('equal')
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [轨迹] 已保存: {path}")
    except Exception as e:
        print(f"  [警告] 轨迹图失败: {e}")


# ============================================================
#  v3 主管道
# ============================================================

class HybridPipelineV3:
    """
    v3 混合管道: 传统SfM精位姿 + 伪3D稠密重建

    Stage 1 ─ 传统SIFT+ORB精确位姿恢复 (POSE ONLY)
    Stage 2 ─ 伪3D稠密匹配 + 已知位姿三角化 (DENSE RECONSTRUCTION)
    Stage 3 ─ 多帧融合 + 世界坐标系对齐
    """

    def __init__(
        self,
        dataset: DatasetLoader,
        K: np.ndarray,
        # Stage 1: 传统位姿参数
        sift_nfeatures: int = 3000,
        orb_nfeatures: int = 3000,
        sift_ratio: float = 0.75,
        ransac_thresh: float = 2.0,
        ransac_prob: float = 0.9999,
        # Stage 2: 密集极线匹配参数
        grid_step: int = 3,
        epipolar_band: int = 2,
        normal_thresh: float = 0.15,
        color_weight: float = 0.2,
        reproj_thresh: float = 4.0,
        # Stage 2 旧参数(保留向后兼容)
        search_radius: float = 25.0,
        p3d_match_thresh: float = 1.0,
        cross_check_px: float = 3.0,
        use_epipolar: bool = True,
        epipolar_width: float = 5.0,
        # 融合参数
        fuse_traditional: bool = True,
        dedup_pixel_thresh: float = 5.0,
        # 深度过滤参数
        max_depth_ratio: float = 5.0,
        min_disparity: float = 1.0,
        # 通用
        output_dir: str = "",
    ):
        self.ds = dataset
        self.K = K
        self.output_dir = output_dir
        self.fuse_traditional = fuse_traditional
        self._dedup_px = dedup_pixel_thresh

        # Stage 1: 传统位姿估计器
        self.pose_estimator = TraditionalPoseEstimator(
            dataset=dataset, K=K,
            sift_nfeatures=sift_nfeatures,
            orb_nfeatures=orb_nfeatures,
            ratio_thresh=sift_ratio,
            ransac_thresh=ransac_thresh,
            ransac_prob=ransac_prob,
        )

        # Stage 2: 伪3D稠密重建器
        self.dense_reconstructor = Pseudo3DDenseReconstructor(
            dataset=dataset, K=K,
            grid_step=grid_step,
            search_radius=search_radius,
            feature_match_thresh=p3d_match_thresh,
            reproj_thresh=reproj_thresh,
            cross_check_px=cross_check_px,
            use_epipolar=use_epipolar,
            epipolar_width=epipolar_width,
            epipolar_band=epipolar_band,
            normal_thresh=normal_thresh,
            color_weight=color_weight,
            max_depth_ratio=max_depth_ratio,
            min_disparity=min_disparity,
        )

    def run(self, max_frames: int = 0) -> dict:
        t0 = time.time()

        # 输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        dir_pose = os.path.join(self.output_dir, "01_pose_recovery")
        dir_dense = os.path.join(self.output_dir, "02_dense_p3d")
        dir_fused = os.path.join(self.output_dir, "03_fused")
        for d in [dir_pose, dir_dense, dir_fused]:
            os.makedirs(d, exist_ok=True)

        n_frames = self.ds.num_frames()
        if max_frames > 0:
            n_frames = min(n_frames, max_frames)

        print("=" * 70)
        print("Hybrid Pipeline v3: Traditional SfM(Pose) + Pseudo-3D(Dense)")
        print("=" * 70)
        print(f"图像尺寸: {self.ds.target_W}x{self.ds.target_H}")
        print(f"帧数: {n_frames}")
        print(f"内参K:\n{self.K}")

        total_pairs = n_frames - 1

        # ============================================================
        #  Stage 1: 传统SfM精确位姿恢复
        # ============================================================
        print(f"\n{'='*70}")
        print("Stage 1: 传统SIFT+ORB精确位姿恢复")
        print(f"{'='*70}\n")

        pair_poses: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        pair_inlier_masks: dict[tuple[int, int], np.ndarray] = {}
        pose_stats: list[dict] = []
        camera_poses: list[np.ndarray] = []
        T_world = np.eye(4, dtype=np.float64)
        camera_poses.append(T_world[:3, 3].copy())

        for i in range(total_pairs):
            fi, fj = i, i + 1

            try:
                result = self.pose_estimator.estimate_pose(fi, fj)
            except Exception as e:
                print(f"  [Stage1] {fi:04d}->{fj:04d} ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

            if result is None:
                print(f"  [Stage1] {fi:04d}->{fj:04d} FAIL")
                continue

            R, t, inlier_mask, stats = result
            pair_poses[(fi, fj)] = (R, t)
            pair_inlier_masks[(fi, fj)] = inlier_mask

            # 累积世界坐标
            R_inv = R.T
            t_inv = -_safe_mm(R.T, t.reshape(3, 1))
            delta_inv = np.eye(4, dtype=np.float64)
            delta_inv[:3, :3] = R_inv
            delta_inv[:3, 3] = t_inv.ravel()
            T_world = _matmul_4x4(T_world, delta_inv)
            camera_poses.append(T_world[:3, 3].copy())

            pose_stats.append({
                "frame_i": fi, "frame_j": fj, **stats,
            })
            print(f"  [Stage1] {fi:04d}->{fj:04d} OK matches={stats['n_matches']}, E_in={stats['E_inliers']}({stats['E_inlier_ratio']:.1%})")

        n_pose_ok = len(pair_poses)
        print(f"\n[Stage1] 位姿恢复: {n_pose_ok}/{total_pairs} 对成功")

        # 保存位姿统计
        if pose_stats:
            pose_csv = os.path.join(dir_pose, "pose_stats.csv")
            with open(pose_csv, "w", encoding="utf-8") as f:
                keys = ["frame_i", "frame_j", "n_matches", "E_inliers", "E_inlier_ratio"]
                f.write(",".join(keys) + "\n")
                for s in pose_stats:
                    f.write(",".join(str(s.get(k, "")) for k in keys) + "\n")

        # 保存相机轨迹 (禁用 — matplotlib可能触发BLAS死锁)
        # if len(camera_poses) > 1:
        #     try:
        #         _save_camera_trajectory(np.array(camera_poses),
        #                                 os.path.join(dir_pose, "camera_trajectory.png"))
        #     except Exception as e:
        #         print(f"  [警告] 轨迹图失败: {e}")
        print("\nDEBUG: Stage 1 done, starting Stage 2...", flush=True)

        # ============================================================
        #  Stage 2: 密集极线匹配 + 传统稀疏点融合 + 已知位姿三角化
        # ============================================================
        print(f"\n{'='*70}")
        print("Stage 2: 密集极线匹配 + 传统融合 + 已知位姿三角化")
        print(f"{'='*70}\n")
        print(f"  密集极线参数: grid_step={self.dense_reconstructor.grid_step}, "
              f"epipolar_band={self.dense_reconstructor._epipolar_band}, "
              f"normal_thresh={self.dense_reconstructor._normal_thresh:.2f}, "
              f"color_weight={self.dense_reconstructor._color_weight:.2f}")
        print(f"  深度过滤: max_depth_ratio={self.dense_reconstructor._max_depth_ratio}, "
              f"min_disparity={self.dense_reconstructor._min_disparity}")
        print(f"  融合传统稀疏点: {self.fuse_traditional}")

        all_pts: list[np.ndarray] = []
        all_cols: list[np.ndarray] = []
        dense_stats: list[dict] = []

        # 重新初始化世界坐标
        T_world = np.eye(4, dtype=np.float64)

        for i in range(total_pairs):
            fi, fj = i, i + 1
            print(f"  [Stage2] {fi:04d}->{fj:04d} ", end=" ")

            # 检查是否有已知位姿
            if (fi, fj) not in pair_poses:
                print("SKIP(no_pose)")
                # 累积E-pose (用P3D RT近似)
                T_i = self.ds.get_rt(fi)
                T_j = self.ds.get_rt(fj)
                R_uvh, t_uvh = Pseudo3DDenseReconstructor._relative_pose_from_abs(T_i, T_j)
                # 用单位变换近似缺失位姿
                T_world_approx = T_world.copy()
                continue

            R, t = pair_poses[(fi, fj)]

            try:
                # 加载网格
                grid_i = self.ds.load_mesh_grid(fi)
                grid_j = self.ds.load_mesh_grid(fj)
            except Exception as e:
                print(f"FAIL: grid load error: {e}")
                R_inv = R.T; t_inv = -_safe_mm(R.T, t.reshape(3, 1))
                delta_inv = np.eye(4, dtype=np.float64); delta_inv[:3, :3] = R_inv; delta_inv[:3, 3] = t_inv.ravel()
                T_world = _matmul_4x4(T_world, delta_inv)
                continue

            # 加载图像（颜色采样）
            img_i = self.ds.load_image(fi)

            try:
                # ---- 传统稀疏点 (先做，获取参考深度) ----
                trad_pts = np.zeros((0, 3), dtype=np.float64)
                trad_cols = np.zeros((0, 3), dtype=np.uint8)
                trad_px = np.zeros((0, 2), dtype=np.float64)
                n_trad = 0
                ref_depth_median = None

                if self.fuse_traditional and (fi, fj) in pair_inlier_masks:
                    pts1_all, pts2_all = self.pose_estimator._match_features(fi, fj)
                    inlier_mask = pair_inlier_masks[(fi, fj)]
                    if len(pts1_all) >= 8:
                        pts1_in = pts1_all[inlier_mask]
                        pts2_in = pts2_all[inlier_mask]
                        if len(pts1_in) >= 5:
                            tri_result = self.dense_reconstructor.triangulate_with_known_pose(
                                pts1_in, pts2_in, R, t)
                            if tri_result is not None:
                                X_cam1, p1_ok, _, _ = tri_result
                                trad_pts = X_cam1
                                trad_px = p1_ok
                                n_trad = len(X_cam1)
                                # 用传统匹配的中值深度作为极线匹配的参考
                                ref_depth_median = float(np.median(X_cam1[:, 2]))
                                if img_i is not None:
                                    xs = np.clip(np.round(p1_ok[:, 0]).astype(int), 0, img_i.shape[1]-1)
                                    ys = np.clip(np.round(p1_ok[:, 1]).astype(int), 0, img_i.shape[0]-1)
                                    trad_cols = img_i[ys, xs, :]
                                else:
                                    trad_cols = np.full((n_trad, 3), 128, dtype=np.uint8)

                # ---- 密集极线匹配 (用传统参考深度过滤) ----
                t_epi = time.time()
                pts1_epi, pts2_epi, X3d_epi, colors_epi = self.dense_reconstructor.match_dense_epipolar(
                    grid_i, grid_j, R, t, img_i=img_i, ref_depth_median=ref_depth_median)
                dt_epi = time.time() - t_epi
                n_epi = len(pts1_epi)

                # ---- 融合: 传统 + 密集极线去重 ----
                epi_px = pts1_epi if n_epi > 0 else np.zeros((0, 2), dtype=np.float64)
                n_unique = 0
                if n_epi > 0 and n_trad > 0:
                    keep = np.ones(n_epi, dtype=bool)
                    for b in range(0, n_epi, 500):
                        be = min(b + 500, n_epi)
                        dists = np.linalg.norm(epi_px[b:be, None, :] - trad_px[None, :, :], axis=2)
                        keep[b:be] = dists.min(axis=1) > self._dedup_px
                    n_unique = keep.sum()
                    X3d_epi_f = X3d_epi[keep]
                    colors_epi_f = colors_epi[keep] if len(colors_epi) > 0 else colors_epi
                else:
                    X3d_epi_f = X3d_epi
                    colors_epi_f = colors_epi

                # 合并点云
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
                    print(f"SKIP(no_points) epi={n_epi} trad={n_trad}")
                    R_inv = R.T; t_inv = -_safe_mm(R.T, t.reshape(3, 1))
                    delta_inv = np.eye(4, dtype=np.float64); delta_inv[:3, :3] = R_inv; delta_inv[:3, 3] = t_inv.ravel()
                    T_world = _matmul_4x4(T_world, delta_inv)
                    continue

                # 世界坐标变换
                X_world = (_safe_mm(T_world[:3, :3], X_merged.T) + T_world[:3, 3:4]).T

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                R_inv = R.T; t_inv = -_safe_mm(R.T, t.reshape(3, 1))
                delta_inv = np.eye(4, dtype=np.float64); delta_inv[:3, :3] = R_inv; delta_inv[:3, 3] = t_inv.ravel()
                T_world = _matmul_4x4(T_world, delta_inv)
                continue

            # 累积E-pose
            R_inv = R.T
            t_inv = -_safe_mm(R.T, t.reshape(3, 1))
            delta_inv = np.eye(4, dtype=np.float64)
            delta_inv[:3, :3] = R_inv
            delta_inv[:3, 3] = t_inv.ravel()
            T_world = _matmul_4x4(T_world, delta_inv)

            # 保存逐对PLY
            _write_ply(X_world, cols_merged,
                       os.path.join(dir_dense, f"dense_{fi:04d}_{fj:04d}.ply"))

            all_pts.append(X_world.astype(np.float32))
            all_cols.append(cols_merged.astype(np.uint8))

            dense_stats.append({
                "frame_i": fi, "frame_j": fj,
                "epi_matches": n_epi,
                "trad_matches": n_trad,
                "epi_unique": n_unique,
                "fused": len(X_merged),
                "dt_epi": round(dt_epi, 2),
            })

            print(f"OK epi={n_epi} trad={n_trad} unique={n_unique} fused={len(X_merged)} ({dt_epi:.1f}s)")

        if not all_pts:
            raise RuntimeError("没有一对帧成功稠密重建！")

        # ============================================================
        #  Stage 3: 融合
        # ============================================================
        print(f"\n{'='*70}")
        print("Stage 3: 多帧融合 + 世界坐标系对齐")
        print(f"{'='*70}\n")

        merged = np.vstack(all_pts)
        merged_cols = np.vstack(all_cols)
        print(f"  原始总点数: {len(merged)}")

        # Voxel降采样
        if len(merged) > 1000:
            vs = 0.3
            voxel_idx = np.floor(merged / vs).astype(np.int64)
            flat = voxel_idx[:, 0] * 100000000 + voxel_idx[:, 1] * 10000 + voxel_idx[:, 2]
            _, first_idx = np.unique(flat, return_index=True)
            merged = merged[first_idx]
            merged_cols = merged_cols[first_idx]
            print(f"  Voxel降采样: {len(first_idx)} 点")

        # 离群点移除
        if len(merged) > 100:
            centroid = np.median(merged, axis=0)
            dists = np.linalg.norm(merged - centroid[None, :], axis=1)
            med = np.median(dists)
            inlier = dists < med * 5.0
            n_outlier = (~inlier).sum()
            if n_outlier > 0:
                merged = merged[inlier]
                merged_cols = merged_cols[inlier]
                print(f"  移除 {n_outlier} 个离群点")

        # 保存最终点云
        _write_ply(merged, merged_cols,
                   os.path.join(dir_fused, "dense_fused_final.ply"))

        # ============================================================
        #  统计输出 (先保存，避免后续可视化崩溃丢失)
        # ============================================================
        elapsed = time.time() - t0

        # 保存逐对统计CSV
        csv_path = os.path.join(self.output_dir, "per_pair_stats.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            # 合并Stage1和Stage2统计
            all_stats_keys = ["frame_i", "frame_j", "n_matches", "E_inliers",
                              "E_inlier_ratio", "epi_matches", "trad_matches",
                              "epi_unique", "fused", "dt_epi"]
            f.write(",".join(all_stats_keys) + "\n")
            # 以Stage2为主，补充Stage1数据
            pose_dict = {(s["frame_i"], s["frame_j"]): s for s in pose_stats}
            dense_dict = {(s["frame_i"], s["frame_j"]): s for s in dense_stats}
            for i in range(total_pairs):
                fi, fj = i, i + 1
                ps = pose_dict.get((fi, fj), {})
                ds_row = dense_dict.get((fi, fj), {})
                row = []
                for k in all_stats_keys:
                    row.append(str(ds_row.get(k, ps.get(k, ""))))
                f.write(",".join(row) + "\n")

        # 保存总结
        total_epi = sum(s["epi_matches"] for s in dense_stats)
        total_trad = sum(s["trad_matches"] for s in dense_stats)
        total_unique = sum(s["epi_unique"] for s in dense_stats)
        total_fused = sum(s["fused"] for s in dense_stats)

        print(f"\n{'='*70}")
        print(f"[完成] 总耗时: {elapsed:.1f}s")
        print(f"[统计] 位姿恢复: {n_pose_ok}/{total_pairs} 对成功")
        print(f"[统计] 稠密重建: {len(all_pts)}/{total_pairs} 对成功")
        print(f"[统计] 密集极线总匹配: {total_epi}")
        print(f"[统计] 传统稀疏总匹配: {total_trad}")
        print(f"[统计] 极线去重后新增: {total_unique}")
        print(f"[统计] 融合后总点数: {total_fused}")
        print(f"[统计] 最终点数(voxel+outlier): {len(merged)}")
        print(f"{'='*70}")

        # 与v2对比
        v2_final = 6792
        improvement = len(merged) / max(1, v2_final)
        print(f"\n[对比v2] v2最终={v2_final}, v3最终={len(merged)}, 倍数={improvement:.1f}x")

        stats_path = os.path.join(self.output_dir, "summary_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"Hybrid Pipeline v3 Statistics\n")
            f.write(f"{'='*40}\n")
            f.write(f"Total pairs:       {total_pairs}\n")
            f.write(f"Pose recovered:    {n_pose_ok}\n")
            f.write(f"Dense pairs:       {len(all_pts)}\n")
            f.write(f"Final points:      {len(merged)}\n")
            f.write(f"Total epi matches: {total_epi}\n")
            f.write(f"Total trad points: {total_trad}\n")
            f.write(f"Epi unique points: {total_unique}\n")
            f.write(f"Total fused:       {total_fused}\n")
            f.write(f"Elapsed:           {elapsed:.1f}s\n")
            f.write(f"v2 comparison:     {improvement:.1f}x ({v2_final} -> {len(merged)})\n")

        # ============================================================
        #  可视化 (可能触发BLAS死锁，放在最后)
        # ============================================================
        try:
            _save_projection_views(merged,
                                   os.path.join(dir_fused, "dense_views.png"),
                                   "v3: Traditional Pose + Pseudo-3D Dense")
        except Exception as e:
            print(f"  [警告] 投影视图失败: {e}")

        # 轨迹图(可能触发BLAS死锁，用try-except保护)
        if len(camera_poses) > 1:
            try:
                _save_camera_trajectory(np.array(camera_poses),
                                        os.path.join(dir_fused, "camera_trajectory.png"))
            except Exception as e:
                print(f"  [警告] 轨迹图失败: {e}")

        return {
            "elapsed_sec": elapsed,
            "pose_pairs": n_pose_ok,
            "dense_pairs": len(all_pts),
            "final_points": len(merged),
            "total_epi": total_epi,
            "total_trad": total_trad,
            "total_unique": total_unique,
            "total_fused": total_fused,
            "per_pair_pose": pose_stats,
            "per_pair_dense": dense_stats,
        }


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Pipeline v3: Traditional SfM(Pose) + Pseudo-3D(Dense)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="原始RGB图像目录")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="棋盘格脚本输出目录")
    parser.add_argument("--output_dir", "-O", type=str, default="",
                        help="输出目录 (默认自动创建)")
    parser.add_argument("--max_frames", "-n", type=int, default=0,
                        help="最大帧数 (0=全部)")
    parser.add_argument("--fx", type=float, default=0.0,
                        help="焦距 (0=自动估计)")
    parser.add_argument("--target_w", type=int, default=640)
    parser.add_argument("--target_h", type=int, default=358)
    # Stage 1: 位姿参数
    parser.add_argument("--sift_features", type=int, default=2000)
    parser.add_argument("--orb_features", type=int, default=2000)
    parser.add_argument("--sift_ratio", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=2.0)
    # Stage 2: 伪3D参数
    parser.add_argument("--grid_step", type=int, default=3,
                        help="密集极线采样步长 (默认3)")
    parser.add_argument("--epipolar_band", type=int, default=2,
                        help="极线搜索带宽(±像素, 默认2)")
    parser.add_argument("--normal_thresh", type=float, default=0.15,
                        help="法线余弦阈值(默认0.15)")
    parser.add_argument("--color_weight", type=float, default=0.2,
                        help="颜色在匹配评分中的权重(默认0.2)")
    parser.add_argument("--no_fuse_traditional", action="store_true", default=False,
                        help="禁用传统稀疏点融合")
    parser.add_argument("--max_depth_ratio", type=float, default=5.0,
                        help="深度过滤: 最大深度=中值*倍数(默认5.0)")
    parser.add_argument("--min_disparity", type=float, default=1.0,
                        help="最小视差阔值(默认1.0像素)")
    parser.add_argument("--search_radius", type=float, default=25.0)
    parser.add_argument("--p3d_match_thresh", type=float, default=1.0,
                        help="伪3D特征匹配阈值 (v2=0.8, v3默认1.0)")
    parser.add_argument("--reproj_thresh", type=float, default=5.0)
    parser.add_argument("--cross_check_px", type=float, default=3.0)
    parser.add_argument("--use_epipolar", action="store_true", default=True,
                        help="使用已知位姿的epipolar约束")
    parser.add_argument("--no_epipolar", action="store_true", default=False,
                        help="禁用epipolar约束")
    parser.add_argument("--epipolar_width", type=float, default=5.0)

    args = parser.parse_args()

    image_dir = os.path.abspath(args.image_dir)
    results_dir = os.path.abspath(args.results_dir)

    for d, name in [(image_dir, "图像"), (results_dir, "结果")]:
        if not os.path.isdir(d):
            print(f"[错误] {name}目录不存在: {d}")
            sys.exit(1)

    if not args.output_dir:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), f"hybrid_v3_{ts}")
    else:
        output_dir = os.path.abspath(args.output_dir)

    target_size = (args.target_w, args.target_h)
    use_epipolar = not args.no_epipolar

    print("=" * 70)
    print("Hybrid Pipeline v3: Traditional SfM(Pose) + Pseudo-3D(Dense)")
    print("=" * 70)

    try:
        ds = DatasetLoader(image_dir, results_dir, target_size)
        K = _make_camera_intrinsics(target_size[0], target_size[1], args.fx)

        pipeline = HybridPipelineV3(
            dataset=ds, K=K,
            sift_nfeatures=args.sift_features,
            orb_nfeatures=args.orb_features,
            sift_ratio=args.sift_ratio,
            ransac_thresh=args.ransac_thresh,
            grid_step=args.grid_step,
            epipolar_band=args.epipolar_band,
            normal_thresh=args.normal_thresh,
            color_weight=args.color_weight,
            reproj_thresh=args.reproj_thresh,
            fuse_traditional=not args.no_fuse_traditional,
            cross_check_px=args.cross_check_px,
            use_epipolar=use_epipolar,
            epipolar_width=args.epipolar_width,
            max_depth_ratio=args.max_depth_ratio,
            min_disparity=args.min_disparity,
            output_dir=output_dir,
        )
        pipeline.run(max_frames=args.max_frames)

    except Exception as e:
        print(f"\n[错误] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
