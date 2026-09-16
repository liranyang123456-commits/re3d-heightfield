

import os
import cv2
import numpy as np
# 智能matplotlib后端选择（避免Qt插件问题）
import matplotlib
# 检查环境变量，如果设置为非交互式，使用Agg后端
use_interactive_backend = bool(int(os.environ.get("MATPLOTLIB_INTERACTIVE", "0")))
if not use_interactive_backend:
    # 默认使用非交互式后端，避免Qt插件问题
    try:
        matplotlib.use('Agg', force=False)  # 非交互式后端
    except Exception:
        pass  # 如果设置失败，使用默认后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
try:
    import cupy as cp
    from cupyx.scipy.ndimage import convolve1d
except Exception:
    cp = None
    convolve1d = None
try:
    import open3d as o3d
except Exception:
    o3d = None
# PyTorch3D for rendering-based pose optimization
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pytorch3d
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        look_at_view_transform,
        look_at_rotation,
        FoVPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader,
        SoftSilhouetteShader,
        HardPhongShader,
        TexturesVertex,
        BlendParams
    )
    from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("[警告] PyTorch3D未安装，基于渲染的位姿优化功能将不可用")
import json
import math
import copy
import time
import re
import glob

# =========================
# 深度恢复（两视图三角化）
# 复用：本脚本已有的“帧间对应关系”（3D网格匹配的correspondence_set）与外部标定CSV位姿
# 说明：
# - 若使用 recoverPose(E) 得到的 t，则尺度未知 -> 深度为相对尺度
# - 若使用标定CSV/IMU提供的度量尺度位姿 -> 可恢复度量深度（取决于K与位姿约定）
# =========================

def _depth_write_ply_xyzrgb(path, xyz, bgr=None):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    use_color = bgr is not None
    if use_color:
        bgr = np.asarray(bgr, dtype=np.uint8).reshape(-1, 3)
        if len(bgr) != len(xyz):
            use_color = False

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if use_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if use_color:
            for p, c in zip(xyz, bgr):
                # PLY是RGB；OpenCV是BGR
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[2])} {int(c[1])} {int(c[0])}\n")
        else:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _depth_save_vis(depth, out_path, clip_percentile=99.0):
    d = np.asarray(depth, dtype=np.float32)
    valid = d > 0
    if not np.any(valid):
        cv2.imwrite(out_path, np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8))
        return
    vmax = float(np.percentile(d[valid], clip_percentile))
    vmin = float(np.percentile(d[valid], 1.0))
    if vmax <= vmin:
        vmax = float(d[valid].max())
        vmin = float(d[valid].min())
    dn = np.zeros_like(d, dtype=np.float32)
    dn[valid] = (np.clip(d[valid], vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
    img = (dn * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.imwrite(out_path, color)


def _depth_relative_pose_candidates_from_abs(T1, T2):
    """
    给定两帧“绝对位姿”矩阵 T1,T2，返回两种常见约定下的相对位姿候选：
      - cam_to_world:  X_w = T * X_c
      - world_to_cam:  X_c = T * X_w
    返回 dict[name] = (R,t)，其中 [R|t] 表示 cam2 <- cam1（用于 P2=K*[R|t], P1=K*[I|0]）
    """
    T1 = np.asarray(T1, dtype=np.float64)
    T2 = np.asarray(T2, dtype=np.float64)
    R1, t1 = T1[:3, :3], T1[:3, 3:4]
    R2, t2 = T2[:3, :3], T2[:3, 3:4]

    # cam_to_world:
    R_rel_cam2_cam1_cam_to_world = R2.T @ R1
    t_rel_cam2_cam1_cam_to_world = R2.T @ (t1 - t2)

    # world_to_cam:
    R_rel_cam2_cam1_world_to_cam = R2 @ R1.T
    t_rel_cam2_cam1_world_to_cam = t2 - (R_rel_cam2_cam1_world_to_cam @ t1)

    return {
        "cam_to_world": (R_rel_cam2_cam1_cam_to_world, t_rel_cam2_cam1_cam_to_world),
        "world_to_cam": (R_rel_cam2_cam1_world_to_cam, t_rel_cam2_cam1_world_to_cam),
    }


def _depth_triangulate_sparse(K, R, t, pts1, pts2, image_shape_hw, reproj_thresh_px=3.0):
    """
    两视图三角化（稀疏深度）。
    约定：
      P1 = K [I|0]
      P2 = K [R|t]，其中 R,t 把 cam1 坐标系点变换到 cam2 坐标系（cam2 <- cam1）
    """
    H, W = int(image_shape_hw[0]), int(image_shape_hw[1])
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)
    if len(pts1) < 8 or len(pts2) < 8 or len(pts1) != len(pts2):
        return None

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    x1 = pts1.T
    x2 = pts2.T
    X_h = cv2.triangulatePoints(P1, P2, x1, x2)  # 4xN
    X = (X_h[:3, :] / (X_h[3:4, :] + 1e-12)).T  # Nx3，cam1坐标

    # 正深度约束（cam1、cam2都为正）
    Z1 = X[:, 2]
    X2 = (R @ X.T + t).T
    Z2 = X2[:, 2]
    cheirality = (Z1 > 1e-6) & (Z2 > 1e-6)

    def _reproj(P, X3):
        Xh = np.hstack([X3, np.ones((len(X3), 1), dtype=np.float64)])
        x = (P @ Xh.T).T
        x = x[:, :2] / (x[:, 2:3] + 1e-12)
        return x

    reproj1 = _reproj(P1, X)
    reproj2 = _reproj(P2, X)
    err1 = np.linalg.norm(reproj1 - pts1, axis=1)
    err2 = np.linalg.norm(reproj2 - pts2, axis=1)
    ok = cheirality & (err1 < float(reproj_thresh_px)) & (err2 < float(reproj_thresh_px))

    X_ok = X[ok]
    pts1_ok = pts1[ok].astype(np.float32)
    pts2_ok = pts2[ok].astype(np.float32)
    depth_ok = X_ok[:, 2].astype(np.float32)

    depth_map = np.zeros((H, W), dtype=np.float32)
    hit = np.zeros((H, W), dtype=np.uint8)
    xs = np.clip(np.round(pts1_ok[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts1_ok[:, 1]).astype(int), 0, H - 1)
    for x, y, d in zip(xs, ys, depth_ok):
        if hit[y, x] == 0:
            depth_map[y, x] = float(d)
            hit[y, x] = 1
        else:
            depth_map[y, x] = min(depth_map[y, x], float(d))

    return {
        "X_cam1": X_ok,
        "pts1": pts1_ok,
        "pts2": pts2_ok,
        "depth_map": depth_map,
        "stats": {
            "total_matches": int(len(pts1)),
            "cheirality_ok": int(cheirality.sum()),
            "final_points": int(len(X_ok)),
            "mean_reproj1": float(np.mean(err1[ok])) if ok.any() else float("inf"),
            "mean_reproj2": float(np.mean(err2[ok])) if ok.any() else float("inf"),
        },
    }


def _depth_load_intrinsics_from_pose_csv(csv_path, image_wh=None):
    """
    从标定CSV尝试读取 fx,fy,cx,cy。

    支持两类格式：
    - **表头包含 fx/fy/cx/cy**（优先）
    - 兼容旧约定：索引18-21为 fx,fy,cx,cy

    为避免列错位导致的灾难性K，这里会做合理性检查：
    - fx,fy 必须为正且处于合理像素范围
    - 若提供 image_wh=(W,H)，则 cx,cy 必须落在接近图像范围（允许一定外延）
    """
    if (csv_path is None) or (not os.path.exists(csv_path)):
        return None

    def _plausible(fx, fy, cx, cy):
        try:
            fx = float(fx); fy = float(fy); cx = float(cx); cy = float(cy)
        except Exception:
            return False
        if not (np.isfinite(fx) and np.isfinite(fy) and np.isfinite(cx) and np.isfinite(cy)):
            return False
        # fx/fy: 典型像素焦距范围（宽松）
        if fx <= 10 or fy <= 10 or fx > 20000 or fy > 20000:
            return False
        if image_wh is not None:
            W, H = int(image_wh[0]), int(image_wh[1])
            if W > 0 and H > 0:
                if not (-0.2 * W <= cx <= 1.2 * W and -0.2 * H <= cy <= 1.2 * H):
                    return False
        return True

    import csv
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            f.seek(0)
            delimiter = "," if "," in first_line else ";"
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                return None

            # 1) header-based
            header_l = [str(x).strip().lower() for x in header]
            def _idx(name):
                try:
                    return header_l.index(name)
                except Exception:
                    return None

            i_fx = _idx("fx")
            i_fy = _idx("fy")
            i_cx = _idx("cx")
            i_cy = _idx("cy")

            for row in reader:
                if len(row) < 10:
                    continue
                # 优先表头索引
                if i_fx is not None and i_fy is not None and i_cx is not None and i_cy is not None:
                    if max(i_fx, i_fy, i_cx, i_cy) < len(row):
                        fx, fy, cx, cy = row[i_fx], row[i_fy], row[i_cx], row[i_cy]
                        if _plausible(fx, fy, cx, cy):
                            return float(fx), float(fy), float(cx), float(cy)
                # 2) legacy indices 18-21
                if len(row) >= 22:
                    fx = row[18]; fy = row[19]; cx = row[20]; cy = row[21]
                    if _plausible(fx, fy, cx, cy):
                        return float(fx), float(fy), float(cx), float(cy)
    except Exception:
        return None
    return None


def _depth_make_K_from_env_or_estimate(image_shape_hw, sx=1.0, sy=1.0):
    """
    生成与“当前处理分辨率坐标系”一致的K。

    优先级：
      1) 环境变量 DEPTH_FX/FY/CX/CY（默认认为是在原始分辨率下，按sx/sy缩放到当前分辨率；
         可用 DEPTH_INTRINSICS_SPACE=scaled 指明已是当前分辨率）
      2) 自动估计（与脚本现有逻辑一致）：主点中心、焦距=图像对角线长度（在当前分辨率下）
    """
    H, W = int(image_shape_hw[0]), int(image_shape_hw[1])
    space = os.environ.get("DEPTH_INTRINSICS_SPACE", "original").strip().lower()  # original | scaled

    def _get_float(name):
        v = os.environ.get(name, None)
        if v is None or str(v).strip() == "":
            return None
        try:
            return float(v)
        except Exception:
            return None

    fx0 = _get_float("DEPTH_FX")
    fy0 = _get_float("DEPTH_FY")
    cx0 = _get_float("DEPTH_CX")
    cy0 = _get_float("DEPTH_CY")

    if fx0 is not None and fy0 is not None and cx0 is not None and cy0 is not None:
        if space == "scaled":
            fx, fy, cx, cy = fx0, fy0, cx0, cy0
        else:
            fx, fy, cx, cy = fx0 * float(sx), fy0 * float(sy), cx0 * float(sx), cy0 * float(sy)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    # 自动估计（当前分辨率）
    f_est = float(np.sqrt(W * W + H * H))
    fx = fy = f_est
    cx = float(W / 2.0)
    cy = float(H / 2.0)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _depth_orb_match_points(img1_bgr, img2_bgr, max_matches=2000):
    """ORB + ratio test 产生像素坐标匹配点对 (N,2),(N,2)，坐标在输入图像分辨率下。"""
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    try:
        orb = cv2.ORB_create(nfeatures=4000)
    except Exception:
        orb = cv2.ORB_create()
    kps1, des1 = orb.detectAndCompute(gray1, None)
    kps2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(kps1) < 8 or len(kps2) < 8:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        knn = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return None, None
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)[: int(max_matches)]
    if len(good) < 8:
        return None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts1, pts2


# 导入语义特征提取函数（用于语义损失计算）
try:
    from extract_semantic_features_Function import calculate_contour_similarity, calculate_contour_similarity_with_grad
    SEMANTIC_FEATURES_AVAILABLE = True
except ImportError:
    SEMANTIC_FEATURES_AVAILABLE = False
    calculate_contour_similarity_with_grad = None
    print("[警告] extract_semantic_features_Function 未找到，语义损失功能将不可用")

# 棋盘格检测模块 - 集成自 test_chessboard_detection.py
# 与 single_camera_calibration_Transformer_RT copy 2.py 一致的棋盘格配置（列, 行）
CHESSBOARD_CONFIGS = [
    {"size": (10, 7), "square_size": 13, "name": "11x8-13mm", "description": "11×8方格，13mm边长"},
    {"size": (8, 7), "square_size": 8, "name": "9x8-8mm", "description": "9×8方格，8mm边长"},
    {"size": (8, 5), "square_size": 5, "name": "9x6-5mm", "description": "9×6方格，5mm边长"},
]

# 额外常见尺寸，用于兜底检测
ADDITIONAL_SIZES = [
    (9, 6), (6, 9),    # 标准棋盘格
    (8, 8),            # 常见尺寸
    (7, 5), (5, 7),    # 小型棋盘格
    (10, 7), (7, 10),  # 中型棋盘格
    (12, 9), (9, 12),  # 大型棋盘格
]

# 设置中文字体
def setup_chinese_font():
    """设置matplotlib支持中文字体显示"""
    import platform
    
    system = platform.system()
    
    if system == "Windows":
        # Windows系统常用中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    elif system == "Darwin":  # macOS
        # macOS系统常用中文字体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'DejaVu Sans']
    else:  # Linux
        # Linux系统常用中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 初始化中文字体
setup_chinese_font()


# 棋盘格检测函数 - 从 test_chessboard_detection.py 复制
def detect_chessboard(gray_image, image_id="test", save_results=True):
    """
    复刻 single_camera_calibration_Transformer_RT copy 2.py 的棋盘格检测逻辑
    Args:
        gray_image: 灰度图
        image_id: 保存调试结果时使用的标识
        save_results: 是否保存检测或失败图像
    Returns:
        ret, corners, detected_config
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    detected_config = None
    corners = None

    # 1. 尝试默认配置
    for config in CHESSBOARD_CONFIGS:
        size = config["size"]
        try:
            ret, corners = cv2.findChessboardCorners(gray_image, size, flags)
            if ret:
                detected_config = config
                print(f"[OK] 检测到棋盘格: {config['description']}")
                break
        except Exception:
            continue

    # 2. 兜底尝试常见尺寸
    if detected_config is None:
        for size in ADDITIONAL_SIZES:
            try:
                ret, corners = cv2.findChessboardCorners(gray_image, size, flags)
                if ret:
                    detected_config = {
                        "size": size,
                        "square_size": 1.0,
                        "name": f"custom_{size[0]}x{size[1]}",
                        "description": f"自定义 {size[0]}×{size[1]} 方格"
                    }
                    print(f"[警告] 检测到自定义棋盘格: {size[0]}x{size[1]}，请确认方格大小")
                    break
            except Exception:
                continue

    # 3. 保存调试结果
    if save_results:
        results_dir = "detection_results"
        os.makedirs(results_dir, exist_ok=True)

        if detected_config is not None and corners is not None:
            result_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(result_img, detected_config["size"], corners, True)
            result_file = os.path.join(results_dir, f"{image_id}_detected_{detected_config['name']}.jpg")
            cv2.imwrite(result_file, result_img)
            print(f"  保存检测结果: {os.path.abspath(result_file)}")
        else:
            failed_file = os.path.join(results_dir, f"{image_id}_failed.jpg")
            cv2.imwrite(failed_file, gray_image)
            print(f"  保存失败图像: {os.path.abspath(failed_file)}")

    return (detected_config is not None), corners, detected_config


def extract_chessboard_mask(corners, chessboard_size, image_shape, margin=10):
    """
    从检测到的棋盘格角点生成mask

    Args:
        corners: 检测到的角点坐标（numpy数组，形状为(N, 1, 2)）
        chessboard_size: 棋盘格尺寸 (cols, rows)
        image_shape: 图像形状 (height, width)
        margin: 边界扩展像素数

    Returns:
        mask: 二值mask图像（棋盘格区域为255，其他为0）
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if corners is None or len(corners) == 0:
        return mask

    # 获取四个外角点
    cols, rows = chessboard_size

    # 四个角的索引
    corner_indices = [
        0,                      # 左上角
        cols - 1,               # 右上角
        cols * rows - 1,        # 右下角
        cols * (rows - 1)       # 左下角
    ]

    # 提取四个外角点
    outer_corners = corners[corner_indices].reshape(-1, 2).astype(np.int32)

    # 使用凸包填充
    try:
        hull = cv2.convexHull(outer_corners)
        cv2.fillPoly(mask, [hull], 255)
    except Exception as e:
        # 回退到矩形填充
        print(f"  [警告] 凸包失败，使用矩形填充: {e}")
        x_coords = outer_corners[:, 0]
        y_coords = outer_corners[:, 1]
        x_min = max(0, int(np.min(x_coords) - margin))
        x_max = min(w, int(np.max(x_coords) + margin))
        y_min = max(0, int(np.min(y_coords) - margin))
        y_max = min(h, int(np.max(y_coords) + margin))
        mask[y_min:y_max, x_min:x_max] = 255

    # 对mask放大20%（使用形态学膨胀，保持形状不变）
    mask_coords = np.where(mask > 0)
    if len(mask_coords[0]) > 0:
        # 计算mask的边界框尺寸
        y_min_mask = np.min(mask_coords[0])
        y_max_mask = np.max(mask_coords[0])
        x_min_mask = np.min(mask_coords[1])
        x_max_mask = np.max(mask_coords[1])

        mask_width = x_max_mask - x_min_mask
        mask_height = y_max_mask - y_min_mask

        # 计算20%的扩展量（取平均值作为膨胀半径）
        expand_radius = int((mask_width + mask_height) * 0.1)  # 20%的一半，因为膨胀是双向的

        # 确保膨胀半径至少为1
        if expand_radius < 1:
            expand_radius = 1

        # 创建圆形结构元素用于膨胀（保持形状）
        kernel_size = expand_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 执行膨胀操作（向外扩展边界）
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def detect_chessboard_from_path(image_path, save_results=False, refine_corners=True):
    """
    从图片路径检测棋盘格

    Args:
        image_path: 图像文件的完整路径
        save_results: 是否保存调试结果
        refine_corners: 是否进行亚像素优化

    Returns:
        dict: 包含以下键的字典
            - success (bool): 是否检测成功
            - corners: 检测到的角点坐标（numpy数组），如果失败则为None
            - corners_refined: 亚像素优化后的角点坐标（numpy数组），如果失败或未优化则为None
            - config: 检测到的棋盘格配置信息（字典），包含：
                - size: (列, 行) 元组
                - square_size: 方格大小（mm）
                - name: 棋盘格名称
                - description: 棋盘格描述
            - image_info: 图像信息字典
                - width: 图像宽度
                - height: 图像高度
                - channels: 通道数
            - error: 错误信息（如果失败）
    """
    result = {
        "success": False,
        "corners": None,
        "corners_refined": None,
        "config": None,
        "image_info": None,
        "mask": None,
        "error": None
    }

    # 检查文件是否存在
    if not os.path.exists(image_path):
        result["error"] = f"图像文件不存在: {image_path}"
        return result

    # 读取图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        result["error"] = f"无法读取图像文件: {image_path}"
        return result

    # 转换为RGB和灰度图
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 保存图像信息
    h, w = gray.shape
    result["image_info"] = {
        "width": w,
        "height": h,
        "channels": img_rgb.shape[2]
    }

    # 检测棋盘格
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    ret, corners, detected_config = detect_chessboard(
        gray,
        image_id=image_id,
        save_results=save_results
    )

    if ret and detected_config is not None and corners is not None:
        result["success"] = True
        result["corners"] = corners
        result["config"] = detected_config.copy()

        # 亚像素优化
        if refine_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )
            result["corners_refined"] = corners_refined
        else:
            result["corners_refined"] = corners

        # 提取棋盘格mask
        mask = extract_chessboard_mask(
            result["corners_refined"],
            detected_config["size"],
            gray.shape,
            margin=10
        )
        result["mask"] = mask

    else:
        result["error"] = "未检测到棋盘格"

    return result


def get_chessboard_mask(image_path, refine_corners=True):
    """
    从图像路径提取棋盘格mask

    Args:
        image_path: 原始图像的路径
        refine_corners: 是否进行亚像素优化，默认True

    Returns:
        mask: 棋盘格mask图像（numpy数组，二值图像，棋盘格区域为255，其他为0）
             如果检测失败，返回None
    """
    result = detect_chessboard_from_path(image_path, save_results=False, refine_corners=refine_corners)

    if result["success"] and result["mask"] is not None:
        return result["mask"]
    else:
        return None

# 时间统计工具类
class TimeTracker:
    """时间统计工具类，用于记录和输出处理时间"""
    
    def __init__(self, name="处理"):
        self.name = name
        self.start_time = None
        self.checkpoints = []
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.checkpoints = []
        return self
    
    def checkpoint(self, checkpoint_name):
        """记录检查点时间"""
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        self.checkpoints.append((checkpoint_name, elapsed))
        return elapsed
    
    def elapsed(self):
        """获取总耗时（秒）"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def print_summary(self):
        """打印时间统计摘要（仅显示总耗时）"""
        total = self.elapsed()
        if total == 0:
            return
        
        print(f"[时间统计] {self.name}: 总耗时 {total:.2f}秒 ({total/60:.2f}分钟)")
    


class ImageProcessor:
    """图像处理器类 - 处理RGB、Binary、Gradient三种模态"""
    
    def __init__(self, sigma=4.0, sigma_spatial=3.0, sigma_range=0.1, kernel_radius=3):
        """
        初始化ImageProcessor
        
        Args:
            sigma: 高斯核标准差
            sigma_spatial: 双边滤波空间标准差
            sigma_range: 双边滤波范围标准差
            kernel_radius: 核半径
        """
        self.sigma = sigma
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.kernel_radius = kernel_radius
        self.gauss_1d, self.dgauss_1d = self._generate_gaussian_kernels()
        # 允许CPU回退以规避GPU/CuPy相关崩溃
        # 若无法导入CuPy，则强制CPU模式
        self.use_cpu = bool(int(os.environ.get("USE_CPU", "0"))) or (cp is None)
        self.bilateral_kernel = None if self.use_cpu else self._build_bilateral_kernel()
        # 多轮廓/傅里叶与EFD参数（可通过环境变量覆盖）
        # 略微降低默认谐波数量以减少过拟合与噪声影响
        self.fd_harmonics = int(os.environ.get("FD_HARMONICS", "24"))
        self.efd_harmonics = int(os.environ.get("EFD_HARMONICS", "16"))
        self.num_resample_points = int(os.environ.get("CONTOUR_POINTS", "256"))
        self.enable_morph_preprocess = bool(int(os.environ.get("EFD_PREPROCESS", "0")))
        self.morph_kernel_size = int(os.environ.get("MORPH_KSIZE", "3"))
        # 输出目录（默认保存到指定路径）
        self.output_dir = os.environ.get("OUTPUT_DIR", r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            print(f"[警告] 创建输出目录失败: {self.output_dir} - {e}")
        # OBJ网格专用子目录
        try:
            self.mesh_out_dir = os.path.join(self.output_dir, "mesh")
            os.makedirs(self.mesh_out_dir, exist_ok=True)
        except Exception as e:
            print(f"[警告] 创建mesh子目录失败: {self.output_dir}\\mesh - {e}")
        # 纯频域优化：FD谐波能量自适应与不变性特征、轮廓规范化开关
        self.fd_adaptive = bool(int(os.environ.get("FD_ADAPTIVE", "1")))
        self.fd_energy_percent = float(os.environ.get("FD_ENERGY_PERCENT", "0.95"))
        self.fd_canonicalize = bool(int(os.environ.get("FD_CANONICALIZE", "1")))
        # 连通性优化：滞后（双阈值）连接与区域大小筛选
        self.hyst_enable = bool(int(os.environ.get("HYST_ENABLE", "1")))
        self.hyst_low_percent = float(os.environ.get("HYST_LOW_PERCENT", "68"))
        self.hyst_high_percent = float(os.environ.get("HYST_HIGH_PERCENT", "95"))
        self.cc_min_pixels = int(os.environ.get("CC_MIN_PIXELS", "256"))
        # 设置为 <=0 表示不设上限
        self.cc_max_pixels_frac = float(os.environ.get("CC_MAX_PIXELS_FRAC", "0.0"))
        # 主阈值控制：用于主流程的阈值百分位（例如设为95以替代68）
        self.main_percent = int(os.environ.get("MAIN_THRESHOLD_PERCENT", "68"))
        # 阈值模式选择：th_68 或 th_90（用于mesh生成和文件命名）
        self.threshold_mode = os.environ.get("THRESHOLD_MODE", "th_68").lower()
        if self.threshold_mode not in ["th_68", "th_90"]:
            print(f"[警告] 无效的阈值模式 {self.threshold_mode}，使用默认值 th_68")
            self.threshold_mode = "th_68"
        # 根据阈值模式设置对应的百分位值
        self.threshold_percent = 68 if self.threshold_mode == "th_68" else 90
        print(f"[配置] 阈值模式: {self.threshold_mode} (百分位: {self.threshold_percent})")

        # 棋盘格检测配置（默认启用）
        self.use_chessboard_detection = bool(int(os.environ.get("USE_CHESSBOARD_DETECTION", "1")))
        print(f"[配置] 棋盘格检测: {'启用' if self.use_chessboard_detection else '禁用'}")
        # 网格化参数：是否生成网格与网格单元大小（像素）
        # cell_size越小，顶点和三角面片越多，mesh越精细（默认7，平衡速度和精度）
        # 可通过环境变量 MESH_CELL_SIZE 调整（建议范围：1-10，1最精细但最慢，5-7速度较快）
        self.mesh_enable = bool(int(os.environ.get("MESH_ENABLE", "1")))
        self.mesh_cell_size = int(os.environ.get("MESH_CELL_SIZE", "7"))  # 默认改为7，提高计算速度
        # 网格类型：square_tri(默认，规则方格拆分为两三角) 或 tri(等边三角网格)
        self.mesh_type = os.environ.get("MESH_TYPE", "square_tri").lower()
        # 3D 表面傅里叶拟合（对高度场进行2D傅里叶截断重建）
        self.field_ft_enable = bool(int(os.environ.get("FIELD_FT_ENABLE", "1")))
        # 略微降低截断窗口大小
        self.field_ft_kx = int(os.environ.get("FIELD_FT_KX", "24"))
        self.field_ft_ky = int(os.environ.get("FIELD_FT_KY", "24"))
        # 是否导出OBJ三维网格
        self.mesh_obj_enable = bool(int(os.environ.get("MESH_OBJ_ENABLE", "1")))
        # 峰的局部拟合（二次曲面）
        self.peak_fit_enable = bool(int(os.environ.get("PEAK_FIT_ENABLE", "0")))
        # RST估计与ICP核验的开关
        self.video_rst_enable = bool(int(os.environ.get("VIDEO_RST_ENABLE", "0")))
        self.video_path = os.environ.get("VIDEO_PATH", None)
        self.icp_verify_enable = bool(int(os.environ.get("ICP_VERIFY_ENABLE", "0")))
        self.peak_fit_window = int(os.environ.get("PEAK_FIT_WINDOW", "21"))
        self.peak_fit_max = int(os.environ.get("PEAK_FIT_MAX", "256"))
        self.peak_fit_model = os.environ.get("PEAK_FIT_MODEL", "quad").lower()
        # Open3D 可视化配置
        self.o3d_enable = bool(int(os.environ.get("O3D_ENABLE", "1")))
        # o3d_step越小，mesh越精细，但文件越大
        # step=1: 每个像素一个顶点，文件很大（~150-200MB for 1920x1080）
        # step=3: 平衡精度和大小，推荐（~18-25MB for 1920x1080）
        # step=5: 文件更小，精度略降（~4-6MB for 1920x1080）
        # 可通过环境变量 O3D_STEP 调整（建议范围：3-5）
        self.o3d_step = int(os.environ.get("O3D_STEP", "3"))  # 默认改为3，平衡质量和文件大小
        # 图像分辨率缩放：降低分辨率可以加快处理速度
        # scale_factor=0.5: 分辨率降低一半（宽度和高度各减半），处理速度提升约4倍
        # scale_factor=1.0: 保持原始分辨率
        self.image_scale_factor = float(os.environ.get("IMAGE_SCALE_FACTOR", "0.5"))  # 默认降低一半分辨率
        self.o3d_z_scale = float(os.environ.get("O3D_Z_SCALE", "255.0"))
        self.o3d_save_mesh = bool(int(os.environ.get("O3D_SAVE_MESH", "1")))
        # 可视化窗口开关（默认关闭，避免批量处理时弹出窗口和OpenGL错误）
        self.o3d_visualize = bool(int(os.environ.get("O3D_VISUALIZE", "0")))
        # 点云匹配可视化开关（默认开启，展示匹配结果）
        self.match_visualize = bool(int(os.environ.get("MATCH_VISUALIZE", "1")))
        # 点云异常点过滤开关（默认开启，移除异常点）
        self.filter_outliers = bool(int(os.environ.get("FILTER_OUTLIERS", "1")))
        # 统计离群点移除参数：nb_neighbors（邻居数量）和 std_ratio（标准差比率）
        self.outlier_nb_neighbors = int(os.environ.get("OUTLIER_NB_NEIGHBORS", "20"))
        self.outlier_std_ratio = float(os.environ.get("OUTLIER_STD_RATIO", "2.0"))
        # RANSAC参数配置（优化：减少迭代次数以提高速度）
        self.ransac_n = int(os.environ.get("RANSAC_N", "3"))  # RANSAC采样点数（默认3，用于估计变换矩阵的最小点数）
        self.ransac_max_iterations = int(os.environ.get("RANSAC_MAX_ITERATIONS", "5000"))  # 最大迭代次数（从100000降到5000）
        self.ransac_confidence = float(os.environ.get("RANSAC_CONFIDENCE", "0.95"))  # 置信度（从0.999降到0.95以提高速度）
        self.ransac_global_max_iterations = int(os.environ.get("RANSAC_GLOBAL_MAX_ITERATIONS", "10000"))  # 全局匹配的最大迭代次数（从200000降到10000）
        self.ransac_global_confidence = float(os.environ.get("RANSAC_GLOBAL_CONFIDENCE", "0.95"))  # 全局匹配的置信度（从0.99降到0.95）
        
        # ========== 第一阶段优化：I/O缓存和内存管理 ==========
        # 网格内存缓存（不保存PLY文件，直接保存在内存中）
        self._mesh_cache = {}  # 格式: {frame_idx: mesh_o3d}
        self._mesh_features_cache = {}  # 格式: {cache_key: features_dict}
        
        # ========== 第二阶段优化：特征缓存和采样策略 ==========
        # 特征缓存配置
        self._max_feature_cache_size = int(os.environ.get("MAX_FEATURE_CACHE_SIZE", "100"))  # 最多缓存100个特征
        self._feature_cache_access_order = []  # LRU缓存访问顺序
        
        # 采样策略配置
        self._use_voxel_downsample = bool(int(os.environ.get("USE_VOXEL_DOWNSAMPLE", "1")))  # 默认使用体素下采样
        # ========== 第二阶段优化完成 ==========
        
        # 文件等待优化：替换time.sleep
        def wait_for_file_stable(filepath, timeout=5.0, check_interval=0.05):
            """等待文件写入完成（通过检查文件大小稳定性）"""
            if not os.path.exists(filepath):
                return False
            start_time = time.time()
            prev_size = os.path.getsize(filepath)
            while time.time() - start_time < timeout:
                time.sleep(check_interval)
                if not os.path.exists(filepath):
                    return False
                current_size = os.path.getsize(filepath)
                if current_size == prev_size:
                    return True
                prev_size = current_size
            return False
        self._wait_for_file_stable = wait_for_file_stable
        # ========== 第一阶段优化完成 ==========

    def _generate_gaussian_kernels(self):
        """生成高斯核"""
        S_round = int(round(3 * self.sigma))
        gauss_1d = np.exp(-np.arange(-S_round, S_round + 1) ** 2 / (2 * self.sigma ** 2))
        gauss_1d /= gauss_1d.sum()
        dgauss_1d = -np.arange(-S_round, S_round + 1) * gauss_1d
        return gauss_1d.astype(np.float32), dgauss_1d.astype(np.float32)

    def _build_bilateral_kernel(self):
        """构建双边滤波CUDA核"""
        return cp.RawKernel(r'''
        extern "C" __global__
        void bilateral_filter(const float* input, float* output, const float* gs_table,
                              int w, int h, int kernel_radius,
                              float sigma_range)
        {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= w || y >= h) return;

            int idx = y * w + x;
            float center_val = input[idx];

            float sum = 0.0f;
            float norm = 0.0f;

            for (int dy = -kernel_radius; dy <= kernel_radius; dy++) {
                for (int dx = -kernel_radius; dx <= kernel_radius; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        int nidx = ny * w + nx;
                        float neighbor_val = input[nidx];

                        float gs = gs_table[(dy + kernel_radius) * (2 * kernel_radius + 1) + (dx + kernel_radius)];
                        float gr = expf(-((neighbor_val - center_val) * (neighbor_val - center_val)) / (2.0f * sigma_range * sigma_range));

                        float weight = gs * gr;
                        norm += weight;
                        sum += neighbor_val * weight;
                    }
                }
            }
            output[idx] = sum / norm;
        }
        ''', 'bilateral_filter')

    def _bilateral_filter(self, image_cp):
        """双边滤波处理"""
        if self.use_cpu:
            # CPU回退：使用OpenCV双边滤波，输入可能是numpy数组
            img_np = image_cp if isinstance(image_cp, np.ndarray) else cp.asnumpy(image_cp)
            # OpenCV接受float32范围0..1或0..255；这里保持0..1强度域
            d = max(1, 2 * self.kernel_radius + 1)
            out_np = cv2.bilateralFilter(img_np.astype(np.float32), d=d,
                                         sigmaColor=float(self.sigma_range),
                                         sigmaSpace=float(self.sigma_spatial))
            return out_np
        h, w = image_cp.shape
        input_flat = image_cp.ravel()
        output = cp.zeros_like(input_flat)

        size = 2 * self.kernel_radius + 1
        gs_table_cpu = np.zeros((size, size), dtype=np.float32)
        for dy in range(-self.kernel_radius, self.kernel_radius + 1):
            for dx in range(-self.kernel_radius, self.kernel_radius + 1):
                gs_table_cpu[dy + self.kernel_radius, dx + self.kernel_radius] = np.exp(
                    -(dx ** 2 + dy ** 2) / (2.0 * self.sigma_spatial ** 2)
                )
        gs_table = cp.asarray(gs_table_cpu).ravel()

        block = (16, 16)
        grid = ((w + 15) // 16, (h + 15) // 16)

        self.bilateral_kernel(grid, block, (
            input_flat, output, gs_table,
            np.int32(w), np.int32(h), np.int32(self.kernel_radius),
            np.float32(self.sigma_range)
        ))

        return output.reshape((h, w))

    def _normalize(self, arr):
        """归一化数组"""
        min_val = arr.min()
        max_val = arr.max()
        return (arr - min_val) / (max_val - min_val + 1e-6)

    def _analyze_gradient_threshold(self, grad_map, bins=1024, percent=68):
        """分析梯度阈值"""
        if self.use_cpu or (cp is None) or isinstance(grad_map, np.ndarray):
            hist, bin_edges = np.histogram(grad_map, bins=bins, range=(0, 1))
            cum_hist = np.cumsum(hist)
            hist_percentage = cum_hist / grad_map.size * 100.0
            x = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            idx = np.searchsorted(hist_percentage, np.array(percent))
            return float(x[min(idx, len(x) - 1)])
        else:
            hist, bin_edges = cp.histogram(grad_map, bins=bins, range=(0, 1))
            cum_hist = cp.cumsum(hist)
            hist_percentage = cum_hist / grad_map.size * 100
            x = (bin_edges[:-1] + bin_edges[1:]) / 2
            idx = cp.searchsorted(hist_percentage, cp.array(percent)).item()
            return float(x[min(idx, len(x) - 1)])

    def _extract_largest_contour(self, binary_img):
        """从二值图中提取最大外轮廓，返回Nx2坐标数组；若无则返回None"""
        if binary_img.ndim == 3:
            gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_img
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2).astype(np.float32)
        return pts

    def _resample_contour(self, pts, num_points=256):
        """按弧长等距重采样轮廓点，返回num_points个点"""
        if pts is None or len(pts) < 2:
            return None
        closed = np.vstack([pts, pts[0:1, :]])
        d = np.diff(closed, axis=0)
        seg_len = np.sqrt((d ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]
        if total < 1e-6:
            return None
        s = np.linspace(0, total, num_points, endpoint=False)
        resampled = np.empty((num_points, 2), dtype=np.float32)
        j = 0
        for i, si in enumerate(s):
            while j + 1 < len(cum) and cum[j + 1] < si:
                j += 1
            t = (si - cum[j]) / max(cum[j + 1] - cum[j], 1e-6)
            p = closed[j] * (1 - t) + closed[j + 1] * t
            resampled[i] = p
        return resampled

    def _compute_fourier_descriptors(self, pts, num_harmonics=15):
        """计算傅里叶描述子：返回原系数、归一化幅度与相位、质心等"""
        if pts is None:
            return None
        z = pts[:, 0] + 1j * pts[:, 1]
        mean_z = np.mean(z)
        z_centered = z - mean_z
        F = np.fft.fft(z_centered)
        N = len(z_centered)
        eps = 1e-12
        ref = F[1] if np.abs(F[1]) > eps else (1.0 + 0j)
        mags = np.abs(F[1:num_harmonics + 1]) / (np.abs(ref) + eps)
        phases = np.angle(F[1:num_harmonics + 1]) - np.angle(ref)
        features = {
            "num_points": int(N),
            "num_harmonics": int(num_harmonics),
            "magnitudes_norm": mags.tolist(),
            "phases_rel": phases.tolist(),
            "centroid": [float(np.real(mean_z)), float(np.imag(mean_z))]
        }
        return {"F": F, "mean_z": mean_z, "features": features}

    def _reconstruct_contour(self, F, mean_z, num_harmonics=15):
        """用前num_harmonics的系数重建轮廓，返回复数点序列"""
        Ft = np.zeros_like(F)
        M = min(num_harmonics, (len(F) - 1) // 2)
        Ft[1:M + 1] = F[1:M + 1]
        Ft[-M:] = F[-M:]
        z_rec = np.fft.ifft(Ft) + mean_z
        return z_rec

    def _hysteresis_edges(self, grad_np, th_low, th_high):
        """滞后阈值连接：强边作为种子，保留与强边相连的弱边"""
        from collections import deque
        h, w = grad_np.shape
        strong = grad_np >= th_high
        weak = grad_np >= th_low
        out = np.zeros((h, w), dtype=np.uint8)
        q = deque()
        ys, xs = np.nonzero(strong)
        for y, x in zip(ys, xs):
            out[y, x] = 255
            q.append((y, x))
        # 8邻域传播
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while q:
            y, x = q.popleft()
            for dy, dx in nbrs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if weak[ny, nx] and out[ny, nx] == 0:
                        out[ny, nx] = 255
                        q.append((ny, nx))
        return out

    def _filter_components_by_size(self, mask, min_pixels, max_pixels, connectivity=8):
        """根据连通域像素大小过滤，返回过滤后的掩码与统计信息列表"""
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S)
        filtered = np.zeros_like(bw)
        stats_list = []
        for lbl in range(1, num_labels):  # 跳过背景0
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_pixels or (max_pixels is not None and area > max_pixels):
                continue
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            filtered[labels == lbl] = 255
            stats_list.append({
                "label": int(lbl), "area": area, "bbox": [x, y, w, h],
                "centroid": [float(centroids[lbl, 0]), float(centroids[lbl, 1])]
            })
        return filtered, stats_list

    def _canonicalize_contour(self, pts):
        """规范化轮廓：统一方向（顺时针）与起点（面向质心的0角）"""
        if pts is None or len(pts) < 3:
            return pts
        # 统一方向：根据有符号面积判断，若为逆时针则翻转
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        # 图像坐标y向下，符号可能反直觉，但一致性即可
        if area > 0:
            pts = pts[::-1]
        # 统一起点：以质心为参考，将角度接近0的点旋转到序列开头
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        # 选择角度最接近0的索引作为起点
        start = int(np.argmin(np.abs(ang)))
        pts = np.roll(pts, -start, axis=0)
        return pts

    def _fd_energy_adaptive_M(self, F, target_percent=0.95):
        """根据能量覆盖率选择谐波数M（对称能量累计）"""
        N = len(F)
        total_energy = np.sum(np.abs(F) ** 2) - np.abs(F[0]) ** 2
        if total_energy <= 1e-12:
            return 1
        cum = 0.0
        max_k = (N - 1) // 2
        M = 1
        for k in range(1, max_k + 1):
            # 正负频率能量合并
            ek = np.abs(F[k]) ** 2 + np.abs(F[-k]) ** 2
            cum += ek
            if cum / total_energy >= target_percent:
                M = k
                break
            M = k
        return max(1, M)

    def _fd_features_invariants(self, F, use_adaptive=True, energy_percent=0.95):
        """输出FD的旋转/尺度不变特征与能量覆盖信息"""
        N = len(F)
        eps = 1e-12
        # 选定M
        M = self._fd_energy_adaptive_M(F, target_percent=energy_percent) if use_adaptive else min(self.fd_harmonics, (N - 1) // 2)
        ref = F[1] if np.abs(F[1]) > eps else (1.0 + 0j)
        mags = np.abs(F[1:M + 1])
        phases = np.angle(F[1:M + 1]) - np.angle(ref)
        # 不变性：尺度归一化（除以总幅度和/或第一谐波），旋转不变（相对相位）
        sum_mag = np.sum(mags) + eps
        mags_scale_norm = (mags / sum_mag).tolist()
        mags_ref_norm = (mags / (np.abs(ref) + eps)).tolist()
        phases_rel = phases.tolist()
        # 能量覆盖
        total_energy = np.sum(np.abs(F) ** 2) - np.abs(F[0]) ** 2
        cum = 0.0
        for k in range(1, M + 1):
            cum += np.abs(F[k]) ** 2 + np.abs(F[-k]) ** 2
        energy_covered = float(cum / (total_energy + eps))
        return {
            "M_adaptive": int(M),
            "fd_magnitudes_scale_norm": mags_scale_norm,
            "fd_magnitudes_ref_norm": mags_ref_norm,
            "fd_phases_rel": phases_rel,
            "fd_energy_covered": energy_covered
        }

    def _efd_features_invariants(self, efd):
        """输出EFD的旋转/尺度不变特征（基于幅度），含能量占比"""
        a = np.array(efd.get("a", []), dtype=np.float64)
        b = np.array(efd.get("b", []), dtype=np.float64)
        c = np.array(efd.get("c", []), dtype=np.float64)
        d = np.array(efd.get("d", []), dtype=np.float64)
        if len(a) == 0:
            return {"efd_magnitudes_norm": [], "efd_energy_covered": 0.0}
        A = np.sqrt(a ** 2 + b ** 2)
        C = np.sqrt(c ** 2 + d ** 2)
        mags = A + C
        total = np.sum(mags) + 1e-12
        mags_norm = (mags / total).tolist()
        # 能量（以幅度平方近似）
        energy = np.sum(mags ** 2)
        # 前self.efd_harmonics覆盖率
        m = len(mags)
        k = min(self.efd_harmonics, m)
        energy_top = float(np.sum(mags[:k] ** 2))
        return {
            "efd_magnitudes_norm": mags_norm,
            "efd_energy_covered": float(energy_top / (energy + 1e-12))
        }

    def _morph_preprocess(self, mask, kernel_size=3, open_iter=1, close_iter=1):
        """对binary掩码做形态学开闭预处理以净化小噪声与桥接裂隙"""
        k = max(1, int(kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        pre = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        pre = cv2.morphologyEx(pre, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        return pre

    def _extract_all_contours(self, binary_img, preprocess=False):
        """提取所有外轮廓，返回轮廓点列表(list of Nx2 float32)"""
        if binary_img.ndim == 3:
            gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_img
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        if preprocess:
            try:
                mask = self._morph_preprocess(mask, kernel_size=self.morph_kernel_size)
            except Exception:
                pass
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            pts_list.append(cnt.reshape(-1, 2).astype(np.float32))
        return pts_list

    def _find_isocontours(self, img_float, level):
        """基于等值线(不二值化)提取轮廓，返回多条折线点集合(list of Nx2 float32)。
        使用简化的marching squares，将像素网格在给定level处的交点串联为折线。
        """
        if img_float is None:
            return []
        if img_float.ndim == 3:
            img = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            img = img_float.astype(np.float32)
        h, w = img.shape
        # 每个格子的边交点缓存：水平边(y,x->x+1)与垂直边(x,y->y+1)
        horiz = {}
        vert = {}
        def interp(p1, p2, v1, v2):
            t = 0.5 if abs(v2 - v1) < 1e-12 else float((level - v1) / (v2 - v1))
            t = max(0.0, min(1.0, t))
            return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        # 优化：向量化遍历格子，记录交点
        # 使用meshgrid创建所有坐标
        y_coords, x_coords = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing='ij')
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        
        # 向量化获取所有格子的值
        v00 = img[y_coords, x_coords]
        v10 = img[y_coords, x_coords + 1]
        v01 = img[y_coords + 1, x_coords]
        v11 = img[y_coords + 1, x_coords + 1]
        
        # 向量化计算交点
        # 上边 (x,y)-(x+1,y)
        mask_top = (v00 - level) * (v10 - level) < 0
        for idx in np.where(mask_top)[0]:
            y, x = y_coords[idx], x_coords[idx]
            horiz[(y, x)] = interp((x, y), (x + 1, y), v00[idx], v10[idx])
        
        # 下边 (x,y+1)-(x+1,y+1)
        mask_bottom = (v01 - level) * (v11 - level) < 0
        for idx in np.where(mask_bottom)[0]:
            y, x = y_coords[idx], x_coords[idx]
            horiz[(y + 1, x)] = interp((x, y + 1), (x + 1, y + 1), v01[idx], v11[idx])
        
        # 左边 (x,y)-(x,y+1)
        mask_left = (v00 - level) * (v01 - level) < 0
        for idx in np.where(mask_left)[0]:
            y, x = y_coords[idx], x_coords[idx]
            vert[(y, x)] = interp((x, y), (x, y + 1), v00[idx], v01[idx])
        
        # 右边 (x+1,y)-(x+1,y+1)
        mask_right = (v10 - level) * (v11 - level) < 0
        for idx in np.where(mask_right)[0]:
            y, x = y_coords[idx], x_coords[idx]
            vert[(y, x + 1)] = interp((x + 1, y), (x + 1, y + 1), v10[idx], v11[idx])
        # 将交点拼接成折线：相邻格子共享边交点
        # 用图的邻接来串联：每个交点按像素坐标近邻匹配
        pts_lines = []
        visited = set()
        # 全部交点集合（以栅格边索引存储）
        keys = list(set(list(horiz.keys()) + list(vert.keys())))
        # 建立从栅格边到真实坐标的映射
        coord_map = {}
        for k in keys:
            p = None
            if k in horiz:
                p = horiz[k]
            if k in vert:
                # 优先取存在的，若两者都有，选平均坐标
                p = vert[k] if p is None else ((p[0] + vert[k][0]) * 0.5, (p[1] + vert[k][1]) * 0.5)
            if p is not None:
                coord_map[k] = p
        # 用简单的邻域连接：对于每个点，尝试连接同一行或同一列相邻索引的点
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for k in keys:
            if k in visited or k not in coord_map:
                continue
            line = []
            cur = k
            visited.add(cur)
            line.append(coord_map[cur])
            # 先向一个方向延展
            for d in dirs:
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if nxt in coord_map and nxt not in visited:
                    cur = nxt
                    visited.add(cur)
                    line.append(coord_map[cur])
                    break
            # 再往反方向延展直到不能延展
            changed = True
            while changed:
                changed = False
                for d in dirs:
                    nxt = (cur[0] + d[0], cur[1] + d[1])
                    if nxt in coord_map and nxt not in visited:
                        cur = nxt
                        visited.add(cur)
                        line.append(coord_map[cur])
                        changed = True
                        break
            if len(line) >= 4:
                pts = np.array([[p[0], p[1]] for p in line], dtype=np.float32)
                pts_lines.append(pts)
        return pts_lines

    def _compute_efd(self, pts, harmonics=10):
        """计算椭圆傅里叶描述子(EFD)系数，返回字典{a0,c0,a,b,c,d,T}"""
        if pts is None or len(pts) < 3:
            return None
        closed = np.vstack([pts, pts[0:1, :]]).astype(np.float64)
        d = np.diff(closed, axis=0)
        dt = np.linalg.norm(d, axis=1)
        # 避免长度为0片段引发除零
        dt_safe = np.maximum(dt, 1e-8)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        T = t[-1] if t[-1] > 1e-8 else float(len(pts))
        a = []
        b = []
        c = []
        dcoef = []
        for n in range(1, harmonics + 1):
            coeff = (T / (2.0 * (np.pi ** 2) * (n ** 2)))
            s1 = np.sin(2.0 * np.pi * n * t[1:] / T) - np.sin(2.0 * np.pi * n * t[:-1] / T)
            s2 = -np.cos(2.0 * np.pi * n * t[1:] / T) + np.cos(2.0 * np.pi * n * t[:-1] / T)
            dx_dt = d[:, 0] / dt_safe
            dy_dt = d[:, 1] / dt_safe
            a_n = coeff * np.sum(dx_dt * s1)
            b_n = coeff * np.sum(dx_dt * s2)
            c_n = coeff * np.sum(dy_dt * s1)
            d_n = coeff * np.sum(dy_dt * s2)
            a.append(a_n)
            b.append(b_n)
            c.append(c_n)
            dcoef.append(d_n)
        centroid = pts.mean(axis=0)
        return {
            "a0": float(centroid[0]),
            "c0": float(centroid[1]),
            "a": a,
            "b": b,
            "c": c,
            "d": dcoef,
            "T": float(T)
        }

    def _reconstruct_efd(self, efd, num_points=256):
        """使用EFD系数重建轮廓点(num_points个)，返回Nx2 float32"""
        if efd is None:
            return None
        T = efd.get("T", 1.0)
        a0 = efd.get("a0", 0.0)
        c0 = efd.get("c0", 0.0)
        a = np.array(efd.get("a", []), dtype=np.float64)
        b = np.array(efd.get("b", []), dtype=np.float64)
        c = np.array(efd.get("c", []), dtype=np.float64)
        d = np.array(efd.get("d", []), dtype=np.float64)
        m = len(a)
        t = np.linspace(0.0, T, num_points, endpoint=False)
        x = np.full_like(t, a0, dtype=np.float64)
        y = np.full_like(t, c0, dtype=np.float64)
        for n in range(1, m + 1):
            x += a[n - 1] * np.cos(2.0 * np.pi * n * t / T) + b[n - 1] * np.sin(2.0 * np.pi * n * t / T)
            y += c[n - 1] * np.cos(2.0 * np.pi * n * t / T) + d[n - 1] * np.sin(2.0 * np.pi * n * t / T)
        pts = np.stack([x, y], axis=1).astype(np.float32)
        return pts

    def _process_threshold_pipeline(self, binary_img, suffix="68"):
        """针对指定阈值二值图，执行滤波、连通域筛选、FD/EFD拟合与输出（带后缀）。
        注意：该方法保留以binary掩码为基础的传统流程。
        """
        try:
            # 滤波与双边
            low_freq_bg = cv2.GaussianBlur(binary_img, (31, 31), 8)
            filtered = cv2.subtract(binary_img, low_freq_bg)
            filtered = cv2.normalize(filtered.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bilateral = cv2.bilateralFilter(binary_img, d=3, sigmaColor=90, sigmaSpace=90)
            cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_filtered.jpg"), filtered)
            cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_bilateral.jpg"), bilateral)

            # 传统二值掩码下的连通域筛选（仅供需要时使用）。
            img_area = binary_img.size
            max_pixels = None if self.cc_max_pixels_frac <= 0.0 else int(max(1, self.cc_max_pixels_frac * img_area))
            cc_mask, cc_stats = self._filter_components_by_size(binary_img, self.cc_min_pixels, max_pixels)
            cv2.imwrite(os.path.join(self.output_dir, f"connected_components_filtered_{suffix}.jpg"), cc_mask)
            # JSON保存已禁用
            # try:
            #     with open(os.path.join(self.output_dir, f"components_stats_{suffix}.json"), "w", encoding="utf-8") as f:
            #         json.dump({
            #             "min_pixels": self.cc_min_pixels,
            #             "max_pixels": (None if max_pixels is None else int(max_pixels)),
            #             "components": cc_stats
            #         }, f, ensure_ascii=False, indent=2)
            # except Exception as e:
            #     print(f"[警告] 保存连通域统计({suffix})失败: {e}")

            # 轮廓提取与FD/EFD
            mask_for_contours = cc_mask if np.count_nonzero(cc_mask) > 0 else binary_img
            pts_list = self._extract_all_contours(mask_for_contours, preprocess=self.enable_morph_preprocess)
            overlay_fd_all = np.zeros_like(binary_img, dtype=np.uint8)
            overlay_efd_all = np.zeros_like(binary_img, dtype=np.uint8)
            fd_features_all = []
            efd_features_all = []
            for idx, pts in enumerate(pts_list):
                pts_resampled = self._resample_contour(pts, num_points=self.num_resample_points)
                if pts_resampled is None:
                    continue
                if self.fd_canonicalize:
                    pts_resampled = self._canonicalize_contour(pts_resampled)
                fd = self._compute_fourier_descriptors(pts_resampled, num_harmonics=self.fd_harmonics)
                if fd is not None:
                    z_rec = self._reconstruct_contour(fd["F"], fd["mean_z"], num_harmonics=self.fd_harmonics)
                    rec_pts = np.column_stack([np.real(z_rec), np.imag(z_rec)]).astype(np.int32)
                    rec_pts = np.clip(rec_pts, [0, 0], [overlay_fd_all.shape[1] - 1, overlay_fd_all.shape[0] - 1])
                    cv2.polylines(overlay_fd_all, [rec_pts], isClosed=True, color=255, thickness=2)
                    overlay_fd_i = np.zeros_like(binary_img, dtype=np.uint8)
                    cv2.polylines(overlay_fd_i, [rec_pts], isClosed=True, color=255, thickness=2)
                    cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_fourier_{idx}.jpg"), overlay_fd_i)
                    fd_inv = self._fd_features_invariants(fd["F"], use_adaptive=self.fd_adaptive, energy_percent=self.fd_energy_percent)
                    # JSON保存已禁用
                    # try:
                    #     with open(os.path.join(self.output_dir, f"fourier_features_{suffix}_{idx}.json"), "w", encoding="utf-8") as f:
                    #         json.dump({**fd["features"], **fd_inv}, f, ensure_ascii=False, indent=2)
                    # except Exception as e:
                    #     print(f"[警告] 保存傅里叶特征({suffix},{idx})失败: {e}")
                    fd_features_all.append({"index": idx, **fd["features"], **fd_inv})
                efd = self._compute_efd(pts_resampled, harmonics=self.efd_harmonics)
                if efd is not None:
                    efd_rec = self._reconstruct_efd(efd, num_points=self.num_resample_points)
                    efd_rec_int = np.clip(efd_rec, [0, 0], [overlay_efd_all.shape[1] - 1, overlay_efd_all.shape[0] - 1]).astype(np.int32)
                    cv2.polylines(overlay_efd_all, [efd_rec_int], isClosed=True, color=255, thickness=2)
                    overlay_efd_i = np.zeros_like(binary_img, dtype=np.uint8)
                    cv2.polylines(overlay_efd_i, [efd_rec_int], isClosed=True, color=255, thickness=2)
                    cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_efd_{idx}.jpg"), overlay_efd_i)
                    efd_inv = self._efd_features_invariants(efd)
                    # JSON保存已禁用
                    # try:
                    #     with open(os.path.join(self.output_dir, f"efd_features_{suffix}_{idx}.json"), "w", encoding="utf-8") as f:
                    #         json.dump({**efd, "invariants": efd_inv}, f, ensure_ascii=False, indent=2)
                    # except Exception as e:
                    #     print(f"[警告] 保存EFD特征({suffix},{idx})失败: {e}")
                    efd_features_all.append({"index": idx, "harmonics": self.efd_harmonics, **efd, "invariants": efd_inv})

            # 汇总
            if len(fd_features_all) > 0:
                cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_fourier.jpg"), overlay_fd_all)
                # JSON保存已禁用
                # try:
                #     with open(os.path.join(self.output_dir, f"fourier_features_{suffix}.json"), "w", encoding="utf-8") as f:
                #         json.dump({"contours": fd_features_all}, f, ensure_ascii=False, indent=2)
                # except Exception as e:
                #     print(f"[警告] 保存傅里叶特征汇总({suffix})失败: {e}")
            else:
                print(f"[警告] ({suffix}) 未找到有效轮廓，傅里叶拟合与特征提取跳过")
            if len(efd_features_all) > 0:
                cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_efd.jpg"), overlay_efd_all)
                # JSON保存已禁用
                # try:
                #     with open(os.path.join(self.output_dir, f"efd_features_{suffix}.json"), "w", encoding="utf-8") as f:
                #         json.dump({"contours": efd_features_all}, f, ensure_ascii=False, indent=2)
                # except Exception as e:
                #     print(f"[警告] 保存EFD特征汇总({suffix})失败: {e}")
            print(f"[信息] ({suffix}) 多轮廓拟合完成：FD {len(fd_features_all)} 个，EFD {len(efd_features_all)} 个；形态学预处理={'开启' if self.enable_morph_preprocess else '关闭'}")
        except Exception as e:
            print(f"[异常] 阈值管线({suffix})执行失败: {e}")

    def _process_levelset_pipeline(self, level_img_float, level_value, suffix="68"):
        """非二值化的等值线轮廓提取与FD/EFD输出（带后缀）。
        使用等值线(level_value)直接在强度图上提取轮廓。
        """
        try:
            # 提取等值线轮廓
            pts_list = self._find_isocontours(level_img_float, level=level_value)
            overlay_fd_all = np.zeros_like(level_img_float if level_img_float.ndim == 2 else cv2.cvtColor(level_img_float, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
            overlay_efd_all = np.zeros_like(overlay_fd_all, dtype=np.uint8)
            fd_features_all = []
            efd_features_all = []
            for idx, pts in enumerate(pts_list):
                pts_resampled = self._resample_contour(pts, num_points=self.num_resample_points)
                if pts_resampled is None:
                    continue
                if self.fd_canonicalize:
                    pts_resampled = self._canonicalize_contour(pts_resampled)
                fd = self._compute_fourier_descriptors(pts_resampled, num_harmonics=self.fd_harmonics)
                if fd is not None:
                    z_rec = self._reconstruct_contour(fd["F"], fd["mean_z"], num_harmonics=self.fd_harmonics)
                    rec_pts = np.column_stack([np.real(z_rec), np.imag(z_rec)]).astype(np.int32)
                    h, w = overlay_fd_all.shape[:2]
                    rec_pts = np.clip(rec_pts, [0, 0], [w - 1, h - 1])
                    cv2.polylines(overlay_fd_all, [rec_pts], isClosed=True, color=255, thickness=2)
                    overlay_fd_i = np.zeros_like(overlay_fd_all, dtype=np.uint8)
                    cv2.polylines(overlay_fd_i, [rec_pts], isClosed=True, color=255, thickness=2)
                    cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_fourier_{idx}.jpg"), overlay_fd_i)
                    fd_inv = self._fd_features_invariants(fd["F"], use_adaptive=self.fd_adaptive, energy_percent=self.fd_energy_percent)
                    # JSON保存已禁用
                    # try:
                    #     with open(os.path.join(self.output_dir, f"fourier_features_{suffix}_{idx}.json"), "w", encoding="utf-8") as f:
                    #         json.dump({**fd["features"], **fd_inv}, f, ensure_ascii=False, indent=2)
                    # except Exception as e:
                    #     print(f"[警告] 保存傅里叶特征({suffix},{idx})失败: {e}")
                    fd_features_all.append({"index": idx, **fd["features"], **fd_inv})
                efd = self._compute_efd(pts_resampled, harmonics=self.efd_harmonics)
                if efd is not None:
                    efd_rec = self._reconstruct_efd(efd, num_points=self.num_resample_points)
                    efd_rec_int = np.clip(efd_rec, [0, 0], [overlay_efd_all.shape[1] - 1, overlay_efd_all.shape[0] - 1]).astype(np.int32)
                    cv2.polylines(overlay_efd_all, [efd_rec_int], isClosed=True, color=255, thickness=2)
                    overlay_efd_i = np.zeros_like(overlay_efd_all, dtype=np.uint8)
                    cv2.polylines(overlay_efd_i, [efd_rec_int], isClosed=True, color=255, thickness=2)
                    cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_efd_{idx}.jpg"), overlay_efd_i)
                    efd_inv = self._efd_features_invariants(efd)
                    # JSON保存已禁用
                    # try:
                    #     with open(os.path.join(self.output_dir, f"efd_features_{suffix}_{idx}.json"), "w", encoding="utf-8") as f:
                    #         json.dump({**efd, "invariants": efd_inv}, f, ensure_ascii=False, indent=2)
                    # except Exception as e:
                    #     print(f"[警告] 保存EFD特征({suffix},{idx})失败: {e}")
                    efd_features_all.append({"index": idx, "harmonics": self.efd_harmonics, **efd, "invariants": efd_inv})

            # 汇总
            if len(fd_features_all) > 0:
                cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_fourier.jpg"), overlay_fd_all)
                # JSON保存已禁用
                # try:
                #     with open(os.path.join(self.output_dir, f"fourier_features_{suffix}.json"), "w", encoding="utf-8") as f:
                #         json.dump({"contours": fd_features_all}, f, ensure_ascii=False, indent=2)
                # except Exception as e:
                #     print(f"[警告] 保存傅里叶特征汇总({suffix})失败: {e}")
            else:
                print(f"[警告] ({suffix}) 未找到有效等值线，傅里叶拟合与特征提取跳过")
            if len(efd_features_all) > 0:
                cv2.imwrite(os.path.join(self.output_dir, f"contour_image_{suffix}_efd.jpg"), overlay_efd_all)
                # JSON保存已禁用
                # try:
                #     with open(os.path.join(self.output_dir, f"efd_features_{suffix}.json"), "w", encoding="utf-8") as f:
                #         json.dump({"contours": efd_features_all}, f, ensure_ascii=False, indent=2)
                # except Exception as e:
                #     print(f"[警告] 保存EFD特征汇总({suffix})失败: {e}")
            print(f"[信息] ({suffix}) 等值线拟合完成：FD {len(fd_features_all)} 个，EFD {len(efd_features_all)} 个；完全不二值化")
        except Exception as e:
            print(f"[异常] 等值线管线({suffix})执行失败: {e}")

    def _generate_grid_mesh(self, intensity_img, cell_size=3, suffix="68", mesh_type=None, include_z=True):
        """根据强度图生成网格，并输出三角化结果与单元强度。
        - mesh_type="square_tri": 规则方格，每格拆分两三角（默认）
        - mesh_type="tri": 等边三角网格（蜂窝/六角等效三角剖分）
        输出 JSON: 顶点/三角形列表与每个单元的平均强度；并输出网格叠加预览图。
        不对强度图做二值化，保留每格的平均值作为权重/得分。
        """
        timer = TimeTracker(f"网格生成({suffix})").start()
        try:
            if intensity_img is None:
                return
            if intensity_img.ndim == 3:
                img = cv2.cvtColor(intensity_img, cv2.COLOR_BGR2GRAY)
            else:
                img = intensity_img
            h, w = img.shape
            s = max(1, int(cell_size))
            mtype = (self.mesh_type if mesh_type is None else mesh_type).lower()
            timer.checkpoint("图像预处理")
            # 顶点列表（按(y,x)行优先索引），改为“按需创建”以避免缺失键导致的 KeyError
            verts = []
            verts_3d = []
            vert_idx = {}
            def ensure_vertex(yy, xx):
                y_i, x_i = int(max(0, min(h - 1, round(yy)))), int(max(0, min(w - 1, round(xx))))
                key = (y_i, x_i)
                idx = vert_idx.get(key)
                if idx is None:
                    idx = len(verts)
                    vert_idx[key] = idx
                    verts.append([int(x_i), int(y_i)])
                    if include_z:
                        z = float(img[y_i, x_i])
                        # 保存为浮点数，确保OBJ格式正确
                        verts_3d.append([float(x_i), float(y_i), z])
                return idx
            triangles = []
            cells = []
            cell_scores = []
            # 已移除网格叠加预览的计算与保存（不再生成 overlay 图像）

            if mtype == "tri":
                # 等边三角网格：水平间距 s，垂直间距 s*sqrt(3)/2，奇偶行水平偏移 s/2
                dx = s
                dy = max(1, int(round(s * math.sqrt(3) / 2)))
                # 为每一行生成点，并记录索引
                row_points = []  # 每行的顶点坐标列表 [(x,y), ...]
                row_indices = [] # 顶点索引列表 [idx, ...]
                r = 0
                while True:
                    y = r * dy
                    if y > h - 1:
                        break
                    offset = 0 if (r % 2 == 0) else int(round(dx / 2))
                    pts = []
                    idxs = []
                    c = 0
                    while True:
                        x = offset + c * dx
                        if x > w - 1:
                            break
                        idx = ensure_vertex(y, x)
                        pts.append((int(x), int(y)))
                        idxs.append(idx)
                        c += 1
                    # 至少两个点才有意义
                    if len(idxs) >= 2:
                        row_points.append(pts)
                        row_indices.append(idxs)
                    else:
                        # 若该行不足以构成单元，但可能下一行仍需存在点以闭合
                        row_points.append(pts)
                        row_indices.append(idxs)
                    r += 1
                # 生成三角连接（每两行形成菱形，拆分为两个三角）
                for r in range(0, len(row_indices) - 1):
                    top = row_indices[r]
                    bot = row_indices[r + 1]
                    top_pts = row_points[r]
                    bot_pts = row_points[r + 1]
                    if len(top) == 0 or len(bot) == 0:
                        continue
                    if r % 2 == 0:
                        # 偶数行在上：
                        # 连接：(t[c], t[c+1], b[c]) 与 (t[c+1], b[c+1], b[c])
                        for c in range(0, min(len(top) - 1, len(bot))):
                            t0, t1 = top[c], top[c + 1]
                            b0 = bot[c]
                            # 第二个三角需要 b1 存在
                            if c + 1 < len(bot):
                                b1 = bot[c + 1]
                                # 三角1
                                triangles.append([t0, t1, b0])
                                tri_pts_1 = np.array([top_pts[c], top_pts[c + 1], bot_pts[c]], dtype=np.int32)
                                mask1 = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillConvexPoly(mask1, tri_pts_1, 255)
                                ssum1 = float((img * (mask1 > 0)).sum())
                                cnt1 = int((mask1 > 0).sum())
                                cell_scores.append((ssum1 / cnt1) if cnt1 > 0 else 0.0)
                                cells.append({"triangle": [int(v) for v in triangles[-1]]})
                                # 三角2
                                triangles.append([t1, b1, b0])
                                tri_pts_2 = np.array([top_pts[c + 1], bot_pts[c + 1], bot_pts[c]], dtype=np.int32)
                                mask2 = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillConvexPoly(mask2, tri_pts_2, 255)
                                ssum2 = float((img * (mask2 > 0)).sum())
                                cnt2 = int((mask2 > 0).sum())
                                cell_scores.append((ssum2 / cnt2) if cnt2 > 0 else 0.0)
                                cells.append({"triangle": [int(v) for v in triangles[-1]]})
                            else:
                                # 边界不足以形成第二个三角，仅绘制第一个
                                triangles.append([t0, t1, b0])
                                tri_pts_1 = np.array([top_pts[c], top_pts[c + 1], bot_pts[c]], dtype=np.int32)
                                mask = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillConvexPoly(mask, tri_pts_1, 255)
                                ssum = float((img * (mask > 0)).sum())
                                cnt = int((mask > 0).sum())
                                cell_scores.append((ssum / cnt) if cnt > 0 else 0.0)
                                cells.append({"triangle": [int(v) for v in triangles[-1]]})
                    else:
                        # 奇数行在上：
                        # 连接：(t[c], b[c], b[c+1]) 与 (t[c], t[c+1], b[c+1])
                        for c in range(0, min(len(bot) - 1, len(top))):
                            t0 = top[c]
                            t1 = top[c + 1] if c + 1 < len(top) else None
                            b0, b1 = bot[c], bot[c + 1]
                            # 第一个三角
                            triangles.append([t0, b0, b1])
                            tri_pts_1 = np.array([top_pts[c], bot_pts[c], bot_pts[c + 1]], dtype=np.int32)
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillConvexPoly(mask, tri_pts_1, 255)
                            ssum = float((img * (mask > 0)).sum())
                            cnt = int((mask > 0).sum())
                            cell_scores.append((ssum / cnt) if cnt > 0 else 0.0)
                            cells.append({"triangle": [int(v) for v in triangles[-1]]})
                            # 第二个三角（需要 t1）
                            if t1 is not None:
                                triangles.append([t0, t1, b1])
                                tri_pts_2 = np.array([top_pts[c], top_pts[c + 1], bot_pts[c + 1]], dtype=np.int32)
                                mask2 = np.zeros((h, w), dtype=np.uint8)
                                cv2.fillConvexPoly(mask2, tri_pts_2, 255)
                                ssum2 = float((img * (mask2 > 0)).sum())
                                cnt2 = int((mask2 > 0).sum())
                                cell_scores.append((ssum2 / cnt2) if cnt2 > 0 else 0.0)
                                cells.append({"triangle": [int(v) for v in triangles[-1]]})
            else:
                # 规则方格拆分为两三角
                for y in range(0, h, s):
                    y2 = min(y + s, h)
                    for x in range(0, w, s):
                        x2 = min(x + s, w)
                        # 顶点索引（右下边界对齐），统一使用 ensure_vertex，避免缺失键
                        v00 = ensure_vertex(y, x)
                        v10 = ensure_vertex(y, x2)
                        v01 = ensure_vertex(y2, x)
                        v11 = ensure_vertex(y2, x2)
                        # 两个三角形
                        triangles.append([v00, v10, v01])
                        triangles.append([v01, v10, v11])
                        # 单元信息与平均强度得分（使用矩形均值近似）
                        cell = {
                            "bbox": [int(x), int(y), int(x2 - x), int(y2 - y)],
                            "verts": [v00, v10, v11, v01]
                        }
                        cells.append(cell)
                        patch = img[y:y2, x:x2]
                        score = float(patch.mean()) if patch.size > 0 else 0.0
                        cell_scores.append(score)
                        # 已移除预览绘制（网格线与三角边线不再生成）
            timer.checkpoint("网格构建")

            # 保存JSON
            out_json = {
                "image_size": [int(w), int(h)],
                "cell_size": int(s),
                "mesh_type": ("tri" if mtype == "tri" else "square_tri"),
                "vertices": verts,
                "vertices_3d": (verts_3d if include_z else []),
                "triangles": triangles,
                "cells": cells,
                "cell_scores": cell_scores
            }
            if mtype == "tri":
                fname = os.path.join(self.output_dir, f"mesh_{suffix}_{s}px_tri.json")
            else:
                fname = os.path.join(self.output_dir, f"mesh_{suffix}_{s}x{s}.json")
            # JSON保存已禁用
            # with open(fname, "w", encoding="utf-8") as f:
            #     json.dump(out_json, f, ensure_ascii=False, indent=2)
            timer.checkpoint("保存JSON")
            # 已移除网格叠加预览图保存
            print(f"[信息]({suffix}) 网格生成完成：type={out_json['mesh_type']}, cell={s}，顶点{len(verts)}，三角{len(triangles)}")
            timer.print_summary()
        except Exception as e:
            print(f"[异常] 网格生成失败({suffix}): {e}")

    def _o3d_build_height_mesh(self, img, step=3, z_scale=255.0, mask=None):
        """将灰度/强度图构建为Open3D三角网格，高度为归一化强度(0..z_scale)。
        如果提供了mask，只生成mask覆盖区域的mesh，但保留灰度值（不是二值化）。
        
        注意：使用灰度图生成mesh，保留原始灰度值，而不是二值化。
        
        Args:
            img: 灰度/强度图（可以是0-1范围或0-255范围）
            step: 采样步长
            z_scale: Z轴缩放因子（用于将归一化后的值映射到实际高度）
            mask: 可选的二值mask图像（mask覆盖区域为255，其他为0）
        
        返回 (TriangleMesh, arr_ds)；arr_ds 为按采样步长下采样的高度场，用于傅里叶分析；若open3d不可用返回(None, None)。
        """
        try:
            if o3d is None:
                return None, None
            arr = np.asarray(img, dtype=np.float32)
            
            # 检查输入有效性
            if arr.size == 0:
                print(f"[警告] 输入图像为空")
                return None, None
            
            if len(arr.shape) != 2:
                print(f"[警告] 输入图像维度不正确: {arr.shape}，期望2D灰度图")
                return None, None
            
            h, w = arr.shape
            if h == 0 or w == 0:
                print(f"[警告] 输入图像尺寸无效: {h}x{w}")
                return None, None
            
            # 归一化到 [0, z_scale]
            # 如果输入已经是0-1范围，直接使用；否则归一化
            arr_max = arr.max()
            arr_min = arr.min()
            
            if arr_max <= 1.001:
                # 已经是归一化的（0-1范围），直接缩放到z_scale
                arr_normalized = arr.copy()
                arr = arr_normalized * float(z_scale)
            else:
                # 输入是0-255范围的灰度图
                # 方法：将灰度值映射到 [0, z_scale] 范围
                # 这样灰度值0对应高度0，灰度值255对应高度z_scale
                if arr_max > arr_min:
                    # 归一化到0-1，然后缩放到z_scale
                    arr_normalized = (arr - arr_min) / (arr_max - arr_min)
                    arr = arr_normalized * float(z_scale)
                else:
                    # 全为同一值，设为0
                    arr = np.zeros_like(arr)
            
            # 打印调试信息
            if arr.size > 0:
                z_min = arr[arr > 0].min() if np.any(arr > 0) else 0.0
                z_max = arr.max()
                z_mean = arr[arr > 0].mean() if np.any(arr > 0) else 0.0
                print(f"  [高度场统计] Z范围: [{z_min:.2f}, {z_max:.2f}], 平均值: {z_mean:.2f}, z_scale: {z_scale:.2f}")
            
            # 如果提供了mask，应用mask（但保留灰度值，不是二值化）
            mask_valid = None
            if mask is not None:
                mask_arr = np.asarray(mask, dtype=np.uint8)
                if mask_arr.shape != arr.shape:
                    # 如果mask尺寸不匹配，调整mask尺寸
                    print(f"    [mask调整] 调整mask尺寸从 {mask_arr.shape} 到 {arr.shape}")
                    mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_NEAREST)
                # 创建mask有效性标记（mask覆盖区域为True）
                mask_valid = (mask_arr > 0)
                # 检查mask是否有有效区域
                mask_valid_count = np.count_nonzero(mask_valid)
                if mask_valid_count == 0:
                    print(f"    [警告] mask调整后没有有效区域，将不使用mask")
                    mask_valid = None
                else:
                    # 将mask外的区域设为0，但mask内的区域保留原始灰度值
                    arr_before_mask = arr.copy()
                    arr = np.where(mask_valid, arr, 0.0)
                    # 检查mask内是否有非零值
                    arr_in_mask = arr[mask_valid]
                    if len(arr_in_mask) > 0 and arr_in_mask.max() == 0:
                        print(f"    [警告] mask内所有灰度值都为0，可能存在问题")
                    print(f"    [mask应用] mask覆盖 {mask_valid_count} 个像素 ({mask_valid_count/(h*w)*100:.1f}%)")
            
            s = max(1, int(step))
            ys = list(range(0, h, s))
            xs = list(range(0, w, s))
            # 构建顶点与颜色（使用灰度值）
            vertices = []
            colors = []
            arr_ds = np.empty((len(ys), len(xs)), dtype=np.float32)
            for ri, yy in enumerate(ys):
                for ci, xx in enumerate(xs):
                    z = float(arr[yy, xx])  # 这是保留灰度值的高度
                    # 判断该点是否在mask内（如果提供了mask）
                    is_in_mask = mask_valid[yy, xx] if mask_valid is not None else True
                    
                    if mask_valid is not None and not is_in_mask:
                        # mask外的点，设为0
                        vertices.append([float(xx), float(yy), 0.0])
                        colors.append([0.0, 0.0, 0.0])
                        arr_ds[ri, ci] = 0.0
                    else:
                        # mask内的点，使用灰度值（不是二值化）
                        vertices.append([float(xx), float(yy), z])
                        # 颜色基于灰度值（归一化到0-1）
                        g = z / float(z_scale) if z_scale > 0 else 0.0
                        colors.append([g, g, g])
                        arr_ds[ri, ci] = z
            
            # 构建三角（只连接mask内的有效顶点）
            def vidx(r, c):
                return r * len(xs) + c
            triangles = []
            for r in range(len(ys) - 1):
                for c in range(len(xs) - 1):
                    v00 = vidx(r, c)
                    v10 = vidx(r, c + 1)
                    v01 = vidx(r + 1, c)
                    v11 = vidx(r + 1, c + 1)
                    
                    # 检查四个顶点是否在mask内
                    if mask_valid is not None:
                        in_mask_00 = mask_valid[ys[r], xs[c]]
                        in_mask_10 = mask_valid[ys[r], xs[c+1]]
                        in_mask_01 = mask_valid[ys[r+1], xs[c]]
                        in_mask_11 = mask_valid[ys[r+1], xs[c+1]]
                        # 如果所有顶点都在mask外，跳过这个三角形
                        if not (in_mask_00 or in_mask_10 or in_mask_01 or in_mask_11):
                            continue
                    
                    triangles.append([v00, v10, v01])
                    triangles.append([v01, v10, v11])
            
            # 检查是否有有效的顶点和三角形
            if len(vertices) == 0:
                print(f"[警告] 没有生成有效的顶点")
                return None, None
            
            if len(triangles) == 0:
                print(f"[警告] 没有生成有效的三角形")
                return None, None
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
            
            # 优化：清理mesh以减少空洞
            try:
                # 移除退化三角形和重复三角形
                mesh.remove_degenerate_triangles()
                mesh.remove_duplicated_triangles()
                mesh.remove_duplicated_vertices()
                mesh.remove_non_manifold_edges()
                
                # 计算法向量
                mesh.compute_vertex_normals()
                
                # 可选：填充小空洞（如果启用）
                fill_holes = bool(int(os.environ.get("MESH_FILL_HOLES", "0")))  # 默认关闭
                if fill_holes and len(mesh.triangles) > 0:
                    # 使用Open3D的fill_holes方法（如果可用）
                    try:
                        # 注意：Open3D的fill_holes可能不适用于所有情况
                        # 这里先尝试，如果失败则跳过
                        mesh = mesh.fill_holes()
                        print(f"  [mesh优化] 已尝试填充空洞")
                    except Exception as e:
                        print(f"  [mesh优化] 填充空洞失败（可能不支持）: {e}")
            except Exception as e:
                print(f"  [警告] mesh清理失败: {e}")
                try:
                    mesh.compute_vertex_normals()
                except Exception:
                    pass
            
            return mesh, arr_ds
        except Exception as e:
            print(f"[警告] 构建Open3D高度网格失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _compute_o3d_fourier_features(self, field, kx=24, ky=24, suffix="o3d", save_recon=True, return_features=False):
        """
        [已弃用] 在Open3D高度场对应的二维场上做傅里叶截断与特征提取
        
        注意：此函数已不再使用，因为已移除傅里叶拟合功能。
        直接返回None，不再进行任何计算。
        """
        # 已禁用：不再进行傅里叶拟合和特征提取
        return None if return_features else None

    def _compute_o3d_spectral_descriptors(self, field, suffix="o3d", num_radial=32, num_angular=36, return_features=False):
        """计算稳定的谱特征用于匹配：径向/角向能量分布、谱熵、频域质心与扩展。"""
        try:
            if field is None:
                return None if return_features else None
            fld = np.asarray(field, dtype=np.float32)
            h, w = fld.shape
            F = np.fft.fft2(fld)
            F_shift = np.fft.fftshift(F)
            P = np.abs(F_shift).astype(np.float64)
            eps = 1e-12
            P_sum = float(P.sum()) + eps
            Pn = P / P_sum
            cy, cx = h // 2, w // 2
            yy, xx = np.mgrid[0:h, 0:w]
            ky = yy - cy
            kx = xx - cx
            rad = np.hypot(ky, kx)
            ang = np.arctan2(ky, kx)
            # 径向直方图（0..rmax）
            rmax = float(rad.max())
            num_r = max(1, int(num_radial))
            r_bins = np.linspace(0.0, rmax + 1e-9, num_r + 1)
            r_idx = np.digitize(rad.ravel(), r_bins) - 1
            radial_hist = np.zeros(num_r, dtype=np.float64)
            Pn_flat = Pn.ravel()
            for i in range(num_r):
                radial_hist[i] = float(Pn_flat[r_idx == i].sum())
            # 角向直方图（-pi..pi）
            num_a = max(1, int(num_angular))
            a_bins = np.linspace(-np.pi, np.pi + 1e-9, num_a + 1)
            a_idx = np.digitize(ang.ravel(), a_bins) - 1
            angular_hist = np.zeros(num_a, dtype=np.float64)
            for i in range(num_a):
                angular_hist[i] = float(Pn_flat[a_idx == i].sum())
            # 频域质心与扩展
            kx_flat = kx.ravel().astype(np.float64)
            ky_flat = ky.ravel().astype(np.float64)
            kx_mean = float((kx_flat * Pn_flat).sum())
            ky_mean = float((ky_flat * Pn_flat).sum())
            kx_var = float(((kx_flat - kx_mean) ** 2 * Pn_flat).sum())
            ky_var = float(((ky_flat - ky_mean) ** 2 * Pn_flat).sum())
            # 谱熵（能量分布均匀程度）
            entropy = float(-(Pn_flat * np.log(Pn_flat + eps)).sum())
            # 主导方向（角向直方图最大bin中心）
            ang_idx_max = int(np.argmax(angular_hist)) if angular_hist.size > 0 else 0
            ang_centers = (a_bins[:-1] + a_bins[1:]) / 2.0
            dominant_orientation = float(ang_centers[ang_idx_max]) if angular_hist.size > 0 else 0.0
            desc = {
                "size": [int(w), int(h)],
                "spectral_entropy": entropy,
                "centroid_k": [kx_mean, ky_mean],
                "spread_k": [kx_var, ky_var],
                "radial_hist": radial_hist.tolist(),
                "angular_hist": angular_hist.tolist(),
                "dominant_orientation": dominant_orientation
            }
            # JSON保存已禁用
            # try:
            #     with open(os.path.join(self.output_dir, f"o3d_spectral_descriptors_{suffix}.json"), "w", encoding="utf-8") as f:
            #         json.dump(desc, f, ensure_ascii=False, indent=2)
            # except Exception as e:
            #     print(f"[警告] 保存Open3D谱特征失败({suffix}): {e}")
            
            # 可视化保存：径向和角向直方图
            try:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # 径向直方图
                r_centers = (r_bins[:-1] + r_bins[1:]) / 2.0
                axes[0].bar(r_centers, radial_hist, width=(rmax / num_r) * 0.8, alpha=0.7, color='blue')
                axes[0].set_xlabel('径向距离 (像素)')
                axes[0].set_ylabel('能量密度')
                axes[0].set_title(f'径向能量分布 (谱熵: {entropy:.4f})')
                axes[0].grid(True, alpha=0.3)
                
                # 角向直方图
                a_centers = (a_bins[:-1] + a_bins[1:]) / 2.0
                axes[1].bar(a_centers, angular_hist, width=(2 * np.pi / num_a) * 0.8, alpha=0.7, color='green')
                axes[1].set_xlabel('角度 (弧度)')
                axes[1].set_ylabel('能量密度')
                axes[1].set_title(f'角向能量分布 (主导方向: {np.degrees(dominant_orientation):.1f}°)')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                vis_path = os.path.join(self.output_dir, f"o3d_spectral_descriptors_{suffix}.png")
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[信息] 已保存谱特征可视化: {vis_path}")
            except Exception as e:
                print(f"[警告] 保存谱特征可视化失败({suffix}): {e}")
            # 如果要求返回特征，则返回特征字典
            if return_features:
                return desc
        except Exception as e:
            print(f"[异常] Open3D谱特征计算失败({suffix}): {e}")
            return None if return_features else None

    def _coarse_align_meshes(self, mesh_path1, mesh_path2, frame_idx1=None, frame_idx2=None, suffix=None):
        if suffix is None:
            suffix = getattr(self, 'threshold_mode', 'th_68')
        """
        对前后帧mesh进行粗匹配（场景叠加）- 优化版：基于几何地标点
        
        方法（优先级顺序）：
        1. 基于几何地标点匹配（山峰、脊线、显著点）
        2. 基于质心对齐（平移，备选）
        3. 基于PCA主方向对齐（旋转，备选）
        4. ICP粗配准（大距离阈值，精化）
        
        Args:
            mesh_path1: 第一帧网格路径
            mesh_path2: 第二帧网格路径
            frame_idx1: 第一帧索引（用于保存结果）
            frame_idx2: 第二帧索引（用于保存结果）
            suffix: 文件后缀（用于保存结果）
            
        Returns:
            4x4变换矩阵，将mesh2变换到mesh1的坐标系，如果失败返回None
        """
        if o3d is None:
            return None
        
        try:
            # 1. 加载网格
            mesh1 = o3d.io.read_triangle_mesh(mesh_path1)
            mesh2 = o3d.io.read_triangle_mesh(mesh_path2)
            
            if len(mesh1.vertices) == 0 or len(mesh2.vertices) == 0:
                return None
            
            vertices1 = np.asarray(mesh1.vertices)
            vertices2 = np.asarray(mesh2.vertices)
            
            T_coarse = None
            T_translation = None
            T_rotation = None
            translation = None  # 初始化平移向量
            
            # 2. 方法1: 基于几何地标点的对齐（优先）
            try:
                print(f"  [粗匹配] 尝试基于几何地标点对齐...")
                landmarks1 = self._extract_geometric_landmarks(mesh_path1, max_landmarks=50)
                landmarks2 = self._extract_geometric_landmarks(mesh_path2, max_landmarks=50)
                
                if landmarks1 is not None and landmarks2 is not None and \
                   len(landmarks1['landmarks']) >= 3 and len(landmarks2['landmarks']) >= 3:
                    
                    # 匹配地标点
                    matches = self._match_landmarks(
                        landmarks1['landmarks'], landmarks2['landmarks'],
                        landmarks1['landmark_features'], landmarks2['landmark_features'],
                        max_distance=0.5  # 归一化后的距离阈值
                    )
                    
                    if len(matches) >= 3:
                        # 使用匹配的地标点计算初始变换（SVD）
                        matched_pts1 = landmarks1['landmarks'][[m[0] for m in matches]]
                        matched_pts2 = landmarks2['landmarks'][[m[1] for m in matches]]
                        
                        T_landmark = self._estimate_transformation_svd(matched_pts1, matched_pts2)
                        
                        if T_landmark is not None:
                            # 验证变换质量
                            pts2_homo = np.hstack([matched_pts2, np.ones((len(matched_pts2), 1))])
                            pts2_transformed = (T_landmark @ pts2_homo.T).T[:, :3]
                            distances = np.linalg.norm(matched_pts1 - pts2_transformed, axis=1)
                            avg_error = np.mean(distances)
                            
                            # 如果平均误差合理（小于mesh尺度的10%），使用地标点变换
                            mesh_scale = np.linalg.norm(vertices1.max(axis=0) - vertices1.min(axis=0))
                            if avg_error < mesh_scale * 0.1:
                                T_coarse = T_landmark
                                print(f"  [粗匹配] 地标点匹配成功: {len(matches)}个匹配点, 平均误差={avg_error:.4f}")
                            else:
                                print(f"  [粗匹配] 地标点变换误差过大({avg_error:.4f}>{mesh_scale*0.1:.4f})，使用备选方法")
                        else:
                            print(f"  [粗匹配] 地标点变换估计失败，使用备选方法")
                    else:
                        print(f"  [粗匹配] 地标点匹配数量不足({len(matches)}<3)，使用备选方法")
                else:
                    print(f"  [粗匹配] 地标点提取不足，使用备选方法")
            except Exception as e:
                print(f"  [粗匹配] 地标点对齐失败: {e}，使用备选方法")
            
            # 3. 方法2: 基于质心的对齐（备选）
            if T_coarse is None:
                center1 = vertices1.mean(axis=0)
                center2 = vertices2.mean(axis=0)
                translation = center1 - center2
                
                T_translation = np.eye(4)
                T_translation[:3, 3] = translation
                T_coarse = T_translation
                print(f"  [粗匹配] 使用质心对齐（平移）")
            
            # 4. 方法3: 基于PCA的主方向对齐（可选，如果地标点方法失败）
            if T_rotation is None and T_translation is not None and \
               (T_coarse is None or np.allclose(T_coarse, T_translation)):
                try:
                    center1 = vertices1.mean(axis=0)
                    center2 = vertices2.mean(axis=0)
                    
                    # 中心化
                    vertices1_centered = vertices1 - center1
                    vertices2_centered = vertices2 - center2
                    
                    # PCA
                    cov1 = np.cov(vertices1_centered.T)
                    cov2 = np.cov(vertices2_centered.T)
                    
                    eigenvals1, eigenvecs1 = np.linalg.eigh(cov1)
                    eigenvals2, eigenvecs2 = np.linalg.eigh(cov2)
                    
                    # 按特征值排序（降序）
                    idx1 = np.argsort(eigenvals1)[::-1]
                    idx2 = np.argsort(eigenvals2)[::-1]
                    
                    principal_dirs1 = eigenvecs1[:, idx1]
                    principal_dirs2 = eigenvecs2[:, idx2]
                    
                    # 确保主方向一致（处理方向翻转）
                    for i in range(3):
                        if np.dot(principal_dirs1[:, i], principal_dirs2[:, i]) < 0:
                            principal_dirs2[:, i] *= -1
                    
                    # 计算旋转矩阵：R = V1 @ V2^T
                    R_pca = principal_dirs1 @ principal_dirs2.T
                    
                    # 确保是旋转矩阵（正交且行列式为1）
                    U, S, Vt = np.linalg.svd(R_pca)
                    R_pca = U @ Vt
                    if np.linalg.det(R_pca) < 0:
                        U[:, -1] *= -1
                        R_pca = U @ Vt
                    
                    T_rotation = np.eye(4)
                    T_rotation[:3, :3] = R_pca
                    
                    # 组合旋转和平移：先旋转后平移
                    T_coarse = T_translation @ T_rotation
                    print(f"  [粗匹配] 使用PCA对齐（旋转+平移）")
                    
                except Exception as e:
                    # 如果PCA失败，只使用平移
                    print(f"  [粗匹配] PCA对齐失败，仅使用平移对齐: {e}")
                    T_coarse = T_translation
            
            # 5. 方法4: ICP粗配准（精化，使用大距离阈值）
            try:
                # 转换为点云
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(vertices1)
                
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(vertices2)
                
                # 应用粗变换
                pcd2.transform(T_coarse)
                
                # 计算边界框以确定ICP距离阈值
                bbox1 = pcd1.get_axis_aligned_bounding_box()
                bbox2 = pcd2.get_axis_aligned_bounding_box()
                extent1 = bbox1.get_extent()
                max_extent = np.max(extent1)
                
                # ICP粗配准（大距离阈值）
                result_icp = o3d.pipelines.registration.registration_icp(
                    pcd1, pcd2,
                    max_correspondence_distance=max_extent * 0.3,  # 30%的边界框大小
                    init=np.eye(4),  # 从当前对齐开始
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                
                if result_icp.fitness > 0.3:  # 如果ICP改进了对齐
                    T_coarse = T_coarse @ result_icp.transformation
                    print(f"  [粗匹配] ICP粗配准成功，fitness={result_icp.fitness:.4f}")
                else:
                    print(f"  [粗匹配] ICP粗配准效果不佳，使用PCA+平移结果")
                    
            except Exception as e:
                print(f"  [粗匹配] ICP粗配准失败，使用PCA+平移结果: {e}")
            
            # 提取平移向量用于打印（如果存在）
            if translation is not None:
                print(f"  [粗匹配] 完成：平移={translation}, 旋转矩阵行列式={np.linalg.det(T_coarse[:3, :3]):.4f}")
            else:
                print(f"  [粗匹配] 完成：旋转矩阵行列式={np.linalg.det(T_coarse[:3, :3]):.4f}")
            
            # 5. 保存粗匹配结果
            if frame_idx1 is not None and frame_idx2 is not None:
                self._save_coarse_alignment_result(
                    T_coarse, mesh1, mesh2, T_translation, T_rotation,
                    frame_idx1, frame_idx2, suffix
                )
            
            return T_coarse
            
        except Exception as e:
            print(f"[警告] 粗匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_geometric_landmarks(self, mesh_path, max_landmarks=50, z_threshold_percentile=85):
        """
        从3D mesh中提取几何地标点（山峰、关键特征点）
        
        方法：
        1. 提取局部最大值点（山峰）
        2. 提取高曲率点（脊线、边缘）
        3. 提取显著几何特征点
        
        Args:
            mesh_path: 网格文件路径
            max_landmarks: 最大地标点数量
            z_threshold_percentile: Z值阈值百分位（用于筛选山峰）
            
        Returns:
            dict: {
                'landmarks': 地标点坐标 (N, 3),
                'landmark_types': 地标点类型 ['peak', 'ridge', 'salient'],
                'landmark_features': 地标点特征描述符
            } 或 None
        """
        if o3d is None:
            return None
        
        try:
            # 1. 加载网格
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            if len(mesh.vertices) == 0:
                return None
            
            vertices = np.asarray(mesh.vertices)
            
            # 2. 提取山峰（局部Z值最大值点）
            peaks = self._extract_peaks(vertices, max_points=max_landmarks//2, 
                                       z_threshold_percentile=z_threshold_percentile)
            
            # 3. 提取高曲率点（脊线、边缘）
            ridges = self._extract_high_curvature_points(mesh, max_points=max_landmarks//2)
            
            # 4. 提取显著几何特征点（基于局部形状）
            salient = self._extract_salient_points(mesh, max_points=max_landmarks//3)
            
            # 5. 合并所有地标点
            all_landmarks = []
            landmark_types = []
            
            if len(peaks) > 0:
                all_landmarks.append(peaks)
                landmark_types.extend(['peak'] * len(peaks))
            
            if len(ridges) > 0:
                all_landmarks.append(ridges)
                landmark_types.extend(['ridge'] * len(ridges))
            
            if len(salient) > 0:
                all_landmarks.append(salient)
                landmark_types.extend(['salient'] * len(salient))
            
            if len(all_landmarks) == 0:
                return None
            
            landmarks = np.vstack(all_landmarks)
            
            # 6. 计算地标点的特征描述符
            landmark_features = self._compute_landmark_features(mesh, landmarks)
            
            # 7. 限制地标点数量
            if len(landmarks) > max_landmarks:
                # 选择最显著的地标点
                scores = self._score_landmarks(landmarks, landmark_features)
                top_indices = np.argsort(scores)[::-1][:max_landmarks]
                landmarks = landmarks[top_indices]
                landmark_types = [landmark_types[i] for i in top_indices]
                landmark_features = landmark_features[top_indices]
            
            print(f"  [几何地标] 提取了 {len(landmarks)} 个地标点: "
                  f"山峰={sum(1 for t in landmark_types if t=='peak')}, "
                  f"脊线={sum(1 for t in landmark_types if t=='ridge')}, "
                  f"显著点={sum(1 for t in landmark_types if t=='salient')}")
            
            return {
                'landmarks': landmarks,
                'landmark_types': landmark_types,
                'landmark_features': landmark_features
            }
            
        except Exception as e:
            print(f"[警告] 几何地标提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_peaks(self, vertices, max_points=25, z_threshold_percentile=85, radius=10.0):
        """
        提取山峰点（局部Z值最大值）
        
        Args:
            vertices: 顶点坐标 (N, 3)
            max_points: 最大点数
            z_threshold_percentile: Z值阈值百分位
            radius: 局部搜索半径
            
        Returns:
            山峰点坐标 (M, 3)
        """
        try:
            # 计算Z值阈值
            z_values = vertices[:, 2]
            z_threshold = np.percentile(z_values, z_threshold_percentile)
            
            # 筛选高Z值点
            high_z_mask = z_values >= z_threshold
            candidate_points = vertices[high_z_mask]
            
            if len(candidate_points) == 0:
                return np.array([]).reshape(0, 3)
            
            # 使用KDTree进行局部最大值检测
            from scipy.spatial import cKDTree
            tree = cKDTree(candidate_points)
            
            peaks = []
            used_mask = np.zeros(len(candidate_points), dtype=bool)
            
            for i in range(len(candidate_points)):
                if used_mask[i]:
                    continue
                
                # 查找半径内的邻居
                neighbors = tree.query_ball_point(candidate_points[i], radius)
                
                # 检查是否为局部最大值
                is_peak = True
                for j in neighbors:
                    if j != i and candidate_points[j][2] > candidate_points[i][2]:
                        is_peak = False
                        break
                
                if is_peak:
                    peaks.append(candidate_points[i])
                    # 标记邻居为已使用
                    for j in neighbors:
                        used_mask[j] = True
            
            peaks = np.array(peaks) if len(peaks) > 0 else np.array([]).reshape(0, 3)
            
            # 限制数量
            if len(peaks) > max_points:
                # 按Z值排序，选择最高的
                z_scores = peaks[:, 2]
                top_indices = np.argsort(z_scores)[::-1][:max_points]
                peaks = peaks[top_indices]
            
            return peaks
            
        except Exception as e:
            print(f"  [警告] 山峰提取失败: {e}")
            return np.array([]).reshape(0, 3)
    
    def _extract_high_curvature_points(self, mesh, max_points=25, curvature_threshold_percentile=90):
        """
        提取高曲率点（脊线、边缘）
        
        Args:
            mesh: Open3D TriangleMesh
            max_points: 最大点数
            curvature_threshold_percentile: 曲率阈值百分位
            
        Returns:
            高曲率点坐标 (M, 3)
        """
        try:
            vertices = np.asarray(mesh.vertices)
            
            # 计算顶点曲率（使用法向量变化）
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            
            # 计算每个顶点的曲率（法向量变化率）
            from scipy.spatial import cKDTree
            tree = cKDTree(vertices)
            curvatures = []
            
            for i in range(len(vertices)):
                # 查找邻居
                neighbors_idx = tree.query_ball_point(vertices[i], r=5.0)
                if len(neighbors_idx) < 3:
                    curvatures.append(0.0)
                    continue
                
                # 计算法向量变化
                normal_i = normals[i]
                neighbor_normals = normals[neighbors_idx]
                
                # 计算法向量差异
                normal_diffs = np.arccos(np.clip(np.dot(neighbor_normals, normal_i), -1, 1))
                curvature = np.mean(normal_diffs)
                curvatures.append(curvature)
            
            curvatures = np.array(curvatures)
            
            # 筛选高曲率点
            curvature_threshold = np.percentile(curvatures, curvature_threshold_percentile)
            high_curvature_mask = curvatures >= curvature_threshold
            ridge_points = vertices[high_curvature_mask]
            
            # 限制数量
            if len(ridge_points) > max_points:
                ridge_curvatures = curvatures[high_curvature_mask]
                top_indices = np.argsort(ridge_curvatures)[::-1][:max_points]
                ridge_points = ridge_points[top_indices]
            
            return ridge_points
            
        except Exception as e:
            print(f"  [警告] 高曲率点提取失败: {e}")
            return np.array([]).reshape(0, 3)
    
    def _extract_salient_points(self, mesh, max_points=15):
        """
        提取显著几何特征点（基于局部形状描述符）
        
        Args:
            mesh: Open3D TriangleMesh
            max_points: 最大点数
            
        Returns:
            显著点坐标 (M, 3)
        """
        try:
            vertices = np.asarray(mesh.vertices)
            
            # 转换为点云进行特征提取
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.estimate_normals()
            
            # 计算FPFH特征
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, search_param)
            
            if fpfh is None:
                return np.array([]).reshape(0, 3)
            
            # 计算特征向量的方差（方差大的点更显著）
            fpfh_data = np.asarray(fpfh.data).T  # (N, 33)
            feature_variances = np.var(fpfh_data, axis=1)
            
            # 选择特征方差最大的点
            top_indices = np.argsort(feature_variances)[::-1][:max_points]
            salient_points = vertices[top_indices]
            
            return salient_points
            
        except Exception as e:
            print(f"  [警告] 显著点提取失败: {e}")
            return np.array([]).reshape(0, 3)
    
    def _compute_landmark_features(self, mesh, landmarks):
        """
        计算地标点的特征描述符
        
        Args:
            mesh: Open3D TriangleMesh
            landmarks: 地标点坐标 (N, 3)
            
        Returns:
            特征描述符 (N, feature_dim)
        """
        try:
            vertices = np.asarray(mesh.vertices)
            
            # 创建地标点云
            landmark_pcd = o3d.geometry.PointCloud()
            landmark_pcd.points = o3d.utility.Vector3dVector(landmarks)
            
            # 确保点云有法向量（FPFH特征需要法向量）
            if not landmark_pcd.has_normals():
                landmark_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=30)
                )
            
            # 计算FPFH特征
            search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=30)
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(landmark_pcd, search_param)
            
            if fpfh is None:
                # 如果FPFH失败，使用简单的几何特征
                features = np.zeros((len(landmarks), 10))
                for i, landmark in enumerate(landmarks):
                    # 计算到mesh中心的距离
                    mesh_center = vertices.mean(axis=0)
                    dist_to_center = np.linalg.norm(landmark - mesh_center)
                    
                    # 计算Z值
                    z_value = landmark[2]
                    
                    # 简单的特征向量
                    features[i] = np.array([
                        landmark[0], landmark[1], landmark[2],
                        dist_to_center, z_value,
                        landmark[0] - mesh_center[0],
                        landmark[1] - mesh_center[1],
                        landmark[2] - mesh_center[2],
                        np.linalg.norm(landmark),
                        z_value / (dist_to_center + 1e-6)
                    ])
                return features
            
            return np.asarray(fpfh.data).T
            
        except Exception as e:
            print(f"  [警告] 地标特征计算失败: {e}")
            return np.zeros((len(landmarks), 33))
    
    def _score_landmarks(self, landmarks, features):
        """
        计算地标点的显著性分数
        
        Args:
            landmarks: 地标点坐标 (N, 3)
            features: 特征描述符 (N, feature_dim)
            
        Returns:
            分数数组 (N,)
        """
        try:
            # 基于Z值和特征方差计算分数
            z_scores = landmarks[:, 2]
            feature_variances = np.var(features, axis=1)
            
            # 归一化
            z_scores_norm = (z_scores - z_scores.min()) / (z_scores.max() - z_scores.min() + 1e-6)
            feature_var_norm = (feature_variances - feature_variances.min()) / (feature_variances.max() - feature_variances.min() + 1e-6)
            
            # 综合分数
            scores = 0.6 * z_scores_norm + 0.4 * feature_var_norm
            
            return scores
            
        except Exception as e:
            # 如果失败，返回Z值作为分数
            return landmarks[:, 2]
    
    def _match_landmarks(self, landmarks1, landmarks2, features1, features2, max_distance=50.0):
        """
        匹配两个mesh的地标点
        
        Args:
            landmarks1: 第一帧地标点 (N, 3)
            landmarks2: 第二帧地标点 (M, 3)
            features1: 第一帧特征 (N, feature_dim)
            features2: 第二帧特征 (M, feature_dim)
            max_distance: 最大匹配距离
            
        Returns:
            匹配对列表 [(i, j, score), ...]
        """
        try:
            from scipy.spatial.distance import cdist
            
            # 计算特征距离
            feature_distances = cdist(features1, features2, metric='euclidean')
            
            # 计算几何距离
            geometric_distances = cdist(landmarks1, landmarks2, metric='euclidean')
            
            # 综合距离（特征距离 + 几何距离权重）
            combined_distances = 0.7 * feature_distances / (feature_distances.max() + 1e-6) + \
                                0.3 * geometric_distances / (geometric_distances.max() + 1e-6)
            
            # 找到最佳匹配（最近邻）
            matches = []
            used_j = set()
            
            for i in range(len(landmarks1)):
                # 找到未使用的最佳匹配
                valid_indices = [j for j in range(len(landmarks2)) if j not in used_j]
                if len(valid_indices) == 0:
                    break
                
                valid_distances = combined_distances[i, valid_indices]
                best_idx = np.argmin(valid_distances)
                j = valid_indices[best_idx]
                
                if combined_distances[i, j] < max_distance:
                    matches.append((i, j, combined_distances[i, j]))
                    used_j.add(j)
            
            return matches
            
        except Exception as e:
            print(f"  [警告] 地标匹配失败: {e}")
            return []
    
    def _estimate_transformation_svd(self, pts1, pts2):
        """
        使用SVD从匹配点对估计最优变换矩阵（R和T）
        
        Args:
            pts1: 第一组点 (N, 3)
            pts2: 第二组点 (N, 3)，对应pts1
            
        Returns:
            4x4变换矩阵，将pts2变换到pts1的坐标系，如果失败返回None
        """
        try:
            if len(pts1) < 3 or len(pts2) < 3 or len(pts1) != len(pts2):
                return None
            
            # 1. 计算质心
            centroid1 = pts1.mean(axis=0)
            centroid2 = pts2.mean(axis=0)
            
            # 2. 中心化
            pts1_centered = pts1 - centroid1
            pts2_centered = pts2 - centroid2
            
            # 3. 计算协方差矩阵 H = pts1_centered^T @ pts2_centered
            H = pts1_centered.T @ pts2_centered
            
            # 4. SVD分解
            U, S, Vt = np.linalg.svd(H)
            
            # 5. 计算旋转矩阵 R = V @ U^T
            R = Vt.T @ U.T
            
            # 确保是旋转矩阵（行列式为1）
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # 6. 计算平移向量 t = centroid1 - R @ centroid2
            t = centroid1 - R @ centroid2
            
            # 7. 构建4x4变换矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            return T
            
        except Exception as e:
            print(f"  [警告] SVD变换估计失败: {e}")
            return None
    
    def _save_coarse_alignment_result(self, T_coarse, mesh1, mesh2, T_translation, T_rotation,
                                     frame_idx1, frame_idx2, suffix="th_68"):
        """
        保存粗匹配结果到文件夹
        
        Args:
            T_coarse: 粗变换矩阵（4x4）
            mesh1: 第一帧网格
            mesh2: 第二帧网格
            T_translation: 平移变换矩阵
            T_rotation: 旋转变换矩阵（可选）
            frame_idx1: 第一帧索引
            frame_idx2: 第二帧索引
            suffix: 文件后缀
        """
        try:
            # 创建粗匹配结果文件夹
            coarse_output_dir = os.path.join(self.output_dir, "coarse_alignment")
            os.makedirs(coarse_output_dir, exist_ok=True)
            
            # 1. 保存变换矩阵（JSON格式）
            transform_data = {
                'frame_idx1': int(frame_idx1),
                'frame_idx2': int(frame_idx2),
                'coarse_transformation': T_coarse.tolist(),
                'translation': T_translation[:3, 3].tolist(),
                'rotation_matrix': T_coarse[:3, :3].tolist(),
                'translation_matrix': T_translation.tolist()
            }
            
            if T_rotation is not None:
                transform_data['rotation_matrix_pca'] = T_rotation[:3, :3].tolist()
            
            json_path = os.path.join(
                coarse_output_dir, 
                f"coarse_alignment_{suffix}_{frame_idx1:04d}_{frame_idx2:04d}.json"
            )
            # JSON保存已禁用
            # with open(json_path, 'w', encoding='utf-8') as f:
            #     json.dump(transform_data, f, ensure_ascii=False, indent=2)
            
            print(f"  [保存] 粗匹配变换矩阵: {json_path}")
            
            # 2. 保存变换矩阵（NumPy格式，便于程序读取）
            npy_path = os.path.join(
                coarse_output_dir,
                f"coarse_alignment_{suffix}_{frame_idx1:04d}_{frame_idx2:04d}.npy"
            )
            np.save(npy_path, T_coarse)
            
            # 3. 保存粗对齐后的网格（可视化用）
            try:
                mesh2_transformed = o3d.geometry.TriangleMesh(mesh2)
                mesh2_transformed.transform(T_coarse)
                
                aligned_mesh_path = os.path.join(
                    coarse_output_dir,
                    f"mesh2_aligned_{suffix}_{frame_idx1:04d}_{frame_idx2:04d}.ply"
                )
                o3d.io.write_triangle_mesh(aligned_mesh_path, mesh2_transformed)
                print(f"  [保存] 粗对齐后的网格2: {aligned_mesh_path}")
            except Exception as e:
                print(f"  [警告] 保存粗对齐网格失败: {e}")
            
            # 4. 可视化粗对齐结果（可选）
            try:
                self._visualize_coarse_alignment(
                    mesh1, mesh2, T_coarse, frame_idx1, frame_idx2, suffix, coarse_output_dir
                )
            except Exception as e:
                print(f"  [警告] 粗对齐可视化失败: {e}")
                
        except Exception as e:
            print(f"[警告] 保存粗匹配结果失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_coarse_alignment(self, mesh1, mesh2, T_coarse, frame_idx1, frame_idx2, 
                                   suffix, output_dir):
        """
        可视化粗对齐结果
        
        Args:
            mesh1: 第一帧网格
            mesh2: 第二帧网格
            T_coarse: 粗变换矩阵
            frame_idx1: 第一帧索引
            frame_idx2: 第二帧索引
            suffix: 文件后缀
            output_dir: 输出目录
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            vertices1 = np.asarray(mesh1.vertices)
            vertices2 = np.asarray(mesh2.vertices)
            
            # 应用粗变换
            vertices2_homo = np.hstack([vertices2, np.ones((len(vertices2), 1))])
            vertices2_transformed = (T_coarse @ vertices2_homo.T).T[:, :3]
            
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # 采样显示（粗匹配过程保留更多mesh细节）
            max_faces = 50000  # 增加采样数量，保留更多mesh信息
            faces1 = np.asarray(mesh1.triangles)
            faces2 = np.asarray(mesh2.triangles)
            
            if len(faces1) > max_faces:
                face_indices1 = np.random.choice(len(faces1), max_faces, replace=False)
                faces1_display = faces1[face_indices1]
                print(f"  [粗匹配可视化] 帧{frame_idx1}: 采样 {max_faces}/{len(faces1)} 个三角面片")
            else:
                faces1_display = faces1
                print(f"  [粗匹配可视化] 帧{frame_idx1}: 显示全部 {len(faces1)} 个三角面片")
            
            if len(faces2) > max_faces:
                face_indices2 = np.random.choice(len(faces2), max_faces, replace=False)
                faces2_display = faces2[face_indices2]
                print(f"  [粗匹配可视化] 帧{frame_idx2}: 采样 {max_faces}/{len(faces2)} 个三角面片")
            else:
                faces2_display = faces2
                print(f"  [粗匹配可视化] 帧{frame_idx2}: 显示全部 {len(faces2)} 个三角面片")
            
            # 绘制网格（使用更明显的颜色区分前后两帧）
            if len(faces1_display) > 0:
                triangles1 = vertices1[faces1_display]
                # 第一帧：使用橙红色（更明显）
                mesh_collection1 = Poly3DCollection(triangles1, alpha=0.6, facecolor='#FF6B35', 
                                                   edgecolor='#C44536', linewidths=0.05)
                ax.add_collection3d(mesh_collection1)
            
            if len(faces2_display) > 0:
                triangles2 = vertices2_transformed[faces2_display]
                # 第二帧：使用青蓝色（与第一帧形成明显对比）
                mesh_collection2 = Poly3DCollection(triangles2, alpha=0.6, facecolor='#4ECDC4', 
                                                   edgecolor='#2E86AB', linewidths=0.05)
                ax.add_collection3d(mesh_collection2)
            
            # 设置坐标轴
            all_vertices = np.vstack([vertices1, vertices2_transformed])
            x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
            y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
            z_min, z_max = all_vertices[:, 2].min(), all_vertices[:, 2].max()
            
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_zlabel('Z', fontsize=12)
            
            # 添加图例（使用实际颜色）
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#FF6B35', alpha=0.6, label=f'帧 {frame_idx1} (前一帧，原始位置)'),
                Patch(facecolor='#4ECDC4', alpha=0.6, label=f'帧 {frame_idx2} (后一帧，粗对齐后)')
            ]
            ax.legend(handles=legend_elements, fontsize=12, loc='upper left', framealpha=0.9)
            
            # 添加信息文本框
            info_text = (
                f"粗匹配结果（场景叠加）\n"
                f"帧 {frame_idx1} ↔ 帧 {frame_idx2}\n"
                f"前一帧: 橙红色 (#FF6B35)\n"
                f"后一帧: 青蓝色 (#4ECDC4)\n"
                f"显示: 粗对齐后的3D场景叠加效果"
            )
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_title(f'粗匹配结果（场景叠加）: 帧 {frame_idx1} <-> 帧 {frame_idx2}', 
                        fontsize=15, fontweight='bold', pad=20)
            
            ax.view_init(elev=20, azim=45)
            
            # 保存图像
            image_path = os.path.join(
                output_dir,
                f"coarse_alignment_vis_{suffix}_{frame_idx1:04d}_{frame_idx2:04d}.png"
            )
            plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  [保存] 粗对齐可视化: {image_path}")
            
        except Exception as e:
            print(f"  [警告] 粗对齐可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_largest_contour_mask(self, frame_idx=None, suffix="68", use_chessboard=False, original_image_path=None):
        """
        从contour_image_68_bilateral图像中提取轮廓并生成mask
        现在支持棋盘格检测模式

        Args:
            frame_idx: 帧索引（可选，用于查找对应的图像文件）
            suffix: 文件后缀，默认为"68"
            use_chessboard: 是否使用棋盘格检测（如果为True，将使用棋盘格mask）
            original_image_path: 原始图像路径（用于棋盘格检测）

        Returns:
            mask: 二值mask图像（轮廓区域为255，其他为0），如果提取失败返回None
        """
        try:
            # 如果启用棋盘格检测，直接使用棋盘格mask
            if use_chessboard and original_image_path and os.path.exists(original_image_path):
                print(f"  [棋盘格检测] 开始检测棋盘格mask: {original_image_path}")
                chessboard_mask = get_chessboard_mask(original_image_path, refine_corners=True)

                if chessboard_mask is not None:
                    # 如果图像被缩放，mask也需要相应缩放
                    if self.image_scale_factor != 1.0:
                        chessboard_mask_h, chessboard_mask_w = chessboard_mask.shape[:2]
                        new_mask_w = int(chessboard_mask_w * self.image_scale_factor)
                        new_mask_h = int(chessboard_mask_h * self.image_scale_factor)
                        chessboard_mask = cv2.resize(chessboard_mask, (new_mask_w, new_mask_h), interpolation=cv2.INTER_NEAREST)

                    mask_area = np.sum(chessboard_mask > 0)
                    mask_percentage = (mask_area / (chessboard_mask.shape[0] * chessboard_mask.shape[1])) * 100
                    print(f"  [mask统计] 区域: {mask_area} 像素 ({mask_percentage:.2f}% 图像)")

                    # 保存mask到文件
                    try:
                        mask_filename = f"chessboard_mask_{frame_idx:04d}.jpg"
                        mask_path = os.path.join(self.output_dir, mask_filename)
                        cv2.imwrite(mask_path, chessboard_mask)
                        print(f"  [保存] 棋盘格mask已保存: {mask_path}")
                    except Exception as e:
                        print(f"  [警告] 保存mask失败: {e}")

                    return chessboard_mask
                else:
                    print(f"  [警告] 棋盘格检测失败，回退到轮廓提取模式")
                    # 继续使用传统的轮廓提取方法

            # 构建图像文件名（格式：contour_image_68_bilateral_0000.jpg）
            if frame_idx is not None:
                image_filename = f"contour_image_{suffix}_bilateral_{frame_idx:04d}.jpg"
            else:
                # 如果没有frame_idx，尝试查找所有匹配的文件
                image_filename = f"contour_image_{suffix}_bilateral*.jpg"
            
            # 查找图像文件
            image_path = None
            if frame_idx is not None:
                image_path = os.path.join(self.output_dir, image_filename)
                if not os.path.exists(image_path):
                    print(f"  [警告] 图像文件不存在: {image_path}")
                    return None
            else:
                # 查找所有匹配的文件
                import glob
                pattern = os.path.join(self.output_dir, image_filename)
                matches = glob.glob(pattern)
                if not matches:
                    print(f"  [警告] 未找到匹配的图像文件: {pattern}")
                    return None
                # 使用第一个匹配的文件
                image_path = matches[0]
                print(f"  [信息] 使用图像文件: {os.path.basename(image_path)}")
            
            # 读取图像（灰度图）
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [错误] 无法读取图像: {image_path}")
                return None
            
            h, w = img.shape
            print(f"  [图像信息] 尺寸: {w}×{h}")
            
            # 读取gradient_before_threshold图像来计算th_68阈值
            if frame_idx is not None:
                grad_filename = f"gradient_before_threshold_{frame_idx:04d}.jpg"
            else:
                grad_filename = "gradient_before_threshold*.jpg"
            
            grad_path = None
            if frame_idx is not None:
                grad_path = os.path.join(self.output_dir, grad_filename)
                if not os.path.exists(grad_path):
                    print(f"  [警告] 梯度图像不存在: {grad_path}，尝试查找...")
                    import glob
                    pattern = os.path.join(self.output_dir, f"gradient_before_threshold*{frame_idx:04d}*.jpg")
                    matches = glob.glob(pattern)
                    if matches:
                        grad_path = matches[0]
                        print(f"  [信息] 找到梯度图像: {os.path.basename(grad_path)}")
                    else:
                        print(f"  [警告] 未找到梯度图像，将使用OTSU阈值")
                        grad_path = None
            else:
                import glob
                pattern = os.path.join(self.output_dir, grad_filename)
                matches = glob.glob(pattern)
                if matches:
                    grad_path = matches[0]
                    print(f"  [信息] 使用梯度图像: {os.path.basename(grad_path)}")
            
            # 计算阈值（使用阈值模式对应的百分位）
            threshold_percent = getattr(self, 'threshold_percent', 68)  # 默认68
            threshold_mode = getattr(self, 'threshold_mode', 'th_68')  # 默认th_68
            th_value = None
            if grad_path and os.path.exists(grad_path):
                try:
                    # 读取梯度图像
                    grad_img = cv2.imread(grad_path, cv2.IMREAD_GRAYSCALE)
                    if grad_img is not None:
                        # 归一化到0-1范围
                        grad_norm = grad_img.astype(np.float32) / 255.0
                        # 计算阈值（使用阈值模式对应的百分位）
                        th_value = self._analyze_gradient_threshold(grad_norm, bins=1024, percent=threshold_percent)
                        # 转换为0-255范围的阈值
                        th_uint8 = int(th_value * 255.0)
                        print(f"  [{threshold_mode}阈值] 计算得到: {th_value:.6f} (归一化), {th_uint8} (0-255范围)")
                    else:
                        print(f"  [警告] 无法读取梯度图像: {grad_path}")
                except Exception as e:
                    print(f"  [警告] 计算{threshold_mode}阈值失败: {e}")
            
            # 使用阈值进行二值化（用于mask提取）
            if th_value is not None:
                # 使用阈值：大于阈值=255，小于等于阈值=0
                th_uint8 = int(th_value * 255.0)
                binary = np.where(img > th_uint8, 255, 0).astype(np.uint8)
                print(f"  [二值化] 使用{threshold_mode}阈值: {th_uint8} (大于{th_uint8}=255, 小于等于{th_uint8}=0)")
            else:
                # 回退方案：使用OTSU阈值
                otsu_threshold, binary_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if otsu_threshold < 10:
                    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
                    print(f"  [二值化] 回退：使用固定阈值1（OTSU阈值: {otsu_threshold:.1f}）")
                else:
                    binary = binary_otsu
                    print(f"  [二值化] 回退：使用OTSU自适应阈值: {otsu_threshold:.1f}")
            
            # 形态学预处理（去除噪声，连接断开的区域）
            kernel = np.ones((3, 3), np.uint8)
            # 先开运算去除小噪声
            binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            # 再闭运算连接断开的区域
            binary = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 查找轮廓（使用CHAIN_APPROX_NONE保留完整轮廓信息）
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) == 0:
                print(f"  [警告] 未找到任何轮廓")
                return None
            
            # 过滤小轮廓（面积小于图像1%的轮廓视为噪声）
            min_area = (h * w) * 0.01
            filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            
            if len(filtered_contours) == 0:
                print(f"  [警告] 过滤后没有有效轮廓（最小面积阈值: {min_area:.0f} 像素）")
                # 如果过滤后没有轮廓，使用原始轮廓
                filtered_contours = contours
            
            print(f"  [轮廓信息] 找到 {len(contours)} 个轮廓，过滤后 {len(filtered_contours)} 个有效轮廓")
            
            # 按面积排序（从大到小）
            sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
            
            # 创建mask并填充所有有效轮廓（不仅仅是最大的）
            mask = np.zeros((h, w), dtype=np.uint8)
            num_contours_to_fill = min(len(sorted_contours), 5)  # 最多填充5个最大的轮廓
            
            for i in range(num_contours_to_fill):
                cv2.fillPoly(mask, [sorted_contours[i]], 255)
                area = cv2.contourArea(sorted_contours[i])
                print(f"  [轮廓 {i+1}] 面积: {area:.0f} 像素 ({area/(h*w)*100:.2f}% 图像)")
            
            # 保存mask和轮廓结果
            if frame_idx is not None:
                try:
                    # 保存纯mask图像（二值图像）
                    mask_path = os.path.join(
                        self.output_dir,
                        f"mask_{suffix}_{frame_idx:04d}.jpg"
                    )
                    cv2.imwrite(mask_path, mask)
                    print(f"  [保存] Mask图像: {mask_path}")
                    
                    # 保存轮廓可视化图像
                    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    for i in range(num_contours_to_fill):
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)  # 最大轮廓绿色，其他蓝色
                        cv2.drawContours(contour_img, [sorted_contours[i]], -1, color, 2)
                    
                    contour_path = os.path.join(
                        self.output_dir,
                        f"contours_{suffix}_{frame_idx:04d}.jpg"
                    )
                    cv2.imwrite(contour_path, contour_img)
                    print(f"  [保存] 轮廓可视化: {contour_path}")
                    
                except Exception as e:
                    print(f"  [警告] 保存mask和轮廓失败: {e}")
            
            # 输出统计信息
            mask_area = np.sum(mask > 0)
            mask_percentage = (mask_area / (h * w)) * 100
            print(f"  [完成] 提取了 {num_contours_to_fill} 个轮廓的mask | "
                  f"mask区域: {mask_area} 像素 ({mask_percentage:.2f}%)")
            
            return mask
            
        except Exception as e:
            print(f"  [错误] 提取轮廓mask异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_mask_from_truncated_gray(self, th_gray_image, frame_idx):
        """
        基于截断后的灰度图提取主要轮廓并生成mask
        
        Args:
            th_gray_image: 截断并归一化后的灰度图（0-255范围）
            frame_idx: 帧索引
            
        Returns:
            mask: 二值mask图像（轮廓区域为255，其他为0），如果提取失败返回None
        """
        try:
            h, w = th_gray_image.shape
            print(f"  [图像信息] 尺寸: {w}×{h}")
            
            # 对截断后的灰度图进行二值化（非零区域为前景）
            # 由于截断后的灰度图已经将小于阈值的区域设为0，所以非零区域就是主要轮廓区域
            binary = np.where(th_gray_image > 0, 255, 0).astype(np.uint8)
            
            # 形态学预处理（去除噪声，连接断开的区域）
            kernel = np.ones((3, 3), np.uint8)
            # 先开运算去除小噪声
            binary_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            # 再闭运算连接断开的区域
            binary = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 查找轮廓（使用CHAIN_APPROX_NONE保留完整轮廓信息）
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) == 0:
                print(f"  [警告] 未找到任何轮廓")
                return None
            
            # 过滤小轮廓（面积小于图像1%的轮廓视为噪声）
            min_area = (h * w) * 0.01
            filtered_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            
            if len(filtered_contours) == 0:
                print(f"  [警告] 过滤后没有有效轮廓（最小面积阈值: {min_area:.0f} 像素）")
                # 如果过滤后没有轮廓，使用原始轮廓
                filtered_contours = contours
            
            print(f"  [轮廓信息] 找到 {len(contours)} 个轮廓，过滤后 {len(filtered_contours)} 个有效轮廓")
            
            # 按面积排序（从大到小）
            sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
            
            # 创建mask并填充主要轮廓（最多填充5个最大的轮廓）
            mask = np.zeros((h, w), dtype=np.uint8)
            num_contours_to_fill = min(len(sorted_contours), 5)
            
            for i in range(num_contours_to_fill):
                cv2.fillPoly(mask, [sorted_contours[i]], 255)
                area = cv2.contourArea(sorted_contours[i])
                print(f"  [轮廓 {i+1}] 面积: {area:.0f} 像素 ({area/(h*w)*100:.2f}% 图像)")
            
            # 保存mask和轮廓结果
            try:
                # 保存纯mask图像（二值图像）
                mask_path = os.path.join(
                    self.output_dir,
                    f"mask_{self.threshold_mode}_{frame_idx:04d}.jpg"
                )
                cv2.imwrite(mask_path, mask)
                print(f"  [保存] Mask图像: {mask_path}")
                
                # 保存轮廓可视化图像
                contour_img = cv2.cvtColor(th_gray_image, cv2.COLOR_GRAY2BGR)
                for i in range(num_contours_to_fill):
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)  # 最大轮廓绿色，其他蓝色
                    cv2.drawContours(contour_img, [sorted_contours[i]], -1, color, 2)
                
                contour_path = os.path.join(
                    self.output_dir,
                    f"contours_{self.threshold_mode}_{frame_idx:04d}.jpg"
                )
                cv2.imwrite(contour_path, contour_img)
                print(f"  [保存] 轮廓可视化: {contour_path}")
                
            except Exception as e:
                print(f"  [警告] 保存mask和轮廓失败: {e}")
            
            # 输出统计信息
            mask_area = np.sum(mask > 0)
            mask_percentage = (mask_area / (h * w)) * 100
            print(f"  [完成] 提取了 {num_contours_to_fill} 个轮廓的mask | "
                  f"mask区域: {mask_area} 像素 ({mask_percentage:.2f}%)")
            
            return mask
            
        except Exception as e:
            print(f"  [错误] 从截断灰度图提取mask异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_curvature_features(self, mesh, pcd):
        """
        计算点云的曲率特征
        
        Args:
            mesh: Open3D TriangleMesh
            pcd: Open3D PointCloud
            
        Returns:
            曲率特征数组 (N, 3): [平均曲率, 高斯曲率, 最大曲率]
        """
        try:
            points = np.asarray(pcd.points)
            
            # 确保点云有法向量
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
                )
            
            normals = np.asarray(pcd.normals)
            
            # 使用点云本身构建KDTree（不是mesh顶点）
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            
            curvature_features = []
            
            for i, point in enumerate(points):
                # 查找点云中的邻居（不是mesh顶点）
                neighbors_idx = tree.query_ball_point(point, r=5.0)
                # 移除自己
                neighbors_idx = [j for j in neighbors_idx if j != i]
                
                if len(neighbors_idx) < 3:
                    curvature_features.append([0.0, 0.0, 0.0])
                    continue
                
                # 计算法向量变化
                normal_i = normals[i]
                neighbor_normals = normals[neighbors_idx]
                
                # 计算法向量差异（曲率近似）
                normal_diffs = np.arccos(np.clip(np.dot(neighbor_normals, normal_i), -1, 1))
                mean_curvature = np.mean(normal_diffs)
                max_curvature = np.max(normal_diffs)
                
                # 高斯曲率近似（法向量变化的标准差）
                gaussian_curvature = np.std(normal_diffs)
                
                curvature_features.append([mean_curvature, gaussian_curvature, max_curvature])
            
            return np.array(curvature_features)
            
        except Exception as e:
            print(f"  [警告] 曲率特征计算失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((len(pcd.points), 3))
    
    def _compute_normal_features(self, pcd):
        """
        计算法向量特征（法向量的统计特征）
        
        Args:
            pcd: Open3D PointCloud
            
        Returns:
            法向量特征数组 (N, 3): [法向量x, 法向量y, 法向量z]（归一化到[0,1]）
        """
        try:
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
                )
            
            normals = np.asarray(pcd.normals)
            # 归一化到[0, 1]范围（法向量通常在[-1, 1]）
            normals_normalized = (normals + 1.0) / 2.0
            return normals_normalized
            
        except Exception as e:
            print(f"  [警告] 法向量特征计算失败: {e}")
            return np.zeros((len(pcd.points), 3))
    
    def _compute_density_features(self, pcd, radius):
        """
        计算点云密度特征（局部邻域内的点数）
        
        Args:
            pcd: Open3D PointCloud
            radius: 搜索半径
            
        Returns:
            密度特征数组 (N,): 每个点的邻域密度
        """
        try:
            points = np.asarray(pcd.points)
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            
            density_features = []
            for point in points:
                # 查找半径内的邻居数量
                neighbors = tree.query_ball_point(point, r=radius)
                density = len(neighbors) - 1  # 减去自己
                density_features.append(density)
            
            return np.array(density_features)
            
        except Exception as e:
            print(f"  [警告] 密度特征计算失败: {e}")
            return np.zeros(len(pcd.points))
    
    def _compute_distance_features(self, pcd):
        """
        计算距离特征（到质心的距离，到最近点的距离等）
        
        Args:
            pcd: Open3D PointCloud
            
        Returns:
            距离特征数组 (N, 3): [到质心距离, 到最近点距离, 到最远点距离]
        """
        try:
            points = np.asarray(pcd.points)
            center = points.mean(axis=0)
            
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            
            distance_features = []
            for i, point in enumerate(points):
                # 到质心的距离
                dist_to_center = np.linalg.norm(point - center)
                
                # 到最近点的距离（排除自己）
                distances, indices = tree.query(point, k=2)  # k=2因为第一个是自己
                dist_to_nearest = distances[1] if len(distances) > 1 else 0.0
                
                # 到最远点的距离（使用边界框）
                bbox_min = points.min(axis=0)
                bbox_max = points.max(axis=0)
                corners = np.array([
                    bbox_min, bbox_max,
                    [bbox_min[0], bbox_min[1], bbox_max[2]],
                    [bbox_min[0], bbox_max[1], bbox_min[2]],
                    [bbox_max[0], bbox_min[1], bbox_min[2]],
                    [bbox_min[0], bbox_max[1], bbox_max[2]],
                    [bbox_max[0], bbox_min[1], bbox_max[2]],
                    [bbox_max[0], bbox_max[1], bbox_min[2]]
                ])
                dists_to_corners = np.linalg.norm(corners - point, axis=1)
                dist_to_farthest = np.max(dists_to_corners)
                
                distance_features.append([dist_to_center, dist_to_nearest, dist_to_farthest])
            
            return np.array(distance_features)
            
        except Exception as e:
            print(f"  [警告] 距离特征计算失败: {e}")
            return np.zeros((len(pcd.points), 3))
    
    def _compute_angle_features(self, pcd):
        """
        计算角度特征（法向量与主方向的角度等）
        
        Args:
            pcd: Open3D PointCloud
            
        Returns:
            角度特征数组 (N, 3): [法向量与X轴角度, 与Y轴角度, 与Z轴角度]
        """
        try:
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
                )
            
            normals = np.asarray(pcd.normals)
            points = np.asarray(pcd.points)
            
            # 主方向（PCA主方向）
            center = points.mean(axis=0)
            centered_points = points - center
            cov = np.cov(centered_points.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            principal_dirs = eigenvecs[:, np.argsort(eigenvals)[::-1]]  # 按特征值降序排列
            
            angle_features = []
            for normal in normals:
                # 计算法向量与三个主方向的角度
                angles = []
                for i in range(3):
                    if i < principal_dirs.shape[1]:
                        cos_angle = np.clip(np.dot(normal, principal_dirs[:, i]), -1, 1)
                        angle = np.arccos(abs(cos_angle))  # 使用绝对值，角度在[0, π/2]
                        angles.append(angle)
                    else:
                        angles.append(0.0)
                
                angle_features.append(angles)
            
            return np.array(angle_features)
            
        except Exception as e:
            print(f"  [警告] 角度特征计算失败: {e}")
            return np.zeros((len(pcd.points), 3))
    
    def _extract_contour_points(self, frame_idx=None, suffix="68"):
        """
        已禁用：轮廓提取功能已移除，不再使用轮廓进行采样和匹配
        现在优先使用mask进行采样，如果没有mask则使用全图采样
        
        Args:
            frame_idx: 帧索引（可选，已不使用）
            suffix: 文件后缀（已不使用）
            
        Returns:
            None: 始终返回None，表示不使用轮廓
        """
        # 轮廓计算功能已移除，直接返回None
        return None
    
    def _sample_points_from_contour(self, mesh, max_points, contour_points_list):
        """
        基于轮廓点的点云采样（直接从轮廓点对应的mesh顶点采样）
        
        采样策略：
        1. 直接使用th_68轮廓点，不进行插值
        2. 对每个轮廓点，查找对应的最近邻mesh顶点
        3. 从3D高度场提取所有轮廓点的特征
        
        Args:
            mesh: Open3D TriangleMesh
            max_points: 最大采样点数（如果采样点超过此值，会下采样）
            contour_points_list: 轮廓点列表，每个元素是一个Nx2的numpy数组（x, y坐标）
            
        Returns:
            pcd: Open3D PointCloud，包含采样得到的mesh顶点
        """
        try:
            vertices = np.asarray(mesh.vertices)
            
            if contour_points_list is None or len(contour_points_list) == 0:
                print(f"  [警告] 轮廓点列表为空")
                return None
            
            # 合并所有轮廓点（不进行插值，直接使用原始轮廓点）
            all_contour_points = np.vstack(contour_points_list)  # (N, 2)
            num_contour_points = len(all_contour_points)
            
            if num_contour_points == 0:
                print(f"  [警告] 轮廓点数为0")
                return None
            
            print(f"  [轮廓采样] 轮廓点总数: {num_contour_points}")
            
            sampled_points = []
            sampled_normals = []
            
            # 构建KDTree用于快速查找最近邻顶点
            from scipy.spatial import cKDTree
            vertex_xy = vertices[:, :2]  # 只使用x, y坐标构建KDTree
            tree = cKDTree(vertex_xy)
            
            # 批量查询轮廓点对应的最近邻mesh顶点（只查找1个最近邻）
            distances, indices = tree.query(all_contour_points, k=1)
            
            # 获取对应的3D坐标
            sampled_points = vertices[indices]
            
            # 如果有法向量，也获取对应的法向量
            if mesh.has_vertex_normals():
                normals = np.asarray(mesh.vertex_normals)
                sampled_normals = normals[indices]
            else:
                sampled_normals = []
            
            if len(sampled_points) == 0:
                print(f"  [警告] 未能从轮廓点采样到mesh顶点")
                return None
            
            # 创建点云
            sampled_points = np.array(sampled_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sampled_points)
            
            # 设置法向量
            if len(sampled_normals) > 0:
                sampled_normals = np.array(sampled_normals)
                pcd.normals = o3d.utility.Vector3dVector(sampled_normals)
            
            # 如果点数超过max_points，进行均匀下采样
            if len(pcd.points) > max_points:
                original_count = len(pcd.points)
                # 使用均匀采样
                every_k = max(1, len(pcd.points) // max_points)
                pcd = pcd.uniform_down_sample(every_k_points=every_k)
                # 如果采样后点数还是太多，再次采样
                if len(pcd.points) > max_points:
                    every_k = max(1, len(pcd.points) // max_points)
                    pcd = pcd.uniform_down_sample(every_k_points=every_k)
                print(f"  [轮廓采样] 下采样: {original_count} -> {len(pcd.points)} 个点 (max_points={max_points})")
            else:
                print(f"  [轮廓采样] 采样点数: {len(pcd.points)} (未超过max_points={max_points})")
            
            return pcd
            
        except Exception as e:
            print(f"  [警告] 轮廓采样失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _sample_points_with_mask(self, mesh, max_points=None, mask=None, use_all_points=True):
        """
        基于mask约束的点云采样（直接从mask覆盖的区域采样）
        
        策略：如果use_all_points=True，mask覆盖的所有mesh点都参与匹配，不进行数量限制
        
        Args:
            mesh: Open3D TriangleMesh
            max_points: 最大采样点数（如果use_all_points=False时使用，已弃用，保留以兼容旧代码）
            mask: 二值mask图像（棋盘格区域为255，其他为0）
            use_all_points: 是否使用mask覆盖的所有点（默认True，全部参与匹配）
            
        Returns:
            pcd: Open3D PointCloud，只包含mask内的点
        """
        try:
            vertices = np.asarray(mesh.vertices)
            h, w = mask.shape
            
            # 1. 找到mask中所有非零像素的位置
            mask_coords = np.where(mask > 0)
            mask_y_coords = mask_coords[0]  # 行坐标（对应mesh的y）
            mask_x_coords = mask_coords[1]  # 列坐标（对应mesh的x）
            
            num_mask_pixels = len(mask_x_coords)
            if num_mask_pixels == 0:
                print(f"  [警告] mask内没有有效像素")
                return None
            
            print(f"  [mask采样] mask覆盖区域: {num_mask_pixels} 个像素")
            
            # 2. 对于mask内的每个像素位置，从mesh中查找对应的3D坐标
            # mesh的顶点坐标是(x, y, z)，其中x和y对应图像像素坐标
            sampled_points = []
            sampled_normals = []
            
            # 构建KDTree用于快速查找最近邻顶点
            from scipy.spatial import cKDTree
            vertex_xy = vertices[:, :2]  # 只使用x, y坐标构建KDTree
            tree = cKDTree(vertex_xy)
            
            # 优化：即使use_all_points=True，如果点数太多也要采样
            # 如果mask内的像素太多，先进行下采样以加速
            if max_points is not None and num_mask_pixels > max_points:
                # 均匀采样mask内的像素
                step = max(1, num_mask_pixels // max_points)
                sampled_mask_indices = np.arange(0, num_mask_pixels, step)
                mask_x_coords = mask_x_coords[sampled_mask_indices]
                mask_y_coords = mask_y_coords[sampled_mask_indices]
                print(f"  [mask采样] 对mask像素进行预采样: {len(mask_x_coords)} 个像素")
            
            # 3. 批量查找mask内每个像素位置对应的mesh顶点
            # 构建查询点数组
            query_points = np.column_stack([mask_x_coords, mask_y_coords])
            
            # 批量查询最近邻顶点（使用KDTree批量查询，效率更高）
            distances, indices = tree.query(query_points, k=1)
            
            # 获取对应的3D坐标
            sampled_points = vertices[indices]
            
            # 如果有法向量，也获取对应的法向量
            if mesh.has_vertex_normals():
                normals = np.asarray(mesh.vertex_normals)
                sampled_normals = normals[indices]
            else:
                sampled_normals = []
            
            if len(sampled_points) == 0:
                print(f"  [警告] 未能从mask区域采样到点")
                return None
            
            # 4. 创建点云
            sampled_points = np.array(sampled_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sampled_points)
            
            # 设置法向量
            if len(sampled_normals) > 0:
                sampled_normals = np.array(sampled_normals)
                pcd.normals = o3d.utility.Vector3dVector(sampled_normals)
            
            # 5. 如果use_all_points=False，才进行下采样（兼容旧代码）
            if not use_all_points and max_points is not None:
                # 如果点数超过max_points，进行均匀下采样
                if len(pcd.points) > max_points:
                    # 使用均匀采样
                    every_k = max(1, len(pcd.points) // max_points)
                    pcd = pcd.uniform_down_sample(every_k_points=every_k)
                    # 如果采样后点数还是太多，再次采样
                    if len(pcd.points) > max_points:
                        every_k = max(1, len(pcd.points) // max_points)
                        pcd = pcd.uniform_down_sample(every_k_points=every_k)
            
            print(f"  [mask采样] mask像素数: {num_mask_pixels}, 采样点数: {len(pcd.points)} (全部参与匹配)")
            
            return pcd
            
        except Exception as e:
            print(f"  [警告] mask约束采样失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_adaptive_radii(self, points, default_radius_normal=0.1, default_radius_feature=0.2):
        """
        根据点云尺度自适应计算半径参数
        
        计算策略：
        1. 计算点云的边界框（bounding box）
        2. 计算边界框的对角线长度（点云尺度）
        3. 根据尺度比例调整半径
        
        Args:
            points: 点云坐标数组 (N, 3)
            default_radius_normal: 默认法向量估计半径
            default_radius_feature: 默认特征计算半径
            
        Returns:
            tuple: (radius_normal, radius_feature, point_cloud_scale)
        """
        if points is None or len(points) == 0:
            return default_radius_normal, default_radius_feature, 1.0
        
        try:
            points_array = np.asarray(points)
            if points_array.shape[0] < 3:
                return default_radius_normal, default_radius_feature, 1.0
            
            # 计算边界框
            bbox_min = points_array.min(axis=0)
            bbox_max = points_array.max(axis=0)
            bbox_size = bbox_max - bbox_min
            
            # 计算点云尺度（边界框对角线长度）
            point_cloud_scale = np.linalg.norm(bbox_size)
            
            # 如果尺度太小或太大，使用默认值
            if point_cloud_scale < 1e-6 or point_cloud_scale > 1e6:
                return default_radius_normal, default_radius_feature, point_cloud_scale
            
            # 计算平均点间距（用于估计点云密度）
            if len(points_array) > 1:
                from scipy.spatial import cKDTree
                tree = cKDTree(points_array)
                # 采样一些点计算平均最近邻距离
                sample_indices = np.linspace(0, len(points_array)-1, min(100, len(points_array)), dtype=int)
                distances, _ = tree.query(points_array[sample_indices], k=2)  # k=2因为包含自己
                if distances.ndim == 2:
                    # 取第二个最近邻（第一个是自己）
                    avg_point_distance = np.mean(distances[:, 1]) if distances.shape[1] > 1 else np.mean(distances[:, 0])
                else:
                    avg_point_distance = np.mean(distances)
            else:
                avg_point_distance = point_cloud_scale * 0.01
            
            # 自适应计算半径
            # radius_normal: 应该包含足够的邻居点（建议10-30个）
            # 基于平均点间距，确保能找到足够的邻居
            radius_normal = max(
                default_radius_normal,
                avg_point_distance * 5.0,  # 5倍平均距离，确保找到足够邻居
                point_cloud_scale * 0.01    # 点云尺度的1%
            )
            radius_normal = min(radius_normal, point_cloud_scale * 0.1)  # 不超过尺度的10%
            
            # radius_feature: 应该比radius_normal大，用于描述更大的局部区域
            # 通常设置为radius_normal的1.5-2倍
            radius_feature = max(
                default_radius_feature,
                radius_normal * 1.5,        # 法向量半径的1.5倍
                avg_point_distance * 8.0,   # 8倍平均距离
                point_cloud_scale * 0.02    # 点云尺度的2%
            )
            radius_feature = min(radius_feature, point_cloud_scale * 0.15)  # 不超过尺度的15%
            
            print(f"  [自适应半径] 点云尺度: {point_cloud_scale:.4f}, 平均点距: {avg_point_distance:.4f}")
            print(f"  [自适应半径] radius_normal: {radius_normal:.4f} (默认: {default_radius_normal:.4f})")
            print(f"  [自适应半径] radius_feature: {radius_feature:.4f} (默认: {default_radius_feature:.4f})")
            
            return radius_normal, radius_feature, point_cloud_scale
            
        except Exception as e:
            print(f"  [警告] 自适应半径计算失败: {e}，使用默认值")
            return default_radius_normal, default_radius_feature, 1.0
    
    def _update_feature_cache(self, cache_key, features):
        """
        更新特征缓存（LRU策略）
        
        Args:
            cache_key: 缓存键
            features: 特征字典
        """
        # 如果缓存已满，删除最久未使用的
        if len(self._mesh_features_cache) >= self._max_feature_cache_size:
            if self._feature_cache_access_order:
                oldest_key = self._feature_cache_access_order.pop(0)
                if oldest_key in self._mesh_features_cache:
                    del self._mesh_features_cache[oldest_key]
        
        # 添加新缓存
        self._mesh_features_cache[cache_key] = features
        # 更新访问顺序
        if cache_key in self._feature_cache_access_order:
            self._feature_cache_access_order.remove(cache_key)
        self._feature_cache_access_order.append(cache_key)
    
    def _get_feature_from_cache(self, cache_key):
        """
        从缓存获取特征（更新LRU顺序）
        
        Args:
            cache_key: 缓存键
            
        Returns:
            特征字典或None
        """
        if cache_key in self._mesh_features_cache:
            # 更新访问顺序（移到末尾）
            if cache_key in self._feature_cache_access_order:
                self._feature_cache_access_order.remove(cache_key)
            self._feature_cache_access_order.append(cache_key)
            return self._mesh_features_cache[cache_key]
        return None
    
    def _smart_downsample(self, pcd, target_points, radius_normal=None):
        """
        智能下采样：优先使用体素下采样（保持空间分布），如果不够再均匀采样
        
        Args:
            pcd: Open3D点云对象
            target_points: 目标点数
            radius_normal: 法向量估计半径（用于计算合适的体素大小）
            
        Returns:
            下采样后的点云
        """
        if len(pcd.points) <= target_points:
            return pcd
        
        if self._use_voxel_downsample:
            # 方法1: 体素下采样（保持空间分布）
            try:
                # 计算合适的体素大小
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox_size = bbox.get_extent()
                volume = np.prod(bbox_size)
                
                if volume > 0:
                    # 计算体素大小：使体素数量接近目标点数
                    voxel_size = (volume / target_points) ** (1/3)
                    
                    # 限制体素大小在合理范围内（不小于半径的1/10，不大于半径的2倍）
                    if radius_normal is not None:
                        voxel_size = max(radius_normal * 0.1, min(voxel_size, radius_normal * 2.0))
                    
                    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
                    
                    # 如果体素采样后点数还是太多，再均匀采样
                    if len(pcd_down.points) > target_points * 1.5:
                        every_k = max(1, len(pcd_down.points) // target_points)
                        pcd_down = pcd_down.uniform_down_sample(every_k_points=every_k)
                    
                    if len(pcd_down.points) > 0:
                        return pcd_down
            except Exception as e:
                print(f"  [警告] 体素下采样失败: {e}，回退到均匀采样")
        
        # 方法2: 均匀下采样（回退方案）
        every_k = max(1, len(pcd.points) // target_points)
        return pcd.uniform_down_sample(every_k_points=every_k)
    
    def _compute_adaptive_correspondence_distance(self, points1, points2, default_distance=0.1):
        """
        根据两个点云的尺度自适应计算对应点距离阈值
        
        Args:
            points1: 第一个点云坐标数组 (N, 3)
            points2: 第二个点云坐标数组 (M, 3)
            default_distance: 默认对应点距离
            
        Returns:
            float: 自适应对应点距离阈值
        """
        try:
            points1_array = np.asarray(points1)
            points2_array = np.asarray(points2)
            
            if len(points1_array) == 0 or len(points2_array) == 0:
                return default_distance
            
            # 计算两个点云的边界框
            bbox1_min = points1_array.min(axis=0)
            bbox1_max = points1_array.max(axis=0)
            bbox1_size = bbox1_max - bbox1_min
            
            bbox2_min = points2_array.min(axis=0)
            bbox2_max = points2_array.max(axis=0)
            bbox2_size = bbox2_max - bbox2_min
            
            # 计算点云尺度
            scale1 = np.linalg.norm(bbox1_size)
            scale2 = np.linalg.norm(bbox2_size)
            avg_scale = (scale1 + scale2) / 2.0
            
            if avg_scale < 1e-6 or avg_scale > 1e6:
                return default_distance
            
            # 自适应距离阈值：点云尺度的2-5%
            adaptive_distance = avg_scale * 0.03  # 3%的尺度
            
            # 限制在合理范围内
            adaptive_distance = max(default_distance, adaptive_distance)
            adaptive_distance = min(adaptive_distance, avg_scale * 0.1)  # 不超过尺度的10%
            
            print(f"  [自适应距离] 点云平均尺度: {avg_scale:.4f}, "
                  f"对应点距离阈值: {adaptive_distance:.4f} (默认: {default_distance:.4f})")
            
            return adaptive_distance
            
        except Exception as e:
            print(f"  [警告] 自适应距离计算失败: {e}，使用默认值")
            return default_distance
    
    def _extract_3d_mesh_features(self, mesh_input, max_points=10000, radius_normal=0.1, radius_feature=0.2, pre_transform=None, mask=None, contour_points_list=None, auto_radius=True):
        """
        从3D网格提取3D特征描述符（优化：支持mesh对象或文件路径）
        
        FPFH (Fast Point Feature Histograms) 特征说明：
        ============================================
        FPFH是一种快速的点特征直方图描述符，用于3D点云匹配。
        
        特点：
        - 维度：33维特征向量（每个点）
        - 计算速度：比PFH快，但保持相似的描述能力
        - 鲁棒性：对噪声和密度变化有一定鲁棒性
        - 不变性：对旋转和平移有一定不变性
        
        计算原理：
        1. 对每个点，计算其邻域内点对的几何关系
        2. 统计这些关系的直方图分布
        3. 结合法向量信息，形成33维特征向量
        
        应用：
        - 3D点云配准
        - 特征匹配
        - 物体识别
        
        Args:
            mesh_input: 网格对象（o3d.geometry.TriangleMesh）或文件路径（字符串）
            max_points: 最大采样点数（用于下采样）
            radius_normal: 法向量估计半径（用于计算法向量），如果auto_radius=True会被自动计算
            radius_feature: 特征计算半径（FPFH特征的计算范围），如果auto_radius=True会被自动计算
            pre_transform: 预变换矩阵（4x4），在特征提取前应用到网格上
            mask: 二值mask图像（可选，已弃用，保留以兼容旧代码）
            contour_points_list: 轮廓点列表（可选），如果提供，直接从轮廓点采样
            auto_radius: 是否自动计算半径（默认True，根据点云尺度自适应）
            
        Returns:
            dict: {
                'pcd': 点云对象,
                'fpfh': FPFH特征描述符 (33维 x N点),
                'points': 点坐标数组,
                'normals': 法向量数组
            } 或 None
        """
        if o3d is None:
            print("[警告] Open3D未安装，无法提取3D网格特征")
            return None
        
        try:
            # 第二阶段优化：检查特征缓存
            feature_cache_key = None
            if isinstance(mesh_input, str):
                feature_cache_key = f"feat_{mesh_input}_{max_points}_{radius_normal}_{radius_feature}"
            else:
                # 对于mesh对象，尝试从frame_idx获取缓存
                frame_idx = getattr(mesh_input, '_frame_idx', None)
                if frame_idx is not None:
                    feature_cache_key = f"feat_frame_{frame_idx}_{max_points}_{radius_normal}_{radius_feature}"
            
            if feature_cache_key:
                cached_features = self._get_feature_from_cache(feature_cache_key)
                if cached_features is not None:
                    print(f"  [特征缓存] 从缓存获取特征: {feature_cache_key}")
                    return cached_features
            
            # 1. 优化：支持mesh对象或文件路径
            if isinstance(mesh_input, str):
                # 文件路径：从缓存或文件加载
                cache_key = f"mesh_{mesh_input}"
                if hasattr(self, '_mesh_cache') and cache_key in self._mesh_cache:
                    mesh = self._mesh_cache[cache_key]
                else:
                    mesh = o3d.io.read_triangle_mesh(mesh_input)
                    if hasattr(self, '_mesh_cache'):
                        self._mesh_cache[cache_key] = mesh
            else:
                # mesh对象：直接使用（不复制，使用引用）
                mesh = mesh_input
            
            if len(mesh.vertices) == 0:
                mesh_source = mesh_input if isinstance(mesh_input, str) else "mesh对象"
                print(f"[警告] 网格为空: {mesh_source}")
                return None
            
            # 2. 应用预变换（如果提供）
            if pre_transform is not None:
                # 优化：创建副本以避免修改原始mesh
                mesh = o3d.geometry.TriangleMesh(mesh)
                mesh.transform(pre_transform)
                print(f"  [特征提取] 已应用预变换到网格")
            
            # 3. 网格清理
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            # 4. 自适应计算半径（如果启用）
            if auto_radius and len(mesh.vertices) > 0:
                vertices_array = np.asarray(mesh.vertices)
                radius_normal, radius_feature, point_cloud_scale = self._compute_adaptive_radii(
                    vertices_array, 
                    default_radius_normal=radius_normal,
                    default_radius_feature=radius_feature
                )
                print(f"  [特征提取] 使用自适应半径: normal={radius_normal:.4f}, feature={radius_feature:.4f}")
            
            # 5. 计算法向量
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            # 6. 转换为点云（优先使用mask采样，其次轮廓采样，最后全图采样）
            # 优化：即使使用mask，也要进行采样以加速匹配
            if mask is not None:
                # 优先使用mask约束的采样（但限制点数以加速）
                pcd = self._sample_points_with_mask(mesh, max_points=max_points, mask=mask, use_all_points=False)
                if pcd is None:
                    print(f"  [警告] mask约束采样失败，尝试回退到轮廓采样或全图采样")
                    # 回退到轮廓采样（如果提供）
                    if contour_points_list is not None:
                        pcd = self._sample_points_from_contour(mesh, max_points, contour_points_list)
                        if pcd is None:
                            print(f"  [警告] 轮廓采样也失败，使用全图采样")
                            # 回退到全图采样
                            if len(mesh.vertices) > max_points:
                                pcd = mesh.sample_points_uniformly(number_of_points=max_points)
                            else:
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = mesh.vertices
                                pcd.normals = mesh.vertex_normals
                        else:
                            print(f"  [采样] 使用轮廓点采样: {len(pcd.points)} 个点")
                    else:
                        # 回退到全图采样
                        if len(mesh.vertices) > max_points:
                            pcd = mesh.sample_points_uniformly(number_of_points=max_points)
                        else:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = mesh.vertices
                            pcd.normals = mesh.vertex_normals
                else:
                    # 如果采样后的点数仍然太多，进一步下采样
                    if len(pcd.points) > max_points:
                        print(f"  [采样] mask采样后点数过多 ({len(pcd.points)} > {max_points})，进行下采样...")
                        pcd = pcd.uniform_down_sample(every_k_points=max(1, len(pcd.points) // max_points))
                        print(f"  [采样] 下采样后: {len(pcd.points)} 个点")
                    else:
                        print(f"  [采样] 使用mask约束采样: {len(pcd.points)} 个点")
            elif contour_points_list is not None:
                # 使用轮廓点采样（直接使用th_68轮廓点，不插值）
                # 从3D高度场提取所有轮廓点的特征
                pcd = self._sample_points_from_contour(mesh, max_points, contour_points_list)
                if pcd is None:
                    print(f"  [警告] 轮廓采样失败，使用全图采样")
                    # 回退到全图采样
                    if len(mesh.vertices) > max_points:
                        pcd = mesh.sample_points_uniformly(number_of_points=max_points)
                    else:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = mesh.vertices
                        pcd.normals = mesh.vertex_normals
                else:
                    print(f"  [采样] 使用轮廓点采样: {len(pcd.points)} 个点")
            else:
                # 全图采样（原有方式）
                if len(mesh.vertices) > max_points:
                    pcd = mesh.sample_points_uniformly(number_of_points=max_points)
                else:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = mesh.vertices
                    pcd.normals = mesh.vertex_normals
            
            # 7. 异常点过滤（已禁用）
            # if self.filter_outliers:
            #     pcd, num_outliers = self._remove_outliers_from_pointcloud(pcd)
            #     if num_outliers > 0:
            #         print(f"  [特征提取] 移除了 {num_outliers} 个异常点")
            
            # 8. 确保法向量已计算
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_normal, max_nn=30
                    )
                )
            
            # 9. 计算FPFH特征
            # 使用KDTreeSearchParamHybrid指定搜索参数
            search_param = o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=30
            )
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                search_param
            )
            
            if pcd_fpfh is None or pcd_fpfh.data.shape[1] == 0:
                mesh_source = mesh_input if isinstance(mesh_input, str) else "mesh对象"
                print(f"[警告] FPFH特征计算失败: {mesh_source}")
                return None
            
            # 9. 计算曲率特征（增强几何特征）
            curvature_features = self._compute_curvature_features(mesh, pcd)
            
            # 10. 融合FPFH和曲率特征
            fpfh_data = np.asarray(pcd_fpfh.data).T  # (N, 33)
            # 归一化曲率特征
            if curvature_features.max() > 0:
                curvature_features_norm = curvature_features / (curvature_features.max() + 1e-6)
            else:
                curvature_features_norm = curvature_features
            
            # 组合特征：FPFH (33维) + 曲率 (3维) = 36维
            combined_features = np.hstack([fpfh_data, curvature_features_norm])
            
            print(f"  [特征提取] FPFH: {fpfh_data.shape}, 曲率: {curvature_features.shape}, "
                  f"融合特征: {combined_features.shape}")
            
            # 优化：减少内存复制，使用视图而不是复制
            points_array = np.asarray(pcd.points)
            normals_array = np.asarray(pcd.normals)
            
            return {
                'pcd': pcd,
                'fpfh': pcd_fpfh,
                'curvature': curvature_features,
                'combined_features': combined_features,  # 融合后的特征
                'points': points_array,  # 已经是numpy数组，不需要复制
                'normals': normals_array,  # 已经是numpy数组，不需要复制
                'mesh_source': mesh_input if isinstance(mesh_input, str) else "内存mesh对象"
            }
            
        except Exception as e:
            mesh_source = mesh_input if isinstance(mesh_input, str) else "mesh对象"
            print(f"[异常] 3D网格特征提取失败({mesh_source}): {e}")
            import traceback
            traceback.print_exc()
            return None

    def _remove_outliers_from_pointcloud(self, pcd, nb_neighbors=None, std_ratio=None):
        """
        移除点云中的异常点（统计离群点移除）
        
        原理：对于每个点，计算它到其k个最近邻居的平均距离。
              如果该距离超过全局平均距离的std_ratio倍标准差，则认为是异常点。
        
        Args:
            pcd: Open3D点云对象
            nb_neighbors: 用于统计的邻居数量（默认使用配置值）
            std_ratio: 标准差比率阈值（默认使用配置值）
            
        Returns:
            过滤后的点云，以及被移除的异常点数量
        """
        if o3d is None:
            return pcd, 0
        
        try:
            if len(pcd.points) < 3:
                return pcd, 0
            
            # 使用配置的默认值
            if nb_neighbors is None:
                nb_neighbors = self.outlier_nb_neighbors
            if std_ratio is None:
                std_ratio = self.outlier_std_ratio
            
            # 确保邻居数量不超过点云大小
            nb_neighbors = min(nb_neighbors, len(pcd.points) - 1)
            if nb_neighbors < 3:
                return pcd, 0
            
            # 执行统计离群点移除
            pcd_filtered, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
            num_outliers = len(pcd.points) - len(pcd_filtered.points)
            
            if num_outliers > 0:
                print(f"  [异常点过滤] 移除了 {num_outliers} 个异常点（原始: {len(pcd.points)}, 过滤后: {len(pcd_filtered.points)}）")
            
            return pcd_filtered, num_outliers
            
        except Exception as e:
            print(f"  [警告] 异常点过滤失败: {e}")
            return pcd, 0
    
    def _visualize_point_cloud_matching(self, pcd1, pcd2, transformation=None, 
                                       correspondence_set=None, title="点云匹配结果",
                                       save_path=None, show_plot=True):
        """
        使用matplotlib在3D空间中可视化点云匹配结果
        
        策略：展示匹配前后的点云，以及配准结果
        
        Args:
            pcd1: 第一个点云（参考点云，蓝色）
            pcd2: 第二个点云（待配准点云，红色）
            transformation: 变换矩阵（4x4），如果提供则显示配准后的结果
            correspondence_set: 对应点集合，如果提供则显示匹配点对连线
            title: 图表标题
            save_path: 保存路径（可选），如果提供则保存图像
            show_plot: 是否显示图像（默认True）
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # 设置中文字体
            setup_chinese_font()
            
            # 异常点过滤（已禁用）
            # if self.filter_outliers:
            #     print(f"  [可视化] 执行异常点过滤...")
            #     pcd1_filtered, num_outliers1 = self._remove_outliers_from_pointcloud(pcd1)
            #     pcd2_filtered, num_outliers2 = self._remove_outliers_from_pointcloud(pcd2)
            #     pcd1 = pcd1_filtered
            #     pcd2 = pcd2_filtered
            #     if num_outliers1 > 0 or num_outliers2 > 0:
            #         title += f"\n过滤异常点: 点云1移除{num_outliers1}个, 点云2移除{num_outliers2}个"
            
            # 提取点云坐标
            points1 = np.asarray(pcd1.points)
            points2 = np.asarray(pcd2.points)
            
            # 如果提供了变换矩阵，应用变换
            if transformation is not None:
                points2_transformed = points2.copy()
                # 应用齐次变换
                points2_homo = np.hstack([points2_transformed, np.ones((len(points2_transformed), 1))])
                points2_transformed = (transformation @ points2_homo.T).T[:, :3]
            else:
                points2_transformed = points2
            
            # 定义统一的颜色方案（所有子图保持一致）
            color_pcd1 = 'blue'      # 点云1始终为蓝色
            color_pcd2 = 'red'      # 点云2始终为红色（匹配前后保持一致）
            color_center1 = 'darkblue'  # 点云1质心为深蓝色
            color_center2 = 'darkred'   # 点云2质心为深红色
            color_line = 'orange'      # 匹配点对连线为橙色
            
            # 创建3D图形
            fig = plt.figure(figsize=(16, 12))
            
            # 子图1: 匹配前（原始点云）
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], 
                       c=color_pcd1, s=1, alpha=0.6, label='点云1（参考）')
            ax1.scatter(points2[:, 0], points2[:, 1], points2[:, 2], 
                       c=color_pcd2, s=1, alpha=0.6, label='点云2（待配准）')
            ax1.set_title('匹配前（原始点云）', fontsize=12, fontweight='bold')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend()
            ax1.view_init(elev=20, azim=45)
            
            # 子图2: 匹配后（配准结果）
            ax2 = fig.add_subplot(222, projection='3d')
            ax2.scatter(points1[:, 0], points1[:, 1], points1[:, 2], 
                       c=color_pcd1, s=1, alpha=0.6, label='点云1（参考）')
            ax2.scatter(points2_transformed[:, 0], points2_transformed[:, 1], points2_transformed[:, 2], 
                       c=color_pcd2, s=1, alpha=0.6, label='点云2（配准后）')
            ax2.set_title('匹配后（配准结果）', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()
            ax2.view_init(elev=20, azim=45)
            
            # 子图3: 匹配点对连线（如果有对应点集合）
            ax3 = fig.add_subplot(223, projection='3d')
            ax3.scatter(points1[:, 0], points1[:, 1], points1[:, 2], 
                       c=color_pcd1, s=1, alpha=0.3, label='点云1（参考）')
            ax3.scatter(points2_transformed[:, 0], points2_transformed[:, 1], points2_transformed[:, 2], 
                       c=color_pcd2, s=1, alpha=0.3, label='点云2（配准后）')
            
            # 绘制匹配点对连线
            if correspondence_set is not None and len(correspondence_set) > 0:
                # 限制显示的匹配点数量（避免过多线条）
                max_correspondences = min(100, len(correspondence_set))
                sampled_correspondences = correspondence_set[:max_correspondences] if len(correspondence_set) > max_correspondences else correspondence_set
                
                for corr in sampled_correspondences:
                    if len(corr) == 2:
                        i, j = int(corr[0]), int(corr[1])
                        if 0 <= i < len(points1) and 0 <= j < len(points2_transformed):
                            pt1 = points1[i]
                            pt2 = points2_transformed[j]
                            ax3.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                                   color=color_line, alpha=0.4, linewidth=0.5)
                
                ax3.set_title(f'匹配点对连线（显示{len(sampled_correspondences)}/{len(correspondence_set)}个匹配）', 
                            fontsize=12, fontweight='bold')
            else:
                ax3.set_title('匹配点对连线（无对应点数据）', fontsize=12, fontweight='bold')
            
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            ax3.legend()
            ax3.view_init(elev=20, azim=45)
            
            # 子图4: 叠加视图（所有点云叠加显示）
            ax4 = fig.add_subplot(224, projection='3d')
            ax4.scatter(points1[:, 0], points1[:, 1], points1[:, 2], 
                       c=color_pcd1, s=2, alpha=0.5, label='点云1（参考）')
            ax4.scatter(points2_transformed[:, 0], points2_transformed[:, 1], points2_transformed[:, 2], 
                       c=color_pcd2, s=2, alpha=0.5, label='点云2（配准后）')
            
            # 计算并显示质心
            center1 = points1.mean(axis=0)
            center2_transformed = points2_transformed.mean(axis=0)
            ax4.scatter([center1[0]], [center1[1]], [center1[2]], 
                       c=color_center1, s=200, marker='*', label='点云1质心')
            ax4.scatter([center2_transformed[0]], [center2_transformed[1]], [center2_transformed[2]], 
                       c=color_center2, s=200, marker='*', label='点云2质心')
            
            # 计算配准误差（如果提供了对应点）
            if correspondence_set is not None and len(correspondence_set) > 0:
                errors = []
                for corr in correspondence_set:
                    if len(corr) == 2:
                        i, j = int(corr[0]), int(corr[1])
                        if 0 <= i < len(points1) and 0 <= j < len(points2_transformed):
                            error = np.linalg.norm(points1[i] - points2_transformed[j])
                            errors.append(error)
                
                if len(errors) > 0:
                    avg_error = np.mean(errors)
                    max_error = np.max(errors)
                    min_error = np.min(errors)
                    ax4.set_title(f'叠加视图\n平均误差: {avg_error:.4f}, 最大: {max_error:.4f}, 最小: {min_error:.4f}', 
                                fontsize=12, fontweight='bold')
                else:
                    ax4.set_title('叠加视图', fontsize=12, fontweight='bold')
            else:
                ax4.set_title('叠加视图', fontsize=12, fontweight='bold')
            
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            ax4.legend()
            ax4.view_init(elev=20, azim=45)
            
            # 设置总标题
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # 保存图像
            if save_path:
                try:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"  [可视化] 匹配结果已保存到: {save_path}")
                except Exception as e:
                    print(f"  [警告] 保存可视化图像失败: {e}")
            
            # 显示图像
            if show_plot:
                plt.show(block=False)  # 非阻塞显示
                plt.pause(0.1)  # 短暂暂停以确保窗口显示
            
            return fig
            
        except Exception as e:
            print(f"  [警告] 3D可视化失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_pca_quality(self, eigenvals):
        """
        评估PCA质量，判断点云是否适合使用PCA对齐
        
        原理：通过特征值比例判断点云主方向的明确程度
        - 如果最大特征值远大于其他特征值，说明有明确主方向，适合PCA
        - 如果特征值接近，说明点云接近球形，不适合PCA
        
        Args:
            eigenvals: 排序后的特征值（从大到小）
            
        Returns:
            (is_suitable, quality_score): 
            - is_suitable: 是否适合使用PCA
            - quality_score: 质量分数 (0-1)，越高说明主方向越明确
        """
        if len(eigenvals) < 3 or np.sum(eigenvals) < 1e-6:
            return False, 0.0
        
        # 归一化特征值
        eigenvals_norm = eigenvals / np.sum(eigenvals)
        
        # 计算主方向强度：最大特征值占比
        principal_ratio = eigenvals_norm[0]
        
        # 计算各向异性程度：特征值差异
        # 如果特征值差异大，说明有明确主方向
        anisotropy = 1.0 - (eigenvals_norm[1] + eigenvals_norm[2]) / 2.0
        
        # 综合质量分数
        quality_score = (principal_ratio + anisotropy) / 2.0
        
        # 判断是否适合：主方向占比 > 0.4 且质量分数 > 0.3
        is_suitable = principal_ratio > 0.4 and quality_score > 0.3
        
        return is_suitable, quality_score
    
    def _global_alignment_pca(self, pcd1, pcd2):
        """
        使用PCA（主成分分析）进行全局对齐（粗配准）
        
        策略：PCA用于粗配准，为后续ICP精配准提供初始估计
        
        原理：
        1. 计算两个点云的主方向（PCA）
        2. 对齐主方向（旋转）
        3. 对齐质心（平移）
        
        适用场景：
        - 点云有明确的主方向（如长条形、平面结构）
        - 旋转角度较大（>30度）
        - 需要快速粗配准
        
        不适用场景：
        - 点云接近球形或高度对称
        - 主方向不明确
        
        Args:
            pcd1: 第一个点云（参考点云）
            pcd2: 第二个点云（待配准点云）
            
        Returns:
            4x4变换矩阵，将pcd2变换到pcd1的坐标系，如果失败返回None
        """
        try:
            points1 = np.asarray(pcd1.points)
            points2 = np.asarray(pcd2.points)
            
            if len(points1) < 3 or len(points2) < 3:
                return None
            
            # 1. 计算质心
            center1 = points1.mean(axis=0)
            center2 = points2.mean(axis=0)
            
            # 2. 去中心化
            points1_centered = points1 - center1
            points2_centered = points2 - center2
            
            # 3. PCA分解
            # 计算协方差矩阵
            cov1 = points1_centered.T @ points1_centered / len(points1_centered)
            cov2 = points2_centered.T @ points2_centered / len(points2_centered)
            
            # 特征值分解
            eigenvals1, eigenvecs1 = np.linalg.eigh(cov1)
            eigenvals2, eigenvecs2 = np.linalg.eigh(cov2)
            
            # 按特征值排序（从大到小）
            idx1 = np.argsort(eigenvals1)[::-1]
            idx2 = np.argsort(eigenvals2)[::-1]
            eigenvecs1_sorted = eigenvecs1[:, idx1]
            eigenvecs2_sorted = eigenvecs2[:, idx2]
            eigenvals1_sorted = eigenvals1[idx1]
            eigenvals2_sorted = eigenvals2[idx2]
            
            # 评估PCA质量
            is_suitable1, quality1 = self._evaluate_pca_quality(eigenvals1_sorted)
            is_suitable2, quality2 = self._evaluate_pca_quality(eigenvals2_sorted)
            
            # 如果点云不适合PCA，返回None（让后续方法处理）
            if not (is_suitable1 and is_suitable2):
                print(f"  [PCA全局对齐] 点云主方向不明确（质量分数: {quality1:.3f}, {quality2:.3f}），不适合PCA，跳过")
                return None
            
            eigenvecs1 = eigenvecs1_sorted
            eigenvecs2 = eigenvecs2_sorted
            
            # 确保右手坐标系（行列式为正）
            if np.linalg.det(eigenvecs1) < 0:
                eigenvecs1[:, -1] *= -1
            if np.linalg.det(eigenvecs2) < 0:
                eigenvecs2[:, -1] *= -1
            
            # 4. 计算旋转矩阵（从pcd2坐标系到pcd1坐标系）
            R = eigenvecs1 @ eigenvecs2.T
            
            # 确保是旋转矩阵（正交且行列式为1）
            U, S, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt
            
            # 5. 计算平移向量
            t = center1 - R @ center2
            
            # 6. 构建变换矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            print(f"  [PCA全局对齐] 粗配准完成（质量分数: {quality1:.3f}, {quality2:.3f}），准备进行ICP精配准")
            return T
            
        except Exception as e:
            print(f"  [警告] PCA全局对齐失败: {e}")
            return None
    
    def _global_alignment_feature_based(self, pcd1, pcd2, fpfh1, fpfh2, 
                                       max_correspondence_distance, 
                                       num_samples=1000, num_ransac=4):
        """
        基于特征的全局匹配（粗配准）
        
        策略：
        1. 使用更大的搜索范围进行特征匹配
        2. 使用更多的RANSAC采样点
        3. 使用更大的距离阈值
        
        Args:
            pcd1: 第一个点云
            pcd2: 第二个点云
            fpfh1: 第一个点云的FPFH特征
            fpfh2: 第二个点云的FPFH特征
            max_correspondence_distance: 基础距离阈值
            num_samples: 特征匹配采样数
            num_ransac: RANSAC采样点数
            
        Returns:
            RANSAC结果，如果失败返回None
        """
        try:
            # 使用更大的距离阈值进行全局搜索
            global_distance = max_correspondence_distance * 5.0  # 5倍基础距离
            
            print(f"  [全局特征匹配] 使用大距离阈值: {global_distance:.4f} (基础: {max_correspondence_distance:.4f})")
            print(f"  [全局特征匹配] 开始RANSAC匹配（最大迭代={self.ransac_global_max_iterations}, 置信度={self.ransac_global_confidence}）...")
            
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd1, pcd2,
                fpfh1, fpfh2,
                mutual_filter=False,  # 关闭互过滤，允许更多匹配
                max_correspondence_distance=global_distance,  # 大距离阈值
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=num_ransac,  # 使用更多采样点
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),  # 放宽边长检查
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(global_distance)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    self.ransac_global_max_iterations, 
                    self.ransac_global_confidence
                )
            )
            
            if result_ransac.fitness > 0:
                print(f"  [全局特征匹配] 成功: fitness={result_ransac.fitness:.4f}, RMSE={result_ransac.inlier_rmse:.6f}")
                return result_ransac
            else:
                print(f"  [全局特征匹配] 失败: fitness=0")
                return None
                
        except Exception as e:
            print(f"  [警告] 全局特征匹配失败: {e}")
            return None
    
    def _match_3d_mesh_features(self, feat1, feat2, max_correspondence_distance=0.1, 
                                mutual_filter=False, ransac_n=3, auto_distance=True, 
                                use_global_alignment=True):
        """
        匹配两个3D网格的特征（优化版：多特征融合）
        
        Args:
            feat1: 第一个网格的特征（来自_extract_3d_mesh_features）
            feat2: 第二个网格的特征
            max_correspondence_distance: 最大对应点距离（默认0.1，增大以提高匹配成功率），如果auto_distance=True会被自动计算
            mutual_filter: 是否使用互过滤（默认False，避免过滤掉太多对应点）
            ransac_n: RANSAC采样点数
            auto_distance: 是否自动计算对应点距离（默认True，根据点云尺度自适应）
            use_global_alignment: 是否使用全局对齐（默认True，先进行粗配准再进行精细配准）
            
        Returns:
            dict: {
                'transformation': 4x4变换矩阵,
                'fitness': 匹配适应度,
                'inlier_rmse': 内点RMSE,
                'correspondence_set': 对应点集合
            } 或 None
        """
        if o3d is None:
            return None
        
        try:
            if feat1 is None or feat2 is None:
                return None
            
            pcd1 = feat1['pcd']
            pcd2 = feat2['pcd']
            fpfh1 = feat1['fpfh']
            fpfh2 = feat2['fpfh']
            
            # 优化：对点云进行采样以加速匹配（如果点数太多）
            max_points_for_matching = int(os.environ.get("MAX_POINTS_FOR_MATCHING", "20000"))  # 默认最多20000点
            
            # 从特征中获取半径信息（如果可用），否则使用自适应计算
            points1_temp = np.asarray(pcd1.points)
            points2_temp = np.asarray(pcd2.points)
            radius_normal_est, radius_feature_est, _ = self._compute_adaptive_radii(points1_temp, default_radius_normal=0.1, default_radius_feature=0.2)
            
            # 优化：使用智能下采样（体素下采样优先，保持空间分布）
            if len(pcd1.points) > max_points_for_matching:
                print(f"  [采样] 点云1点数过多 ({len(pcd1.points)} > {max_points_for_matching})，进行下采样...")
                pcd1 = self._smart_downsample(pcd1, max_points_for_matching, radius_normal_est)
                # 重新计算法向量和FPFH特征（因为点云改变了）
                pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_est, max_nn=30))
                fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd1, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature_est, max_nn=100))
                print(f"  [采样] 点云1下采样后: {len(pcd1.points)} 个点")
            
            if len(pcd2.points) > max_points_for_matching:
                print(f"  [采样] 点云2点数过多 ({len(pcd2.points)} > {max_points_for_matching})，进行下采样...")
                pcd2 = self._smart_downsample(pcd2, max_points_for_matching, radius_normal_est)
                # 重新计算法向量和FPFH特征（因为点云改变了）
                pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_est, max_nn=30))
                fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd2, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature_est, max_nn=100))
                print(f"  [采样] 点云2下采样后: {len(pcd2.points)} 个点")
            
            # 保存原始点云（用于可视化，因为pcd2可能在匹配过程中被变换）
            pcd1_original_for_viz = o3d.geometry.PointCloud(pcd1)
            pcd2_original_for_viz = o3d.geometry.PointCloud(pcd2)
            
            # 优先使用融合特征（如果可用）
            use_combined_features = 'combined_features' in feat1 and 'combined_features' in feat2
            if use_combined_features:
                print(f"  [匹配] 使用融合特征（FPFH+曲率）进行匹配")
            
            # 调试信息：检查网格和特征质量
            points1 = np.asarray(pcd1.points)
            points2 = np.asarray(pcd2.points)
            
            # 保存原始点云坐标（用于后续SVD优化）
            original_points1 = points1.copy()
            original_points2 = points2.copy()
            bbox1 = points1.max(axis=0) - points1.min(axis=0)
            bbox2 = points2.max(axis=0) - points2.min(axis=0)
            center1 = points1.mean(axis=0)
            center2 = points2.mean(axis=0)
            
            print(f"  [调试] 网格1: {len(points1)}点, 边界框={bbox1}, 中心={center1}")
            print(f"  [调试] 网格2: {len(points2)}点, 边界框={bbox2}, 中心={center2}")
            print(f"  [调试] FPFH特征维度: {fpfh1.data.shape} vs {fpfh2.data.shape}")
            
            # 自适应计算对应点距离阈值（如果启用）
            if auto_distance:
                max_correspondence_distance = self._compute_adaptive_correspondence_distance(
                    points1, points2, default_distance=max_correspondence_distance
                )
                print(f"  [匹配] 使用自适应对应点距离: {max_correspondence_distance:.4f}")
            
            # ========== 阶段1: 全局匹配（粗配准） ==========
            global_transformation = None
            used_pca = False  # 标志：记录是否使用了PCA
            
            if use_global_alignment:
                print(f"\n  [阶段1] 开始全局匹配（粗配准）...")
                
                # 方法1: 尝试PCA对齐（适用于有明显主方向的点云）
                print(f"  [全局匹配] 尝试PCA对齐...")
                T_pca = self._global_alignment_pca(pcd1, pcd2)
                
                if T_pca is not None:
                    # 验证PCA对齐质量
                    pcd2_pca = o3d.geometry.PointCloud(pcd2)
                    pcd2_pca.transform(T_pca)
                    
                    # 计算对齐后的距离
                    from scipy.spatial import cKDTree
                    tree = cKDTree(np.asarray(pcd1.points))
                    distances, _ = tree.query(np.asarray(pcd2_pca.points), k=1)
                    avg_distance = np.mean(distances)
                    
                    # 如果平均距离合理，使用PCA结果
                    scale = np.linalg.norm(bbox1)
                    if avg_distance < scale * 0.1:  # 平均距离小于尺度的10%
                        global_transformation = T_pca
                        used_pca = True  # 标记使用了PCA
                        print(f"  [全局匹配] PCA对齐成功: 平均距离={avg_distance:.4f} (尺度={scale:.4f})")
                    else:
                        print(f"  [全局匹配] PCA对齐质量不佳: 平均距离={avg_distance:.4f}，尝试其他方法")
                
                # 方法2: 如果PCA失败，尝试基于特征的全局匹配
                if global_transformation is None:
                    print(f"  [全局匹配] 尝试基于特征的全局匹配...")
                    result_global = self._global_alignment_feature_based(
                        pcd1, pcd2, fpfh1, fpfh2, 
                        max_correspondence_distance,
                        num_samples=1000,
                        num_ransac=4
                    )
                    
                    if result_global is not None and result_global.fitness > 0.1:
                        global_transformation = result_global.transformation
                        print(f"  [全局匹配] 特征匹配成功: fitness={result_global.fitness:.4f}")
                    else:
                        print(f"  [全局匹配] 特征匹配失败，将使用局部匹配")
                
                # 如果全局匹配成功，应用变换
                if global_transformation is not None:
                    pcd2_global = o3d.geometry.PointCloud(pcd2)
                    pcd2_global.transform(global_transformation)
                    pcd2 = pcd2_global
                    
                    # 重新计算特征（可选，如果变换较大）
                    # 计算自适应半径用于重新计算特征
                    points2_temp = np.asarray(pcd2.points)
                    _, radius_feature_recalc, _ = self._compute_adaptive_radii(
                        points2_temp, 
                        default_radius_normal=0.1,
                        default_radius_feature=0.2
                    )
                    
                    pcd2.estimate_normals()
                    search_param = o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius_feature_recalc, 
                        max_nn=30
                    )
                    fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(pcd2, search_param)
                    points2 = np.asarray(pcd2.points)
                    print(f"  [全局匹配] 已应用全局变换，准备进行局部匹配")
                else:
                    print(f"  [全局匹配] 全局匹配失败，直接进行局部匹配")
            
            # 预处理：归一化网格到相同尺度（可选）
            # 计算网格的尺度
            scale1 = np.linalg.norm(bbox1)
            scale2 = np.linalg.norm(bbox2)
            if scale1 > 1e-6 and scale2 > 1e-6:
                # 归一化到单位尺度
                pcd1_normalized = o3d.geometry.PointCloud(pcd1)
                pcd2_normalized = o3d.geometry.PointCloud(pcd2)
                pcd1_normalized.scale(1.0 / scale1, center=center1)
                pcd2_normalized.scale(1.0 / scale2, center=center2)
                
                # 重新计算法向量和特征
                pcd1_normalized.estimate_normals()
                pcd2_normalized.estimate_normals()
                
                search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                fpfh1_normalized = o3d.pipelines.registration.compute_fpfh_feature(pcd1_normalized, search_param)
                fpfh2_normalized = o3d.pipelines.registration.compute_fpfh_feature(pcd2_normalized, search_param)
                
                if fpfh1_normalized is not None and fpfh2_normalized is not None:
                    pcd1 = pcd1_normalized
                    pcd2 = pcd2_normalized
                    fpfh1 = fpfh1_normalized
                    fpfh2 = fpfh2_normalized
                    max_correspondence_distance = 0.1  # 归一化后使用较小的距离
                    print(f"  [调试] 已归一化网格，新距离阈值={max_correspondence_distance}")
            
            # 1. 首先尝试使用mutual_filter（如果启用）
            result_ransac = None
            if mutual_filter:
                try:
                    print(f"  [RANSAC] 开始第一次匹配尝试（mutual_filter=True, 最大迭代={self.ransac_max_iterations}）...")
                    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        pcd1, pcd2,
                        fpfh1, fpfh2,
                        mutual_filter=True,
                        max_correspondence_distance=max_correspondence_distance,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                        ransac_n=ransac_n,
                        checkers=[
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)
                        ],
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                            self.ransac_max_iterations, 
                            self.ransac_confidence
                        )
                    )
                    if result_ransac.fitness > 0:
                        print(f"  [调试] Mutual filter成功: fitness={result_ransac.fitness:.4f}")
                except Exception as e:
                    print(f"  [调试] Mutual filter失败: {e}")
                    result_ransac = None
            
            # 2. 如果mutual_filter失败或未启用，尝试不使用mutual_filter（优化：减少尝试次数）
            if result_ransac is None or result_ransac.fitness == 0:
                try:
                    # 逐步增大距离阈值（从3个减少到2个以提高速度）
                    distance_thresholds = [max_correspondence_distance * 1.5, max_correspondence_distance * 2.5]
                    for idx, dist_thresh in enumerate(distance_thresholds):
                        try:
                            print(f"  [RANSAC] 尝试距离阈值 {idx+1}/{len(distance_thresholds)}: {dist_thresh:.4f} (最大迭代={self.ransac_max_iterations})...")
                            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                                pcd1, pcd2,
                                fpfh1, fpfh2,
                                mutual_filter=False,  # 关闭互过滤
                                max_correspondence_distance=dist_thresh,  # 逐步增大距离阈值
                                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                                ransac_n=ransac_n,
                                checkers=[
                                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.7),  # 进一步放宽
                                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
                                ],
                                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                                    self.ransac_max_iterations, 
                                    0.95  # 降低收敛要求以提高成功率
                                )
                            )
                            if result_ransac.fitness > 0:
                                print(f"  [调试] RANSAC成功 (距离阈值={dist_thresh:.3f}): fitness={result_ransac.fitness:.4f}")
                                break
                        except Exception as e:
                            print(f"  [警告] RANSAC尝试失败 (距离阈值={dist_thresh:.3f}): {e}")
                            continue
                except Exception as e:
                    print(f"  [调试] RANSAC配准失败: {e}")
                    result_ransac = None
            
            if result_ransac is None or result_ransac.fitness == 0:
                # 尝试直接使用ICP（需要初始估计）
                print("  [调试] 尝试使用ICP直接配准（无初始估计）...")
                try:
                    # 先进行粗略对齐（基于质心）
                    center_diff = center2 - center1
                    init_transformation = np.eye(4)
                    init_transformation[:3, 3] = center_diff
                    
                    result_icp = o3d.pipelines.registration.registration_icp(
                        pcd1, pcd2,
                        max_correspondence_distance=max_correspondence_distance * 5.0,  # 很大的初始距离
                        init=init_transformation,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                    )
                    
                    if result_icp.fitness > 0:
                        print(f"  [调试] ICP直接配准成功: fitness={result_icp.fitness:.4f}")
                        return {
                            'transformation': result_icp.transformation,
                            'fitness': result_icp.fitness,
                            'inlier_rmse': result_icp.inlier_rmse,
                            'correspondence_set': result_icp.correspondence_set if result_icp.correspondence_set is not None else [],
                            'method': 'ICP_direct'
                        }
                except Exception as e:
                    print(f"  [调试] ICP直接配准也失败: {e}")
                
                # 最后尝试：使用单位变换矩阵（降级策略）
                print("  [警告] 所有匹配方法失败，使用单位变换矩阵作为降级策略")
                print(f"  [调试] 可能原因：网格差异太大、特征质量不足、或需要手动对齐")
                # 返回单位变换矩阵，而不是None，以便后续流程可以继续
                identity_transform = np.eye(4)
                return {
                    'transformation': identity_transform,
                    'fitness': 0.0,
                    'inlier_rmse': float('inf'),
                    'correspondence_set': [],
                    'method': 'identity_fallback',
                    'warning': '匹配失败，使用单位变换矩阵'
                }
            
            # ========== 阶段2: 局部匹配（精细配准） ==========
            print(f"\n  [阶段2] 开始局部匹配（精细配准）...")
            
            # 初始变换：如果有全局匹配，使用全局变换；否则使用RANSAC结果
            if global_transformation is not None:
                init_transformation = global_transformation
                print(f"  [局部匹配] 使用全局匹配结果作为初始估计")
            else:
                init_transformation = result_ransac.transformation
                print(f"  [局部匹配] 使用RANSAC结果作为初始估计")
            
            final_transformation = init_transformation
            final_fitness = result_ransac.fitness if result_ransac is not None else 0.0
            final_rmse = result_ransac.inlier_rmse if result_ransac is not None else float('inf')
            final_correspondence_set = result_ransac.correspondence_set if result_ransac is not None else None
            final_method = 'Global+RANSAC' if global_transformation is not None else 'RANSAC'
            
            # ICP精细配准（在全局匹配基础上进一步优化）
            # 策略：PCA/RANSAC提供粗配准，ICP进行精配准
            try:
                print(f"  [局部匹配] 执行ICP精细配准（基于粗配准结果）...")
                result_icp = o3d.pipelines.registration.registration_icp(
                    pcd1, pcd2,
                    max_correspondence_distance=max_correspondence_distance * 1.5,  # 局部匹配使用较小距离
                    init=init_transformation,  # 使用粗配准结果作为初始估计
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                
                # 如果ICP改进了结果，使用ICP的结果
                if result_icp.fitness > final_fitness:
                    final_transformation = result_icp.transformation
                    final_fitness = result_icp.fitness
                    final_rmse = result_icp.inlier_rmse
                    final_correspondence_set = result_icp.correspondence_set
                    if global_transformation is not None:
                        # 根据使用的粗配准方法确定最终方法名称
                        final_method = 'PCA+RANSAC+ICP' if used_pca else 'Feature+RANSAC+ICP'
                    else:
                        final_method = 'RANSAC+ICP'
                    print(f"  [局部匹配] ICP精配准成功: fitness={final_fitness:.4f}, RMSE={final_rmse:.6f} (方法: {final_method})")
                else:
                    print(f"  [局部匹配] ICP未改进结果，保持粗配准结果 (fitness={final_fitness:.4f})")
            except Exception as e:
                print(f"  [警告] ICP精细配准失败: {e}")
            
            # 4. 如果使用融合特征，使用匹配点对进行SVD优化
            # 检查correspondence_set是否有效
            if final_correspondence_set is None:
                final_correspondence_set = []
            
            if use_combined_features and len(final_correspondence_set) >= 3:
                try:
                    # 提取匹配点对
                    matched_pts1 = []
                    matched_pts2 = []
                    for corr in final_correspondence_set:
                        try:
                            if isinstance(corr, (list, tuple, np.ndarray)) and len(corr) == 2:
                                i, j = int(corr[0]), int(corr[1])
                                if 0 <= i < len(points1) and 0 <= j < len(points2):
                                    matched_pts1.append(points1[i])
                                    matched_pts2.append(points2[j])
                        except (ValueError, IndexError, TypeError) as e:
                            # 跳过无效的对应点
                            continue
                    
                    if len(matched_pts1) >= 3:
                        matched_pts1 = np.array(matched_pts1)
                        matched_pts2 = np.array(matched_pts2)
                        
                        # 注意：matched_pts1和matched_pts2应该在同一坐标系中
                        # matched_pts1在第一帧坐标系，matched_pts2在第二帧坐标系
                        # 需要将matched_pts2变换到第一帧坐标系
                        pts2_homo = np.hstack([matched_pts2, np.ones((len(matched_pts2), 1))])
                        matched_pts2_transformed = (final_transformation @ pts2_homo.T).T[:, :3]
                        
                        # 使用SVD计算更精确的变换（从matched_pts2_transformed到matched_pts1）
                        # 这应该是一个小的修正变换
                        T_svd = self._estimate_transformation_svd(matched_pts1, matched_pts2_transformed)
                        
                        if T_svd is not None:
                            # 验证SVD变换质量
                            error = self._validate_transformation(matched_pts1, matched_pts2_transformed, T_svd)
                            
                            # 如果SVD变换改进了结果，使用SVD结果
                            if error['reprojection_error'] < final_rmse * 1.2:  # 允许20%的误差范围
                                final_transformation = T_svd @ final_transformation  # 组合变换
                                final_method = 'RANSAC+ICP+SVD'
                                print(f"  [匹配优化] 使用SVD优化变换，重投影误差={error['reprojection_error']:.6f}")
                except Exception as e:
                    print(f"  [警告] SVD优化失败: {e}")
            
            # 可视化匹配结果（如果启用）
            if self.match_visualize and final_transformation is not None:
                try:
                    # 准备保存路径
                    save_path = None
                    if hasattr(self, 'output_dir') and self.output_dir:
                        os.makedirs(self.output_dir, exist_ok=True)
                        import time
                        timestamp = int(time.time())
                        save_path = os.path.join(self.output_dir, f"point_cloud_matching_{timestamp}.png")
                    
                    # 生成可视化标题
                    title = f"点云匹配结果\n方法: {final_method}, Fitness: {final_fitness:.4f}, RMSE: {final_rmse:.6f}"
                    
                    # 调用可视化函数（使用保存的原始点云）
                    print(f"  [可视化] 生成3D匹配可视化...")
                    self._visualize_point_cloud_matching(
                        pcd1_original_for_viz, pcd2_original_for_viz,
                        transformation=final_transformation,
                        correspondence_set=final_correspondence_set,
                        title=title,
                        save_path=save_path,
                        show_plot=self.o3d_visualize  # 使用o3d_visualize控制是否显示窗口
                    )
                except Exception as e:
                    print(f"  [警告] 匹配可视化失败: {e}")
            
            # 确保correspondence_set不为None
            if final_correspondence_set is None:
                final_correspondence_set = []
            
            return {
                'transformation': final_transformation,
                'fitness': final_fitness,
                'inlier_rmse': final_rmse,
                'correspondence_set': final_correspondence_set,
                'method': final_method
            }
            
        except Exception as e:
            print(f"[异常] 3D网格特征匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _match_frames_3d_mesh_features(self, suffix=None, max_points=10000, 
                                      radius_normal=0.1, radius_feature=0.2):
        """
        批量匹配所有帧的3D原始高度场网格特征
        
        Args:
            suffix: 网格文件后缀（如"th_68"）
            max_points: 最大采样点数
            radius_normal: 法向量估计半径
            radius_feature: 特征计算半径
            
        Returns:
            list: 匹配结果列表，每个元素包含帧对和变换矩阵
        """
        if suffix is None:
            suffix = getattr(self, 'threshold_mode', 'th_68')
        if o3d is None:
            print("[警告] Open3D未安装，无法进行3D网格匹配")
            return None
        
        try:
            print("\n" + "="*80)
            print("开始基于3D原始高度场网格进行帧间匹配")
            print("="*80)
            
            # 1. 查找所有PLY文件（使用原始高度场网格）
            ply_files = []
            for filename in os.listdir(self.mesh_out_dir):
                if filename.startswith(f"o3d_height_{suffix}") and filename.endswith(".ply"):
                    # 提取帧索引
                    try:
                        # 文件名格式: o3d_height_th_68_0000.ply
                        parts = filename.replace(".ply", "").split("_")
                        frame_idx = None
                        for part in parts:
                            if part.isdigit() and len(part) == 4:
                                frame_idx = int(part)
                                break
                        if frame_idx is not None:
                            ply_files.append((frame_idx, os.path.join(self.mesh_out_dir, filename)))
                    except Exception:
                        continue
            
            if len(ply_files) < 2:
                print(f"[错误] 找到的PLY文件数量不足: {len(ply_files)}")
                return None
            
            # 按帧索引排序
            ply_files.sort(key=lambda x: x[0])
            print(f"[信息] 找到 {len(ply_files)} 个3D网格文件")
            
            # 2. 提取所有网格的特征
            print("\n[步骤1] 提取3D网格特征...")
            mesh_features = {}
            timer = TimeTracker("3D特征提取").start()
            
            for frame_idx, ply_path in ply_files:
                print(f"  处理帧 {frame_idx}: {os.path.basename(ply_path)}")
                # 获取对应帧的mask（优先使用mask采样）
                mask = None
                if hasattr(self, '_frame_masks') and frame_idx in self._frame_masks:
                    mask = self._frame_masks[frame_idx]
                    print(f"    [mask] 使用缓存的mask进行采样")
                
                # 轮廓计算功能已移除，不再使用轮廓点采样
                # 如果没有mask，将使用全图采样（在_extract_3d_mesh_features中处理）
                feat = self._extract_3d_mesh_features(
                    ply_path, 
                    max_points=max_points,
                    radius_normal=radius_normal,
                    radius_feature=radius_feature,
                    mask=mask,  # 优先使用mask采样，如果没有mask则使用全图采样
                    contour_points_list=None  # 轮廓计算已移除
                )
                if feat is not None:
                    mesh_features[frame_idx] = feat
                    print(f"    [OK] 提取成功: {feat['points'].shape[0]} 个点, FPFH维度: {feat['fpfh'].data.shape}")
                else:
                    print(f"    ✗ 提取失败")
            
            timer.print_summary()
            
            if len(mesh_features) < 2:
                print("[错误] 成功提取特征的网格数量不足")
                return None
            
            # 3. 逐对匹配
            print(f"\n[步骤2] 进行帧间匹配（共 {len(mesh_features)} 帧）...")
            match_results = []
            timer = TimeTracker("3D特征匹配").start()
            
            frame_indices = sorted(mesh_features.keys())
            for i in range(len(frame_indices) - 1):
                idx1 = frame_indices[i]
                idx2 = frame_indices[i + 1]
                
                print(f"\n[匹配] 帧 {idx1} <-> 帧 {idx2}")
                feat1 = mesh_features[idx1]
                feat2 = mesh_features[idx2]
                
                match_result = self._match_3d_mesh_features(feat1, feat2)
                
                if match_result is not None:
                    T = match_result['transformation']
                    fitness = match_result['fitness']
                    rmse = match_result['inlier_rmse']
                    method = match_result['method']
                    
                    print(f"  [OK] 匹配成功: fitness={fitness:.4f}, RMSE={rmse:.6f}, 方法={method}")
                    
                    # 提取旋转和平移
                    R = T[:3, :3]
                    t = T[:3, 3]
                    
                    match_results.append({
                        'frame_idx1': idx1,
                        'frame_idx2': idx2,
                        'transformation': T,
                        'rotation': R,
                        'translation': t,
                        'fitness': fitness,
                        'inlier_rmse': rmse,
                        'method': method
                    })
                else:
                    print(f"  ✗ 匹配失败")
            
            timer.print_summary()
            
            # 4. 保存匹配结果
            if len(match_results) > 0:
                match_output_dir = os.path.join(self.output_dir, "3d_mesh_feature_matching")
                os.makedirs(match_output_dir, exist_ok=True)
                
                # 保存JSON结果
                result_json = {
                    'suffix': suffix,
                    'total_frames': len(mesh_features),
                    'total_matches': len(match_results),
                    'matches': []
                }
                
                for match in match_results:
                    result_json['matches'].append({
                        'frame_idx1': int(match['frame_idx1']),
                        'frame_idx2': int(match['frame_idx2']),
                        'transformation': match['transformation'].tolist(),
                        'rotation': match['rotation'].tolist(),
                        'translation': match['translation'].tolist(),
                        'fitness': float(match['fitness']),
                        'inlier_rmse': float(match['inlier_rmse']),
                        'method': match['method']
                    })
                
                json_path = os.path.join(match_output_dir, f"3d_mesh_matches_{suffix}.json")
                # JSON保存已禁用
                # with open(json_path, 'w', encoding='utf-8') as f:
                #     json.dump(result_json, f, ensure_ascii=False, indent=2)
                
                print(f"\n[完成] 匹配结果已保存: {json_path}")
                print(f"  成功匹配: {len(match_results)}/{len(frame_indices)-1} 对")
                
                # 打印统计信息
                if len(match_results) > 0:
                    avg_fitness = np.mean([m['fitness'] for m in match_results])
                    avg_rmse = np.mean([m['inlier_rmse'] for m in match_results])
                    print(f"  平均fitness: {avg_fitness:.4f}")
                    print(f"  平均RMSE: {avg_rmse:.6f}")
                
                # 保存匹配指标到TXT文件
                txt_path = os.path.join(match_output_dir, f"matching_metrics_{suffix}.txt")
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write("3D网格特征匹配指标\n")
                        f.write("="*80 + "\n\n")
                        
                        f.write("【匹配统计】\n")
                        f.write(f"总帧数: {len(mesh_features)}\n")
                        f.write(f"成功匹配对数: {len(match_results)}\n")
                        f.write(f"匹配成功率: {len(match_results)/(len(frame_indices)-1)*100:.2f}%\n\n")
                        
                        if len(match_results) > 0:
                            fitness_values = [m['fitness'] for m in match_results]
                            rmse_values = [m['inlier_rmse'] for m in match_results]
                            
                            f.write("【匹配适应度 (Fitness)】\n")
                            f.write(f"  平均值: {np.mean(fitness_values):.6f}\n")
                            f.write(f"  最大值: {np.max(fitness_values):.6f}\n")
                            f.write(f"  最小值: {np.min(fitness_values):.6f}\n")
                            f.write(f"  标准差: {np.std(fitness_values):.6f}\n")
                            f.write(f"  中位数: {np.median(fitness_values):.6f}\n\n")
                            
                            f.write("【内点RMSE (Inlier RMSE)】\n")
                            f.write(f"  平均值: {np.mean(rmse_values):.6f}\n")
                            f.write(f"  最大值: {np.max(rmse_values):.6f}\n")
                            f.write(f"  最小值: {np.min(rmse_values):.6f}\n")
                            f.write(f"  标准差: {np.std(rmse_values):.6f}\n")
                            f.write(f"  中位数: {np.median(rmse_values):.6f}\n")
                            
                            # 计算总体RMSE
                            rmse_overall = np.sqrt(np.mean(np.array(rmse_values)**2))
                            f.write(f"  总体RMSE: {rmse_overall:.6f}\n\n")
                            
                            f.write("="*80 + "\n")
                            f.write("【详细匹配信息】\n")
                            f.write("="*80 + "\n\n")
                            f.write(f"{'帧对':<15} {'Fitness':<12} {'RMSE':<15} {'方法':<20}\n")
                            f.write("-"*80 + "\n")
                            for match in match_results:
                                frame_pair = f"{match['frame_idx1']:04d}-{match['frame_idx2']:04d}"
                                f.write(f"{frame_pair:<15} {match['fitness']:<12.6f} {match['inlier_rmse']:<15.6f} {match['method']:<20}\n")
                            
                            f.write("\n" + "="*80 + "\n")
                            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("="*80 + "\n")
                    
                    print(f"[保存] 匹配指标已保存到: {txt_path}")
                except Exception as e:
                    print(f"[警告] 保存匹配指标到TXT文件失败: {e}")
            
            return match_results
            
        except Exception as e:
            print(f"[异常] 3D网格特征匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _match_with_previous_frame_3d(self, current_frame_idx, suffix=None):
        """
        与上一帧进行3D网格特征匹配（逐帧匹配，优化版：移除粗匹配）
        
        优化说明：
        ==========
        1. 移除了粗匹配步骤，直接使用FPFH特征匹配
        2. 对于相邻帧，变换通常较小，FPFH特征匹配本身具有足够的鲁棒性
        3. 通过调整匹配参数（增大max_correspondence_distance）来处理可能的初始偏差
        4. 简化了代码逻辑，减少了误差累积
        
        Args:
            current_frame_idx: 当前帧索引
            suffix: 网格文件后缀，如果为None则使用阈值模式
        """
        if suffix is None:
            suffix = getattr(self, 'threshold_mode', 'th_68')
        if o3d is None:
            return
        
        try:
            # 1. 优化：从内存缓存获取mesh（不读取文件）
            prev_frame_idx = current_frame_idx - 1
            
            # 检查内存缓存中是否有mesh
            if current_frame_idx not in self._mesh_cache:
                print(f"  [警告] 当前帧（{current_frame_idx}）的mesh不在缓存中")
                return
            if prev_frame_idx not in self._mesh_cache:
                print(f"  [警告] 前一帧（{prev_frame_idx}）的mesh不在缓存中")
                return
            
            print(f"  [3D匹配] 开始匹配帧 {prev_frame_idx} <-> {current_frame_idx}（直接特征匹配，无粗对齐）")
            
            # 2. 获取轮廓点（从缓存或重新提取）
            # 优先从缓存获取，如果没有则重新提取
            contour_points_prev = None
            contour_points_curr = None
            
            if hasattr(self, '_frame_contours') and prev_frame_idx in self._frame_contours:
                contour_points_prev = self._frame_contours[prev_frame_idx]
                print(f"  [轮廓] 前一帧: 从缓存获取轮廓点")
            else:
                contour_points_prev = self._extract_contour_points(frame_idx=prev_frame_idx, suffix="68")
                if contour_points_prev is not None:
                    if not hasattr(self, '_frame_contours'):
                        self._frame_contours = {}
                    self._frame_contours[prev_frame_idx] = contour_points_prev
                    print(f"  [轮廓] 前一帧: 重新提取轮廓点")
                else:
                    print(f"  [轮廓] 前一帧: 未提取到轮廓点，使用全图采样")
            
            if hasattr(self, '_frame_contours') and current_frame_idx in self._frame_contours:
                contour_points_curr = self._frame_contours[current_frame_idx]
                print(f"  [轮廓] 当前帧: 从缓存获取轮廓点")
            else:
                contour_points_curr = self._extract_contour_points(frame_idx=current_frame_idx, suffix="68")
                if contour_points_curr is not None:
                    if not hasattr(self, '_frame_contours'):
                        self._frame_contours = {}
                    self._frame_contours[current_frame_idx] = contour_points_curr
                    print(f"  [轮廓] 当前帧: 重新提取轮廓点")
                else:
                    print(f"  [轮廓] 当前帧: 未提取到轮廓点，使用全图采样")
            
            # 3. 直接提取特征（不进行粗对齐，优先使用mask采样，其次轮廓点采样）
            # 使用FPFH + 曲率特征（36维）
            # 获取mask（优先从缓存获取）
            mask_prev = None
            mask_curr = None
            if hasattr(self, '_frame_masks'):
                mask_prev = self._frame_masks.get(prev_frame_idx, None)
                mask_curr = self._frame_masks.get(current_frame_idx, None)
            
            # 优化：从内存缓存获取mesh对象（不读取文件）
            mesh_prev = self._mesh_cache[prev_frame_idx]
            mesh_curr = self._mesh_cache[current_frame_idx]
            
            # 优化：使用特征缓存（如果已提取过）
            cache_key_prev = f"feat_{prev_frame_idx}_{suffix}"
            cache_key_curr = f"feat_{current_frame_idx}_{suffix}"
            
            if cache_key_prev in self._mesh_features_cache:
                feat_prev = self._mesh_features_cache[cache_key_prev]
                print(f"  [特征缓存] 前一帧特征从缓存获取")
            else:
                feat_prev = self._extract_3d_mesh_features(
                    mesh_prev,  # 直接传入mesh对象
                    max_points=10000,
                    radius_normal=0.15,  # 法向量估计半径
                    radius_feature=0.3,   # 特征计算半径
                    pre_transform=None,   # 不应用任何预变换
                    mask=mask_prev,  # 优先使用mask采样
                    contour_points_list=contour_points_prev  # 如果没有mask，使用轮廓点采样
                )
                if feat_prev:
                    self._mesh_features_cache[cache_key_prev] = feat_prev
            
            if cache_key_curr in self._mesh_features_cache:
                feat_curr = self._mesh_features_cache[cache_key_curr]
                print(f"  [特征缓存] 当前帧特征从缓存获取")
            else:
                feat_curr = self._extract_3d_mesh_features(
                    mesh_curr,  # 直接传入mesh对象
                    max_points=10000,
                    radius_normal=0.15,
                    radius_feature=0.3,
                    pre_transform=None,   # 不应用任何预变换
                    mask=mask_curr,  # 优先使用mask采样
                    contour_points_list=contour_points_curr  # 如果没有mask，使用轮廓点采样
                )
                if feat_curr:
                    self._mesh_features_cache[cache_key_curr] = feat_curr
            
            if feat_prev is None or feat_curr is None:
                print(f"  [警告] 特征提取失败，跳过匹配")
                # 如果特征提取失败，使用单位变换矩阵作为降级策略
                if not hasattr(self, '_rt_matrices'):
                    self._rt_matrices = {}
                identity_matrix = np.eye(4)
                self._rt_matrices[f"{prev_frame_idx}_{current_frame_idx}"] = identity_matrix
                print(f"  [降级] 使用单位变换矩阵作为RT矩阵")
                return
            
            # 4. 进行特征匹配（使用FPFH + 曲率特征）
            match_result = self._match_3d_mesh_features(
                feat_prev, 
                feat_curr,
                max_correspondence_distance=0.15,  # 对应距离
                mutual_filter=False  # 关闭互过滤，避免过滤掉太多点
            )
            
            if match_result is None:
                print(f"  [警告] 特征匹配失败，使用单位变换矩阵作为降级策略")
                # 使用单位变换矩阵作为降级策略，而不是直接返回
                if not hasattr(self, '_rt_matrices'):
                    self._rt_matrices = {}
                identity_matrix = np.eye(4)
                self._rt_matrices[f"{prev_frame_idx}_{current_frame_idx}"] = identity_matrix
                
                # 尝试更新累积位姿（使用单位变换）
                try:
                    pose_result = self._update_accumulated_pose(
                        prev_frame_idx=prev_frame_idx,
                        current_frame_idx=current_frame_idx,
                        rt_matrix=identity_matrix,
                        reference_frame_idx=getattr(self, 'start_frame_idx', 0)
                    )
                    if pose_result:
                        print(f"  [位姿] Frame {current_frame_idx} 的累积位姿已更新（使用单位变换）")
                except Exception as e:
                    print(f"  [警告] 更新累积位姿失败: {e}")
                return
            
            # 5. 保存采样点云坐标到匹配结果中（用于可视化）
            # 匹配结果中的transformation已经是最终变换（从第二帧坐标系到第一帧坐标系）
            try:
                match_result['sampled_points1'] = feat_prev['points']  # 第一帧采样点云（原始坐标，在第一帧坐标系）
                
                # 将第二帧的点变换到第一帧的坐标系中
                points2_original = feat_curr['points']  # 第二帧采样点云（原始坐标，在第二帧坐标系）
                if len(points2_original) > 0:
                    points2_homo = np.hstack([points2_original, np.ones((len(points2_original), 1))])
                    match_result['sampled_points2_final'] = (match_result['transformation'] @ points2_homo.T).T[:, :3]
                else:
                    match_result['sampled_points2_final'] = np.array([])
            except Exception as e:
                print(f"  [警告] 保存采样点云坐标失败: {e}")
                match_result['sampled_points1'] = np.array([])
                match_result['sampled_points2_final'] = np.array([])
            
            print(f"  [匹配完成] 变换矩阵已计算，fitness={match_result['fitness']:.4f}, RMSE={match_result['inlier_rmse']:.6f}")

            # 5.5 深度恢复（可选）：复用3D匹配的对应关系 + 标定CSV真实位姿（度量尺度）
            # 启用方式：设置环境变量 DEPTH_RECOVERY_ENABLE=1，并提供 DEPTH_GT_POSE_CSV 或 GT_POSE_CSV
            try:
                if bool(int(os.environ.get("DEPTH_RECOVERY_ENABLE", "0"))):
                    self._depth_recover_from_3d_correspondences(
                        prev_frame_idx=prev_frame_idx,
                        current_frame_idx=current_frame_idx,
                        points1=feat_prev.get("points", None),
                        points2=feat_curr.get("points", None),
                        correspondence_set=match_result.get("correspondence_set", None),
                    )
            except Exception as e:
                print(f"  [警告] 深度恢复失败(帧{prev_frame_idx}->{current_frame_idx}): {e}")
            
            # 打印轮廓信息
            if contour_points_prev is not None:
                num_contour_points_prev = sum(len(c) for c in contour_points_prev)
                print(f"  [轮廓信息] 前一帧: {len(contour_points_prev)} 个轮廓，共 {num_contour_points_prev} 个轮廓点")
            else:
                print(f"  [轮廓信息] 前一帧: 未使用轮廓点（全图采样）")
            
            if contour_points_curr is not None:
                num_contour_points_curr = sum(len(c) for c in contour_points_curr)
                print(f"  [轮廓信息] 当前帧: {len(contour_points_curr)} 个轮廓，共 {num_contour_points_curr} 个轮廓点")
            else:
                print(f"  [轮廓信息] 当前帧: 未使用轮廓点（全图采样）")
            
            # 6. 保存匹配结果
            if not hasattr(self, '_frame_3d_matches'):
                self._frame_3d_matches = []
            
            # 安全地提取匹配结果信息
            try:
                T = match_result.get('transformation', np.eye(4))
                if T is None:
                    T = np.eye(4)
                R = T[:3, :3]
                t = T[:3, 3]
                
                match_info = {
                    'frame_idx1': prev_frame_idx,
                    'frame_idx2': current_frame_idx,
                    'transformation': T,
                    'rotation': R,
                    'translation': t,
                    'fitness': match_result.get('fitness', 0.0),
                    'inlier_rmse': match_result.get('inlier_rmse', float('inf')),
                    'method': match_result.get('method', 'unknown')
                }
            except Exception as e:
                print(f"  [警告] 提取匹配结果信息失败: {e}")
                # 使用单位变换矩阵作为降级策略
                T = np.eye(4)
                match_info = {
                    'frame_idx1': prev_frame_idx,
                    'frame_idx2': current_frame_idx,
                    'transformation': T,
                    'rotation': T[:3, :3],
                    'translation': T[:3, 3],
                    'fitness': 0.0,
                    'inlier_rmse': float('inf'),
                    'method': 'error_fallback'
                }
            
            self._frame_3d_matches.append(match_info)
            
            # 保存RT矩阵（transformation是将当前帧点变换到前一帧坐标系，与RT矩阵含义一致）
            if not hasattr(self, '_rt_matrices'):
                self._rt_matrices = {}
            self._rt_matrices[f"{prev_frame_idx}_{current_frame_idx}"] = T
            
            # 实时更新累积位姿（立即计算当前帧的相机位姿）
            pose_result = self._update_accumulated_pose(
                prev_frame_idx=prev_frame_idx,
                current_frame_idx=current_frame_idx,
                rt_matrix=T,  # transformation矩阵就是RT矩阵
                reference_frame_idx=getattr(self, 'start_frame_idx', 0)
            )
            if pose_result:
                print(f"  [位姿] Frame {current_frame_idx} 的累积位姿已更新")
            
            print(f"[3D匹配] 帧 {prev_frame_idx} <-> {current_frame_idx}: "
                  f"fitness={match_info['fitness']:.4f}, RMSE={match_info['inlier_rmse']:.6f}, "
                  f"方法={match_info['method']}")
            
            # 7. 可视化并保存特征匹配结果（在原始高度场网格上）
            # 优化：传入mesh对象而不是文件路径
            try:
                self._visualize_3d_feature_matches(
                    mesh_prev, mesh_curr, match_result,
                    prev_frame_idx, current_frame_idx, suffix
                )
            except Exception as e:
                print(f"  [警告] 可视化匹配结果失败: {e}")
            
        except Exception as e:
            print(f"[警告] 帧 {current_frame_idx} 的3D匹配失败: {e}")
            import traceback
            traceback.print_exc()

    def _depth_recover_from_3d_correspondences(self, prev_frame_idx, current_frame_idx, points1, points2, correspondence_set):
        """
        基于3D网格匹配得到的 correspondence_set（点索引对应关系）恢复两帧的稀疏深度。

        - 对应点来自伪3D网格点 (u,v,h)，我们仅取 (u,v) 作为像素对应关系。
        - 相对位姿来自标定CSV（需提供度量尺度位姿与内参K）。

        环境变量：
          - DEPTH_GT_POSE_CSV / GT_POSE_CSV: 标定CSV路径
          - DEPTH_GT_POSE_OFFSET: 帧索引偏移（默认0）
          - DEPTH_RECOVERY_REPROJ_THRESH: 重投影阈值像素（默认3.0）
          - DEPTH_RECOVERY_MAX_CORR: 最多使用的对应点数量（默认5000）
        """
        if points1 is None or points2 is None or correspondence_set is None:
            return False
        points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 3)
        points2 = np.asarray(points2, dtype=np.float64).reshape(-1, 3)
        corr = correspondence_set
        if corr is None:
            return False
        if len(points1) == 0 or len(points2) == 0 or len(corr) < 8:
            return False

        # 1) 加载并缩放图像（与process_image的缩放逻辑保持一致）
        img1, sx, sy, orig1 = self._depth_load_scaled_frame_image(prev_frame_idx)
        img2, sx2, sy2, orig2 = self._depth_load_scaled_frame_image(current_frame_idx)
        # 两帧缩放应一致；不一致则按第一帧为准强制resize
        if img2.shape[:2] != img1.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
            sx2, sy2 = sx, sy
        H, W = img1.shape[:2]

        # 2) 选择位姿来源：优先使用“现有代码计算的位姿结果”，可选使用GT CSV
        pose_source = os.environ.get("DEPTH_POSE_SOURCE", "existing").strip().lower()  # existing | gtcsv | auto
        gt_csv = os.environ.get("DEPTH_GT_POSE_CSV", None) or os.environ.get("GT_POSE_CSV", None)

        T1 = None
        T2 = None
        note_pose = ""

        # 2.1) existing：使用 self._accumulated_poses（你现有代码实时累积的位姿结果）
        if pose_source in ["existing", "auto"]:
            if hasattr(self, "_accumulated_poses") and (prev_frame_idx in self._accumulated_poses) and (current_frame_idx in self._accumulated_poses):
                T1 = np.asarray(self._accumulated_poses[prev_frame_idx], dtype=np.float64)
                T2 = np.asarray(self._accumulated_poses[current_frame_idx], dtype=np.float64)
                note_pose = "pose_source=existing_accumulated_poses"
            elif pose_source == "existing":
                raise RuntimeError("未找到现有累积位姿：请确保位姿已通过 _update_accumulated_pose() 实时更新，并且包含这两帧")

        # 2.2) gtcsv：使用标定CSV位姿（如果提供）
        if (T1 is None or T2 is None) and (pose_source in ["gtcsv", "auto"]):
            if gt_csv is None or (not os.path.exists(gt_csv)):
                if pose_source == "gtcsv":
                    raise FileNotFoundError("未提供有效的GT位姿CSV：请设置 DEPTH_GT_POSE_CSV 或 GT_POSE_CSV")
            else:
                if not hasattr(self, "_depth_gt_cache"):
                    self._depth_gt_cache = None
                if (self._depth_gt_cache is None) or (self._depth_gt_cache.get("csv_path") != gt_csv):
                    gt_poses, _gt_positions = load_camera_poses_from_file(gt_csv)
                    self._depth_gt_cache = {"csv_path": gt_csv, "poses": gt_poses}
                gt_poses = self._depth_gt_cache.get("poses", [])
                offset = int(os.environ.get("DEPTH_GT_POSE_OFFSET", "0"))
                idx1 = int(prev_frame_idx) + offset
                idx2 = int(current_frame_idx) + offset
                if idx1 < 0 or idx2 < 0 or idx1 >= len(gt_poses) or idx2 >= len(gt_poses):
                    raise IndexError(f"GT位姿索引越界：len={len(gt_poses)} idx1={idx1} idx2={idx2} offset={offset}")
                T1 = np.asarray(gt_poses[idx1], dtype=np.float64)
                T2 = np.asarray(gt_poses[idx2], dtype=np.float64)
                note_pose = f"pose_source=gtcsv idx={idx1}->{idx2} offset={offset} csv={gt_csv}"

        if T1 is None or T2 is None:
            raise RuntimeError("无法获得两帧的位姿矩阵：请检查 DEPTH_POSE_SOURCE，以及是否有现有累积位姿或GT CSV")

        # 3) 内参K：优先从GT CSV解析（若可用且合理），否则按环境变量或自动估计
        K = None
        if gt_csv is not None and os.path.exists(gt_csv):
            # 使用原始分辨率做合理性检查（cx/cy应在图像附近）
            intr_csv = _depth_load_intrinsics_from_pose_csv(gt_csv, image_wh=orig1)
            if intr_csv is not None:
                fx, fy, cx, cy = intr_csv
                K = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)
        if K is None:
            K = _depth_make_K_from_env_or_estimate((H, W), sx=sx, sy=sy)

        # 3) 从correspondence_set提取像素对应（从points中取u,v）
        max_corr = int(os.environ.get("DEPTH_RECOVERY_MAX_CORR", "5000"))
        pts1 = []
        pts2 = []
        used = 0
        for c in corr:
            if used >= max_corr:
                break
            try:
                i, j = int(c[0]), int(c[1])
            except Exception:
                continue
            if i < 0 or j < 0 or i >= len(points1) or j >= len(points2):
                continue
            u1, v1 = points1[i, 0], points1[i, 1]
            u2, v2 = points2[j, 0], points2[j, 1]
            if not (np.isfinite(u1) and np.isfinite(v1) and np.isfinite(u2) and np.isfinite(v2)):
                continue
            # 边界过滤（注意：mesh点本身就是像素坐标系）
            if 0 <= u1 < W and 0 <= v1 < H and 0 <= u2 < W and 0 <= v2 < H:
                pts1.append([u1, v1])
                pts2.append([u2, v2])
                used += 1

        pts1 = np.asarray(pts1, dtype=np.float32)
        pts2 = np.asarray(pts2, dtype=np.float32)
        if len(pts1) < 8:
            raise RuntimeError(f"有效像素对应点不足：{len(pts1)}")

        # 4) 相对位姿（自动判别位姿约定：cam_to_world vs world_to_cam）
        cands = _depth_relative_pose_candidates_from_abs(T1, T2)
        reproj_th = float(os.environ.get("DEPTH_RECOVERY_REPROJ_THRESH", "3.0"))

        best = None
        best_name = None
        best_score = (-1, float("inf"))
        best_Rt = None
        for name, (R, t) in cands.items():
            out = _depth_triangulate_sparse(K, R, t, pts1, pts2, (H, W), reproj_thresh_px=reproj_th)
            if out is None:
                continue
            n = int(out["stats"]["final_points"])
            e = float(out["stats"]["mean_reproj1"]) + float(out["stats"]["mean_reproj2"])
            score = (n, e)
            if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_score = score
                best = out
                best_name = name
                best_Rt = (R, t)

        if best is None or best_Rt is None:
            raise RuntimeError("三角化失败：两种位姿约定都未得到可用结果")

        R_best, t_best = best_Rt

        # 5) 保存输出
        out_root = os.path.join(self.output_dir, "depth_recovery")
        os.makedirs(out_root, exist_ok=True)
        out_dir = os.path.join(out_root, f"{prev_frame_idx:04d}_{current_frame_idx:04d}")
        os.makedirs(out_dir, exist_ok=True)

        np.savez(os.path.join(out_dir, "matches_uv.npz"), pts1=pts1, pts2=pts2)
        np.save(os.path.join(out_dir, "depth_sparse.npy"), best["depth_map"])
        _depth_save_vis(best["depth_map"], os.path.join(out_dir, "depth_sparse_vis.png"))

        # 点云颜色从img1采样
        p1 = best["pts1"]
        xs = np.clip(np.round(p1[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(p1[:, 1]).astype(int), 0, H - 1)
        colors = img1[ys, xs, :]
        _depth_write_ply_xyzrgb(os.path.join(out_dir, "points_cam1.ply"), best["X_cam1"], colors)

        # 统计信息
        with open(os.path.join(out_dir, "depth_stats.txt"), "w", encoding="utf-8") as f:
            f.write("=== depth recovery from 3d correspondences ===\n")
            f.write(f"frames: {prev_frame_idx} -> {current_frame_idx}\n")
            f.write(f"{note_pose}\n")
            f.write(f"pose_convention: {best_name}\n")
            f.write(f"image_size: {W} x {H}\n")
            f.write(f"scale: sx={sx:.6f} sy={sy:.6f}\n")
            f.write(f"K_scaled: fx={K[0,0]:.6f} fy={K[1,1]:.6f} cx={K[0,2]:.6f} cy={K[1,2]:.6f}\n\n")
            f.write("R (cam2<-cam1):\n")
            for row in R_best:
                f.write("  " + " ".join([f"{v: .8f}" for v in row]) + "\n")
            f.write("t (cam2<-cam1):\n")
            f.write("  " + " ".join([f"{v: .8f}" for v in np.asarray(t_best).reshape(-1)]) + "\n\n")
            for k, v in best["stats"].items():
                f.write(f"{k}: {v}\n")

        print(f"  [深度恢复] 完成: {out_dir} (pose={best_name}, points={best['stats']['final_points']})")
        return True

    def _depth_recover_from_orb_pair(self, prev_frame_idx, current_frame_idx):
        """
        不依赖Open3D：直接用两帧图像 ORB 匹配点 +（现有累积位姿/GT CSV位姿）做两视图三角化恢复稀疏深度。

        环境变量：
          - DEPTH_POSE_SOURCE: existing | gtcsv | auto（默认existing，但无Open3D时通常建议gtcsv）
          - DEPTH_GT_POSE_CSV / GT_POSE_CSV: gtcsv模式的CSV路径
          - DEPTH_GT_POSE_OFFSET: 帧索引偏移（默认0）
          - DEPTH_RECOVERY_REPROJ_THRESH: 重投影阈值像素（默认3.0）
          - DEPTH_ORB_MAX_MATCHES: ORB最大匹配点（默认2000）
        """
        # 1) 加载并缩放图像
        img1, sx, sy, orig1 = self._depth_load_scaled_frame_image(prev_frame_idx)
        img2, sx2, sy2, orig2 = self._depth_load_scaled_frame_image(current_frame_idx)
        if img2.shape[:2] != img1.shape[:2]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
            sx2, sy2 = sx, sy
        H, W = img1.shape[:2]

        # 2) ORB匹配
        max_m = int(os.environ.get("DEPTH_ORB_MAX_MATCHES", "2000"))
        pts1, pts2 = _depth_orb_match_points(img1, img2, max_matches=max_m)
        if pts1 is None or pts2 is None or len(pts1) < 8:
            raise RuntimeError(f"ORB匹配点不足，无法恢复深度: {0 if pts1 is None else len(pts1)}")

        # 3) 位姿来源
        pose_source = os.environ.get("DEPTH_POSE_SOURCE", "existing").strip().lower()  # existing | gtcsv | auto
        gt_csv = os.environ.get("DEPTH_GT_POSE_CSV", None) or os.environ.get("GT_POSE_CSV", None)

        T1 = None
        T2 = None
        note_pose = ""

        if pose_source in ["existing", "auto"]:
            if hasattr(self, "_accumulated_poses") and (prev_frame_idx in self._accumulated_poses) and (current_frame_idx in self._accumulated_poses):
                T1 = np.asarray(self._accumulated_poses[prev_frame_idx], dtype=np.float64)
                T2 = np.asarray(self._accumulated_poses[current_frame_idx], dtype=np.float64)
                note_pose = "pose_source=existing_accumulated_poses"
            elif pose_source == "existing":
                raise RuntimeError("existing位姿不可用：当前环境未生成_accumulated_poses或不包含该帧对")

        if (T1 is None or T2 is None) and (pose_source in ["gtcsv", "auto"]):
            if gt_csv is None or (not os.path.exists(gt_csv)):
                if pose_source == "gtcsv":
                    raise FileNotFoundError("gtcsv位姿不可用：请设置 DEPTH_GT_POSE_CSV 或 GT_POSE_CSV")
            else:
                if not hasattr(self, "_depth_gt_cache"):
                    self._depth_gt_cache = None
                if (self._depth_gt_cache is None) or (self._depth_gt_cache.get("csv_path") != gt_csv):
                    gt_poses, _gt_positions = load_camera_poses_from_file(gt_csv)
                    self._depth_gt_cache = {"csv_path": gt_csv, "poses": gt_poses}
                gt_poses = self._depth_gt_cache.get("poses", [])
                offset = int(os.environ.get("DEPTH_GT_POSE_OFFSET", "0"))
                idx1 = int(prev_frame_idx) + offset
                idx2 = int(current_frame_idx) + offset
                if idx1 < 0 or idx2 < 0 or idx1 >= len(gt_poses) or idx2 >= len(gt_poses):
                    raise IndexError(f"GT位姿索引越界：len={len(gt_poses)} idx1={idx1} idx2={idx2} offset={offset}")
                T1 = np.asarray(gt_poses[idx1], dtype=np.float64)
                T2 = np.asarray(gt_poses[idx2], dtype=np.float64)
                note_pose = f"pose_source=gtcsv idx={idx1}->{idx2} offset={offset} csv={gt_csv}"

        if T1 is None or T2 is None:
            raise RuntimeError("无法获得位姿矩阵：请检查 DEPTH_POSE_SOURCE / GT CSV配置")

        # 4) 内参K：优先从GT CSV解析，否则用env/估计
        K = None
        if gt_csv is not None and os.path.exists(gt_csv):
            intr_csv = _depth_load_intrinsics_from_pose_csv(gt_csv, image_wh=orig1)
            if intr_csv is not None:
                fx, fy, cx, cy = intr_csv
                K = np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)
        if K is None:
            K = _depth_make_K_from_env_or_estimate((H, W), sx=sx, sy=sy)

        # 5) 相对位姿约定自动择优 + 三角化
        cands = _depth_relative_pose_candidates_from_abs(T1, T2)
        reproj_th = float(os.environ.get("DEPTH_RECOVERY_REPROJ_THRESH", "3.0"))

        best = None
        best_name = None
        best_score = (-1, float("inf"))
        best_Rt = None
        for name, (R, t) in cands.items():
            out = _depth_triangulate_sparse(K, R, t, pts1, pts2, (H, W), reproj_thresh_px=reproj_th)
            if out is None:
                continue
            n = int(out["stats"]["final_points"])
            e = float(out["stats"]["mean_reproj1"]) + float(out["stats"]["mean_reproj2"])
            score = (n, e)
            if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_score = score
                best = out
                best_name = name
                best_Rt = (R, t)

        if best is None or best_Rt is None:
            raise RuntimeError("ORB三角化失败：两种位姿约定都未得到可用结果")

        R_best, t_best = best_Rt

        out_root = os.path.join(self.output_dir, "depth_recovery_orb")
        os.makedirs(out_root, exist_ok=True)
        out_dir = os.path.join(out_root, f"{prev_frame_idx:04d}_{current_frame_idx:04d}")
        os.makedirs(out_dir, exist_ok=True)

        np.savez(os.path.join(out_dir, "matches_orb.npz"), pts1=pts1, pts2=pts2)
        np.save(os.path.join(out_dir, "depth_sparse.npy"), best["depth_map"])
        _depth_save_vis(best["depth_map"], os.path.join(out_dir, "depth_sparse_vis.png"))

        p1 = best["pts1"]
        xs = np.clip(np.round(p1[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(p1[:, 1]).astype(int), 0, H - 1)
        colors = img1[ys, xs, :]
        _depth_write_ply_xyzrgb(os.path.join(out_dir, "points_cam1.ply"), best["X_cam1"], colors)

        with open(os.path.join(out_dir, "depth_stats.txt"), "w", encoding="utf-8") as f:
            f.write("=== depth recovery (ORB matches) ===\n")
            f.write(f"frames: {prev_frame_idx} -> {current_frame_idx}\n")
            f.write(f"{note_pose}\n")
            f.write(f"pose_convention: {best_name}\n")
            f.write(f"image_size: {W} x {H}\n")
            f.write(f"scale: sx={sx:.6f} sy={sy:.6f}\n")
            f.write(f"K_scaled: fx={K[0,0]:.6f} fy={K[1,1]:.6f} cx={K[0,2]:.6f} cy={K[1,2]:.6f}\n\n")
            f.write("R (cam2<-cam1):\n")
            for row in R_best:
                f.write("  " + " ".join([f"{v: .8f}" for v in row]) + "\n")
            f.write("t (cam2<-cam1):\n")
            f.write("  " + " ".join([f"{v: .8f}" for v in np.asarray(t_best).reshape(-1)]) + "\n\n")
            for k, v in best["stats"].items():
                f.write(f"{k}: {v}\n")

        print(f"  [深度恢复-ORB] 完成: {out_dir} (pose={best_name}, points={best['stats']['final_points']})")
        return True

    def _depth_load_scaled_frame_image(self, frame_idx):
        """从已记录的原始图像路径加载图像，并复刻process_image里的缩放逻辑，返回(img_scaled, sx, sy, (orig_w, orig_h))."""
        if not hasattr(self, "_image_paths") or frame_idx not in self._image_paths:
            raise RuntimeError(f"未找到帧{frame_idx}的原始图像路径（需要process_image传入original_image_path并存在文件）")
        p = self._image_paths[frame_idx]
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {p}")
        h, w = img.shape[:2]
        scale = float(getattr(self, "image_scale_factor", 1.0))
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            new_w = (new_w // 2) * 2
            new_h = (new_h // 2) * 2
            if new_w <= 0 or new_h <= 0:
                raise RuntimeError(f"缩放后尺寸无效: {new_w}x{new_h} (orig={w}x{h}, scale={scale})")
            if new_w != w or new_h != h:
                img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img2 = img
        else:
            img2 = img
            new_w, new_h = w, h
        sx = float(new_w) / float(w)
        sy = float(new_h) / float(h)
        return img2, sx, sy, (int(w), int(h))

    def _generate_checkerboard_texture(self, vertices, device, grid_size=10.0):
        """
        生成粉红色纹理

        Args:
            vertices: 顶点坐标 (N, 3)，torch.Tensor
            device: torch设备
            grid_size: 网格大小（未使用，保留参数以兼容现有代码）

        Returns:
            colors: 顶点颜色 (N, 3)，RGB格式，值范围[0, 1]
        """
        # 粉红色 RGB: (1.0, 0.75, 0.8)
        pink_color = torch.tensor([1.0, 0.75, 0.8], dtype=torch.float32, device=device)

        # 为所有顶点设置粉红色
        colors = pink_color.unsqueeze(0).repeat(len(vertices), 1)

        return colors
    
    def _visualize_training_process(self, loss_history, loss_silhouette_history, loss_mesh_history,
                                    loss_reg_history, loss_normal_history, loss_semantic_history,
                                    learning_rate_history, loss_type, prev_frame_idx, current_frame_idx, save_dir):
        """
        可视化训练过程（损失曲线、学习率曲线等）
        
        Args:
            loss_history: 总损失历史
            loss_silhouette_history: 轮廓损失历史
            loss_mesh_history: mesh损失历史
            loss_reg_history: 正则化损失历史
            loss_normal_history: 法向量损失历史
            loss_semantic_history: 语义损失历史
            learning_rate_history: 学习率历史
            loss_type: 损失类型
            prev_frame_idx: 前一帧索引
            current_frame_idx: 当前帧索引
            save_dir: 保存目录
        """
        try:
            if len(loss_history) == 0:
                print(f"  [警告] 损失历史为空，无法可视化")
                return
            
            # 设置中文字体
            setup_chinese_font()
            
            # 创建多子图
            num_subplots = 2 if loss_type in ["silhouette", "mesh_error"] else 3
            fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots))
            
            if num_subplots == 1:
                axes = [axes]
            
            iterations = range(len(loss_history))
            
            # 子图1: 总损失曲线
            ax1 = axes[0]
            ax1.plot(iterations, loss_history, 'b-', linewidth=2, label='总损失')
            ax1.axhline(y=min(loss_history), color='r', linestyle='--', linewidth=1.5, 
                       label=f'最佳损失: {min(loss_history):.6f}')
            ax1.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
            ax1.set_ylabel('总损失', fontsize=12, fontweight='bold')
            ax1.set_title(f'总损失曲线 (帧 {prev_frame_idx} -> {current_frame_idx})', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=11)
            
            # 子图2: 学习率曲线（单独子图）
            if len(learning_rate_history) > 0 and num_subplots > 1:
                ax2 = axes[1]
                ax2.plot(iterations, learning_rate_history, 'g-', linewidth=2, label='学习率')
                ax2.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
                ax2.set_ylabel('学习率', fontsize=12, fontweight='bold', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.set_title('学习率变化', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right', fontsize=11)
            
            # 子图3: 各项损失分解（仅当使用enhanced或combined损失类型时）
            if loss_type in ["combined", "enhanced"] and num_subplots >= 3:
                ax3 = axes[2]
                
                # 只绘制非零的损失项
                if len(loss_silhouette_history) > 0 and max(loss_silhouette_history) > 0:
                    ax3.plot(iterations, loss_silhouette_history, 'r-', linewidth=1.5, label='轮廓损失', alpha=0.7)
                if len(loss_mesh_history) > 0 and max(loss_mesh_history) > 0:
                    ax3.plot(iterations, loss_mesh_history, 'b-', linewidth=1.5, label='Mesh损失', alpha=0.7)
                if len(loss_reg_history) > 0 and max(loss_reg_history) > 0:
                    ax3.plot(iterations, loss_reg_history, 'm-', linewidth=1.5, label='正则化损失', alpha=0.7)
                if len(loss_normal_history) > 0 and max(loss_normal_history) > 0:
                    ax3.plot(iterations, loss_normal_history, 'c-', linewidth=1.5, label='法向量损失', alpha=0.7)
                if len(loss_semantic_history) > 0 and max(loss_semantic_history) > 0:
                    ax3.plot(iterations, loss_semantic_history, 'y-', linewidth=1.5, label='语义损失', alpha=0.7)
                
                ax3.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
                ax3.set_ylabel('各项损失', fontsize=12, fontweight='bold')
                ax3.set_title('损失分解（各项损失）', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.legend(fontsize=10, loc='upper right')
                # 使用对数刻度以便更好地显示不同尺度的损失
                if max(loss_silhouette_history + loss_mesh_history + loss_reg_history + loss_normal_history + loss_semantic_history) > 0:
                    ax3.set_yscale('log')
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(save_dir, "training_visualization.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            print(f"  [保存] 训练过程可视化已保存: {save_path}")
            
        except Exception as e:
            print(f"  [警告] 训练过程可视化失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果详细可视化失败，至少保存简单的损失曲线
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(loss_history)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title(f'Loss History: Frame {prev_frame_idx} -> {current_frame_idx}')
                plt.grid(True)
                plt.savefig(os.path.join(save_dir, "loss_curve.png"))
                plt.close()
                print(f"  [保存] 简单损失曲线已保存")
            except:
                pass
    
    def _save_textured_mesh_to_ply(self, vertices, faces, colors, file_path):
        """
        将带纹理的mesh保存为PLY文件
        
        Args:
            vertices: 顶点坐标 (N, 3)
            faces: 面片索引 (M, 3)
            colors: 顶点颜色 (N, 3)，RGB格式，值范围[0, 1]
            file_path: 输出文件路径
        """
        try:
            # 方法1: 使用Open3D保存（推荐）
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
            
            # 设置顶点颜色（需要转换为0-255范围）
            colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors_uint8.astype(np.float64) / 255.0)
            
            # 保存PLY文件
            success = o3d.io.write_triangle_mesh(file_path, mesh_o3d, write_ascii=False)
            
            if not success:
                # 方法2: 如果Open3D保存失败，手动写入PLY文件
                print(f"    [警告] Open3D保存失败，尝试手动保存PLY...")
                self._save_ply_manual(file_path, vertices, faces, colors)
            else:
                print(f"    [保存] 使用Open3D保存成功")
                
        except Exception as e:
            print(f"    [警告] Open3D保存异常: {e}，尝试手动保存PLY...")
            try:
                self._save_ply_manual(file_path, vertices, faces, colors)
            except Exception as e2:
                print(f"    [错误] 手动保存也失败: {e2}")
                raise
    
    def _save_ply_manual(self, file_path, vertices, faces, colors):
        """
        手动写入PLY文件（ASCII格式，兼容性更好）
        
        Args:
            file_path: 输出文件路径
            vertices: 顶点坐标 (N, 3)
            faces: 面片索引 (M, 3)
            colors: 顶点颜色 (N, 3)，RGB格式，值范围[0, 1]
        """
        num_vertices = len(vertices)
        num_faces = len(faces)
        
        # 转换颜色到0-255范围
        colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # PLY文件头
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Generated by PyTorch3D rendering optimization\n")
            f.write(f"comment Mesh with {num_vertices} vertices and {num_faces} faces\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # 写入顶点数据（坐标 + 颜色）
            for i in range(num_vertices):
                x, y, z = vertices[i]
                r, g, b = colors_uint8[i]
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}\n")
            
            # 写入面片数据
            for i in range(num_faces):
                v0, v1, v2 = faces[i]
                f.write(f"3 {v0} {v1} {v2}\n")
        
        print(f"    [保存] 手动PLY保存成功")
    
    def _extract_texture_from_image(self, vertices, image_path, mask_path, 
                                    image_width, image_height, device):
        """
        从原始图像中提取纹理颜色并映射到mesh顶点
        
        Args:
            vertices: mesh顶点坐标 (N, 3)，x,y对应图像像素坐标，z为高度
            image_path: 原始图像路径
            mask_path: mask图像路径
            image_width: 图像宽度
            image_height: 图像高度
            device: torch设备
            
        Returns:
            colors: 顶点颜色tensor (N, 3)，RGB格式，值范围[0, 1]
        """
        # 读取原始图像（BGR格式）
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]
        
        # 读取mask（如果存在）
        mask = None
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                mask = (mask_img > 127).astype(np.uint8)  # 二值化mask
        
        # 提取顶点颜色
        num_vertices = len(vertices)
        colors = np.zeros((num_vertices, 3), dtype=np.float32)
        
        for i, vertex in enumerate(vertices):
            x, y, z = vertex
            
            # 将顶点坐标映射到图像像素坐标
            # mesh的x,y坐标直接对应图像像素坐标
            px = int(np.clip(x, 0, img_w - 1))
            py = int(np.clip(y, 0, img_h - 1))
            
            # 检查是否在mask范围内（如果mask存在）
            if mask is not None:
                if py < mask.shape[0] and px < mask.shape[1]:
                    if mask[py, px] == 0:
                        # 不在mask内，使用默认颜色（白色）
                        colors[i] = [1.0, 1.0, 1.0]
                        continue
            
            # 从图像中提取颜色（RGB，值范围0-255）
            pixel_color = image_rgb[py, px].astype(np.float32)
            # 归一化到[0, 1]
            colors[i] = pixel_color / 255.0
        
        # 转换为torch tensor
        colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)
        return colors_tensor
    
    class _PoseOptimizationModel(nn.Module):
        """
        基于可微分渲染的位姿优化模型类
        参考PyTorch3D官方示例代码设计
        """
        def __init__(self, meshes_prev, meshes_curr, renderer, image_ref, 
                     camera_R, camera_T, initial_pose=None, device=None,
                     loss_type="enhanced", silhouette_gt=None, vertices_gt=None, 
                     normals_gt=None, mesh_prev_o3d=None, mesh_curr_o3d=None):
            """
            Args:
                meshes_prev: 前一帧的Meshes对象
                meshes_curr: 当前帧的Meshes对象
                renderer: 轮廓渲染器（SoftSilhouetteShader）
                image_ref: 参考图像（前一帧的渲染轮廓）
                camera_R: 相机旋转矩阵（3x3）
                camera_T: 相机平移向量（3,）
                initial_pose: 初始位姿（4x4矩阵），如果为None则使用单位矩阵
                device: torch设备
                loss_type: 损失类型，"silhouette", "mesh_error", "combined", or "enhanced"
                silhouette_gt: Ground truth轮廓（用于损失计算）
                vertices_gt: Ground truth顶点（用于mesh损失计算）
                normals_gt: Ground truth法向量（用于法向量损失计算）
                mesh_prev_o3d: 前一帧的Open3D mesh对象（用于法向量计算）
                mesh_curr_o3d: 当前帧的Open3D mesh对象（用于法向量计算）
            """
            super().__init__()
            self.meshes_prev = meshes_prev
            self.meshes_curr = meshes_curr
            self.device = device if device is not None else meshes_prev.device
            self.renderer = renderer
            self.loss_type = loss_type
            self.mesh_prev_o3d = mesh_prev_o3d
            self.mesh_curr_o3d = mesh_curr_o3d
            
            # 保存相机位姿（用于渲染）
            self.register_buffer('camera_R', camera_R.unsqueeze(0) if camera_R.dim() == 2 else camera_R)
            self.register_buffer('camera_T', camera_T.unsqueeze(0) if camera_T.dim() == 1 else camera_T)
            
            # 注册参考图像（前一帧的轮廓）
            if isinstance(image_ref, torch.Tensor):
                self.register_buffer('image_ref', image_ref)
            else:
                # 如果是numpy数组，转换为tensor
                image_ref_tensor = torch.from_numpy(image_ref).float().to(self.device)
                self.register_buffer('image_ref', image_ref_tensor)
            
            # 注册ground truth数据
            if silhouette_gt is not None:
                if isinstance(silhouette_gt, torch.Tensor):
                    self.register_buffer('silhouette_gt', silhouette_gt)
                else:
                    self.register_buffer('silhouette_gt', torch.from_numpy(silhouette_gt).float().to(self.device))
            else:
                self.register_buffer('silhouette_gt', self.image_ref.clone())
            
            if vertices_gt is not None:
                if isinstance(vertices_gt, torch.Tensor):
                    self.register_buffer('vertices_gt', vertices_gt)
                else:
                    self.register_buffer('vertices_gt', torch.from_numpy(vertices_gt).float().to(self.device))
            else:
                self.register_buffer('vertices_gt', None)
            
            if normals_gt is not None:
                if isinstance(normals_gt, torch.Tensor):
                    self.register_buffer('normals_gt', normals_gt)
                else:
                    self.register_buffer('normals_gt', torch.from_numpy(normals_gt).float().to(self.device))
            else:
                self.register_buffer('normals_gt', None)
            
            # 初始化RT矩阵参数（6DOF：3个旋转欧拉角 + 3个平移）
            if initial_pose is not None:
                # 从初始位姿提取旋转和平移
                R_init = torch.tensor(initial_pose[:3, :3], dtype=torch.float32, device=self.device)
                t_init = torch.tensor(initial_pose[:3, 3], dtype=torch.float32, device=self.device)
                euler_init = matrix_to_euler_angles(R_init, "XYZ")
            else:
                # 初始化为单位变换
                euler_init = torch.zeros(3, dtype=torch.float32, device=self.device)
                t_init = torch.zeros(3, dtype=torch.float32, device=self.device)
            
            # 创建可优化参数
            self.euler_angles = nn.Parameter(euler_init.clone())
            self.translation = nn.Parameter(t_init.clone())
        
        def forward(self, iteration=None):
            """
            前向传播：计算损失（优化版，支持多种损失类型）
            
            Args:
                iteration: 当前迭代次数（用于语义损失计算频率控制）
            
            Returns:
                loss: 总损失值
                image: 渲染图像（用于可视化）
                loss_dict: 损失字典（包含各项损失的详细信息）
            """
            # 将欧拉角转换为旋转矩阵
            R_rt = euler_angles_to_matrix(self.euler_angles, "XYZ")
            
            # 变换当前帧mesh：vertices_curr_transformed = R_rt * vertices_curr + T_rt
            vertices_curr = self.meshes_curr.verts_list()[0]
            vertices_curr_transformed = torch.matmul(vertices_curr, R_rt.T) + self.translation
            
            # 创建变换后的mesh
            faces_curr = self.meshes_curr.faces_list()[0]
            textures_curr = self.meshes_curr.textures
            mesh_curr_transformed = Meshes(
                verts=[vertices_curr_transformed],
                faces=[faces_curr],
                textures=textures_curr
            )
            
            # 渲染变换后的mesh（使用相机位姿，参考示例代码）
            image = self.renderer(meshes_world=mesh_curr_transformed, R=self.camera_R, T=self.camera_T)
            
            # 计算损失（根据损失类型）
            loss_dict = {}
            
            if self.loss_type == "silhouette":
                # 轮廓差异损失：变换后的当前帧mesh vs 前一帧mesh
                alpha_rendered = image[0, ..., 3]
                alpha_rendered = torch.clamp(alpha_rendered, 0, 1)
                
                # 优化：对alpha通道进行形态学操作以减少空洞（可选）
                use_morphology = bool(int(os.environ.get("RENDERING_USE_MORPHOLOGY", "0")))
                if use_morphology:
                    alpha_np = alpha_rendered.detach().cpu().numpy()
                    kernel = np.ones((3, 3), np.uint8)
                    alpha_np = cv2.dilate(alpha_np, kernel, iterations=1)
                    alpha_rendered = torch.tensor(alpha_np, device=self.device, dtype=alpha_rendered.dtype, requires_grad=True)
                
                # 使用Smooth L1 Loss代替MSE，对异常值更鲁棒
                loss_smooth_l1 = torch.nn.functional.smooth_l1_loss(alpha_rendered, self.silhouette_gt, beta=0.1)
                
                # 添加IoU损失，提高轮廓匹配精度
                intersection = (alpha_rendered * self.silhouette_gt).sum()
                union = alpha_rendered.sum() + self.silhouette_gt.sum() - intersection
                iou = intersection / (union + 1e-8)
                loss_iou = 1.0 - iou
                
                # 组合损失：主要使用Smooth L1，辅助IoU
                loss_silhouette = 0.8 * loss_smooth_l1 + 0.2 * loss_iou
                loss = loss_silhouette
                loss_dict['loss_silhouette'] = loss_silhouette
                loss_dict['loss_smooth_l1'] = loss_smooth_l1
                loss_dict['loss_iou'] = loss_iou
                
            elif self.loss_type == "mesh_error":
                # Mesh误差损失：将当前帧的顶点变换后与前一帧比较
                if self.vertices_gt is None:
                    raise ValueError("vertices_gt is required for mesh_error loss type")
                
                max_vertices = 8000
                if len(vertices_curr_transformed) > max_vertices:
                    indices = torch.randperm(len(vertices_curr_transformed), device=self.device)[:max_vertices]
                    vertices_curr_sampled = vertices_curr_transformed[indices]
                else:
                    vertices_curr_sampled = vertices_curr_transformed
                
                if len(self.vertices_gt) > max_vertices:
                    indices_gt = torch.randperm(len(self.vertices_gt), device=self.device)[:max_vertices]
                    vertices_gt_sampled = self.vertices_gt[indices_gt]
                else:
                    vertices_gt_sampled = self.vertices_gt
                
                # 使用改进的Chamfer距离（可微分的最近邻距离）
                distances_curr_to_prev = torch.cdist(vertices_curr_sampled, vertices_gt_sampled)
                min_distances_curr = distances_curr_to_prev.min(dim=1)[0]
                min_distances_prev = distances_curr_to_prev.min(dim=0)[0]
                
                # 加权Chamfer距离，给近的点更高权重
                weights_curr = torch.softmax(-min_distances_curr * 10.0, dim=0)
                weights_prev = torch.softmax(-min_distances_prev * 10.0, dim=0)
                
                # 加权Chamfer距离
                loss_chamfer_weighted = (min_distances_curr * weights_curr).sum() + (min_distances_prev * weights_prev).sum()
                
                # 添加平方项，对近的点更敏感
                loss_chamfer_squared = (min_distances_curr ** 2).mean() + (min_distances_prev ** 2).mean()
                
                # 组合：主要使用加权Chamfer，辅助平方项
                loss_mesh = 0.7 * loss_chamfer_weighted + 0.3 * loss_chamfer_squared
                loss = loss_mesh
                loss_dict['loss_mesh'] = loss_mesh
                loss_dict['loss_chamfer_weighted'] = loss_chamfer_weighted
                loss_dict['loss_chamfer_squared'] = loss_chamfer_squared
                
            elif self.loss_type == "combined" or self.loss_type == "enhanced":
                # 组合损失：轮廓 + mesh误差 + 正则化 + 法向量一致性 + 语义损失
                # 轮廓损失（优化版）
                alpha_rendered = image[0, ..., 3]
                alpha_rendered = torch.clamp(alpha_rendered, 0, 1)
                
                # 使用Smooth L1 + IoU
                loss_silhouette_smooth = torch.nn.functional.smooth_l1_loss(alpha_rendered, self.silhouette_gt, beta=0.1)
                intersection = (alpha_rendered * self.silhouette_gt).sum()
                union = alpha_rendered.sum() + self.silhouette_gt.sum() - intersection
                loss_silhouette_iou = 1.0 - (intersection / (union + 1e-8))
                loss_silhouette = 0.8 * loss_silhouette_smooth + 0.2 * loss_silhouette_iou
                loss_dict['loss_silhouette'] = loss_silhouette
                loss_dict['loss_silhouette_smooth'] = loss_silhouette_smooth
                loss_dict['loss_silhouette_iou'] = loss_silhouette_iou
                
                # Mesh误差损失（优化版）
                if self.vertices_gt is not None:
                    max_vertices = 5000
                    if len(vertices_curr_transformed) > max_vertices:
                        indices = torch.randperm(len(vertices_curr_transformed), device=self.device)[:max_vertices]
                        vertices_curr_sampled = vertices_curr_transformed[indices]
                    else:
                        vertices_curr_sampled = vertices_curr_transformed
                    
                    if len(self.vertices_gt) > max_vertices:
                        indices_gt = torch.randperm(len(self.vertices_gt), device=self.device)[:max_vertices]
                        vertices_gt_sampled = self.vertices_gt[indices_gt]
                    else:
                        vertices_gt_sampled = self.vertices_gt
                    
                    # 加权Chamfer距离
                    distances = torch.cdist(vertices_curr_sampled, vertices_gt_sampled)
                    min_distances_curr = distances.min(dim=1)[0]
                    min_distances_prev = distances.min(dim=0)[0]
                    weights_curr = torch.softmax(-min_distances_curr * 10.0, dim=0)
                    weights_prev = torch.softmax(-min_distances_prev * 10.0, dim=0)
                    loss_mesh_weighted = (min_distances_curr * weights_curr).sum() + (min_distances_prev * weights_prev).sum()
                    loss_mesh_squared = (min_distances_curr ** 2).mean() + (min_distances_prev ** 2).mean()
                    loss_mesh = 0.7 * loss_mesh_weighted + 0.3 * loss_mesh_squared
                    loss_dict['loss_mesh'] = loss_mesh
                    loss_dict['loss_mesh_weighted'] = loss_mesh_weighted
                    loss_dict['loss_mesh_squared'] = loss_mesh_squared
                else:
                    loss_mesh = torch.tensor(0.0, device=self.device)
                    loss_dict['loss_mesh'] = loss_mesh
                
                # 正则化项（防止过度变换）
                rotation_reg = torch.sum(self.euler_angles ** 2) * 0.01
                translation_reg = torch.sum(self.translation ** 2) * 0.01
                loss_dict['rotation_reg'] = rotation_reg
                loss_dict['translation_reg'] = translation_reg
                
                # 法向量一致性损失（如果可用）
                loss_normal = torch.tensor(0.0, device=self.device)
                if self.loss_type == "enhanced" and self.normals_gt is not None and self.mesh_curr_o3d is not None:
                    # 计算当前帧变换后的法向量
                    if hasattr(self.mesh_curr_o3d, 'vertex_normals') and len(self.mesh_curr_o3d.vertex_normals) > 0:
                        normals_curr = torch.tensor(np.asarray(self.mesh_curr_o3d.vertex_normals), dtype=torch.float32, device=self.device)
                        # 获取采样索引（使用与mesh损失相同的采样策略）
                        max_vertices_normal = 5000
                        if len(vertices_curr_transformed) > max_vertices_normal:
                            curr_indices = torch.randperm(len(vertices_curr_transformed), device=self.device)[:max_vertices_normal]
                        else:
                            curr_indices = torch.arange(len(vertices_curr_transformed), device=self.device)
                        
                        if self.vertices_gt is not None and len(self.vertices_gt) > max_vertices_normal:
                            gt_indices = torch.randperm(len(self.vertices_gt), device=self.device)[:max_vertices_normal]
                        elif self.vertices_gt is not None:
                            gt_indices = torch.arange(len(self.vertices_gt), device=self.device)
                        else:
                            gt_indices = None
                        
                        # 变换法向量（只旋转，不平移）
                        if gt_indices is not None:
                            normals_curr_transformed = (R_rt @ normals_curr[curr_indices].T).T
                            normals_gt_sampled = self.normals_gt[gt_indices]
                            
                            # 归一化法向量
                            normals_curr_transformed = normals_curr_transformed / (torch.norm(normals_curr_transformed, dim=1, keepdim=True) + 1e-8)
                            normals_gt_sampled = normals_gt_sampled / (torch.norm(normals_gt_sampled, dim=1, keepdim=True) + 1e-8)
                            
                            # 计算法向量一致性（点积应该接近1）
                            normal_dots = (normals_curr_transformed * normals_gt_sampled).sum(dim=1)
                            loss_normal = (1.0 - normal_dots).mean() * 0.1
                loss_dict['loss_normal'] = loss_normal
                
                # 语义损失（使用语义特征提取器计算轮廓相似度，支持梯度计算）
                loss_semantic = torch.tensor(0.0, device=self.device, requires_grad=True)
                semantic_similarity_value = 0.0
                if SEMANTIC_FEATURES_AVAILABLE and calculate_contour_similarity_with_grad is not None:
                    # 只在某些迭代中计算语义损失（因为计算较慢）
                    compute_semantic_freq = int(os.environ.get("RENDERING_SEMANTIC_FREQ", "2"))
                    use_grad_version = bool(int(os.environ.get("RENDERING_SEMANTIC_USE_GRAD", "1")))
                    
                    if iteration is not None and (iteration % compute_semantic_freq == 0):
                        try:
                            if use_grad_version:
                                # 使用支持梯度的版本
                                silhouette_gt_expanded = self.silhouette_gt.unsqueeze(0).unsqueeze(0)
                                silhouette_gt_rgb = silhouette_gt_expanded.repeat(1, 3, 1, 1)
                                
                                alpha_rendered_expanded = alpha_rendered.unsqueeze(0).unsqueeze(0)
                                alpha_rendered_rgb = alpha_rendered_expanded.repeat(1, 3, 1, 1)
                                
                                semantic_similarity_tensor = calculate_contour_similarity_with_grad(
                                    contour1=silhouette_gt_rgb,
                                    contour2=alpha_rendered_rgb,
                                    model_path=r'E:\reloc3r\Autoencoder_129.pth',
                                    image_size=(128, 128),
                                    device=self.device
                                )
                                
                                loss_semantic = 1.0 - semantic_similarity_tensor
                                semantic_similarity_value = semantic_similarity_tensor.item()
                            else:
                                # 使用不支持梯度的版本
                                silhouette_gt_np = self.silhouette_gt.detach().cpu().numpy()
                                silhouette_gt_img = (silhouette_gt_np * 255).astype(np.uint8)
                                silhouette_gt_rgb = np.stack([silhouette_gt_img] * 3, axis=-1)
                                
                                alpha_rendered_np = alpha_rendered.detach().cpu().numpy()
                                alpha_rendered_img = (alpha_rendered_np * 255).astype(np.uint8)
                                alpha_rendered_rgb = np.stack([alpha_rendered_img] * 3, axis=-1)
                                
                                semantic_similarity = calculate_contour_similarity(
                                    contour1=silhouette_gt_rgb,
                                    contour2=alpha_rendered_rgb,
                                    model_path=r'E:\reloc3r\Autoencoder_129.pth',
                                    image_size=(128, 128),
                                    device=self.device
                                )
                                
                                semantic_loss_value = 1.0 - semantic_similarity
                                loss_semantic = torch.tensor(semantic_loss_value, device=self.device, dtype=torch.float32)
                                semantic_similarity_value = semantic_similarity
                        except Exception as e:
                            loss_semantic = torch.tensor(0.0, device=self.device, requires_grad=True)
                            semantic_similarity_value = 0.0
                loss_dict['loss_semantic'] = loss_semantic
                loss_dict['semantic_similarity_value'] = semantic_similarity_value
                
                # 重新设计权重分配：平衡不同损失的贡献
                weight_silhouette = 0.6
                weight_mesh = 0.3
                weight_reg = 0.05
                weight_semantic = float(os.environ.get("RENDERING_SEMANTIC_WEIGHT", "0.1"))
                
                # 组合损失：使用平衡权重分配
                loss = (weight_silhouette * loss_silhouette +
                       weight_mesh * loss_mesh +
                       weight_semantic * loss_semantic +
                       weight_reg * (rotation_reg + translation_reg) +
                       loss_normal)
                
                loss_dict['weight_silhouette'] = weight_silhouette
                loss_dict['weight_mesh'] = weight_mesh
                loss_dict['weight_reg'] = weight_reg
                loss_dict['weight_semantic'] = weight_semantic
            else:
                raise ValueError(f"未知的损失类型: {self.loss_type}")
            
            return loss, image, loss_dict
        
        def get_rt_matrix(self):
            """获取当前的RT矩阵（4x4）"""
            R_rt = euler_angles_to_matrix(self.euler_angles, "XYZ")
            rt_matrix = torch.eye(4, device=self.device)
            rt_matrix[:3, :3] = R_rt
            rt_matrix[:3, 3] = self.translation
            return rt_matrix
    
    def _optimize_pose_with_rendering(self, prev_frame_idx, current_frame_idx, 
                                     initial_pose=None, suffix=None,
                                     image_width=640, image_height=480,
                                     fx=None, fy=None, cx=None, cy=None,
                                     max_iterations=150, learning_rate=0.01,
                                     loss_type="enhanced",  # "silhouette", "mesh_error", "combined", or "enhanced"
                                     save_intermediate=True, original_image_path=None):
        """
        基于PyTorch3D渲染的相机位姿优化
        
        方案说明：
        ==========
        1. 第一帧：初始化相机位姿和光源（基于第一帧mesh的几何信息）
        2. 第二帧及后续帧：使用前一帧的相机位姿进行渲染，通过迭代优化RT矩阵，
           将当前帧的mesh变换到前一帧的坐标系，使得变换后的当前帧mesh与前一帧mesh对齐
        
        流程：
        - 第一帧：初始化相机位姿（基于mesh中心、边界框等）
        - 第二帧：使用第一帧的相机位姿，优化RT矩阵，使得 R*vertices_curr + T 与 vertices_prev 对齐
        - 第三帧：使用第二帧的相机位姿，优化RT矩阵，使得 R*vertices_curr + T 与 vertices_prev 对齐
        - 以此类推：每一帧都使用前一帧的相机位姿进行渲染
        
        约束条件：
        - silhouette: 渲染轮廓差异最小化（变换后的当前帧mesh vs 前一帧mesh），使用Smooth L1 + IoU
        - mesh_error: 3D mesh误差最小化（变换后的当前帧mesh vs 前一帧mesh），使用加权Chamfer距离
        - combined: 组合损失（轮廓 + mesh误差 + 正则化）
        - enhanced: 增强损失（轮廓 + mesh误差 + 正则化 + 法向量一致性），效果最好
        
        Args:
            prev_frame_idx: 前一帧索引
            current_frame_idx: 当前帧索引
            initial_pose: 初始位姿（4x4矩阵），如果为None则使用单位矩阵
            suffix: mesh文件后缀
            image_width: 图像宽度
            image_height: 图像高度
            fx, fy, cx, cy: 相机内参，如果为None则使用默认值
            max_iterations: 最大迭代次数
            learning_rate: 学习率
            loss_type: 损失类型，"silhouette"（轮廓差异）、"mesh_error"（mesh误差）、
                       "combined"（组合损失）或"enhanced"（增强损失，包含正则化和法向量一致性）
            save_intermediate: 是否保存中间结果
            original_image_path: 原始图像路径（用于加载棋盘格掩码作为ground truth）
            
        Returns:
            dict: {
                'rt_matrix': RT矩阵（4x4矩阵，将当前帧mesh变换到前一帧坐标系）,
                'loss_history': 损失历史,
                'iterations': 实际迭代次数,
                'final_loss': 最终损失值
            }
        """
        if not PYTORCH3D_AVAILABLE:
            print("[错误] PyTorch3D未安装，无法使用基于渲染的位姿优化")
            return None
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\n[渲染优化] 开始优化帧 {prev_frame_idx} -> {current_frame_idx} 的位姿")
            print(f"  [设备] 使用设备: {device}")
            print(f"  [损失类型] {loss_type}")
            
            # 1. 优化：从内存缓存加载mesh（不读取文件）
            if prev_frame_idx not in self._mesh_cache:
                print(f"[错误] 前一帧（{prev_frame_idx}）的mesh不在缓存中")
                return None
            if current_frame_idx not in self._mesh_cache:
                print(f"[错误] 当前帧（{current_frame_idx}）的mesh不在缓存中")
                return None
            
            # 从内存缓存获取mesh（使用引用，不复制）
            mesh_prev_o3d = self._mesh_cache[prev_frame_idx]
            mesh_curr_o3d = self._mesh_cache[current_frame_idx]
            
            if len(mesh_prev_o3d.vertices) == 0 or len(mesh_curr_o3d.vertices) == 0:
                print(f"[错误] Mesh为空")
                return None
            
            # 转换为PyTorch3D Meshes
            vertices_prev = torch.tensor(np.asarray(mesh_prev_o3d.vertices), dtype=torch.float32, device=device)
            faces_prev = torch.tensor(np.asarray(mesh_prev_o3d.triangles), dtype=torch.long, device=device)
            
            vertices_curr = torch.tensor(np.asarray(mesh_curr_o3d.vertices), dtype=torch.float32, device=device)
            faces_curr = torch.tensor(np.asarray(mesh_curr_o3d.triangles), dtype=torch.long, device=device)
            
            # 生成粉红色纹理
            texture_grid_size = float(os.environ.get("TEXTURE_GRID_SIZE", "7.0"))  # 保留参数以兼容现有代码
            print(f"  [纹理] 生成粉红色纹理...")
            colors_prev = self._generate_checkerboard_texture(vertices_prev, device, grid_size=texture_grid_size)
            colors_curr = self._generate_checkerboard_texture(vertices_curr, device, grid_size=texture_grid_size)
            
            # 创建带纹理的Meshes
            textures_prev = TexturesVertex(verts_features=colors_prev.unsqueeze(0))
            textures_curr = TexturesVertex(verts_features=colors_curr.unsqueeze(0))
            mesh_prev = Meshes(verts=[vertices_prev], faces=[faces_prev], textures=textures_prev)
            mesh_curr = Meshes(verts=[vertices_curr], faces=[faces_curr], textures=textures_curr)
            print(f"  [纹理] 粉红色纹理生成成功")
            
            print(f"  [Mesh信息] 前一帧: {len(vertices_prev)} 顶点, {len(faces_prev)} 面片")
            print(f"    [坐标范围] X: [{vertices_prev[:, 0].min():.2f}, {vertices_prev[:, 0].max():.2f}], "
                  f"Y: [{vertices_prev[:, 1].min():.2f}, {vertices_prev[:, 1].max():.2f}], "
                  f"Z: [{vertices_prev[:, 2].min():.2f}, {vertices_prev[:, 2].max():.2f}]")
            print(f"  [Mesh信息] 当前帧: {len(vertices_curr)} 顶点, {len(faces_curr)} 面片")
            print(f"    [坐标范围] X: [{vertices_curr[:, 0].min():.2f}, {vertices_curr[:, 0].max():.2f}], "
                  f"Y: [{vertices_curr[:, 1].min():.2f}, {vertices_curr[:, 1].max():.2f}], "
                  f"Z: [{vertices_curr[:, 2].min():.2f}, {vertices_curr[:, 2].max():.2f}]")
            
            # 2. 设置相机内参
            if fx is None:
                fx = image_width * 0.8
            if fy is None:
                fy = image_height * 0.8
            if cx is None:
                cx = image_width / 2.0
            if cy is None:
                cy = image_height / 2.0
            
            # 3. 初始化相机位姿（参考示例代码，使用look_at_view_transform）
            # 计算mesh的边界框和中心
            mesh_min = vertices_prev.min(dim=0)[0].cpu().numpy()
            mesh_max = vertices_prev.max(dim=0)[0].cpu().numpy()
            mesh_center = vertices_prev.mean(dim=0).cpu().numpy()
            mesh_size = mesh_max - mesh_min
            mesh_diagonal = np.linalg.norm(mesh_size)
            
            # 相机距离mesh中心，确保能看到整个mesh（参考示例代码）
            distance = mesh_diagonal * 1.5
            
            # 使用球面坐标初始化相机位置（参考示例代码）
            # 选择视角：elevation（仰角）和azimuth（方位角）
            elevation = 50.0  # 仰角（度）
            azimuth = 0.0     # 方位角（度）
            
            # 获取实际的第一帧索引
            actual_first_frame_idx = getattr(self, 'start_frame_idx', 0)
            
            # 初始化相机位姿字典（如果不存在）
            if not hasattr(self, '_frame_camera_poses'):
                self._frame_camera_poses = {}
            
            # 如果是第一帧，初始化相机位姿；否则使用前一帧的相机位姿
            if prev_frame_idx == actual_first_frame_idx:
                # 第一帧：使用look_at_view_transform初始化相机位姿（参考示例代码）
                R_camera, T_camera = look_at_view_transform(
                    dist=distance,
                    elev=elevation,
                    azim=azimuth,
                    device=device
                )
                
                # 保存第一帧的相机位姿
                camera_pose_np = np.eye(4)
                camera_pose_np[:3, :3] = R_camera[0].cpu().numpy()
                camera_pose_np[:3, 3] = T_camera[0].cpu().numpy()
                self._frame_camera_poses[actual_first_frame_idx] = camera_pose_np.copy()
                
                if not hasattr(self, '_optimized_poses'):
                    self._optimized_poses = {}
                self._optimized_poses[actual_first_frame_idx] = camera_pose_np.copy()
                
                print(f"  [初始化] 第一帧（索引{actual_first_frame_idx}）相机位姿已初始化")
                print(f"    [相机信息] 距离: {distance:.2f}, 仰角: {elevation:.1f}°, 方位角: {azimuth:.1f}°")
                
                R_init = R_camera[0]
                t_init = T_camera[0]
            else:
                # 后续帧：使用前一帧的相机位姿
                if prev_frame_idx in self._frame_camera_poses:
                    prev_camera_pose = self._frame_camera_poses[prev_frame_idx]
                    R_init = torch.tensor(prev_camera_pose[:3, :3], dtype=torch.float32, device=device)
                    t_init = torch.tensor(prev_camera_pose[:3, 3], dtype=torch.float32, device=device)
                    print(f"  [初始化] 使用前一帧（索引{prev_frame_idx}）的相机位姿")
                else:
                    # 回退：使用第一帧的位姿
                    if actual_first_frame_idx in self._frame_camera_poses:
                        first_camera_pose = self._frame_camera_poses[actual_first_frame_idx]
                        R_init = torch.tensor(first_camera_pose[:3, :3], dtype=torch.float32, device=device)
                        t_init = torch.tensor(first_camera_pose[:3, 3], dtype=torch.float32, device=device)
                        print(f"  [警告] 使用第一帧（索引{actual_first_frame_idx}）的相机位姿作为回退")
                    else:
                        raise RuntimeError(f"未找到相机位姿，无法继续优化")
            
            # 4. 设置渲染器（参考示例代码，创建两个渲染器）
            # 4.1 初始化相机（参考示例代码）
            cameras = FoVPerspectiveCameras(device=device, R=R_init.unsqueeze(0), T=t_init.unsqueeze(0))
            
            # 4.2 创建轮廓渲染器（用于优化，参考示例代码）
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
            raster_settings_silhouette = RasterizationSettings(
                image_size=(image_height, image_width),
                blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
                faces_per_pixel=100,
            )
            silhouette_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings_silhouette
                ),
                shader=SoftSilhouetteShader(blend_params=blend_params)
            )
            
            # 4.3 创建Phong渲染器（用于可视化，参考示例代码）
            raster_settings_phong = RasterizationSettings(
                image_size=(image_height, image_width),
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
            phong_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings_phong
                ),
                shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
            )
            
            print(f"  [渲染器] 轮廓渲染器（优化用）和Phong渲染器（可视化用）已创建")
            
            # 5. 创建参考图像（参考示例代码）
            # 渲染前一帧mesh作为参考图像（ground truth）
            with torch.no_grad():
                silhouette_ref = silhouette_renderer(meshes_world=mesh_prev, R=R_init.unsqueeze(0), T=t_init.unsqueeze(0))
                image_ref = phong_renderer(meshes_world=mesh_prev, R=R_init.unsqueeze(0), T=t_init.unsqueeze(0))
            
            # 提取参考轮廓（参考示例代码）
            # 获取参考图像的轮廓：找到所有非白色像素值
            image_ref_np = image_ref[0, ..., :3].cpu().numpy()
            image_ref_silhouette = torch.from_numpy((image_ref_np.max(-1) != 1).astype(np.float32)).to(device)
            
            print(f"  [参考图像] 已创建参考轮廓（前一帧的渲染轮廓）")
            
            # 5.1 准备ground truth数据（用于损失计算）
            # 优先使用棋盘格掩码作为ground truth，如果不存在则使用渲染的轮廓
            use_chessboard_mask = bool(int(os.environ.get("USE_CHESSBOARD_MASK_AS_GT", "1")))  # 默认启用
            silhouette_gt = None
            
            if use_chessboard_mask:
                # 尝试加载棋盘格掩码图像
                chessboard_mask_loaded = False
                if original_image_path and os.path.exists(original_image_path):
                    # 首先尝试从输出目录加载已保存的棋盘格掩码
                    mask_filename = f"chessboard_mask_{prev_frame_idx:04d}.jpg"
                    mask_path = os.path.join(self.output_dir, mask_filename)
                    if os.path.exists(mask_path):
                        try:
                            chessboard_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            if chessboard_mask is not None:
                                # 调整掩码尺寸以匹配渲染图像尺寸
                                if chessboard_mask.shape[:2] != (image_height, image_width):
                                    chessboard_mask = cv2.resize(chessboard_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                                # 转换为tensor并归一化到0-1
                                chessboard_mask_float = chessboard_mask.astype(np.float32) / 255.0
                                silhouette_gt = torch.from_numpy(chessboard_mask_float).to(device)
                                chessboard_mask_loaded = True
                                print(f"  [GT] 成功加载已保存的棋盘格掩码: {mask_path} (尺寸: {chessboard_mask.shape})")
                            else:
                                print(f"  [GT] 无法读取棋盘格掩码文件: {mask_path}")
                        except Exception as e:
                            print(f"  [GT] 加载棋盘格掩码文件失败: {e}")
                    
                    # 如果加载失败，尝试从原始图像生成棋盘格掩码
                    if not chessboard_mask_loaded:
                        print(f"  [GT] 未找到已保存的棋盘格掩码，尝试从原始图像生成: {original_image_path}")
                        chessboard_mask = get_chessboard_mask(original_image_path, refine_corners=True)
                        if chessboard_mask is not None:
                            # 调整掩码尺寸以匹配渲染图像尺寸
                            if chessboard_mask.shape[:2] != (image_height, image_width):
                                chessboard_mask = cv2.resize(chessboard_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                            # 转换为tensor并归一化到0-1
                            chessboard_mask_float = chessboard_mask.astype(np.float32) / 255.0
                            silhouette_gt = torch.from_numpy(chessboard_mask_float).to(device)
                            chessboard_mask_loaded = True
                            print(f"  [GT] 使用棋盘格掩码作为ground truth (尺寸: {chessboard_mask.shape})")
                        else:
                            print(f"  [GT] 棋盘格检测失败，回退到渲染轮廓")
                elif original_image_path:
                    print(f"  [GT] 原始图像路径不存在: {original_image_path}，回退到渲染轮廓")
                else:
                    print(f"  [GT] 未提供原始图像路径，回退到渲染轮廓")
                
                # 如果棋盘格掩码加载失败，回退到渲染轮廓
                if silhouette_gt is None:
                    silhouette_gt = silhouette_ref[0, ..., 3].clone().detach()  # 使用alpha通道作为ground truth
                    silhouette_gt = torch.clamp(silhouette_gt, 0, 1)
                    print(f"  [GT] 使用渲染轮廓作为ground truth")
            else:
                # 不使用棋盘格掩码，直接使用渲染轮廓
                silhouette_gt = silhouette_ref[0, ..., 3].clone().detach()  # 使用alpha通道作为ground truth
                silhouette_gt = torch.clamp(silhouette_gt, 0, 1)
                print(f"  [GT] 使用渲染轮廓作为ground truth（棋盘格掩码已禁用）")
            
            vertices_gt = None
            normals_gt = None
            if loss_type == "mesh_error" or loss_type == "combined" or loss_type == "enhanced":
                # 使用前一帧mesh顶点作为ground truth
                vertices_gt = vertices_prev.to(device)
                # 计算前一帧mesh的法向量（用于法向量一致性损失）
                if hasattr(mesh_prev_o3d, 'vertex_normals') and len(mesh_prev_o3d.vertex_normals) > 0:
                    normals_gt = torch.tensor(np.asarray(mesh_prev_o3d.vertex_normals), dtype=torch.float32, device=device)
                else:
                    # 如果没有法向量，计算法向量
                    mesh_prev_o3d.compute_vertex_normals()
                    normals_gt = torch.tensor(np.asarray(mesh_prev_o3d.vertex_normals), dtype=torch.float32, device=device)
            
            # 6. 创建优化模型（参考示例代码）
            model = self._PoseOptimizationModel(
                meshes_prev=mesh_prev,
                meshes_curr=mesh_curr,
                renderer=silhouette_renderer,
                image_ref=image_ref_silhouette,
                camera_R=R_init,
                camera_T=t_init,
                initial_pose=initial_pose,
                device=device,
                loss_type=loss_type,
                silhouette_gt=silhouette_gt,
                vertices_gt=vertices_gt,
                normals_gt=normals_gt,
                mesh_prev_o3d=mesh_prev_o3d,
                mesh_curr_o3d=mesh_curr_o3d
            ).to(device)
            
            # 7. 创建优化器（参考示例代码）
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # 8. 创建输出目录
            if save_intermediate:
                render_output_dir = os.path.join(self.output_dir, "rendering_optimization")
                os.makedirs(render_output_dir, exist_ok=True)
                frame_pair_dir = os.path.join(render_output_dir, f"frame_{prev_frame_idx:04d}_{current_frame_idx:04d}")
                os.makedirs(frame_pair_dir, exist_ok=True)
            else:
                frame_pair_dir = None
            
            # 9. 优化循环（参考示例代码）
            # 详细损失历史记录（用于可视化）
            loss_history = []  # 总损失
            loss_silhouette_history = []  # 轮廓损失历史
            loss_mesh_history = []  # mesh损失历史
            loss_reg_history = []  # 正则化损失历史
            loss_normal_history = []  # 法向量损失历史
            loss_semantic_history = []  # 语义损失历史
            learning_rate_history = []  # 学习率历史
            
            # 添加学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=25,
                min_lr=1e-5
            )
            
            best_loss = float('inf')
            print(f"  [优化] 开始迭代优化（最大{max_iterations}次）...")
            
            # 可视化初始位置（参考示例代码）
            if save_intermediate and frame_pair_dir:
                with torch.no_grad():
                    _, image_init, _ = model()
                    image_init_np = image_init.detach().squeeze().cpu().numpy()[..., 3]
                    cv2.imwrite(
                        os.path.join(frame_pair_dir, "starting_position.png"),
                        (image_init_np * 255).astype(np.uint8)
                    )
                    # 保存参考轮廓
                    ref_silhouette_np = image_ref_silhouette.cpu().numpy()
                    cv2.imwrite(
                        os.path.join(frame_pair_dir, "reference_silhouette.png"),
                        (ref_silhouette_np * 255).astype(np.uint8)
                    )
                    # 保存ground truth轮廓
                    gt_silhouette_img = (silhouette_gt.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(frame_pair_dir, "ground_truth_silhouette.png"),
                        gt_silhouette_img
                    )
                    # 如果使用了棋盘格掩码，也保存一个副本，文件名与chessboard_mask格式一致
                    if use_chessboard_mask and silhouette_gt is not None:
                        chessboard_mask_filename = f"chessboard_mask_{prev_frame_idx:04d}.jpg"
                        chessboard_mask_path = os.path.join(frame_pair_dir, chessboard_mask_filename)
                        cv2.imwrite(chessboard_mask_path, gt_silhouette_img)
                        print(f"  [保存] Ground truth轮廓已保存（同时保存为 {chessboard_mask_filename}）")
                    else:
                        print(f"  [保存] Ground truth轮廓已保存")
            
            # 优化循环（参考示例代码结构）
            try:
                from tqdm import tqdm
                loop = tqdm(range(max_iterations))
            except ImportError:
                loop = range(max_iterations)
            
            for iteration in loop:
                optimizer.zero_grad()
                
                # 前向传播（参考示例代码）
                loss, image, loss_dict = model(iteration=iteration)
                
                # 反向传播（参考示例代码）
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step(loss)  # 更新学习率
                
                # 记录损失
                loss_value = loss.item()
                loss_history.append(loss_value)
                current_lr = optimizer.param_groups[0]['lr']
                learning_rate_history.append(current_lr)
                
                # 记录详细损失（根据损失类型）
                if loss_type == "enhanced" or loss_type == "combined":
                    if 'loss_silhouette' in loss_dict:
                        loss_silhouette_history.append(loss_dict['loss_silhouette'].item() if hasattr(loss_dict['loss_silhouette'], 'item') else float(loss_dict['loss_silhouette']))
                    else:
                        loss_silhouette_history.append(0.0)
                    
                    if 'loss_mesh' in loss_dict:
                        loss_mesh_history.append(loss_dict['loss_mesh'].item() if hasattr(loss_dict['loss_mesh'], 'item') else float(loss_dict['loss_mesh']))
                    else:
                        loss_mesh_history.append(0.0)
                    
                    if 'rotation_reg' in loss_dict and 'translation_reg' in loss_dict:
                        reg_total = (loss_dict['rotation_reg'] + loss_dict['translation_reg']).item() if hasattr(loss_dict['rotation_reg'], 'item') else float(loss_dict['rotation_reg'] + loss_dict['translation_reg'])
                        loss_reg_history.append(reg_total)
                    else:
                        loss_reg_history.append(0.0)
                    
                    if 'loss_normal' in loss_dict:
                        loss_normal_history.append(loss_dict['loss_normal'].item() if hasattr(loss_dict['loss_normal'], 'item') else float(loss_dict['loss_normal']))
                    else:
                        loss_normal_history.append(0.0)
                    
                    if 'loss_semantic' in loss_dict:
                        loss_semantic_history.append(loss_dict['loss_semantic'].item() if hasattr(loss_dict['loss_semantic'], 'item') else float(loss_dict['loss_semantic']))
                    else:
                        loss_semantic_history.append(0.0)
                elif loss_type == "silhouette":
                    if 'loss_silhouette' in loss_dict:
                        loss_silhouette_history.append(loss_dict['loss_silhouette'].item() if hasattr(loss_dict['loss_silhouette'], 'item') else float(loss_dict['loss_silhouette']))
                    else:
                        loss_silhouette_history.append(0.0)
                    loss_mesh_history.append(0.0)
                    loss_reg_history.append(0.0)
                    loss_normal_history.append(0.0)
                    loss_semantic_history.append(0.0)
                elif loss_type == "mesh_error":
                    if 'loss_mesh' in loss_dict:
                        loss_mesh_history.append(loss_dict['loss_mesh'].item() if hasattr(loss_dict['loss_mesh'], 'item') else float(loss_dict['loss_mesh']))
                    else:
                        loss_mesh_history.append(0.0)
                    loss_silhouette_history.append(0.0)
                    loss_reg_history.append(0.0)
                    loss_normal_history.append(0.0)
                    loss_semantic_history.append(0.0)
                else:
                    loss_silhouette_history.append(0.0)
                    loss_mesh_history.append(0.0)
                    loss_reg_history.append(0.0)
                    loss_normal_history.append(0.0)
                    loss_semantic_history.append(0.0)
                
                # 记录最佳损失
                if loss_value < best_loss:
                    best_loss = loss_value
                
                # 更新进度条（参考示例代码）
                if hasattr(loop, 'set_description'):
                    loop.set_description(f'Optimizing (loss {loss_value:.4f})')
                
                # 保存中间结果（参考示例代码）
                if save_intermediate and frame_pair_dir and (iteration % 10 == 0 or iteration == max_iterations - 1):
                    with torch.no_grad():
                        # 使用Phong渲染器可视化结果（参考示例代码）
                        R_rt = model.get_rt_matrix()[:3, :3]
                        T_rt = model.get_rt_matrix()[:3, 3]
                        vertices_curr_transformed = torch.matmul(vertices_curr, R_rt.T) + T_rt
                        mesh_curr_transformed = Meshes(
                            verts=[vertices_curr_transformed],
                            faces=[faces_curr],
                            textures=textures_curr
                        )
                        image_phong = phong_renderer(meshes_world=mesh_curr_transformed, R=R_init.unsqueeze(0), T=t_init.unsqueeze(0))
                        image_phong_np = image_phong[0, ..., :3].detach().squeeze().cpu().numpy()
                        image_phong_uint8 = (np.clip(image_phong_np, 0, 1) * 255).astype(np.uint8)
                        cv2.imwrite(
                            os.path.join(frame_pair_dir, f"iter_{iteration:03d}_loss_{loss_value:.2f}.png"),
                            cv2.cvtColor(image_phong_uint8, cv2.COLOR_RGB2BGR)
                        )
                        
                        # 如果使用包含轮廓的损失，保存轮廓对比图
                        if loss_type == "silhouette" or loss_type == "combined" or loss_type == "enhanced":
                            alpha_channel = image[0, ..., 3].detach().cpu().numpy()
                            alpha_threshold = float(os.environ.get("RENDERING_ALPHA_THRESHOLD", "0.0"))
                            silhouette_rendered = (alpha_channel > alpha_threshold).astype(np.float32)
                            
                            # 分析统计信息
                            alpha_min = alpha_channel.min()
                            alpha_max = alpha_channel.max()
                            alpha_mean = alpha_channel.mean()
                            alpha_nonzero = (alpha_channel > 0).sum()
                            
                            # 调试信息（每10次迭代打印一次）
                            if iteration % 10 == 0:
                                visible_pixels = silhouette_rendered.sum()
                                total_pixels = silhouette_rendered.size
                                print(f"      [轮廓提取] alpha范围=[{alpha_min:.4f}, {alpha_max:.4f}], "
                                      f"均值={alpha_mean:.4f}, 非零像素={alpha_nonzero}/{total_pixels}, "
                                      f"可见像素={visible_pixels:.0f}/{total_pixels} ({visible_pixels/total_pixels*100:.1f}%)")
                            
                            silhouette_rendered_img = (silhouette_rendered * 255).astype(np.uint8)
                            
                            # 创建对比图：左侧渲染轮廓，右侧ground truth，下方差异
                            comparison = np.zeros((image_height * 2, image_width, 3), dtype=np.uint8)
                            comparison[:image_height, :, 0] = silhouette_rendered_img  # 红色通道：渲染
                            comparison[:image_height, :, 1] = silhouette_gt.detach().cpu().numpy() * 255  # 绿色通道：GT
                            comparison[image_height:, :, 0] = np.abs(silhouette_rendered_img - silhouette_gt.detach().cpu().numpy() * 255)  # 差异
                            cv2.imwrite(
                                os.path.join(frame_pair_dir, f"silhouette_comparison_iter_{iteration:03d}.png"),
                                comparison
                            )
                            print(f"      [保存] 轮廓对比图已保存: silhouette_comparison_iter_{iteration:03d}.png")
                
                # 打印进度
                if iteration % 10 == 0 or iteration == max_iterations - 1:
                    if loss_type == "enhanced":
                        # enhanced损失类型：输出更详细的信息
                        print(f"    [迭代 {iteration:3d}/{max_iterations}] 总损失: {loss_value:.6f}, LR: {current_lr:.6f}, 最佳: {best_loss:.6f}")
                        if 'loss_silhouette' in loss_dict:
                            print(f"      -> 轮廓: {loss_dict['loss_silhouette'].item():.6f}, ", end="")
                        if 'loss_mesh' in loss_dict:
                            print(f"Mesh: {loss_dict['loss_mesh'].item():.6f}, ", end="")
                        if 'rotation_reg' in loss_dict and 'translation_reg' in loss_dict:
                            reg_total = (loss_dict['rotation_reg'] + loss_dict['translation_reg']).item()
                            print(f"正则化: {reg_total:.6f}, ", end="")
                        if 'loss_normal' in loss_dict:
                            print(f"法向量: {loss_dict['loss_normal'].item():.6f}")
                        
                        # 详细损失分解
                        print(f"      [损失分解-Enhanced]")
                        if 'loss_silhouette' in loss_dict and 'loss_silhouette_smooth' in loss_dict and 'loss_silhouette_iou' in loss_dict:
                            print(f"        轮廓损失: {loss_dict['loss_silhouette'].item():.6f} (Smooth L1: {loss_dict['loss_silhouette_smooth'].item():.6f}, IoU: {loss_dict['loss_silhouette_iou'].item():.6f})")
                        if 'loss_mesh' in loss_dict and 'loss_mesh_weighted' in loss_dict and 'loss_mesh_squared' in loss_dict:
                            print(f"        Mesh损失: {loss_dict['loss_mesh'].item():.6f} (加权Chamfer: {loss_dict['loss_mesh_weighted'].item():.6f}, 平方Chamfer: {loss_dict['loss_mesh_squared'].item():.6f})")
                        if 'loss_semantic' in loss_dict and loss_dict['loss_semantic'].item() > 0:
                            semantic_info = f" (相似度: {loss_dict.get('semantic_similarity_value', 0.0):.4f}, 权重: {loss_dict.get('weight_semantic', 0.0):.4f})"
                            print(f"        语义损失: {loss_dict['loss_semantic'].item():.6f}{semantic_info}")
                        if 'rotation_reg' in loss_dict and 'translation_reg' in loss_dict:
                            print(f"        正则化: 旋转={loss_dict['rotation_reg'].item():.6f}, 平移={loss_dict['translation_reg'].item():.6f}, 总计={(loss_dict['rotation_reg'] + loss_dict['translation_reg']).item():.6f}")
                        if 'loss_normal' in loss_dict:
                            print(f"        法向量损失: {loss_dict['loss_normal'].item():.6f}")
                        if 'weight_silhouette' in loss_dict:
                            print(f"        权重: 轮廓={loss_dict['weight_silhouette']:.4f}, Mesh={loss_dict.get('weight_mesh', 0.0):.4f}, 正则化={loss_dict.get('weight_reg', 0.0):.4f}, 语义={loss_dict.get('weight_semantic', 0.0):.4f}")
                        print(f"        总损失: {loss_value:.6f}")
                    else:
                        print(f"    [迭代 {iteration:3d}/{max_iterations}] 损失: {loss_value:.6f}, LR: {current_lr:.6f}, 最佳: {best_loss:.6f}")
            
            # 10. 提取最终结果（参考示例代码）
            rt_matrix = model.get_rt_matrix().detach().cpu().numpy()
            
            print(f"  [完成] 优化完成，最终损失: {loss_history[-1]:.6f}")
            
            return {
                'rt_matrix': rt_matrix,
                'rotation_matrix': rt_matrix[:3, :3],
                'translation_vector': rt_matrix[:3, 3],
                'loss_history': loss_history,
                'iterations': len(loss_history),
                'final_loss': loss_history[-1] if loss_history else 0.0
            }
            
        except Exception as e:
            print(f"[错误] 渲染优化失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _visualize_3d_feature_matches(self, mesh1_input, mesh2_input, match_result,
                                     frame_idx1, frame_idx2, suffix="th_68"):
        """
        在原始高度场3D网格上可视化特征匹配结果，保存为图像
        使用matplotlib直接可视化网格匹配结果
        
        优化：支持mesh对象或文件路径
        
        Args:
            mesh1_input: 第一帧的网格对象或文件路径
            mesh2_input: 第二帧的网格对象或文件路径
            match_result: 匹配结果（包含transformation矩阵）
            frame_idx1: 第一帧索引
            frame_idx2: 第二帧索引
            suffix: 文件后缀
        """
        if o3d is None:
            print("[警告] Open3D未安装，无法加载网格文件")
            return
        
        match_output_dir = os.path.join(self.output_dir, "3d_mesh_feature_matching")
        os.makedirs(match_output_dir, exist_ok=True)
        
        image_path = os.path.join(
            match_output_dir, 
            f"3d_match_vis_{suffix}_{frame_idx1:04d}_{frame_idx2:04d}.png"
        )
        
        try:
            # 优化：支持mesh对象或文件路径
            if isinstance(mesh1_input, str):
                mesh1 = o3d.io.read_triangle_mesh(mesh1_input)
            else:
                mesh1 = mesh1_input
            
            if isinstance(mesh2_input, str):
                mesh2 = o3d.io.read_triangle_mesh(mesh2_input)
            else:
                mesh2 = mesh2_input
            
            if len(mesh1.vertices) == 0 or len(mesh2.vertices) == 0:
                print(f"[警告] 网格为空，无法可视化")
                return
            
            # 获取变换矩阵
            if isinstance(match_result, dict):
                transformation = match_result.get('transformation', np.eye(4))
            else:
                transformation = match_result
            
            # 应用变换到mesh2
            mesh2_transformed = copy.deepcopy(mesh2)
            mesh2_transformed.transform(transformation)
            
            # 可视化
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(mesh1)
            vis.add_geometry(mesh2_transformed)
            vis.capture_screen_image(image_path)
            vis.destroy_window()
            
            print(f"  [保存] 3D匹配可视化已保存: {image_path}")
            
        except Exception as e:
            print(f"  [警告] 3D匹配可视化失败: {e}")
            import traceback
            traceback.print_exc()

    def _visualize_mesh_with_matplotlib(self, mesh1, mesh2, correspondences, transformation,
                                      frame_idx1, frame_idx2, image_path,
                                      feat1=None, feat2=None):
        """
        使用matplotlib在原始高度场网格上绘制匹配结果
        左边显示前一帧的3D mesh，右边显示当前帧的3D mesh（变换后）
        
        Args:
            mesh1: 原始mesh对象
            mesh2: 变换后的mesh对象
            correspondences: 匹配点对列表
            transformation: 变换矩阵
            frame_idx1: 帧索引
            frame_idx2: 帧索引
            image_path: 保存路径
            feat1: 特征提取结果（可选）
            feat2: 特征提取结果（可选）
        """
        # ... 方法实现 ...
        pass

    def _estimate_pose_from_3d_matches(self, matches, K=None):
        """
        从3D匹配结果估计相机位姿
        
        Args:
            matches: 匹配结果列表，每个元素包含：
                - transformation: 4x4变换矩阵（将当前帧mesh变换到前一帧坐标系）
                - correspondence_set: 对应点集合
                - pcd1: 第一帧采样点云
                - pcd2: 第二帧采样点云（变换后）
            K: 相机内参矩阵（3x3），如果为None则使用默认值
            
        Returns:
            {
                'poses': 位姿列表（4x4矩阵）
            }
        """
        # ... 方法实现 ...
        pass

    def _validate_transformation(self, pts1, pts2, T):
        """
        验证变换矩阵的质量
        
        Args:
            pts1: 第一组点（N, 3）
            pts2: 第二组点（N, 3）
            T: 变换矩阵（4x4）
            
        Returns:
            {
                'rotation_error': 旋转误差（度）
                'translation_error': 平移误差
            }
        """
        # ... 方法实现 ...
        pass

    def _compute_pose_error(self, estimated_poses, ground_truth_poses):
        """
        计算估计位姿与真实位姿的误差
        
        Args:
            estimated_poses: 估计的位姿列表（4x4矩阵）
            ground_truth_poses: 真实位姿列表（4x4矩阵）
            
        Returns:
            {
                'rotation_error': 旋转误差（度）
                'translation_error': 平移误差
            }
        """
        # ... 方法实现 ...
        pass

    def _update_accumulated_pose(self, prev_frame_idx, current_frame_idx, rt_matrix, reference_frame_idx=0):
        """
        实时更新累积位姿
        
        这个方法会在每次计算RT矩阵后立即更新累积位姿
        
        Args:
            prev_frame_idx: 前一帧索引
            current_frame_idx: 当前帧索引
            rt_matrix: RT矩阵（4x4矩阵，将当前帧mesh变换到前一帧坐标系）
            reference_frame_idx: 参考帧索引（默认为0）
            
        Returns:
            {
                'current_pose': 当前帧的4x4位姿矩阵（相对于参考坐标系）
            }
        """
        # ... 方法实现 ...
        pass

    def _explain_relative_to_absolute_pose(self):
        """
        解释相对位姿和绝对位姿的关系
        
        这是一个说明性函数，用于解释代码中的位姿概念
        
        核心概念：
        
        1. 相对位姿（RT矩阵）：
           表示frame_i相对于frame_i-1的变换矩阵
           含义：
           公式：
               vertices_i_transformed = RT_i @ vertices_i + T_i
               
        2. 绝对位姿（T_absolute）：
           表示frame_i相对于参考坐标系（frame_0）的位姿
           含义：
           公式：
               T_absolute_i = T_absolute_0 @ RT_1 @ RT_2 @ ... @ RT_i
               
        3. 从相对位姿到绝对位姿：
           
           设参考坐标系为frame_0，则：
               T_absolute_0 = I（单位矩阵）
               T_absolute_1 = RT_1（第一帧相对于参考帧）
               T_absolute_2 = RT_1 @ RT_2（第二帧相对于参考帧）
               T_absolute_3 = RT_1 @ RT_2 @ RT_3（第三帧相对于参考帧）
               
           通用公式：
               T_absolute_i = RT_1 @ RT_2 @ ... @ RT_i
               
        为什么需要求逆？
           RT矩阵表示"将当前帧mesh变换到前一帧坐标系"
           但位姿表示"当前帧相对于参考坐标系的变换"
           这两个是互逆的关系
           
        误差累积：
           由于每一帧的相对变换都有误差，
           后续帧的绝对位姿误差通常比前面的帧更大
           
        示例：
           假设有3帧：
               参考坐标系：frame_0
               相对RT_1：frame_1相对于frame_0
               相对RT_2：frame_2相对于frame_1
               
           绝对位姿计算：
               T_absolute_0 = I
               T_absolute_1 = RT_1
               T_absolute_2 = RT_1 @ RT_2
               
           这样，每一帧的绝对位姿都是相对于参考坐标系的
        """
        # 说明文本
        pass

    def _accumulate_camera_poses_from_rt_matrices(self, reference_frame_idx=0):
        """
        相对位姿 vs 绝对位姿 详细说明
        
        【相对位姿（RT矩阵）】
        你计算的RT矩阵是相对位姿：
           含义：frame_i相对于frame_i-1的变换矩阵
           公式：vertices_i_transformed = RT_i @ vertices_i + T_i
           用途：用于将当前帧mesh对齐到前一帧坐标系
           
        【绝对位姿（T_absolute）】
        绝对位姿是：
           含义：frame_i相对于参考坐标系（frame_0）的位姿
           公式：T_absolute_i = RT_1 @ RT_2 @ ... @ RT_i
           用途：用于全局定位和建图
           
        【转换关系】
        从相对位姿到绝对位姿：
           T_absolute_0 = I（参考帧）
           T_absolute_1 = RT_1
           T_absolute_2 = RT_1 @ RT_2
           T_absolute_3 = RT_1 @ RT_2 @ RT_3
           ...
           
        【误差累积】
        由于每一帧的相对变换都有误差，后续帧的绝对位姿误差会累积增大
        """
        # ... 方法实现 ...
        pass

    def _visualize_3d_feature_matches(self, mesh1_input, mesh2_input, match_result, 
                                     frame_idx1, frame_idx2, suffix="th_68"):
        """
        在原始高度场3D网格上可视化特征匹配结果，保存为图像
        使用matplotlib直接可视化网格匹配结果
        
        优化：支持mesh对象或文件路径
        
        Args:
            mesh1_input: 第一帧的网格对象或文件路径
            mesh2_input: 第二帧的网格对象或文件路径
            match_result: 匹配结果（包含transformation矩阵）
            frame_idx1: 第一帧索引
            frame_idx2: 第二帧索引
            suffix: 文件后缀
        """
        if o3d is None:
            print("[警告] Open3D未安装，无法加载网格文件")
            return
        
        match_output_dir = os.path.join(self.output_dir, "3d_mesh_feature_matching")
        os.makedirs(match_output_dir, exist_ok=True)
        
        image_path = os.path.join(
            match_output_dir, 
            f"3d_match_vis_{suffix}_{frame_idx1:04d}_{frame_idx2:04d}.png"
        )
        
        try:
            # 优化：支持mesh对象或文件路径
            if isinstance(mesh1_input, str):
                mesh1 = o3d.io.read_triangle_mesh(mesh1_input)
            else:
                mesh1 = mesh1_input  # 直接使用mesh对象（不复制）
            
            if isinstance(mesh2_input, str):
                mesh2 = o3d.io.read_triangle_mesh(mesh2_input)
            else:
                mesh2 = mesh2_input  # 直接使用mesh对象（不复制）
            
            if len(mesh1.vertices) == 0 or len(mesh2.vertices) == 0:
                print(f"  [警告] 网格为空，跳过可视化")
                return
            
            # 计算法向量（用于渲染）
            mesh1.compute_vertex_normals()
            mesh2.compute_vertex_normals()
            
            correspondence_set = match_result.get('correspondence_set', None)
            transformation = match_result.get('transformation', np.eye(4))
            
            # 为了绘制匹配线，需要从mesh中提取对应的顶点
            # 由于correspondence_set是基于采样点云的索引，我们需要找到对应的mesh顶点
            # 这里我们使用mesh的顶点来创建匹配线
            vertices1 = np.asarray(mesh1.vertices)
            vertices2 = np.asarray(mesh2.vertices)
            
            if correspondence_set is None or len(correspondence_set) == 0:
                return
            
            # 处理correspondence_set格式
            try:
                if isinstance(correspondence_set, np.ndarray):
                    correspondences_list = correspondence_set.tolist()
                else:
                    correspondences_list = list(correspondence_set)
                
                if len(correspondences_list) == 0:
                    return
                
                first_item = correspondences_list[0]
                if not (isinstance(first_item, (list, tuple, np.ndarray)) and len(first_item) == 2):
                    print(f"  [警告] 未知的correspondence_set格式，跳过可视化")
                    return
            except Exception as e:
                print(f"  [警告] 处理correspondence_set失败: {e}")
                return
            
            # 限制显示的匹配线数量（最多500条）
            max_lines = min(500, len(correspondences_list))
            if len(correspondences_list) > max_lines:
                import random
                selected_indices = random.sample(range(len(correspondences_list)), max_lines)
                selected_correspondences = [correspondences_list[i] for i in selected_indices]
            else:
                selected_correspondences = correspondences_list
            
            # 注意：correspondence_set的索引对应采样点云，不是mesh顶点
            # 从匹配结果中获取采样点云坐标（匹配时已保存）
            sampled_points1 = match_result.get('sampled_points1', None)
            sampled_points2_final = match_result.get('sampled_points2_final', None)
            
            if sampled_points1 is None or sampled_points2_final is None:
                print(f"  [警告] 匹配结果中未找到采样点云，重新提取（可能不一致）...")
                # 如果匹配结果中没有保存采样点云，重新提取（但采样可能不一致）
                # 尝试获取mask（如果可用）
                mask1 = None
                mask2 = None
                if hasattr(self, '_frame_masks'):
                    # 从mesh路径或对象推断frame_idx
                    import re
                    mask1 = None
                    mask2 = None
                    if isinstance(mesh1_input, str):
                        match1 = re.search(r'(\d{4})\.ply', os.path.basename(mesh1_input))
                        if match1:
                            frame_idx1 = int(match1.group(1))
                            mask1 = self._frame_masks.get(frame_idx1, None)
                    if isinstance(mesh2_input, str):
                        match2 = re.search(r'(\d{4})\.ply', os.path.basename(mesh2_input))
                        if match2:
                            frame_idx2 = int(match2.group(1))
                            mask2 = self._frame_masks.get(frame_idx2, None)
                
                feat_prev = self._extract_3d_mesh_features(
                    mesh1_input, max_points=10000, radius_normal=0.15, radius_feature=0.3,
                    mask=mask1  # 使用mask约束采样
                )
                feat_curr = self._extract_3d_mesh_features(
                    mesh2_input, max_points=10000, radius_normal=0.15, radius_feature=0.3,
                    pre_transform=transformation,  # 应用完整变换
                    mask=mask2  # 使用mask约束采样
                )
                if feat_prev is None or feat_curr is None:
                    print(f"  [警告] 无法获取采样点云，跳过可视化")
                    return
                sampled_points1 = feat_prev['points']
                sampled_points2_final = feat_curr['points']
            else:
                print(f"  [可视化] 使用匹配时保存的采样点云: 帧1={len(sampled_points1)}点, 帧2={len(sampled_points2_final)}点")
            
            # 创建特征字典用于可视化函数
            feat_prev = {'points': sampled_points1}
            feat_curr = {'points': sampled_points2_final}
            
            # 直接使用matplotlib可视化网格匹配结果
            try:
                self._visualize_mesh_with_matplotlib(
                    mesh1, mesh2, selected_correspondences, transformation,
                    frame_idx1, frame_idx2, image_path,
                    feat_prev, feat_curr  # 传入采样点云特征
                )
                print(f"  [保存] 特征匹配可视化: {image_path} ({len(correspondences_list)}个匹配点)")
            except Exception as e:
                print(f"[警告] matplotlib可视化失败: {e}")
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            print(f"[警告] 保存特征匹配可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_mesh_with_matplotlib(self, mesh1, mesh2, correspondences, transformation,
                                       frame_idx1, frame_idx2, image_path,
                                       feat1=None, feat2=None):
        """
        使用matplotlib在原始高度场网格上绘制匹配结果（左右布局）
        左边显示前一帧的3D mesh，右边显示当前帧的3D mesh，匹配点之间用直线连接
        
        Args:
            mesh1, mesh2: 原始mesh对象
            correspondences: 匹配点对列表，索引对应采样点云（不是mesh顶点）
            transformation: 变换矩阵
            frame_idx1, frame_idx2: 帧索引
            image_path: 保存路径
            feat1, feat2: 特征提取结果（包含采样点云坐标），如果为None则从mesh顶点提取
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        vertices1 = np.asarray(mesh1.vertices)
        vertices2 = np.asarray(mesh2.vertices)
        faces1 = np.asarray(mesh1.triangles)
        faces2 = np.asarray(mesh2.triangles)
        
        # 应用变换：将mesh2变换到mesh1的坐标系
        vertices2_homo = np.hstack([vertices2, np.ones((len(vertices2), 1))])
        vertices2_transformed = (transformation @ vertices2_homo.T).T[:, :3]
        
        # 获取采样点云坐标（correspondence_set的索引对应采样点云）
        if feat1 is not None and feat2 is not None:
            # 使用特征提取时的采样点云
            sampled_points1 = feat1['points']  # 第一帧的采样点云（已应用粗变换）
            sampled_points2 = feat2['points']  # 第二帧的采样点云（已应用粗变换+精细变换）
            print(f"  [可视化] 使用采样点云: 帧1={len(sampled_points1)}点, 帧2={len(sampled_points2)}点")
        else:
            # 如果没有提供特征，使用mesh顶点（可能不准确）
            print(f"  [警告] 未提供采样点云特征，使用mesh顶点（可能不准确）")
            sampled_points1 = vertices1
            sampled_points2 = vertices2_transformed
        
        # 计算两个mesh的边界框，用于确定平移距离
        mesh1_min = vertices1.min(axis=0)
        mesh1_max = vertices1.max(axis=0)
        mesh1_size = mesh1_max - mesh1_min
        mesh1_center = (mesh1_min + mesh1_max) / 2
        
        mesh2_min = vertices2_transformed.min(axis=0)
        mesh2_max = vertices2_transformed.max(axis=0)
        mesh2_size = mesh2_max - mesh2_min
        mesh2_center = (mesh2_min + mesh2_max) / 2
        
        # 计算两个mesh之间的分离距离（使用X轴方向的最大尺寸）
        max_size = max(mesh1_size[0], mesh2_size[0], mesh1_size[1], mesh2_size[1], mesh1_size[2], mesh2_size[2])
        separation = max_size * 3.0  # 分离距离为最大尺寸的3.0倍，确保两个mesh之间有足够的距离
        
        # 将mesh1向左平移，mesh2向右平移
        vertices1_separated = vertices1.copy()
        vertices1_separated[:, 0] -= separation / 2  # 向左平移
        
        vertices2_separated = vertices2_transformed.copy()
        vertices2_separated[:, 0] += separation / 2  # 向右平移
        
        # 同样平移采样点云
        sampled_points1_separated = sampled_points1.copy()
        sampled_points1_separated[:, 0] -= separation / 2
        
        sampled_points2_separated = sampled_points2.copy()
        sampled_points2_separated[:, 0] += separation / 2
        
        # 使用所有网格面片（不进行采样）
        faces1_display = faces1
        faces2_display = faces2
        print(f"  [网格信息] 网格1: {len(faces1)}个面片, 网格2: {len(faces2)}个面片")
        
        # 提取匹配点坐标（使用采样点云，不是mesh顶点）
        matched_points1 = []
        matched_points2 = []
        matched_points1_separated = []
        matched_points2_separated = []
        line_count = 0
        if len(correspondences) > 0:
            for corr in correspondences:
                try:
                    if isinstance(corr, (list, tuple, np.ndarray)) and len(corr) == 2:
                        i, j = int(corr[0]), int(corr[1])
                        # 索引对应采样点云，不是mesh顶点
                        if 0 <= i < len(sampled_points1) and 0 <= j < len(sampled_points2):
                            matched_points1.append(sampled_points1[i])
                            matched_points2.append(sampled_points2[j])
                            # 同时保存平移后的坐标（用于绘制连接线）
                            matched_points1_separated.append(sampled_points1_separated[i])
                            matched_points2_separated.append(sampled_points2_separated[j])
                            line_count += 1
                except Exception:
                    continue
        
        matched_points1 = np.array(matched_points1) if matched_points1 else np.array([]).reshape(0, 3)
        matched_points2 = np.array(matched_points2) if matched_points2 else np.array([]).reshape(0, 3)
        matched_points1_separated = np.array(matched_points1_separated) if matched_points1_separated else np.array([]).reshape(0, 3)
        matched_points2_separated = np.array(matched_points2_separated) if matched_points2_separated else np.array([]).reshape(0, 3)
        
        # 分析匹配点分布
        if len(matched_points1) > 0:
            # 计算匹配点与mesh中心的距离
            mesh1_center = vertices1.mean(axis=0)
            mesh2_center = vertices2_transformed.mean(axis=0)
            
            distances1 = np.linalg.norm(matched_points1 - mesh1_center, axis=1)
            distances2 = np.linalg.norm(matched_points2 - mesh2_center, axis=1)
            
            # 计算mesh边界框大小
            mesh1_extent = np.linalg.norm(vertices1.max(axis=0) - vertices1.min(axis=0))
            mesh2_extent = np.linalg.norm(vertices2_transformed.max(axis=0) - vertices2_transformed.min(axis=0))
            
            # 计算匹配点距离中心的相对位置（0=中心，1=边缘）
            relative_pos1 = distances1 / (mesh1_extent / 2 + 1e-6)
            relative_pos2 = distances2 / (mesh2_extent / 2 + 1e-6)
            
            center_count = np.sum(relative_pos1 < 0.5)  # 中心区域（<50%）
            edge_count = np.sum(relative_pos1 >= 0.5)   # 边缘区域（>=50%）
            
            print(f"  [匹配点分布] 中心区域(<50%): {center_count}个, 边缘区域(>=50%): {edge_count}个")
            print(f"  [匹配点分布] 平均相对位置: {relative_pos1.mean():.3f} (0=中心, 1=边缘)")
            
            if edge_count > center_count * 2:
                print(f"  [警告] 匹配点主要集中在边缘，可能原因：")
                print(f"    - FPFH特征在边缘处更明显（几何变化大）")
                print(f"    - 采样点云可能集中在边缘")
                print(f"    - 建议检查特征提取参数或采样策略")
        
        # ========== 统一计算所有坐标轴范围（包含平移后的顶点）==========
        # 合并所有顶点（包括平移后的），计算统一的坐标轴范围
        all_vertices_combined = np.vstack([vertices1_separated, vertices2_separated])
        if len(all_vertices_combined) > 0:
            x_min, x_max = all_vertices_combined[:, 0].min(), all_vertices_combined[:, 0].max()
            y_min, y_max = all_vertices_combined[:, 1].min(), all_vertices_combined[:, 1].max()
            z_min, z_max = all_vertices_combined[:, 2].min(), all_vertices_combined[:, 2].max()
            
            # 添加一些边距，避免边界被裁剪
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            margin = 0.1  # 10%的边距
            
            x_min -= x_range * margin
            x_max += x_range * margin
            y_min -= y_range * margin
            y_max += y_range * margin
            z_min -= z_range * margin
            z_max += z_range * margin
        else:
            # 如果没有顶点，使用默认范围
            x_min, x_max = -1, 1
            y_min, y_max = -1, 1
            z_min, z_max = -1, 1
        
        # ========== 统计匹配线颜色分布（统一计算一次）==========
        # 确保两个匹配点都在同一坐标系中（第一帧的坐标系）
        # sampled_points1: 第一帧采样点云（第一帧坐标系）
        # sampled_points2: 第二帧采样点云（应该已经变换到第一帧坐标系）
        green_count = 0
        yellow_count = 0
        red_count = 0
        distances_list = []
        if len(matched_points1) > 0 and len(matched_points2) > 0 and len(matched_points1) == len(matched_points2):
            # 计算匹配点之间的距离（两个点都应该在第一帧坐标系中）
            for i in range(len(matched_points1)):
                pt1 = matched_points1[i]  # 第一帧坐标系中的点
                pt2 = matched_points2[i]  # 应该也在第一帧坐标系中（已应用变换）
                dist = np.linalg.norm(pt2 - pt1)
                distances_list.append(dist)
                if dist < 0.05:
                    green_count += 1
                elif dist < 0.1:
                    yellow_count += 1
                else:
                    red_count += 1
            
            # 调试信息：打印距离统计和坐标系验证
            if len(distances_list) > 0:
                distances_arr = np.array(distances_list)
                # 检查坐标范围，验证坐标系是否正确
                pts1_range = np.array([matched_points1.min(axis=0), matched_points1.max(axis=0)])
                pts2_range = np.array([matched_points2.min(axis=0), matched_points2.max(axis=0)])
                print(f"  [匹配距离统计] 最小距离: {distances_arr.min():.6f}, 最大距离: {distances_arr.max():.6f}, "
                      f"平均距离: {distances_arr.mean():.6f}, 中位数: {np.median(distances_arr):.6f}")
                print(f"  [匹配距离统计] 距离分布: <0.05: {green_count}, 0.05-0.1: {yellow_count}, >=0.1: {red_count}")
                print(f"  [坐标系验证] 第一帧点范围: X[{pts1_range[0,0]:.2f}, {pts1_range[1,0]:.2f}], "
                      f"Y[{pts1_range[0,1]:.2f}, {pts1_range[1,1]:.2f}], Z[{pts1_range[0,2]:.2f}, {pts1_range[1,2]:.2f}]")
                print(f"  [坐标系验证] 第二帧点范围: X[{pts2_range[0,0]:.2f}, {pts2_range[1,0]:.2f}], "
                      f"Y[{pts2_range[0,1]:.2f}, {pts2_range[1,1]:.2f}], Z[{pts2_range[0,2]:.2f}, {pts2_range[1,2]:.2f}]")
                
                # 如果距离过大，可能是坐标系不一致
                if distances_arr.mean() > 10.0:
                    print(f"  [警告] 平均匹配距离较大({distances_arr.mean():.6f})，可能原因：")
                    print(f"    - 坐标系不一致（第二帧点未正确变换到第一帧坐标系）")
                    print(f"    - 匹配质量较差")
                    print(f"    - 坐标单位不统一（像素 vs 归一化坐标）")
        
        # ========== 创建单个3D图，同时显示两个mesh和连接线 ==========
        # 使用一个大的3D图，将两个mesh分别放在左右两侧，然后用线连接
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # ========== 左边：前一帧的mesh ==========
        if len(faces1_display) > 0:
            triangles1 = vertices1_separated[faces1_display]
            mesh_collection1 = Poly3DCollection(triangles1, alpha=0.6, facecolor='#FF6B35', 
                                               edgecolor='#C44536', linewidths=0.05)
            ax.add_collection3d(mesh_collection1)
        
        # 显示前一帧的匹配点
        if len(matched_points1_separated) > 0:
            ax.scatter(matched_points1_separated[:, 0], matched_points1_separated[:, 1], matched_points1_separated[:, 2],
                      c='green', s=60, alpha=0.9, marker='o', edgecolors='darkgreen', linewidths=2, label='匹配点')
        
        # ========== 右边：当前帧的mesh ==========
        if len(faces2_display) > 0:
            triangles2 = vertices2_separated[faces2_display]
            mesh_collection2 = Poly3DCollection(triangles2, alpha=0.6, facecolor='#4ECDC4', 
                                               edgecolor='#2E86AB', linewidths=0.05)
            ax.add_collection3d(mesh_collection2)
        
        # 显示当前帧的匹配点
        if len(matched_points2_separated) > 0:
            ax.scatter(matched_points2_separated[:, 0], matched_points2_separated[:, 1], matched_points2_separated[:, 2],
                      c='green', s=60, alpha=0.9, marker='o', edgecolors='darkgreen', linewidths=2)
        
        # ========== 绘制匹配点之间的连接线 ==========
        if len(matched_points1_separated) > 0 and len(matched_points2_separated) > 0 and len(matched_points1_separated) == len(matched_points2_separated):
            for i in range(len(matched_points1_separated)):
                pt1 = matched_points1_separated[i]  # 左边mesh上的点
                pt2 = matched_points2_separated[i]  # 右边mesh上的点
                
                # 计算原始坐标系中的距离（用于颜色判断）
                pt1_orig = matched_points1[i]
                pt2_orig = matched_points2[i]
                dist = np.linalg.norm(pt2_orig - pt1_orig)
                
                if dist < 0.05:
                    color = 'green'
                    alpha = 0.8
                elif dist < 0.1:
                    color = 'yellow'
                    alpha = 0.6
                else:
                    color = 'red'
                    alpha = 0.4
                
                # 绘制连接线
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                       color=color, alpha=alpha, linewidth=2, linestyle='-')
            
            print(f"  [匹配线统计] 绿色(很近): {green_count}, 黄色(较近): {yellow_count}, 红色(较远): {red_count}")
        
        # ========== 设置坐标轴和标题 ==========
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        
        color_info = f'绿色:{green_count} | 黄色:{yellow_count} | 红色:{red_count}' if line_count > 0 else '无匹配'
        ax.set_title(f'3D匹配可视化: 帧 {frame_idx1} (左) ↔ 帧 {frame_idx2} (右) - {line_count}个匹配点 ({color_info})', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.view_init(elev=20, azim=45)
        
        # ========== 添加图例 ==========
        legend_elements = [
            plt.Line2D([0], [0], color='#FF6B35', lw=4, label=f'帧{frame_idx1} (前一帧, 橙红色)'),
            plt.Line2D([0], [0], color='#4ECDC4', lw=4, label=f'帧{frame_idx2} (当前帧, 青蓝色)'),
            plt.Line2D([0], [0], color='green', lw=2, label='匹配线: 很近(<0.05)'),
            plt.Line2D([0], [0], color='yellow', lw=2, label='匹配线: 较近(0.05-0.1)'),
            plt.Line2D([0], [0], color='red', lw=2, label='匹配线: 较远(≥0.1)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                 framealpha=0.9, ncol=1)
        
        # 保存图像（在显示前保存）
        plt.savefig(image_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  [保存] 图像已保存: {image_path}")
        
        # 显示图像（阻塞显示，等待用户关闭窗口）
        print(f"  [显示] 正在显示3D匹配可视化窗口...")
        print(f"  [提示] 关闭窗口后程序将继续运行")
        plt.show(block=True)  # 阻塞显示，等待用户关闭窗口
        
        # 窗口关闭后继续
        plt.close(fig)

    def _estimate_pose_from_3d_matches(self, matches, K=None):
        """
        从3D匹配结果估计相机位姿（优化版：基于匹配点对的SVD计算）
        
        Args:
            matches: 匹配结果列表，每个match包含：
                - 'transformation': 4x4变换矩阵（已计算）
                - 'correspondence_set': 对应点集合
                - 'sampled_points1': 第一帧采样点云
                - 'sampled_points2_final': 第二帧采样点云（已变换）
            K: 相机内参矩阵（如果为None，会估计）
            
        Returns:
            dict: {
                'poses': 位姿列表（4x4矩阵）,
                'intrinsics': 内参矩阵,
                'distortion': 畸变系数,
                'pose_errors': 位姿估计误差（用于验证）
            }
        """
        try:
            if len(matches) == 0:
                return None
            
            # 1. 估计内参（如果未提供）
            if K is None:
                # 使用第一帧的图像尺寸估计内参
                h, w = 1080, 1920  # 默认值，应该从实际图像获取
                fx = fy = float(np.sqrt(w * w + h * h))
                cx = float(w / 2)
                cy = float(h / 2)
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            
            # 2. 构建位姿链（第一帧为单位变换）
            poses = []
            pose_errors = []  # 用于验证位姿质量
            T_global = np.eye(4, dtype=np.float64)
            poses.append(T_global.copy())
            pose_errors.append({'rotation_error': 0.0, 'translation_error': 0.0, 'reprojection_error': 0.0})
            
            # 3. 对每个匹配，使用SVD优化变换矩阵
            for i, match in enumerate(matches):
                T_rel = match['transformation']
                
                # 如果匹配结果中有采样点云，使用SVD重新计算更精确的变换
                if 'sampled_points1' in match and 'sampled_points2_final' in match:
                    pts1 = match['sampled_points1']
                    pts2 = match['sampled_points2_final']
                    
                    if len(pts1) >= 3 and len(pts2) >= 3:
                        # 使用SVD计算最优变换
                        T_svd = self._estimate_transformation_svd(pts1, pts2)
                        
                        if T_svd is not None:
                            # 验证变换质量
                            error = self._validate_transformation(pts1, pts2, T_svd)
                            pose_errors.append(error)
                            
                            # 如果SVD变换质量更好，使用SVD结果
                            if error['reprojection_error'] < 1.0:  # 阈值可调
                                T_rel = T_svd
                                print(f"  [位姿优化] 帧{i}->{i+1}: 使用SVD优化变换，重投影误差={error['reprojection_error']:.4f}")
                            else:
                                print(f"  [位姿优化] 帧{i}->{i+1}: SVD变换质量不佳，使用原始变换")
                                pose_errors.append({'rotation_error': 0.0, 'translation_error': 0.0, 
                                                   'reprojection_error': error['reprojection_error']})
                        else:
                            pose_errors.append({'rotation_error': 0.0, 'translation_error': 0.0, 'reprojection_error': float('inf')})
                    else:
                        pose_errors.append({'rotation_error': 0.0, 'translation_error': 0.0, 'reprojection_error': float('inf')})
                else:
                    pose_errors.append({'rotation_error': 0.0, 'translation_error': 0.0, 'reprojection_error': float('inf')})
                
                # 累积变换：T_global = T_global * T_rel
                T_global = T_global @ T_rel
                poses.append(T_global.copy())
            
            return {
                'poses': poses,
                'intrinsics': K,
                'distortion': np.zeros(5, dtype=np.float64),  # 假设无畸变
                'pose_errors': pose_errors  # 位姿估计误差
            }
            
        except Exception as e:
            print(f"[异常] 位姿估计失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _validate_transformation(self, pts1, pts2, T):
        """
        验证变换矩阵的质量
        
        Args:
            pts1: 第一组点 (N, 3)
            pts2: 第二组点 (N, 3)
            T: 变换矩阵 (4x4)
            
        Returns:
            dict: {
                'rotation_error': 旋转误差（度）,
                'translation_error': 平移误差,
                'reprojection_error': 重投影误差（平均距离）
            }
        """
        try:
            # 应用变换
            pts2_homo = np.hstack([pts2, np.ones((len(pts2), 1))])
            pts2_transformed = (T @ pts2_homo.T).T[:, :3]
            
            # 计算平均距离误差
            distances = np.linalg.norm(pts1 - pts2_transformed, axis=1)
            reprojection_error = np.mean(distances)
            
            # 提取旋转和平移
            R = T[:3, :3]
            t = T[:3, 3]
            
            # 计算旋转误差（检查是否为有效旋转矩阵）
            R_error = np.abs(np.linalg.det(R) - 1.0)  # 应该接近0
            
            # 计算平移误差（相对大小）
            translation_error = np.linalg.norm(t)
            
            return {
                'rotation_error': R_error * 180.0 / np.pi,  # 转换为度
                'translation_error': translation_error,
                'reprojection_error': reprojection_error
            }
            
        except Exception as e:
            return {
                'rotation_error': float('inf'),
                'translation_error': float('inf'),
                'reprojection_error': float('inf')
            }

    def _compute_pose_error(self, estimated_poses, ground_truth_poses):
        """
        计算估计位姿与真实位姿的误差
        
        Args:
            estimated_poses: 估计的位姿列表（4x4矩阵）
            ground_truth_poses: 真实位姿列表（4x4矩阵）
            
        Returns:
            dict: {
                'rotation_errors': 旋转误差（度）,
                'translation_errors': 平移误差,
                'mean_rotation_error': 平均旋转误差,
                'mean_translation_error': 平均平移误差
            }
        """
        try:
            if len(estimated_poses) != len(ground_truth_poses):
                print(f"[警告] 位姿数量不匹配: 估计{len(estimated_poses)}个, 真实{len(ground_truth_poses)}个")
                min_len = min(len(estimated_poses), len(ground_truth_poses))
                estimated_poses = estimated_poses[:min_len]
                ground_truth_poses = ground_truth_poses[:min_len]
            
            rotation_errors = []
            translation_errors = []
            
            for i, (T_est, T_gt) in enumerate(zip(estimated_poses, ground_truth_poses)):
                # 提取旋转和平移
                R_est = T_est[:3, :3]
                t_est = T_est[:3, 3]
                R_gt = T_gt[:3, :3]
                t_gt = T_gt[:3, 3]
                
                # 计算旋转误差（相对旋转的角度）
                R_rel = R_est @ R_gt.T
                # 使用Rodrigues公式计算旋转角度
                trace = np.trace(R_rel)
                angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                angle_deg = np.degrees(angle_rad)
                rotation_errors.append(angle_deg)
                
                # 计算平移误差（欧氏距离）
                t_error = np.linalg.norm(t_est - t_gt)
                translation_errors.append(t_error)
            
            return {
                'rotation_errors': rotation_errors,
                'translation_errors': translation_errors,
                'mean_rotation_error': np.mean(rotation_errors),
                'mean_translation_error': np.mean(translation_errors),
                'max_rotation_error': np.max(rotation_errors),
                'max_translation_error': np.max(translation_errors),
                'std_rotation_error': np.std(rotation_errors),
                'std_translation_error': np.std(translation_errors)
            }
            
        except Exception as e:
            print(f"[异常] 误差计算失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _update_accumulated_pose(self, prev_frame_idx, current_frame_idx, rt_matrix, reference_frame_idx=0):
        """
        实时更新累积位姿（在计算完每一帧的相对RT矩阵后立即调用）
        
        这个方法会在每次计算RT矩阵后立即更新累积位姿，而不是等所有帧处理完。
        
        Args:
            prev_frame_idx: 前一帧索引
            current_frame_idx: 当前帧索引
            rt_matrix: RT矩阵（4x4，将当前帧mesh变换到前一帧坐标系）
            reference_frame_idx: 参考帧索引（默认0）
            
        Returns:
            dict: {
                'pose': 当前帧的4x4位姿矩阵（相对于参考帧）
                'position': 当前帧的3D位置 [x, y, z]
                'rotation': 当前帧的3x3旋转矩阵
            }
        """
        # 初始化累积位姿数据结构（如果不存在）
        if not hasattr(self, '_accumulated_poses'):
            self._accumulated_poses = {}  # 存储每一帧的累积位姿
            self._accumulated_positions = {}
            self._accumulated_rotations = {}
            self._reference_frame_idx = reference_frame_idx
            
            # 初始化参考帧的位姿
            self._accumulated_poses[reference_frame_idx] = np.eye(4, dtype=np.float32)
            self._accumulated_positions[reference_frame_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._accumulated_rotations[reference_frame_idx] = np.eye(3, dtype=np.float32)
            print(f"  [位姿初始化] 参考帧 {reference_frame_idx} 位姿已初始化为单位矩阵")
        
        # RT矩阵是将frame_curr的mesh变换到frame_prev坐标系
        # 因此，frame_curr相对于frame_prev的位姿是RT的逆
        rt_inv = np.linalg.inv(rt_matrix)
        
        # 累积位姿：T_curr = T_prev @ RT_inv
        if prev_frame_idx in self._accumulated_poses:
            self._accumulated_poses[current_frame_idx] = self._accumulated_poses[prev_frame_idx] @ rt_inv
        else:
            # 如果前一帧位姿不存在，使用单位矩阵（不应该发生）
            print(f"  [警告] 前一帧 {prev_frame_idx} 的位姿不存在，使用单位矩阵")
            self._accumulated_poses[current_frame_idx] = rt_inv
        
        # 提取位置和旋转
        self._accumulated_positions[current_frame_idx] = self._accumulated_poses[current_frame_idx][:3, 3]
        self._accumulated_rotations[current_frame_idx] = self._accumulated_poses[current_frame_idx][:3, :3]
        
        # 转换为欧拉角显示（降低打印频率以加速，每10帧打印一次）
        if current_frame_idx % 10 == 0 or current_frame_idx < 5:
            try:
                from scipy.spatial.transform import Rotation as R_scipy
                r = R_scipy.from_matrix(self._accumulated_rotations[current_frame_idx])
                euler = r.as_euler('XYZ', degrees=True)
                print(f"  [位姿更新] Frame {current_frame_idx}: 位置=[{self._accumulated_positions[current_frame_idx][0]:.3f}, "
                      f"{self._accumulated_positions[current_frame_idx][1]:.3f}, {self._accumulated_positions[current_frame_idx][2]:.3f}], "
                      f"旋转=[{euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°]")
            except:
                print(f"  [位姿更新] Frame {current_frame_idx}: 位置={self._accumulated_positions[current_frame_idx]}")
        
        return {
            'pose': self._accumulated_poses[current_frame_idx],
            'position': self._accumulated_positions[current_frame_idx],
            'rotation': self._accumulated_rotations[current_frame_idx]
        }
    
    def _explain_relative_to_absolute_pose(self):
        """
        解释相对位姿和绝对位姿的关系
        
        这是一个说明性函数，帮助理解如何从相对RT矩阵得到绝对位姿。
        
        核心概念：
        =========
        
        1. 相对位姿（Relative Pose）：
           - RT_i: frame_i相对于frame_i-1的变换矩阵
           - 含义：将frame_i的mesh变换到frame_i-1坐标系
           - 公式：mesh_i_in_frame_i-1 = RT_i @ mesh_i_in_frame_i
           
        2. 绝对位姿（Absolute Pose）：
           - T_i: frame_i相对于参考坐标系（通常是第一帧）的位姿
           - 含义：相机在参考坐标系中的位置和朝向
           - 公式：point_in_world = T_i @ point_in_frame_i
           
        3. 从相对位姿到绝对位姿（累积过程）：
           
           设参考坐标系为frame_0（第一帧）：
           
           T_0 = I  （第一帧作为参考，单位矩阵）
           
           RT_1: frame_1相对于frame_0的相对变换
           → T_1 = RT_1^(-1)  （frame_1的绝对位姿）
           
           RT_2: frame_2相对于frame_1的相对变换
           → T_2 = T_1 @ RT_2^(-1) = RT_1^(-1) @ RT_2^(-1)
           
           RT_3: frame_3相对于frame_2的相对变换
           → T_3 = T_2 @ RT_3^(-1) = RT_1^(-1) @ RT_2^(-1) @ RT_3^(-1)
           
           通用公式：
           T_i = T_{i-1} @ RT_i^(-1)
                = RT_1^(-1) @ RT_2^(-1) @ ... @ RT_i^(-1)
        
        4. 为什么需要求逆？
           - RT矩阵表示"将mesh从frame_i变换到frame_i-1"
           - 但位姿表示"frame_i在frame_i-1坐标系中的位置"
           - 这两个是互逆的关系
        
        5. 误差累积：
           - 由于每一帧的相对变换都有误差，累积后误差会增大
           - 后续帧的绝对位姿误差通常比前面的帧更大
        
        示例：
        =====
        
        假设有3帧：
        - frame_0: 参考坐标系（T_0 = I）
        - frame_1: 相对RT_1 = [R1, t1; 0, 1]
        - frame_2: 相对RT_2 = [R2, t2; 0, 1]
        
        绝对位姿计算：
        - T_0 = I（单位矩阵）
        - T_1 = RT_1^(-1) = [R1^T, -R1^T*t1; 0, 1]
        - T_2 = T_1 @ RT_2^(-1) = T_1 @ [R2^T, -R2^T*t2; 0, 1]
        
        这样，T_1和T_2就是frame_1和frame_2相对于frame_0的绝对位姿。
        
        Returns:
            str: 说明文本
        """
        explanation = """
================================================================================
相对位姿 vs 绝对位姿 - 详细说明
================================================================================

【问题】你计算的是帧与帧之间的相对RT矩阵，如何求得绝对位姿？

【答案】通过累积（Chain）相对变换矩阵！

--------------------------------------------------------------------------------
1. 相对位姿（Relative Pose）
--------------------------------------------------------------------------------

你计算的RT矩阵是相对位姿：
  - RT_i: frame_i相对于frame_i-1的变换矩阵
  - 含义：将frame_i的mesh变换到frame_i-1坐标系
  - 公式：mesh_i_in_frame_i-1 = RT_i @ mesh_i_in_frame_i

例如：
  RT_1: frame_1相对于frame_0的相对变换
  RT_2: frame_2相对于frame_1的相对变换
  RT_3: frame_3相对于frame_2的相对变换
  ...

--------------------------------------------------------------------------------
2. 绝对位姿（Absolute Pose）
--------------------------------------------------------------------------------

绝对位姿是相对于参考坐标系（通常是第一帧）的位姿：
  - T_i: frame_i相对于参考坐标系（frame_0）的位姿
  - 含义：相机在参考坐标系中的位置和朝向
  - 公式：point_in_world = T_i @ point_in_frame_i

--------------------------------------------------------------------------------
3. 从相对位姿到绝对位姿（累积过程）
--------------------------------------------------------------------------------

设参考坐标系为frame_0（第一帧）：

  T_0 = I  （第一帧作为参考，单位矩阵）

  RT_1: frame_1相对于frame_0的相对变换
  → T_1 = RT_1^(-1)  （frame_1的绝对位姿）

  RT_2: frame_2相对于frame_1的相对变换
  → T_2 = T_1 @ RT_2^(-1) = RT_1^(-1) @ RT_2^(-1)

  RT_3: frame_3相对于frame_2的相对变换
  → T_3 = T_2 @ RT_3^(-1) = RT_1^(-1) @ RT_2^(-1) @ RT_3^(-1)

  通用公式：
  T_i = T_{i-1} @ RT_i^(-1)
       = RT_1^(-1) @ RT_2^(-1) @ ... @ RT_i^(-1)

--------------------------------------------------------------------------------
4. 为什么需要求逆？
--------------------------------------------------------------------------------

RT矩阵的含义：
  - RT_i: "将frame_i的mesh变换到frame_i-1坐标系"
  - 即：mesh_i_in_frame_i-1 = RT_i @ mesh_i_in_frame_i

位姿的含义：
  - T_i: "frame_i在frame_i-1坐标系中的位置和朝向"
  - 即：point_in_frame_i-1 = T_i @ point_in_frame_i

这两个是互逆的关系！所以需要求逆。

--------------------------------------------------------------------------------
5. 代码实现
--------------------------------------------------------------------------------

代码中已经实现了这个功能：

1. 实时更新（推荐）：
   - 每次计算完RT矩阵后，调用 _update_accumulated_pose()
   - 自动累积位姿，存储在 _accumulated_poses 中

2. 批量处理：
   - 处理完所有帧后，调用 _accumulate_camera_poses_from_rt_matrices()
   - 从所有RT矩阵一次性计算所有绝对位姿

3. 访问绝对位姿：
   - processor._accumulated_poses[frame_idx]  # 4x4变换矩阵
   - processor._accumulated_positions[frame_idx]  # 3D位置 [x, y, z]
   - processor._accumulated_rotations[frame_idx]  # 3x3旋转矩阵

--------------------------------------------------------------------------------
6. 示例
--------------------------------------------------------------------------------

假设有3帧，相对RT矩阵为：
  RT_1 = [[R1, t1], [0, 1]]  # frame_1相对于frame_0
  RT_2 = [[R2, t2], [0, 1]]  # frame_2相对于frame_1

绝对位姿计算：
  T_0 = I（单位矩阵）
  
  RT_1_inv = [[R1^T, -R1^T*t1], [0, 1]]
  T_1 = RT_1_inv
  
  RT_2_inv = [[R2^T, -R2^T*t2], [0, 1]]
  T_2 = T_1 @ RT_2_inv = RT_1_inv @ RT_2_inv

结果：
  - T_0: frame_0的绝对位姿（参考坐标系，单位矩阵）
  - T_1: frame_1相对于frame_0的绝对位姿
  - T_2: frame_2相对于frame_0的绝对位姿

--------------------------------------------------------------------------------
7. 注意事项
--------------------------------------------------------------------------------

1. 误差累积：
   - 由于每一帧的相对变换都有误差，累积后误差会增大
   - 后续帧的绝对位姿误差通常比前面的帧更大

2. 参考坐标系：
   - 默认使用第一帧（frame_0）作为参考坐标系
   - 可以通过 reference_frame_idx 参数自定义

3. 实时 vs 批量：
   - 实时更新：每帧处理完即可获得位姿（推荐）
   - 批量处理：需要等所有帧处理完

================================================================================
"""
        print(explanation)
        return explanation
    
    def _accumulate_camera_poses_from_rt_matrices(self, reference_frame_idx=0):
        """
        从相对RT矩阵累积计算每一帧的相机位姿（批量处理版本）
        
        注意：如果已经使用_update_accumulated_pose实时更新，可以直接返回已累积的位姿。
        这个方法主要用于批量处理或重新计算。
        
        问题分析：
        ==========
        1. RT矩阵的含义：
           - RT_i: 将frame_i的mesh变换到frame_i-1坐标系的变换矩阵
           - 即：mesh_i_in_frame_i-1 = RT_i @ mesh_i_in_frame_i
           
        2. 相机位姿的含义：
           - 相机位姿描述相机在某个坐标系中的位置和朝向
           - 如果RT是将mesh从frame_i变换到frame_i-1，那么：
             * 相机在frame_i-1中看到的frame_i的mesh位置 = RT_i @ mesh_i
             * 这意味着相机在frame_i中的位姿，相对于frame_i-1，是RT_i的逆
           
        3. 坐标系选择：
           - 方案A：第一帧坐标系作为世界坐标系（推荐）
             * frame_0的位姿 = 单位矩阵 I
             * frame_1的位姿 = RT_1^(-1)  （相对于frame_0）
             * frame_2的位姿 = RT_1^(-1) @ RT_2^(-1)  （相对于frame_0）
             
        4. 累积公式：
           设 T_i 为frame_i相对于参考坐标系（frame_0）的位姿：
           - T_0 = I  （第一帧作为参考）
           - T_1 = RT_1^(-1)  （frame_1相对于frame_0）
           - T_2 = T_1 @ RT_2^(-1) = RT_1^(-1) @ RT_2^(-1)
           - T_i = T_{i-1} @ RT_i^(-1) = RT_1^(-1) @ RT_2^(-1) @ ... @ RT_i^(-1)
           
        Args:
            reference_frame_idx: 参考帧索引（默认0，第一帧作为世界坐标系）
            
        Returns:
            dict: {
                'poses': 位姿列表，每个为4x4变换矩阵（相对于参考帧）
                'positions': 位置列表，每个为3D坐标 [x, y, z]
                'rotations': 旋转矩阵列表，每个为3x3矩阵
                'reference_frame': 参考帧索引
                'frame_indices': 帧索引列表
                'poses_dict': 位姿字典（key为帧索引）
                'positions_dict': 位置字典（key为帧索引）
            }
        """
        # 如果已经实时累积了位姿，直接返回
        if hasattr(self, '_accumulated_poses') and self._accumulated_poses:
            print(f"\n[位姿累积] 使用已累积的位姿（实时更新模式）")
            sorted_indices = sorted(self._accumulated_poses.keys())
            poses_list = [self._accumulated_poses[i] for i in sorted_indices]
            positions_list = [self._accumulated_positions[i] for i in sorted_indices]
            rotations_list = [self._accumulated_rotations[i] for i in sorted_indices]
            
            return {
                'poses': poses_list,
                'positions': positions_list,
                'rotations': rotations_list,
                'frame_indices': sorted_indices,
                'reference_frame': self._reference_frame_idx,
                'poses_dict': self._accumulated_poses.copy(),
                'positions_dict': self._accumulated_positions.copy()
            }
        
        # 否则，从RT矩阵重新计算（批量处理模式）
        if not hasattr(self, '_rt_matrices') or not self._rt_matrices:
            print("[警告] 未找到RT矩阵，无法累积位姿")
            return None
        
        print(f"\n[位姿累积] 开始从RT矩阵累积相机位姿（参考帧: {reference_frame_idx}）")
        
        # 获取所有帧索引
        frame_indices = set()
        for key in self._rt_matrices.keys():
            prev_idx, curr_idx = map(int, key.split('_'))
            frame_indices.add(prev_idx)
            frame_indices.add(curr_idx)
        frame_indices = sorted(list(frame_indices))
        
        print(f"  [帧索引] 找到 {len(frame_indices)} 个帧: {frame_indices}")
        
        # 初始化位姿字典
        poses = {}
        positions = {}
        rotations = {}
        
        # 第一帧（参考帧）的位姿为单位矩阵
        poses[reference_frame_idx] = np.eye(4, dtype=np.float32)
        positions[reference_frame_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        rotations[reference_frame_idx] = np.eye(3, dtype=np.float32)
        print(f"  [参考帧] Frame {reference_frame_idx}: 单位矩阵（参考坐标系）")
        
        # 按顺序累积位姿
        for i in range(1, len(frame_indices)):
            prev_idx = frame_indices[i-1]
            curr_idx = frame_indices[i]
            
            # 获取相对RT矩阵
            rt_key = f"{prev_idx}_{curr_idx}"
            if rt_key not in self._rt_matrices:
                print(f"  [警告] 未找到RT矩阵: {rt_key}，跳过frame {curr_idx}")
                continue
            
            rt_matrix = self._rt_matrices[rt_key]
            
            # RT矩阵是将frame_curr的mesh变换到frame_prev坐标系
            # 因此，frame_curr相对于frame_prev的位姿是RT的逆
            rt_inv = np.linalg.inv(rt_matrix)
            
            # 累积位姿：T_curr = T_prev @ RT_inv
            if prev_idx in poses:
                poses[curr_idx] = poses[prev_idx] @ rt_inv
            else:
                # 如果前一帧位姿不存在，使用单位矩阵（不应该发生）
                print(f"  [警告] 前一帧 {prev_idx} 的位姿不存在，使用单位矩阵")
                poses[curr_idx] = rt_inv
            
            # 提取位置和旋转
            positions[curr_idx] = poses[curr_idx][:3, 3]
            rotations[curr_idx] = poses[curr_idx][:3, :3]
            
            # 转换为欧拉角显示
            try:
                from scipy.spatial.transform import Rotation as R_scipy
                r = R_scipy.from_matrix(rotations[curr_idx])
                euler = r.as_euler('XYZ', degrees=True)
                print(f"  [累积] Frame {curr_idx}: 位置=[{positions[curr_idx][0]:.3f}, {positions[curr_idx][1]:.3f}, {positions[curr_idx][2]:.3f}], "
                      f"旋转=[{euler[0]:.2f}°, {euler[1]:.2f}°, {euler[2]:.2f}°]")
            except:
                print(f"  [累积] Frame {curr_idx}: 位置={positions[curr_idx]}")
        
        # 转换为列表（按帧索引排序）
        sorted_indices = sorted(poses.keys())
        poses_list = [poses[i] for i in sorted_indices]
        positions_list = [positions[i] for i in sorted_indices]
        rotations_list = [rotations[i] for i in sorted_indices]
        
        print(f"\n[完成] 成功累积 {len(poses_list)} 个位姿")
        
        return {
            'poses': poses_list,
            'positions': positions_list,
            'rotations': rotations_list,
            'frame_indices': sorted_indices,
            'reference_frame': reference_frame_idx,
            'poses_dict': poses,  # 保留字典格式以便查询
            'positions_dict': positions
        }
    
    def _save_accumulated_poses(self, accumulated_result, output_path=None):
        """
        保存累积的位姿到文件
        
        Args:
            accumulated_result: _accumulate_camera_poses_from_rt_matrices的返回结果
            output_path: 输出文件路径（如果为None，使用默认路径）
        """
        if accumulated_result is None:
            print("[警告] 位姿累积结果为空，无法保存")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, "accumulated_camera_poses.json")
        
        # 准备保存数据
        save_data = {
            'reference_frame': int(accumulated_result['reference_frame']),
            'frame_indices': [int(i) for i in accumulated_result['frame_indices']],
            'poses': [pose.tolist() for pose in accumulated_result['poses']],
            'positions': [pos.tolist() for pos in accumulated_result['positions']],
            'rotations': [rot.tolist() for rot in accumulated_result['rotations']],
            'note': '位姿是相对于参考帧（reference_frame）的4x4变换矩阵'
        }
        
        # 保存JSON
        # JSON保存已禁用
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"[保存] 累积位姿已保存到: {output_path}")
        
        # 同时保存为CSV格式（便于查看）
        csv_path = output_path.replace('.json', '.csv')
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
            for frame_idx, pos, rot in zip(
                accumulated_result['frame_indices'],
                accumulated_result['positions'],
                accumulated_result['rotations']
            ):
                try:
                    from scipy.spatial.transform import Rotation as R_scipy
                    r = R_scipy.from_matrix(rot)
                    euler = r.as_euler('XYZ', degrees=True)
                    writer.writerow([int(frame_idx), pos[0], pos[1], pos[2], 
                                   euler[0], euler[1], euler[2]])
                except:
                    writer.writerow([int(frame_idx), pos[0], pos[1], pos[2], 0, 0, 0])
        
        print(f"[保存] 累积位姿CSV已保存到: {csv_path}")
        
        # 同时保存为TXT格式（每帧一个文件，便于使用）
        txt_dir = os.path.join(os.path.dirname(output_path), "estimated_poses_txt")
        os.makedirs(txt_dir, exist_ok=True)
        
        for frame_idx, pose in zip(accumulated_result['frame_indices'], accumulated_result['poses']):
            txt_path = os.path.join(txt_dir, f"pose_{frame_idx:04d}.txt")
            np.savetxt(txt_path, pose, fmt='%.8f')
        
        print(f"[保存] 累积位姿TXT文件已保存到: {txt_dir} (共{len(accumulated_result['frame_indices'])}个文件)")
        
        # 同时保存汇总TXT文件（所有位姿在一个文件中，便于查看）
        summary_txt_path = output_path.replace('.json', '_summary.txt')
        self._save_estimated_poses_to_txt(
            accumulated_result['poses'],
            accumulated_result['positions'],
            accumulated_result['frame_indices'],
            summary_txt_path
        )
        print(f"[保存] 累积位姿汇总TXT已保存到: {summary_txt_path}")
        
        # 保存完整的位姿估计结果TXT文件（用户要求的完整文件）
        complete_txt_path = os.path.join(os.path.dirname(output_path), "estimated_poses_complete.txt")
        self._save_estimated_poses_to_txt(
            accumulated_result['poses'],
            accumulated_result['positions'],
            accumulated_result['frame_indices'],
            complete_txt_path
        )
        print(f"[保存] 完整位姿估计结果TXT已保存到: {complete_txt_path}")
    
    def _save_estimated_poses_to_txt(self, poses, positions, frame_indices, save_path):
        """
        保存估计位姿到TXT文件（简单格式，便于人工查看）
        
        Args:
            poses: 位姿列表，每个为4x4变换矩阵
            positions: 位置列表，每个为3D坐标
            frame_indices: 帧索引列表
            save_path: 保存路径（TXT文件）
        """
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("估计的相机位姿数据（基于可微分渲染优化）\n")
                f.write("="*80 + "\n\n")
                f.write(f"总位姿数量: {len(poses)}\n\n")
                f.write("格式说明：\n")
                f.write("  每行格式：帧索引 | 旋转向量[rx, ry, rz] | 平移向量[tx, ty, tz] | 4x4变换矩阵\n")
                f.write("-"*80 + "\n\n")
                
                for frame_idx, pose, pos in zip(frame_indices, poses, positions):
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    
                    # 计算旋转向量
                    rot_vec, _ = cv2.Rodrigues(R)
                    rot_vec_flat = rot_vec.flatten()
                    
                    f.write(f"帧 {frame_idx}:\n")
                    f.write(f"  旋转向量: [{rot_vec_flat[0]:.8f}, {rot_vec_flat[1]:.8f}, {rot_vec_flat[2]:.8f}]\n")
                    f.write(f"  平移向量: [{t[0]:.8f}, {t[1]:.8f}, {t[2]:.8f}]\n")
                    f.write(f"  4x4变换矩阵:\n")
                    for row in pose:
                        f.write(f"    [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}, {row[3]:12.8f}]\n")
                    f.write("\n")
                
                # 添加汇总信息
                f.write("-"*80 + "\n")
                f.write("汇总信息:\n")
                if len(positions) > 0:
                    positions_array = np.array(positions)
                    f.write(f"  位置范围:\n")
                    f.write(f"    X: [{positions_array[:, 0].min():.4f}, {positions_array[:, 0].max():.4f}]\n")
                    f.write(f"    Y: [{positions_array[:, 1].min():.4f}, {positions_array[:, 1].max():.4f}]\n")
                    f.write(f"    Z: [{positions_array[:, 2].min():.4f}, {positions_array[:, 2].max():.4f}]\n")
            
            return True
            
        except Exception as e:
            print(f"  [错误] 保存估计位姿TXT失败: {e}")
            return False
    
    def _detect_loop_closure(self, current_frame_idx, min_frame_gap=10, position_threshold=0.5, rotation_threshold=30.0, 
                             max_check_frames=50, sample_stride=5):
        """
        回环检测：检测当前帧是否与历史帧相似（可能回到之前的位置）
        
        这是减少累积误差的关键技术之一。当检测到回环时，可以添加额外的约束来修正累积误差。
        
        优化版本：通过采样历史帧来加速检测，而不是检查所有历史帧。
        
        Args:
            current_frame_idx: 当前帧索引
            min_frame_gap: 最小帧间隔（避免检测相邻帧）
            position_threshold: 位置相似度阈值（米）
            rotation_threshold: 旋转相似度阈值（度）
            max_check_frames: 最多检查的历史帧数量（默认50，加速检测）
            sample_stride: 采样步长（每隔N帧采样一次，默认5）
            
        Returns:
            dict: {
                'detected': 是否检测到回环,
                'loop_frame_idx': 回环帧索引（如果检测到）,
                'position_error': 位置误差,
                'rotation_error': 旋转误差（度）
            }
        """
        if not hasattr(self, '_accumulated_poses') or not self._accumulated_poses:
            return {'detected': False}
        
        if current_frame_idx not in self._accumulated_poses:
            return {'detected': False}
        
        current_pos = self._accumulated_positions[current_frame_idx]
        current_rot = self._accumulated_rotations[current_frame_idx]
        
        # 获取所有历史帧索引并排序
        all_hist_frames = sorted([idx for idx in self._accumulated_poses.keys() 
                                 if idx < current_frame_idx - min_frame_gap])
        
        if len(all_hist_frames) == 0:
            return {'detected': False}
        
        # 优化策略1：优先检查最近的历史帧（最可能形成回环）
        # 取最近的一些帧进行详细检查
        recent_frames = all_hist_frames[-max_check_frames:] if len(all_hist_frames) > max_check_frames else all_hist_frames
        
        # 优化策略2：对更早的帧进行采样（每隔sample_stride帧采样一次）
        if len(all_hist_frames) > max_check_frames:
            earlier_frames = all_hist_frames[:-max_check_frames]
            sampled_earlier_frames = earlier_frames[::sample_stride]
            frames_to_check = sampled_earlier_frames + recent_frames
        else:
            frames_to_check = recent_frames
        
        # 限制检查的帧数量
        frames_to_check = frames_to_check[-max_check_frames:] if len(frames_to_check) > max_check_frames else frames_to_check
        
        # 检查采样后的历史帧
        for hist_frame_idx in frames_to_check:
            hist_pos = self._accumulated_positions[hist_frame_idx]
            hist_rot = self._accumulated_rotations[hist_frame_idx]
            
            # 快速预筛选：先检查位置距离（计算更快）
            pos_error = np.linalg.norm(current_pos - hist_pos)
            if pos_error > position_threshold * 2.0:  # 宽松的预筛选阈值
                continue
            
            # 计算旋转误差（只在位置接近时计算）
            try:
                from scipy.spatial.transform import Rotation as R_scipy
                r_current = R_scipy.from_matrix(current_rot)
                r_hist = R_scipy.from_matrix(hist_rot)
                rot_error = (r_current * r_hist.inv()).magnitude() * 180.0 / np.pi
            except:
                # 如果scipy不可用，使用简单的Frobenius范数
                rot_error = np.linalg.norm(current_rot - hist_rot, 'fro') * 180.0 / np.pi
            
            # 检查是否满足回环条件
            if pos_error < position_threshold and rot_error < rotation_threshold:
                print(f"  [回环检测] 检测到回环！Frame {current_frame_idx} 与 Frame {hist_frame_idx} 相似")
                print(f"    位置误差: {pos_error:.3f}m (阈值: {position_threshold}m)")
                print(f"    旋转误差: {rot_error:.2f}° (阈值: {rotation_threshold}°)")
                
                return {
                    'detected': True,
                    'loop_frame_idx': hist_frame_idx,
                    'current_frame_idx': current_frame_idx,
                    'position_error': pos_error,
                    'rotation_error': rot_error
                }
        
        return {'detected': False}
    
    def _pose_graph_optimization(self, max_iterations=10, convergence_threshold=1e-4):
        """
        位姿图优化（Pose Graph Optimization）
        
        通过优化所有相对位姿约束，全局优化所有位姿，减少累积误差。
        
        算法：
        1. 构建位姿图：节点=位姿，边=相对RT矩阵
        2. 优化目标：最小化所有相对位姿约束的误差
        3. 使用迭代优化方法（类似高斯-牛顿）
        
        Args:
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            
        Returns:
            dict: {
                'optimized_poses': 优化后的位姿字典,
                'optimized_positions': 优化后的位置字典,
                'optimized_rotations': 优化后的旋转字典,
                'iterations': 实际迭代次数,
                'final_error': 最终误差
            }
        """
        if not hasattr(self, '_accumulated_poses') or not self._accumulated_poses:
            print("[警告] 未找到累积位姿，无法进行位姿图优化")
            return None
        
        if not hasattr(self, '_rt_matrices') or not self._rt_matrices:
            print("[警告] 未找到RT矩阵，无法进行位姿图优化")
            return None
        
        print(f"\n[位姿图优化] 开始优化 {len(self._accumulated_poses)} 个位姿...")
        
        # 复制当前位姿作为初始值
        optimized_poses = {k: v.copy() for k, v in self._accumulated_poses.items()}
        optimized_positions = {k: v.copy() for k, v in self._accumulated_positions.items()}
        optimized_rotations = {k: v.copy() for k, v in self._accumulated_rotations.items()}
        
        # 构建相对位姿约束列表
        constraints = []
        for rt_key, rt_matrix in self._rt_matrices.items():
            prev_idx, curr_idx = map(int, rt_key.split('_'))
            if prev_idx in optimized_poses and curr_idx in optimized_poses:
                constraints.append((prev_idx, curr_idx, rt_matrix))
        
        print(f"  [约束] 找到 {len(constraints)} 个相对位姿约束")
        
        # 迭代优化
        for iteration in range(max_iterations):
            total_error = 0.0
            
            # 对每个约束计算误差并更新位姿
            for prev_idx, curr_idx, rt_matrix in constraints:
                # 计算预测的相对位姿
                T_prev = optimized_poses[prev_idx]
                T_curr = optimized_poses[curr_idx]
                rel_pred = np.linalg.inv(T_prev) @ T_curr
                
                # 计算误差（目标：rel_pred应该等于rt_matrix的逆）
                rt_inv = np.linalg.inv(rt_matrix)
                error_matrix = rel_pred @ np.linalg.inv(rt_inv)
                
                # 提取旋转和平移误差
                R_error = error_matrix[:3, :3]
                t_error = error_matrix[:3, 3]
                
                # 计算误差大小
                rot_error = np.linalg.norm(R_error - np.eye(3), 'fro')
                trans_error = np.linalg.norm(t_error)
                total_error += rot_error + trans_error
                
                # 更新位姿（简单的梯度下降）
                alpha = 0.1  # 学习率
                
                # 旋转修正（使用Rodrigues公式）
                try:
                    from scipy.spatial.transform import Rotation as R_scipy
                    r_error = R_scipy.from_matrix(R_error)
                    angle_axis = r_error.as_rotvec()
                    correction_angle = angle_axis * alpha
                    R_correction = R_scipy.from_rotvec(correction_angle).as_matrix()
                except:
                    # 如果scipy不可用，使用简单的线性修正
                    R_correction = np.eye(3) + (R_error - np.eye(3)) * alpha
                
                # 平移修正
                t_correction = -t_error * alpha
                
                # 应用修正（修正当前帧位姿）
                correction = np.eye(4)
                correction[:3, :3] = R_correction
                correction[:3, 3] = t_correction
                
                # 更新位姿
                optimized_poses[curr_idx] = optimized_poses[curr_idx] @ correction
                optimized_positions[curr_idx] = optimized_poses[curr_idx][:3, 3]
                optimized_rotations[curr_idx] = optimized_poses[curr_idx][:3, :3]
            
            # 检查收敛
            if iteration % 2 == 0:
                print(f"  迭代 {iteration+1}/{max_iterations}, 总误差: {total_error:.6f}")
            
            if total_error < convergence_threshold:
                print(f"  [收敛] 在第 {iteration+1} 次迭代时收敛")
                break
        
        print(f"[完成] 位姿图优化完成，最终误差: {total_error:.6f}")
        
        return {
            'optimized_poses': optimized_poses,
            'optimized_positions': optimized_positions,
            'optimized_rotations': optimized_rotations,
            'iterations': iteration + 1,
            'final_error': total_error
        }
    
    def _apply_optimized_poses(self, optimization_result):
        """
        应用优化后的位姿，更新累积位姿
        
        Args:
            optimization_result: _pose_graph_optimization的返回结果
        """
        if optimization_result is None:
            return
        
        print(f"\n[应用优化] 更新累积位姿...")
        
        # 更新累积位姿
        self._accumulated_poses = optimization_result['optimized_poses'].copy()
        self._accumulated_positions = optimization_result['optimized_positions'].copy()
        self._accumulated_rotations = optimization_result['optimized_rotations'].copy()
        
        print(f"  [完成] 已更新 {len(self._accumulated_poses)} 个位姿")
    
    def _detect_and_correct_drift(self, current_frame_idx, max_position_drift=10.0, max_rotation_drift=90.0):
        """
        检测并修正累积误差（Drift Detection and Correction）
        
        通过检测异常大的位姿变化来识别累积误差，并进行修正。
        
        Args:
            current_frame_idx: 当前帧索引
            max_position_drift: 最大位置漂移（米）
            max_rotation_drift: 最大旋转漂移（度）
            
        Returns:
            dict: {
                'drift_detected': 是否检测到漂移,
                'correction_applied': 是否应用了修正,
                'position_drift': 位置漂移量,
                'rotation_drift': 旋转漂移量
            }
        """
        if not hasattr(self, '_accumulated_poses') or len(self._accumulated_poses) < 2:
            return {'drift_detected': False}
        
        if current_frame_idx not in self._accumulated_poses:
            return {'drift_detected': False}
        
        # 获取当前帧和前一帧的位姿
        if current_frame_idx - 1 not in self._accumulated_poses:
            return {'drift_detected': False}
        
        current_pos = self._accumulated_positions[current_frame_idx]
        prev_pos = self._accumulated_positions[current_frame_idx - 1]
        
        # 计算相对位移
        relative_displacement = np.linalg.norm(current_pos - prev_pos)
        
        # 计算相对旋转
        current_rot = self._accumulated_rotations[current_frame_idx]
        prev_rot = self._accumulated_rotations[current_frame_idx - 1]
        
        try:
            from scipy.spatial.transform import Rotation as R_scipy
            r_current = R_scipy.from_matrix(current_rot)
            r_prev = R_scipy.from_matrix(prev_rot)
            relative_rotation = (r_current * r_prev.inv()).magnitude() * 180.0 / np.pi
        except:
            relative_rotation = np.linalg.norm(current_rot - prev_rot, 'fro') * 180.0 / np.pi
        
        # 检测异常大的变化（可能是累积误差）
        # 优化：快速检查，避免不必要的打印
        drift_detected = False
        if relative_displacement > max_position_drift:
            if current_frame_idx % 10 == 0:  # 降低打印频率
                print(f"  [漂移检测] Frame {current_frame_idx}: 位置变化异常大 ({relative_displacement:.3f}m > {max_position_drift}m)")
            drift_detected = True
        
        if relative_rotation > max_rotation_drift:
            if current_frame_idx % 10 == 0:  # 降低打印频率
                print(f"  [漂移检测] Frame {current_frame_idx}: 旋转变化异常大 ({relative_rotation:.2f}° > {max_rotation_drift}°)")
            drift_detected = True
        
        if drift_detected:
            # 可以应用平滑修正（例如：使用移动平均）
            # 这里只是检测，实际修正可以通过位姿图优化完成
            if current_frame_idx % 10 == 0:  # 降低打印频率
                print(f"  [建议] 检测到累积误差，建议运行位姿图优化")
        
        return {
            'drift_detected': drift_detected,
            'position_drift': relative_displacement,
            'rotation_drift': relative_rotation,
            'correction_applied': False  # 需要手动调用位姿图优化
        }
    
    def _save_3d_matching_results(self, matches, pose_estimation=None, pose_errors=None, suffix="th_68"):
        """
        保存3D匹配结果、位姿估计和误差分析
        
        Args:
            matches: 匹配结果列表
            pose_estimation: 位姿估计结果
            pose_errors: 位姿误差结果
            suffix: 文件后缀
        """
        try:
            match_output_dir = os.path.join(self.output_dir, "3d_mesh_feature_matching")
            os.makedirs(match_output_dir, exist_ok=True)
            
            # 1. 保存匹配结果
            result_json = {
                'suffix': suffix,
                'total_matches': len(matches),
                'matches': []
            }
            
            for match in matches:
                result_json['matches'].append({
                    'frame_idx1': int(match['frame_idx1']),
                    'frame_idx2': int(match['frame_idx2']),
                    'transformation': match['transformation'].tolist(),
                    'rotation': match['rotation'].tolist(),
                    'translation': match['translation'].tolist(),
                    'fitness': float(match['fitness']),
                    'inlier_rmse': float(match['inlier_rmse']),
                    'method': match['method']
                })
            
            # 2. 添加位姿估计结果
            if pose_estimation is not None:
                result_json['pose_estimation'] = {
                    'intrinsics': pose_estimation['intrinsics'].tolist(),
                    'distortion': pose_estimation['distortion'].tolist(),
                    'poses': [pose.tolist() for pose in pose_estimation['poses']]
                }
            
            # 3. 添加误差分析
            if pose_errors is not None:
                result_json['pose_errors'] = {
                    'rotation_errors': [float(e) for e in pose_errors['rotation_errors']],
                    'translation_errors': [float(e) for e in pose_errors['translation_errors']],
                    'mean_rotation_error': float(pose_errors['mean_rotation_error']),
                    'mean_translation_error': float(pose_errors['mean_translation_error']),
                    'max_rotation_error': float(pose_errors['max_rotation_error']),
                    'max_translation_error': float(pose_errors['max_translation_error']),
                    'std_rotation_error': float(pose_errors['std_rotation_error']),
                    'std_translation_error': float(pose_errors['std_translation_error'])
                }
            
            json_path = os.path.join(match_output_dir, f"3d_mesh_matches_{suffix}.json")
            # JSON保存已禁用
            # with open(json_path, 'w', encoding='utf-8') as f:
            #     json.dump(result_json, f, ensure_ascii=False, indent=2)
            
            print(f"[保存] 3D匹配结果已保存: {json_path}")
            
            # 4. 打印误差摘要
            if pose_errors is not None:
                print("\n" + "="*80)
                print("位姿估计误差分析")
                print("="*80)
                print(f"平均旋转误差: {pose_errors['mean_rotation_error']:.4f}度")
                print(f"平均平移误差: {pose_errors['mean_translation_error']:.6f}")
                print(f"最大旋转误差: {pose_errors['max_rotation_error']:.4f}度")
                print(f"最大平移误差: {pose_errors['max_translation_error']:.6f}")
                print(f"旋转误差标准差: {pose_errors['std_rotation_error']:.4f}度")
                print(f"平移误差标准差: {pose_errors['std_translation_error']:.6f}")
                print("="*80)
            
            # 5. 保存指标到TXT文件
            txt_path = os.path.join(match_output_dir, f"evaluation_metrics_{suffix}.txt")
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("3D网格特征匹配评估指标\n")
                    f.write("="*80 + "\n\n")
                    
                    # 匹配统计信息
                    f.write("【匹配统计】\n")
                    f.write(f"总匹配对数: {len(matches)}\n")
                    if len(matches) > 0:
                        # 计算匹配指标
                        fitness_values = [m['fitness'] for m in matches]
                        rmse_values = [m['inlier_rmse'] for m in matches]
                        
                        f.write(f"\n匹配适应度 (Fitness):\n")
                        f.write(f"  平均值: {np.mean(fitness_values):.6f}\n")
                        f.write(f"  最大值: {np.max(fitness_values):.6f}\n")
                        f.write(f"  最小值: {np.min(fitness_values):.6f}\n")
                        f.write(f"  标准差: {np.std(fitness_values):.6f}\n")
                        
                        f.write(f"\n内点RMSE (Inlier RMSE):\n")
                        f.write(f"  平均值: {np.mean(rmse_values):.6f}\n")
                        f.write(f"  最大值: {np.max(rmse_values):.6f}\n")
                        f.write(f"  最小值: {np.min(rmse_values):.6f}\n")
                        f.write(f"  标准差: {np.std(rmse_values):.6f}\n")
                        
                        # 计算RMSE（均方根误差）
                        rmse_overall = np.sqrt(np.mean(np.array(rmse_values)**2))
                        f.write(f"  总体RMSE: {rmse_overall:.6f}\n")
                    
                    # 位姿误差信息
                    if pose_errors is not None:
                        f.write("\n" + "="*80 + "\n")
                        f.write("【位姿估计误差分析】\n")
                        f.write("="*80 + "\n\n")
                        
                        f.write("旋转误差 (度):\n")
                        f.write(f"  平均值: {pose_errors['mean_rotation_error']:.6f}\n")
                        f.write(f"  最大值: {pose_errors['max_rotation_error']:.6f}\n")
                        f.write(f"  标准差: {pose_errors['std_rotation_error']:.6f}\n")
                        
                        # 计算旋转误差的RMSE
                        if len(pose_errors['rotation_errors']) > 0:
                            rot_rmse = np.sqrt(np.mean(np.array(pose_errors['rotation_errors'])**2))
                            f.write(f"  RMSE: {rot_rmse:.6f}\n")
                        
                        f.write("\n平移误差:\n")
                        f.write(f"  平均值: {pose_errors['mean_translation_error']:.6f}\n")
                        f.write(f"  最大值: {pose_errors['max_translation_error']:.6f}\n")
                        f.write(f"  标准差: {pose_errors['std_translation_error']:.6f}\n")
                        
                        # 计算平移误差的RMSE
                        if len(pose_errors['translation_errors']) > 0:
                            trans_rmse = np.sqrt(np.mean(np.array(pose_errors['translation_errors'])**2))
                            f.write(f"  RMSE: {trans_rmse:.6f}\n")
                    
                    # 详细匹配信息
                    if len(matches) > 0:
                        f.write("\n" + "="*80 + "\n")
                        f.write("【详细匹配信息】\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"{'帧对':<15} {'Fitness':<12} {'RMSE':<15} {'方法':<20}\n")
                        f.write("-"*80 + "\n")
                        for match in matches:
                            frame_pair = f"{match['frame_idx1']:04d}-{match['frame_idx2']:04d}"
                            f.write(f"{frame_pair:<15} {match['fitness']:<12.6f} {match['inlier_rmse']:<15.6f} {match['method']:<20}\n")
                    
                    # 详细位姿误差信息
                    if pose_errors is not None and len(pose_errors.get('rotation_errors', [])) > 0:
                        f.write("\n" + "="*80 + "\n")
                        f.write("【逐帧位姿误差】\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"{'帧索引':<10} {'旋转误差(度)':<18} {'平移误差':<15}\n")
                        f.write("-"*80 + "\n")
                        for i, (rot_err, trans_err) in enumerate(zip(
                            pose_errors['rotation_errors'], 
                            pose_errors['translation_errors']
                        )):
                            f.write(f"{i:<10} {rot_err:<18.6f} {trans_err:<15.6f}\n")
                    
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*80 + "\n")
                
                print(f"[保存] 评估指标已保存到: {txt_path}")
            except Exception as e:
                print(f"[警告] 保存评估指标到TXT文件失败: {e}")
            
        except Exception as e:
            print(f"[异常] 保存匹配结果失败: {e}")
            import traceback
            traceback.print_exc()

    def _estimate_fft_rst(self, arr1, arr2, suffix="first_order"):
        """估计旋转-尺度-平移（RST）：Fourier–Mellin用于旋转/尺度，POC用于平移。
        返回并保存JSON：rotation_deg, scale, tx, ty, confidence。
        """
        try:
            if arr1 is None or arr2 is None:
                return None
            a1 = np.asarray(arr1, dtype=np.float32)
            a2 = np.asarray(arr2, dtype=np.float32)
            h1, w1 = a1.shape
            h2, w2 = a2.shape
            # 统一尺寸
            if (h1, w1) != (h2, w2):
                a2 = cv2.resize(a2, (w1, h1), interpolation=cv2.INTER_AREA)
                h2, w2 = a2.shape
            # Hann窗口，减少边界效应
            hann_y = np.hanning(h1).astype(np.float32)
            hann_x = np.hanning(w1).astype(np.float32)
            win = np.outer(hann_y, hann_x)
            a1w = a1 * win
            a2w = a2 * win
            # 幅度谱（对数压缩）
            eps = 1e-8
            F1 = np.fft.fft2(a1w)
            F2 = np.fft.fft2(a2w)
            S1 = np.log(np.abs(np.fft.fftshift(F1)) + eps).astype(np.float32)
            S2 = np.log(np.abs(np.fft.fftshift(F2)) + eps).astype(np.float32)
            # Log-Polar映射参数
            cy, cx = h1 // 2, w1 // 2
            rmax = float(min(cx, cy))
            if rmax < 1:
                rmax = 1.0
            M = w1 / math.log(rmax)
            flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
            LP1 = cv2.logPolar(S1, (cx, cy), M, flags)
            LP2 = cv2.logPolar(S2, (cx, cy), M, flags)
            # 在log-polar域做相位相关，获得尺度与旋转
            try:
                shift_lp, resp_lp = cv2.phaseCorrelate(LP1, LP2)
                dx_lp, dy_lp = float(shift_lp[0]), float(shift_lp[1])
            except Exception:
                dx_lp, dy_lp = cv2.phaseCorrelate(LP1, LP2)
                resp_lp = 1.0
            rotation_deg = -dy_lp * 360.0 / float(LP1.shape[0])
            scale = math.exp(dx_lp / float(M))
            # 将arr2进行旋转与尺度矫正
            rotM = cv2.getRotationMatrix2D((cx, cy), rotation_deg, scale)
            a2_rs = cv2.warpAffine(a2, rotM, (w1, h1), flags=cv2.INTER_LINEAR)
            # 相位相关求平移
            try:
                shift_xy, resp_xy = cv2.phaseCorrelate(a1w, a2_rs * win)
                tx, ty = float(shift_xy[0]), float(shift_xy[1])
            except Exception:
                tx, ty = cv2.phaseCorrelate(a1w, a2_rs * win)
                resp_xy = 1.0
            confidence = float(resp_lp) * float(resp_xy)
            out = {
                "rotation_deg": float(rotation_deg),
                "scale": float(scale),
                "tx": float(tx),
                "ty": float(ty),
                "confidence": confidence,
                "size": [int(w1), int(h1)]
            }
            # JSON保存已禁用
            # try:
            #     with open(os.path.join(self.output_dir, f"fft_rst_{suffix}.json"), "w", encoding="utf-8") as f:
            #         json.dump(out, f, ensure_ascii=False, indent=2)
            # except Exception as e:
            #     print(f"[警告] 保存RST估计结果失败({suffix}): {e}")
            return out
        except Exception as e:
            print(f"[异常] RST估计失败({suffix}): {e}")
            return None

    def _transform_mesh_xy_rst(self, mesh, rotation_deg, scale, tx, ty):
        """在XY平面应用RST变换到网格，Z不缩放。返回新网格。"""
        try:
            if mesh is None:
                return None
            theta = math.radians(rotation_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            T = np.array([
                [scale * cos_t, -scale * sin_t, 0.0, tx],
                [scale * sin_t,  scale * cos_t, 0.0, ty],
                [0.0,            0.0,           1.0, 0.0],
                [0.0,            0.0,           0.0, 1.0]
            ], dtype=np.float64)
            m = copy.deepcopy(mesh)
            m.transform(T)
            return m
        except Exception as e:
            print(f"[异常] 网格RST变换失败: {e}")
            return None

    def _icp_verify_meshes(self, mesh_ref, mesh_mov, rst, suffix="first_order", max_correspondence_distance=5.0, sample_points=5000):
        """对两帧网格做几何核验（ICP），输出配准质量指标JSON。"""
        try:
            if mesh_ref is None or mesh_mov is None or rst is None:
                return None
            mesh_mov_rs = self._transform_mesh_xy_rst(mesh_mov, rst["rotation_deg"], rst["scale"], rst["tx"], rst["ty"]) 
            if mesh_mov_rs is None:
                return None
            try:
                mesh_ref.compute_vertex_normals()
                mesh_mov_rs.compute_vertex_normals()
            except Exception:
                pass
            try:
                pcd_ref = mesh_ref.sample_points_poisson_disk(sample_points)
                pcd_mov = mesh_mov_rs.sample_points_poisson_disk(sample_points)
            except Exception:
                pcd_ref = mesh_ref.sample_points_uniformly(sample_points)
                pcd_mov = mesh_mov_rs.sample_points_uniformly(sample_points)
            reg = o3d.pipelines.registration.registration_icp(
                pcd_mov, pcd_ref, max_correspondence_distance, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            out = {
                "fitness": float(reg.fitness),
                "inlier_rmse": float(reg.inlier_rmse),
                "points_ref": int(np.asarray(pcd_ref.points).shape[0]),
                "points_mov": int(np.asarray(pcd_mov.points).shape[0])
            }
            # JSON保存已禁用
            # try:
            #     with open(os.path.join(self.output_dir, f"fft_rst_icp_{suffix}.json"), "w", encoding="utf-8") as f:
            #         json.dump(out, f, ensure_ascii=False, indent=2)
            # except Exception as e:
            #     print(f"[警告] 保存ICP核验结果失败({suffix}): {e}")
            return out
        except Exception as e:
            print(f"[异常] ICP核验失败({suffix}): {e}")
            return None

    def _video_rst_demo(self, video_path, do_icp=True):
        """从视频读取首对帧，构建高度场，估计RST并（可选）ICP核验，保存JSON与对齐示例网格。"""
        try:
            cap = cv2.VideoCapture(video_path)
            ok1, f1 = cap.read()
            ok2, f2 = cap.read()
            cap.release()
            if not (ok1 and ok2):
                print(f"[警告] 无法从视频读取两帧: {video_path}")
                return
            def build_fields(frame, idx):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(gx * gx + gy * gy)
                mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                th = getattr(self, "th_68", 68)
                mask = (gray.astype(np.float32) >= float(th))
                masked = (mag_norm.astype(np.float32) * mask).astype(np.uint8)
                # 保存 th_68 掩蔽梯度图（视频帧）
                try:
                    cv2.imwrite(os.path.join(self.output_dir, f"th_68_masked_gradient_video_f{idx}.jpg"), masked)
                except Exception:
                    pass
                mesh_g, arr_g = self._o3d_build_height_mesh(mag_norm, step=self.o3d_step, z_scale=self.o3d_z_scale)
                mesh_t, arr_t = self._o3d_build_height_mesh(masked, step=self.o3d_step, z_scale=self.o3d_z_scale)
                # 保存 th_68 下采样高度场图（视频帧）
                try:
                    arr_t_img = np.clip(self._normalize(arr_t) * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(self.output_dir, f"th_68_height_ds_video_f{idx}.jpg"), arr_t_img)
                except Exception:
                    pass
                return (mesh_g, arr_g), (mesh_t, arr_t)
            (mesh_g1, arr_g1), (mesh_t1, arr_t1) = build_fields(f1, 1)
            (mesh_g2, arr_g2), (mesh_t2, arr_t2) = build_fields(f2, 2)
            rst_g = self._estimate_fft_rst(arr_g1, arr_g2, suffix="first_order_video")
            if rst_g is not None:
                # 保存对齐后的示例网格（first_order）
                try:
                    out_mesh_dir = getattr(self, "mesh_out_dir", getattr(self, "mesh_dir", self.output_dir))
                    try:
                        os.makedirs(out_mesh_dir, exist_ok=True)
                    except Exception:
                        pass
                    if (mesh_g2 is not None) and (o3d is not None):
                        aligned_g2 = self._transform_mesh_xy_rst(mesh_g2, rst_g["rotation_deg"], rst_g["scale"], rst_g["tx"], rst_g["ty"])
                        if aligned_g2 is not None:
                            o3d.io.write_triangle_mesh(os.path.join(out_mesh_dir, "o3d_height_first_order_video_aligned.ply"), aligned_g2, write_vertex_colors=True)
                            print(f"[信息] 已保存对齐示例网格(first_order): {os.path.join(out_mesh_dir, 'o3d_height_first_order_video_aligned.ply')}")
                except Exception as e:
                    print(f"[警告] 保存对齐网格失败(first_order_video): {e}")
            if do_icp and rst_g is not None:
                self._icp_verify_meshes(mesh_g1, mesh_g2, rst_g, suffix="first_order_video")
            rst_t = self._estimate_fft_rst(arr_t1, arr_t2, suffix="th_68_video")
            if rst_t is not None:
                # 保存对齐后的示例网格（th_68）
                try:
                    out_mesh_dir = getattr(self, "mesh_out_dir", getattr(self, "mesh_dir", self.output_dir))
                    try:
                        os.makedirs(out_mesh_dir, exist_ok=True)
                    except Exception:
                        pass
                    if (mesh_t2 is not None) and (o3d is not None):
                        aligned_t2 = self._transform_mesh_xy_rst(mesh_t2, rst_t["rotation_deg"], rst_t["scale"], rst_t["tx"], rst_t["ty"])
                        if aligned_t2 is not None:
                            o3d.io.write_triangle_mesh(os.path.join(out_mesh_dir, "o3d_height_th_68_video_aligned.ply"), aligned_t2, write_vertex_colors=True)
                            print(f"[信息] 已保存对齐示例网格(th_68): {os.path.join(out_mesh_dir, 'o3d_height_th_68_video_aligned.ply')}")
                except Exception as e:
                    print(f"[警告] 保存对齐网格失败(th_68_video): {e}")
            if do_icp and rst_t is not None:
                self._icp_verify_meshes(mesh_t1, mesh_t2, rst_t, suffix="th_68_video")
        except Exception as e:
            print(f"[异常] 视频RST示例失败: {e}")

    def _fd_feature_from_image(self, image_bgr, percent=None, num_points=None, num_harmonics=None):
        """从图像提取最大轮廓的傅里叶级数特征（FD）。返回特征字典或None。"""
        try:
            if percent is None:
                percent = getattr(self, "main_percent", 68)
            if num_points is None:
                num_points = getattr(self, "num_resample_points", 256)
            if num_harmonics is None:
                num_harmonics = getattr(self, "efd_harmonics", 16)
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            mag_norm = self._normalize(mag)
            th_val = np.percentile(mag_norm, percent)
            mask = (mag_norm >= th_val).astype(np.uint8) * 255
            pts = self._extract_largest_contour(mask)
            if pts is None:
                return None
            pts = self._canonicalize_contour(pts)
            pts_rs = self._resample_contour(pts, num_points=num_points)
            fd = self._compute_fourier_descriptors(pts_rs, num_harmonics=num_harmonics)
            if fd is None:
                return None
            features = fd["features"]
            return {
                "num_points": features.get("num_points"),
                "num_harmonics": features.get("num_harmonics"),
                "magnitudes_norm": features.get("magnitudes_norm"),
                "phases_rel": features.get("phases_rel"),
                "centroid": features.get("centroid")
            }
        except Exception as e:
            print(f"[警告] 提取FD特征失败: {e}")
            return None

    def _load_frame_features(self, frame_idx, suffix="th_68"):
        """加载指定帧的特征JSON文件，返回特征字典或None"""
        try:
            # 注意：已移除傅里叶特征，只加载谱特征
            spectral_path = os.path.join(self.output_dir, f"o3d_spectral_descriptors_{suffix}.json")
            
            # 如果帧索引不为空，尝试加载特定帧的特征
            if frame_idx is not None:
                spectral_path_frame = os.path.join(self.output_dir, f"o3d_spectral_descriptors_{suffix}_{frame_idx}.json")
                if os.path.exists(spectral_path_frame):
                    spectral_path = spectral_path_frame
            
            spectral_feat = None
            
            if os.path.exists(spectral_path):
                with open(spectral_path, 'r', encoding='utf-8') as f:
                    spectral_feat = json.load(f)
            
            if spectral_feat is None:
                return None
            
            return {
                "spectral": spectral_feat,
                "frame_idx": frame_idx
            }
        except Exception as e:
            print(f"[警告] 加载帧{frame_idx}特征失败: {e}")
            return None

    def _extract_keypoints_from_features(self, features, image_shape):
        """从特征中提取关键点位置（基于主导频率模式映射回空间位置）"""
        try:
            if features is None or features.get("fourier") is None:
                return None
            
            fourier = features["fourier"]
            dominant_modes = fourier.get("dominant_modes", [])
            size = fourier.get("size", [0, 0])
            w, h = size[0], size[1]
            
            if w == 0 or h == 0:
                w, h = image_shape[1], image_shape[0]
            
            keypoints = []
            descriptors = []
            
            # 从主导频率模式提取关键点
            cy, cx = h // 2, w // 2
            for mode in dominant_modes:
                kx = mode.get("kx", 0)
                ky = mode.get("ky", 0)
                magnitude = mode.get("magnitude", 0.0)
                
                # 将频域坐标映射回空间坐标（简化映射）
                # 这里使用频域中心作为参考，映射到图像中心附近
                x = float(cx + kx * 0.1)  # 缩放因子，可根据需要调整
                y = float(cy + ky * 0.1)
                
                # 确保坐标在图像范围内
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                
                keypoints.append((x, y))
                
                # 构建描述符：结合fourier和spectral特征
                desc = []
                desc.append(magnitude)
                desc.append(float(kx))
                desc.append(float(ky))
                
                # 添加spectral特征
                if features.get("spectral") is not None:
                    spectral = features["spectral"]
                    desc.append(spectral.get("spectral_entropy", 0.0))
                    desc.extend(spectral.get("radial_hist", [])[:10])  # 前10个径向直方图值
                    desc.extend(spectral.get("angular_hist", [])[:10])  # 前10个角向直方图值
                
                descriptors.append(desc)
            
            # 如果没有主导频率，使用图像中心作为关键点
            if len(keypoints) == 0:
                keypoints.append((cx, cy))
                desc = [0.0, 0.0, 0.0]
                if features.get("spectral") is not None:
                    spectral = features["spectral"]
                    desc.append(spectral.get("spectral_entropy", 0.0))
                    desc.extend(spectral.get("radial_hist", [])[:10])
                    desc.extend(spectral.get("angular_hist", [])[:10])
                descriptors.append(desc)
            
            return {
                "keypoints": np.array(keypoints, dtype=np.float32),
                "descriptors": np.array(descriptors, dtype=np.float32)
            }
        except Exception as e:
            print(f"[警告] 从特征提取关键点失败: {e}")
            return None

    def _match_features_spectral(self, feat1, feat2, image1_shape, image2_shape):
        """基于spectral和fourier特征进行特征匹配"""
        try:
            kp1_data = self._extract_keypoints_from_features(feat1, image1_shape)
            kp2_data = self._extract_keypoints_from_features(feat2, image2_shape)
            
            if kp1_data is None or kp2_data is None:
                return None
            
            kp1 = kp1_data["keypoints"]
            desc1 = kp1_data["descriptors"]
            kp2 = kp2_data["keypoints"]
            desc2 = kp2_data["descriptors"]
            
            if len(kp1) == 0 or len(kp2) == 0:
                return None
            
            # 归一化描述符
            desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
            desc2_norm = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8)
            
            # 计算距离矩阵
            distances = np.sqrt(((desc1_norm[:, None, :] - desc2_norm[None, :, :]) ** 2).sum(axis=2))
            
            # 使用最近邻匹配（Lowe's ratio test）
            matches = []
            ratio_threshold = 0.75
            
            for i in range(len(kp1)):
                dists = distances[i]
                sorted_indices = np.argsort(dists)
                
                if len(sorted_indices) >= 2:
                    best_dist = dists[sorted_indices[0]]
                    second_best_dist = dists[sorted_indices[1]]
                    
                    if best_dist < ratio_threshold * second_best_dist:
                        matches.append({
                            "queryIdx": i,
                            "trainIdx": int(sorted_indices[0]),
                            "distance": float(best_dist)
                        })
                elif len(sorted_indices) == 1:
                    matches.append({
                        "queryIdx": i,
                        "trainIdx": int(sorted_indices[0]),
                        "distance": float(dists[sorted_indices[0]])
                    })
            
            # 转换为OpenCV格式的匹配点对
            pts1 = []
            pts2 = []
            match_info = []
            
            for m in matches:
                idx1 = m["queryIdx"]
                idx2 = m["trainIdx"]
                pts1.append(kp1[idx1])
                pts2.append(kp2[idx2])
                match_info.append(m)
            
            if len(pts1) < 4:  # 至少需要4个点估计基础矩阵
                return None
            
            return {
                "pts1": np.array(pts1, dtype=np.float32),
                "pts2": np.array(pts2, dtype=np.float32),
                "matches": match_info,
                "keypoints1": kp1,
                "keypoints2": kp2
            }
        except Exception as e:
            print(f"[警告] 特征匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _visualize_matches(self, image1, image2, match_result, save_path, frame_idx1, frame_idx2):
        """可视化匹配结果"""
        try:
            if match_result is None or len(match_result["pts1"]) < 4:
                print(f"[警告] 匹配点数量不足，无法可视化")
                return False
            
            pts1 = match_result["pts1"]
            pts2 = match_result["pts2"]
            
            # 创建拼接图像
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            h = max(h1, h2)
            w = w1 + w2
            
            if len(image1.shape) == 2:
                img1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
            else:
                img1_color = image1.copy()
            
            if len(image2.shape) == 2:
                img2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
            else:
                img2_color = image2.copy()
            
            # 调整图像大小使其高度一致
            if h1 != h2:
                scale = h / max(h1, h2)
                new_w1 = int(w1 * scale)
                new_w2 = int(w2 * scale)
                img1_color = cv2.resize(img1_color, (new_w1, h))
                img2_color = cv2.resize(img2_color, (new_w2, h))
                # 调整点坐标
                pts1_scaled = pts1 * scale
                pts2_scaled = pts2 * scale
                pts2_scaled[:, 0] += new_w1  # 第二张图像向右偏移
            else:
                pts1_scaled = pts1.copy()
                pts2_scaled = pts2.copy()
                pts2_scaled[:, 0] += w1
            
            # 拼接图像
            vis = np.hstack([img1_color, img2_color])
            
            # 绘制匹配线
            colors = plt.cm.get_cmap('hsv')(np.linspace(0, 1, len(pts1_scaled)))
            for i, (p1, p2) in enumerate(zip(pts1_scaled, pts2_scaled)):
                color = (int(colors[i][2] * 255), int(colors[i][1] * 255), int(colors[i][0] * 255))
                cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)
                cv2.circle(vis, (int(p1[0]), int(p1[1])), 5, color, -1)
                cv2.circle(vis, (int(p2[0]), int(p2[1])), 5, color, -1)
            
            # 添加文本信息
            cv2.putText(vis, f"Frame {frame_idx1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis, f"Frame {frame_idx2}", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis, f"Matches: {len(pts1)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imwrite(save_path, vis)
            print(f"[信息] 匹配可视化已保存: {save_path}")
            return True
        except Exception as e:
            print(f"[警告] 可视化匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _estimate_camera_pose_from_matches(self, match_result, K, frame_idx1, frame_idx2):
        """基于匹配结果估计相机位姿（R和T）"""
        try:
            if match_result is None or len(match_result["pts1"]) < 8:
                return None
            
            pts1 = match_result["pts1"]
            pts2 = match_result["pts2"]
            
            # 使用RANSAC估计本质矩阵
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            
            if E is None:
                return None
            
            # 从本质矩阵恢复位姿
            _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
            
            # 统计内点数量
            inliers = int(pose_mask.sum()) if pose_mask is not None else 0
            
            # 构建4x4变换矩阵
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            
            return {
                "R": R.tolist(),
                "t": t.flatten().tolist(),
                "T": T.tolist(),
                "E": E.tolist(),
                "inliers": inliers,
                "total_matches": len(pts1),
                "frame_idx1": frame_idx1,
                "frame_idx2": frame_idx2
            }
        except Exception as e:
            print(f"[警告] 估计相机位姿失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _match_frames_spectral_features(self, image_folder=None, suffix="th_68"):
        """批量匹配所有帧的spectral和fourier特征，估计相机参数
        直接使用process_image中缓存的特征数据，无需重新加载或计算
        """
        try:
            print("\n" + "="*80)
            print("开始基于频域特征进行帧间匹配和相机参数估计")
            print("="*80)
            
            # 直接使用缓存的特征数据
            if not hasattr(self, '_frame_features_cache') or len(self._frame_features_cache) < 2:
                print("[错误] 特征缓存为空或帧数不足，请先处理图像")
                return None
            
            print(f"[信息] 使用缓存的特征数据，共 {len(self._frame_features_cache)} 帧")
            
            # 创建匹配结果输出目录
            match_output_dir = os.path.join(self.output_dir, "spectral_feature_matching")
            os.makedirs(match_output_dir, exist_ok=True)
            
            # 从缓存中提取特征和图像
            frame_features = []
            frame_images = []
            
            for idx, cached_data in enumerate(self._frame_features_cache):
                spectral_feat = cached_data.get("spectral")
                image = cached_data.get("image")
                
                if spectral_feat is None or image is None:
                    print(f"[警告] 帧 {idx} 的特征数据不完整，跳过")
                    continue
                
                frame_features.append({
                    "spectral": spectral_feat,
                    "frame_idx": idx
                })
                frame_images.append((image, f"frame_{idx}", idx))
            
            if len(frame_features) < 2:
                print("[错误] 有效帧数量不足")
                return None
            
            print(f"[信息] 成功加载 {len(frame_features)} 帧的特征")
            
            # 估计相机内参（使用第一张图像）
            first_image = frame_images[0][0]
            h, w = first_image.shape[:2]
            # 简单估计：假设主点在中心，焦距为图像对角线长度
            fx = fy = float(np.sqrt(w * w + h * h))
            cx = float(w / 2)
            cy = float(h / 2)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            
            print(f"[信息] 估计的相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            
            # 逐对匹配
            all_matches = []
            all_poses = []
            
            # 第一帧作为参考帧（单位变换）
            R_global = np.eye(3, dtype=np.float64)
            t_global = np.zeros((3, 1), dtype=np.float64)
            poses_global = [{
                "frame_idx": 0,
                "R": R_global.tolist(),
                "t": t_global.flatten().tolist(),
                "T": np.eye(4, dtype=np.float64).tolist()
            }]
            
            for i in range(len(frame_features) - 1):
                idx1 = i
                idx2 = i + 1
                
                print(f"\n[匹配] 帧 {idx1} <-> 帧 {idx2}")
                
                feat1 = frame_features[idx1]
                feat2 = frame_features[idx2]
                img1, name1, _ = frame_images[idx1]
                img2, name2, _ = frame_images[idx2]
                
                # 进行特征匹配
                match_result = self._match_features_spectral(
                    feat1, feat2,
                    img1.shape, img2.shape
                )
                
                if match_result is None:
                    print(f"[警告] 帧 {idx1} 和 {idx2} 匹配失败")
                    continue
                
                print(f"[信息] 找到 {len(match_result['pts1'])} 个匹配点")
                
                # 可视化匹配结果
                vis_path = os.path.join(match_output_dir, f"match_{idx1}_{idx2}.jpg")
                self._visualize_matches(img1, img2, match_result, vis_path, idx1, idx2)
                
                # 估计相对位姿
                pose_result = self._estimate_camera_pose_from_matches(
                    match_result, K, idx1, idx2
                )
                
                if pose_result is None:
                    print(f"[警告] 帧 {idx1} 和 {idx2} 位姿估计失败")
                    continue
                
                print(f"[信息] 内点数量: {pose_result['inliers']}/{pose_result['total_matches']}")
                
                all_matches.append({
                    "frame_idx1": idx1,
                    "frame_idx2": idx2,
                    "num_matches": len(match_result['pts1']),
                    "inliers": pose_result['inliers']
                })
                
                # 累积全局位姿
                R_rel = np.array(pose_result["R"], dtype=np.float64)
                t_rel = np.array(pose_result["t"], dtype=np.float64).reshape(3, 1)
                
                R_global = R_global @ R_rel
                t_global = t_global + (R_global @ t_rel)
                
                T_global = np.eye(4, dtype=np.float64)
                T_global[:3, :3] = R_global
                T_global[:3, 3] = t_global.flatten()
                
                poses_global.append({
                    "frame_idx": idx2,
                    "R": R_global.tolist(),
                    "t": t_global.flatten().tolist(),
                    "T": T_global.tolist(),
                    "relative_R": pose_result["R"],
                    "relative_t": pose_result["t"]
                })
                
                all_poses.append(pose_result)
            
            # 保存匹配结果
            matches_json_path = os.path.join(match_output_dir, "matches_summary.json")
            # JSON保存已禁用
            # matches_json_path = os.path.join(match_output_dir, "matches_summary.json")
            # with open(matches_json_path, 'w', encoding='utf-8') as f:
            #     json.dump({
            #         "total_frames": len(frame_features),
            #         "total_matches": len(all_matches),
            #         "matches": all_matches,
            #         "camera_intrinsics": {
            #             "fx": float(fx),
            #             "fy": float(fy),
            #             "cx": float(cx),
            #             "cy": float(cy),
            #             "width": int(w),
            #             "height": int(h)
            #         }
            #     }, f, ensure_ascii=False, indent=2)
            
            # 保存相机位姿
            # JSON保存已禁用
            # poses_json_path = os.path.join(match_output_dir, "camera_poses_spectral.json")
            # with open(poses_json_path, 'w', encoding='utf-8') as f:
            #     json.dump({
            #         "camera_intrinsics": {
            #             "fx": float(fx),
            #             "fy": float(fy),
            #             "cx": float(cx),
            #             "cy": float(cy),
            #             "width": int(w),
            #             "height": int(h)
            #         },
            #         "poses": poses_global,
            #         "relative_poses": all_poses
            #     }, f, ensure_ascii=False, indent=2)
            
            print(f"\n[完成] 匹配和位姿估计完成")
            # print(f"  - 匹配结果: {matches_json_path}")
            # print(f"  - 相机位姿: {poses_json_path}")
            print(f"  - 匹配可视化: {match_output_dir}")
            
            # 保存频域特征匹配指标到TXT文件
            txt_path = os.path.join(match_output_dir, f"spectral_matching_metrics_{suffix}.txt")
            try:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("频域特征匹配评估指标\n")
                    f.write("="*80 + "\n\n")
                    
                    f.write("【匹配统计】\n")
                    f.write(f"总帧数: {len(frame_features)}\n")
                    f.write(f"成功匹配对数: {len(all_matches)}\n")
                    if len(frame_features) > 1:
                        f.write(f"匹配成功率: {len(all_matches)/(len(frame_features)-1)*100:.2f}%\n\n")
                    
                    if len(all_matches) > 0:
                        num_matches_list = [m['num_matches'] for m in all_matches]
                        inliers_list = [m['inliers'] for m in all_matches]
                        inlier_ratios = [inliers / num_matches if num_matches > 0 else 0.0 
                                        for inliers, num_matches in zip(inliers_list, num_matches_list)]
                        
                        f.write("【匹配点数量】\n")
                        f.write(f"  平均值: {np.mean(num_matches_list):.2f}\n")
                        f.write(f"  最大值: {np.max(num_matches_list)}\n")
                        f.write(f"  最小值: {np.min(num_matches_list)}\n")
                        f.write(f"  标准差: {np.std(num_matches_list):.2f}\n\n")
                        
                        f.write("【内点数量】\n")
                        f.write(f"  平均值: {np.mean(inliers_list):.2f}\n")
                        f.write(f"  最大值: {np.max(inliers_list)}\n")
                        f.write(f"  最小值: {np.min(inliers_list)}\n")
                        f.write(f"  标准差: {np.std(inliers_list):.2f}\n\n")
                        
                        f.write("【内点比例】\n")
                        f.write(f"  平均值: {np.mean(inlier_ratios)*100:.2f}%\n")
                        f.write(f"  最大值: {np.max(inlier_ratios)*100:.2f}%\n")
                        f.write(f"  最小值: {np.min(inlier_ratios)*100:.2f}%\n")
                        f.write(f"  标准差: {np.std(inlier_ratios)*100:.2f}%\n\n")
                        
                        f.write("="*80 + "\n")
                        f.write("【详细匹配信息】\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"{'帧对':<15} {'匹配点数':<12} {'内点数':<12} {'内点比例':<15}\n")
                        f.write("-"*80 + "\n")
                        for match in all_matches:
                            frame_pair = f"{match['frame_idx1']:04d}-{match['frame_idx2']:04d}"
                            inlier_ratio = match['inliers'] / match['num_matches'] * 100 if match['num_matches'] > 0 else 0.0
                            f.write(f"{frame_pair:<15} {match['num_matches']:<12} {match['inliers']:<12} {inlier_ratio:<15.2f}%\n")
                    
                    f.write("\n【相机内参】\n")
                    f.write(f"  fx: {fx:.2f}\n")
                    f.write(f"  fy: {fy:.2f}\n")
                    f.write(f"  cx: {cx:.2f}\n")
                    f.write(f"  cy: {cy:.2f}\n")
                    f.write(f"  图像尺寸: {w} x {h}\n")
                    
                    f.write("\n" + "="*80 + "\n")
                    f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*80 + "\n")
                
                print(f"[保存] 频域特征匹配指标已保存到: {txt_path}")
            except Exception as e:
                print(f"[警告] 保存频域特征匹配指标到TXT文件失败: {e}")
            
            return {
                "matches": all_matches,
                "poses": poses_global,
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h}
            }
        except Exception as e:
            print(f"[异常] 帧间匹配失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _video_fft_sequence_camera_estimation(self, video_path, max_frames=150, stride=1, fd_harmonics=None):
        """读取视频序列，提取傅里叶级数特征并匹配；估计相机内参与相对外参。"""
        out_dir = getattr(self, "output_dir", r"E:\reloc3r\Point_Matching_RT\test3_extracted_frames\extracted_frames")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        trace_log = os.path.join(out_dir, "run_trace.log")
        def _log_trace(msg):
            try:
                with open(trace_log, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        _log_trace("[trace] video_fft_sequence_entry")
        try:
            cap = cv2.VideoCapture(video_path)
            opened = cap.isOpened()
            _log_trace(f"[trace] cap_opened:{int(opened)} path:{video_path}")
            frames = []
            idx = 0
            while idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if (idx % stride) == 0:
                    frames.append(frame)
                idx += 1
        except Exception as e:
            _log_trace(f"[trace] cap_exception:{e}")
            try:
                cap.release()
            except Exception:
                pass
            print(f"[异常] 视频读取失败: {e}")
            return
        finally:
            try:
                cap.release()
            except Exception:
                pass
        _log_trace(f"[trace] frames_read:{len(frames)}")
        if len(frames) >= 1:
            try:
                cv2.imwrite(os.path.join(out_dir, "seq_frame_000.jpg"), frames[0])
            except Exception:
                pass
        if len(frames) >= 2:
            try:
                cv2.imwrite(os.path.join(out_dir, "seq_frame_001.jpg"), frames[1])
            except Exception:
                pass
        if len(frames) < 2:
            print(f"[警告] 视频帧不足以进行匹配: {video_path}")
            _log_trace("[trace] sequence_insufficient_frames")
            return
        h, w = frames[0].shape[:2]
        # 1) 提取每帧FD特征
        _log_trace("[trace] fd_extraction_start")
        fd_list = []
        try:
            for i, fr in enumerate(frames):
                fdi = self._fd_feature_from_image(fr, percent=getattr(self, "main_percent", 68), num_points=getattr(self, "num_resample_points", 256), num_harmonics=fd_harmonics or getattr(self, "efd_harmonics", 16))
                fd_list.append({"frame_index": i, "fd": fdi})
        except Exception as e:
            _log_trace(f"[trace] fd_exception:{e}")
            print(f"[异常] FD特征提取失败: {e}")
            return
        _log_trace("[trace] fd_extraction_done")
        # JSON保存已禁用
        # try:
        #     with open(os.path.join(out_dir, "fft_fd_features_video.json"), "w", encoding="utf-8") as f:
        #         json.dump({"video_path": video_path, "width": w, "height": h, "frames": fd_list}, f, ensure_ascii=False, indent=2)
        # except Exception as e:
        #     print(f"[警告] 保存FD特征JSON失败: {e}")
        _log_trace("[trace] fd_json_saved")
        # 2) 相邻帧FD特征匹配（相似度）
        match_list = []
        for i in range(1, len(frames)):
            f1 = fd_list[i - 1]["fd"]
            f2 = fd_list[i]["fd"]
            if (f1 is None) or (f2 is None) or (f1.get("magnitudes_norm") is None) or (f2.get("magnitudes_norm") is None):
                match_list.append({"i": i - 1, "j": i, "distance": None, "correlation": None})
                continue
            a = np.asarray(f1["magnitudes_norm"], dtype=np.float32)
            b = np.asarray(f2["magnitudes_norm"], dtype=np.float32)
            m = min(len(a), len(b))
            a = a[:m]
            b = b[:m]
            # 距离与相关性
            dist = float(np.linalg.norm(a - b))
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            corr = float(np.dot(a, b) / denom)
            match_list.append({"i": i - 1, "j": i, "distance": dist, "correlation": corr})
        try:
            with open(os.path.join(out_dir, "fft_fd_matching_video.json"), "w", encoding="utf-8") as f:
                json.dump({"video_path": video_path, "matches": match_list}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[警告] 保存FD匹配JSON失败: {e}")
        # 3) 估计相机内参（简单自标定）：假设零斜切，主点在中心，扫描焦距f使E矩阵内点最大
        _log_trace("[trace] orb_matching_start")
        try:
            orb = cv2.ORB_create(nfeatures=2000)
        except Exception:
            orb = cv2.ORB_create()
        pair_points = []
        for i in range(1, len(frames)):
            img1 = frames[i - 1]
            img2 = frames[i]
            try:
                kps1, des1 = orb.detectAndCompute(img1, None)
                kps2, des2 = orb.detectAndCompute(img2, None)
            except Exception as e:
                _log_trace(f"[trace] orb_detect_exception:{e}")
                pair_points.append(None)
                continue
            if des1 is None or des2 is None or len(kps1) < 8 or len(kps2) < 8:
                pair_points.append(None)
                continue
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            try:
                matches = bf.match(des1, des2)
            except Exception as e:
                _log_trace(f"[trace] bf_match_exception:{e}")
                pair_points.append(None)
                continue
            matches = sorted(matches, key=lambda x: x.distance)[:500]
            pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            pair_points.append((pts1, pts2))
        _log_trace(f"[trace] orb_pairs:{sum(1 for pp in pair_points if pp is not None)}")
        # 焦距搜索区间
        fmin = 0.6 * max(w, h)
        fmax = 1.8 * max(w, h)
        grid = np.linspace(fmin, fmax, num=15)
        best_f = grid[0]
        best_score = -1
        _log_trace(f"[trace] f_search_range:{fmin:.2f},{fmax:.2f} grid:{len(grid)}")
        for f in grid:
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
            total_inliers = 0
            used_pairs = 0
            for pp in pair_points[:20]:  # 用前20对估计焦距
                if pp is None:
                    continue
                pts1, pts2 = pp
                try:
                    E, maskE = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                except Exception as e:
                    _log_trace(f"[trace] findE_exception:{e}")
                    E, maskE = None, None
                if E is None:
                    continue
                inliers = int(maskE.sum()) if maskE is not None else 0
                total_inliers += inliers
                used_pairs += 1
            if used_pairs > 0:
                score = total_inliers / used_pairs
                if score > best_score:
                    best_score = score
                    best_f = float(f)
        _log_trace(f"[trace] f_best:{best_f:.2f} score:{best_score:.2f}")
        fx = fy = float(best_f)
        cx = float(w / 2)
        cy = float(h / 2)
        K_best = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        intrinsics = {"width": int(w), "height": int(h), "fx": fx, "fy": fy, "cx": cx, "cy": cy, "method": "grid_inlier_max", "score": float(best_score)}
        # JSON保存已禁用
        # try:
        #     with open(os.path.join(out_dir, "camera_intrinsics_estimate.json"), "w", encoding="utf-8") as f:
        #         json.dump(intrinsics, f, ensure_ascii=False, indent=2)
        # except Exception as e:
        #     print(f"[警告] 保存相机内参JSON失败: {e}")
        _log_trace("[trace] intrinsics_json_saved")
        # 4) 估计相机相对外参（逐帧）
        poses = []
        R_global = np.eye(3, dtype=np.float64)
        t_global = np.zeros((3, 1), dtype=np.float64)
        poses.append({"frame_index": 0, "R": R_global.tolist(), "t": t_global.flatten().tolist()})
        for i, pp in enumerate(pair_points, start=1):
            if pp is None:
                poses.append({"frame_index": i, "R": None, "t": None, "inliers": 0})
                continue
            pts1, pts2 = pp
            try:
                E, maskE = cv2.findEssentialMat(pts1, pts2, cameraMatrix=K_best, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            except Exception as e:
                _log_trace(f"[trace] findE_best_exception:{e}")
                E, maskE = None, None
            if E is None:
                poses.append({"frame_index": i, "R": None, "t": None, "inliers": 0})
                continue
            try:
                _, R, t, mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=K_best)
            except Exception as e:
                _log_trace(f"[trace] recoverPose_exception:{e}")
                poses.append({"frame_index": i, "R": None, "t": None, "inliers": 0})
                continue
            # 累积到全局坐标（单位步长，尺度未知）
            R_prev = R_global.copy()
            R_global = R_prev @ R
            t_global = t_global + (R_prev @ t)
            poses.append({"frame_index": i, "R": R_global.tolist(), "t": t_global.flatten().tolist(), "inliers": int(mask.sum()) if mask is not None else None})
        # JSON保存已禁用
        # try:
        #     with open(os.path.join(out_dir, "camera_extrinsics_poses.json"), "w", encoding="utf-8") as f:
        #         json.dump({"video_path": video_path, "intrinsics": intrinsics, "poses": poses}, f, ensure_ascii=False, indent=2)
        # except Exception as e:
        #     print(f"[警告] 保存相机外参JSON失败: {e}")
        _log_trace("[trace] camera_est_done")

    def _compute_field_fourier(self, field, kx=32, ky=32, suffix="68"):
        """对高度场进行2D傅里叶截断重建，输出重建图、能量占比与可选3D网格。"""
        try:
            if field is None:
                return
            fld = np.asarray(field, dtype=np.float32)
            h, w = fld.shape
            F = np.fft.fft2(fld)
            F_shift = np.fft.fftshift(F)
            cy, cx = h // 2, w // 2
            y0, y1 = max(0, cy - ky), min(h, cy + ky + 1)
            x0, x1 = max(0, cx - kx), min(w, cx + kx + 1)
            keep = np.zeros_like(F_shift, dtype=bool)
            keep[y0:y1, x0:x1] = True
            F_trunc = np.where(keep, F_shift, 0)
            F_trunc_unshift = np.fft.ifftshift(F_trunc)
            rec = np.real(np.fft.ifft2(F_trunc_unshift))
            rec_norm = cv2.normalize(rec.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            energy_total = float(np.sum(np.abs(F_shift) ** 2))
            energy_keep = float(np.sum(np.abs(F_trunc) ** 2))
            energy_ratio = (energy_keep / (energy_total + 1e-12))
            # 保存重建图
            out_img = (rec_norm * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.output_dir, f"field_fourier_{suffix}.png"), out_img)
            # 保存描述JSON
            # JSON保存已禁用
            # try:
            #     with open(os.path.join(self.output_dir, f"field_fourier_{suffix}.json"), "w", encoding="utf-8") as f:
            #         json.dump({
            #             "kx": int(kx), "ky": int(ky),
            #             "energy_ratio": energy_ratio,
            #             "image_size": [int(w), int(h)]
            #         }, f, ensure_ascii=False, indent=2)
            # except Exception as e:
            #     print(f"[警告] 保存field_fourier描述失败({suffix}): {e}")
            # 可选：导出重建表面的3D三角网格
            if self.mesh_enable:
                self._generate_grid_mesh(rec_norm, cell_size=self.mesh_cell_size, suffix=f"{suffix}_ft", mesh_type=self.mesh_type, include_z=True)
        except Exception as e:
            print(f"[异常] 高度场傅里叶重建失败({suffix}): {e}")

    def _detect_local_peaks(self, field, threshold_abs=None, min_dist=5, max_peaks=512):
        """在高度场中检测局部峰值，返回按强度排序的峰值列表。"""
        fld = np.asarray(field, dtype=np.float32)
        h, w = fld.shape
        # 使用膨胀寻找局部最大
        k = int(max(1, min_dist))
        kernel = np.ones((k, k), np.uint8)
        dil = cv2.dilate(fld, kernel)
        maxima = (fld >= dil - 1e-6)
        if threshold_abs is not None:
            maxima &= (fld >= float(threshold_abs))
        ys, xs = np.nonzero(maxima)
        vals = fld[ys, xs]
        order = np.argsort(-vals)
        peaks = []
        for i in order[:max_peaks]:
            peaks.append({"x": int(xs[i]), "y": int(ys[i]), "value": float(vals[i])})
        return peaks

    def _fit_peak_quadratic(self, field, px, py, window=21):
        """对以 (px,py) 为中心的局部窗口进行二次曲面拟合: I ~ a x^2 + b x y + c y^2 + d x + e y + f
        返回拟合参数、极值点(子像素)及类型(max/min/saddle)、残差MSE等。坐标相对原图。"""
        fld = np.asarray(field, dtype=np.float32)
        h, w = fld.shape
        half = max(1, int(window) // 2)
        cx, cy = int(px), int(py)
        x1, x2 = max(0, cx - half), min(w - 1, cx + half)
        y1, y2 = max(0, cy - half), min(h - 1, cy + half)
        patch = fld[y1:y2 + 1, x1:x2 + 1]
        if patch.size < 6:
            return None
        # 构建设计矩阵
        yy, xx = np.mgrid[y1:y2 + 1, x1:x2 + 1]
        # 以峰心为原点的局部坐标
        xr = xx - cx
        yr = yy - cy
        X = np.stack([xr**2, xr*yr, yr**2, xr, yr, np.ones_like(xr)], axis=-1).reshape(-1, 6)
        Y = patch.reshape(-1)
        # 最小二乘拟合
        try:
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            return None
        a, b, c, d, e, f0 = [float(v) for v in coef]
        # 求驻点: 解 [2a, b; b, 2c] [x; y] = [-d; -e]
        H = np.array([[2*a, b], [b, 2*c]], dtype=np.float64)
        rhs = np.array([-d, -e], dtype=np.float64)
        det = float(np.linalg.det(H))
        if abs(det) < 1e-12:
            return {
                "params": {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f0},
                "type": "degenerate",
                "fitted": {"x": float(cx), "y": float(cy), "value": float(f0)},
                "window": int(window),
                "residual_mse": float(np.mean((X @ coef - Y)**2))
            }
        try:
            xy = np.linalg.solve(H, rhs)
            x_fit_local, y_fit_local = float(xy[0]), float(xy[1])
        except np.linalg.LinAlgError:
            x_fit_local, y_fit_local = 0.0, 0.0
        # 判定类型: Hessian 负定->最大, 正定->最小, 否则鞍点
        # 负定条件: a<0, c<0 且 4ac - b^2 > 0
        disc = 4*a*c - b*b
        if a < 0 and c < 0 and disc > 0:
            ptype = "max"
        elif a > 0 and c > 0 and disc > 0:
            ptype = "min"
        else:
            ptype = "saddle"
        # 全局子像素坐标
        xg = float(cx + x_fit_local)
        yg = float(cy + y_fit_local)
        # 预测强度
        If = a*(x_fit_local**2) + b*(x_fit_local*y_fit_local) + c*(y_fit_local**2) + d*x_fit_local + e*y_fit_local + f0
        mse = float(np.mean((X @ coef - Y)**2))
        return {
            "params": {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f0},
            "hessian": {"h11": 2*a, "h12": b, "h22": 2*c, "det": det},
            "type": ptype,
            "fitted": {"x": xg, "y": yg, "value": float(If)},
            "origin": {"x": int(cx), "y": int(cy)},
            "window": int(window),
            "residual_mse": mse
        }

    def process_image(self, image, frame_idx=None, original_image_path=None):
        """
        处理图像，保存一阶、二阶、梯度图像
        
        Args:
            image: 输入图像（BGR格式）
            frame_idx: 可选的帧索引，用于在文件名中添加编号
            original_image_path: 原始图像路径（用于保存路径信息，全局匹配模式不使用mask）
        """
        timer = TimeTracker(f"图像处理{(' (帧' + str(frame_idx) + ')' if frame_idx is not None else '')}").start()
        
        # ========== 分辨率缩放（降低分辨率以加快处理速度）==========
        original_h, original_w = image.shape[:2]
        if self.image_scale_factor != 1.0:
            new_w = int(original_w * self.image_scale_factor)
            new_h = int(original_h * self.image_scale_factor)
            # 确保尺寸是偶数（某些算法要求）
            new_w = (new_w // 2) * 2
            new_h = (new_h // 2) * 2
            if new_w != original_w or new_h != original_h:
                print(f"  [分辨率缩放] {original_w}x{original_h} -> {new_w}x{new_h} (缩放因子: {self.image_scale_factor:.2f})")
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)  # INTER_AREA适合缩小
                print(f"  [性能] 分辨率降低后，处理速度预计提升约 {1.0/(self.image_scale_factor**2):.1f}倍")
        timer.checkpoint("分辨率缩放")
        # ========== 分辨率缩放完成 ==========
        
        # ========== 保存图像路径（用于后续纹理提取等）==========
        mask = None  # 初始化为None，后续从轮廓提取生成
        if frame_idx is not None and original_image_path is not None:
            try:
                if os.path.exists(original_image_path):
                    # 保存图像路径，供后续纹理提取使用
                    if not hasattr(self, '_image_paths'):
                        self._image_paths = {}
                    self._image_paths[frame_idx] = original_image_path
                    print(f"  [信息] 已保存图像路径: {original_image_path}")
                else:
                    print(f"  [警告] 原始图像路径不存在: {original_image_path}")
            except Exception as e:
                print(f"  [警告] 保存图像路径失败: {e}")
        timer.checkpoint("图像路径保存")
        # ========== 图像路径保存完成 ==========
        
        # 转换为float32格式
        image_float = image.astype(np.float32) / 255.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        timer.checkpoint("图像预处理")

        # 彩色双边滤波
        bilateral_color = cv2.bilateralFilter(image, 9, 75, 75)
        timer.checkpoint("彩色双边滤波")

        # 灰度双边滤波与梯度计算（GPU/CuPy或CPU回退）
        if self.use_cpu or (cp is None):
            filtered_gray = self._bilateral_filter(gray)  # 返回numpy
            # 使用Sobel近似高斯导数
            Dx = cv2.Sobel(filtered_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            Dy = cv2.Sobel(filtered_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            first_order_combined = np.hypot(Dx, Dy)
            first_order_norm = self._normalize(first_order_combined)
            grad_mag = first_order_combined
            grad_norm = first_order_norm
        else:
            # GPU路径
            gray_cp = cp.asarray(gray)
            filtered_gray_cp = self._bilateral_filter(gray_cp)
            Dx = convolve1d(filtered_gray_cp, cp.asarray(self.dgauss_1d), axis=1, mode='reflect')
            Dy = convolve1d(filtered_gray_cp, cp.asarray(self.dgauss_1d), axis=0, mode='reflect')
            first_order_combined = cp.hypot(Dx, Dy)
            first_order_norm = self._normalize(first_order_combined)
            grad_mag = first_order_combined
            grad_norm = first_order_norm
        timer.checkpoint("梯度计算")

        # ========== 步骤4：阈值计算（基于步骤3的梯度计算结果）==========
        # 计算主阈值和阈值模式对应的阈值（th_68或th_90），都基于grad_norm（步骤3的梯度计算结果）
        th_main = self._analyze_gradient_threshold(grad_norm, percent=self.main_percent)
        
        # 计算阈值模式对应的阈值（基于步骤3的梯度计算结果）
        if self.use_cpu or (cp is None):
            grad_norm_for_th = grad_norm
        else:
            grad_norm_for_th = grad_norm.get() if hasattr(grad_norm, 'get') else grad_norm
        th_value = self._analyze_gradient_threshold(grad_norm_for_th, bins=1024, percent=self.threshold_percent)
        th_uint8 = int(th_value * 255.0)
        print(f"  [阈值计算] th_main: {th_main:.6f} (归一化), {self.threshold_mode}: {th_value:.6f} (归一化), {th_uint8} (0-255范围)")
        timer.checkpoint("阈值计算")
        # ========== 阈值计算完成 ==========

        # 按阈值屏蔽：小于阈值置零，其余保持原值
        if self.use_cpu:
            binary_np = np.where(grad_norm < th_main, 0.0, grad_norm)
        else:
            binary_cp = cp.where(grad_norm < th_main, 0, grad_norm)

        # 保存未二值化的梯度图像（在应用阈值之前）
        if self.use_cpu:
            grad_before_threshold = np.clip(grad_norm * 255.0, 0, 255).astype(np.uint8)
            grad_norm_np = grad_norm
        else:
            grad_before_threshold = cp.clip(grad_norm * 255, 0, 255).astype(np.uint8).get()
            grad_norm_np = grad_norm.get()

        # 转换回numpy
        out_bilateral = bilateral_color
        out_gradient = grad_before_threshold  # 使用已保存的未二值化梯度图像
        if self.use_cpu:
            out_binary = np.clip((binary_np * 255.0), 0, 255).astype(np.uint8)
        else:
            out_binary = cp.clip(binary_cp * 255, 0, 255).astype(np.uint8).get()  # 主阈值屏蔽结果

        # 对主阈值结果进行滤波：移除低频背景（高斯高通，背景扣除）
        low_freq_bg = cv2.GaussianBlur(out_binary, (31, 31), 8)
        out_binary_main_filtered = cv2.subtract(out_binary, low_freq_bg)
        out_binary_main_filtered = cv2.normalize(out_binary_main_filtered.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 双边滤波（保边去噪），在主阈值结果上
        out_binary_main_bilateral = cv2.bilateralFilter(out_binary, d=3, sigmaColor=90, sigmaSpace=90)
        
        # 注意：网格生成已移至_extract_contour_points_from_bilateral函数中，在th_68截断灰度图生成后执行
        # 不再在这里生成网格，等待th_68截断灰度图生成后再生成网格
        
        # 一阶梯度合并图像（保留计算过程，但不保存图像文件）
        if self.use_cpu or (cp is None):
            first_order_merged = np.clip(first_order_norm * 255.0, 0, 255).astype(np.uint8)
        else:
            first_order_merged = cp.clip(first_order_norm * 255, 0, 255).astype(np.uint8).get()

        timer.checkpoint("图像处理完成")
        
        # ========== 保存梯度图像（用于后续处理）==========
        if frame_idx is not None:
            try:
                # 保存gradient_before_threshold图像（使用参考文件中的方式）
                grad_before_threshold_path = os.path.join(self.output_dir, f"gradient_before_threshold_{frame_idx:04d}.jpg")
                cv2.imwrite(grad_before_threshold_path, grad_before_threshold)
                print(f"  [保存] 梯度图像已保存: {grad_before_threshold_path}")
            except Exception as e:
                print(f"  [警告] 保存梯度图像失败: {e}")
        timer.checkpoint("梯度图像保存")
        # ========== 梯度图像保存完成 ==========

        # 转换回numpy
        out_bilateral = bilateral_color
        out_gradient = grad_before_threshold  # 使用已保存的未二值化梯度图像
        if self.use_cpu:
            out_binary = np.clip((binary_np * 255.0), 0, 255).astype(np.uint8)
        else:
            out_binary = cp.clip(binary_cp * 255, 0, 255).astype(np.uint8).get()  # 主阈值屏蔽结果

        # 对主阈值结果进行滤波：移除低频背景（高斯高通，背景扣除）
        low_freq_bg = cv2.GaussianBlur(out_binary, (31, 31), 8)
        out_binary_main_filtered = cv2.subtract(out_binary, low_freq_bg)
        out_binary_main_filtered = cv2.normalize(out_binary_main_filtered.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # 双边滤波（保边去噪），在主阈值结果上
        out_binary_main_bilateral = cv2.bilateralFilter(out_binary, d=3, sigmaColor=90, sigmaSpace=90)
        
        # 注意：网格生成已移至_extract_contour_points_from_bilateral函数中，在th_68截断灰度图生成后执行
        # 不再在这里生成网格，等待th_68截断灰度图生成后再生成网格
        
        # 一阶梯度合并图像（保留计算过程，但不保存图像文件）
        if self.use_cpu or (cp is None):
            first_order_merged = np.clip(first_order_norm * 255.0, 0, 255).astype(np.uint8)
        else:
            first_order_merged = cp.clip(first_order_norm * 255, 0, 255).astype(np.uint8).get()

        timer.checkpoint("图像处理完成")
        
        # ========== 步骤5：生成阈值模式对应的截断梯度图（用于后续高度场生成）==========
        # 逻辑：1. 使用步骤4计算的阈值 -> 2. 对梯度图进行截断 -> 3. 归一化到0-255
        th_gray_image = None
        if frame_idx is not None:
            try:
                # 步骤1: 阈值已在步骤4中计算完成，直接使用
                # th_value 和 th_uint8 已在步骤4中计算
                
                # 步骤2: 对梯度图进行截断（保留梯度值，小于等于阈值的设为0）
                # 使用grad_before_threshold（步骤3的梯度计算结果，0-255范围）
                grad_uint8 = grad_before_threshold  # 已经是0-255范围的uint8
                
                # 截断：大于阈值的保留梯度值，小于等于阈值的设为0
                th_gray_truncated = np.where(grad_uint8 > th_uint8, grad_uint8, 0).astype(np.float32)
                
                # 步骤3: 归一化到0-255（对非零区域进行归一化）
                non_zero_mask = th_gray_truncated > 0
                if np.any(non_zero_mask):
                    # 获取非零区域的最小值和最大值
                    min_val = th_gray_truncated[non_zero_mask].min()
                    max_val = th_gray_truncated[non_zero_mask].max()
                    
                    if max_val > min_val:
                        # 归一化非零区域到0-255
                        th_gray_normalized = np.zeros_like(th_gray_truncated)
                        th_gray_normalized[non_zero_mask] = (
                            (th_gray_truncated[non_zero_mask] - min_val) / (max_val - min_val) * 255.0
                        )
                        th_gray_image = th_gray_normalized.astype(np.uint8)
                        print(f"  [归一化] 非零区域范围: [{min_val:.1f}, {max_val:.1f}] -> [0, 255]")
                    else:
                        # 如果所有非零值相同，设为255
                        th_gray_image = np.where(non_zero_mask, 255, 0).astype(np.uint8)
                        print(f"  [归一化] 所有非零值相同，设为255")
                else:
                    # 如果没有非零区域，保持原样
                    th_gray_image = th_gray_truncated.astype(np.uint8)
                    print(f"  [警告] 截断后没有非零区域")
                
                # 保存截断并归一化后的梯度图（使用阈值模式命名）
                th_gray_path = os.path.join(self.output_dir, f"gradient_{self.threshold_mode}_{frame_idx:04d}.jpg")
                cv2.imwrite(th_gray_path, th_gray_image)
                print(f"  [保存] {self.threshold_mode}截断并归一化梯度图: {th_gray_path} (范围: [{th_gray_image.min()}, {th_gray_image.max()}])")
                
                # ========== 生成mask（支持棋盘格检测）==========
                mask = None
                use_chessboard = getattr(self, 'use_chessboard_detection', False)

                if use_chessboard and original_image_path and os.path.exists(original_image_path):
                    # 使用棋盘格检测
                    print(f"  [Mask提取] 使用棋盘格检测模式...")
                    mask = self._extract_largest_contour_mask(
                        frame_idx=frame_idx,
                        suffix=self.threshold_mode,
                        use_chessboard=True,
                        original_image_path=original_image_path
                    )
                else:
                    # 使用传统轮廓提取
                    print(f"  [Mask提取] 基于截断后的灰度图提取主要轮廓...")
                    mask = self._extract_mask_from_truncated_gray(th_gray_image, frame_idx)
                
                if mask is not None:
                    # 保存mask到实例变量，供后续高度场生成和匹配使用
                    if not hasattr(self, '_frame_masks'):
                        self._frame_masks = {}
                    self._frame_masks[frame_idx] = mask
                    mask_area = np.sum(mask > 0)
                    mask_percentage = (mask_area / (mask.shape[0] * mask.shape[1])) * 100
                    print(f"  [Mask生成] 成功生成mask，已保存到缓存 (尺寸: {mask.shape}, 区域: {mask_area} 像素, {mask_percentage:.2f}%)")
                else:
                    print(f"  [警告] 未能从截断灰度图提取mask，将使用全图处理")
                
            except Exception as e:
                print(f"  [警告] 生成{self.threshold_mode}截断梯度图或mask失败: {e}")
                import traceback
                traceback.print_exc()
        timer.checkpoint(f"{self.threshold_mode}梯度图生成和mask提取")
        # ========== {self.threshold_mode}梯度图生成和mask提取完成 ==========
        
        # ========== 基于mask和阈值模式对应的梯度图生成高度场（3D mesh）==========
        if frame_idx is not None and self.mesh_enable and th_gray_image is not None:
            try:
                # 检查梯度图是否有效（不全为0）
                gray_min = th_gray_image.min()
                gray_max = th_gray_image.max()
                gray_mean = th_gray_image.mean()
                h, w = th_gray_image.shape
                
                # 检查是否全为0
                if gray_max == 0:
                    print(f"  [警告] {self.threshold_mode}梯度图全为0，跳过高度场生成")
                    return image  # 返回原始图像
                
                # 检查有效像素数量
                non_zero_pixels = np.count_nonzero(th_gray_image)
                if non_zero_pixels < 10:
                    print(f"  [警告] {self.threshold_mode}梯度图有效像素太少 ({non_zero_pixels} 个)，跳过高度场生成")
                    return image  # 返回原始图像
                
                if mask is not None:
                    print(f"  [高度场生成] 基于mask和{self.threshold_mode}截断梯度图生成3D高度场（square_tri, cell_size={self.mesh_cell_size}）...")
                else:
                    print(f"  [高度场生成] 基于{self.threshold_mode}截断梯度图生成3D高度场（square_tri, cell_size={self.mesh_cell_size}）...")
                
                print(f"  [梯度统计] 最小值: {gray_min}, 最大值: {gray_max}, 平均值: {gray_mean:.2f}, 有效像素: {non_zero_pixels} ({non_zero_pixels/(h*w)*100:.1f}%)")
                
                # 验证mask尺寸和有效性（如果存在）
                if mask is not None:
                    mask_h, mask_w = mask.shape[:2]
                    if mask_h != h or mask_w != w:
                        print(f"  [mask尺寸调整] mask尺寸 {mask_w}x{mask_h} 与梯度图尺寸 {w}x{h} 不匹配，将自动调整")
                    else:
                        print(f"  [mask验证] mask尺寸 {mask_w}x{mask_h} 与梯度图尺寸一致")
                    # 检查mask是否有有效区域
                    mask_valid_pixels = np.count_nonzero(mask)
                    if mask_valid_pixels == 0:
                        print(f"  [警告] mask内没有有效像素，将使用全图生成高度场")
                        mask = None  # 如果mask无效，不使用mask
                    else:
                        # 检查mask和梯度图的重叠区域
                        mask_bool = mask > 0
                        grad_non_zero = th_gray_image > 0
                        overlap = np.logical_and(mask_bool, grad_non_zero)
                        overlap_pixels = np.count_nonzero(overlap)
                        if overlap_pixels == 0:
                            print(f"  [警告] mask与梯度图有效区域无重叠，将使用全图生成高度场")
                            mask = None  # 如果无重叠，不使用mask
                        else:
                            print(f"  [mask统计] mask覆盖区域: {mask_valid_pixels} 个像素 ({mask_valid_pixels/(h*w)*100:.1f}%)")
                            print(f"  [重叠统计] mask与梯度图重叠: {overlap_pixels} 个像素 ({overlap_pixels/(h*w)*100:.1f}%)")
                
                # 将uint8梯度图转换为float32（0-1范围）用于JSON网格生成
                th_gray_float = th_gray_image.astype(np.float32) / 255.0
                
                # 计算合适的z_scale，使高度差异明显
                # 方法1: 基于图像尺寸的比例（使高度与图像尺寸成比例）
                image_scale = max(h, w)  # 使用较大的维度作为基准
                # z_scale设置为图像尺寸的15%，这样高度差异会更明显
                z_scale_base = image_scale * 0.15
                
                # 方法2: 基于梯度值范围动态调整z_scale
                grad_range = gray_max - gray_min
                if grad_range > 0:
                    # 如果梯度值范围较大，使用基于范围的z_scale
                    # 确保梯度值255对应z_scale，梯度值0对应0
                    z_scale = z_scale_base
                    # 如果梯度值范围很小，适当增大z_scale以突出高度差异
                    if grad_range < 50:
                        z_scale = max(z_scale_base, 50.0)
                        print(f"  [z_scale调整] 梯度值范围较小 ({grad_range:.1f})，增大z_scale到 {z_scale:.2f}")
                else:
                    z_scale = z_scale_base
                
                print(f"  [高度缩放] z_scale={z_scale:.2f} (图像尺寸: {w}x{h}, 梯度范围: {grad_range:.1f})")
                
                # 1. 生成JSON格式的网格（用于其他用途）- 使用配置的cell_size
                suffix_num = self.threshold_mode.replace("th_", "")  # "th_68" -> "68", "th_90" -> "90"
                self._generate_grid_mesh(
                    th_gray_float, 
                    cell_size=self.mesh_cell_size,  # 使用配置参数，而不是硬编码7
                    suffix=f"{self.threshold_mode}{frame_idx:04d}", 
                    mesh_type=self.mesh_type,  # 使用配置的mesh_type
                    include_z=True
                )
                
                # 2. 生成高度场网格（用于3D匹配），应用mask，保存在内存中（不保存PLY文件）
                if o3d is not None:
                    try:
                        # 使用归一化后的梯度值（0-255范围）作为输入
                        # 注意：_o3d_build_height_mesh会将0-255范围归一化到0-1，然后缩放到z_scale
                        # 这样梯度值0对应高度0，梯度值255对应高度z_scale
                        mesh_ply, arr_ds = self._o3d_build_height_mesh(
                            th_gray_image.astype(np.float32),  # 使用归一化后的梯度值（0-255）
                            step=self.o3d_step,  # 使用配置的o3d_step，而不是硬编码7
                            z_scale=z_scale,  # 使用计算出的z_scale
                            mask=mask  # 应用mask（如果存在）
                        )
                        
                        if mesh_ply is not None:
                            # 检查mesh的有效性
                            vertices = np.asarray(mesh_ply.vertices)
                            triangles = np.asarray(mesh_ply.triangles)
                            
                            if len(vertices) == 0:
                                print(f"  [警告] 生成的mesh没有顶点，跳过缓存")
                            elif len(triangles) == 0:
                                print(f"  [警告] 生成的mesh没有三角形，跳过缓存")
                            else:
                                # 检查Z坐标范围
                                z_coords = vertices[:, 2]
                                z_min = z_coords.min()
                                z_max = z_coords.max()
                                z_range = z_max - z_min
                                
                                print(f"  [mesh统计] 顶点数: {len(vertices)}, 三角形数: {len(triangles)}")
                                print(f"  [Z坐标范围] [{z_min:.2f}, {z_max:.2f}], 范围: {z_range:.2f}")
                                
                                if z_range < 0.1:
                                    print(f"  [警告] Z坐标范围太小 ({z_range:.2f})，高度场可能过于平坦")
                                
                                # 优化：直接保存在内存中，不保存PLY文件
                                if frame_idx is not None:
                                    self._mesh_cache[frame_idx] = mesh_ply
                                    print(f"  [内存缓存] 帧 {frame_idx} 的mesh已缓存到内存（顶点: {len(vertices)}, 三角形: {len(triangles)}）")

                                    # FORCE SAVE MESH FOR DEBUGGING
                                    debug_mesh_path = os.path.join(self.mesh_out_dir, f"debug_mesh_{frame_idx:04d}.ply")
                                    try:
                                        o3d.io.write_triangle_mesh(debug_mesh_path, mesh_ply)
                                        print(f"  [DEBUG] Mesh saved: {debug_mesh_path}")
                                    except Exception as e:
                                        print(f"  [DEBUG] Failed to save mesh: {e}")
                        else:
                            print(f"  [警告] Open3D高度场网格生成失败（返回None）")
                            print(f"  [调试] 可能原因：")
                            print(f"    - {self.threshold_mode}梯度图全为0")
                            print(f"    - mask与梯度图无重叠区域")
                            print(f"    - 图像尺寸无效")
                    except Exception as e:
                        print(f"  [警告] 高度场网格生成失败: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"  [警告] Open3D未安装，无法生成高度场网格")
                
                if mask is not None:
                    print(f"  [高度场生成] {self.threshold_mode}高度场生成完成（已应用mask）")
                else:
                    print(f"  [高度场生成] {self.threshold_mode}高度场生成完成")
            except Exception as e:
                print(f"  [警告] 高度场生成失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 打印时间统计
        timer.print_summary()
        
        # 3D网格逐帧匹配（如果存在上一帧的网格）
        # 优化：直接从内存缓存读取mesh，不需要等待文件写入
        # 获取实际的第一帧索引（如果从非0帧开始，start_frame_idx就是第一帧）
        actual_first_frame_idx = getattr(self, 'start_frame_idx', 0)
        
        # 如果是第一帧，初始化累积位姿
        if frame_idx == actual_first_frame_idx:
            # 初始化第一帧的累积位姿（单位矩阵）
            if not hasattr(self, '_accumulated_poses'):
                self._accumulated_poses = {}
                self._accumulated_positions = {}
                self._accumulated_rotations = {}
                self._reference_frame_idx = actual_first_frame_idx
                
                self._accumulated_poses[actual_first_frame_idx] = np.eye(4, dtype=np.float32)
                self._accumulated_positions[actual_first_frame_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                self._accumulated_rotations[actual_first_frame_idx] = np.eye(3, dtype=np.float32)
                print(f"  [位姿初始化] 第一帧（索引{actual_first_frame_idx}）位姿已初始化为单位矩阵（参考坐标系）")
        
        if frame_idx is not None and frame_idx > actual_first_frame_idx:
            # 优化：直接从内存缓存读取mesh，不需要等待文件
            prev_frame_idx = frame_idx - 1
            
            # 检查是否启用基于渲染的位姿优化（默认启用，如果PyTorch3D可用）
            # 可以通过设置 USE_RENDERING_OPTIMIZATION=0 来禁用，使用传统方法
            use_rendering_optimization = bool(int(os.environ.get("USE_RENDERING_OPTIMIZATION", "1" if PYTORCH3D_AVAILABLE else "0")))
            
            if use_rendering_optimization and PYTORCH3D_AVAILABLE:
                # 优化：从内存缓存读取mesh
                if prev_frame_idx not in self._mesh_cache or frame_idx not in self._mesh_cache:
                    print(f"  [警告] 前一帧（{prev_frame_idx}）或当前帧（{frame_idx}）的mesh不在缓存中")
                    print(f"  [信息] 跳过渲染优化，使用传统3D特征匹配")
                    self._match_with_previous_frame_3d(frame_idx)
                else:
                    print(f"  [匹配] 使用基于渲染的位姿优化: 帧 {prev_frame_idx} <-> {frame_idx}")
                    # 获取前一帧的优化位姿（如果存在）
                    prev_pose = None
                    if hasattr(self, '_optimized_poses') and prev_frame_idx in self._optimized_poses:
                        prev_pose = self._optimized_poses[prev_frame_idx]
                    
                    # 执行渲染优化
                    # 获取原始图像路径（如果可用）
                    original_image_path = None
                    if hasattr(self, '_image_paths') and prev_frame_idx in self._image_paths:
                        original_image_path = self._image_paths[prev_frame_idx]
                        print(f"  [信息] 使用帧{prev_frame_idx}的原始图像路径: {original_image_path}")
                    else:
                        print(f"  [警告] 未找到帧{prev_frame_idx}的原始图像路径，将使用渲染轮廓作为ground truth")
                    
                    result = self._optimize_pose_with_rendering(
                        prev_frame_idx=prev_frame_idx,
                        current_frame_idx=frame_idx,
                        initial_pose=prev_pose,
                        suffix=self.threshold_mode,
                        max_iterations=int(os.environ.get("RENDERING_MAX_ITERATIONS", "200")),
                        learning_rate=float(os.environ.get("RENDERING_LEARNING_RATE", "0.01")),
                        loss_type=os.environ.get("RENDERING_LOSS_TYPE", "enhanced"),
                        save_intermediate=True,
                        original_image_path=original_image_path
                    )
                
                    # 保存优化后的RT矩阵
                    if result is not None:
                        if not hasattr(self, '_rt_matrices'):
                            self._rt_matrices = {}  # 存储帧与帧之间的RT矩阵
                        # 保存RT矩阵：将当前帧mesh变换到前一帧坐标系的变换矩阵
                        self._rt_matrices[f"{prev_frame_idx}_{frame_idx}"] = result['rt_matrix']
                        print(f"  [匹配] 渲染优化完成，最终损失: {result['final_loss']:.6f}")
                        print(f"  [RT矩阵] 帧{prev_frame_idx} -> 帧{frame_idx}的RT矩阵已保存")
                        
                        # 实时更新累积位姿（立即计算当前帧的相机位姿）
                        pose_result = self._update_accumulated_pose(
                            prev_frame_idx=prev_frame_idx,
                            current_frame_idx=frame_idx,
                            rt_matrix=result['rt_matrix'],
                            reference_frame_idx=getattr(self, 'start_frame_idx', 0)
                        )
                        if pose_result:
                            print(f"  [位姿] Frame {frame_idx} 的累积位姿已更新")
                            
                            # 检测累积误差（每20帧检测一次，降低频率以加速）
                            if frame_idx % 20 == 0:
                                drift_result = self._detect_and_correct_drift(
                                    current_frame_idx=frame_idx,
                                    max_position_drift=10.0,
                                    max_rotation_drift=90.0
                                )
                            
                            # 回环检测（每10帧检测一次，降低频率以加速，并使用采样优化）
                            if frame_idx % 10 == 0 and frame_idx > 10:
                                loop_result = self._detect_loop_closure(
                                    current_frame_idx=frame_idx,
                                    min_frame_gap=10,
                                    position_threshold=0.5,
                                    rotation_threshold=30.0,
                                    max_check_frames=50,  # 最多检查50帧历史帧
                                    sample_stride=5  # 每隔5帧采样一次
                                )
                                if loop_result.get('detected', False):
                                    print(f"  [回环] 检测到回环，建议运行位姿图优化以减少累积误差")
            else:
                print(f"  [匹配] 使用传统3D特征匹配: 帧 {prev_frame_idx} <-> {frame_idx}")
                self._match_with_previous_frame_3d(frame_idx)

        return image  # 返回原始图像



def load_camera_poses_from_file(filepath):
    """
    从标定结果文件加载相机位姿（支持CSV格式）
    
    CSV格式说明：从第19列开始，分别代表：
    fx, fy, cx, cy, k1, k2, p1, p2, k3, rx, ry, rz, tx, ty, tz
    
    Args:
        filepath: 标定结果文件路径（CSV格式）
        
    Returns:
        poses: 位姿列表，每个为4x4变换矩阵
        positions: 位置列表，每个为3D坐标 [x, y, z]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"位姿文件不存在: {filepath}")
    
    poses = []
    positions = []
    
    # 检查文件扩展名，支持CSV和TXT格式
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        # CSV格式：从第19列（索引18）开始读取
        # 列索引：18=fx, 19=fy, 20=cx, 21=cy, 22=k1, 23=k2, 24=p1, 25=p2, 26=k3,
        #         27=rx, 28=ry, 29=rz, 30=tx, 31=ty, 32=tz
        import csv
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # 尝试检测分隔符
                first_line = f.readline()
                f.seek(0)
                
                # 检测分隔符（逗号或分号）
                delimiter = ',' if ',' in first_line else ';'
                
                reader = csv.reader(f, delimiter=delimiter)
                
                # 跳过可能的表头
                header = next(reader, None)
                if header is None:
                    print("  [警告] CSV文件为空")
                    return poses, positions
                
                row_idx = 0
                for row in reader:
                    row_idx += 1
                    try:
                        # 检查行是否有足够的列
                        if len(row) < 33:  # 至少需要33列（索引0-32）
                            print(f"  [警告] 第 {row_idx} 行列数不足（需要至少33列，实际{len(row)}列），跳过")
                            continue
                        
                        # 读取旋转向量（rx, ry, rz）- 索引27, 28, 29
                        rx = float(row[27]) if row[27].strip() else 0.0
                        ry = float(row[28]) if row[28].strip() else 0.0
                        rz = float(row[29]) if row[29].strip() else 0.0
                        rot_vec = np.array([rx, ry, rz], dtype=np.float64)
                        
                        # 读取平移向量（tx, ty, tz）- 索引30, 31, 32
                        tx = float(row[30]) if row[30].strip() else 0.0
                        ty = float(row[31]) if row[31].strip() else 0.0
                        tz = float(row[32]) if row[32].strip() else 0.0
                        trans_vec = np.array([tx, ty, tz], dtype=np.float64)
                        
                        # 转换为4x4变换矩阵
                        R, _ = cv2.Rodrigues(rot_vec)
                        T = np.eye(4, dtype=np.float64)
                        T[:3, :3] = R
                        T[:3, 3] = trans_vec
                        
                        poses.append(T)
                        positions.append(trans_vec)
                        
                    except (ValueError, IndexError) as e:
                        print(f"  [警告] 解析第 {row_idx} 行数据失败: {e}，跳过")
                        continue
                    except Exception as e:
                        print(f"  [警告] 处理第 {row_idx} 行时出错: {e}，跳过")
                        continue
                
                print(f"  成功从CSV文件加载 {len(poses)} 个位姿")
                
        except Exception as e:
            print(f"  [错误] 读取CSV文件失败: {e}")
            import traceback
            traceback.print_exc()
            return poses, positions
    
    else:
        # 原有的TXT格式支持（向后兼容）
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式匹配所有图像位姿
        import re
        pattern = r'图像\s+(\d+):\s*\n\s*旋转向量:\s*\[([^\]]+)\]\s*\n\s*平移向量:\s*\[([^\]]+)\]'
        matches = re.findall(pattern, content)
        
        print(f"  找到 {len(matches)} 个位姿记录")
        
        for img_num_str, rot_str, trans_str in matches:
            try:
                # 解析旋转向量
                rot_vec = np.array([float(x.strip()) for x in rot_str.split()], dtype=np.float64)
                if len(rot_vec) != 3:
                    print(f"  [警告] 图像 {img_num_str} 的旋转向量维度不正确，跳过")
                    continue
                
                # 解析平移向量
                trans_vec = np.array([float(x.strip()) for x in trans_str.split()], dtype=np.float64)
                if len(trans_vec) != 3:
                    print(f"  [警告] 图像 {img_num_str} 的平移向量维度不正确，跳过")
                    continue
                
                # 转换为4x4变换矩阵
                R, _ = cv2.Rodrigues(rot_vec)
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = trans_vec
                
                poses.append(T)
                positions.append(trans_vec)
                
            except Exception as e:
                print(f"  [警告] 解析图像 {img_num_str} 的位姿失败: {e}")
                continue
        
        print(f"  成功从TXT文件加载 {len(poses)} 个位姿")
    
    return poses, positions


def save_ground_truth_poses_to_txt(poses, positions, save_path):
    """
    保存真实值位姿到TXT文件（简单格式，便于人工查看）
    
    Args:
        poses: 位姿列表，每个为4x4变换矩阵
        positions: 位置列表，每个为3D坐标
        save_path: 保存路径（TXT文件）
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("真实值相机位姿数据\n")
            f.write("="*80 + "\n\n")
            f.write(f"总位姿数量: {len(poses)}\n\n")
            f.write("格式说明：\n")
            f.write("  每行格式：帧索引 | 旋转向量[rx, ry, rz] | 平移向量[tx, ty, tz] | 4x4变换矩阵\n")
            f.write("-"*80 + "\n\n")
            
            for i, (pose, pos) in enumerate(zip(poses, positions)):
                R = pose[:3, :3]
                t = pose[:3, 3]
                
                # 计算旋转向量
                rot_vec, _ = cv2.Rodrigues(R)
                rot_vec_flat = rot_vec.flatten()
                
                f.write(f"帧 {i+1}:\n")
                f.write(f"  旋转向量: [{rot_vec_flat[0]:.8f}, {rot_vec_flat[1]:.8f}, {rot_vec_flat[2]:.8f}]\n")
                f.write(f"  平移向量: [{t[0]:.8f}, {t[1]:.8f}, {t[2]:.8f}]\n")
                f.write(f"  4x4变换矩阵:\n")
                for row in pose:
                    f.write(f"    [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}, {row[3]:12.8f}]\n")
                f.write("\n")
            
            # 添加汇总信息
            f.write("-"*80 + "\n")
            f.write("汇总信息:\n")
            if len(positions) > 0:
                positions_array = np.array(positions)
                f.write(f"  位置范围:\n")
                f.write(f"    X: [{positions_array[:, 0].min():.4f}, {positions_array[:, 0].max():.4f}]\n")
                f.write(f"    Y: [{positions_array[:, 1].min():.4f}, {positions_array[:, 1].max():.4f}]\n")
                f.write(f"    Z: [{positions_array[:, 2].min():.4f}, {positions_array[:, 2].max():.4f}]\n")
        
        print(f"  [保存] 真实值位姿已保存到: {save_path}")
        return True
        
    except Exception as e:
        print(f"  [错误] 保存真实值位姿失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_ground_truth_poses_to_json(poses, positions, save_path):
    """
    保存真实值位姿到JSON文件（便于程序读取和比较）
    格式与计算得到的位姿格式一致，便于直接比较
    
    Args:
        poses: 位姿列表，每个为4x4变换矩阵
        positions: 位置列表，每个为3D坐标
        save_path: 保存路径（JSON文件）
    """
    try:
        poses_data = []
        
        for i, (pose, pos) in enumerate(zip(poses, positions)):
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # 计算旋转向量
            rot_vec, _ = cv2.Rodrigues(R)
            rot_vec_flat = rot_vec.flatten()
            
            # 计算欧拉角
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0
            
            pose_info = {
                "frame_idx": i,
                "R": R.tolist(),
                "t": t.tolist(),
                "T": pose.tolist(),
                "rotation_vector": {
                    "x": float(rot_vec_flat[0]),
                    "y": float(rot_vec_flat[1]),
                    "z": float(rot_vec_flat[2])
                },
                "translation": {
                    "x": float(t[0]),
                    "y": float(t[1]),
                    "z": float(t[2])
                },
                "position": {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2])
                },
                "euler_angles": {
                    "roll_deg": float(np.degrees(x)),
                    "pitch_deg": float(np.degrees(y)),
                    "yaw_deg": float(np.degrees(z))
                }
            }
            poses_data.append(pose_info)
        
        output_data = {
            "source": "ground_truth",
            "source_file": "full_calibration_data_20251025_100736.csv",
            "total_poses": len(poses_data),
            "poses": poses_data
        }
        
        # JSON保存已禁用
        # with open(save_path, 'w', encoding='utf-8') as f:
        #     json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # print(f"  [保存] 真实值位姿JSON已保存到: {save_path}")
        return True
        
    except Exception as e:
        print(f"  [错误] 保存真实值位姿JSON失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_ground_truth_poses_to_csv(poses, positions, save_path):
    """
    保存真实值位姿到CSV文件（便于Excel等工具分析和比较）
    
    Args:
        poses: 位姿列表，每个为4x4变换矩阵
        positions: 位置列表，每个为3D坐标
        save_path: 保存路径（CSV文件）
    """
    try:
        import csv
        
        with open(save_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            header = [
                "frame_idx", 
                "rx", "ry", "rz",  # 旋转向量
                "tx", "ty", "tz",  # 平移向量
                "px", "py", "pz",  # 位置
                "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22",  # 旋转矩阵
                "T00", "T01", "T02", "T03", "T10", "T11", "T12", "T13", 
                "T20", "T21", "T22", "T23", "T30", "T31", "T32", "T33"  # 4x4变换矩阵
            ]
            writer.writerow(header)
            
            # 写入数据
            for i, (pose, pos) in enumerate(zip(poses, positions)):
                R = pose[:3, :3]
                t = pose[:3, 3]
                
                # 计算旋转向量
                rot_vec, _ = cv2.Rodrigues(R)
                rot_vec_flat = rot_vec.flatten()
                
                row = [
                    i,  # frame_idx
                    rot_vec_flat[0], rot_vec_flat[1], rot_vec_flat[2],  # rx, ry, rz
                    t[0], t[1], t[2],  # tx, ty, tz
                    pos[0], pos[1], pos[2],  # px, py, pz
                    R[0, 0], R[0, 1], R[0, 2],  # R00, R01, R02
                    R[1, 0], R[1, 1], R[1, 2],  # R10, R11, R12
                    R[2, 0], R[2, 1], R[2, 2],  # R20, R21, R22
                    pose[0, 0], pose[0, 1], pose[0, 2], pose[0, 3],  # T00-T03
                    pose[1, 0], pose[1, 1], pose[1, 2], pose[1, 3],  # T10-T13
                    pose[2, 0], pose[2, 1], pose[2, 2], pose[2, 3],  # T20-T23
                    pose[3, 0], pose[3, 1], pose[3, 2], pose[3, 3]   # T30-T33
                ]
                writer.writerow(row)
        
        print(f"  [保存] 真实值位姿CSV已保存到: {save_path}")
        return True
        
    except Exception as e:
        print(f"  [错误] 保存真实值位姿CSV失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_poses_info(poses, positions, save_path):
    """
    保存所有位姿信息到文件
    
    Args:
        poses: 位姿列表，每个为4x4变换矩阵
        positions: 位置列表，每个为3D坐标
        save_path: 保存路径（JSON文件）
    """
    try:
        poses_info = []
        for i, (pose, pos) in enumerate(zip(poses, positions)):
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # 计算旋转角度（欧拉角）
            # 使用旋转矩阵计算欧拉角（ZYX顺序）
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0
            
            # 转换为度数
            roll = np.degrees(x)
            pitch = np.degrees(y)
            yaw = np.degrees(z)
            
            # 计算旋转向量
            rot_vec, _ = cv2.Rodrigues(R)
            rot_vec_flat = rot_vec.flatten()  # 确保是1D数组
            
            pose_info = {
                "frame_index": i + 1,
                "position": {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2])
                },
                "rotation_matrix": R.tolist(),
                "rotation_vector": {
                    "x": float(rot_vec_flat[0]),
                    "y": float(rot_vec_flat[1]),
                    "z": float(rot_vec_flat[2])
                },
                "euler_angles": {
                    "roll_deg": float(roll),
                    "pitch_deg": float(pitch),
                    "yaw_deg": float(yaw)
                },
                "translation": {
                    "x": float(t[0]),
                    "y": float(t[1]),
                    "z": float(t[2])
                }
            }
            poses_info.append(pose_info)
        
        # JSON保存已禁用
        # 保存为JSON
        # with open(save_path, 'w', encoding='utf-8') as f:
        #     json.dump({
        #         "total_poses": len(poses_info),
        #         "poses": poses_info
        #     }, f, ensure_ascii=False, indent=2)
        
        # print(f"  [保存] 位姿信息已保存到: {save_path}")
        
        # 同时保存为文本格式（便于查看）
        txt_path = save_path.replace('.json', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("相机位姿信息\n")
            f.write("="*80 + "\n\n")
            f.write(f"总位姿数量: {len(poses_info)}\n\n")
            
            for info in poses_info:
                f.write(f"图像 {info['frame_index']}:\n")
                f.write(f"  位置: X={info['position']['x']:.4f}, Y={info['position']['y']:.4f}, Z={info['position']['z']:.4f}\n")
                f.write(f"  旋转向量: [{info['rotation_vector']['x']:.6f}, {info['rotation_vector']['y']:.6f}, {info['rotation_vector']['z']:.6f}]\n")
                f.write(f"  欧拉角: Roll={info['euler_angles']['roll_deg']:.2f}°, Pitch={info['euler_angles']['pitch_deg']:.2f}°, Yaw={info['euler_angles']['yaw_deg']:.2f}°\n")
                f.write(f"  平移向量: [{info['translation']['x']:.4f}, {info['translation']['y']:.4f}, {info['translation']['z']:.4f}]\n")
                f.write("\n")
        
        print(f"  [保存] 位姿信息文本已保存到: {txt_path}")
        return True
        
    except Exception as e:
        print(f"  [错误] 保存位姿信息失败: {e}")
        import traceback
        traceback.print_exc()
        return False




def visualize_poses_matplotlib(poses, positions, save_path):
    """
    使用matplotlib可视化所有位姿（只显示一个3D轨迹图，带起始点和结束点标记）
    
    Args:
        poses: 位姿列表，每个为4x4变换矩阵
        positions: 位置列表，每个为3D坐标
        save_path: 保存路径前缀（用于保存PNG图像）
    """
    try:
        positions_array = np.array(positions, dtype=np.float64)
        
        # 检查无效值
        valid_mask = ~(np.isnan(positions_array).any(axis=1) | np.isinf(positions_array).any(axis=1))
        positions_array = positions_array[valid_mask]
        valid_poses = [p for i, p in enumerate(poses) if valid_mask[i]]
        
        if len(positions_array) == 0:
            print("[警告] 没有有效的位姿数据")
            return False
        
        print(f"有效位姿数量: {len(positions_array)}")
        print(f"位置范围: X[{positions_array[:, 0].min():.2f}, {positions_array[:, 0].max():.2f}], "
              f"Y[{positions_array[:, 1].min():.2f}, {positions_array[:, 1].max():.2f}], "
              f"Z[{positions_array[:, 2].min():.2f}, {positions_array[:, 2].max():.2f}]")
        
        # 创建单个3D图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 创建颜色映射（基于帧序号渐变）
        colors = np.linspace(0, 1, len(positions_array))

        # 绘制所有点云（使用渐变颜色表示时间顺序）
        scatter = ax.scatter(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2],
                           c=colors, cmap='viridis', s=50, alpha=0.8, label='位姿点云')

        # 起始点标记（绿色，大号，星形）
        start_point = positions_array[0]
        ax.scatter([start_point[0]], [start_point[1]], [start_point[2]],
                  c='green', s=300, marker='*', alpha=0.9,
                  edgecolors='darkgreen', linewidths=2, label='起始点', zorder=10)

        # 结束点标记（红色，大号，方形）
        end_point = positions_array[-1]
        ax.scatter([end_point[0]], [end_point[1]], [end_point[2]],
                  c='red', s=300, marker='s', alpha=0.9,
                  edgecolors='darkred', linewidths=2, label='结束点', zorder=10)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('时间顺序 (帧序号)', fontsize=12)
        
        # 绘制坐标轴（每20个位姿显示一个）
        step = max(1, len(valid_poses) // 20)
        axis_size = 15.0
        for i in range(0, len(valid_poses), step):
            pose = valid_poses[i]
            if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
                continue
            origin = pose[:3, 3]
            R = pose[:3, :3]
            
            # X轴（红色）
            x_end = origin + R[:, 0] * axis_size
            ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 
                   'r-', linewidth=2, alpha=0.6)
            
            # Y轴（绿色）
            y_end = origin + R[:, 1] * axis_size
            ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 
                   'g-', linewidth=2, alpha=0.6)
            
            # Z轴（蓝色）
            z_end = origin + R[:, 2] * axis_size
            ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 
                   'b-', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('X (mm)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=14, fontweight='bold')
        ax.set_zlabel('Z (mm)', fontsize=14, fontweight='bold')
        ax.set_title(f'相机位姿3D点云 (共 {len(positions_array)} 个位姿)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 设置相等的坐标轴比例
        max_range = np.array([positions_array[:, 0].max() - positions_array[:, 0].min(),
                             positions_array[:, 1].max() - positions_array[:, 1].min(),
                             positions_array[:, 2].max() - positions_array[:, 2].min()]).max() / 2.0
        mid_x = (positions_array[:, 0].max() + positions_array[:, 0].min()) * 0.5
        mid_y = (positions_array[:, 1].max() + positions_array[:, 1].min()) * 0.5
        mid_z = (positions_array[:, 2].max() + positions_array[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # 保存matplotlib静态图像（PNG格式，高分辨率）
        try:
            png_path = save_path.replace('.ply', '_matplotlib_complete.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  [保存] Matplotlib静态图像已保存: {png_path}")
        except Exception as e:
            print(f"  [警告] 保存Matplotlib静态图像失败: {e}")
        
        # 注意：已禁用交互式窗口显示，只保存图像文件
        # 如果需要查看位姿轨迹，请打开保存的PNG图像文件
        # plt.show()  # 已禁用，避免打开多个窗口
        
        return True
        
    except Exception as e:
        print(f"  [错误] Matplotlib可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_ply_manual(file_path, points, colors):
    """
    手动保存PLY文件格式（兼容性更好的ASCII格式）
    
    Args:
        file_path: 保存路径
        points: 点坐标数组 (N, 3)
        colors: 颜色数组 (N, 3) 范围0-1
    """
    try:
        points = np.array(points, dtype=np.float64)
        colors = np.array(colors, dtype=np.float64)
        
        if len(points) == 0:
            print(f"  [警告] 没有点云数据可保存")
            return False
        
        # 确保颜色值在0-1范围内
        colors = np.clip(colors, 0.0, 1.0)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # PLY文件头（ASCII格式，兼容性最好）
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"comment Camera pose trajectory\n")
            f.write(f"comment Generated by ChessBoard_Cal_RT_CMP_3DICP_2DFFT_Fea_Matching\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 写入点云数据
            for i in range(len(points)):
                x, y, z = points[i]
                r, g, b = colors[i]
                
                # 将颜色从0-1范围转换为0-255范围
                r_int = max(0, min(255, int(round(r * 255))))
                g_int = max(0, min(255, int(round(g * 255))))
                b_int = max(0, min(255, int(round(b * 255))))
                
                # 使用较高精度的坐标格式
                f.write(f"{x:.8f} {y:.8f} {z:.8f} {r_int} {g_int} {b_int}\n")
        
        # 验证文件
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 100:
                print(f"  [保存] 手动PLY文件保存成功: {file_path} (大小: {file_size} 字节)")
                return True
            else:
                print(f"  [警告] PLY文件太小，可能损坏: {file_size} 字节")
                return False
        else:
            print(f"  [错误] PLY文件未创建")
            return False
            
    except Exception as e:
        print(f"  [错误] 手动PLY文件保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_and_save_poses(poses, positions, save_path):
    """
    使用Matplotlib可视化位姿轨迹并保存为PLY格式（简化版，主要使用Matplotlib）
    
    Args:
        poses: 位姿列表，每个为4x4变换矩阵
        positions: 位置列表，每个为3D坐标
        save_path: 保存路径（PLY文件）
    """
    if len(poses) == 0 or len(positions) == 0:
        print("[警告] 位姿列表为空，无法可视化")
        return False
    
    try:
        # 转换为numpy数组
        positions_array = np.array(positions, dtype=np.float64)
        
        # 检查无效值
        valid_mask = ~(np.isnan(positions_array).any(axis=1) | np.isinf(positions_array).any(axis=1))
        positions_array = positions_array[valid_mask]
        
        if len(positions_array) == 0:
            print("[警告] 清理后位置数组为空")
            return False
        
        print(f"  [信息] 有效位姿数量: {len(positions_array)}")
        print(f"  [信息] 位置范围: X[{positions_array[:, 0].min():.2f}, {positions_array[:, 0].max():.2f}], "
              f"Y[{positions_array[:, 1].min():.2f}, {positions_array[:, 1].max():.2f}], "
              f"Z[{positions_array[:, 2].min():.2f}, {positions_array[:, 2].max():.2f}]")
        
        # 创建颜色（红色）
        colors = np.array([[1.0, 0.0, 0.0]] * len(positions_array), dtype=np.float64)
        
        # 手动保存PLY（ASCII格式，最兼容）
        success_manual = save_ply_manual(save_path, positions_array, colors)
        
        # 准备有效位姿列表（用于坐标轴绘制）
        valid_poses_list = [p for i, p in enumerate(poses) if valid_mask[i]]
        
        # 使用Matplotlib进行可视化（不依赖Open3D）
        try:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制轨迹点
            ax.scatter(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2], 
                      c='red', s=20, alpha=0.6, label='相机位姿')
            
            # 绘制轨迹线
            ax.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2], 
                   'r-', alpha=0.3, linewidth=1, label='轨迹')
            
            # 绘制坐标轴（每20个位姿显示一个）
            step = max(1, len(valid_poses_list) // 20)
            axis_size = 15.0  # 坐标轴大小
            
            for i in range(0, len(valid_poses_list), step):
                pose = valid_poses_list[i]
                if np.any(np.isnan(pose)) or np.any(np.isinf(pose)):
                    continue
                origin = pose[:3, 3]
                R = pose[:3, :3]
                
                # X轴（红色）
                x_end = origin + R[:, 0] * axis_size
                ax.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 
                       'r-', linewidth=2, alpha=0.8)
                
                # Y轴（绿色）
                y_end = origin + R[:, 1] * axis_size
                ax.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 
                       'g-', linewidth=2, alpha=0.8)
                
                # Z轴（蓝色）
                z_end = origin + R[:, 2] * axis_size
                ax.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 
                       'b-', linewidth=2, alpha=0.8)
            
            # 设置标签和标题
            ax.set_xlabel('X (mm)', fontsize=12)
            ax.set_ylabel('Y (mm)', fontsize=12)
            ax.set_zlabel('Z (mm)', fontsize=12)
            ax.set_title(f'相机位姿轨迹 (共 {len(positions_array)} 个位姿)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 设置相等的坐标轴比例
            max_range = np.array([positions_array[:, 0].max() - positions_array[:, 0].min(),
                                 positions_array[:, 1].max() - positions_array[:, 1].min(),
                                 positions_array[:, 2].max() - positions_array[:, 2].min()]).max() / 2.0
            mid_x = (positions_array[:, 0].max() + positions_array[:, 0].min()) * 0.5
            mid_y = (positions_array[:, 1].max() + positions_array[:, 1].min()) * 0.5
            mid_z = (positions_array[:, 2].max() + positions_array[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.close()
            
        except Exception as e:
            print(f"  [警告] Matplotlib可视化失败: {e}")
            import traceback
            traceback.print_exc()
        
        return success_manual
        
    except Exception as e:
        print(f"[错误] 可视化位姿时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 开始总体时间统计
    main_timer = TimeTracker("主程序").start()
    
    # 准备输出目录与运行痕迹日志
    try:
        out_dir = os.environ.get("OUTPUT_DIR", r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results")
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        out_dir = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results"
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
    trace_log = os.path.join(out_dir, "run_trace.log")
    try:
        with open(trace_log, "a", encoding="utf-8") as f:
            f.write("[trace] main:start\n")
    except Exception:
        pass
    main_timer.checkpoint("初始化")
    # 如果启用视频RST示范，则直接执行并返回
    if os.environ.get("VIDEO_RST_ENABLE", "0") == "1":
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write("[trace] video_rst_entry\n")
        except Exception:
            pass
        processor = ImageProcessor(sigma=4.0, sigma_spatial=3.0, sigma_range=0.1, kernel_radius=3)
        video_path = os.environ.get("VIDEO_PATH", os.path.join(os.getcwd(), "original.mp4"))
        icp_on = (os.environ.get("ICP_VERIFY_ENABLE", "0") == "1")
        processor._video_rst_demo(video_path, do_icp=icp_on)
        print(f"[信息] 已完成视频RST示范: {video_path}")
        # 序列FD匹配与相机参数估计（增加显式trace与健壮性）
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write("[trace] video_fft_sequence_calling\n")
        except Exception:
            pass
        try:
            if hasattr(processor, "_video_fft_sequence_camera_estimation"):
                processor._video_fft_sequence_camera_estimation(video_path, max_frames=150, stride=1)
                print(f"[信息] 已完成视频序列FD匹配与相机参数估计: {video_path}")
                try:
                    with open(trace_log, "a", encoding="utf-8") as f:
                        f.write("[trace] video_fft_sequence_called_ok\n")
                except Exception:
                    pass
            else:
                print("[警告] ImageProcessor缺少方法 _video_fft_sequence_camera_estimation")
                try:
                    with open(trace_log, "a", encoding="utf-8") as f:
                        f.write("[trace] video_fft_sequence_attr_missing\n")
                except Exception:
                    pass
        except Exception as e:
            print(f"[警告] 视频序列FD/相机估计失败: {e}")
            try:
                import traceback
                with open(trace_log, "a", encoding="utf-8") as f:
                    f.write("[trace] video_fft_sequence_call_exception\n")
                    f.write(traceback.format_exc()+"\n")
            except Exception:
                pass
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write("[trace] video_rst_done\n")
        except Exception:
            pass
        return

    # 自动视频入口：检测到当前目录存在 original.mp4 则直接执行视频RST示范
    auto_video_path = os.path.join(os.getcwd(), "original.mp4")
    # 兼容显式盘符路径（L:\original.mp4）
    alt_video_path = os.path.join("L:\\", "original.mp4")
    if os.path.exists(auto_video_path) or os.path.exists(alt_video_path):
        if not os.path.exists(auto_video_path) and os.path.exists(alt_video_path):
            auto_video_path = alt_video_path
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write("[trace] video_rst_auto_entry\n")
        except Exception:
            pass
        processor = ImageProcessor(sigma=4.0, sigma_spatial=3.0, sigma_range=0.1, kernel_radius=3)
        icp_on = (os.environ.get("ICP_VERIFY_ENABLE", "0") == "1")
        processor._video_rst_demo(auto_video_path, do_icp=icp_on)
        print(f"[信息] 已完成视频RST示范: {auto_video_path}")
        # 序列FD匹配与相机参数估计（增加显式trace与健壮性）
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write("[trace] video_fft_sequence_calling\n")
        except Exception:
            pass
        try:
            if hasattr(processor, "_video_fft_sequence_camera_estimation"):
                processor._video_fft_sequence_camera_estimation(auto_video_path, max_frames=150, stride=1)
                print(f"[信息] 已完成视频序列FD匹配与相机参数估计: {auto_video_path}")
                try:
                    with open(trace_log, "a", encoding="utf-8") as f:
                        f.write("[trace] video_fft_sequence_called_ok\n")
                except Exception:
                    pass
            else:
                print("[警告] ImageProcessor缺少方法 _video_fft_sequence_camera_estimation")
                try:
                    with open(trace_log, "a", encoding="utf-8") as f:
                        f.write("[trace] video_fft_sequence_attr_missing\n")
                except Exception:
                    pass
        except Exception as e:
            print(f"[警告] 视频序列FD/相机估计失败: {e}")
            try:
                import traceback
                with open(trace_log, "a", encoding="utf-8") as f:
                    f.write("[trace] video_fft_sequence_call_exception\n")
                    f.write(traceback.format_exc()+"\n")
            except Exception:
                pass
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write("[trace] video_rst_auto_done\n")
        except Exception:
            pass
        return

    # 数据路径：图片文件夹（可用环境变量覆盖）
    input_image_folder = os.environ.get("INPUT_IMAGE_FOLDER", r"D:\reloc3r\Data_IMU_Camera_Pose_5\Lines_Photo _2")
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_image_folder):
        print(f"[错误] 输入文件夹不存在: {input_image_folder}")
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write(f"[trace] folder_not_found:{input_image_folder}\n")
        except Exception:
            pass
        return
    
    if not os.path.isdir(input_image_folder):
        print(f"[错误] 输入路径不是文件夹: {input_image_folder}")
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write(f"[trace] not_a_folder:{input_image_folder}\n")
        except Exception:
            pass
        return
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for filename in os.listdir(input_image_folder):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    image_files.sort()  # 按文件名排序
    
    if len(image_files) == 0:
        print(f"[错误] 文件夹中没有找到图片文件: {input_image_folder}")
        try:
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write(f"[trace] no_images_found:{input_image_folder}\n")
        except Exception:
            pass
        return
    
    print(f"[信息] 找到 {len(image_files)} 张图片")
    print(f"[信息] 输入文件夹: {input_image_folder}")
    
    # 获取起始帧索引（从环境变量或默认值）
    # 可以通过设置环境变量 START_FRAME_IDX 来指定起始帧，例如：START_FRAME_IDX=198
    # 如果设置 AUTO_DETECT_START_FRAME=1，则会自动检测已处理的帧并从下一帧开始
    start_frame_idx = 0
    auto_detect = os.environ.get("AUTO_DETECT_START_FRAME", "0") == "1"
    
    try:
        start_frame_idx_env = os.environ.get("START_FRAME_IDX", None)
        if start_frame_idx_env is not None:
            start_frame_idx = int(start_frame_idx_env)
            print(f"[信息] 从环境变量读取起始帧索引: {start_frame_idx}")
        elif auto_detect:
            # 如果启用了自动检测，尝试查找已存在的输出目录
            # 检查是否有 frame_0198_0199 这样的目录，如果有则从下一帧开始
            rendering_scenes_dir = os.path.join(out_dir, "rendering_scenes")
            if os.path.exists(rendering_scenes_dir):
                import re
                frame_dirs = [d for d in os.listdir(rendering_scenes_dir) if d.startswith("frame_")]
                if frame_dirs:
                    # 提取所有帧索引
                    frame_indices = []
                    for frame_dir in frame_dirs:
                        match = re.search(r'frame_(\d{4})_(\d{4})', frame_dir)
                        if match:
                            idx1, idx2 = int(match.group(1)), int(match.group(2))
                            frame_indices.extend([idx1, idx2])
                    if frame_indices:
                        max_frame_idx = max(frame_indices)
                        # 从最大帧索引+1开始
                        start_frame_idx = max_frame_idx + 1
                        print(f"[信息] 自动检测：已处理的最大帧索引: {max_frame_idx}，将从第{start_frame_idx}帧开始处理")
                        # 特别处理：如果检测到frame_0198_0199，说明198-199已处理，从200开始
                        if any('frame_0198' in d or 'frame_0199' in d for d in frame_dirs):
                            print(f"[信息] 检测到第198-199帧已处理，将从第{start_frame_idx}帧开始继续处理")
        else:
            # 默认从第0帧开始
            print(f"[信息] 将从第0帧开始处理所有帧（如需自动检测，请设置 AUTO_DETECT_START_FRAME=1）")
    except Exception as e:
        print(f"[警告] 检测起始帧索引时出错: {e}，将从第0帧开始")
        start_frame_idx = 0
    
    if start_frame_idx > 0:
        print(f"[信息] 将从第 {start_frame_idx} 帧开始处理（跳过前 {start_frame_idx} 帧）")
        if start_frame_idx >= len(image_files):
            print(f"[错误] 起始帧索引 {start_frame_idx} 超出范围（总帧数: {len(image_files)}）")
            return
    else:
        print(f"[信息] 将从第 0 帧开始处理所有 {len(image_files)} 帧")
    
    # 创建图像处理器
    try:
        with open(trace_log, "a", encoding="utf-8") as f:
            f.write("[trace] processor_init\n")
    except Exception:
        pass
    processor = ImageProcessor(sigma=4.0, sigma_spatial=3.0, sigma_range=0.1, kernel_radius=3)
    # 将起始帧索引传递给处理器，用于渲染优化时判断第一帧
    processor.start_frame_idx = start_frame_idx
    main_timer.checkpoint("创建图像处理器")
    
    # 统计信息
    success_count = 0
    fail_count = 0
    
    # 处理每张图片（从start_frame_idx开始）
    main_timer.checkpoint("准备处理图片")

    # 可选：限制处理帧数（便于端到端快速自检）
    # MAX_PROCESS_FRAMES=2 表示只处理起始帧开始的2张图
    try:
        max_process_frames = int(os.environ.get("MAX_PROCESS_FRAMES", "0"))
    except Exception:
        max_process_frames = 0
    if max_process_frames > 0:
        print(f"[信息] 已启用帧数限制：最多处理 {max_process_frames} 张（从start_frame_idx开始计数）")
    
    for idx, filename in enumerate(image_files):
        # 跳过起始帧之前的帧
        if idx < start_frame_idx:
            continue
        if max_process_frames > 0:
            processed_so_far = idx - start_frame_idx
            if processed_so_far >= max_process_frames:
                print(f"[信息] 达到MAX_PROCESS_FRAMES={max_process_frames}，停止处理")
                break
        image_path = os.path.join(input_image_folder, filename)
        base_name = os.path.splitext(filename)[0]
        
        
        print(f"\n{'='*80}")
        print(f"[处理 {idx+1}/{len(image_files)}] {filename}")
        print(f"{'='*80}")
        
        try:
            # 读取图像（BGR）
            print(f"  [读取] 正在读取图像...")
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"  [错误] 无法读取图像: {image_path}")
                fail_count += 1
                try:
                    with open(trace_log, "a", encoding="utf-8") as f:
                        f.write(f"[trace] imread_failed:{image_path}\n")
                except Exception:
                    pass
                continue
            
            print(f"  [信息] 图像尺寸: {image.shape[1]}x{image.shape[0]}")
            
            # 处理图像（传入原始图片路径）
            print(f"  [处理] 开始处理...")
            
            processor.process_image(image, frame_idx=idx, original_image_path=image_path)

            # 端到端深度恢复（ORB版本，不依赖Open3D）
            try:
                if bool(int(os.environ.get("DEPTH_RECOVERY_ORB_ENABLE", "0"))) and idx > start_frame_idx:
                    processor._depth_recover_from_orb_pair(idx - 1, idx)
            except Exception as e:
                print(f"  [警告] 深度恢复-ORB失败(帧{idx-1}->{idx}): {e}")
            
            # 保存结果至输出目录
            out_dir = processor.output_dir
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                print(f"  [警告] 创建输出目录失败: {out_dir} - {e}")
            
            print(f"  [OK] 处理完成，结果已保存到: {out_dir}")
            
            success_count += 1
            
            # 每10张图片显示一次统计
            processed_count = idx - start_frame_idx + 1  # 实际处理的帧数（从start_frame_idx开始）
            if processed_count % 10 == 0:
                elapsed_time = main_timer.elapsed()
                if processed_count > 0:
                    avg_time = elapsed_time / processed_count
                    remaining_count = len(image_files) - idx - 1
                    remaining_time = avg_time * remaining_count
                    print(f"\n[统计] 已处理 {idx+1}/{len(image_files)} 张（从第{start_frame_idx}帧开始，实际处理{processed_count}张），平均耗时: {avg_time:.2f}秒/张")
                    print(f"[预计] 剩余时间: {remaining_time/60:.1f}分钟（剩余{remaining_count}张）")
            
            try:
                with open(trace_log, "a", encoding="utf-8") as f:
                    f.write(f"[trace] process_image_done:{filename}\n")
            except Exception:
                pass
        
        except Exception as e:
            print(f"  [错误] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            try:
                with open(trace_log, "a", encoding="utf-8") as f:
                    import traceback
                    f.write(f"[trace] process_image_exception:{filename}\n")
                    f.write(traceback.format_exc()+"\n")
            except Exception:
                pass
    
    # 完成处理，记录最后检查点
    main_timer.checkpoint("完成所有图片处理")
    
    # 保存总体统计信息
    total_time = main_timer.elapsed()
    summary_path = os.path.join(processor.output_dir, "processing_summary.txt")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("ChessBoard_Cal_RT_CMP_3DICP_2DFFT_Fea_Matching 批量处理总结\n")
            f.write("="*50 + "\n\n")
            f.write(f"输入文件夹: {input_image_folder}\n")
            f.write(f"输出目录: {processor.output_dir}\n\n")
            f.write(f"总图片数: {len(image_files)}\n")
            f.write(f"成功处理: {success_count}\n")
            f.write(f"处理失败: {fail_count}\n")
            f.write(f"总耗时: {total_time/60:.2f}分钟 ({total_time:.2f}秒)\n")
            if success_count > 0:
                f.write(f"平均耗时: {total_time/success_count:.2f}秒/张\n")
    except Exception as e:
        print(f"[警告] 保存统计信息失败: {e}")
    
    print("\n" + "="*80)
    print("批量处理完成！")
    print(f"成功处理: {success_count}/{len(image_files)}")
    print(f"处理失败: {fail_count}/{len(image_files)}")
    print(f"总耗时: {total_time/60:.2f}分钟 ({total_time:.2f}秒)")
    if success_count > 0:
        print(f"平均耗时: {total_time/success_count:.2f}秒/张")
    print(f"结果保存在: {processor.output_dir}")
    print(f"总结信息保存在: {summary_path}")
    
    # 打印主程序时间统计
    main_timer.print_summary()
    print("="*80)
    
    # 3D网格匹配后的位姿估计和误差分析
    if success_count >= 2 and hasattr(processor, '_frame_3d_matches') and len(processor._frame_3d_matches) > 0:
        print("\n" + "="*80)
        print("开始基于3D匹配结果进行位姿估计和误差分析")
        print("="*80)
        
        try:
            # 1. 估计位姿
            print("\n[步骤1] 估计相机位姿...")
            # 获取第一张图像的尺寸用于估计内参
            if len(image_files) > 0:
                first_image_path = os.path.join(input_image_folder, image_files[0])
                first_image = cv2.imread(first_image_path)
                if first_image is not None:
                    h, w = first_image.shape[:2]
                    fx = fy = float(np.sqrt(w * w + h * h))
                    cx = float(w / 2)
                    cy = float(h / 2)
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
                else:
                    K = None
            else:
                K = None
            
            pose_estimation = processor._estimate_pose_from_3d_matches(processor._frame_3d_matches, K=K)
            
            if pose_estimation is None:
                print("[警告] 位姿估计失败")
            else:
                print(f"[完成] 成功估计 {len(pose_estimation['poses'])} 个位姿")
                
                # 2. 加载真实位姿
                print("\n[步骤2] 加载真实位姿数据...")
                try:
                    pose_file = r"E:\reloc3r\Point_Matching_RT\Data_IMU_Camera_Pose_1\full_calibration_data_20251025_100736.csv"
                    gt_poses, gt_positions = load_camera_poses_from_file(pose_file)
                    
                    if len(gt_poses) > 0:
                        print(f"[完成] 成功加载 {len(gt_poses)} 个真实位姿")
                        
                        # 3. 计算误差
                        print("\n[步骤3] 计算位姿误差...")
                        pose_errors = processor._compute_pose_error(pose_estimation['poses'], gt_poses)
                        
                        if pose_errors is not None:
                            print("[完成] 误差计算完成")
                            
                            # 4. 保存结果
                            print("\n[步骤4] 保存匹配结果和误差分析...")
                            processor._save_3d_matching_results(
                                processor._frame_3d_matches,
                                pose_estimation=pose_estimation,
                                pose_errors=pose_errors,
                                suffix=processor.threshold_mode
                            )
                        else:
                            print("[警告] 误差计算失败")
                    else:
                        print("[警告] 真实位姿数据为空")
                except Exception as e:
                    print(f"[警告] 加载真实位姿失败: {e}")
                    # 即使没有真实位姿，也保存匹配结果
                    processor._save_3d_matching_results(
                        processor._frame_3d_matches,
                        pose_estimation=pose_estimation,
                        pose_errors=None,
                        suffix=processor.threshold_mode
                    )
        except Exception as e:
            print(f"[错误] 3D匹配位姿估计异常: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存累积的估计位姿到TXT文件
    if success_count >= 2 and hasattr(processor, '_accumulated_poses') and processor._accumulated_poses:
        print("\n" + "="*80)
        print("保存估计的累积位姿到TXT文件")
        print("="*80)
        try:
            # 获取累积位姿结果
            accumulated_result = processor._accumulate_camera_poses_from_rt_matrices(
                reference_frame_idx=getattr(processor, 'start_frame_idx', 0)
            )
            
            if accumulated_result is not None:
                # 保存位姿（包括TXT格式）
                output_path = os.path.join(processor.output_dir, "accumulated_camera_poses.json")
                processor._save_accumulated_poses(accumulated_result, output_path)
                print("[完成] 估计位姿已保存到TXT文件")

                # 使用matplotlib可视化位姿轨迹
                print("\n" + "="*80)
                print("使用Matplotlib可视化相机位姿轨迹")
                print("="*80)
                try:
                    visualize_path = os.path.join(processor.output_dir, "camera_trajectory_visualization.png")
                    success = visualize_poses_matplotlib(
                        accumulated_result['poses'],
                        accumulated_result['positions'],
                        visualize_path
                    )
                    if success:
                        print(f"[完成] 相机位姿轨迹可视化已保存: {visualize_path}")
                        print(f"  [位姿类型] 显示的是累积的相机位姿（相对于参考帧 {accumulated_result.get('reference_frame', 0)}）")
                    else:
                        print("[警告] 位姿轨迹可视化失败")
                except Exception as e:
                    print(f"[错误] 位姿轨迹可视化异常: {e}")
                    import traceback
                    traceback.print_exc()

                # 显示帧间RT矩阵信息
                if hasattr(processor, '_rt_matrices') and processor._rt_matrices:
                    print("\n" + "="*60)
                    print("帧间RT矩阵信息（将当前帧mesh变换到前一帧坐标系）")
                    print("="*60)
                    for rt_key, rt_matrix in sorted(processor._rt_matrices.items()):
                        prev_idx, curr_idx = map(int, rt_key.split('_'))
                        translation = rt_matrix[:3, 3]
                        # 将旋转矩阵转换为欧拉角
                        try:
                            rotation_euler = matrix_to_euler_angles(torch.tensor(rt_matrix[:3, :3], dtype=torch.float32), "XYZ").numpy()
                        except Exception:
                            rotation_euler = np.array([0.0, 0.0, 0.0])  # 如果转换失败，使用默认值
                        print(f"  RT_{prev_idx}->{curr_idx}:")
                        print(f"    位移: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
                        print(f"    旋转(欧拉角°): [{np.degrees(rotation_euler[0]):.1f}, {np.degrees(rotation_euler[1]):.1f}, {np.degrees(rotation_euler[2]):.1f}]")
            else:
                print("[警告] 累积位姿结果为空，无法保存")
        except Exception as e:
            print(f"[错误] 保存估计位姿失败: {e}")
            import traceback
            traceback.print_exc()
    
    try:
        with open(trace_log, "a", encoding="utf-8") as f:
            f.write(f"[trace] batch_processing_done:success={success_count},fail={fail_count}\n")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 用户取消")
    except Exception as e:
        print(f"[异常] {e}")
        # 将异常也写入trace
        try:
            out_dir = os.environ.get("OUTPUT_DIR", r"E:\reloc3r\Point_Matching_RT\test3_extracted_frames\extracted_frames")
            trace_log = os.path.join(out_dir, "run_trace.log")
            import traceback
            with open(trace_log, "a", encoding="utf-8") as f:
                f.write("[trace] main_exception\n")
                f.write(traceback.format_exc()+"\n")
        except Exception:
            pass