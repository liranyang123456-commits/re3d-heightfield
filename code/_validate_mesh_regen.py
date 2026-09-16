#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate that debug_mesh_XXXX.ply z-values equal the saved gradient map
at the grid vertices, which would let us regenerate meshes for any
threshold/cell-size without the Op3 monolith."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover_true3d_sift_pseudo3d_v3 import _parse_ply_binary

RES = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2"

verts, _ = _parse_ply_binary(os.path.join(RES, "mesh", "debug_mesh_0000.ply"))
grad = cv2.imread(os.path.join(RES, "gradient_th_68_0000.jpg"), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(os.path.join(RES, "chessboard_mask_0000.jpg"), cv2.IMREAD_GRAYSCALE)
raw = cv2.imread(os.path.join(RES, "gradient_before_threshold_0000.jpg"), cv2.IMREAD_GRAYSCALE)

print("mesh verts:", len(verts), "z range:", verts[:, 2].min(), verts[:, 2].max())
print("grad img:", grad.shape, grad.dtype, "range:", grad.min(), grad.max())
print("mask img:", mask.shape, mask.dtype, "range:", mask.min(), mask.max(),
      "unique:", np.unique(mask)[:5])
print("raw  img:", raw.shape, raw.dtype, "range:", raw.min(), raw.max())

# compare z at vertex (x,y) with grad pixel
xs = verts[:, 0].astype(int)
ys = verts[:, 1].astype(int)
z_mesh = verts[:, 2]
z_grad = grad[ys, xs].astype(float)
print("\ncorrelation(mesh z, grad pixel):", np.corrcoef(z_mesh, z_grad)[0, 1])
print("z_mesh / z_grad ratio (where grad>0):")
m = z_grad > 0
ratios = z_mesh[m] / z_grad[m]
print("  median:", np.median(ratios), "mean:", ratios.mean(), "std:", ratios.std())

# test hypothesis: z = grad * 96/255
z_pred = z_grad * 96.0 / 255.0
err = np.abs(z_pred - z_mesh)
print("hypothesis z=grad*96/255: mean abs err:", err.mean(), "max:", err.max())

# maybe grad jpg is in [0,255] and mesh z = grad*96/255 exactly?
# check a few samples
for k in [0, 100, 5000, 12000]:
    print(f"  v{k}: mesh_z={z_mesh[k]:.3f} grad={z_grad[k]:.1f} pred={z_pred[k]:.3f}")
