#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect vertex coordinate ranges of Op3-generated mesh PLYs."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recover_true3d_sift_pseudo3d_v3 import _parse_ply_binary

p = r"D:\reloc3r\Data_IMU_Camera_Pose_5\Line_Photo_2_results2\mesh"
for fn in ["debug_mesh_0000.ply", "debug_mesh_0001.ply"]:
    verts, cols = _parse_ply_binary(os.path.join(p, fn))
    print(fn, "n=", len(verts))
    print("  x range:", verts[:, 0].min(), verts[:, 0].max())
    print("  y range:", verts[:, 1].min(), verts[:, 1].max())
    print("  z range:", verts[:, 2].min(), verts[:, 2].max())
    ux = np.unique(np.round(verts[:, 0], 3))
    uy = np.unique(np.round(verts[:, 1], 3))
    print("  unique x count:", len(ux), "first:", ux[:5])
    print("  unique y count:", len(uy), "first:", uy[:5])
    dx = np.diff(np.sort(ux))
    print("  x spacing median:", np.median(dx) if len(dx) else None)
