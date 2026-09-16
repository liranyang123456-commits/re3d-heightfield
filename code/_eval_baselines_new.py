#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate all finished baselines on all 4 new sessions (shape + pose vs GT)."""
import os, subprocess, sys

PY = r"C:\Users\lry\.conda\envs\py3d\python.exe"
EVAL = r"e:\MIS_TMI_Re_3D\re3d_cmpb_results\code\_eval_new_capture.py"
BENCH = r"E:\MIS_TMI_Re_3D\benchmark_new"
SESSIONS = ["pnp_seq_20260911_011115", "vio_seq_20260911_010902",
            "vio_seq_20260911_011013", "vio_seq_20260912_104835"]

# method -> (pose_subdir, pose_prefix, cloud_filename)
METHODS = {
    "orb_sfm": ("orb_poses", "c2w_", "orb_sfm.ply"),
    "vggt": ("estimated_poses_txt", "pose_", "vggt_pointcloud.ply"),
    "colmap": ("estimated_poses_txt", "pose_", "colmap_sfm.ply"),
}

for method, (sub, pre, cloud_fn) in METHODS.items():
    for sess in SESSIONS:
        run_dir = os.path.join(BENCH, method, sess)
        cloud = os.path.join(run_dir, cloud_fn)
        if not os.path.isdir(run_dir):
            print(f"[skip] {method}/{sess}: no run dir"); continue
        cmd = [PY, EVAL, "--run_dir", run_dir, "--session", sess,
               "--pose_subdir", sub, "--pose_prefix", pre,
               "--out_name", f"eval_{method}.json"]
        if os.path.exists(cloud):
            cmd += ["--cloud", cloud]
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("[")]
        print(f"{method:10s} {sess}: " + (" | ".join(line) if line else r.stdout.strip()[-200:]))
        if r.returncode != 0:
            print("  ERR:", r.stderr.strip()[-300:])

print("\nEVAL DONE")
