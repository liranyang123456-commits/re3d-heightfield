#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N2: board-occupancy and working-distance stats from existing GT.

The 12x9-3mm board is the only target with enough valid frames for ATE.
Secondary boards (9x6-5mm, 9x8-8mm) are counted but not used as a second
metric protocol. Working distance is ||t|| of the saved c2w inverse (mm).
"""
from __future__ import annotations
import os, json
import numpy as np

NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
CSV_ROOT = r"D:\reloc3r\Data_IMU_Camera_Pose_Sequences\stereo_endoscope"
OUT = r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\n2_boards.json"
ALIAS = {
    "vio_seq_20260911_010902": "V1",
    "vio_seq_20260911_011013": "V2",
    "pnp_seq_20260911_011115": "P1",
    "vio_seq_20260912_104835": "V3",
}


def working_distance(gt_dir):
    dists = []
    for fn in os.listdir(gt_dir):
        if not fn.startswith("pose_") or not fn.endswith(".txt"):
            continue
        c2w = np.loadtxt(os.path.join(gt_dir, fn))
        w2c = np.linalg.inv(c2w)
        dists.append(float(np.linalg.norm(w2c[:3, 3])))
    if not dists:
        return None
    a = np.array(dists)
    return {
        "n": int(len(a)),
        "mean_mm": float(a.mean()),
        "median_mm": float(np.median(a)),
        "p05_mm": float(np.percentile(a, 5)),
        "p95_mm": float(np.percentile(a, 95)),
    }


def extra_board_csv(session):
    csv_path = os.path.join(CSV_ROOT, session, "frames_labeled.csv")
    if not os.path.exists(csv_path):
        return None
    import pandas as pd
    df = pd.read_csv(csv_path)
    v = df[df["pose_valid"] == 1]
    out = {}
    for name, g in v.groupby("chessboard_name"):
        t = np.sqrt(g["tvec_x"] ** 2 + g["tvec_y"] ** 2 + g["tvec_z"] ** 2)
        out[str(name)] = {
            "n": int(len(g)),
            "mean_dist_mm": float(t.mean()),
            "median_dist_mm": float(t.median()),
        }
    return out


def main():
    rows = []
    for sess, alias in ALIAS.items():
        meta_path = os.path.join(NEW_ROOT, sess, "gt_meta.json")
        meta = json.load(open(meta_path, encoding="utf-8"))
        wd = working_distance(os.path.join(NEW_ROOT, sess, "gt_poses"))
        extra = extra_board_csv(sess)
        row = {
            "alias": alias,
            "session": sess,
            "n_frames": meta["n_frames"],
            "n_pose_valid": meta["n_pose_valid"],
            "pose_valid_pct": round(100.0 * meta["n_traj_poses"] / meta["n_frames"], 1),
            "boards": meta["boards"],
            "traj_board": meta["traj_board"],
            "n_traj": meta["n_traj_poses"],
            "working_distance_12x9": wd,
            "boards_from_csv": extra,
        }
        rows.append(row)
        print(f"[{alias}] frames={row['n_frames']} traj={row['n_traj']} "
              f"({row['pose_valid_pct']}%) boards={row['boards']}")
        if wd:
            print(f"       12x9 working dist mean={wd['mean_mm']:.1f} mm "
                  f"median={wd['median_mm']:.1f} mm")

    payload = {
        "note": "Secondary boards have 1-19 frames and are not a second ATE protocol.",
        "sessions": rows,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("saved", OUT)


if __name__ == "__main__":
    main()
