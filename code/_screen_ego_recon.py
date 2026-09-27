#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen EGO-Mo captures for monocular reconstruction, not IMU displacement.

A real session is usable when the left camera has images, both IMUs were
recorded, a chessboard pose file exists, and the rig was not rejected as
non-rigid. A synthetic session is usable only when rendered images and
ground_truth.npz both exist. The large IMU-only corpus is listed as rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EGO = Path(r"E:\EGO_Mo\datasets")
SPLIT = EGO / "trajectory_split_20260924.json"
CALIB = EGO / "calib_intrinsics_20260922_123120" / "camera_calibration_cam0.json"
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_screen.json")


def _jpg(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() == ".jpg")


def screen_real(split: dict) -> list[dict]:
    excluded = {item["name"]: item["reason"] for item in split.get("excluded", [])}
    role = {}
    for key in ("train", "val", "test", "extra_train"):
        for name in split.get(key, []):
            role[name] = key
    rows = []
    for summary_path in sorted(EGO.glob("*/capture_summary.json")):
        name = summary_path.parent.name
        if not (name.startswith("traj_") or name.startswith("rigid_")):
            continue
        sess = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        n0 = _jpg(sess / "cam0" / "images")
        pose_path = EGO / "pose_gt_raw" / f"{name}.npz"
        usable_pose = 0
        if pose_path.is_file():
            usable_pose = int(np.load(pose_path)["usable"].sum())
        reasons = []
        if name in excluded:
            reasons.append(excluded[name])
        if n0 < 80:
            reasons.append(f"cam0 images {n0}")
        if not pose_path.is_file():
            reasons.append("no chessboard pose")
        elif usable_pose < 40:
            reasons.append(f"usable board frames {usable_pose}")
        if float(summary.get("seconds") or 0) < 20:
            reasons.append("shorter than 20 s")
        rows.append({
            "name": name,
            "role": role.get(name),
            "seconds": round(float(summary.get("seconds") or 0), 1),
            "cam0": n0,
            "cam1": _jpg(sess / "cam1" / "images"),
            "usable_pose_frames": usable_pose,
            "image_dir": str(sess / "cam0" / "images"),
            "pose": str(pose_path),
            "usable": not reasons,
            "reasons": reasons,
        })
    return rows


def screen_sim() -> list[dict]:
    rows = []
    for gt in EGO.rglob("ground_truth.npz"):
        sess = gt.parent
        # skip the IMU-only corpus trees by requiring images next to the npz
        n0 = _jpg(sess / "cam0" / "images")
        if n0 < 8:
            continue
        z = np.load(gt)
        rows.append({
            "name": sess.name,
            "frames": int(len(z["t"])) if "t" in z.files else n0,
            "cam0": n0,
            "image_dir": str(sess / "cam0" / "images"),
            "pose": str(gt),
            "kind": "rendered_sim",
        })
    return rows


def main():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    real = screen_real(split)
    sim = screen_sim()
    # one rendered clip per smoke folder, longest first, cap 4
    sim_sorted = sorted(sim, key=lambda r: -r["cam0"])[:4]
    out = {
        "purpose": "monocular reconstruction, not IMU displacement",
        "K": calib["camera_matrix"],
        "image_size": calib["image_size"],
        "square_mm": calib["square_size_mm"],
        "real_usable": [r for r in real if r["usable"]],
        "real_rejected": [r for r in real if not r["usable"]],
        "sim_with_images": sim_sorted,
        "sim_note": (
            "corpus_20260925_135023 has IMU and pose ground truth but no "
            "rendered images, so it cannot be used for reconstruction."
        ),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"real usable {len(out['real_usable'])} rejected {len(out['real_rejected'])}")
    print(f"rendered sim clips {len(sim_sorted)}")
    for r in out["real_usable"]:
        print(f"  {r['name']:28} role={r['role']} cam0={r['cam0']} pose={r['usable_pose_frames']}")
    for r in sim_sorted:
        print(f"  sim {r['name']} frames={r['cam0']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
