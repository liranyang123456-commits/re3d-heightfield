#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chain Reloc3r relative poses on the same 30-frame subsets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\reloc3r_code")
from reloc3r.reloc3r_relpose import Reloc3rRelpose, inference_relpose
from reloc3r.utils.image import load_images, check_images_shape_format
from reloc3r.utils.device import to_numpy

SCREEN = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_screen.json")
OUT = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results\ego_recon_reloc3r.json")
CKPT = r"D:\reloc3r_models\reloc3r-224"
MAX_FRAMES = 30
REAL_NAMES = ("traj_20260923_023422", "traj_20260924_144019")


def sim3_ate(est_c, gt_c):
    A = np.asarray(est_c, dtype=np.float64)
    B = np.asarray(gt_c, dtype=np.float64)
    ac, bc = A.mean(0), B.mean(0)
    ad, bd = A - ac, B - bc
    sa = np.sqrt((ad ** 2).sum() / len(A))
    sb = np.sqrt((bd ** 2).sum() / len(B))
    s = sb / max(sa, 1e-12)
    U, _, Vt = np.linalg.svd(ad.T @ bd)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    Aa = s * (R @ ad.T).T
    ate = float(np.sqrt(((Aa - bd) ** 2).sum(1).mean()))
    return ate, float(s)


def subsample_idx(n, k):
    return np.unique(np.linspace(0, n - 1, min(k, n)).astype(int))


def real_paths(row):
    pose = np.load(row["pose"])
    usable = np.flatnonzero(pose["usable"] == 1)
    kept = [int(i) for i in usable if (Path(row["image_dir"]) / f"{int(i):06d}.jpg").is_file()]
    sel = subsample_idx(len(kept), MAX_FRAMES)
    ids = [kept[i] for i in sel]
    paths = [str(Path(row["image_dir"]) / f"{i:06d}.jpg") for i in ids]
    return paths, pose["p"][ids] * 1000.0


def sim_paths(row):
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all = gt_all * 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, MAX_FRAMES)
    return [str(paths[i]) for i in sel], gt_all[sel]


def chain(paths, model, device):
    poses = [np.eye(4)]
    for a, b in zip(paths, paths[1:]):
        images = load_images([a, b], size=224)
        images = check_images_shape_format(images, device)
        rel = to_numpy(inference_relpose([images[0], images[1]], model, device)[0])
        rel = np.asarray(rel, dtype=np.float64)
        nrm = np.linalg.norm(rel[:3, 3])
        if nrm > 1e-8:
            rel[:3, 3] /= nrm
        poses.append(poses[-1] @ rel)
    return np.array([T[:3, 3] for T in poses])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading Reloc3r", flush=True)
    model = Reloc3rRelpose.from_pretrained(CKPT).to(device).eval()
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    real = {r["name"]: r for r in screen["real_usable"]}
    jobs = []
    for name in REAL_NAMES:
        if name in real:
            jobs.append((name, "real", real[name].get("role"), *real_paths(real[name])))
    if screen["sim_with_images"]:
        row = screen["sim_with_images"][0]
        jobs.append((row["name"], "sim", None, *sim_paths(row)))
    rows = []
    for name, kind, role, paths, gt in jobs:
        print("reloc3r", name, flush=True)
        est = chain(paths, model, device)
        ate, scale = sim3_ate(est, gt)
        rec = {"name": name, "kind": kind, "role": role, "n": len(paths),
               "ate_mm": round(ate, 2), "sim3_scale": round(scale, 3)}
        rows.append(rec)
        print(rec, flush=True)
        OUT.write_text(json.dumps({"method": "Reloc3r", "max_frames": MAX_FRAMES, "results": rows}, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
