#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Relative pose head trained only on train and extra_train captures.

The reported test and val sequences are held out. Each example is a pair of
frames. The target is the chessboard relative rotation and the camera-frame
translation, the same composition that reconstructs the reference trajectory
when the targets are exact.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _opt_ego_pose import sim3_ate, so3, subsample_idx
from _train_pose_fusion import H, W, geodesic, rot6d_to_matrix

EGO = Path(r"E:\EGO_Mo\datasets")
SPLIT = EGO / "trajectory_split_20260924.json"
HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
HELD_OUT = {"traj_20260923_023422", "traj_20260924_144019"}
GAPS = (8, 16, 24, 32)


class RelHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 9))

    def forward(self, x):
        out = self.fc(self.backbone(x).flatten(1))
        return rot6d_to_matrix(out[:, :6]), out[:, 6:9]


def session(name, max_frames=700):
    folder = EGO / name / "cam0" / "images"
    pose_path = EGO / "pose_gt_raw" / f"{name}.npz"
    if not folder.is_dir() or not pose_path.is_file():
        return None
    pose = np.load(pose_path)
    kept = [int(i) for i in np.flatnonzero(pose["usable"] == 1)
            if (folder / f"{int(i):06d}.jpg").is_file()]
    if len(kept) > max_frames:
        sel = subsample_idx(len(kept), max_frames)
        kept = [kept[i] for i in sel]
    if len(kept) < 40:
        return None
    imgs, Rs, ps = [], [], []
    for i in kept:
        g = cv2.imread(str(folder / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        imgs.append(cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA))
        Rs.append(np.asarray(pose["R"][i], np.float32))
        ps.append(np.asarray(pose["p"][i], np.float32) * 1000.0)
    return {"name": name, "img": np.stack(imgs), "R": np.stack(Rs), "p": np.stack(ps)}


def make_pairs(packs, gaps, per_gap=80):
    xs, Rs, ts = [], [], []
    rng = np.random.default_rng(0)
    for pack in packs:
        n = len(pack["img"])
        for gap in gaps:
            if n <= gap + 1:
                continue
            starts = rng.choice(n - gap, size=min(per_gap, n - gap), replace=False)
            for i in starts:
                j = i + gap
                pair = np.stack([pack["img"][i], pack["img"][j]], axis=0).astype(np.float32) / 255.0
                Ra, Rb = pack["R"][i], pack["R"][j]
                Rrel = so3(Ra.T @ Rb).astype(np.float32)
                tcam = (Ra.T @ (pack["p"][j] - pack["p"][i])).astype(np.float32)
                xs.append(pair)
                Rs.append(Rrel)
                ts.append(tcam)
    return np.stack(xs), np.stack(Rs), np.stack(ts)


def chain_ate(model, pack, device, step):
    model.eval()
    c = np.eye(4)
    centers = [c[:3, 3].copy()]
    n = len(pack["img"])
    with torch.no_grad():
        for i in range(0, n - 1, step):
            j = min(i + step, n - 1)
            pair = np.stack([pack["img"][i], pack["img"][j]], 0).astype(np.float32) / 255.0
            x = torch.from_numpy(pair)[None].to(device)
            R, t = model(x)
            T = np.eye(4)
            T[:3, :3] = R[0].cpu().numpy()
            T[:3, 3] = t[0].cpu().numpy()
            c = c @ T
            # hold this pose across the frames inside the step
            for _ in range(j - i):
                centers.append(c[:3, 3].copy())
    centers = np.asarray(centers)
    m = min(len(centers), n)
    sel = subsample_idx(m, 30)
    ate, scale = sim3_ate(centers[sel], pack["p"][sel])
    return float(ate), float(scale)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    names = [n for n in split["train"] + split["extra_train"] if n not in HELD_OUT]
    packs = []
    for name in names:
        pack = session(name)
        if pack is None:
            print("skip", name, flush=True)
            continue
        print(f"{name} frames {len(pack['img'])}", flush=True)
        packs.append(pack)
    x, R, t = make_pairs(packs, GAPS)
    print(f"pairs {len(x)} device {device}", flush=True)
    model = RelHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(x)
    Rt = torch.from_numpy(R)
    tt = torch.from_numpy(t)
    n = len(xt)
    rng = np.random.default_rng(1)
    for epoch in range(1, 21):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n, 64):
            idx = order[s:s + 64]
            xb = xt[idx].to(device)
            Rp, tp = model(xb)
            rot = geodesic(Rp, Rt[idx].to(device)).mean()
            trans = (tp - tt[idx].to(device)).norm(dim=1).mean()
            loss = rot + trans / 15.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"epoch {epoch:02d} loss {total/n:.3f}", flush=True)
    summary = {}
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        pack = session(name, max_frames=900)
        print(name, flush=True)
        summary[name] = {}
        for step in (8, 16, 24):
            ate, scale = chain_ate(model, pack, device, step)
            summary[name][f"step{step}"] = {"ate_mm": round(ate, 2), "scale": round(scale, 3)}
            print(f"  step {step:2} sim3 {ate:.2f} scale {scale:.3f}", flush=True)
    torch.save(model.state_dict(), HERE / "rel_pose_head.pt")
    (HERE / "ego_rel_pose.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", HERE / "ego_rel_pose.json", flush=True)


if __name__ == "__main__":
    main()
