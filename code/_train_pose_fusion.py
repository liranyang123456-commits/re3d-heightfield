#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a rig-specific pose head on sequences other than the reported test and val.

Labels are the chessboard poses of the training captures. The reported
test sequence traj_20260923_023422 and val sequence traj_20260924_144019
are never used as training images. The head sees a downsized grayscale frame
and predicts rotation and camera position.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _opt_ego_pose import sim3_ate, subsample_idx

EGO = Path(r"E:\EGO_Mo\datasets")
SPLIT = EGO / "trajectory_split_20260924.json"
HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
CKPT = HERE / "pose_fusion_head.pt"
H, W = 112, 192
HELD_OUT = {"traj_20260923_023422", "traj_20260924_144019"}
INTERNAL_VAL = "traj_20260923_023241"


def rot6d_to_matrix(d6):
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def geodesic(R_pred, R_gt):
    delta = R_pred.transpose(1, 2) @ R_gt
    tr = delta[:, 0, 0] + delta[:, 1, 1] + delta[:, 2, 2]
    cos = torch.clamp((tr - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


class PoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 9))

    def forward(self, x):
        z = self.backbone(x).flatten(1)
        out = self.fc(z)
        return rot6d_to_matrix(out[:, :6]), out[:, 6:9]


def load_split_frames(name, max_frames=500):
    folder = EGO / name / "cam0" / "images"
    pose_path = EGO / "pose_gt_raw" / f"{name}.npz"
    if not folder.is_dir() or not pose_path.is_file():
        return None
    pose = np.load(pose_path)
    usable = np.flatnonzero(pose["usable"] == 1)
    kept = [int(i) for i in usable if (folder / f"{int(i):06d}.jpg").is_file()]
    if len(kept) < 30:
        return None
    if len(kept) > max_frames:
        sel = subsample_idx(len(kept), max_frames)
        kept = [kept[i] for i in sel]
    images, Rs, ts = [], [], []
    for i in kept:
        bgr = cv2.imread(str(folder / f"{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if bgr is None:
            continue
        small = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
        images.append(small.astype(np.float32) / 255.0)
        Rs.append(np.asarray(pose["R"][i], np.float64))
        ts.append(np.asarray(pose["p"][i], np.float64) * 1000.0)
    if len(images) < 20:
        return None
    return {
        "name": name,
        "x": np.stack(images)[:, None],
        "R": np.stack(Rs),
        "t": np.stack(ts),
        "ids": kept,
    }


def load_sim(name, gt_path, image_dir, max_frames=200):
    if not Path(image_dir).is_dir() or not Path(gt_path).is_file():
        return None
    z = np.load(gt_path)
    key = "p_W_C" if "p_W_C" in z.files else "p"
    p = np.asarray(z[key], np.float64)
    if np.nanmax(np.abs(p)) < 5:
        p = p * 1000.0
    R = np.asarray(z["R_W_C"] if "R_W_C" in z.files else z["R"], np.float64) if (
        "R_W_C" in z.files or "R" in z.files) else None
    paths = sorted(Path(image_dir).glob("*.jpg"))
    n = min(len(paths), len(p))
    sel = subsample_idx(n, min(max_frames, n))
    images, ts, Rs = [], [], []
    for i in sel:
        bgr = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
        if bgr is None:
            continue
        images.append(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)
        ts.append(p[i])
        if R is not None:
            Rs.append(R[i])
    if len(images) < 15 or R is None:
        return None
    return {"name": name, "x": np.stack(images)[:, None], "R": np.stack(Rs), "t": np.stack(ts)}


def batch_loss(model, x, R, t):
    Rp, tp = model(x)
    rot = geodesic(Rp, R).mean()
    trans = (tp - t).norm(dim=1).mean()
    return rot + trans / 20.0, rot.detach(), trans.detach()


@torch.no_grad()
def ate_of(model, pack, device):
    model.eval()
    x = torch.from_numpy(pack["x"]).to(device)
    preds_t, preds_R = [], []
    for i in range(0, len(x), 64):
        R, t = model(x[i:i + 64])
        preds_t.append(t.cpu().numpy())
        preds_R.append(R.cpu().numpy())
    est = np.concatenate(preds_t)
    # Protocol-style number: 30 evenly spaced frames, Sim3 on positions.
    sel = subsample_idx(len(est), 30)
    ate, scale = sim3_ate(est[sel], pack["t"][sel])
    # Also the direct error, no similarity.
    direct = float(np.linalg.norm(est - pack["t"], axis=1).mean())
    return float(ate), float(scale), direct


def concat(packs):
    return {
        "name": "+".join(p["name"] for p in packs),
        "x": np.concatenate([p["x"] for p in packs]),
        "R": np.concatenate([p["R"] for p in packs]),
        "t": np.concatenate([p["t"] for p in packs]),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_names = [n for n in split["train"] + split["extra_train"] if n not in HELD_OUT]
    print("loading", train_names, flush=True)
    packs = []
    for name in train_names:
        pack = load_split_frames(name)
        if pack is None:
            print("  skip", name, flush=True)
            continue
        print(f"  {name} {len(pack['t'])}", flush=True)
        packs.append(pack)
    train = concat(packs)
    internal = load_split_frames(INTERNAL_VAL, max_frames=300)
    test = load_split_frames("traj_20260923_023422", max_frames=800)
    val = load_split_frames("traj_20260924_144019", max_frames=800)
    print(f"train images {len(train['t'])} device {device}", flush=True)
    model = PoseHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(train["x"])
    R = torch.from_numpy(train["R"].astype(np.float32))
    t = torch.from_numpy(train["t"].astype(np.float32))
    # Center the translation targets so the head learns residuals around the rig.
    t_mean = t.mean(0, keepdim=True)
    t = t - t_mean
    n = len(x)
    rng = np.random.default_rng(0)
    best_state, best_val = None, 1e9
    for epoch in range(1, 31):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n, 64):
            idx = order[s:s + 64]
            xb = x[idx].to(device)
            # light brightness jitter
            if model.training:
                xb = torch.clamp(xb * (0.8 + 0.4 * torch.rand(len(xb), 1, 1, 1, device=device)), 0, 1)
            loss, rot, trans = batch_loss(model, xb, R[idx].to(device), t[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        msg = f"epoch {epoch:02d} loss {total/n:.3f}"
        if internal is not None and epoch % 5 == 0:
            ate, scale, direct = ate_of(model, {
                "x": internal["x"],
                "t": internal["t"] - t_mean.numpy(),
            }, device)
            # ate_of compares to pack t which we shifted; shift back by evaluating raw
            msg += f" internal_sim3 {ate:.2f}"
            if ate < best_val:
                best_val = ate
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    # Evaluate in the original millimetre frame: add the training mean back.
    def report(pack, title):
        model.eval()
        xx = torch.from_numpy(pack["x"]).to(device)
        est = []
        with torch.no_grad():
            for i in range(0, len(xx), 64):
                _, tp = model(xx[i:i + 64])
                est.append(tp.cpu().numpy() + t_mean.numpy())
        est = np.concatenate(est)
        sel = subsample_idx(len(est), 30)
        ate, scale = sim3_ate(est[sel], pack["t"][sel])
        direct = float(np.linalg.norm(est[sel] - pack["t"][sel], axis=1).mean())
        print(f"{title} sim3_ate {ate:.2f} scale {scale:.3f} direct_mm {direct:.2f}", flush=True)
        return {"ate_mm": round(float(ate), 2), "scale": round(float(scale), 3),
                "direct_mm": round(direct, 2), "n": 30}

    summary = {
        "test_traj_20260923_023422": report(test, "TEST"),
        "val_traj_20260924_144019": report(val, "VAL"),
        "note": "trained on train+extra_train only; reported test and val held out",
    }
    torch.save({"state": model.state_dict(), "t_mean": t_mean.cpu()}, CKPT)
    (HERE / "ego_pose_fusion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", CKPT, flush=True)


if __name__ == "__main__":
    main()
