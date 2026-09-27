#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrain the pose head on far-camera captures only.

The held-out test camera sits near 186 mm from the board and travels farther
than most training sessions. This run keeps sequences whose median camera
distance is at least 150 mm, uses every usable frame up to 1200, and still
excludes traj_20260923_023422 and traj_20260924_144019.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import _train_pose_fusion as base

EGO = base.EGO
HERE = base.HERE
FAR_M = 0.15
MAX_FRAMES = 1200


def median_z(name):
    path = EGO / "pose_gt_raw" / f"{name}.npz"
    if not path.is_file():
        return None
    z = np.load(path)
    if "usable" not in z.files or int(z["usable"].sum()) < 30:
        return None
    return float(np.median(z["p"][z["usable"] == 1][:, 2]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = json.loads(base.SPLIT.read_text(encoding="utf-8"))
    names = []
    for name in split["train"] + split["extra_train"]:
        if name in base.HELD_OUT:
            continue
        z = median_z(name)
        if z is None:
            print("skip missing", name, flush=True)
            continue
        if z > -FAR_M:
            print(f"skip near {name} z {z*1000:.0f} mm", flush=True)
            continue
        print(f"keep {name} z {z*1000:.0f} mm", flush=True)
        names.append(name)
    packs = []
    for name in names:
        pack = base.load_split_frames(name, max_frames=MAX_FRAMES)
        if pack is None:
            continue
        print(f"  loaded {name} {len(pack['t'])}", flush=True)
        packs.append(pack)
    train = base.concat(packs)
    internal = base.load_split_frames(base.INTERNAL_VAL, max_frames=400)
    test = base.load_split_frames("traj_20260923_023422", max_frames=900)
    val = base.load_split_frames("traj_20260924_144019", max_frames=900)
    print(f"train {len(train['t'])} on {device}", flush=True)
    model = base.PoseHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    x = torch.from_numpy(train["x"])
    R = torch.from_numpy(train["R"].astype(np.float32))
    t = torch.from_numpy(train["t"].astype(np.float32))
    t_mean = t.mean(0, keepdim=True)
    t = t - t_mean
    n = len(x)
    rng = np.random.default_rng(2)
    best_state, best_val = None, 1e9
    for epoch in range(1, 41):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n, 64):
            idx = order[s:s + 64]
            xb = x[idx].to(device)
            xb = torch.clamp(xb * (0.85 + 0.3 * torch.rand(len(xb), 1, 1, 1, device=device)), 0, 1)
            loss, _, _ = base.batch_loss(model, xb, R[idx].to(device), t[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        msg = f"epoch {epoch:02d} loss {total / n:.3f}"
        if internal is not None and epoch % 5 == 0:
            ate, _, _ = base.ate_of(model, {"x": internal["x"], "t": internal["t"] - t_mean.numpy()}, device)
            msg += f" internal_sim3 {ate:.2f}"
            if ate < best_val:
                best_val = ate
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    def report(pack, title):
        model.eval()
        xx = torch.from_numpy(pack["x"]).to(device)
        est = []
        with torch.no_grad():
            for i in range(0, len(xx), 64):
                _, tp = model(xx[i:i + 64])
                est.append(tp.cpu().numpy() + t_mean.numpy())
        est = np.concatenate(est)
        sel = base.subsample_idx(len(est), 30)
        ate, scale = base.sim3_ate(est[sel], pack["t"][sel])
        direct = float(np.linalg.norm(est[sel] - pack["t"][sel], axis=1).mean())
        print(f"{title} sim3 {ate:.2f} scale {scale:.3f} direct {direct:.2f}", flush=True)
        return {"ate_mm": round(float(ate), 2), "scale": round(float(scale), 3),
                "direct_mm": round(direct, 2)}

    summary = {
        "test": report(test, "TEST"),
        "val": report(val, "VAL"),
        "train_sequences": names,
        "note": "far-camera subset only; test and reported val held out",
    }
    (HERE / "ego_pose_far.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", HERE / "ego_pose_far.json", flush=True)


if __name__ == "__main__":
    main()
