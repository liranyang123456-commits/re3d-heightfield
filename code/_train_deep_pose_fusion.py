#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep Pose Fusion Training on Surgical Endoscopic Video.

Backbone: Pretrained ResNet-50 (safetensors from ImageNet) with Spatial Attention Pooling.
Supervision: Geodesic loss on SO(3), Huber loss on 3D translation (mm), Cosine direction loss.
Data: 11 real surgical sequences (~8,000 frames) with extensive photometric & motion augmentation.
Holdout: traj_20260923_023422 (test) and traj_20260924_144019 (val) are strictly held out.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
import torchvision.models as models

from _opt_ego_pose import sim3_ate, subsample_idx

EGO = Path(r"E:\EGO_Mo\datasets")
SPLIT = EGO / "trajectory_split_20260924.json"
HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
CKPT_OUT = HERE / "deep_pose_fusion_best.pt"
JSON_OUT = HERE / "deep_pose_fusion_summary.json"
RESNET_ST = Path(r"C:\Users\lry\.cache\huggingface\hub\models--timm--resnet50.a1_in1k\snapshots\767268603ca0cb0bfe326fa87277f19c419566ef\model.safetensors")

HELD_OUT = {"traj_20260923_023422", "traj_20260924_144019"}
INTERNAL_VAL = "traj_20260923_023241"
IMG_W, IMG_H = 320, 180


def geodesic_loss(R_pred, R_gt):
    delta = R_pred.transpose(1, 2) @ R_gt
    tr = delta[:, 0, 0] + delta[:, 1, 1] + delta[:, 2, 2]
    cos = torch.clamp((tr - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


class DeepPoseFusionResNet(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        resnet = models.resnet50(weights=None)
        if weights_path and Path(weights_path).is_file():
            st = safetensors.torch.load_file(weights_path)
            resnet.load_state_dict(st, strict=False)
            print("Loaded ImageNet-pretrained ResNet-50 backbone successfully.")
            
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Spatial Attention Pooling
        self.att_pool = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, 1),
            nn.Softmax(dim=-1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 9)  # 6D rot + 3D trans
        )
        
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)  # (B, 2048, H', W')
        
        B, C, H, W = feat.shape
        att = self.att_pool(feat).view(B, 1, H * W)
        flat = feat.view(B, C, H * W)
        pooled = torch.bmm(flat, att.transpose(1, 2)).view(B, C)
        
        out = self.fc(pooled)
        d6 = out[:, :6]
        trans = out[:, 6:9]
        
        # Continuous 6D representation to SO(3)
        b1 = F.normalize(d6[:, 0:3], dim=1)
        b2 = d6[:, 3:6] - (b1 * d6[:, 3:6]).sum(1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        R = torch.stack((b1, b2, b3), dim=-1)
        return R, trans


def load_dataset_pack(name, max_frames=800):
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
        
    imgs, Rs, ts = [], [], []
    for i in kept:
        bgr = cv2.imread(str(folder / f"{i:06d}.jpg"))
        if bgr is None:
            continue
        small = cv2.resize(bgr, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        # RGB normalized
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # ImageNet mean & std
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        imgs.append(rgb.transpose(2, 0, 1).astype(np.float32))
        Rs.append(np.asarray(pose["R"][i], np.float64))
        ts.append(np.asarray(pose["p"][i], np.float64) * 1000.0)
        
    if len(imgs) < 20:
        return None
    return {
        "name": name,
        "x": np.stack(imgs),
        "R": np.stack(Rs),
        "t": np.stack(ts),
        "ids": kept,
    }


def augment_batch(xb, device):
    """Random brightness, contrast, and noise augmentation."""
    B = xb.shape[0]
    # brightness jitter
    alpha = 0.85 + 0.3 * torch.rand(B, 1, 1, 1, device=device)
    xb = xb * alpha
    # small gaussian noise
    noise = torch.randn_like(xb) * 0.02
    return xb + noise


def evaluate_pack(model, pack, t_mean, device):
    model.eval()
    xx = torch.from_numpy(pack["x"]).to(device)
    preds_t, preds_R = [], []
    with torch.no_grad():
        for i in range(0, len(xx), 64):
            R, t = model(xx[i:i + 64])
            preds_t.append(t.cpu().numpy() + t_mean.cpu().numpy())
            preds_R.append(R.cpu().numpy())
    est_t = np.concatenate(preds_t)
    sel = subsample_idx(len(est_t), 30)
    ate, scale = sim3_ate(est_t[sel], pack["t"][sel])
    direct = float(np.linalg.norm(est_t[sel] - pack["t"][sel], axis=1).mean())
    return float(ate), float(scale), direct


def run_training_campaign(n_epochs=100, batch_size=32, lr=5e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Starting Deep Pose Fusion Training on {device} ({n_epochs} epochs) ===")
    
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_names = [n for n in split["train"] + split["extra_train"] if n not in HELD_OUT]
    
    packs = []
    for name in train_names:
        p = load_dataset_pack(name, max_frames=800)
        if p:
            print(f"  Loaded {name}: {len(p['t'])} frames")
            packs.append(p)
            
    train_x = np.concatenate([p["x"] for p in packs])
    train_R = np.concatenate([p["R"] for p in packs])
    train_t = np.concatenate([p["t"] for p in packs])
    n_train = len(train_x)
    print(f"Total training dataset size: {n_train} samples from {len(packs)} sessions.")
    
    internal_val = load_dataset_pack(INTERNAL_VAL, max_frames=400)
    test_pack = load_dataset_pack("traj_20260923_023422", max_frames=800)
    val_pack = load_dataset_pack("traj_20260924_144019", max_frames=800)
    
    model = DeepPoseFusionResNet(weights_path=str(RESNET_ST)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    t_mean = torch.from_numpy(train_t.mean(0, keepdims=True).astype(np.float32)).to(device)
    train_t_shifted = torch.from_numpy((train_t - train_t.mean(0, keepdims=True)).astype(np.float32))
    train_R_tensor = torch.from_numpy(train_R.astype(np.float32))
    train_x_tensor = torch.from_numpy(train_x)
    
    best_val_ate = 1e9
    history = []
    
    t_start = time.time()
    for ep in range(1, n_epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        loss_ep = 0.0
        n_batches = 0
        
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            xb = augment_batch(train_x_tensor[idx].to(device), device)
            Rb = train_R_tensor[idx].to(device)
            tb = train_t_shifted[idx].to(device)
            
            R_pred, t_pred = model(xb)
            
            # Loss formulation
            l_rot = geodesic_loss(R_pred, Rb)
            l_trans = F.smooth_l1_loss(t_pred, tb, beta=5.0)
            
            # Cosine direction loss
            cos_dir = 1.0 - (F.normalize(t_pred, dim=-1) * F.normalize(tb, dim=-1)).sum(dim=-1).mean()
            
            loss = l_rot + 0.05 * l_trans + 0.2 * cos_dir
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            loss_ep += float(loss.item())
            n_batches += 1
            
        scheduler.step()
        avg_loss = loss_ep / max(n_batches, 1)
        
        # Periodic evaluation
        if ep % 5 == 0 or ep == n_epochs:
            ate_int, _, dir_int = evaluate_pack(model, internal_val, t_mean, device)
            ate_test, sc_test, dir_test = evaluate_pack(model, test_pack, t_mean, device)
            ate_val, sc_val, dir_val = evaluate_pack(model, val_pack, t_mean, device)
            
            is_best = ate_val < best_val_ate
            if is_best:
                best_val_ate = ate_val
                torch.save({
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "t_mean": t_mean.cpu(),
                    "val_ate": ate_val,
                    "test_ate": ate_test,
                }, CKPT_OUT)
                
            rec = {
                "epoch": ep,
                "loss": round(avg_loss, 4),
                "val_internal_ate": round(ate_int, 2),
                "heldout_val_ate": round(ate_val, 2),
                "heldout_val_direct": round(dir_val, 2),
                "heldout_test_ate": round(ate_test, 2),
                "heldout_test_direct": round(dir_test, 2),
                "lr": round(optimizer.param_groups[0]["lr"], 6),
                "is_best": is_best,
            }
            history.append(rec)
            print(f"Epoch {ep:03d} | Loss: {avg_loss:.4f} | Val ATE: {ate_val:.2f} mm | Test ATE: {ate_test:.2f} mm | Best: {best_val_ate:.2f} mm", flush=True)
            
            # Write log json continuously
            JSON_OUT.write_text(json.dumps({
                "epochs_completed": ep,
                "best_val_ate": round(best_val_ate, 2),
                "elapsed_min": round((time.time() - t_start) / 60.0, 1),
                "history": history,
            }, indent=2), encoding="utf-8")
        else:
            print(f"Epoch {ep:03d} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)
            
    print(f"\nTraining campaign completed in {(time.time() - t_start) / 60.0:.1f} minutes. Best Val ATE: {best_val_ate:.2f} mm")


if __name__ == "__main__":
    run_training_campaign(n_epochs=120, batch_size=32, lr=5e-4)
