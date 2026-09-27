#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradient-anchored planar/essential VO for the EGO clips.

Image-space heightfield ICP does not follow the camera center: the mesh
lives on the pixel grid. This estimator keeps the paper's gradient anchors,
tracks them, and recovers pose with the calibrated camera. A planar board
uses a homography; a general scene uses the essential matrix. A failed or
near-static pair holds the camera still instead of repeating the last step.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(r"E:\MIS_TMI_Re_3D\re3d_cmpb_results\results")
SCREEN = HERE / "ego_recon_screen.json"
CALIB_SIZE = (1280, 720)
MAX_ROT_VIDEO = 12.0
MAX_ROT_GAP = 80.0


def sim3_ate(est_c, gt_c):
    A = np.asarray(est_c, dtype=np.float64)
    B = np.asarray(gt_c, dtype=np.float64)
    ac, bc = A.mean(0), B.mean(0)
    ad, bd = A - ac, B - bc
    sa = np.sqrt((ad ** 2).sum() / len(A))
    sb = np.sqrt((bd ** 2).sum() / len(B))
    if sa < 1e-8:
        return 1e9, 0.0
    s = sb / sa
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


def scale_K(K, wh):
    w, h = wh
    out = np.array(K, dtype=np.float64, copy=True)
    out[0, :] *= w / CALIB_SIZE[0]
    out[1, :] *= h / CALIB_SIZE[1]
    return out


def so3(R):
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rot_deg(R):
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def prep(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray, 7, 25, 7)
    dx = cv2.Sobel(filt, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(filt, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(dx, dy)
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fov = ((gray > 18).astype(np.uint8)) * 255
    fov = cv2.erode(fov, np.ones((15, 15), np.uint8))
    return gray, mag_u8, fov


def _lk(g1, g2, p1):
    lk = dict(winSize=(25, 25), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p2, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p1, None, **lk)
    if p2 is None:
        return None
    p1b, st2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, p2, None, **lk)
    if p1b is None:
        return None
    fb = np.linalg.norm(p1.reshape(-1, 2) - p1b.reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.0)
    if int(ok.sum()) < 25:
        return None
    return p1.reshape(-1, 2)[ok].astype(np.float64), p2.reshape(-1, 2)[ok].astype(np.float64)


def track_lk(g1, mag, fov, g2):
    pts = cv2.goodFeaturesToTrack(g1, 1200, 0.005, 6, mask=fov, blockSize=7)
    if pts is None or len(pts) < 40:
        return None
    xs = np.clip(np.round(pts[:, 0, 0]).astype(int), 0, mag.shape[1] - 1)
    ys = np.clip(np.round(pts[:, 0, 1]).astype(int), 0, mag.shape[0] - 1)
    strength = mag[ys, xs]
    if len(pts) > 80:
        pts = pts[strength >= np.percentile(strength, 35)]
    if len(pts) < 30:
        return None
    return _lk(g1, g2, pts)


def track_sift(g1, fov, g2):
    sift = cv2.SIFT_create(nfeatures=2500)
    k1, d1 = sift.detectAndCompute(g1, fov)
    k2, d2 = sift.detectAndCompute(g2, fov)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    p1, p2 = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            p1.append(k1[a.queryIdx].pt)
            p2.append(k2[a.trainIdx].pt)
    if len(p1) < 30:
        return None
    return np.asarray(p1, np.float64), np.asarray(p2, np.float64)


def parallax(R, p1, p2, K):
    x = np.hstack([p1, np.ones((len(p1), 1))])
    pred = K @ (R @ (np.linalg.inv(K) @ x.T))
    z = np.clip(pred[2], 1e-6, None)
    xy = (pred[:2] / z).T
    return float(np.median(np.linalg.norm(p2 - xy, axis=1)))


def solve_essential(p1, p2, K, max_rot):
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.2)
    if E is None or E.shape != (3, 3):
        return None
    n_in, R, t, m2 = cv2.recoverPose(E, p1, p2, K, mask=mask)
    if n_in < 20:
        return None
    R = so3(R)
    if rot_deg(R) > max_rot:
        return None
    t = t.ravel()
    t = t / max(np.linalg.norm(t), 1e-8)
    inl = m2.ravel() > 0
    if int(inl.sum()) >= 15:
        p1, p2 = p1[inl], p2[inl]
    para = parallax(R, p1, p2, K)
    if not np.isfinite(para) or para > 80:
        return None
    return R, t, float(para), int(n_in) / max(len(inl), 1)


def solve_homography(p1, p2, K, max_rot):
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 2.5)
    if H is None or mask is None or int(mask.sum()) < 25:
        return None
    inl = mask.ravel().astype(bool)
    ratio = float(inl.mean())
    p1i, p2i = p1[inl], p2[inl]
    try:
        nsol, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    except cv2.error:
        return None
    P0 = K @ np.eye(3, 4)
    best = None
    best_score = -1
    for R, t, n in zip(Rs, ts, ns):
        R = so3(R)
        if rot_deg(R) > max_rot:
            continue
        t = np.asarray(t, dtype=np.float64).reshape(3)
        n = np.asarray(n, dtype=np.float64).reshape(3)
        if n[2] < 0:
            n = -n
            t = -t
        P1 = K @ np.hstack([R, t.reshape(3, 1)])
        Xh = cv2.triangulatePoints(P0, P1, p1i.T, p2i.T)
        X = (Xh[:3] / np.clip(Xh[3], 1e-8, None)).T
        X2 = (R @ X.T + t.reshape(3, 1)).T
        front = int(((X[:, 2] > 0) & (X2[:, 2] > 0)).sum())
        if front > best_score:
            best_score = front
            tn = t / max(np.linalg.norm(t), 1e-8)
            best = (R, tn, best_score)
    if best is None or best_score < 20:
        return None
    R, t, score = best
    para = parallax(R, p1i, p2i, K)
    if not np.isfinite(para) or para > 80:
        return None
    return R, t, float(para), ratio


def step_from_pair(g1, mag, fov, g2, K, wide, max_rot):
    tracked = track_sift(g1, fov, g2) if wide else track_lk(g1, mag, fov, g2)
    if tracked is None and wide:
        tracked = track_lk(g1, mag, fov, g2)
    if tracked is None:
        return {"hold": "lost"}
    p1, p2 = tracked
    flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
    if flow < 0.45:
        return {"hold": "still", "flow": flow}
    ess = solve_essential(p1, p2, K, max_rot)
    homo = solve_homography(p1, p2, K, max_rot)
    if ess is None and homo is None:
        return {"hold": "pose", "flow": flow}
    return {"hold": None, "flow": flow, "ess": ess, "homo": homo, "n": len(p1)}


def chain(steps, model, invert):
    c = np.eye(4)
    centers = [c[:3, 3].copy()]
    used = 0
    for rec in steps:
        R = np.eye(3)
        t = np.zeros(3)
        if rec.get("hold") is None:
            src = rec["homo"] if model == "homo" else rec["ess"]
            if model == "auto":
                homo, ess = rec["homo"], rec["ess"]
                if homo is not None and homo[3] >= 0.45:
                    src = homo
                else:
                    src = ess if ess is not None else homo
            if src is not None:
                R, tdir, para, _q = src
                t = tdir * max(para, 0.0)
                used += 1
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        if invert:
            T = np.linalg.inv(T)
        c = c @ T
        centers.append(c[:3, 3].copy())
    return np.asarray(centers), used


def relatives(paths, K, wide, max_rot):
    steps = []
    counts = {"lost": 0, "still": 0, "pose": 0, "ok": 0}
    img = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    g1, mag, fov = prep(img)
    for i, path in enumerate(paths[1:], start=1):
        nxt = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if nxt is None:
            steps.append({"hold": "lost"})
            counts["lost"] += 1
            continue
        g2, mag2, fov2 = prep(nxt)
        rec = step_from_pair(g1, mag, fov, g2, K, wide, max_rot)
        key = rec["hold"] if rec["hold"] else "ok"
        counts[key] = counts.get(key, 0) + 1
        steps.append(rec)
        g1, mag, fov = g2, mag2, fov2
        if i % 250 == 0:
            print(f"    {i}/{len(paths)-1} {counts}", flush=True)
    return steps, counts


def load_jobs(screen, K0):
    real = {r["name"]: r for r in screen["real_usable"]}
    sim = {r["name"]: r for r in screen["sim_with_images"]}
    jobs = []
    for name in ("traj_20260923_023422", "traj_20260924_144019"):
        row = real[name]
        pose = np.load(row["pose"])
        usable = np.flatnonzero(pose["usable"] == 1)
        kept = [int(i) for i in usable
                if (Path(row["image_dir"]) / f"{int(i):06d}.jpg").is_file()]
        sel = subsample_idx(len(kept), 30)
        ids = [kept[i] for i in sel]
        span = []
        for i in range(ids[0], ids[-1] + 1):
            p = Path(row["image_dir"]) / f"{i:06d}.jpg"
            if p.is_file():
                span.append((i, p))
        jobs.append({
            "name": name,
            "span": span,
            "ids": ids,
            "gt": pose["p"][ids] * 1000.0,
            "K": K0,
        })
    row = sim["sim_20260924_172955_s10095"]
    z = np.load(row["pose"])
    key = "p_W_C" if "p_W_C" in z.files else "p"
    gt_all = np.asarray(z[key], dtype=np.float64)
    if np.nanmax(np.abs(gt_all)) < 5:
        gt_all *= 1000.0
    paths = sorted(Path(row["image_dir"]).glob("*.jpg"))
    n = min(len(paths), len(gt_all))
    sel = subsample_idx(n, 30)
    jobs.append({
        "name": row["name"] if False else "sim_20260924_172955_s10095",
        "span": [(i, paths[i]) for i in range(n)],
        "ids": sel.tolist(),
        "gt": gt_all[sel],
        "K": K0,
    })
    return jobs


def evaluate(job):
    paths = [p for _, p in job["span"]]
    id_to_k = {i: k for k, (i, _) in enumerate(job["span"])}
    sample = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    h, w = sample.shape[:2]
    K = scale_K(job["K"], (w, h))
    print(f"\n{job['name']} video-frames {len(paths)} {w}x{h}", flush=True)
    rows = []
    proto = [dict(job["span"])[i] for i in job["ids"]]
    passes = [
        ("f30", proto, list(range(len(proto))), True, MAX_ROT_GAP),
        ("video", paths, [id_to_k[i] for i in job["ids"]], False, MAX_ROT_VIDEO),
    ]
    for label, seq, index, wide, max_rot in passes:
        print(f"  {label} n={len(seq)} wide={wide}", flush=True)
        steps, counts = relatives(seq, K, wide, max_rot)
        print(f"  counts {counts}", flush=True)
        gt = np.asarray(job["gt"])[:len(index)]
        for model in ("auto", "homo", "ess"):
            for invert in (False, True):
                centers, used = chain(steps, model, invert)
                ate, scale = sim3_ate(centers[index], gt)
                tag = f"{label}:{model}{'_inv' if invert else ''}"
                rec = {"tag": tag, "ate_mm": round(ate, 2), "scale": round(scale, 3),
                       "posed_steps": used}
                rows.append(rec)
                print(f"  {tag:22} ate {rec['ate_mm']:8} scale {rec['scale']}", flush=True)
    return rows


def main():
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    K0 = np.asarray(screen["K"], dtype=np.float64)
    summary = {}
    for job in load_jobs(screen, K0):
        summary[job["name"]] = evaluate(job)
        (HERE / "ego_pose_opt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", HERE / "ego_pose_opt.json", flush=True)


if __name__ == "__main__":
    main()
