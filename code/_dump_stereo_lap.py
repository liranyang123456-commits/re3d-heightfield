#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump per-sequence Stereo-Lap metrics as LaTeX rows for the supplement."""
import json

d = json.load(open(r"E:\MIS_TMI_Re_3D\benchmark_stereo_lap\stereo_lap_merged.json"))
seqs = d["sequences"]

print("seq & ours_absrel & ours_d1 & sgm_absrel & sgm_d1 & mono_absrel & mono_d1 & o_shape & s_shape & m_shape")
for k in sorted(seqs):
    s = seqs[k]
    o, g, m = s["ours"], s["sgm"], s.get("mono")
    mo = f"{m['AbsRel']:.3f} & {m['d1_25']:.3f}" if m else "-- & --"
    ms = m["shape"][0] if m else "-"
    print(f"{k} & {o['AbsRel']:.3f} & {o['d1_25']:.3f} & {g['AbsRel']:.3f} & "
          f"{g['d1_25']:.3f} & {mo} & {o['shape'][0]} & {g['shape'][0]} & {ms} \\\\")

agg = d["summary"]["aggregate"]
for name in ("Ours", "Stereo SGM", "Mono"):
    a = agg[name]
    print(f"{name}: avg_absrel={a['avg_absrel']:.4f} avg_l31={a['avg_lambda3_lambda1']:.3f} "
          f"N/F/C={a['normal3d']}/{a['flat']}/{a['cone']} time={a['avg_time']:.2f}s")
