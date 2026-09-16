#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the Dec-14 pipeline-of-record on all four newly captured sequences."""
import os, subprocess, sys

PY = r"C:\Users\lry\.conda\envs\py3d\python.exe"
DRIVER = r"e:\MIS_TMI_Re_3D\re3d_cmpb_results\code\_sweep_op3_driver.py"
OP3 = r"e:\MIS_TMI_Re_3D\re3d_cmpb_results\code\Op3_Dec14_newcapture.py"
NEW_ROOT = r"E:\MIS_TMI_Re_3D\new_capture"
RUN_ROOT = r"E:\MIS_TMI_Re_3D\new_capture_runs"

SESSIONS = [
    "pnp_seq_20260911_011115",
    "vio_seq_20260911_010902",
    "vio_seq_20260911_011013",
    "vio_seq_20260912_104835",
]

for sess in SESSIONS:
    env = dict(os.environ)
    env["OP3_PATH"] = OP3
    env["OP3_IMG_DIR"] = os.path.join(NEW_ROOT, sess, "images")
    env["OP3_SWEEP_ROOT"] = RUN_ROOT
    print(f"\n{'='*70}\n[RUN] {sess}\n{'='*70}", flush=True)
    r = subprocess.run(
        [PY, DRIVER, "--param", "default", "--frames", "2000", "--tag", sess],
        env=env)
    print(f"[DONE] {sess} exit={r.returncode}", flush=True)

print("\nALL SESSIONS DONE")
