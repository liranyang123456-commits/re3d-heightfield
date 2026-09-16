#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restart session 4 with auto-retry (transient CUDA faults) and checkpoints."""
import os, subprocess, time

PY = r"C:\Users\lry\.conda\envs\py3d\python.exe"
DRIVER = r"e:\MIS_TMI_Re_3D\re3d_cmpb_results\code\_sweep_op3_driver.py"
OP3 = r"e:\MIS_TMI_Re_3D\re3d_cmpb_results\code\Op3_Dec14_newcapture.py"
SESS = "vio_seq_20260912_104835"
MAX_TRY = 3

for attempt in range(1, MAX_TRY + 1):
    env = dict(os.environ)
    env["OP3_PATH"] = OP3
    env["OP3_IMG_DIR"] = rf"E:\MIS_TMI_Re_3D\new_capture\{SESS}\images"
    env["OP3_SWEEP_ROOT"] = r"E:\MIS_TMI_Re_3D\new_capture_runs"
    # the 200-iter render loop hung after iter ~150 on this long sequence;
    # cap at 100 (loss is near-converged by then) to stay clear of it
    env["RENDERING_MAX_ITERATIONS"] = "100"
    print(f"\n[RESTART {attempt}/{MAX_TRY}] {SESS}  {time.strftime('%H:%M:%S')}", flush=True)
    log_path = rf"E:\MIS_TMI_Re_3D\new_capture_runs\{SESS}_attempt{attempt}.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        r = subprocess.run([PY, DRIVER, "--param", "default", "--frames", "2000",
                            "--tag", SESS], env=env, stdout=lf,
                           stderr=subprocess.STDOUT)
    print(f"[EXIT {attempt}] code={r.returncode}, full log: {log_path}", flush=True)
    if r.returncode == 0:
        print(f"[DONE] {SESS} succeeded on attempt {attempt}", flush=True)
        break
    time.sleep(10)
else:
    print(f"[FAIL] {SESS} failed after {MAX_TRY} attempts", flush=True)
