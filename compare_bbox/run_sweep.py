# -*- coding: utf-8 -*-
"""
run_sweep.py — f ごとに frenet_transform.py の auto → convert を回す

使い方:
    python make_refpoint_csv.py      # 先に f ごとのCSVを作る
    python run_sweep.py
    python evaluate_refpoint.py
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRENET     = os.path.join(SCRIPT_DIR, "frenet_transform.py")

BASE_NAME = "scene_A_track2_recal2"
OUT_DIR   = "out_" + BASE_NAME   # make_refpoint_csv.py と同じ規則
F_VALUES  = [i / 8 for i in range(9)]   # 0, 0.125, ..., 1 の対称な9点

for f in F_VALUES:
    csv = os.path.join(OUT_DIR, f"{BASE_NAME}_f{f:.3f}.csv")
    npy = os.path.join(OUT_DIR, f"centerline_f{f:.3f}.npy")
    print(f"\n########## f = {f} ##########")
    subprocess.run([sys.executable, FRENET, "auto", csv, npy], check=True)
    subprocess.run([sys.executable, FRENET, "convert", csv, npy], check=True)