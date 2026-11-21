#!/usr/bin/env python3
"""
Build degraded evaluation sets from raw EuRoC / TUM-VI sequences.

For each raw sequence folder (containing ``mav0/imu0/data.csv`` and a ground
truth ``mav0/state_groundtruth_estimate0/data.csv`` or ``mav0/mocap0/data.csv``)
this script produces, under --out:

    <seq>_<variant>/imu0/data.csv   degraded IMU in EuRoC format (gyro, accel)
    <seq>_<variant>.zip             the same, zipped like the evaluation input
    <seq>_<variant>/gt.tum          ground truth in TUM format
    <seq>_<variant>/degrade_meta.json  statistics from the organiser script

Variants
    ordered : columns are re-ordered to (ts, accel, gyro) before calling the
              organiser script, i.e. the noise model is applied as intended.
    literal : the organiser script is run directly on the EuRoC file, whose
              column order is (ts, gyro, accel). The script then applies the
              *accelerometer* noise model to the gyroscope and vice versa.
              We test against both because we cannot know which one the
              hidden evaluation set used.

The organiser script is NOT part of this repository; pass its path with
--degrade (default: $DEGRADE_IMU or ../personal/degrade_imu.py).

Requires pandas (development dependency only).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

EUROC_HEADER = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
                "a_RS_S_z [m s^-2]")


def find_file(seq_dir, sub):
    for root, _, files in os.walk(seq_dir):
        if root.replace("\\", "/").endswith(sub) and "data.csv" in files:
            return os.path.join(root, "data.csv")
    return None


def read_imu(path):
    df = pd.read_csv(path, comment=None)
    df = df.iloc[:, :7]
    df.columns = ["ts", "wx", "wy", "wz", "ax", "ay", "az"]
    df["ts"] = df["ts"].astype(np.int64)
    return df


def write_euroc(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(EUROC_HEADER + "\n")
        for row in df.itertuples(index=False):
            f.write("%d,%s\n" % (row[0], ",".join("%.6f" % v for v in row[1:7])))


def gt_to_tum(gt_csv, out_path):
    gt = pd.read_csv(gt_csv)
    gt = gt.iloc[:, :8]
    ts = gt.iloc[:, 0].astype(np.int64).to_numpy()
    p = gt.iloc[:, 1:4].to_numpy(float)
    q = gt.iloc[:, 4:8].to_numpy(float)  # w x y z
    with open(out_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for t, pp, qq in zip(ts, p, q):
            f.write("%d.%09d %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (
                t // 10**9, t % 10**9, pp[0], pp[1], pp[2], qq[1], qq[2], qq[3], qq[0]))


def run_degrade(script, inp, out, meta):
    cmd = [sys.executable, script, inp, out, "--meta", meta]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def build(seq_dir, name, script, out_root, variants):
    imu = find_file(seq_dir, "imu0")
    gt = find_file(seq_dir, "state_groundtruth_estimate0") or find_file(seq_dir, "mocap0")
    if imu is None or gt is None:
        print("skip %s: missing imu0 or ground truth" % seq_dir)
        return
    df = read_imu(imu)
    with tempfile.TemporaryDirectory() as tmp:
        for var in variants:
            dst = os.path.join(out_root, "%s_%s" % (name, var))
            meta = os.path.join(dst, "degrade_meta.json")
            os.makedirs(dst, exist_ok=True)
            if var == "ordered":
                src = os.path.join(tmp, "in_ordered.csv")
                df[["ts", "ax", "ay", "az", "wx", "wy", "wz"]].to_csv(
                    src, index=False, header=["#timestamp [ns]", "ax", "ay", "az", "wx", "wy", "wz"])
                deg = os.path.join(tmp, "out_ordered.csv")
                run_degrade(script, src, deg, meta)
                out = pd.read_csv(deg)
                out.columns = ["ts", "ax", "ay", "az", "wx", "wy", "wz"]
                out = out[["ts", "wx", "wy", "wz", "ax", "ay", "az"]]
            else:  # literal
                deg = os.path.join(tmp, "out_literal.csv")
                run_degrade(script, imu, deg, meta)
                out = pd.read_csv(deg)
                out.columns = ["ts", "wx", "wy", "wz", "ax", "ay", "az"]
            out["ts"] = out["ts"].astype(np.int64)
            csv_path = os.path.join(dst, "imu0", "data.csv")
            write_euroc(out, csv_path)
            with zipfile.ZipFile(dst + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
                z.write(csv_path, "imu0/data.csv")
            gt_to_tum(gt, os.path.join(dst, "gt.tum"))
            # keep the clean IMU timestamps for reference (dropout analysis)
            np.save(os.path.join(dst, "clean_ts.npy"), df["ts"].to_numpy())
            with open(meta) as f:
                m = json.load(f)
            print("%-28s rows %6d -> %6d  dur %6.1fs  dropped %5d  dup %3d  failed %4d" % (
                os.path.basename(dst), m["rows_in"], m["rows_out"], m["duration_s"],
                m["dropped_total"], m["duplicated_rows"], m["failure_frames"]))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_script = os.environ.get(
        "DEGRADE_IMU", os.path.join(here, "..", "..", "personal", "degrade_imu.py"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", nargs="+", help="raw sequence folders")
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "degraded"))
    ap.add_argument("--degrade", default=default_script, help="organiser degrade_imu.py")
    ap.add_argument("--variants", default="ordered,literal")
    args = ap.parse_args()
    if not os.path.isfile(args.degrade):
        sys.exit("organiser script not found: %s (use --degrade)" % args.degrade)
    os.makedirs(args.out, exist_ok=True)
    for seq in args.raw:
        name = os.path.basename(os.path.normpath(seq)).replace("dataset-", "")
        build(seq, name, os.path.abspath(args.degrade), args.out, args.variants.split(","))


if __name__ == "__main__":
    main()
