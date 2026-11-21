#!/usr/bin/env python3
"""
Evaluation harness that mirrors the organisers' scoring as closely as we can.

For every degraded sequence folder under --data (built by make_dataset.py):
  1. runs src/main.py on the zip (in-process, same code path as the CLI),
  2. validates the output format (row count == input rows, identical
     timestamps, finite values, unit quaternions),
  3. aligns the estimate to ground truth with SE(3) Umeyama (no scale, as evo
     ``-a``) and reports ATE RMSE for
        all       - whole trajectory   (proxy for the waypoint score, 30 %)
        blind     - last 5 s after IMU failure            (20 %)
        dropout   - poses within 1 s after a >=50 ms gap  (10 %)
     plus rotation RMSE and RPE (1 s translational drift),
  4. reports the B0 baseline (constant position) for reference,
  5. optionally measures peak heap (tracemalloc) and runtime per IMU second.

Usage
    python tools/evaluate.py [--data data/degraded] [--filter room] \
                             [--set key=val ...] [--mem] [--json out.json]
"""

import argparse
import glob
import importlib.util
import io
import json
import os
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_main():
    spec = importlib.util.spec_from_file_location("pf_main", os.path.join(ROOT, "src", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_tum(path):
    ts, rows = [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            sec, _, frac = p[0].partition(".")
            frac = (frac + "000000000")[:9]
            ts.append(int(sec) * 10**9 + int(frac))
            rows.append([float(v) for v in p[1:8]])
    return np.array(ts, np.int64), np.array(rows, float)


def read_input_ts(csv_path):
    ts = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ts.append(int(line.split(",", 1)[0]))
    return np.array(ts, np.int64)


def umeyama_se3(src, dst):
    """R, t minimising ||dst - (R src + t)|| (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    return R, mu_d - R @ mu_s


def quat_to_R(q):  # q: [N,4] x y z w
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def associate(est_ts, gt_ts, max_diff_ns=10_000_000):
    """Nearest-neighbour association (as evo, default max diff 10 ms)."""
    j = np.searchsorted(gt_ts, est_ts)
    j = np.clip(j, 1, len(gt_ts) - 1)
    left, right = gt_ts[j - 1], gt_ts[j]
    jj = np.where(np.abs(est_ts - left) <= np.abs(right - est_ts), j - 1, j)
    ok = np.abs(gt_ts[jj] - est_ts) <= max_diff_ns
    return ok, jj


def metrics(est_ts, est, gt_ts, gt, in_ts):
    ok, jj = associate(est_ts, gt_ts)
    E, Gt = est[ok], gt[jj[ok]]
    t = est_ts[ok]
    R, tr = umeyama_se3(E[:, :3], Gt[:, :3])
    P = E[:, :3] @ R.T + tr
    err = np.linalg.norm(P - Gt[:, :3], axis=1)
    # rotation error after alignment
    Re = np.einsum("ij,njk->nik", R, quat_to_R(E[:, 3:7]))
    Rg = quat_to_R(Gt[:, 3:7])
    rel = np.einsum("nji,njk->nik", Rg, Re)
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    # segments
    t_end = in_ts[-1]
    blind = t >= t_end - 5 * 10**9
    gaps = np.flatnonzero(np.diff(np.unique(in_ts)) >= 50_000_000)
    uts = np.unique(in_ts)
    drop = np.zeros(len(t), bool)
    for g in gaps:
        t1 = uts[g + 1]
        drop |= (t >= t1) & (t < t1 + 10**9)
    drop &= ~blind

    def rmse(m):
        return float(np.sqrt(np.mean(err[m] ** 2))) if m.any() else float("nan")

    # baseline B0: a constant position aligned to the GT centroid
    b0 = np.linalg.norm(Gt[:, :3] - Gt[:, :3].mean(0), axis=1)

    def rmse_b0(m):
        return float(np.sqrt(np.mean(b0[m] ** 2))) if m.any() else float("nan")

    # RPE: translational drift over 1 s windows (aligned frame)
    dt = 10**9
    k = np.searchsorted(t, t + dt)
    v = k < len(t)
    i0, i1 = np.flatnonzero(v), k[v]
    rpe = np.linalg.norm((P[i1] - P[i0]) - (Gt[i1, :3] - Gt[i0, :3]), axis=1)
    all_m = np.ones(len(t), bool)
    return {
        "matched": int(ok.sum()),
        "ate_all": rmse(all_m), "ate_blind": rmse(blind), "ate_dropout": rmse(drop),
        "b0_all": rmse_b0(all_m), "b0_blind": rmse_b0(blind), "b0_dropout": rmse_b0(drop),
        "rot_rmse_deg": float(np.sqrt(np.mean(ang ** 2))),
        "rpe_1s": float(np.sqrt(np.mean(rpe ** 2))) if len(rpe) else float("nan"),
        "n_gaps": int(len(gaps)),
    }


def check_format(est_ts, est, in_ts):
    issues = []
    if len(est_ts) != len(in_ts):
        issues.append("row count %d != input %d" % (len(est_ts), len(in_ts)))
    elif not np.array_equal(est_ts, in_ts):
        issues.append("timestamps differ from input")
    if not np.isfinite(est).all():
        issues.append("non-finite values")
    qn = np.linalg.norm(est[:, 3:7], axis=1)
    if np.abs(qn - 1).max() > 1e-5:
        issues.append("non-unit quaternion (max dev %.2e)" % np.abs(qn - 1).max())
    return issues


def measure(mod, zip_path, out_path, cfg, mem):
    t0 = time.perf_counter()
    mod.run(zip_path, out_path, cfg)
    wall = time.perf_counter() - t0
    peak = None
    if mem:
        tracemalloc.start()
        mod.run(zip_path, out_path, cfg)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return wall, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "degraded"))
    ap.add_argument("--filter", default="", help="substring filter on sequence names")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--mem", action="store_true", help="also measure tracemalloc peak")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    mod = load_main()
    cfg = dict(mod.CFG)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v

    seqs = sorted(d for d in glob.glob(os.path.join(args.data, "*"))
                  if os.path.isdir(d) and args.filter in os.path.basename(d))
    out_dir = os.path.join(ROOT, "data", "out")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    hdr = "%-26s %7s %7s %7s | %7s %7s %7s | %6s %6s | %6s %7s %s" % (
        "sequence", "ATE", "blind", "drop", "B0", "B0bl", "B0dr", "rotdeg", "RPE1s",
        "ms/s", "peakMB", "format")
    if not args.quiet:
        print(hdr)
        print("-" * len(hdr))
    for d in seqs:
        name = os.path.basename(d)
        zip_path = d + ".zip"
        if not os.path.isfile(zip_path):
            continue
        out_path = os.path.join(out_dir, name + ".txt")
        wall, peak = measure(mod, zip_path, out_path, cfg, args.mem)
        in_ts = read_input_ts(os.path.join(d, "imu0", "data.csv"))
        est_ts, est = read_tum(out_path)
        gt_ts, gt = read_tum(os.path.join(d, "gt.tum"))
        issues = check_format(est_ts, est, in_ts)
        m = metrics(est_ts, est, gt_ts, gt, in_ts)
        dur = (in_ts[-1] - in_ts[0]) * 1e-9
        m["ms_per_s"] = 1000 * wall / dur
        m["peak_mb"] = peak / 2**20 if peak is not None else None
        m["format_issues"] = issues
        results[name] = m
        if not args.quiet:
            print("%-26s %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | %6.2f %6.3f | %6.2f %7s %s" % (
                name, m["ate_all"], m["ate_blind"], m["ate_dropout"],
                m["b0_all"], m["b0_blind"], m["b0_dropout"], m["rot_rmse_deg"], m["rpe_1s"],
                m["ms_per_s"], "%.2f" % m["peak_mb"] if peak else "-",
                "OK" if not issues else "; ".join(issues)))
    if results:
        keys = ["ate_all", "ate_blind", "ate_dropout", "b0_all", "b0_blind", "b0_dropout",
                "rot_rmse_deg", "rpe_1s", "ms_per_s"]
        mean = {k: float(np.nanmean([r[k] for r in results.values()])) for k in keys}
        results["_mean"] = mean
        if not args.quiet:
            print("-" * len(hdr))
            print("%-26s %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | %6.2f %6.3f | %6.2f" % (
                "MEAN", *[mean[k] for k in keys]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"cfg": cfg, "results": results}, f, indent=2)
    return results


if __name__ == "__main__":
    main()
