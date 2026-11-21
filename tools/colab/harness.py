# Colab evaluation farm: build degraded variants of every downloaded sequence
# and score src/main.py across all of them (same metrics as tools/evaluate.py).
#
# Setup in a fresh Colab runtime (see tools/colab/README.md):
#   /content/pf/src/main.py          <- copy of src/main.py
#   /content/pf/tools/degrade_imu.py <- organiser script (NOT in this repo)
#   /content/pf/tools/harness.py     <- this file
#   /content/pf/data/raw/<SEQ>/mav0/{imu0,state_groundtruth_estimate0}/data.csv
#
#   import sys; sys.path.insert(0, '/content/pf/tools'); import harness
#   harness.build('MH_01_easy'); harness.evaluate({'vel_hp_s': 6.0})
import os, sys, glob, importlib.util, zipfile, itertools, json
import numpy as np, pandas as pd

ROOT = os.environ.get("PF_ROOT", "/content/pf")
sys.path.insert(0, os.path.join(ROOT, "tools"))
from degrade_imu import degrade, CONFIG, SEED  # noqa: E402  (organiser code)

RAW, DEG = ROOT + "/data/raw", ROOT + "/data/degraded"
HDR = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
       "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")


def build(seq):
    imu = pd.read_csv(f"{RAW}/{seq}/mav0/imu0/data.csv"); imu.columns = ["ts", "wx", "wy", "wz", "ax", "ay", "az"]
    gt = pd.read_csv(f"{RAW}/{seq}/mav0/state_groundtruth_estimate0/data.csv").iloc[:, :8]
    g = np.c_[gt.iloc[:, 0].to_numpy(np.int64), gt.iloc[:, 1:8].to_numpy(float)]
    for var in ("ordered", "literal"):
        dst = f"{DEG}/{seq}_{var}"
        os.makedirs(dst, exist_ok=True)
        if var == "ordered":   # noise model applied as the organiser intended
            df = imu[["ts", "ax", "ay", "az", "wx", "wy", "wz"]].copy()
            df.columns = ["#timestamp [ns]", "ax", "ay", "az", "wx", "wy", "wz"]
            out, _ = degrade(df, dict(CONFIG), SEED)
            out.columns = ["ts", "ax", "ay", "az", "wx", "wy", "wz"]
            out = out[["ts", "wx", "wy", "wz", "ax", "ay", "az"]]
        else:                  # script run directly on EuRoC column order
            df = imu.copy(); df.columns = ["#timestamp [ns]", "wx", "wy", "wz", "ax", "ay", "az"]
            out, _ = degrade(df, dict(CONFIG), SEED)
            out.columns = ["ts", "wx", "wy", "wz", "ax", "ay", "az"]
        lines = [HDR] + ["%d,%s" % (r[0], ",".join("%.6f" % v for v in r[1:])) for r in out.itertuples(index=False)]
        with zipfile.ZipFile(dst + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("imu0/data.csv", "\n".join(lines) + "\n")
        np.save(dst + "/in_ts.npy", out["ts"].to_numpy(np.int64))
        np.save(dst + "/gt.npy", g)


def build_new():
    done = {os.path.basename(p)[:-len("_ordered.zip")] for p in glob.glob(DEG + "/*_ordered.zip")}
    for seq in sorted(os.listdir(RAW)):
        if seq not in done and os.path.isfile(f"{RAW}/{seq}/mav0/state_groundtruth_estimate0/data.csv"):
            build(seq); print("built", seq)


def load_main():
    spec = importlib.util.spec_from_file_location("pf_main", ROOT + "/src/main.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def read_tum(path):
    ts, rows = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split(); s, _, f = p[0].partition(".")
        ts.append(int(s) * 10**9 + int((f + "000000000")[:9])); rows.append([float(v) for v in p[1:8]])
    return np.array(ts, np.int64), np.array(rows)


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); S = (dst - md).T @ (src - ms) / len(src)
    U, _, Vt = np.linalg.svd(S); D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt; return R, md - R @ ms


def score(name, cfg, main_mod):
    d = f"{DEG}/{name}"; out = f"{ROOT}/data/out/{name}.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    main_mod.run(d + ".zip", out, cfg)
    ets, est = read_tum(out); in_ts = np.load(d + "/in_ts.npy"); g = np.load(d + "/gt.npy")
    assert len(ets) == len(in_ts) and np.array_equal(ets, in_ts), "format"
    gts = g[:, 0].astype(np.int64); j = np.clip(np.searchsorted(gts, ets), 1, len(gts) - 1)
    jj = np.where(np.abs(ets - gts[j - 1]) <= np.abs(gts[j] - ets), j - 1, j)
    ok = np.abs(gts[jj] - ets) <= 10_000_000
    E, G_ = est[ok, :3], g[jj[ok], 1:4]; t = ets[ok]
    R, tr = umeyama(E, G_); err = np.linalg.norm(E @ R.T + tr - G_, axis=1)
    b0 = np.linalg.norm(G_ - G_.mean(0), axis=1)
    blind = t >= in_ts[-1] - 5 * 10**9
    u = np.unique(in_ts); gaps = np.flatnonzero(np.diff(u) >= 50_000_000); drop = np.zeros(len(t), bool)
    for k in gaps:
        drop |= (t >= u[k + 1]) & (t < u[k + 1] + 10**9)
    drop &= ~blind
    r = lambda e, m: float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else float("nan")
    a = np.ones(len(t), bool)
    return dict(ate=r(err, a), blind=r(err, blind), drop=r(err, drop),
                b0=r(b0, a), b0_blind=r(b0, blind), b0_drop=r(b0, drop))


def evaluate(overrides=None, filt=""):
    m = load_main(); cfg = dict(m.CFG); cfg.update(overrides or {})
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(DEG + "/*.zip") if filt in p)
    return {n: score(n, cfg, m) for n in names}


def norm_score(res):
    """Rubric-weighted error relative to the do-nothing baseline (lower is
    better, 1.0 = same as B0): 0.5 ATE + 0.33 blind + 0.17 dropout."""
    s = [0.5 * v["ate"] / v["b0"] + 0.33 * v["blind"] / v["b0_blind"] + 0.17 * v["drop"] / v["b0_drop"]
         for v in res.values()]
    return float(np.mean(s)), float(np.max(s))


def sweep(grid, fixed=None, filt=""):
    keys = list(grid); rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        ov = dict(zip(keys, combo)); ov.update(fixed or {})
        r = evaluate(ov, filt); mean, worst = norm_score(r)
        rows.append((mean, worst, ov))
        print("%-60s norm %.3f (worst %.3f)" % (json.dumps(ov)[:60], mean, worst), flush=True)
    rows.sort(key=lambda x: x[0])
    return rows
