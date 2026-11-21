#!/usr/bin/env python3
"""
Path Finder - 6-DoF trajectory estimation from degraded IMU telemetry.

    python main.py <input> [-o estimated_trajectory.txt]

<input> may be
    * the evaluation zip (any member ending in ``imu0/data.csv`` is used),
    * a dataset folder containing ``imu0/data.csv`` (e.g. ``MH_01/mav0``), or
    * the CSV file itself.

Output: TUM pose file, one row per input IMU row (duplicates included),
``timestamp tx ty tz qx qy qz qw`` with the timestamp in seconds.

Pipeline (see README.md for details)
    1. Streaming CSV parse into compact int64/float32 buffers.
    2. Sensor sanity: column-order + unit detection, invalid / stuck / duplicate
       / failed (all-zero) rows, timestamp gaps (dropouts).
    3. Robust spike rejection (windowed Hampel filter).
    4. Static initialisation from the clean lead-in: gravity direction,
       gravity magnitude, gyro bias.
    5. Attitude: Mahony-style complementary filter (gyro + gravity vector)
       with online gyro-bias estimation; gaps bridged at constant rate.
    6. Position: gravity-compensated world acceleration, integrated with
       zero-phase drift suppression (high-pass) on velocity and position.
    7. Post-failure dead reckoning: damped constant-velocity extrapolation.
    8. Interpolation back onto every input timestamp, streamed to disk.

Only NumPy and the standard library are used; all large buffers are float32
or int64 and processed in chunks so the heap stays well below 8 MB for
typical EuRoC / TUM-VI sequences.
"""

import os

# The challenge mandates a single CPU thread: pin every BLAS/OpenMP backend
# before NumPy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

G0 = 9.80665

# Tunable parameters (values chosen on degraded EuRoC / TUM-VI sequences; see
# tools/evaluate.py and the README results table).
CFG = {
    "init_window_s": 1.8,     # clean lead-in used for initialisation
    "hampel_half": 4,         # half window (samples) of the spike filter
    "hampel_k": 8.0,          # rejection threshold in robust sigmas
    "att_kp": 0.05,           # accel->attitude correction gain [1/s]
    "att_ki": 0.002,          # gyro-bias integral gain [1/s^2]
    "att_acc_sigma": 1.5,     # |a|-g tolerance for the correction [m/s^2]
    "adapt_block_s": 2.0,     # block length for gyro-noise monitoring [s]
    "adapt_kp_max_ratio": 40.0,  # max gain boost when the gyro degrades (1 = off)
    "conf_power": 1.0,        # velocity confidence = noise_ratio^-power (0 = off)
    "blind_vmax_xy": 1.5,     # clamp on extrapolated horizontal speed [m/s]
    "blind_vmax_z": 0.5,      # clamp on extrapolated vertical speed [m/s]
    "vel_hp_s": 30.0,         # velocity high-pass window [s] (0 = off)
    "pos_hp_s": 30.0,        # position high-pass window [s] (0 = off)
    "pos_mode": "drag",        # "hp" | "drag" | "zero" (baseline: no translation)
    "drag_lp_s": 0.0,         # extra low-pass on the drag velocity [s]
    "att_mode": "filter",     # "filter" | "identity" (baseline)
    "blind_tau_s": 1.0,       # velocity decay constant after IMU failure
    "blind_vel_win_s": 1.0,   # window to estimate the last velocity
    "chunk": 2048,            # rows per processing chunk
}


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
def _make_opener(path):
    """Return a zero-arg callable that opens the IMU CSV as a binary stream."""
    if os.path.isdir(path):
        best = None
        for root, dirs, files in os.walk(path):
            dirs.sort()
            if "data.csv" in files and os.path.basename(root) == "imu0":
                cand = os.path.join(root, "data.csv")
                if best is None or len(cand) < len(best):
                    best = cand
        if best is None:
            raise FileNotFoundError("no imu0/data.csv under %s" % path)
        return lambda: open(best, "rb")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("imu0/data.csv")]
            if not names:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError("no CSV inside %s" % path)
        member = sorted(names, key=len)[0]

        class _ZipStream:
            def __enter__(self):
                self.z = zipfile.ZipFile(path)
                self.f = self.z.open(member)
                return self.f

            def __exit__(self, *exc):
                self.f.close()
                self.z.close()

        return _ZipStream
    return lambda: open(path, "rb")


def load_imu(path, chunk=1024):
    """Parse the CSV. Returns (ts int64[N], raw float32[N,6]) in file order."""
    opener = _make_opener(path)
    with opener() as f:
        n_lines = sum(1 for _ in f)
    ts = np.empty(n_lines, np.int64)
    raw = np.empty((n_lines, 6), np.float32)
    nan = float("nan")
    k = 0
    buf_t, buf_v = [], []
    with opener() as f:
        for line in f:
            s = line.strip()
            if not s or s[:1] == b"#":
                continue
            parts = s.split(b",")
            try:
                t = int(parts[0])
            except ValueError:
                try:
                    t = int(round(float(parts[0])))
                except ValueError:
                    continue  # header / garbage line
            vals = []
            for p in parts[1:7]:
                try:
                    vals.append(float(p))
                except ValueError:
                    vals.append(nan)
            while len(vals) < 6:
                vals.append(nan)
            buf_t.append(t)
            buf_v.append(vals)
            if len(buf_t) == chunk:
                m = len(buf_t)
                ts[k:k + m] = buf_t
                raw[k:k + m] = buf_v
                k += m
                buf_t, buf_v = [], []
        if buf_t:
            m = len(buf_t)
            ts[k:k + m] = buf_t
            raw[k:k + m] = buf_v
            k += m
    return ts[:k], raw[:k]


# --------------------------------------------------------------------------
# Sensor sanity
# --------------------------------------------------------------------------
def _sample_rows(n, cap=4000):
    step = max(1, n // cap)
    return slice(0, n, step)


def detect_layout(raw, ok):
    """Decide which triplet is the accelerometer and its unit scale.

    Returns (gyro_cols, acc_cols, acc_scale, swapped).
    The specification says gyro first, but we verify using the fact that the
    accelerometer norm is ~g on average.
    """
    sel = raw[_sample_rows(len(raw))][ok[_sample_rows(len(raw))]]
    if len(sel) == 0:
        return (0, 1, 2), (3, 4, 5), 1.0, False
    na = float(np.median(np.linalg.norm(sel[:, 0:3], axis=1)))
    nb = float(np.median(np.linalg.norm(sel[:, 3:6], axis=1)))

    def score(nm):  # distance to g in either m/s^2 or g units
        return min(abs(nm - G0) / G0, abs(nm - 1.0))

    swapped = score(na) < score(nb)
    acc_norm = na if swapped else nb
    acc_scale = G0 if abs(acc_norm - 1.0) < abs(acc_norm - G0) / G0 else 1.0
    if swapped:
        return (3, 4, 5), (0, 1, 2), acc_scale, True
    return (0, 1, 2), (3, 4, 5), acc_scale, False


def validity_mask(ts, raw):
    """Usable rows: finite, not all-zero, not a stuck/duplicate repeat,
    strictly increasing timestamp."""
    n = len(ts)
    ok = np.isfinite(raw).all(axis=1)
    ok &= np.any(raw != 0.0, axis=1)
    if n > 1:
        same = np.all(raw[1:] == raw[:-1], axis=1)      # stuck or duplicated
        ok[1:] &= ~same
        # strictly increasing time w.r.t. the running maximum
        run_max = np.maximum.accumulate(ts)
        ok[1:] &= ts[1:] > run_max[:-1]
    return ok


# --------------------------------------------------------------------------
# Spike rejection
# --------------------------------------------------------------------------
def hampel_inplace(x, half, k, chunk):
    """Windowed Hampel filter over the rows of x (float32 [M,C]), in place.
    Outliers are replaced by the local median. Processed in chunks."""
    m = len(x)
    if m < 2 * half + 3:
        return 0
    swv = np.lib.stride_tricks.sliding_window_view
    # global per-channel noise floor from first differences (robust)
    floor = np.empty(x.shape[1], np.float32)
    for c in range(x.shape[1]):
        d = np.abs(np.diff(x[:, c]))
        floor[c] = 1.4826 * float(np.median(d)) / math.sqrt(2.0) + 1e-6
        del d
    out_fixes = 0
    # Work on a copy of each chunk's neighbourhood so in-place updates do not
    # influence neighbouring medians.
    prev_tail = None
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        lo, hi = max(0, s - half), min(m, e + half)
        seg = np.array(x[lo:hi], dtype=np.float32)
        if prev_tail is not None:
            # restore original (unfiltered) values for the overlap
            seg[: len(prev_tail)] = prev_tail
        prev_tail = np.array(x[max(0, e - half):e], dtype=np.float32)
        padl = s - lo
        padr = hi - e
        # pad edges by reflection so every row has a full window
        if padl < half:
            seg = np.concatenate([seg[1:half - padl + 1][::-1], seg])
        if padr < half:
            seg = np.concatenate([seg, seg[len(seg) - 1 - (half - padr):-1][::-1]])
        win = swv(seg, 2 * half + 1, axis=0)           # [rows, C, W]
        med = np.median(win, axis=2)
        mad = np.median(np.abs(win - med[:, :, None]), axis=2)
        cur = seg[half:half + (e - s)]
        sigma = np.maximum(1.4826 * mad, floor)
        bad = np.abs(cur - med) > k * sigma
        if bad.any():
            out_fixes += int(bad.sum())
            blk = x[s:e]
            blk[bad] = med[bad]
    return out_fixes


# --------------------------------------------------------------------------
# Adaptive attitude gain
# --------------------------------------------------------------------------
def adaptive_kp(tv, gyr, acc, cfg):
    """Per-block accelerometer-correction gain.

    Jamming adds isotropic white noise that grows over the flight. Rotor
    vibration is strongly anisotropic and depends on the manoeuvre, so we
    track the noise of the QUIETEST gyro axis (robust sample-to-sample
    spread) in blocks, relative to its early-flight value. Where it rises,
    the gyro is being degraded and the attitude filter leans on gravity more
    (gain ~ ratio^2, the optimal complementary crossover for white noise).
    Returns (list of per-block gains, per-block noise ratio >= 1)."""
    blk = cfg["adapt_block_s"]
    t0 = float(tv[0])
    nb = int((float(tv[-1]) - t0) / blk) + 1
    edges = np.searchsorted(tv, t0 + blk * np.arange(nb + 1))
    sg = np.zeros(nb)
    sa = np.zeros(nb)
    for b in range(nb):
        lo, hi = edges[b], edges[b + 1]
        if hi - lo > 10:
            sg[b] = float(np.median(np.abs(np.diff(gyr[lo:hi], axis=0)), axis=0).min())
            sa[b] = float(np.median(np.abs(np.diff(acc[lo:hi], axis=0)), axis=0).min())
    valid = sg > 0
    lead = max(1, int(np.ceil(cfg["init_window_s"] / blk)))
    if valid.sum() < lead + 4:
        return [cfg["att_kp"]] * nb, np.ones(nb)
    # "in flight" blocks: accelerometer vibration well above the lead-in level
    sa_lead = float(np.median(sa[:lead][valid[:lead]])) if valid[:lead].any() else 0.0
    flight = valid & (sa > 3.0 * sa_lead)
    flight[:lead] = False
    if flight.sum() < 4:
        flight = valid.copy()
        flight[:lead] = False
    fidx = np.flatnonzero(flight)
    ref = float(np.median(sg[fidx[:max(3, len(fidx) // 4)]]))
    ref = max(ref, 1e-9)
    ratio = np.where(flight, sg / ref, 1.0)
    if nb >= 3:
        ratio = np.convolve(np.pad(ratio, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    gain = np.clip(ratio ** 2, 1.0, cfg["adapt_kp_max_ratio"])
    return (cfg["att_kp"] * gain).tolist(), np.maximum(ratio, 1.0)


# --------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z), body -> world
# --------------------------------------------------------------------------
def quat_from_two_vectors(a, b):
    """Quaternion rotating unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return (0.0, float(axis[0]), float(axis[1]), float(axis[2]))
    w = 1.0 + d
    q = np.array([w, c[0], c[1], c[2]])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


# --------------------------------------------------------------------------
# Attitude + world acceleration (pure-Python inner loop for speed)
# --------------------------------------------------------------------------
def attitude_and_world_acc(tv, gyr, acc, q0, bias0, g_mag, cfg, quat_out,
                           thrust_axis=None, kp_blocks=None):
    """Integrate attitude over valid samples.

    tv  : float64 [M] seconds
    gyr : float32 [M,3] (bias NOT removed), overwritten with nothing
    acc : float32 [M,3] body specific force; overwritten in place with the
          gravity-compensated world-frame acceleration
    quat_out : float32 [M,4] (w,x,y,z) written in place
    thrust_axis : optional body unit vector. If given, the gyro buffer is
          overwritten with the world-frame specific force perpendicular to
          this axis (multirotor rotor-drag signal, proportional to -velocity).
    """
    drag = thrust_axis is not None
    if drag:
        tx_, ty_, tz_ = (float(v) for v in thrust_axis)
    kp, ki = cfg["att_kp"], cfg["att_ki"]
    blk_s = cfg["adapt_block_s"]
    t_first = float(tv[0])
    nkb = len(kp_blocks) if kp_blocks else 0
    sig = cfg["att_acc_sigma"]
    inv2sig2 = 1.0 / (2.0 * sig * sig)
    identity = cfg["att_mode"] == "identity"
    qw, qx, qy, qz = q0
    bx, by, bz = bias0
    m = len(tv)
    chunk = cfg["chunk"]
    pgx = pgy = pgz = None
    tprev = None
    exp, sqrt, cos, sin = math.exp, math.sqrt, math.cos, math.sin
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        T = tv[s:e].tolist()
        Gm = gyr[s:e].tolist()
        Am = acc[s:e].tolist()
        Q = []
        W = []
        D = []
        for i in range(e - s):
            t = T[i]
            gx, gy, gz = Gm[i]
            ax, ay, az = Am[i]
            if tprev is not None and not identity:
                dt = t - tprev
                # trapezoidal rate (constant rate across gaps)
                wx = 0.5 * (gx + pgx) - bx
                wy = 0.5 * (gy + pgy) - by
                wz = 0.5 * (gz + pgz) - bz
                an = sqrt(ax * ax + ay * ay + az * az)
                if an > 1e-6:
                    # estimated gravity direction in body: R^T * [0,0,1]
                    vx = 2.0 * (qx * qz - qw * qy)
                    vy = 2.0 * (qy * qz + qw * qx)
                    vz = qw * qw - qx * qx - qy * qy + qz * qz
                    ux, uy, uz = ax / an, ay / an, az / an
                    ex = uy * vz - uz * vy
                    ey = uz * vx - ux * vz
                    ez = ux * vy - uy * vx
                    dev = an - g_mag
                    wgt = exp(-dev * dev * inv2sig2)
                    if dt > 0.05:     # do not trust correction over gaps
                        wgt *= 0.05 / dt
                    if nkb:
                        kb = int((t - t_first) / blk_s)
                        kpw = kp_blocks[kb if kb < nkb else nkb - 1] * wgt
                    else:
                        kpw = kp * wgt
                    wx += kpw * ex
                    wy += kpw * ey
                    wz += kpw * ez
                    kiw = ki * wgt * dt
                    bx -= kiw * ex
                    by -= kiw * ey
                    bz -= kiw * ez
                # q <- q * exp(0.5 * w * dt)
                hx, hy, hz = 0.5 * wx * dt, 0.5 * wy * dt, 0.5 * wz * dt
                th = sqrt(hx * hx + hy * hy + hz * hz)
                if th > 1e-12:
                    c = cos(th)
                    sn = sin(th) / th
                    dw, dx, dy, dz = c, hx * sn, hy * sn, hz * sn
                else:
                    dw, dx, dy, dz = 1.0, hx, hy, hz
                nw = qw * dw - qx * dx - qy * dy - qz * dz
                nx = qw * dx + qx * dw + qy * dz - qz * dy
                ny = qw * dy - qx * dz + qy * dw + qz * dx
                nz = qw * dz + qx * dy - qy * dx + qz * dw
                nn = 1.0 / sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
                qw, qx, qy, qz = nw * nn, nx * nn, ny * nn, nz * nn
            tprev = t
            pgx, pgy, pgz = gx, gy, gz
            # world-frame acceleration: R a - [0,0,g]
            r00 = 1 - 2 * (qy * qy + qz * qz)
            r01 = 2 * (qx * qy - qw * qz)
            r02 = 2 * (qx * qz + qw * qy)
            r10 = 2 * (qx * qy + qw * qz)
            r11 = 1 - 2 * (qx * qx + qz * qz)
            r12 = 2 * (qy * qz - qw * qx)
            r20 = 2 * (qx * qz - qw * qy)
            r21 = 2 * (qy * qz + qw * qx)
            r22 = 1 - 2 * (qx * qx + qy * qy)
            W.append((r00 * ax + r01 * ay + r02 * az,
                      r10 * ax + r11 * ay + r12 * az,
                      r20 * ax + r21 * ay + r22 * az - g_mag))
            Q.append((qw, qx, qy, qz))
            if drag:
                dp = ax * tx_ + ay * ty_ + az * tz_
                px, py, pz = ax - dp * tx_, ay - dp * ty_, az - dp * tz_
                D.append((r00 * px + r01 * py + r02 * pz,
                          r10 * px + r11 * py + r12 * pz,
                          r20 * px + r21 * py + r22 * pz))
        acc[s:e] = W
        quat_out[s:e] = Q
        if drag:
            gyr[s:e] = D


# --------------------------------------------------------------------------
# Zero-phase drift suppression
# --------------------------------------------------------------------------
def _moving_average(y, half):
    """Centered moving average with shrinking window at the edges (float64).
    Allocates only the cumulative-sum and output arrays."""
    n = len(y)
    if half <= 0 or n == 0:
        return np.array(y, np.float64)
    c = np.empty(n + 1, np.float64)
    c[0] = 0.0
    np.cumsum(y, out=c[1:])
    out = np.empty(n, np.float64)
    w = 2 * half + 1
    if n > w - 1:
        np.subtract(c[w:], c[:n + 1 - w], out=out[half:n - half])
        out[half:n - half] /= w
    k = min(half, n)
    i = np.arange(k)                       # left edge
    hi = np.minimum(i + half + 1, n)
    out[:k] = c[hi] / hi
    i = np.arange(max(n - half, k), n)     # right edge
    lo = np.maximum(i - half, 0)
    out[i] = (c[n] - c[lo]) / (n - lo)
    return out


def _highpass(y, half):
    """y - MA(MA(y)): zero-phase high-pass (triangular kernel)."""
    if half <= 0:
        return y
    return y - _moving_average(_moving_average(y, half), half)


def _vel_from_acc(a, dt):
    """Trapezoidal cumulative integral; consumes nothing, returns float64."""
    inc = np.empty(len(a), np.float64)
    inc[0] = 0.0
    np.add(a[1:], a[:-1], out=inc[1:])
    inc[1:] *= dt[1:]
    inc[1:] *= 0.5
    np.cumsum(inc, out=inc)
    return inc


def integrate_position(tv, acc_w, cfg, pos_out, drag_w=None, vel_w=None):
    """acc_w float32 [M,3] world accel; pos_out float32 [M,3].

    drag_w (optional, float32 [M,3]): world-frame rotor-drag specific force.
    For a multirotor it is approximately -k * horizontal velocity, i.e. a
    drift-free (but scaled and noisy) velocity observation. Its scale is
    fitted against the band-passed integrated acceleration, and the two are
    fused complementarily: high frequencies from the integral, low from drag.

    vel_w (optional, per adapt-block weights in [0,1]): confidence in the
    velocity direction; velocity is scaled by it before integration.

    Returns the final smoothed velocity (for dead reckoning)."""
    m = len(tv)
    if cfg["pos_mode"] == "zero" or m < 3:
        pos_out[:] = 0.0
        return np.zeros(3)
    dt = np.empty(m, np.float64)
    dt[0] = 0.0
    np.subtract(tv[1:], tv[:-1], out=dt[1:])
    med_dt = float(np.median(dt[1:])) if m > 1 else 0.005
    hv = int(round(cfg["vel_hp_s"] / med_dt / 2.0))
    hp = int(round(cfg["pos_hp_s"] / med_dt / 2.0))
    hd = int(round(cfg["drag_lp_s"] / med_dt / 2.0))
    nwin = max(1, int(round(cfg["blind_vel_win_s"] / med_dt)))
    use_drag = drag_w is not None and cfg["pos_mode"] == "drag"
    scale = 0.0
    if use_drag:
        # fit v ~= scale * drag on band-passed signals (horizontal axes)
        hs = max(1, int(round(0.25 / med_dt)))
        num = den = 0.0
        for ax in range(2):
            v = _vel_from_acc(acc_w[:, ax].astype(np.float64), dt)
            v = _moving_average(_highpass(v, hv), hs)
            d = _moving_average(_highpass(drag_w[:, ax].astype(np.float64), hv), hs)
            num += float(np.dot(v, d))
            den += float(np.dot(d, d))
            del v, d
        scale = num / den if den > 0 else 0.0
        if scale > 0:          # physically the drag opposes velocity
            scale = 0.0
    v_end = np.zeros(3)
    for ax in range(3):
        v = _vel_from_acc(acc_w[:, ax].astype(np.float64), dt)
        v = _highpass(v, hv)
        if use_drag and ax < 2 and scale != 0.0:
            d = drag_w[:, ax].astype(np.float64) * scale
            d = _moving_average(_moving_average(d, hv), hv)
            if hd > 0:
                d = _moving_average(d, hd)
            v += d
            del d
        if vel_w is not None:
            blk = cfg["adapt_block_s"]
            kb = ((tv - tv[0]) / blk).astype(np.int32)
            np.minimum(kb, len(vel_w) - 1, out=kb)
            v *= vel_w[kb]
            del kb
        v_end[ax] = float(np.mean(v[-nwin:]))
        p = _vel_from_acc(v, dt)
        del v
        p = _highpass(p, hp)
        pos_out[:, ax] = p
        del p
    return v_end


# --------------------------------------------------------------------------
# Main estimator
# --------------------------------------------------------------------------
def estimate(ts, raw, cfg=CFG, log=None):
    """Returns (tv[M] seconds rel. ts0, pos[M,3], quat[M,4] wxyz, extras)."""
    say = log or (lambda *a: None)
    n = len(ts)
    ok = validity_mask(ts, raw)
    gcols, acols, ascale, swapped = detect_layout(raw, ok)
    say("rows=%d valid=%d swapped_columns=%s acc_scale=%.3f"
        % (n, int(ok.sum()), swapped, ascale))

    idx = np.flatnonzero(ok).astype(np.int32)
    del ok
    if len(idx) < 10:
        raise RuntimeError("not enough valid IMU rows")
    m = len(idx)
    ts0 = int(ts[0])
    # Compact the valid rows in place at the front of the raw buffer (safe:
    # idx[k] >= k, so a row is always read before its slot is overwritten).
    tv = np.empty(m, np.float64)
    ch = cfg["chunk"]
    for s in range(0, m, ch):
        sl = idx[s:s + ch]
        tv[s:s + ch] = (ts[sl] - ts0) * 1e-9
        raw[s:s + len(sl)] = raw[sl]
    del idx
    gyr = raw[:m, gcols[0]:gcols[0] + 3]   # views, no copies
    acc = raw[:m, acols[0]:acols[0] + 3]
    if ascale != 1.0:
        acc *= ascale

    # spike rejection
    nfix = hampel_inplace(gyr, cfg["hampel_half"], cfg["hampel_k"], ch)
    nfix += hampel_inplace(acc, cfg["hampel_half"], cfg["hampel_k"], ch)
    say("spike samples replaced=%d" % nfix)

    # static initialisation from the clean lead-in
    t_init = tv[0] + cfg["init_window_s"]
    k_init = max(20, int(np.searchsorted(tv, t_init)))
    k_init = min(k_init, m)
    a_mean = acc[:k_init].astype(np.float64).mean(axis=0)
    g_mag = float(np.median(np.linalg.norm(acc[:k_init], axis=1)))
    gyr_std = float(gyr[:k_init].astype(np.float64).std(axis=0).max())
    acc_std = float(acc[:k_init].astype(np.float64).std(axis=0).max())
    static = gyr_std < 0.05 and acc_std < 0.5
    bias0 = gyr[:k_init].astype(np.float64).mean(axis=0) if static else np.zeros(3)
    if not (8.0 < g_mag < 11.5):
        g_mag = G0
    q0 = quat_from_two_vectors(a_mean, np.array([0.0, 0.0, 1.0]))
    say("init: static=%s g=%.4f bias=%s" % (static, g_mag, np.round(bias0, 5)))

    quat = np.empty((m, 4), np.float32)
    thrust = a_mean / np.linalg.norm(a_mean) if cfg["pos_mode"] == "drag" else None
    kp_blocks, noise_ratio = adaptive_kp(tv, gyr, acc, cfg)
    say("gyro noise ratio (max/early)=%.2f  kp range %.3f..%.3f"
        % (float(noise_ratio.max()), min(kp_blocks), max(kp_blocks)))
    attitude_and_world_acc(tv, gyr, acc, q0, tuple(bias0), g_mag, cfg, quat, thrust,
                           kp_blocks)
    pos = np.empty((m, 3), np.float32)
    # confidence weight on velocity: heading degrades with gyro jamming
    pw = cfg["conf_power"]
    vel_w = (noise_ratio ** -pw) if pw > 0 else None
    v_end = integrate_position(tv, acc, cfg, pos, gyr if thrust is not None else None, vel_w)
    del acc, gyr
    vxy = float(np.hypot(v_end[0], v_end[1]))
    if vxy > cfg["blind_vmax_xy"]:
        v_end[:2] *= cfg["blind_vmax_xy"] / vxy
    v_end[2] = float(np.clip(v_end[2], -cfg["blind_vmax_z"], cfg["blind_vmax_z"]))
    return tv, pos, quat, {"v_end": v_end}


def write_tum(out_path, ts, tv, pos, quat, v_end, cfg=CFG, chunk=1024):
    """Interpolate the estimate onto every input timestamp and stream it out."""
    ts0 = int(ts[0])
    t_last = float(tv[-1])
    tau = cfg["blind_tau_s"]
    # make quaternion sign continuous for safe component interpolation
    m = len(quat)
    parity = 0
    prev = quat[0].copy()
    for s in range(1, m, chunk):
        blk = quat[s:s + chunk]
        prv = np.empty_like(blk)
        prv[0] = prev
        prv[1:] = blk[:-1]
        neg = np.einsum("ij,ij->i", blk, prv) < 0
        par = (np.cumsum(neg) + parity) % 2 == 1
        parity = int(par[-1])
        prev = blk[-1].copy()
        blk[par] *= -1.0
        del prv, neg, par
    with open(out_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        n = len(ts)
        for s in range(0, n, chunk):
            tsc = ts[s:s + chunk]
            t = (tsc - ts0).astype(np.float64) * 1e-9
            P = np.empty((len(t), 3))
            Q = np.empty((len(t), 4))
            for c in range(3):
                P[:, c] = np.interp(t, tv, pos[:, c])
            for c in range(4):
                Q[:, c] = np.interp(t, tv, quat[:, c])
            # dead reckoning after the last valid sample
            after = t > t_last
            if after.any():
                dtb = t[after] - t_last
                if tau > 0:
                    gain = tau * (1.0 - np.exp(-dtb / tau))
                else:
                    gain = np.zeros_like(dtb)
                P[after] = pos[-1].astype(np.float64) + gain[:, None] * v_end[None, :]
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            lines = []
            for i, tsi in enumerate(tsc.tolist()):
                p = P[i]
                q = Q[i]
                lines.append("%d.%09d %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (
                    tsi // 1000000000, tsi % 1000000000, p[0], p[1], p[2],
                    q[1], q[2], q[3], q[0]))
            f.write("".join(lines))


def run(inp, out, cfg=CFG, verbose=False):
    log = (lambda *a: print("[pathfinder]", *a, file=sys.stderr)) if verbose else None
    ts, raw = load_imu(inp)
    tv, pos, quat, ex = estimate(ts, raw, cfg, log)
    del raw
    write_tum(out, ts, tv, pos, quat, ex["v_end"], cfg)
    return len(ts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="zip / dataset folder / imu0 data.csv")
    ap.add_argument("-o", "--output", default="estimated_trajectory.txt")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a CFG parameter (development use)")
    args = ap.parse_args(argv)
    cfg = dict(CFG)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in cfg:
            ap.error("unknown parameter %s" % k)
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
    n = run(args.input, args.output, cfg, args.verbose)
    if args.verbose:
        print("[pathfinder] wrote %d poses to %s" % (n, args.output), file=sys.stderr)


if __name__ == "__main__":
    main()
