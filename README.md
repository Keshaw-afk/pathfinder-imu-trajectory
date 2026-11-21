# Path Finder — Trajectory Estimation Under Sensor Degradation

**XR Hackathon 2026 · Track 3** · 6-DoF trajectory reconstruction and 5-second
post-failure dead reckoning from degraded IMU telemetry only.

`src/main.py` reads a degraded EuRoC-format IMU stream and writes
`estimated_trajectory.txt` in TUM format: exactly one pose per input row,
using a single CPU thread, NumPy only, with a small heap footprint.

**Team Tri-Netra** · Thapar Institute of Engineering & Technology
Ashutosh Mehta · Keshaw Tiwari · Priyanshu Soni (Computer Engineering, 3rd year)

---

## Quick start

```bash
pip install -r requirements.txt            # numpy only
python src/main.py <input> -o estimated_trajectory.txt
```

`<input>` can be the evaluation **zip** (any member ending in `imu0/data.csv`),
a dataset **folder** containing `imu0/data.csv`, or the **CSV** itself.
Add `-v` for a short diagnostic log on stderr.

Docker:

```bash
docker build -t pathfinder .
docker run --rm -v "$PWD:/work" pathfinder /work/input.zip -o /work/estimated_trajectory.txt
```

## Input / output contract

| | Format |
|---|---|
| Input | `imu0/data.csv`: `#timestamp [ns], w_RS_S_{x,y,z} [rad/s], a_RS_S_{x,y,z} [m/s²]` |
| Output | `estimated_trajectory.txt`, space separated: `timestamp tx ty tz qx qy qz qw` |

* Timestamps are copied **exactly** from the input (ns → s with 9 decimals,
  formatted from the integer, so there is no float rounding).
* One output row per input row, **including duplicated rows** and the rows in
  the failed (blind) window. No rows are added or removed.
* Quaternions are unit-norm Hamilton quaternions, body → world, world z up.

## Method

```
 CSV ──► streaming parser ──► sanity checks ──► spike filter ──► static init
                              (layout, units,     (Hampel)        (gravity, g,
                               stuck / dup /                       gyro bias)
                               zero / gaps)                            │
                                                                       ▼
 TUM file ◄── interpolation onto ◄── dead reckoning ◄── drift-suppressed ◄── attitude filter
              every input stamp      (blind 5 s)       integration          (gyro + gravity,
                                                                              bias tracking)
```

1. **Streaming parse.** int64 nanosecond timestamps and float32 channels,
   parsed in blocks, so the raw text is never held in memory.
2. **Sensor sanity.**
   * The accelerometer is identified as the triplet whose magnitude is ≈ g, so
     the code is robust to a swapped column order and to data in g units.
   * Non-finite, all-zero (failed), stuck (exact repeat), duplicated and
     out-of-order rows are excluded from estimation.
   * Timestamp gaps (dropouts) are bridged at a constant rate.
3. **Spike rejection.** A windowed Hampel filter with a global noise floor
   replaces samples more than 8 robust σ from the local median.
4. **Static initialisation.** The undamaged lead-in gives the gravity
   direction (roll and pitch). Gyro bias and the local gravity magnitude come
   from the quietest 2 s window in the first 60 s, because EuRoC flights
   often start hand-carried. If no window is still enough, the thresholds
   are relaxed 3×.
5. **Attitude.** A Mahony-style complementary filter integrates the gyro on
   SO(3). The accelerometer pulls the gravity direction back slowly, with
   lower weight when |a| deviates from g. An integral term tracks gyro-bias
   drift.
   * **Jamming-aware gain.** Rotor vibration is anisotropic; injected
     jamming noise is isotropic. The noise level of the *quietest* gyro axis,
     per 2 s block, relative to early flight, detects gyro corruption. The
     correction gain grows with its square.
6. **Position.** The gravity-compensated world acceleration is integrated
   twice. Each integration is followed by a **zero-phase high-pass** (the
   signal minus its triangular moving average). This removes the quadratic
   drift that bias and tilt errors cause while keeping the motion an IMU can
   actually observe.
   * **Edge handling.** Near the ends of the recording the smoothing window is
     one-sided. A local-linear variant (`hp_edge="linear"`) removes the lag
     and is essential for long windows: MH_04 blind 5 s goes 11.4 → 6.0 m at
     30 s. At the short windows we ship, the plain one-sided average was
     better on 15 of 22 blind windows, so it is the default.
   * **Rotor-drag velocity** (`pos_mode="drag"`). On a multirotor, the
     specific force perpendicular to the thrust axis is ≈ −k·v. This
     drift-free velocity cue is scaled by regression and fused complementarily
     with the integrated velocity.
   * **Confidence weighting.** Velocity is scaled down when the jamming
     detector reports a corrupted gyro: holding position costs less ATE than
     moving in a wrong direction.
7. **Dead reckoning.** After the last valid sample, the orientation is held
   and the position continues along the recent smoothed velocity with
   exponential decay.
8. **Output.** Poses are interpolated (linear position, sign-consistent
   normalised quaternion) onto every input timestamp and streamed to disk.

## Efficiency

* Pure Python inner loop over native floats. Everything else is vectorised
  NumPy on float32 buffers processed in fixed-size chunks.
* Measured with `tracemalloc` (heap) and wall time; see the results below.
* BLAS/OpenMP thread pools are pinned to one thread.

## Results

Evaluated on **11 EuRoC MAV flights** (MH_01–05, V1_01–03, V2_01–03). Each
was degraded with the organisers' script (seed 42) in two variants: noise
applied as intended, and the script run as written on EuRoC column order,
which corrupts the gyro far more. That gives 22 runs, all scored with
Umeyama SE(3)-aligned ATE. **B0** is the do-nothing baseline: a constant
position.

| Mean over 22 runs | Ours | B0 | Change |
|---|---|---|---|
| ATE, whole trajectory | 3.23 m | 3.45 m | −6.4 % |
| ATE, blind last 5 s | 2.69 m | 3.09 m | −13.0 % |
| ATE, 1 s after dropouts | 3.30 m | 3.48 m | −5.3 % |

* 19 of 22 runs beat B0 on the rubric-weighted score
  (0.5·ATE + 0.33·blind + 0.17·dropout, each ÷ B0). Mean score 0.913; worst
  run 1.11 (V1_03, script-as-written variant).
* Settings were chosen on the 22-run mean, never on a single flight.
  Longer filter windows track more motion on some flights but are up to 5×
  worse on others.
* Efficiency on MH_04 (19 753 rows, 101 s): peak heap 3.0 MB (tracemalloc),
  5.2–5.8 ms per second of IMU, single thread.

IMU-only position under this level of corruption is weakly observable. The
method keeps only the motion it can trust, which is what the alignment-based
metric rewards.

## Reproducing the evaluation

```bash
pip install -r requirements-dev.txt
python tools/fetch_euroc.py MH_01_easy MH_04_difficult      # IMU + GT only
python tools/make_dataset.py data/raw/MH_01_easy --degrade /path/to/degrade_imu.py
python tools/evaluate.py --mem
```

`make_dataset.py` produces two variants of every sequence: `ordered` (the
organiser noise model applied as intended) and `literal` (the organiser script
run directly on the EuRoC column order, which puts accelerometer-grade noise on
the gyroscope). The estimator is tuned to handle both.

`evaluate.py` mirrors the official scoring: SE(3) Umeyama alignment (no
scale), then ATE RMSE overall, in the blind 5 s and within 1 s after
dropouts. It also reports rotation RMSE and 1 s RPE, and compares against the
trivial baseline of a constant position.

## Repository layout

```
src/main.py             submission entrypoint (NumPy + stdlib only)
tools/fetch_euroc.py    range-download IMU + ground truth from the EuRoC archives
tools/make_dataset.py   apply the organiser degradation (both column orders)
tools/evaluate.py       format checks, ATE / RPE, heap and runtime
Dockerfile              reproducible runtime
```

## Limitations

* Without any external aiding, absolute position is unobservable from an IMU
  alone. The estimator therefore favours bounded, low-drift motion estimates
  over open-loop integration.
* Yaw is unobservable from gravity, so heading drifts with the residual gyro
  bias.
