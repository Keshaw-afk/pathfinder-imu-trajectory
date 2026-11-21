# src — submission entrypoint

```bash
pip install numpy
python main.py <input> -o estimated_trajectory.txt
```

* `<input>`: the evaluation zip, a folder containing `imu0/data.csv`, or the
  CSV file itself (EuRoC IMU format, timestamps in ns).
* Output: TUM format `timestamp tx ty tz qx qy qz qw`, timestamps in seconds,
  exactly one pose per input row (duplicated rows and the failed final window
  included).
* Requirements: Python ≥ 3.8 and NumPy. Runs on a single CPU thread (BLAS
  thread pools are pinned to 1 inside the script) and needs no network or GPU.
* `-v` prints a short diagnostic log (detected column layout, rows rejected,
  initial bias) to stderr.

The method, its parameters and the evaluation results are documented in the
repository root `README.md`.
