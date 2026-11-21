#!/usr/bin/env python3
"""
Build the Track 3 submission bundle (slide 17 layout):

    <team>.zip
    ├── estimated_trajectory.txt
    ├── metadata.json            runtime / peak heap / CPU info
    └── src/
        ├── main.py
        └── README.md

    python tools/package_submission.py --team CollegeName_TeamName \
        --input data/degraded/MH_01_easy_ordered.zip
"""

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="CollegeName_TeamName")
    ap.add_argument("--input", required=True, help="sample degraded input (zip/folder/csv)")
    ap.add_argument("--out", default=os.path.join(ROOT, "submission"))
    args = ap.parse_args()

    stage = os.path.join(args.out, args.team)
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(os.path.join(stage, "src"))
    shutil.copy(os.path.join(ROOT, "src", "main.py"), os.path.join(stage, "src", "main.py"))
    shutil.copy(os.path.join(ROOT, "src", "README.md"), os.path.join(stage, "src", "README.md"))
    traj = os.path.join(stage, "estimated_trajectory.txt")

    # runtime: fresh interpreter, exactly as the evaluator would run it
    t0 = time.perf_counter()
    subprocess.run([sys.executable, os.path.join(stage, "src", "main.py"), args.input, "-o", traj],
                   check=True)
    wall = time.perf_counter() - t0

    # algorithm-level heap peak (tracemalloc started after imports)
    spec = importlib.util.spec_from_file_location("pf_main", os.path.join(ROOT, "src", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    t1 = time.perf_counter()
    mod.run(args.input, traj + ".tmp")
    algo = time.perf_counter() - t1
    tracemalloc.start()
    mod.run(args.input, traj + ".tmp")
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    os.remove(traj + ".tmp")

    first = last = None
    rows = 0
    with open(traj) as f:
        for line in f:
            if line.startswith("#"):
                continue
            ts = float(line.split()[0])
            first = ts if first is None else first
            last = ts
            rows += 1
    dur = last - first
    meta = {
        "team": args.team,
        "track": 3,
        "input": os.path.basename(args.input),
        "poses": rows,
        "imu_duration_s": round(dur, 3),
        "runtime_total_s": round(wall, 4),
        "runtime_algorithm_s": round(algo, 4),
        "ms_per_imu_second": round(1000 * algo / dur, 3),
        "peak_heap_mb_tracemalloc": round(peak / 2**20, 3),
        "threads": 1,
        "cpu": cpu_name(),
        "python": platform.python_version(),
        "numpy": mod.np.__version__,
        "os": platform.platform(),
    }
    with open(os.path.join(stage, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    zpath = os.path.join(args.out, args.team + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in ("estimated_trajectory.txt", "metadata.json", "src/main.py", "src/README.md"):
            z.write(os.path.join(stage, rel), rel)
    print(json.dumps(meta, indent=2))
    print("bundle:", zpath)


if __name__ == "__main__":
    main()
