#!/usr/bin/env python3
"""Grid-search CFG parameters with tools/evaluate.py.

    python tools/sweep.py --data data/degraded --filter MH \
        --grid vel_hp_s=4,8,16 --grid pos_hp_s=20,40 [--fixed pos_mode=drag]
"""
import argparse
import itertools
import os
import subprocess
import sys
import json
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "degraded"))
ap.add_argument("--filter", default="")
ap.add_argument("--grid", action="append", default=[])
ap.add_argument("--fixed", action="append", default=[])
ap.add_argument("--metric", default="ate_all")
a = ap.parse_args()
keys = [g.split("=")[0] for g in a.grid]
vals = [g.split("=")[1].split(",") for g in a.grid]
rows = []
for combo in itertools.product(*vals):
    sets = ["%s=%s" % kv for kv in zip(keys, combo)] + a.fixed
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        jp = tf.name
    cmd = [sys.executable, os.path.join(HERE, "evaluate.py"), "--data", a.data,
           "--filter", a.filter, "--quiet", "--json", jp]
    for s in sets:
        cmd += ["--set", s]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    r = json.load(open(jp))["results"]
    os.remove(jp)
    per = {k: v for k, v in r.items() if k != "_mean"}
    m = r["_mean"]
    rows.append((m[a.metric], combo, m, per))
    print("%-40s ATE %.3f  blind %.3f  drop %.3f | %s" % (
        " ".join(sets[:len(keys)]), m["ate_all"], m["ate_blind"], m["ate_dropout"],
        "  ".join("%s:%.2f" % (k.replace("_ordered", "/o").replace("_literal", "/l")[-14:], v["ate_all"])
                   for k, v in per.items())), flush=True)
rows.sort(key=lambda x: x[0])
print("best:", " ".join("%s=%s" % kv for kv in zip(keys, rows[0][1])), "-> %.3f" % rows[0][0])
