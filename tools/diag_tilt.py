#!/usr/bin/env python3
"""Diagnostic: tilt (roll/pitch) error of an estimate vs ground truth,
independent of yaw and of the trajectory alignment."""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from evaluate import read_tum, associate, quat_to_R

est_ts, est = read_tum(sys.argv[1])
gt_ts, gt = read_tum(sys.argv[2])
ok, jj = associate(est_ts, gt_ts)
Re, Rg = quat_to_R(est[ok, 3:7]), quat_to_R(gt[jj[ok], 3:7])
# gravity direction expressed in body frame: R^T e_z
ze, zg = Re[:, 2, :], Rg[:, 2, :]
ang = np.degrees(np.arccos(np.clip(np.sum(ze * zg, 1), -1, 1)))
t = (est_ts[ok] - est_ts[0]) * 1e-9
for q in (0.1, 0.25, 0.5, 0.75, 0.95):
    i = int(q * (len(t) - 1))
    print("t=%6.1fs tilt err %.2f deg" % (t[i], ang[i]))
print("tilt RMSE %.2f deg" % np.sqrt(np.mean(ang ** 2)))
# full attitude error relative to the first pose (removes the unknown yaw offset)
D0 = Rg[0] @ Re[0].T
rel = np.einsum("nji,njk->nik", Rg, np.einsum("ij,njk->nik", D0, Re))
full = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
print("full attitude err (start-anchored): median %.2f  end %.2f deg" % (np.median(full), full[-1]))
