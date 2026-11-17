#!/usr/bin/env python3
"""Report of a run (runs/<run>/node.npz): driving and timing; also written to runs/<run>/report.txt.

Driving: laps along the reference trajectory in the driving direction, |speed - reference speed|, |lateral offset| from
the reference line (mean and largest), stops at a wall (the car, moving at >= 0.2 m/s, stands still at the next
decision although the command asked for >= 0.3 m/s), decisions not driven by the controller (stale input, controller
error), takeovers by the safety driver and the time driven by the human, by nobody and by the policy. Timing: decision
period, age of the scan, odometry and pose at each decision, the controller's compute time.

    python3 -m safer_deploy.report runs/<run> [...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from safer_deploy import observation as ob                                          # noqa: E402
from safer_deploy.reference import ReferenceTrajectory                              # noqa: E402


def report(run_dir):
    meta = json.load(open(os.path.join(run_dir, 'meta.json')))
    path = os.path.join(run_dir, 'node.npz')
    if not os.path.exists(path):
        text = '%s: no decision logged' % run_dir
        print(text)
        return text
    d = np.load(path)
    ref = ReferenceTrajectory(os.path.join(HERE, 'data', '%s_reference.npz' % meta['config']['track']))
    direction = int(meta['direction'])
    P, speed, C, tr = d['pose'], d['speed'], d['command'], d['t_ros']
    idx = ob.nearest_index(ref, P[0, 0], P[0, 1])
    s_prev, progress, lat, vref = None, 0.0, [], []
    for x, y, _ in P:
        idx, s, la = ob.project(ref, x, y, idx)
        if s_prev is not None:
            progress += direction * ((s - s_prev + 0.5 * ref.length) % ref.length - 0.5 * ref.length)
        s_prev = s
        lat.append(la)
        vref.append((ref.v_ccw if direction > 0 else ref.v_cw)[idx])
    lat, vref = np.abs(np.asarray(lat)), np.asarray(vref)
    stops = int(np.sum((speed[1:] < 0.05) & (speed[:-1] >= 0.2) & (C[:-1, 0] >= 0.3)))
    status = d['status']
    bad = {s: int(np.sum(status == s)) for s in set(status.tolist()) if s != 'ok'}
    dt = np.diff(tr) if len(tr) > 1 else np.zeros(1)
    comp = d['compute_ms']
    ages = {k: tr - d[k + '_stamp'] for k in ('scan', 'odom', 'pose')}
    if 'driver' in d.files:
        drv = d['driver']
        takeovers = int(np.sum((drv[1:] == 'human') & (drv[:-1] != 'human')) + int(drv[0] == 'human'))
        period = float(np.median(dt)) if len(tr) > 1 else 0.1
        who = ('takeovers by the safety driver: %d (human driving %.1f s), not driven %.1f s, policy %.1f s'
               % (takeovers, period * np.sum(drv == 'human'), period * np.sum(drv == 'none'),
                  period * np.sum(drv == 'policy')))
    else:
        who = 'takeovers: not logged'
    lines = [
        '%s: %s (%s), direction %+d, %d decisions, %.1f s' % (os.path.basename(os.path.normpath(run_dir)),
                                                             meta['controller'], os.path.basename(meta['checkpoint']),
                                                             direction, len(tr), tr[-1] - tr[0]),
        '  driving: %.2f laps (%.1f m), |v - v_ref| %.3f m/s, |lateral| %.3f m (largest %.3f m), stops at a wall %d'
        % (progress / ref.length, progress, np.mean(np.abs(speed - vref)), lat.mean(), lat.max(), stops),
        '  decisions not driven by the controller: %s' % (bad if bad else 'none'),
        '  ' + who,
        '  timing: period %.3f s (largest %.3f), input age at a decision: scan %.0f ms, odometry %.0f ms, pose %.0f ms; '
        'compute %.1f ms (99th pct %.1f, largest %.1f)' % (dt.mean(), dt.max(), 1000 * ages['scan'].mean(),
                                                           1000 * ages['odom'].mean(), 1000 * ages['pose'].mean(),
                                                           comp.mean(), np.percentile(comp, 99), comp.max()),
    ]
    text = '\n'.join(lines)
    print(text)
    open(os.path.join(run_dir, 'report.txt'), 'w').write(text + '\n')
    return text


if __name__ == '__main__':
    for r in sys.argv[1:]:
        report(r)
