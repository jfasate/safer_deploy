"""The track's reference trajectory (its centre line, sampled every ~0.1 m, with a reference speed per point and driving
direction), as the training environment uses it (mixed_quality_f1tenth/env.Track), in the map frame of
data/<track>.yaml."""
from __future__ import annotations

import json

import numpy as np


class ReferenceTrajectory:
    def __init__(self, path: str):
        d = np.load(path)
        self.name = str(d["name"])
        self.xy, self.yaw, self.kappa, self.width = d["xy"], d["yaw"], d["kappa"], d["width"]
        self.v_ccw, self.v_cw = d["v_ccw"], d["v_cw"]       # direction +1 (index order) and -1
        self.n = int(len(self.xy))
        self.length = float(d["length"])
        self.step = self.length / self.n
        self.map = json.loads(str(d["map"]))                # the map the trajectory is in: image, resolution, origin
