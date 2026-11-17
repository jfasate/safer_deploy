"""The policy's observation (definitions in env.py's docstring), shared by the training environment and the ROS node.

Pure functions of the sensor values, the car's pose and the reference trajectory: env.py applies its randomisation
(LiDAR dropout, speed and yaw-rate noise, localisation error) to their inputs, the ROS node (node.py) calls them
with the measured values, so that both build the observation with the same code.
"""
from __future__ import annotations

import math

import numpy as np

LIDAR_RANGE_MIN = 0.10          # the LiDAR's minimum range [m] (mixed_quality_f1tenth/twin.py)


def lidar_features(ranges, cfg) -> np.ndarray:
    """The 1080 ranges [m] (inf or nan: no return) -> 108 values: invalid returns and returns at or below the
    sensor's minimum range set to cfg.lidar_cap, clipped to [0, cap], min-pooled in groups of cfg.lidar_pool, / cap."""
    r = np.array(ranges, dtype=np.float64)
    r[~np.isfinite(r) | (r <= LIDAR_RANGE_MIN)] = cfg.lidar_cap
    return np.clip(r, 0.0, cfg.lidar_cap).reshape(-1, cfg.lidar_pool).min(axis=1) / cfg.lidar_cap


def project(track, x: float, y: float, hint: int, window: int = 40) -> tuple[int, float, float]:
    """Nearest trajectory sample within +-window samples of `hint` (so the projection never jumps to a neighbouring
    lane behind a wall): its index, the continuous arc length [m] and the signed lateral offset [m] (positive to the
    left of the trajectory's counter-clockwise order)."""
    idx = (hint + np.arange(-window, window + 1)) % track.n
    d2 = (track.xy[idx, 0] - x) ** 2 + (track.xy[idx, 1] - y) ** 2
    i = int(idx[np.argmin(d2)])
    dx, dy = x - track.xy[i, 0], y - track.xy[i, 1]
    c, s = math.cos(track.yaw[i]), math.sin(track.yaw[i])
    return i, (i * track.step + dx * c + dy * s) % track.length, -dx * s + dy * c


def nearest_index(track, x: float, y: float) -> int:
    """The trajectory sample nearest to (x, y) over the whole lap (a first projection hint)."""
    return int(np.argmin((track.xy[:, 0] - x) ** 2 + (track.xy[:, 1] - y) ** 2))


def reference_features(track, direction: int, x: float, y: float, yaw: float, hint: int, cfg) -> np.ndarray:
    """32 values from the pose: lateral offset / lat_scale (positive to the left of the driving direction), heading
    error / pi, and the trajectory points cfg.preview metres ahead in the driving direction, each (x / preview_scale,
    y / preview_scale, reference speed / v_max) in the car's frame (x forward, y left)."""
    i, _, lat = project(track, x, y, hint)
    lat *= direction
    ref_yaw = track.yaw[i] + (0.0 if direction > 0 else math.pi)
    head = (yaw - ref_yaw + math.pi) % (2 * math.pi) - math.pi
    k = (i + direction * np.round(np.asarray(cfg.preview) / track.step).astype(int)) % track.n
    dx, dy = track.xy[k, 0] - x, track.xy[k, 1] - y
    cy, sy = math.cos(yaw), math.sin(yaw)
    px, py = cy * dx + sy * dy, -sy * dx + cy * dy
    v = (track.v_ccw if direction > 0 else track.v_cw)[k]
    pts = np.stack([px / cfg.preview_scale, py / cfg.preview_scale, v / cfg.v_max], 1).reshape(-1)
    return np.concatenate([[lat / cfg.lat_scale, head / math.pi], pts])


def observation(lidar: np.ndarray, vx: float, yaw_rate: float, last_action, reference: np.ndarray, cfg) -> np.ndarray:
    """The 144-value observation: LiDAR (108), speed / v_max and yaw rate / yaw_rate_scale, the last action (2), the
    reference features (32), clipped to [-1, 1]."""
    ego = [vx / cfg.v_max, yaw_rate / cfg.yaw_rate_scale]
    obs = np.concatenate([lidar, ego, last_action, reference])
    return np.clip(obs, -1.0, 1.0).astype(np.float32)


class ObsConfig:
    """The observation's constants: the training environment's (mixed_quality_f1tenth/env.EnvConfig)."""
    lidar_cap = 10.0                                   # [m]
    lidar_pool = 10                                    # 1080 beams -> 108
    v_max = 2.0                                        # [m/s], also the speed at alpha = +1
    yaw_rate_scale = 4.0                               # [rad/s]
    lat_scale = 1.5                                    # [m]
    preview = tuple(0.5 * (k + 1) for k in range(10))  # [m] ahead along the reference trajectory
    preview_scale = 5.0                                # [m]


DELTA_MAX = 0.4189                                     # steering angle at delta = +-1 [rad] (the car's limit)
