#!/usr/bin/env python3
"""Pre-flight check: does the car match what the policies were trained for? Run with the car standing on the track.

Listens to the configured topics for a few seconds and checks
  1. topics       every input arrives, and how often (scan >= 10 Hz, odometry and pose >= 20 Hz)
  2. LiDAR        1080 beams over -2.35..2.35 rad, ranges in metres
  3. map frame    the scan, placed at the localisation's pose (LiDAR lidar_mount_x ahead of the pose), lies on the walls
                  of data/<track>.png: median distance of the returns within 10 m to the nearest wall cell. If it does
                  not, a search over shifts (+-1 m) and rotations (+-10 deg) reports the offset that would align them.
  4. reference    the pose's lateral offset from the reference trajectory and heading error; the driving direction
  5. drive        somebody subscribes to the drive topic
  6. compute      decision time of the PPO expert and of SAFER on this machine (SAFER must stay well below 100 ms)
and writes the report to runs/check_setup_<time>.txt (send it to us with the runs).

    python3 -m safer_deploy.check_setup --config config/real_car.yaml [--listen 5] [--no_safer]
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rclpy                                                                         # noqa: E402
from ackermann_msgs.msg import AckermannDriveStamped                                 # noqa: E402
from nav_msgs.msg import Odometry                                                    # noqa: E402
from rclpy.node import Node                                                          # noqa: E402
from rclpy.parameter import Parameter                                                # noqa: E402
from rclpy.qos import qos_profile_sensor_data                                        # noqa: E402
from sensor_msgs.msg import Imu, LaserScan                                           # noqa: E402

from safer_deploy import observation as ob                                          # noqa: E402
from safer_deploy.node import LIDAR_MAX, LIDAR_MIN, LIDAR_N, POSE_TYPES, load_config, pose_of, rel, stamp_s  # noqa: E402
from safer_deploy.reference import ReferenceTrajectory                              # noqa: E402


class Listener(Node):
    def __init__(self, cfg):
        super().__init__('safer_check_setup',
                         parameter_overrides=[Parameter('use_sim_time', value=bool(cfg['control']['use_sim_time']))])
        t = cfg['topics']
        self.msgs = {k: [] for k in ('scan', 'odom', 'pose', 'imu')}
        self.create_subscription(LaserScan, t['scan'], lambda m: self.msgs['scan'].append(m), qos_profile_sensor_data)
        self.create_subscription(Odometry, t['odom'], lambda m: self.msgs['odom'].append(m), qos_profile_sensor_data)
        if not (t['pose'] == t['odom'] and t['pose_type'] == 'odometry'):
            self.create_subscription(POSE_TYPES[t['pose_type']], t['pose'], lambda m: self.msgs['pose'].append(m),
                                     qos_profile_sensor_data)
        if t['yaw_rate'] == 'imu':
            self.create_subscription(Imu, t['imu'], lambda m: self.msgs['imu'].append(m), qos_profile_sensor_data)
        self.drive = self.create_publisher(AckermannDriveStamped, t['drive'], 10)


def load_map(cfg):
    """Occupied cells of data/<track>.png (ROS map_server convention) and their distance transform [m]."""
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    import yaml
    meta = yaml.safe_load(open(rel('data/%s.yaml' % cfg['track'])))
    img = np.asarray(Image.open(rel('data/' + meta['image'])).convert('L'), dtype=np.float64)
    p = img / 255.0 if meta.get('negate', 0) else (255.0 - img) / 255.0
    occupied = p > float(meta['occupied_thresh'])
    dist = distance_transform_edt(~occupied) * float(meta['resolution'])
    return dist, float(meta['resolution']), meta['origin']


def wall_distance(points, dist, res, origin):
    """Distance [m] from map points (N,2) to the nearest occupied cell (outside the image: 1 m)."""
    h, w = dist.shape
    col = np.floor((points[:, 0] - origin[0]) / res).astype(int)
    row = h - 1 - np.floor((points[:, 1] - origin[1]) / res).astype(int)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    out = np.ones(len(points))
    out[inside] = dist[row[inside], col[inside]]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=os.path.join(HERE, 'config', 'real_car.yaml'))
    p.add_argument('--listen', type=float, default=5.0, help='seconds of listening (wall clock)')
    p.add_argument('--no_safer', action='store_true', help='skip the SAFER timing (e.g. on a machine without a GPU)')
    a = p.parse_args()
    cfg = load_config(a.config)
    t = cfg['topics']
    lines, results = [], []

    def say(s=''):
        print(s, flush=True)
        lines.append(s)

    def check(name, status, detail):
        results.append((name, status))
        say('%-5s %-28s %s' % (status, name, detail))

    rclpy.init()
    node = Listener(cfg)
    t_end = time.time() + a.listen
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.05)
    now = node.get_clock().now().nanoseconds * 1e-9
    say('safer_deploy check_setup, %s, config %s, listened %.1f s' % (time.strftime('%Y-%m-%d %H:%M:%S'), cfg['_path'],
                                                                      a.listen))
    m = node.msgs
    if not m['pose']:
        m['pose'] = m['odom'] if (t['pose'] == t['odom'] and t['pose_type'] == 'odometry') else []
    # 1. topics
    need = [('scan', t['scan'], 10.0), ('odom', t['odom'], 20.0), ('pose', t['pose'], 20.0)]
    if t['yaw_rate'] == 'imu':
        need.append(('imu', t['imu'], 20.0))
    for key, topic, min_rate in need:
        n = len(m[key])
        if n == 0:
            check('topic ' + key, 'FAIL', '%s: no message' % topic)
            continue
        rate = n / a.listen
        age = now - stamp_s(m[key][-1].header.stamp)
        check('topic ' + key, 'PASS' if rate >= min_rate and age < 0.5 else 'WARN',
              '%s: %.1f Hz (need >= %.0f), newest %.0f ms old' % (topic, rate, min_rate, 1000 * age))
    if not (m['scan'] and m['odom'] and m['pose']):
        say('\ninputs missing: fix the topics in %s first' % cfg['_path'])
        return finish(lines, results)
    scan, odom = m['scan'][-1], m['odom'][-1]
    x, y, yaw = pose_of(m['pose'][-1])
    # 2. LiDAR
    r = np.asarray(scan.ranges, np.float64)
    ok = (len(r) == LIDAR_N and abs(scan.angle_min - LIDAR_MIN) < 1e-3 and abs(scan.angle_max - LIDAR_MAX) < 1e-3)
    valid = np.isfinite(r) & (r > 0.1) & (r < 30.0)
    check('LiDAR', 'PASS' if ok else 'FAIL', '%d beams over [%.3f, %.3f] rad (need %d over [%.2f, %.2f]), %.0f %% valid '
          'returns, median %.2f m' % (len(r), scan.angle_min, scan.angle_max, LIDAR_N, LIDAR_MIN, LIDAR_MAX,
                                      100 * valid.mean(), np.median(r[valid]) if valid.any() else float('nan')))
    # 3. map frame
    dist, res, origin = load_map(cfg)
    ang = scan.angle_min + scan.angle_increment * np.arange(len(r))
    use = valid & (r < 10.0)
    mx = float(cfg['lidar_mount_x'])

    def endpoints(dx=0.0, dy=0.0, dyaw=0.0):
        X, Y, Th = x + dx, y + dy, yaw + dyaw
        lx, ly = X + mx * math.cos(Th), Y + mx * math.sin(Th)
        return np.stack([lx + r[use] * np.cos(Th + ang[use]), ly + r[use] * np.sin(Th + ang[use])], 1)

    d0 = wall_distance(endpoints(), dist, res, origin)
    med = float(np.median(d0))
    status = 'PASS' if med <= 0.08 and (d0 < 0.10).mean() >= 0.8 else ('WARN' if med <= 0.20 else 'FAIL')
    check('map frame', status, 'scan returns lie %.3f m (median) from the walls of data/%s.png, %.0f %% within 0.10 m'
          % (med, cfg['track'], 100 * (d0 < 0.10).mean()))
    if status != 'PASS':
        best = (med, 0.0, 0.0, 0.0)
        for dyaw in np.radians(np.arange(-10, 10.5, 1.0)):
            for dx in np.arange(-1.0, 1.0001, 0.05):
                for dy in np.arange(-1.0, 1.0001, 0.05):
                    mm = float(np.median(wall_distance(endpoints(dx, dy, dyaw), dist, res, origin)))
                    if mm < best[0]:
                        best = (mm, dx, dy, dyaw)
        say('      best alignment within +-1 m, +-10 deg: shift (%.2f, %.2f) m, rotation %.1f deg -> median %.3f m. '
            'If this is small, the pose is in another frame: tell us.' % (best[1], best[2], math.degrees(best[3]), best[0]))
    # 4. reference
    ref = ReferenceTrajectory(rel('data/%s_reference.npz' % cfg['track']))
    i = ob.nearest_index(ref, x, y)
    i, _, lat = ob.project(ref, x, y, i)
    d = cfg['direction']
    direction = (1 if math.cos(yaw - ref.yaw[i]) > 0 else -1) if str(d) == 'auto' else int(d)
    head = (yaw - (ref.yaw[i] + (0.0 if direction > 0 else math.pi)) + math.pi) % (2 * math.pi) - math.pi
    half = 0.5 * float(ref.width[i])
    status = 'PASS' if abs(lat) < half - 0.2 and abs(math.degrees(head)) < 30 else 'WARN'
    check('reference', status, 'pose (%.2f, %.2f, %.0f deg): %.2f m from the reference line (lane half-width %.2f m), '
          'heading error %.0f deg, driving direction %+d' % (x, y, math.degrees(yaw), direction * lat, half,
                                                              math.degrees(head), direction))
    # speed / yaw rate
    say('      speed %.2f m/s and yaw rate %.2f rad/s from %s (should be ~0 while standing)'
        % (odom.twist.twist.linear.x, odom.twist.twist.angular.z, t['odom']))
    # 5. drive
    n_sub = node.count_subscribers(t['drive'])
    check('drive', 'PASS' if n_sub > 0 else 'WARN', '%s: %d subscriber(s)' % (t['drive'], n_sub))
    # 6. compute
    c = ob.ObsConfig()
    obs = ob.observation(ob.lidar_features(r.astype(np.float32), c), odom.twist.twist.linear.x,
                         odom.twist.twist.angular.z, np.zeros(2, np.float32),
                         ob.reference_features(ref, direction, x, y, yaw, i, c), c)
    from safer_deploy.ppo import PPOPolicy
    ppo = PPOPolicy(rel(cfg['ppo']['checkpoint']))
    t0 = time.perf_counter()
    for _ in range(100):
        a_ppo = ppo.act(obs)
    check('compute PPO', 'PASS', '%.2f ms per decision; action now (%.2f, %.2f) = %.2f m/s, %.3f rad'
          % (10 * (time.perf_counter() - t0), a_ppo[0], a_ppo[1], c.v_max * (a_ppo[0] + 1) / 2, a_ppo[1] * ob.DELTA_MAX))
    if not a.no_safer:
        from safer_deploy.planner import SaferPlanner
        planner = SaferPlanner(rel(cfg['safer']['checkpoint']), cfg['safer']['device'], cfg['safer']['risk_weight'])
        for _ in range(3):
            planner.act(obs)
        planner.reset()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            a_s = planner.act(obs)
            times.append(1000 * (time.perf_counter() - t0))
        mean = float(np.mean(times))
        check('compute SAFER', 'PASS' if mean <= 60 else ('WARN' if mean <= 90 else 'FAIL'),
              '%.1f ms per decision (largest %.1f; must stay well below 100) on %s; action now (%.2f, %.2f)'
              % (mean, max(times), cfg['safer']['device'], a_s[0], a_s[1]))
    node.destroy_node()
    rclpy.shutdown()
    return finish(lines, results)


def finish(lines, results):
    worst = 'FAIL' if any(s == 'FAIL' for _, s in results) else ('WARN' if any(s == 'WARN' for _, s in results) else 'PASS')
    lines.append('\noverall: %s' % worst)
    print(lines[-1])
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    path = os.path.join(HERE, 'runs', 'check_setup_%s.txt' % time.strftime('%Y%m%d_%H%M%S'))
    open(path, 'w').write('\n'.join(lines) + '\n')
    print('written %s' % path)
    return 0 if worst != 'FAIL' else 1


if __name__ == '__main__':
    sys.exit(main())
