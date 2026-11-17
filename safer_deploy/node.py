#!/usr/bin/env python3
"""ROS 2 node that drives the car with SAFER, its BC prior or the PPO expert, and logs every decision.

Every `control.period` seconds the newest scan, odometry and pose become the policy's observation (observation.py, the
training code) and the controller acts; the action (alpha, delta) in [-1, 1] is published as
AckermannDriveStamped(speed = 2.0 (alpha + 1) / 2 capped at `control.speed_cap`, steering_angle = delta * 0.4189).
The car stops (speed 0, steering 0) when an input is older than `control.max_input_age` or the controller fails, and
when the node ends (Ctrl-C or --duration).

The run folder (--run_dir) receives node.npz (one row per decision: the 1080 ranges, pose, speed, yaw rate, the
observation, the action and the command, the stamps of the inputs, the controller's compute time and, for SAFER, its
16-step plan and its BC prior's chunk) and meta.json (settings, checkpoint, driving direction, notes). scripts/run.sh
runs this node together with a rosbag recording.

Takeovers (the `human` topics): every row also holds the command the car actually executes (the drive mux's output),
the joystick's driving command and its buttons and axes, and `driver`, who drove since the previous decision:
'policy' (the executed command is the one this node sent last), 'human' (another one: the safety driver), 'none'
(nothing executed since this node's last command, or for max_input_age: autonomous mode released, car coasting or
braking). Without an `executed` topic,
'human' means a recent joystick driving command.

    python3 -m safer_deploy.node --config config/real_car.yaml --controller ppo --run_dir runs/test
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rclpy                                                                         # noqa: E402
from ackermann_msgs.msg import AckermannDriveStamped                                 # noqa: E402
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped                 # noqa: E402
from nav_msgs.msg import Odometry                                                    # noqa: E402
from rclpy.node import Node                                                          # noqa: E402
from rclpy.parameter import Parameter                                                # noqa: E402
from rclpy.qos import qos_profile_sensor_data                                        # noqa: E402
from sensor_msgs.msg import Imu, Joy, LaserScan                                      # noqa: E402

from safer_deploy import observation as ob                                          # noqa: E402
from safer_deploy.reference import ReferenceTrajectory                              # noqa: E402

LIDAR_N, LIDAR_MIN, LIDAR_MAX = 1080, -2.35, 2.35
POSE_TYPES = {'odometry': Odometry, 'pose': PoseStamped, 'pose_cov': PoseWithCovarianceStamped}


def load_config(path):
    cfg = yaml.safe_load(open(path))
    cfg['_path'] = os.path.abspath(path)
    return cfg


def rel(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_s(stamp):
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def pose_of(msg):
    p = msg.pose.pose if isinstance(msg, (Odometry, PoseWithCovarianceStamped)) else msg.pose
    return p.position.x, p.position.y, yaw_of(p.orientation)


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


class Controller:
    """act(obs) -> (alpha, delta) in [-1, 1]; .info: the last decision's plan (SAFER)."""

    def __init__(self, kind, cfg):
        self.kind = kind
        self.info = {}
        if kind == 'ppo':
            from safer_deploy.ppo import PPOPolicy
            self.checkpoint = rel(cfg['ppo']['checkpoint'])
            self.policy = PPOPolicy(self.checkpoint)
        else:
            from safer_deploy.planner import SaferPlanner
            self.checkpoint = rel(cfg['safer']['checkpoint'])
            self.planner = SaferPlanner(self.checkpoint, cfg['safer']['device'], cfg['safer']['risk_weight'])
            for _ in range(int(cfg['control']['warmup_decisions'])):
                self.planner.act(np.zeros(144, np.float32), controller='safer' if kind == 'safer' else 'bc_prior')
            self.planner.reset()

    def act(self, obs):
        if self.kind == 'ppo':
            return self.policy.act(obs)
        a = self.planner.act(obs, controller='safer' if self.kind == 'safer' else 'bc_prior')
        self.info = dict(self.planner.last) if self.kind == 'safer' else {}
        return a


class DriveNode(Node):
    def __init__(self, a, cfg):
        sim = bool(cfg['control']['use_sim_time'])
        super().__init__('safer_driver', parameter_overrides=[Parameter('use_sim_time', value=sim)])
        self.a, self.cfg = a, cfg
        self.obs_cfg = ob.ObsConfig()
        self.ref = ReferenceTrajectory(rel('data/%s_reference.npz' % cfg['track']))
        self.controller = Controller(a.controller, cfg)
        d = cfg['direction']
        self.direction = None if str(d) == 'auto' else int(d)
        self.idx = None
        self.scan = self.odom = self.pose = self.imu = None
        self.executed = self.teleop = self.joy = None          # (ROS receive time, values)
        self.last_cmd, self.last_send = (0.0, 0.0), -np.inf
        self.last_action = np.zeros(2, np.float32)
        self.t0 = None
        self.done = False
        self.log = {k: [] for k in ('t_ros', 't_wall', 'scan_stamp', 'odom_stamp', 'pose_stamp', 'imu_stamp',
                                    'compute_ms', 'pose', 'speed', 'yaw_rate', 'ranges', 'obs', 'action', 'command',
                                    'plan', 'prior', 'status', 'driver', 'executed_cmd', 'executed_age', 'teleop_cmd',
                                    'teleop_age', 'joy_buttons', 'joy_axes')}
        t = cfg['topics']
        self.create_subscription(LaserScan, t['scan'], self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, t['odom'], self._on_odom, qos_profile_sensor_data)
        if t['pose'] != t['odom'] or t['pose_type'] != 'odometry':
            self.create_subscription(POSE_TYPES[t['pose_type']], t['pose'], self._on_pose, qos_profile_sensor_data)
        if t['yaw_rate'] == 'imu':
            self.create_subscription(Imu, t['imu'], self._on_imu, qos_profile_sensor_data)
        h = cfg.get('human') or {}
        if h.get('executed'):
            self.create_subscription(AckermannDriveStamped, h['executed'], self._on_executed, qos_profile_sensor_data)
        if h.get('teleop'):
            self.create_subscription(AckermannDriveStamped, h['teleop'], self._on_teleop, qos_profile_sensor_data)
        if h.get('joy'):
            self.create_subscription(Joy, h['joy'], self._on_joy, qos_profile_sensor_data)
        self.pub = self.create_publisher(AckermannDriveStamped, t['drive'], 10)
        self.create_timer(float(cfg['control']['period']), self._step)
        self.get_logger().info('controller %s (%s), track %s, run folder %s'
                               % (a.controller, self.controller.checkpoint, cfg['track'], a.run_dir))

    # -- inputs
    def _on_scan(self, m):
        if len(m.ranges) != LIDAR_N or abs(m.angle_min - LIDAR_MIN) > 1e-3 or abs(m.angle_max - LIDAR_MAX) > 1e-3:
            self.get_logger().error('scan of %d beams over [%.3f, %.3f] rad: the policy needs %d over [%.2f, %.2f]'
                                    % (len(m.ranges), m.angle_min, m.angle_max, LIDAR_N, LIDAR_MIN, LIDAR_MAX))
            self.done = True
            return
        self.scan = (stamp_s(m.header.stamp), np.asarray(m.ranges, np.float32))

    def _on_odom(self, m):
        self.odom = (stamp_s(m.header.stamp), float(m.twist.twist.linear.x), float(m.twist.twist.angular.z))
        t = self.cfg['topics']
        if t['pose'] == t['odom'] and t['pose_type'] == 'odometry':
            self.pose = (stamp_s(m.header.stamp), pose_of(m))

    def _on_pose(self, m):
        self.pose = (stamp_s(m.header.stamp), pose_of(m))

    def _on_imu(self, m):
        self.imu = (stamp_s(m.header.stamp), float(m.angular_velocity.z))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_executed(self, m):
        self.executed = (self._now(), (float(m.drive.speed), float(m.drive.steering_angle)))

    def _on_teleop(self, m):
        self.teleop = (self._now(), (float(m.drive.speed), float(m.drive.steering_angle)))

    def _on_joy(self, m):
        b = np.zeros(16, np.int8)
        b[:min(16, len(m.buttons))] = m.buttons[:16]
        x = np.zeros(8, np.float32)
        x[:min(8, len(m.axes))] = m.axes[:8]
        self.joy = (self._now(), (b, x))

    def _driver(self, now, age):
        """Who drove since the previous decision (see the module docstring)."""
        h = self.cfg.get('human') or {}
        recent = lambda v: v is not None and now - v[0] <= age          # noqa: E731
        if h.get('executed'):
            # nothing executed lately, or nothing since this node's last command: the mux forwards nobody's
            if not recent(self.executed) or self.executed[0] < self.last_send:
                return 'none'
            (sp, st), (lsp, lst) = self.executed[1], self.last_cmd
            return 'policy' if abs(sp - lsp) <= 0.05 and abs(st - lst) <= 0.01 else 'human'
        if h.get('teleop') and recent(self.teleop):
            return 'human'
        return 'policy'

    def send(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.pub.publish(msg)

    # -- one decision
    def _step(self):
        use_imu = self.cfg['topics']['yaw_rate'] == 'imu'
        if self.done or self.scan is None or self.odom is None or self.pose is None or (use_imu and self.imu is None):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        (ts, ranges), (to, speed, yr_odom), (tp, (x, y, yaw)) = self.scan, self.odom, self.pose
        ti, yaw_rate = (self.imu if use_imu else (to, yr_odom))
        if self.idx is None:
            self.idx = ob.nearest_index(self.ref, x, y)
            if self.direction is None:
                self.direction = 1 if math.cos(yaw - self.ref.yaw[self.idx]) > 0 else -1
            self.get_logger().info('driving direction %+d along the reference trajectory' % self.direction)
        self.idx = ob.project(self.ref, x, y, self.idx)[0]
        c = self.obs_cfg
        age = float(self.cfg['control']['max_input_age'])
        driver = self._driver(now, age)
        last = self.last_action
        if driver == 'human':          # the observation's last action is the executed one, here the safety driver's
            h = self.cfg.get('human') or {}
            src = self.executed if h.get('executed') else self.teleop
            if src is not None:
                last = np.clip([2.0 * src[1][0] / c.v_max - 1.0, src[1][1] / ob.DELTA_MAX], -1.0, 1.0).astype(np.float32)
        obs = ob.observation(ob.lidar_features(ranges, c), speed, yaw_rate, last,
                             ob.reference_features(self.ref, self.direction, x, y, yaw, self.idx, c), c)
        status, action = 'ok', np.zeros(2, np.float32)
        t_c = time.perf_counter()
        if max(now - ts, now - to, now - tp, now - ti) > age:
            status = 'stale input'
        else:
            try:
                action = np.clip(np.asarray(self.controller.act(obs), np.float32), -1.0, 1.0)
            except Exception as e:                          # noqa: BLE001 - any failure stops the car
                status = 'controller error: %s' % e
                self.get_logger().error(status)
        compute_ms = 1000.0 * (time.perf_counter() - t_c)
        if status == 'ok':
            cmd = (min(c.v_max * (float(action[0]) + 1.0) / 2.0, float(self.cfg['control']['speed_cap'])),
                   float(action[1]) * ob.DELTA_MAX)
            self.last_action = action
        else:
            cmd = (0.0, 0.0)
        self.send(*cmd)
        self.last_cmd, self.last_send = cmd, self._now()
        info = self.controller.info if status == 'ok' else {}
        nan2 = (float('nan'), float('nan'))
        ex, te, jo = self.executed, self.teleop, self.joy
        row = dict(t_ros=now, t_wall=time.time(), scan_stamp=ts, odom_stamp=to, pose_stamp=tp, imu_stamp=ti,
                   compute_ms=compute_ms, pose=(x, y, yaw), speed=speed, yaw_rate=yaw_rate,
                   ranges=ranges.astype(np.float16), obs=obs, action=action, command=cmd,
                   plan=info.get('plan', np.full((16, 2), np.nan, np.float32)),
                   prior=info.get('prior', np.full((16, 2), np.nan, np.float32)), status=status, driver=driver,
                   executed_cmd=ex[1] if ex else nan2, executed_age=now - ex[0] if ex else np.inf,
                   teleop_cmd=te[1] if te else nan2, teleop_age=now - te[0] if te else np.inf,
                   joy_buttons=jo[1][0] if jo else np.zeros(16, np.int8),
                   joy_axes=jo[1][1] if jo else np.zeros(8, np.float32))
        for k, v in row.items():
            self.log[k].append(v)
        if self.a.duration > 0 and now - self.t0 >= self.a.duration:
            self.done = True

    def save(self):
        os.makedirs(self.a.run_dir, exist_ok=True)
        meta = dict(controller=self.a.controller, checkpoint=self.controller.checkpoint,
                    checkpoint_sha256=sha256(self.controller.checkpoint), config=self.cfg,
                    direction=self.direction, decisions=len(self.log['t_ros']), notes=self.a.notes,
                    started=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.log['t_wall'][0]))
                    if self.log['t_wall'] else None)
        json.dump(meta, open(os.path.join(self.a.run_dir, 'meta.json'), 'w'), indent=1, default=str)
        if self.log['t_ros']:
            arrays = {k: np.asarray(v) for k, v in self.log.items() if k not in ('status', 'driver')}
            np.savez_compressed(os.path.join(self.a.run_dir, 'node.npz'), status=np.asarray(self.log['status']),
                                driver=np.asarray(self.log['driver']), **arrays)
        self.get_logger().info('wrote %s: %d decisions' % (self.a.run_dir, len(self.log['t_ros'])))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=os.path.join(HERE, 'config', 'real_car.yaml'))
    p.add_argument('--controller', choices=('safer', 'bc_prior', 'ppo'), required=True)
    p.add_argument('--run_dir', required=True)
    p.add_argument('--duration', type=float, default=0.0, help='stop after this many seconds (0: until Ctrl-C)')
    p.add_argument('--notes', default='', help='free text stored in meta.json')
    p.add_argument('--startup_timeout', type=float, default=60.0,
                   help='give up when the inputs have not all arrived after this many wall-clock seconds')
    a = p.parse_args()
    cfg = load_config(a.config)
    rclpy.init()
    node = DriveNode(a, cfg)
    t_start = time.time()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.t0 is None and time.time() - t_start > a.startup_timeout:
                node.get_logger().error('not all inputs arrived within %.0f s (scan %s, odom %s, pose %s): check the '
                                        'topics in %s' % (a.startup_timeout, node.scan is not None,
                                                          node.odom is not None, node.pose is not None, cfg['_path']))
                break
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            try:
                node.send(0.0, 0.0)
                time.sleep(0.02)
            except Exception:                               # noqa: BLE001
                pass
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
