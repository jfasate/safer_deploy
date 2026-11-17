# AI_SETUP.md: setting up this folder on the car (for an AI coding assistant)

Jayesh's steps are in README.md. This file has everything needed to install, configure, check and run the
package on his machine; do the setup, run `check_setup.py`, and report to him what is not PASS.

This folder runs two controllers on your F1TENTH car on **real_track**:

- **SAFER**: our planner. Every 0.1 s a learned world model imagines 270 candidate 1.6-second action sequences
  (two MPPI iterations around the previous plan and a learned prior). Each one is scored by predicted progress, lateral
  error and steering smoothness, the value of where it ends, minus a learned wall-contact risk. The best plan's
  first action is sent.
- **PPO**: the expert SAFER learned from. It is a small neural network that maps the observation directly to an action.

Both were trained in a replica of your `f1tenth_gym_gazebo` simulator. The replica uses the same vehicle model and
parameters as `dynamics_plant.py`, the LiDAR of your URDF, and the real_track map generated from the image you sent us.
They were then checked in your Gazebo simulation, where both drive real_track without touching a wall (results below).

Both controllers read the 2D LiDAR, the wheel-odometry speed, the yaw rate and the localisation pose. They drive laps
along the track's centre line at the reference speed of **1.0 m/s**, and publish `ackermann_msgs/AckermannDriveStamped`
on `/drive`.

`videos/` shows SAFER in simulation (2x speed):
- **Right:** a view around the car, heading up. Every candidate plan is coloured by its predicted reward per step
  (yellow = good); white is the plan SAFER executes; green dots are the LiDAR returns; blue dots are the reference
  points the policy sees.
- **Left:** the whole track.
- **Bottom:** the predicted wall-contact risk and the speed.

---

## Quick start

```bash
# 0. once: dependencies (see "Setup"), then edit the topics in config/real_car.yaml
# 1. car standing on the track, localisation running:
python3 -m safer_deploy.check_setup --config config/real_car.yaml      # all PASS? if not, see "Assumptions"
# 2. one run (Ctrl-C or the dead-man switch ends it; the car stops; a report is printed):
bash scripts/run.sh ppo   ppo_dir+1_a
bash scripts/run.sh safer safer_dir+1_a
# 3. after the session: fill RUN_SHEET.csv, then
bash scripts/pack_runs.sh test1                                         # -> ../safer_runs_test1.tar.gz, send us the link
```

---

## The plan: two field tests

| | Test 1 (now) | Test 2 (after we fine-tune) |
|---|---|---|
| controllers | PPO, then SAFER (offline, `checkpoints/safer_offline.pt`) | SAFER fine-tuned on Test 1's data (`checkpoints/safer_finetuned.pt`, we send it), PPO again if there is time |
| goal | (a) the whole chain works on the real car: sensors, localisation, latency, commands; (b) how PPO and SAFER drive; (c) **15-30 minutes of SAFER driving** for fine-tuning | the demo: fine-tuned SAFER, 5-10 laps, filmed |
| what we do afterwards | turn your runs into training data and fine-tune SAFER's dynamics and reward models on them (the prior, value and risk models stay frozen), check it in simulation, send the new checkpoint | analyse; done |

The code is the same for both tests. Test 2 only changes `safer.checkpoint` in `config/real_car.yaml`.

---

## Assumptions: please confirm before driving

On paper your car and your simulator agree, so these should all hold. If **any** of them differs, or you are not sure,
**tell Yansong before driving**. Most mismatches are cheap to fix on our side if we know early (new topic names, a
different map frame). Some need retraining (track scale, LiDAR geometry, a different reference speed).

| # | we assume | checked by `check_setup.py` | if it differs |
|---|---|---|---|
| 1 | **Track**: the real track is `data/real_track.png` at **0.05 m per pixel**: centre-line lap 155.4 m, tightest bend radius 0.83 m, lane 1.6-6.6 m wide (median 3.0 m) | map frame (3) | tell us: scale or layout changes need retraining |
| 2 | **Localisation**: a pose (x, y, yaw) of `base_link` in **the map frame of `data/real_track.yaml`** (origin [-30.95, -20.325], 0.05 m/px), e.g. your particle filter on that map, at >= 20 Hz, error within ~5 cm / 2 deg | map frame (3), topics (1) | send us your map (png + yaml) and the pose topic: a different frame we can transform, a different map we must check |
| 3 | **LiDAR**: 2D, **1080 beams over -2.35..+2.35 rad** (270 deg, counter-clockwise, beam 0 on the right), ranges in metres (inf or 0 for no return), mounted **0.275 m ahead of `base_link`**, level, facing forward, >= 10 Hz (your sim: 20 Hz) | LiDAR (2), map frame (3) | tell us: other beam counts, angles or mounting need retraining or a conversion |
| 4 | **Speed and yaw rate**: forward speed [m/s] from the wheel odometry (`twist.linear.x`), yaw rate [rad/s, positive = turning left] from the odometry or the IMU (`topics.yaw_rate`) | topics (1) | set the topics; tell us if the yaw rate is not measured |
| 5 | **Commands**: `AckermannDriveStamped`, `drive.speed` in m/s (the VESC speed controller tracks it), `drive.steering_angle` in rad (positive = left), steering limit **0.4189 rad**; a new command every 0.1 s is fine for your driver/mux | drive (5) | tell us your limits |
| 6 | **Dynamics and latency**: the car reaches 1 m/s within ~0.3 s and steers at ~3 rad/s; from the LiDAR measurement to the wheels takes <= ~100 ms (the policies were trained with 0-80 ms extra command delay and +-10-25 % friction, mass and motor variation) | report timing | tell us what you measured |
| 7 | **Speed**: **1.0 m/s** on real_track (your MPC's speed for this map). `control.speed_cap` (1.5 m/s) only guards. Capping below ~1.1 m/s changes the policies' behaviour | - | tell us if you want slower driving: we retrain for it |
| 8 | **Direction**: both directions are trained; the direction is taken from the car's heading at the start (`direction: auto`) | reference (4) | - |
| 9 | **Compute**: SAFER needs a **CUDA GPU** (40-50 ms per decision on our Quadro RTX 6000; on CPU 550 ms, too slow). PPO runs anywhere (< 1 ms) | compute (6) | if the car's computer is too slow, run the node on a laptop with a GPU on the car's ROS 2 network, or run PPO only |
| 10 | **Software**: ROS 2 Humble, Python 3.10 | - | tell us your ROS version |

---

## Setup

1. **Copy** this folder to the computer that will run the node: the car's computer if it has a CUDA GPU, otherwise a
   laptop with a GPU on the car's ROS 2 network (same `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY` unset).

   Workstation setup (the node not on the car):
   - The node judges the inputs' age (`control.max_input_age`) from their header stamps, which come from the car's
     clock. Synchronise the two clocks (chrony or NTP), or change `node.py`'s staleness check to the time a message is
     received. Keep the header stamps in the log: they measure the latency.
   - The WiFi adds delay. `report.txt` shows the input ages at each decision; the policies tolerate up to ~100 ms from
     measurement to command.
   - If the car's setup needs another arrangement altogether, adapt the code: `node.py` is the only ROS-specific
     part.
2. **Install**:
   ```bash
   sudo apt install ros-humble-ackermann-msgs ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins
   pip install -r requirements.txt          # torch: the CUDA build matching your GPU (Jetson: NVIDIA's wheel)
   source /opt/ros/humble/setup.bash        # in every terminal that runs the node
   ```
   If ROS uses another Python than the one with torch, set `PYTHON=/path/to/python` for `scripts/run.sh`. It has to
   import `rclpy` (ROS sourced) and `torch`.
3. **Configure** `config/real_car.yaml`. Normally only the `topics` section changes:
   - `scan`: your LaserScan topic.
   - `odom`: the wheel odometry.
   - `pose` and `pose_type`: the localisation (`odometry`, `pose` or `pose_cov` message).
   - `yaw_rate`: `odom` or `imu`.
   - `drive`: the input of your drive mux or VESC driver.

   Also set the `human` section, so that takeovers by the safety driver are recorded:
   - `executed`: the drive mux's output, i.e. the command the car actually executes, unscaled. The f1tenth_system
     stack uses `/ackermann_cmd`.
   - `teleop`: the joystick's driving commands (`/teleop`).
   - `joy`: the joystick (`/joy`).
   - `record`: the VESC topics to put in the rosbag.

   Every decision then holds who drove since the previous one: `policy`, `human` or `none` (nothing executed, e.g.
   autonomous mode released). The report counts the takeovers. If the stack has no mux output topic, leave `executed`
   empty: `human` then means a recent joystick command. Tested in Gazebo with a stand-in mux: one 5 s takeover and one
   3 s release were labelled exactly.
4. **Check**: put the car on the track, start the drivers and the localisation, and run
   `python3 -m safer_deploy.check_setup --config config/real_car.yaml`. Expected (from your Gazebo simulation):
   ```
   PASS  topic scan          /scan: 16.2 Hz (need >= 10), newest 38 ms old
   PASS  topic odom          /ego_racecar/odom: 80.0 Hz (need >= 20), newest 0 ms old
   PASS  topic pose          /ego_racecar/odom: 80.0 Hz (need >= 20), newest 0 ms old
   PASS  LiDAR               1080 beams over [-2.350, 2.350] rad (need 1080 over [-2.35, 2.35]), 100 % valid returns
   PASS  map frame           scan returns lie 0.050 m (median) from the walls of data/real_track.png, 93 % within 0.10 m
   PASS  reference           pose (-8.11, 4.05, -111 deg): -0.00 m from the reference line, heading error 0 deg, direction +1
   PASS  drive               /drive: 1 subscriber(s)
   PASS  compute PPO         0.04 ms per decision
   PASS  compute SAFER       40.4 ms per decision (largest 48.5; must stay well below 100) on cuda:0
   ```
   **map frame** is the important line. It places the scan at the localisation's pose on our map:
   - If it fails, the pose is not in our map's frame or the track differs from our map.
   - It then also prints the shift and rotation that would align them. Send us that output.

   On the real car, expect a larger median distance than in simulation (real walls, real noise). Up to ~0.08 m is
   fine.

---

## Driving (Test 1)

**Safety**:
- Keep your dead-man switch or joystick override active at all times; it is the primary stop.
- The node itself sends speed 0 when any input is older than 0.3 s, when the controller fails, and when it exits
  (Ctrl-C).
- Stop the run (dead-man first, then Ctrl-C) when:
  - the car leaves the lane or touches a wall;
  - it oscillates or weaves;
  - the localisation jumps;
  - anything looks wrong.

  Note it in `RUN_SHEET.csv`. No intentional crashes, please.

**Procedure**:
1. `check_setup.py` at the start line, car standing. Keep its output: it is saved in `runs/`.
2. **Stand test**, wheels off the ground:
   - Run `bash scripts/run.sh ppo stand_ppo -- --duration 20`.
   - The wheels should turn at about 1 m/s and the steering should move a little.
   - Then the same with `safer`.
3. **Start position** for every run: car on the track near the centre line, pointing along the intended direction
   (within ~30 deg), localisation initialised.
4. **PPO**: 2-3 runs of about 2 minutes (`bash scripts/run.sh ppo ppo_dir+1_a`, `..._b`, ...). Use the other direction
   too if possible.
5. **SAFER**: the same; then more SAFER runs until there are **15-30 minutes of SAFER driving** in total. Fine-tuning
   learns from these.
6. Film a few laps of each controller with a phone.

The run name is free text; `ppo_dir+1_a` style helps us. `--notes "..."` stores a remark in the run's `meta.json`.

---

## Data to send us

After the session:
1. Fill `RUN_SHEET.csv`: one line per run (controller, direction, laps, interventions, wall contacts, floor and light,
   battery, remarks).
2. Run `bash scripts/pack_runs.sh test1`, upload `../safer_runs_test1.tar.gz` (Google Drive, Dropbox, ...) and send
   Yansong the link.
3. Include in the same message:
   - the phone videos;
   - `config/real_car.yaml` if you changed more than the topics;
   - anything unexpected;
   - your answers to the assumption table.

What `runs/` contains and why we need it:

| file | content | used for |
|---|---|---|
| `runs/<run>/node.npz` | every decision: the 1080 LiDAR ranges, pose, speed, yaw rate, the observation, the action, the command, the input stamps, compute time; for SAFER its 16-step plan and its prior's; the executed command, the joystick's command, buttons and axes, and who drove (policy, human, none) | fine-tuning (this is the training data; the safety driver's driving is used too, marked), analysis |
| `runs/<run>/bag/` | rosbag of the scan, odometry, pose, `/drive`, the mux output, `/teleop`, `/joy`, the VESC topics and `/tf` at full rate | checking timing and sensors, rebuilding anything the node did not log |
| `runs/<run>/meta.json`, `config.yaml`, `report.txt` | settings, checkpoint (sha256), driving direction, notes; the run's laps, errors and timing | bookkeeping |
| `runs/check_setup_*.txt` | the pre-flight checks | assumptions |

A 2-minute run is about 40 MB (mostly the rosbag).

---

## Test 2: the fine-tuned SAFER

We send `safer_finetuned.pt` and you put it into `checkpoints/`.
- Set `safer.checkpoint: checkpoints/safer_finetuned.pt` in `config/real_car.yaml`. Keep the offline line commented
  out, to switch back.
- Then repeat the procedure: `check_setup.py`, stand test, track runs.
- Film the demo: 5-10 laps.
- PPO and the offline SAFER again, a few laps each, if there is time (a comparison on the same day).
- Send the runs the same way.

---

## In your Gazebo simulation (optional)

`config/gazebo.yaml` has your simulator's topics and sim time. Start the simulation and run the node:

```bash
ros2 launch f1tenth_gym_gazebo safe_map_launch.py map_name:=real_track     # other terminal
python3 -m safer_deploy.check_setup --config config/gazebo.yaml
CONFIG=config/gazebo.yaml bash scripts/run.sh safer gazebo_test -- --duration 120
```

Our results there (real_track, 200 s per run, spawn heading = direction +1, spawn heading + pi = direction -1):

| controller | direction | laps | wall contacts | speed error | distance from the centre line (largest) | decision time |
|---|---|---|---|---|---|---|
| SAFER | +1 | 1.24 | 0 | 0.017 m/s | 0.072 m (0.25 m) | 49 ms |
| SAFER | -1 | 1.25 | 0 | 0.017 m/s | 0.060 m (0.25 m) | 51 ms |
| PPO | +1 | 1.26 | 0 | 0.023 m/s | 0.064 m (0.24 m) | 0.1 ms |
| PPO | -1 | 1.26 | 0 | 0.022 m/s | 0.053 m (0.23 m) | 0.1 ms |

Gazebo ran at 0.8x real time. The decision period stayed at 0.1 s; at each decision the scan was ~45 ms old and the
odometry ~11 ms.

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `not all inputs arrived` | the topics in `config/real_car.yaml`; `ros2 topic list` / `ros2 topic hz` |
| `scan of N beams ...` and the node stops | the LiDAR is not 1080 beams over +-2.35 rad (assumption 3): tell us |
| `map frame` FAIL | the pose frame or the map differs (assumptions 1 and 2): send us the check output and your map |
| the car does not move | nothing subscribes to `topics.drive`, or the mux gives priority to another input (`check_setup` line `drive`) |
| the car stops by itself | an input is older than 0.3 s: `report.txt` shows the input ages; on a network setup check the WiFi |
| SAFER too slow (`compute SAFER` WARN or FAIL) | use a GPU machine (assumption 9), or run PPO only and tell us |

## Contents

```
README.md (Jayesh's steps), AI_SETUP.md (this file), RUN_SHEET.csv, requirements.txt, MANIFEST.json (sources, sha256)
config/real_car.yaml, config/gazebo.yaml   settings: topics, control period, speed cap, checkpoints
safer_deploy/node.py                      the ROS 2 node (one decision per 0.1 s, logging, safety stops)
safer_deploy/check_setup.py               pre-flight checks
safer_deploy/report.py                    a run's report
safer_deploy/planner.py                   SAFER (MPPI over the world model with the risk term) and its prior
safer_deploy/ppo.py                       the PPO expert
safer_deploy/observation.py               the policies' observation (the training code's functions)
safer_deploy/reference.py                 the reference trajectory
safer_deploy/wm/                          the world-model code, unmodified copies of the training code
checkpoints/safer_offline.pt              SAFER trained offline in simulation
checkpoints/ppo_expert.npz                the PPO expert
data/real_track.png/.yaml                 the map the policies were trained on (0.05 m/px)
data/real_track_reference.npz             its centre line with the reference speed
scripts/run.sh, scripts/pack_runs.sh      one run (node + rosbag); pack the runs to send
videos/safer_sim_real_track_*.mp4         SAFER in simulation, both directions
```

Questions at any point: Yansong.
