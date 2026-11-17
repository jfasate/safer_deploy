#!/usr/bin/env bash
# One run on the car: a rosbag of the inputs and the commands, and the driving node; Ctrl-C ends both (the node sends
# speed 0 first) and the run's report is printed. Everything goes to runs/<date>_<time>_<NAME>/.
#
#   bash scripts/run.sh CONTROLLER NAME [-- extra node arguments, e.g. --duration 120 --notes "dry asphalt"]
#     CONTROLLER  ppo | safer | bc_prior
#     NAME        a short label, e.g. ppo_dir+1_a
#   environment: CONFIG (default config/real_car.yaml), PYTHON (default python3: a Python with rclpy, numpy and torch)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
CTRL=${1:?controller: ppo, safer or bc_prior}
NAME=${2:?run name}
shift 2
[ "${1:-}" = "--" ] && shift
CONFIG="${CONFIG:-$HERE/config/real_car.yaml}"
PYTHON="${PYTHON:-python3}"
RUN="$HERE/runs/$(date +%Y%m%d_%H%M%S)_${NAME}"
mkdir -p "$RUN"
cp "$CONFIG" "$RUN/config.yaml"
TOPICS=$("$PYTHON" -c "
import sys, yaml
t = yaml.safe_load(open(sys.argv[1]))['topics']
h = yaml.safe_load(open(sys.argv[1])).get('human') or {}
s = [t['scan'], t['odom'], t['pose'], t['drive']] + ([t['imu']] if t['yaw_rate'] == 'imu' else []) + list(t.get('record_extra', []))
s += [h[k] for k in ('executed', 'teleop', 'joy') if h.get(k)] + list(h.get('record') or [])
print(' '.join(dict.fromkeys(s)))" "$CONFIG")
ros2 bag record -o "$RUN/bag" $TOPICS > "$RUN/bag.log" 2>&1 &
BAG=$!
stop_bag() { kill -INT "$BAG" 2>/dev/null || true; wait "$BAG" 2>/dev/null || true; }
trap stop_bag EXIT
sleep 1
"$PYTHON" -m safer_deploy.node --config "$CONFIG" --controller "$CTRL" --run_dir "$RUN" "$@" || echo "node ended with an error"
stop_bag
trap - EXIT
"$PYTHON" -m safer_deploy.report "$RUN" || true
echo "run folder: $RUN"
