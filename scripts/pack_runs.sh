#!/usr/bin/env bash
# Pack every run (runs/: the node logs, rosbags, reports, check_setup outputs, run sheet) into one archive to upload.
#   bash scripts/pack_runs.sh [LABEL]     ->  safer_runs_<LABEL or date>.tar.gz next to this folder
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-$(date +%Y%m%d)}"
OUT="$(dirname "$HERE")/safer_runs_${LABEL}.tar.gz"
tar -czf "$OUT" -C "$HERE" runs
du -h "$OUT"
echo "upload $OUT and send us the link"
