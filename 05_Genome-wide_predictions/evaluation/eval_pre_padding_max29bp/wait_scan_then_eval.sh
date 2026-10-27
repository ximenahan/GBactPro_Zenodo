#!/usr/bin/env bash
# Wait for genome scan to finish, then run all evaluations.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_SESSION="${1:-v5_scan_pre_max29bp}"
SCAN_OUT_DIR="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactpro_BH/04_genome_scanning_v5_all_data_pool/output/doublestrand_intervals_all_pre_max29bp"

echo "Waiting for scan completion signal from: $WAIT_SESSION ..."
scan_started=0
while true; do
  if pgrep -fa "scan_double_strand_intervals.py.*--out-dir ${SCAN_OUT_DIR}" >/dev/null 2>&1; then
    scan_started=1
    sleep 10
  else
    if [ "$scan_started" -eq 1 ]; then
      if ! tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] $WAIT_SESSION session ended."
      else
        echo "[$(date '+%H:%M:%S')] Scan process ended (session may remain open)."
      fi
      break
    fi
    if ! tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
      echo "ERROR: Scan tmux ended before scan started. Abort eval."
      exit 1
    fi
    sleep 10
  fi
done
echo "[$(date '+%H:%M:%S')] Launching evaluations..."

exec bash "$SCRIPT_DIR/run_all_evaluations.sh"
