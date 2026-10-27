#!/usr/bin/env bash
# Wait for training Python to finish (tmux session may stay open), then run genome scan.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_SESSION="${1:-v5_train_pre_max29bp}"
TRAIN_MARKER="train_type1_pre_padding_max29bp.py"

echo "Waiting for training to complete (session: $WAIT_SESSION, marker: $TRAIN_MARKER) ..."
started=0
while true; do
  if pgrep -fa "$TRAIN_MARKER" >/dev/null 2>&1; then
    started=1
  fi
  if [ "$started" -eq 1 ] && ! pgrep -fa "$TRAIN_MARKER" >/dev/null 2>&1; then
    echo "[$(date '+%H:%M:%S')] Training process finished. Launching genome scan..."
    break
  fi
  if ! tmux has-session -t "$WAIT_SESSION" 2>/dev/null; then
    if [ "$started" -eq 0 ]; then
      echo "ERROR: Train tmux session ended before training started. Abort scan."
      exit 1
    fi
    echo "[$(date '+%H:%M:%S')] Train session closed after training. Launching genome scan..."
    break
  fi
  sleep 10
done

exec bash "$SCRIPT_DIR/run_scan_all_windows.sh"
