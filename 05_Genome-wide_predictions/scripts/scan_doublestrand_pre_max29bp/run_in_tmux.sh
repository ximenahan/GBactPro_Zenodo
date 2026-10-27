#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${TMUX_SESSION:-v5_scan_pre_max29bp}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR" \
  "bash ./wait_train_then_scan.sh v5_train_pre_max29bp; exec bash"

echo "Started session: $SESSION"
echo "  1. Waits for v5_train_pre_max29bp training process to finish"
echo "  2. Then runs 29 bp genome scan with pre-padding max29bp model"
echo "  Attach: tmux attach -t $SESSION"
