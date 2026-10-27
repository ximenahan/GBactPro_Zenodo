#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${TMUX_SESSION:-v5_eval_pre_max29bp}"
mkdir -p "$SCRIPT_DIR/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux '$SESSION' already exists. Attach: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR" \
  "bash ./wait_scan_then_eval.sh v5_scan_pre_max29bp 2>&1 | tee logs/eval_all_$(date +%Y%m%d_%H%M%S).log; exec bash"

echo "Started session: $SESSION"
echo "  1. Waits for v5_scan_pre_max29bp scan process to finish"
echo "  2. Runs PR + ROC + event-level evaluations (29 bp)"
echo "  Attach: tmux attach -t $SESSION"
