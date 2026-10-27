#!/usr/bin/env bash
# Four tmux windows (one per TSS class), each runs PR+ROC one-to-one (0 and 100 bp) in parallel.
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="prroc_o2o_subsets"

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

tmux new-session -d -s "$SESSION" -n "primary" \
  "cd '$RUN_ROOT' && bash '$RUN_ROOT/run_one_class.sh' primary; echo DONE primary; sleep 3600"

tmux new-window -t "$SESSION" -n "secondary" \
  "cd '$RUN_ROOT' && bash '$RUN_ROOT/run_one_class.sh' secondary; echo DONE secondary; sleep 3600"

tmux new-window -t "$SESSION" -n "internal" \
  "cd '$RUN_ROOT' && bash '$RUN_ROOT/run_one_class.sh' internal; echo DONE internal; sleep 3600"

tmux new-window -t "$SESSION" -n "antisense" \
  "cd '$RUN_ROOT' && bash '$RUN_ROOT/run_one_class.sh' antisense; echo DONE antisense; sleep 3600"

echo "Started tmux session: $SESSION"
echo "  tmux attach -t $SESSION"
echo "Windows: primary, secondary, internal, antisense (four classes in parallel)."
