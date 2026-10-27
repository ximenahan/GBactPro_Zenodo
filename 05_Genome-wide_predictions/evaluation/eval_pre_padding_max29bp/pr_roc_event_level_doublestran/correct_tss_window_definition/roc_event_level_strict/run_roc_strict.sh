#!/usr/bin/env bash
# Strict event-level ROC only (greedy match; no is_positive for FPR).
# Reads peaks from parent correct_tss_window_definition/peaks/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$SCRIPT_DIR/.." && pwd)"

PEAKS_TSV="$PARENT/peaks/NC_000913_peaks.tsv"
REFERENCE_CSV="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
EVAL_PY="$SCRIPT_DIR/eval_roc_strict.py"

OUT_100="$SCRIPT_DIR/output_100bp_tolerance"
OUT_0="$SCRIPT_DIR/output_no_tolerance"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$OUT_100" "$OUT_0" "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "============================================="
echo "Strict event-level ROC (parallel, 0bp + 100bp)"
echo "  Peaks: $PEAKS_TSV"
echo "  Started: $(date -Is)"
echo "============================================="

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

[ -f "$PEAKS_TSV" ]       || { echo "ERROR: Peaks TSV not found: $PEAKS_TSV"; exit 1; }
[ -f "$REFERENCE_CSV" ]   || { echo "ERROR: Reference CSV not found: $REFERENCE_CSV"; exit 1; }
[ -f "$EVAL_PY" ]        || { echo "ERROR: eval_roc_strict.py not found"; exit 1; }

LOG_100="$LOG_DIR/roc_strict_100bp_${STAMP}.log"
LOG_0="$LOG_DIR/roc_strict_0bp_${STAMP}.log"
fail=0

python "$EVAL_PY" \
  --peaks-tsv "$PEAKS_TSV" \
  --reference-csv "$REFERENCE_CSV" \
  --tolerance 100 \
  --output-dir "$OUT_100" \
  > "$LOG_100" 2>&1 &
pid_100=$!

python "$EVAL_PY" \
  --peaks-tsv "$PEAKS_TSV" \
  --reference-csv "$REFERENCE_CSV" \
  --tolerance 0 \
  --output-dir "$OUT_0" \
  > "$LOG_0" 2>&1 &
pid_0=$!

wait "$pid_100" || fail=1
wait "$pid_0"   || fail=1

echo "Logs: $LOG_100  $LOG_0"
if [ "$fail" -ne 0 ]; then
  echo "ERROR: one or more jobs failed"
  exit 1
fi

echo "Outputs:"
ls -la "$OUT_100" "$OUT_0"
echo "Done: $(date -Is)"
