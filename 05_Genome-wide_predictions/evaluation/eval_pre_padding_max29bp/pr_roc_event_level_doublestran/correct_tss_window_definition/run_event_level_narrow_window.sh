#!/usr/bin/env bash
# Event-level PR+ROC with narrow peak–TSS bands (+ [TSS-39,TSS-34], − [TSS+6,TSS+11]).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# correct_tss_window_definition → pr_roc_event_level_doublestran → eval_pre_padding_max29bp → v5_all_data_pool
V5_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

SCAN_DIR="$V5_ROOT/output/doublestrand_intervals_all_pre_max29bp"
INPUT_TSV=""
for candidate in \
    "$SCAN_DIR/NC_000913.2_type1_pre_max29_allwindows.tsv" \
    "$SCAN_DIR/NC_000913.tsv"; do
  [ -f "$candidate" ] && { INPUT_TSV="$candidate"; break; }
done
[ -n "$INPUT_TSV" ] || { echo "ERROR: No all-windows TSV in $SCAN_DIR"; exit 1; }

REFERENCE_CSV="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
PEAKS_PY="$SCRIPT_DIR/scripts/call_peaks_doublestrand.py"
EVENT_PY="$SCRIPT_DIR/scripts/evaluate_event_level.py"

PEAKS_TSV="$SCRIPT_DIR/peaks/NC_000913_peaks.tsv"
OUT_100="$SCRIPT_DIR/output_100bp_tolerance"
OUT_0="$SCRIPT_DIR/output_no_tolerance"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/peaks" "$OUT_100" "$OUT_0" "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

echo "Narrow-window event-level evaluation"
echo "  Input: $INPUT_TSV"
echo "  Peaks → $PEAKS_TSV"

LOG_PEAKS="$LOG_DIR/call_peaks_${STAMP}.log"
python "$PEAKS_PY" --input-tsv "$INPUT_TSV" --output-tsv "$PEAKS_TSV" --distance 29 2>&1 | tee "$LOG_PEAKS"

fail=0
LOG_100="$LOG_DIR/event_100bp_${STAMP}.log"
LOG_0="$LOG_DIR/event_0bp_${STAMP}.log"

python "$EVENT_PY" --peaks-tsv "$PEAKS_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$OUT_100" > "$LOG_100" 2>&1 &
pid_100=$!
python "$EVENT_PY" --peaks-tsv "$PEAKS_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0 --output-dir "$OUT_0" > "$LOG_0" 2>&1 &
pid_0=$!

wait "$pid_100" || fail=1
wait "$pid_0" || fail=1

if [ "$fail" -ne 0 ]; then
  echo "ERROR: evaluation failed (see $LOG_DIR)"
  exit 1
fi

echo "Done. Outputs: $OUT_0 $OUT_100"
