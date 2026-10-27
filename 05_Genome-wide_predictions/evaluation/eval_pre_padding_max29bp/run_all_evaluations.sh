#!/usr/bin/env bash
# Window-level PR/ROC + event-level (29 bp), same methodology as post/internal max29bp eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V5_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SCAN_DIR="$V5_ROOT/output/doublestrand_intervals_all_pre_max29bp"
INPUT_TSV=""
for candidate in \
    "$SCAN_DIR/NC_000913.2_type1_pre_max29_allwindows.tsv" \
    "$SCAN_DIR/NC_000913.tsv"; do
  [ -f "$candidate" ] && { INPUT_TSV="$candidate"; break; }
done
[ -n "$INPUT_TSV" ] || { echo "ERROR: No all-windows TSV in $SCAN_DIR"; exit 1; }

REFERENCE_CSV="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
[ -f "$REFERENCE_CSV" ] || { echo "ERROR: Reference CSV not found: $REFERENCE_CSV"; exit 1; }

PR_PY="$SCRIPT_DIR/scripts/generate_pr_curve_doublestrand.py"
ROC_PY="$SCRIPT_DIR/scripts/generate_roc_curve_doublestrand.py"
PEAKS_PY="$SCRIPT_DIR/scripts/call_peaks_doublestrand.py"
EVENT_PY="$SCRIPT_DIR/scripts/evaluate_event_level.py"

for f in "$PR_PY" "$ROC_PY" "$PEAKS_PY" "$EVENT_PY"; do
  [ -f "$f" ] || { echo "ERROR: Script not found: $f"; exit 1; }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

echo "============================================="
echo "PRE PADDING MAX29BP — full evaluation (29 bp windows)"
echo "  Input TSV: $INPUT_TSV"
echo "  Started: $(date -Is)"
echo "============================================="

echo ""
echo "=== [1/3] Window-level PR curves ==="
PR_OUT_100="$SCRIPT_DIR/pr_curve_doublestrand_dense/output_100bp_tolerance"
PR_OUT_0="$SCRIPT_DIR/pr_curve_doublestrand_dense/output_no_tolerance"
PR_LOG="$SCRIPT_DIR/pr_curve_doublestrand_dense/logs"
mkdir -p "$PR_OUT_100" "$PR_OUT_0" "$PR_LOG"

python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$PR_OUT_100" > "$PR_LOG/pr_100bp_${STAMP}.log" 2>&1 &
pid_pr100=$!
python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0   --output-dir "$PR_OUT_0"   > "$PR_LOG/pr_0bp_${STAMP}.log"   2>&1 &
pid_pr0=$!

echo "=== [2/3] Window-level ROC curves ==="
ROC_OUT_100="$SCRIPT_DIR/roc_curve_doublestrand_dense/output_100bp_tolerance"
ROC_OUT_0="$SCRIPT_DIR/roc_curve_doublestrand_dense/output_no_tolerance"
ROC_LOG="$SCRIPT_DIR/roc_curve_doublestrand_dense/logs"
mkdir -p "$ROC_OUT_100" "$ROC_OUT_0" "$ROC_LOG"

python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$ROC_OUT_100" > "$ROC_LOG/roc_100bp_${STAMP}.log" 2>&1 &
pid_roc100=$!
python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0   --output-dir "$ROC_OUT_0"   > "$ROC_LOG/roc_0bp_${STAMP}.log"   2>&1 &
pid_roc0=$!

echo "=== [3/3] Event-level PR + ROC (peak calling first) ==="
EVT_DIR="$SCRIPT_DIR/pr_roc_event_level_doublestrand"
PEAKS_TSV="$EVT_DIR/peaks/NC_000913_peaks.tsv"
EVT_OUT_100="$EVT_DIR/output_100bp_tolerance"
EVT_OUT_0="$EVT_DIR/output_no_tolerance"
EVT_LOG="$EVT_DIR/logs"
mkdir -p "$EVT_DIR/peaks" "$EVT_OUT_100" "$EVT_OUT_0" "$EVT_LOG"

python "$PEAKS_PY" --input-tsv "$INPUT_TSV" --output-tsv "$PEAKS_TSV" --distance 29 2>&1 | tee "$EVT_LOG/call_peaks_${STAMP}.log"

python "$EVENT_PY" --peaks-tsv "$PEAKS_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$EVT_OUT_100" > "$EVT_LOG/event_100bp_${STAMP}.log" 2>&1 &
pid_evt100=$!
python "$EVENT_PY" --peaks-tsv "$PEAKS_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0   --output-dir "$EVT_OUT_0"   > "$EVT_LOG/event_0bp_${STAMP}.log"   2>&1 &
pid_evt0=$!

echo ""
echo "Waiting for all 6 parallel jobs..."
fail=0
for pid in $pid_pr100 $pid_pr0 $pid_roc100 $pid_roc0 $pid_evt100 $pid_evt0; do
  wait "$pid" || fail=1
done

echo ""
if [ "$fail" -ne 0 ]; then
  echo "ERROR: One or more evaluation jobs failed. Check logs under $SCRIPT_DIR/*/logs/"
  exit 1
fi

echo "============================================="
echo "All evaluations completed: $(date -Is)"
echo "============================================="
echo ""
echo "Results:"
for d in "$PR_OUT_100" "$PR_OUT_0" "$ROC_OUT_100" "$ROC_OUT_0" "$EVT_OUT_100" "$EVT_OUT_0"; do
  echo "  $d"
  ls "$d"/*.png "$d"/*.tsv "$d"/*.txt 2>/dev/null | sed 's/^/    /'
done
