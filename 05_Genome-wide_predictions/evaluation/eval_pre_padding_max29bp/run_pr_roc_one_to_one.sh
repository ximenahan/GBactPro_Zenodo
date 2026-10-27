#!/usr/bin/env bash
# One-to-one window/promoter matching PR + ROC (does not touch original dense folders).
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

PR_PY="$SCRIPT_DIR/pr_curve_doublestrand_dense_one_to_one/scripts/generate_pr_curve_doublestrand_one_to_one.py"
ROC_PY="$SCRIPT_DIR/roc_curve_doublestrand_dense_one_to_one/scripts/generate_roc_curve_doublestrand_one_to_one.py"

for f in "$PR_PY" "$ROC_PY"; do
  [ -f "$f" ] || { echo "ERROR: Script not found: $f"; exit 1; }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p \
  "$SCRIPT_DIR/pr_curve_doublestrand_dense_one_to_one/logs" \
  "$SCRIPT_DIR/roc_curve_doublestrand_dense_one_to_one/logs"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

echo "============================================="
echo "PRE PADDING MAX29BP — PR/ROC one-to-one matching"
echo "  Input TSV: $INPUT_TSV"
echo "  Started: $(date -Is)"
echo "============================================="

PR_OUT_100="$SCRIPT_DIR/pr_curve_doublestrand_dense_one_to_one/output_100bp_tolerance"
PR_OUT_0="$SCRIPT_DIR/pr_curve_doublestrand_dense_one_to_one/output_no_tolerance"
PR_LOG="$SCRIPT_DIR/pr_curve_doublestrand_dense_one_to_one/logs"
mkdir -p "$PR_OUT_100" "$PR_OUT_0"

python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$PR_OUT_100" > "$PR_LOG/pr_100bp_${STAMP}.log" 2>&1 &
pid_pr100=$!
python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0   --output-dir "$PR_OUT_0"   > "$PR_LOG/pr_0bp_${STAMP}.log"   2>&1 &
pid_pr0=$!

ROC_OUT_100="$SCRIPT_DIR/roc_curve_doublestrand_dense_one_to_one/output_100bp_tolerance"
ROC_OUT_0="$SCRIPT_DIR/roc_curve_doublestrand_dense_one_to_one/output_no_tolerance"
ROC_LOG="$SCRIPT_DIR/roc_curve_doublestrand_dense_one_to_one/logs"
mkdir -p "$ROC_OUT_100" "$ROC_OUT_0"

python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$ROC_OUT_100" > "$ROC_LOG/roc_100bp_${STAMP}.log" 2>&1 &
pid_roc100=$!
python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0   --output-dir "$ROC_OUT_0"   > "$ROC_LOG/roc_0bp_${STAMP}.log"   2>&1 &
pid_roc0=$!

echo "Waiting for 4 jobs (PR0, PR100, ROC0, ROC100)..."
fail=0
for pid in $pid_pr100 $pid_pr0 $pid_roc100 $pid_roc0; do
  wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
  echo "ERROR: One or more jobs failed. Check logs under *_one_to_one/logs/"
  exit 1
fi

echo "============================================="
echo "Done: $(date -Is)"
echo "============================================="
for d in "$PR_OUT_100" "$PR_OUT_0" "$ROC_OUT_100" "$ROC_OUT_0"; do
  echo "  $d"
  ls "$d"/*.png "$d"/*.tsv "$d"/*.txt 2>/dev/null | sed 's/^/    /' || true
done
