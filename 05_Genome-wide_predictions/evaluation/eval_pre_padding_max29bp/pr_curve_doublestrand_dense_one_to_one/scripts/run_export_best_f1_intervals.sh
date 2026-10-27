#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
V5_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"

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

PR_CURVE="$SCRIPT_DIR/../output_100bp_tolerance/pr_curve_data.tsv"
OUT_DIR="$SCRIPT_DIR/../output_100bp_tolerance/predicted_intervals_best_f1_100bp_one_to_one"
mkdir -p "$OUT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

echo "INPUT_TSV=$INPUT_TSV"
echo "PR_CURVE=$PR_CURVE"
echo "OUT_DIR=$OUT_DIR"
LOG="$OUT_DIR/run.log"
python "$SCRIPT_DIR/export_predicted_intervals_at_best_f1.py" \
  --pr-curve-tsv "$PR_CURVE" \
  --input-tsv "$INPUT_TSV" \
  --reference-csv "$REFERENCE_CSV" \
  --tolerance 100 \
  --output-dir "$OUT_DIR" 2>&1 | tee "$LOG"
echo "Done. Log: $LOG"
