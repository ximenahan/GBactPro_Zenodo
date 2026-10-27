#!/usr/bin/env bash
# Part 3: v5 double-strand one-to-one PR + ROC vs union reference (11834 ∪ 371).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ANALYSIS_ROOT/../../../.." && pwd)"
EVAL_PRE="$PROJECT_ROOT/ref_code/GBactpro_BH/04_genome_scanning_v5_all_data_pool/eval_pre_padding_max29bp"
V5_POOL="$PROJECT_ROOT/ref_code/GBactpro_BH/04_genome_scanning_v5_all_data_pool"

PR_PY="$EVAL_PRE/pr_curve_doublestrand_dense_one_to_one/scripts/generate_pr_curve_doublestrand_one_to_one.py"
ROC_PY="$EVAL_PRE/roc_curve_doublestrand_dense_one_to_one/scripts/generate_roc_curve_doublestrand_one_to_one.py"

UNION_BED="$ANALYSIS_ROOT/part2_tss_union_bed/all4classes_11834_union_derepressed_371.bed"
REF_CSV="$SCRIPT_DIR/union_11834_derepressed_reference.csv"

SCAN_DIR="$V5_POOL/output/doublestrand_intervals_all_pre_max29bp"
INPUT_TSV=""
for c in "$SCAN_DIR/NC_000913.2_type1_pre_max29_allwindows.tsv" "$SCAN_DIR/NC_000913.tsv"; do
  [ -f "$c" ] && INPUT_TSV="$c" && break
done
[ -n "$INPUT_TSV" ] || { echo "ERROR: No scan TSV in $SCAN_DIR"; exit 1; }

LOG_DIR="$SCRIPT_DIR/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "Part 3: v5 one-to-one PR + ROC (union TSS ref)"
echo "============================================="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "INPUT_TSV: $INPUT_TSV"
echo "REF_CSV:   $REF_CSV"
echo ""

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

[ -f "$UNION_BED" ] || { echo "ERROR: Run part2 first. Missing $UNION_BED"; exit 1; }
[ -f "$PR_PY" ] && [ -f "$ROC_PY" ] || { echo "ERROR: PR/ROC script missing under $EVAL_PRE"; exit 1; }

python3 "$SCRIPT_DIR/convert_bed_to_reference_csv.py" --input "$UNION_BED" --output "$REF_CSV"

fail=0

run_tolerance() {
  local tol="$1"
  local tag="$2"
  local pr_out="$SCRIPT_DIR/${tag}/pr"
  local roc_out="$SCRIPT_DIR/${tag}/roc"
  mkdir -p "$pr_out" "$roc_out"

  echo "=== ${tag} (tolerance=${tol}) ==="
  python3 "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REF_CSV" --tolerance "$tol" --output-dir "$pr_out" \
    >"$LOG_DIR/pr_${tag}_${STAMP}.log" 2>&1 &
  pid_pr=$!
  python3 "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REF_CSV" --tolerance "$tol" --output-dir "$roc_out" \
    >"$LOG_DIR/roc_${tag}_${STAMP}.log" 2>&1 &
  pid_roc=$!

  wait "$pid_pr" || fail=1
  wait "$pid_roc" || fail=1
}

run_tolerance 0 output_no_tolerance
run_tolerance 100 output_100bp_tolerance

if [ "$fail" -ne 0 ]; then
  echo "Part 3 finished with errors."
  exit 1
fi
echo "Part 3 done: $SCRIPT_DIR/output_no_tolerance/ output_100bp_tolerance/"
