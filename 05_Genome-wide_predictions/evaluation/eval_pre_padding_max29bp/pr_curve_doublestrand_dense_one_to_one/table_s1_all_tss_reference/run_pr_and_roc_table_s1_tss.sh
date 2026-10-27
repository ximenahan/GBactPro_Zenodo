#!/usr/bin/env bash
# PR + ROC one-to-one (100 bp and no tolerance) using Table_S1_all_TSS.bed as reference.
# Same scan TSV and workflow as eval_pre_padding_max29bp/run_pr_roc_one_to_one.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$HERE/../.." && pwd)"
V5_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"

SCAN_DIR="$V5_ROOT/output/doublestrand_intervals_all_pre_max29bp"
INPUT_TSV=""
for candidate in \
    "$SCAN_DIR/NC_000913.2_type1_pre_max29_allwindows.tsv" \
    "$SCAN_DIR/NC_000913.tsv"; do
  [ -f "$candidate" ] && { INPUT_TSV="$candidate"; break; }
done
[ -n "$INPUT_TSV" ] || { echo "ERROR: No all-windows TSV in $SCAN_DIR"; exit 1; }

REFERENCE_BED="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/scans/lstm_scan/hns_knockout_derepressed_promoter_analysis/table_s1_analysis/Table_S1_all_TSS.bed"
[ -f "$REFERENCE_BED" ] || { echo "ERROR: Reference BED not found: $REFERENCE_BED"; exit 1; }

PR_PY="$HERE/scripts/generate_pr_curve_doublestrand_one_to_one.py"
ROC_PY="$EVAL_ROOT/roc_curve_doublestrand_dense_one_to_one/table_s1_all_tss_reference/scripts/generate_roc_curve_doublestrand_one_to_one.py"

for f in "$PR_PY" "$ROC_PY"; do
  [ -f "$f" ] || { echo "ERROR: Script not found: $f"; exit 1; }
done

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$HERE/logs" "$EVAL_ROOT/roc_curve_doublestrand_dense_one_to_one/table_s1_all_tss_reference/logs"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

echo "============================================="
echo "Table S1 TSS — PR/ROC one-to-one matching"
echo "  Input TSV:     $INPUT_TSV"
echo "  Reference BED: $REFERENCE_BED"
echo "  Started: $(date -Is)"
echo "============================================="

PR_OUT_100="$HERE/output_100bp_tolerance"
PR_OUT_0="$HERE/output_no_tolerance"
PR_LOG="$HERE/logs"
mkdir -p "$PR_OUT_100" "$PR_OUT_0"

python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_BED" --tolerance 100 --output-dir "$PR_OUT_100" > "$PR_LOG/pr_100bp_${STAMP}.log" 2>&1 &
pid_pr100=$!
python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_BED" --tolerance 0   --output-dir "$PR_OUT_0"   > "$PR_LOG/pr_0bp_${STAMP}.log"   2>&1 &
pid_pr0=$!

ROC_BASE="$EVAL_ROOT/roc_curve_doublestrand_dense_one_to_one/table_s1_all_tss_reference"
ROC_OUT_100="$ROC_BASE/output_100bp_tolerance"
ROC_OUT_0="$ROC_BASE/output_no_tolerance"
ROC_LOG="$ROC_BASE/logs"
mkdir -p "$ROC_OUT_100" "$ROC_OUT_0"

python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_BED" --tolerance 100 --output-dir "$ROC_OUT_100" > "$ROC_LOG/roc_100bp_${STAMP}.log" 2>&1 &
pid_roc100=$!
python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_BED" --tolerance 0   --output-dir "$ROC_OUT_0"   > "$ROC_LOG/roc_0bp_${STAMP}.log"   2>&1 &
pid_roc0=$!

echo "Waiting for 4 jobs (PR0, PR100, ROC0, ROC100)..."
fail=0
for pid in $pid_pr100 $pid_pr0 $pid_roc100 $pid_roc0; do
  wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
  echo "ERROR: One or more jobs failed. Check logs under table_s1_all_tss_reference/logs/"
  exit 1
fi

echo "============================================="
echo "Done: $(date -Is)"
echo "============================================="
for d in "$PR_OUT_100" "$PR_OUT_0" "$ROC_OUT_100" "$ROC_OUT_0"; do
  echo "  $d"
  ls "$d"/*.png "$d"/*.tsv "$d"/*.txt 2>/dev/null | sed 's/^/    /' || true
done
