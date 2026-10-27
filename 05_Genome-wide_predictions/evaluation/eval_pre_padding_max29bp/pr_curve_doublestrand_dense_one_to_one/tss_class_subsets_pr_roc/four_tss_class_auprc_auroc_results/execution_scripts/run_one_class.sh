#!/usr/bin/env bash
# Run PR + ROC one-to-one for one TSS class (0 and 100 bp tolerance), four Python jobs in parallel.
set -euo pipefail

CLASS="${1:?usage: run_one_class.sh <primary|secondary|internal|antisense>}"

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# eval_pre_padding_max29bp (parent of pr_curve_doublestrand_dense_one_to_one)
EVAL_ROOT="$(cd "$RUN_ROOT/../.." && pwd)"
# 04_genome_scanning_v5_all_data_pool — scan outputs live here, not under eval_*
V5_POOL="$(cd "$RUN_ROOT/../../.." && pwd)"
SCRIPT_DIR="$(cd "$RUN_ROOT/.." && pwd)"

SCAN_DIR="$V5_POOL/output/doublestrand_intervals_all_pre_max29bp"
INPUT_TSV=""
for candidate in \
    "$SCAN_DIR/NC_000913.2_type1_pre_max29_allwindows.tsv" \
    "$SCAN_DIR/NC_000913.tsv"; do
  [ -f "$candidate" ] && { INPUT_TSV="$candidate"; break; }
done
[ -n "$INPUT_TSV" ] || { echo "ERROR: No all-windows TSV in $SCAN_DIR"; exit 1; }

REFERENCE_CSV="$RUN_ROOT/references_pos_strand/${CLASS}_pos_strand.csv"
[ -f "$REFERENCE_CSV" ] || { echo "ERROR: Missing $REFERENCE_CSV (run prepare_reference_csvs.py)"; exit 1; }

PR_PY="$SCRIPT_DIR/scripts/generate_pr_curve_doublestrand_one_to_one.py"
ROC_PY="$EVAL_ROOT/roc_curve_doublestrand_dense_one_to_one/scripts/generate_roc_curve_doublestrand_one_to_one.py"
for f in "$PR_PY" "$ROC_PY"; do
  [ -f "$f" ] || { echo "ERROR: Script not found: $f"; exit 1; }
done

OUT_BASE="$RUN_ROOT/${CLASS}"
PR_OUT_100="$OUT_BASE/pr_output_100bp_tolerance"
PR_OUT_0="$OUT_BASE/pr_output_no_tolerance"
ROC_OUT_100="$OUT_BASE/roc_output_100bp_tolerance"
ROC_OUT_0="$OUT_BASE/roc_output_no_tolerance"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$PR_OUT_100" "$PR_OUT_0" "$ROC_OUT_100" "$ROC_OUT_0" "$LOG_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[$CLASS] Input TSV: $INPUT_TSV"
echo "[$CLASS] Reference:  $REFERENCE_CSV"
echo "[$CLASS] Started:    $(date -Is)"

python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$PR_OUT_100" \
  > "$LOG_DIR/${CLASS}_pr_100bp_${STAMP}.log" 2>&1 &
pid_pr100=$!
python "$PR_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0 --output-dir "$PR_OUT_0" \
  > "$LOG_DIR/${CLASS}_pr_0bp_${STAMP}.log" 2>&1 &
pid_pr0=$!
python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 100 --output-dir "$ROC_OUT_100" \
  > "$LOG_DIR/${CLASS}_roc_100bp_${STAMP}.log" 2>&1 &
pid_roc100=$!
python "$ROC_PY" --input-tsv "$INPUT_TSV" --reference-csv "$REFERENCE_CSV" --tolerance 0 --output-dir "$ROC_OUT_0" \
  > "$LOG_DIR/${CLASS}_roc_0bp_${STAMP}.log" 2>&1 &
pid_roc0=$!

fail=0
for pid in $pid_pr100 $pid_pr0 $pid_roc100 $pid_roc0; do
  wait "$pid" || fail=1
done

if [ "$fail" -ne 0 ]; then
  echo "ERROR [$CLASS]: one or more jobs failed; see $LOG_DIR"
  exit 1
fi

echo "[$CLASS] Done: $(date -Is)"
ls "$PR_OUT_100" "$PR_OUT_0" "$ROC_OUT_100" "$ROC_OUT_0" 2>/dev/null | head -40 || true
