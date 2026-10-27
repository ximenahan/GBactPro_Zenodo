#!/usr/bin/env bash
# Monte Carlo random-score baseline: 1000 sims, tolerance 0 bp.
set -euo pipefail
export PYTHONUNBUFFERED=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# eval_pre_padding_max29bp
EVAL="$(cd "$HERE/../../.." && pwd)"
PY="$EVAL/pr_curve_doublestrand_dense_one_to_one/scripts/simulation_baseline_one_to_one.py"
TSV="$EVAL/../output/doublestrand_intervals_all_pre_max29bp/NC_000913.tsv"
REF="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
LOG="$HERE/run_1000sims_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
python3 "$PY" --input-tsv "$TSV" --reference-csv "$REF" --tolerance 0 --out-dir "$HERE" --n-sims 1000 --seed 42
