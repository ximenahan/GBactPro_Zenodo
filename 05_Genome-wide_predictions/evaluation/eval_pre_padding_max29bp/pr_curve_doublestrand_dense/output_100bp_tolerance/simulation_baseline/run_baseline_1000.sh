#!/usr/bin/env bash
# Monte Carlo random-score baseline for dense hybrid PR: 1000 sims, tolerance 100 bp.
set -euo pipefail
export PYTHONUNBUFFERED=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$(cd "$HERE/../../.." && pwd)"
PY="$EVAL/scripts/simulation_baseline_dense.py"
TSV="$EVAL/../output/doublestrand_intervals_all_pre_max29bp/NC_000913.tsv"
REF="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
LOG="$HERE/run_1000sims_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
python3 "$PY" --input-tsv "$TSV" --reference-csv "$REF" --tolerance 100 --out-dir "$HERE" --n-sims 1000 --seed 42
