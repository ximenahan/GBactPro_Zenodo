#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/brucehan/Project_02_promoter_prediction_model"
GB="${ROOT}/ref_code/GBactpro_BH"
V5="${GB}/04_genome_scanning_v5_all_data_pool"
PR_O2O="${V5}/eval_pre_padding_max29bp/pr_curve_doublestrand_dense_one_to_one"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

exec python3 "${HERE}/generate_hns_bar_chart.py" \
  --pr-curve-tsv "${PR_O2O}/output_100bp_tolerance/pr_curve_data.tsv" \
  --predictions-tsv "${V5}/output/doublestrand_intervals_all_pre_max29bp/NC_000913.tsv" \
  --reference-csv "${ROOT}/ref_code/GBactPro_new/data/tss_class_subsets/ecoli_all4classes_tss_filtered.csv" \
  --hns-bed "${ROOT}/igv_visualization/reference/HNS_binding_sites.bed" \
  --match-tolerance 100 \
  --output-dir "${HERE}"
