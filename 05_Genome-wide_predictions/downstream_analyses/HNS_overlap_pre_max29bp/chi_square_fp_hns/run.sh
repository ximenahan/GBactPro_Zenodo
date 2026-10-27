#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactpro_BH/05_HNS_exe"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

exec python3 "${HERE}/chi_square_fp_hns_from_summary.py" \
  --no-tol-summary "${ROOT}/no_tolerance/bar_chart/hns_overlap_summary.tsv" \
  --with-tol-summary "${ROOT}/100_bp_tolerance/bar_chart/hns_overlap_summary.tsv" \
  --output-dir "${HERE}"
