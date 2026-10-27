#!/usr/bin/env bash
# Export TP/FN reference CSVs (v5 O2O best F1, 100 bp) -> BPM scan -> sequence logos.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/brucehan/Project_02_promoter_prediction_model"
GB="${ROOT}/ref_code/GBactpro_BH"
V4_TYPE1="${GB}/04_genome_scanning_v4_all_data_pool/tss_class_subsets_pr_roc_dense_type1/promoter_architecture_scan/scripts/BPM"
GENOME="${ROOT}/ref_code/GBactPro_Pipeline/01_promoter_extraction/input/genome/NC_000913.2.gb"

PA_IN="${HERE}/promoter_architecture_scan/input"
PA_SCRIPTS="${HERE}/promoter_architecture_scan/scripts"

mkdir -p "${PA_IN}" "${PA_SCRIPTS}" "${HERE}/promoter_architecture_scan/logs"

if [ ! -e "${PA_SCRIPTS}/BPM" ]; then
  ln -sfn "${V4_TYPE1}" "${PA_SCRIPTS}/BPM"
  echo "Linked BPM -> ${V4_TYPE1}"
fi
if [ ! -e "${PA_IN}/NC_000913.2.gb" ]; then
  ln -sfn "${GENOME}" "${PA_IN}/NC_000913.2.gb"
  echo "Linked genome -> ${GENOME}"
fi

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

echo "=============================================="
echo "Step 1/3: export TP/FN subset CSVs (v5 O2O)"
echo "=============================================="
cd "${HERE}"
python3 export_tp_fn_subsets.py

echo ""
echo "=============================================="
echo "Step 2/3: BPM promoter scan (tp, fn)"
echo "=============================================="
bash "${HERE}/promoter_architecture_scan/run_scan_all.sh"

echo ""
echo "=============================================="
echo "Step 3/3: sequence logos"
echo "=============================================="
bash "${HERE}/sequence_logos/run_logos.sh"

echo ""
echo "Done. Scan TSVs: ${HERE}/promoter_architecture_scan/output/"
echo "Logos:          ${HERE}/sequence_logos/output/"
