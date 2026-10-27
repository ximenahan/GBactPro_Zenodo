#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/brucehan/Project_02_promoter_prediction_model"
GB="${ROOT}/ref_code/GBactpro_BH"
V4_BPM="${GB}/04_genome_scanning_v4_all_data_pool/tss_class_subsets_pr_roc_dense_type1/promoter_architecture_scan/scripts/BPM"
GENOME="${ROOT}/ref_code/GBactPro_Pipeline/01_promoter_extraction/input/genome/NC_000913.2.gb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
PA_IN="${BASE_DIR}/input"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$LOG_DIR" "$PA_IN"

if [ ! -e "${SCRIPT_DIR}/BPM" ]; then
  ln -sfn "${V4_BPM}" "${SCRIPT_DIR}/BPM"
  echo "Linked BPM -> ${V4_BPM}"
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

cd "$SCRIPT_DIR"

fail=0
for subset in tp fn; do
  log="$LOG_DIR/scan_${subset}.log"
  echo "[$(date '+%H:%M:%S')] scan: $subset -> $log"
  if python3 scan_tss_subset.py --subset "$subset" >"$log" 2>&1; then
    echo "[$(date '+%H:%M:%S')] OK: $subset"
  else
    echo "[$(date '+%H:%M:%S')] FAIL: $subset (see $log)"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "Outputs: $(dirname "$SCRIPT_DIR")/output/{tp,fn}/"
