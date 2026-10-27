#!/bin/bash
# Type 3.5 redo: original PAS Dis (0-13 bp) then 75/25 split.
# Train 1:10 genomic negatives; test 1:1 (same evaluation setup as v3 E.coli models).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/scripts"

echo "============================================="
echo "Type 3.5 redo — PAS [0,14] Dis + dataset gen"
echo "============================================="
echo "Started at: $(date)"
echo ""

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || conda activate base
fi

echo "Step 0: Original PAS scan for 49 species (tss_m10_range=[0,14])"
python run_pas_scan.py
echo ""

echo "Step 1: Split input into train/test (75/25)"
python split_train_test_input.py
echo ""

run_for_split () {
    SPLIT_NAME="$1"
    NEG_RATIO="$2"
    INPUT_SCAN_DIR="../input_split/${SPLIT_NAME}/scan_result_100"
    OUTPUT_BASE="../output/75-25/${SPLIT_NAME}"

    echo "============================================="
    echo "Running split: ${SPLIT_NAME}"
    echo "Input:  ${INPUT_SCAN_DIR}"
    echo "Output: ${OUTPUT_BASE}"
    echo "Ratio: 1 positive : ${NEG_RATIO} negatives"
    echo "============================================="

    export INPUT_SCAN_DIR OUTPUT_BASE NEGATIVES_PER_POSITIVE="$NEG_RATIO"
    python data_type3_5_35s10_dis.py
    echo "Type 3.5 finished for ${SPLIT_NAME}."
    echo ""
}

run_for_split train 10
run_for_split test 1

echo "============================================="
echo "Type 3.5 redo dataset generation completed!"
echo "Finished at: $(date)"
echo "============================================="
echo "Outputs are in: $SCRIPT_DIR/output/75-25/"
