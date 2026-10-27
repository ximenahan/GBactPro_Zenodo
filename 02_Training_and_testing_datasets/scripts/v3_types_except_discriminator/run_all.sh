#!/bin/bash
# Run all data generation scripts with 75/25 train-test split.
# Train and test both use 1:10 genomic (random) negatives.
# Genomic exclusions use train+test TSS rows combined per species (see genomic_negatives.py).
# Type scripts run in parallel for speed.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/scripts"

echo "============================================="
echo "Dataset Generation V3 - split 75/25 + train/test 1:10"
echo "============================================="
echo "Started at: $(date)"
echo ""

# Activate conda environment if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate bioinfo 2>/dev/null || conda activate base
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    conda activate bioinfo 2>/dev/null || conda activate base
fi

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

    echo "Starting types 1–8 in parallel..."
    pids=()
    export INPUT_SCAN_DIR OUTPUT_BASE NEGATIVES_PER_POSITIVE="$NEG_RATIO"
    python data_type1_35s10.py & pids+=($!)
    python data_type2_up35s10_original.py & pids+=($!)
    python data_type3_up35s10_new.py & pids+=($!)
    python data_type4_total50_up8_12.py & pids+=($!)
    python data_type5_up35_40.py & pids+=($!)
    python data_type6_40bp_upstream_tss.py & pids+=($!)
    python data_type7_minus60_plus21.py & pids+=($!)
    python data_type8_35s10_leftpad40.py & pids+=($!)

    ec=0
    for pid in "${pids[@]}"; do
        wait "$pid" || ec=1
    done
    if [ "$ec" -ne 0 ]; then
        echo "ERROR: One or more data_type scripts failed for split ${SPLIT_NAME}"
        return 1
    fi
    echo "Types 1–8 finished for ${SPLIT_NAME}."
    echo ""
}

run_for_split train 10
run_for_split test 10

echo "============================================="
echo "All dataset types completed for train + test!"
echo "Finished at: $(date)"
echo "============================================="
echo "Outputs are in: $SCRIPT_DIR/output/75-25/"
