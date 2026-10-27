#!/usr/bin/env bash
# Double-strand 29 bp scan — all windows — pre-padding-max29bp model (29 bp input).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V5_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GBH="$(cd "$V5_ROOT/.." && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$V5_ROOT/logs"
mkdir -p "$LOG_DIR"
OUT_DIR="${OUT_DIR:-$V5_ROOT/output/doublestrand_intervals_all_pre_max29bp}"
mkdir -p "$OUT_DIR"

GENOME_FASTA="${GENOME_FASTA:-/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/raw/raw_data/genome/NC_000913.2.fasta}"
MODEL_DIR="$GBH/03_model_training_v3_genome_use/scripts/models/lstm/genome_use_49/35+s+10_pre_max29bp/random"
PY="$GBH/04_genome_scanning_v5_all_data_pool/add_double_strand_and_window_interval/scan_double_strand_intervals.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/scan_pre_max29bp_allwindows_${STAMP}.log"

echo "============================================="
echo "Genome scan ALL windows — pre padding max29bp model (29 bp)"
echo "Started: $(date -Is)"
echo "Model: $MODEL_DIR"
echo "Output: $OUT_DIR"
echo "============================================="

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

PREPEND_LIBS=""
if [ -d "/usr/local/cuda-11.4/lib64" ]; then
  PREPEND_LIBS="/usr/local/cuda-11.4/lib64"
elif [ -d "/usr/local/cuda/lib64" ]; then
  PREPEND_LIBS="/usr/local/cuda/lib64"
fi
if [ -n "${CONDA_PREFIX:-}" ]; then
  PYVER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || PYVER="3.8"
  NVDIR="$CONDA_PREFIX/lib/python${PYVER}/site-packages/nvidia"
  [ -d "$NVDIR/cudnn/lib" ] && PREPEND_LIBS="$NVDIR/cudnn/lib:$PREPEND_LIBS"
  [ -d "$NVDIR/cublas/lib" ] && PREPEND_LIBS="$NVDIR/cublas/lib:$PREPEND_LIBS"
fi
[ -n "$PREPEND_LIBS" ] && export LD_LIBRARY_PATH="${PREPEND_LIBS}:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export SCAN_INFER_BATCH_SIZE="${SCAN_INFER_BATCH_SIZE:-2048}"

[ -f "$GENOME_FASTA" ] || { echo "ERROR: Genome not found: $GENOME_FASTA"; exit 1; }
[ -d "$MODEL_DIR" ]    || { echo "ERROR: Model not found: $MODEL_DIR"; exit 1; }

exec > >(tee -a "$LOG_FILE") 2>&1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SCAN_INFER_BATCH_SIZE=$SCAN_INFER_BATCH_SIZE"

python3 "$PY" \
  --genome "$GENOME_FASTA" \
  --model "$MODEL_DIR" \
  --all-windows \
  --out-dir "$OUT_DIR" \
  --prefix NC_000913.2_type1_pre_max29

echo ""
echo "Done: $(date -Is)"
