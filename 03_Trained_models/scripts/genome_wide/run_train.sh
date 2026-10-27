#!/usr/bin/env bash
# Train type1 (35+s+10) with pre padding, 29 bp input, ≤29 bp raw sequences only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_train_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================="
echo "type1_35s10 PRE padding max29bp — train"
echo "Started: $(date -Is)"
echo "============================================="

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TF_FORCE_GPU_ALLOW_GROWTH=true
TF_NVIDIA_LIBS=$(python3 - <<'PY'
import glob, os, site
site_pkgs = site.getsitepackages()[0]
lib_dirs = sorted(set(glob.glob(os.path.join(site_pkgs, "nvidia", "**", "lib"), recursive=True)))
print(":".join(lib_dirs))
PY
)
if [ -n "$TF_NVIDIA_LIBS" ]; then
  export LD_LIBRARY_PATH="${TF_NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

GBH="$(cd "$ROOT/../.." && pwd)"
MODEL_DIR="$GBH/03_model_training_v3_genome_use/scripts/models/lstm/genome_use_49/35+s+10_pre_max29bp/random"
echo "Target model dir: $MODEL_DIR"
rm -rf "$MODEL_DIR"

python3 "$ROOT/train_type1_pre_padding_max29bp.py" random

echo ""
echo "============================================="
echo "Finished: $(date -Is)"
echo "Model: $MODEL_DIR"
echo "============================================="
