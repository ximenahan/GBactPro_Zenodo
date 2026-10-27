#!/usr/bin/env bash
# Full pipeline: Table S1 compare, part1–2 (same as 11834), part3 v5 one-to-one PR/ROC, part4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT/../../../.." && pwd)"
DEREP_BED="$PROJECT_ROOT/ref_code/GBactPro_new/scans/lstm_scan/hns_knockout_derepressed_promoter_analysis/table_s1_analysis/Table_S1_derepressed_TSS.bed"

echo "========================================================"
echo "  H-NS derepressed analysis — v5 one-to-one (11834 ∪ 371)"
echo "========================================================"
echo "Started: $(date -Is)"
echo ""

if [ ! -f "$DEREP_BED" ]; then
  echo "ERROR: Missing $DEREP_BED"
  exit 1
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda/etc/profile.d/conda.sh"
fi
conda activate gbactpro_env 2>/dev/null || conda activate bioinfo 2>/dev/null || true

echo "=== Table S1 vs 11834 ==="
python3 "$ROOT/table_s1_analysis/compare_tss_11834_vs_table_s1.py" --out-dir "$ROOT/table_s1_analysis"
echo ""

echo "=== Part 1 ==="
python3 "$ROOT/part1_derepressed_overlap/overlap_11834_vs_derepressed.py" --out-dir "$ROOT/part1_derepressed_overlap"
echo ""

echo "=== Part 2 ==="
python3 "$ROOT/part2_tss_union_bed/union_11834_derepressed_bed.py" --out-dir "$ROOT/part2_tss_union_bed"
echo ""

echo "=== Part 3 (v5 PR + ROC, no tol then 100 bp; heavy) ==="
chmod +x "$ROOT/part3_pr_roc_curves/run_pr_and_roc_v5_o2o.sh"
bash "$ROOT/part3_pr_roc_curves/run_pr_and_roc_v5_o2o.sh"
echo ""

echo "=== Part 4 ==="
python3 "$ROOT/part4_derepressed_tss_overlap/analyze_derepressed_tss_overlap_v5.py" --output-dir "$ROOT/part4_derepressed_tss_overlap"
echo ""

echo "========================================================"
echo "  ALL DONE: $(date -Is)"
echo "========================================================"
