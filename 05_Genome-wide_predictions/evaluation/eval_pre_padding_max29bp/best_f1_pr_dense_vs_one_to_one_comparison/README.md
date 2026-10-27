# Best F1 comparison: dense vs one-to-one PR (pre-padding max29bp)

Reference TSS count: **11834** (`ecoli_all4classes_tss_filtered.csv`: **5,923** `+` TSS, **5,911** `−` TSS).

## Precision / recall / F1 at best-F1 operating point (all four runs)

Values come from the row with maximum `f1_score` in each `pr_curve_data.tsv`.

| method | tolerance | threshold | precision | recall | F1 | k (preds ≥ thr) |
|--------|-----------|-----------|-----------|--------|-----|-----------------|
| dense | no_tolerance | 0.931509 | 0.211619 | 0.270407 | 0.237428 | 15027 |
| dense | 100bp_tolerance | 0.857844 | 0.329513 | 0.803870 | 0.467425 | 42669 |
| one_to_one | no_tolerance | 0.931509 | 0.210887 | 0.267788 | 0.235955 | 15027 |
| one_to_one | 100bp_tolerance | 0.918698 | 0.300939 | 0.495606 | 0.374485 | 19489 |

## Strand breakdown (+ / −)

Run **`python3 compute_strand_breakdown_all.py`** (loads ~9.3M scan rows once) to regenerate:

- `best_f1_master_with_strand.tsv` — one row per setting: precision/recall/F1, k, and TP/FP/FN split by strand.
- `strand_distribution_best_f1_all.tsv` — long format (`method`, `tolerance_label`, `metric`, `strand`, `count`).

**Dense:** `TP_win_*` / `FP_win_*` = strand of the **predicted window**; `TP_ref_*` / `FN_ref_*` = strand of the **reference TSS** (recall axis).

**One-to-one:** `TP_match_ref_*` = strand of **matched reference**; `FP_pred_*` = strand of **prediction** without match; `FN_ref_*_o2o` = **unmatched** reference strand.

## Definitions

### Dense (`pr_curve_doublestrand_dense`)
- **TP_windows / FP_windows**: Among the **top-k** predictions by score (k = `n_predictions_top_k`), windows that lie fully inside a true promoter (strand-aware) vs not.
- **TP_references / FN_references**: **Recall axis** — TSS with at least one window in their promoter region with score ≥ threshold (max per TSS). This is **not** the same count as TP_windows.

### One-to-one (`pr_curve_doublestrand_dense_one_to_one`)
- **TP_matched_pairs**: Greedy matching in score order; each TSS matched at most once.
- **FP**: Top-k predictions that did not receive a match.
- **FN**: TSS never matched.

## Dense — best F1 per tolerance (aggregated counts)

| method   | tolerance_label   | source_tsv                                                           |   threshold |       f1 |   precision |   recall |   n_predictions_top_k |   TP_windows_in_top_k |   FP_windows_in_top_k |   TP_references_detected_recall_axis |   FN_references_missed_recall_axis |
|:---------|:------------------|:---------------------------------------------------------------------|------------:|---------:|------------:|---------:|----------------------:|----------------------:|----------------------:|-------------------------------------:|-----------------------------------:|
| dense    | no_tolerance      | pr_curve_doublestrand_dense/output_no_tolerance/pr_curve_data.tsv    |    0.931509 | 0.237428 |    0.211619 | 0.270407 |                 15027 |                  3180 |                 11847 |                                 3200 |                               8634 |
| dense    | 100bp_tolerance   | pr_curve_doublestrand_dense/output_100bp_tolerance/pr_curve_data.tsv |    0.857844 | 0.467425 |    0.329513 | 0.80387  |                 42669 |                 14060 |                 28609 |                                 9513 |                               2321 |

## One-to-one — best F1 per tolerance (aggregated counts)

| method     | tolerance_label   | source_tsv                                                                      |   threshold |       f1 |   precision |   recall |   n_predictions_top_k |   TP_matched_pairs |   FP_predictions_no_match |   FN_references_unmatched |
|:-----------|:------------------|:--------------------------------------------------------------------------------|------------:|---------:|------------:|---------:|----------------------:|-------------------:|--------------------------:|--------------------------:|
| one_to_one | no_tolerance      | pr_curve_doublestrand_dense_one_to_one/output_no_tolerance/pr_curve_data.tsv    |    0.931509 | 0.235955 |    0.210887 | 0.267788 |                 15027 |               3169 |                     11858 |                      8665 |
| one_to_one | 100bp_tolerance   | pr_curve_doublestrand_dense_one_to_one/output_100bp_tolerance/pr_curve_data.tsv |    0.918698 | 0.374485 |    0.300939 | 0.495606 |                 19489 |               5865 |                     13624 |                      5969 |

## Files

- `best_f1_metrics.tsv` — combined table (same as dense/o2o sections above).
- `best_f1_comparison_chart.png` — bar chart from `build_best_f1_comparison.py`.
- `best_f1_master_with_strand.tsv`, `strand_distribution_best_f1_all.tsv` — from `compute_strand_breakdown_all.py`.

## Strand distribution snapshot (from `compute_strand_breakdown_all.py`)

| method     | tolerance_label   |   tolerance_bp |   threshold |   precision_at_best_f1 |   recall_at_best_f1 |   f1_at_best_f1 |   n_predictions_top_k |   n_ref_plus |   n_ref_minus | TP_win_plus   | TP_win_minus   | FP_win_plus   | FP_win_minus   | TP_ref_plus   | TP_ref_minus   | FN_ref_plus   | FN_ref_minus   | TP_match_ref_plus   | TP_match_ref_minus   | FP_pred_plus   | FP_pred_minus   | FN_ref_plus_o2o   | FN_ref_minus_o2o   |   k_recomputed |
|:-----------|:------------------|---------------:|------------:|-----------------------:|--------------------:|----------------:|----------------------:|-------------:|--------------:|:--------------|:---------------|:--------------|:---------------|:--------------|:---------------|:--------------|:---------------|:--------------------|:---------------------|:---------------|:----------------|:------------------|:-------------------|---------------:|
| dense      | no_tolerance      |              0 |    0.931509 |               0.211619 |            0.270407 |        0.237428 |                 15027 |         5923 |          5911 | 1639          | 1541           | 6017          | 5830           | 1649          | 1551           | 4274          | 4360           |                     |                      |                |                 |                   |                    |          15027 |
| dense      | 100bp_tolerance   |            100 |    0.857844 |               0.329513 |            0.80387  |        0.467425 |                 42669 |         5923 |          5911 | 7046          | 7014           | 14471         | 14138          | 4785          | 4728           | 1138          | 1183           |                     |                      |                |                 |                   |                    |          42669 |
| one_to_one | no_tolerance      |              0 |    0.931509 |               0.210887 |            0.267788 |        0.235955 |                 15027 |         5923 |          5911 |               |                |               |                |               |                |               |                | 1634                | 1535                 | 6022           | 5836            | 4289              | 4376               |          15027 |
| one_to_one | 100bp_tolerance   |            100 |    0.918698 |               0.300939 |            0.495606 |        0.374485 |                 19489 |         5923 |          5911 |               |                |               |                |               |                |               |                | 2922                | 2943                 | 6948           | 6676            | 3001              | 2968               |          19489 |
