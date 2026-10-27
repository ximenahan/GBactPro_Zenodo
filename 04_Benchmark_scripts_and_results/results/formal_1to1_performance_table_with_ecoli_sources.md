# Sources — 4-decimal 1:1 table (E. coli always 467:467)

Rounding: half-up to 4 decimals from full-precision means.

| Row | Ave AUROC/AUPRC (raw → 4dp) | E.coli AUROC/AUPRC (raw → 4dp) | n_pos:n_neg | Source |
|---|---|---|---|---|
| GBactPro Type 1, 43-species train | 0.994256 / 0.993077 → 0.9943 / 0.9931 | 0.996767 / 0.996364 → 0.9968 / 0.9964 | 467:467 | `03_model_training_v3_integrated_data/.../35+s+10/random/result` |
| GBactPro Type 1, E. coli train | 0.985861 / 0.985332 → 0.9859 / 0.9853 | 0.992636 / 0.992077 → 0.9926 / 0.9921 | 467:467 | `03_model_training_v3/.../1-1/35+s+10/random/result` |
| GBactPro Type 8, 43-species train | 0.943134 / 0.945155 → 0.9431 / 0.9452 | 0.955697 / 0.957589 → 0.9557 / 0.9576 | 467:467 | `.../integrated_43/minus60_plus21_tss/random/result` |
| GBactPro Type 8, E. coli train | 0.839791 / 0.828393 → 0.8398 / 0.8284 | 0.868829 / 0.840471 → 0.8688 / 0.8405 | 467:467 | `03_model_training_v3/.../minus60_plus21_tss/random/result` |
| iPro-MP 81 bp, E. coli train | 0.841557 / 0.847243 → 0.8416 / 0.8472 | 0.905919 / 0.917010 → 0.9059 / 0.9170 | 467:467 | `.../iPro-MP/output/cross_species_metrics.csv` |
| Prompt 81 bp, E. coli train | 0.778422 / 0.764463 → 0.7784 / 0.7645 | 0.784982 / 0.756790 → 0.7850 / 0.7568 | 467:467 | `.../Prompt/output/cross_species_metrics.csv` |
| iPro-WAEL 81 bp, E. coli train | 0.629325 / 0.620244 → 0.6293 / 0.6202 | 0.680158 / 0.694522 → 0.6802 / 0.6945 | 467:467 | `.../iPro-WAEL/output/cross_species_metrics.csv` |
| GBactPro Type 7, 43-species train | 0.942649 / 0.949460 → 0.9426 / 0.9495 | 0.956334 / 0.960488 → 0.9563 / 0.9605 | 467:467 | Re-eval of 43-species Type 7 on the same both-strand 1:1 FASTAs as Promotech: `performance_comparison/formal_1to1_type7_40bp_467/gbactpro_43species_trained_type7/` |
| GBactPro Type 7, E. coli train | 0.906382 / 0.910124 → 0.9064 / 0.9101 | 0.928988 / 0.924578 → 0.9290 / 0.9246 | 467:467 | Re-eval of E.coli-trained Type 7 on new both-strand 1:1 test: `performance_comparison/formal_1to1_type7_40bp_467/gbactpro_ecoli_trained_type7/` |
| Promotech 40 bp, 7-species train | 0.875690 / 0.875039 → 0.8757 / 0.8750 | 0.907098 / 0.903679 → 0.9071 / 0.9037 | 467:467 | Re-eval on same both-strand 1:1 Type 7 FASTAs: `performance_comparison/formal_1to1_type7_40bp_467/promotech_40bp_1to1/` |

Type 7 / Promotech 1:1 both-strand FASTAs (E. coli 224+ / 243− = 467 pos):  
`02_dataset_generation_v3_1to10/output/75-25_test_1to1_type6/test/`  
(does not overwrite the 1:10 test set).
