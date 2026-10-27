# 04 — Benchmark scripts and results

The comparison is a 1:1 test (equal positives and genomic-random negatives) on the 43-species 75/25 split. Means are unweighted averages across those 43 species. *E. coli* is always 467 positives and 467 negatives. Values in the summary table are rounded half-up to 4 decimals from the raw means.

`results/formal_1to1_performance_table_with_ecoli.csv` is the source table. It was not edited to match any external summary. Every AUROC and AUPRC cell matches the manuscript reference. One label does not: Promotech’s training set is “7 cross species” in this file. A requested summary says 9 species. The file was left as stored.

The mean of the 43 published species in `prediction_scores/GBactPro_Type1_43species/auc.csv` is 0.99425584, which rounds to the table’s 0.9943. The *E. coli* value in that file is 0.99676737, which rounds to 0.9968. The same check holds for the other GBactPro rows cited in `formal_1to1_performance_table_with_ecoli_sources.md`.

## Where each row comes from

| Row | Metrics and scores |
|---|---|
| GBactPro Type 1, 43 species | `prediction_scores/GBactPro_Type1_43species/` |
| GBactPro Type 1, *E. coli* | `prediction_scores/GBactPro_Type1_Ecoli/` |
| GBactPro Type 8, 43 species | `prediction_scores/GBactPro_Type8_43species/` |
| GBactPro Type 8, *E. coli* | `prediction_scores/GBactPro_Type8_Ecoli/` |
| GBactPro Type 7, 43 species | `prediction_scores/GBactPro_Type7_43species_1to1_reeval/` |
| GBactPro Type 7, *E. coli* | `prediction_scores/GBactPro_Type7_Ecoli_1to1_reeval/` |
| iPro-MP, Prompt, iPro-WAEL | `results/iPro-MP_81bp_Ecoli/`, `Prompt_81bp_Ecoli/`, `iPro-WAEL_81bp_Ecoli/` |
| Promotech | `results/Promotech_40bp_1to1/` |

Type 7 scores are the later 1:1 re-evaluation. The original score directory next to the Type 7 checkpoint does not reproduce the published *E. coli* AUROC (0.9586 stored there versus 0.9563 in the table) and was not used.

## Scores and labels

Species directories under `prediction_scores/` use underscores in place of spaces (`Escherichia_coli_MG1655`). `species_43.tsv` is the spaced-name crosswalk.

GBactPro `sequences_predictions.csv` has one probability per line and no header. Positives are written first, then negatives, which is the order in `train_lstm_model.py`. For the archived score files the two blocks are the same length, so the label vector is `n_pos` ones followed by `n_pos` zeros. *E. coli* Type 1 has 934 scores (467 + 467). `performance.txt` in the same folder records accuracy, AUROC, and the confusion matrix.

AUROC in the training script is `sklearn.metrics.roc_auc_score` / `auc(fpr, tpr)`. AUPRC for the formal table was computed in the evaluation scripts under `scripts/` (average precision on the same scores and labels).

The 1:10 test FASTA in section 02 is longer than these score files for Types 1 and 8. Do not pair those 1:10 negatives with these scores. Type 7 has a matching 1:1 FASTA in section 02.

iPro-MP, Prompt, iPro-WAEL, and Promotech retained per-species AUROC, AUPRC, and class counts (`cross_species_metrics.csv`). Per-sequence scores were not found and were not invented. Their model weights were not copied. Evaluation wrappers are in `scripts/`. They import third-party code from the local `ref_code/model_comparision` tree and call the original tools; those upstream repositories should be obtained from their authors.

`auc.csv` files copied from the Type 1 and Type 8 result directories still contain six species outside the published 43, because that is how the summary was written. The published mean uses only the 43 names in `species_43.tsv`. The per-species score directories were limited to those 43.
