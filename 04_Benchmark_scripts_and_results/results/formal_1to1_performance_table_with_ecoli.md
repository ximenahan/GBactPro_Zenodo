# Formal 1:1 test performance table (with E. coli columns)

Evaluation: test **pos:neg = 1:1**. All E. coli tests are **467:467**. GBactPro train/test split = **75% / 25%**. Values rounded to **4 decimals**.

- **Species data** = training data.
- **Ave. AUROC / Ave. AUPRC** = unweighted mean over the 43-species 1:1 test sets.
- **E.coli AUROC / E.coli AUPRC** = *E. coli* MG1655 1:1 test (467 pos / 467 neg).

| Tool | Input length | Species data | Ave. AUROC | Ave. AUPRC | E.coli AUROC | E.coli AUPRC |
|---|---|---|---:|---:|---:|---:|
| GBactPro (optimal) | 27–31 bp (Type 1) | 43 cross species | 0.9943 | 0.9931 | 0.9968 | 0.9964 |
| GBactPro | 27–31 bp (Type 1) | E.coli | 0.9859 | 0.9853 | 0.9926 | 0.9921 |
| GBactPro | 81 bp (Type 8) | 43 cross species | 0.9431 | 0.9452 | 0.9557 | 0.9576 |
| GBactPro | 81 bp (Type 8) | E.coli | 0.8398 | 0.8284 | 0.8688 | 0.8405 |
| iPro-MP | 81 bp | E.coli | 0.8416 | 0.8472 | 0.9059 | 0.9170 |
| Prompt | 81 bp | E.coli | 0.7784 | 0.7645 | 0.7850 | 0.7568 |
| iPro-WAEL | 81 bp | E.coli | 0.6293 | 0.6202 | 0.6802 | 0.6945 |
| GBactPro | 40 bp (Type 7) | 43 cross species | 0.9426 | 0.9495 | 0.9563 | 0.9605 |
| GBactPro | 40 bp (Type 7) | E.coli | 0.9064 | 0.9101 | 0.9290 | 0.9246 |
| Promotech | 40 bp | 7 cross species | 0.8757 | 0.8750 | 0.9071 | 0.9037 |

Type numbering (renumbered): Type 1 = 35+s+10 (27–31 bp); Type 7 = 40 bp upstream TSS; Type 8 = −60 to +21 TSS (81 bp).
