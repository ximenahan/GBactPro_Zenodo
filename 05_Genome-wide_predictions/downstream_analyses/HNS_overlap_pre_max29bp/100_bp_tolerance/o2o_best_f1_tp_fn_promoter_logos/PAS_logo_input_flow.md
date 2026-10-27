# PAS → logos: v5 O2O TP vs FN reference TSS (100 bp tol, best F1)

```mermaid
flowchart TD
  A["ecoli_all4classes_tss_filtered.csv<br/>+ NC_000913.tsv predictions<br/>+ pr_curve_data.tsv best F1 k"]
  B["export_tp_fn_subsets.py<br/>v5 greedy 1:1 top-k matching"]
  C["ecoli_tp_tss_subset.csv<br/>ecoli_fn_tss_subset.csv"]
  D["scan_tss_subset.py × 2<br/>BPM/PAS Params_Con17.pkl"]
  E["scan TSVs<br/>Minus35, Spacer, Minus10, …"]
  F["generate_logos.py<br/>logomaker information logos"]
  G["sequence_logos/output/tp/<br/>sequence_logos/output/fn/"]

  A --> B --> C --> D --> E --> F --> G
```

- **PAS/BPM:** runs only in `promoter_architecture_scan` (same BPM bundle as v4 type-1 class workflow).
- **Logos:** `sequence_logos` reads BPM output columns only; no neural scan.

Parent H-NS bar chart (unchanged): `bar_chart/generate_hns_bar_chart.py` uses the same matching rule for overlap plots; this folder reuses that definition in `matching_v5_o2o.py` for subsetting only.
