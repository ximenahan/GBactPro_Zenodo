## Random-score baseline (dense hybrid PR)

- **Scores**: i.i.d. Uniform(0,1) per window; **geometry unchanged** (strand-aware promoter boxes, 29 bp full containment for window TP labels).
- **Hybrid PR** (same as `generate_pr_curve_doublestrand.py`): precision from **window** TP in top-k; recall from **per-TSS max** score ≥ threshold.
- **Null interpretation**: ranking carries no signal; expected AUPRC / best-F1 / FP rates.

Compare to the real curve: `observed_auprc_reference.txt` (from `../pr_curve_data.tsv`).

Simulations: **1000**, seed **42**, tolerance **0** bp.

Outputs: `random_baseline_per_sim.tsv`, `random_baseline_summary.tsv`.
