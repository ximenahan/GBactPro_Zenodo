## Random-order baseline (one-to-one PR)

- **Scores**: i.i.d. Uniform(0,1) on each scan row (same `start`, `end`, `strand` as the real model).
- **Pipeline**: identical to `generate_pr_curve_doublestrand_one_to_one.py` (geom_ok, greedy 1:1 matching, same percentile + upper-tail thresholds).
- **Interpretation**: expected PR / AUPRC when **ranking** of windows carries no signal; strands, tolerances, and discrete window layout are preserved.

- **Simulations**: 1000, seed=42
- **Tolerance**: 100 bp

Files: `random_baseline_per_sim.tsv`, `random_baseline_summary.tsv`.
