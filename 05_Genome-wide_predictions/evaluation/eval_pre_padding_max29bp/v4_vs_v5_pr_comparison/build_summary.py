#!/usr/bin/env python3
"""Regenerate best_f1_summary.tsv from v4 and v5 pr_curve_data.tsv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

N_REF = 11834
ROOT = Path(__file__).resolve().parent
GBACTPRO_BH = ROOT.parent.parent.parent  # .../ref_code/GBactpro_BH
V4_TSV = (
    GBACTPRO_BH
    / "04_genome_scanning_v4_all_data_pool/pr_curve_all4classes_dense/output/type1_dense/pr_curve_data.tsv"
)
V5_TSV = (
    GBACTPRO_BH
    / "04_genome_scanning_v5_all_data_pool/eval_pre_padding_max29bp/pr_curve_doublestrand_dense/output_100bp_tolerance/pr_curve_data.tsv"
)


def main():
    rows = []
    for name, path in [("v4_type1_dense_WIG_classic", V4_TSV), ("v5_doublestrand_dense_100bp", V5_TSV)]:
        df = pd.read_csv(path, sep="\t")
        r = df.loc[df["f1_score"].idxmax()]
        rec = float(r["recall"])
        rows.append(
            {
                "pipeline": name,
                "source_pr_curve_tsv": str(path.relative_to(GBACTPRO_BH)),
                "best_f1": float(r["f1_score"]),
                "threshold": float(r["threshold"]),
                "precision": float(r["precision"]),
                "recall": rec,
                "implied_TP_refs": int(round(rec * N_REF)),
                "implied_FN_refs": N_REF - int(round(rec * N_REF)),
                "num_predictions_at_best_f1": int(r["num_predictions"]),
            }
        )
    out = pd.DataFrame(rows)
    out_path = ROOT / "best_f1_summary.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    print(out.to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
