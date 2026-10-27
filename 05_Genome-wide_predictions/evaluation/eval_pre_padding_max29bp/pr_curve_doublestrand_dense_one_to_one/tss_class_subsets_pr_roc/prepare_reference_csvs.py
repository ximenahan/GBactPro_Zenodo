#!/usr/bin/env python3
"""Build Pos,Strand CSVs for one-to-one PR/ROC from v4 subset TSVs (genome, loc)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MASTER_CSV = Path(
    "/home/brucehan/Project_02_promoter_prediction_model/ref_code/GBactPro_new/data/"
    "tss_class_subsets/ecoli_all4classes_tss_filtered.csv"
)
CLASSES = ("primary", "secondary", "internal", "antisense")


def main() -> None:
    base = Path(__file__).resolve().parent
    ref_tsv = base / "references"
    out_dir = base / "references_pos_strand"
    out_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_CSV)
    if "Pos" not in master.columns or "Strand" not in master.columns:
        raise SystemExit(f"Unexpected master columns: {master.columns.tolist()}")

    for name in CLASSES:
        tsv_path = ref_tsv / f"{name}.tsv"
        sub = pd.read_csv(tsv_path, sep="\t")
        if "loc" not in sub.columns:
            raise SystemExit(f"{tsv_path}: expected column 'loc'")
        pos = sub.rename(columns={"loc": "Pos"})[["Pos"]].drop_duplicates()
        merged = master.merge(pos, on="Pos", how="inner")
        merged = merged.drop_duplicates(subset=["Pos"], keep="first")
        out_path = out_dir / f"{name}_pos_strand.csv"
        merged[["Pos", "Strand"]].to_csv(out_path, index=False)
        print(f"{name}: {len(sub)} TSV rows, {len(pos)} unique Pos, {len(merged)} refs (deduped) -> {out_path}")


if __name__ == "__main__":
    main()
