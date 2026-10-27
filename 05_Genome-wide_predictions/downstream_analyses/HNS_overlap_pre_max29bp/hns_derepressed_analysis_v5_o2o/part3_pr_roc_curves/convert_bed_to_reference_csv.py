#!/usr/bin/env python3
"""Union BED → CSV with Pos,Strand for v5 one-to-one PR/ROC scripts."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="6-column BED (union TSS)")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV Pos,Strand")
    args = ap.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) < 6:
                continue
            start0 = int(p[1])
            strand = p[5]
            pos_1based = start0 + 1
            rows.append({"Pos": pos_1based, "Strand": strand})

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
