#!/usr/bin/env python3
"""
Peak-calling: convert window-level scores to event-level peaks.

For each strand independently:
  1. Build a dense score array indexed by window start position.
  2. Use scipy.signal.find_peaks with distance=29 (one window width)
     to find local maxima at least 29 bp apart.
  3. Each peak = one predicted event with a position and score.

This collapses ~4.6 M windows/strand down to ~150 K peaks/strand,
ensuring overlapping windows near the same TSS are merged into ONE event.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

WINDOW_LEN = 29


def main():
    ap = argparse.ArgumentParser(description="Call peaks from all-windows TSV")
    ap.add_argument("--input-tsv", required=True, help="All-windows TSV")
    ap.add_argument("--output-tsv", required=True, help="Output peaks TSV")
    ap.add_argument("--distance", type=int, default=WINDOW_LEN,
                    help="Minimum distance between peaks (default: 29)")
    args = ap.parse_args()

    print("=" * 70)
    print("  PEAK CALLING (window → event)")
    print("=" * 70)
    print(f"  Input:    {args.input_tsv}")
    print(f"  Output:   {args.output_tsv}")
    print(f"  Distance: {args.distance} bp")
    print()

    t0 = time.time()
    print("Loading windows ...")
    df = pd.read_csv(
        args.input_tsv, sep="\t",
        usecols=["chrom", "start_0based", "start_1based", "end_1based_inclusive",
                 "strand", "score"],
        dtype={"start_0based": np.int64, "start_1based": np.int64,
               "end_1based_inclusive": np.int64, "score": np.float64},
    )
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    chrom_val = df["chrom"].iloc[0]
    peaks_records: list[dict] = []

    for strand_label in ["+", "-"]:
        sdf = df[df["strand"] == strand_label].sort_values("start_0based")
        scores = sdf["score"].values
        positions_0b = sdf["start_0based"].values
        positions_1b = sdf["start_1based"].values
        ends_1b = sdf["end_1based_inclusive"].values

        print(f"  [{strand_label}] {len(scores):,} windows, calling peaks (distance={args.distance}) ...")
        peak_idx, props = find_peaks(scores, distance=args.distance)
        print(f"  [{strand_label}] {len(peak_idx):,} peaks found")

        for i in peak_idx:
            peaks_records.append({
                "chrom": chrom_val,
                "peak_start_1based": int(positions_1b[i]),
                "peak_end_1based": int(ends_1b[i]),
                "strand": strand_label,
                "score": float(scores[i]),
                "window_index_0based": int(positions_0b[i]),
            })

    peaks_df = pd.DataFrame(peaks_records)
    Path(args.output_tsv).parent.mkdir(parents=True, exist_ok=True)
    peaks_df.to_csv(args.output_tsv, sep="\t", index=False)

    elapsed = time.time() - t0
    print(f"\n  Total peaks: {len(peaks_df):,}")
    print(f"    + strand: {(peaks_df['strand']=='+').sum():,}")
    print(f"    - strand: {(peaks_df['strand']=='-').sum():,}")
    print(f"  Score range: [{peaks_df['score'].min():.6f}, {peaks_df['score'].max():.6f}]")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Saved: {args.output_tsv}")


if __name__ == "__main__":
    main()
