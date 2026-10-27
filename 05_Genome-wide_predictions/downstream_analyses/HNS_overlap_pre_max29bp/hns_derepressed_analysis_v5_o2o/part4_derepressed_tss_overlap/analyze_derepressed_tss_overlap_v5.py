#!/usr/bin/env python3
"""
Part 4 (v5): Overlap of scan windows (score ≥ best-F1 threshold) vs Table_S1 derepressed TSS.

Uses v5 promoter geometry: UPSTREAM_MAX=39, UPSTREAM_MIN=6, tolerance 0 or 100 (strand-aware).
Loads the genome scan TSV once; applies both PR best-F1 thresholds from part 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]

UPSTREAM_MIN = 6
UPSTREAM_MAX = 39

DEFAULT_SCAN = (
    PROJECT_ROOT
    / "ref_code"
    / "GBactpro_BH"
    / "04_genome_scanning_v5_all_data_pool"
    / "output"
    / "doublestrand_intervals_all_pre_max29bp"
    / "NC_000913.tsv"
)
DEREPRESSED_BED = (
    PROJECT_ROOT
    / "ref_code"
    / "GBactPro_new"
    / "scans"
    / "lstm_scan"
    / "hns_knockout_derepressed_promoter_analysis"
    / "table_s1_analysis"
    / "Table_S1_derepressed_TSS.bed"
)


def accepted_region(tss: int, strand_plus: bool, tolerance: int) -> tuple[int, int]:
    if strand_plus:
        reg_s = tss - UPSTREAM_MAX - tolerance
        reg_e = tss - UPSTREAM_MIN + tolerance
    else:
        reg_s = tss + UPSTREAM_MIN - tolerance
        reg_e = tss + UPSTREAM_MAX + tolerance
    return reg_s, reg_e


def intervals_overlap(ws: int, we: int, rs: int, re: int) -> bool:
    return not (we < rs or ws > re)


def load_pr_best_f1(pr_path: Path) -> tuple[float, float, float, float]:
    df = pd.read_csv(pr_path, sep="\t")
    idx = df["f1_score"].idxmax()
    row = df.loc[idx]
    return (
        float(row["threshold"]),
        float(row["precision"]),
        float(row["recall"]),
        float(row["f1_score"]),
    )


def load_derepressed_bed(path: Path) -> tuple[list[int], list[bool]]:
    pos = []
    pl = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) < 6:
                continue
            tss = int(p[1]) + 1
            strand_plus = p[5] == "+"
            pos.append(tss)
            pl.append(strand_plus)
    print(f"Loaded {len(pos):,} derepressed TSS from BED")
    return pos, pl


def run_one_condition(
    starts: np.ndarray,
    ends: np.ndarray,
    regions: list[tuple[int, int]],
    thr: float,
    prec: float,
    rec: float,
    f1: float,
    tolerance: int,
    method_name: str,
) -> dict:
    m = np.zeros(len(starts), dtype=np.bool_)
    # Caller passes pre-filtered arrays; n_pred = len(starts)
    n_pred = len(starts)
    n_tss = len(regions)

    pred_hits = 0
    for i in range(n_pred):
        a, b = int(starts[i]), int(ends[i])
        for (rs, re) in regions:
            if intervals_overlap(a, b, rs, re):
                pred_hits += 1
                break

    tss_hit = 0
    for j in range(n_tss):
        rs, re = regions[j]
        hit = False
        for i in range(n_pred):
            if intervals_overlap(int(starts[i]), int(ends[i]), rs, re):
                hit = True
                break
        if hit:
            tss_hit += 1

    pred_pct = 100.0 * pred_hits / n_pred if n_pred else 0.0
    rec_pct = 100.0 * tss_hit / n_tss if n_tss else 0.0

    print(f"\n{method_name} — tolerance ±{tolerance} bp, thr={thr:.6f}, F1={f1:.4f}")
    print(f"  Windows ≥ threshold: {n_pred:,}")
    print(f"  Preds hitting derepressed region: {pred_hits:,} ({pred_pct:.2f}%)")
    print(f"  Derepressed TSS recovered:        {tss_hit:,} / {n_tss:,} ({rec_pct:.2f}%)")

    return {
        "method": method_name,
        "tolerance_bp": tolerance,
        "threshold": thr,
        "precision_at_f1": prec,
        "recall_at_f1": rec,
        "f1_score": f1,
        "n_predictions_windows": n_pred,
        "n_derepressed_tss": n_tss,
        "predictions_hitting": pred_hits,
        "prediction_hit_percentage": pred_pct,
        "tss_hit": tss_hit,
        "tss_recovery_percentage": rec_pct,
    }


def save_results(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "derepressed_tss_overlap_summary.txt"
    with open(summary, "w") as f:
        f.write("DEREPRESSED TSS OVERLAP — v5 double-strand windows @ best F1 (one-to-one PR)\n")
        f.write("=" * 72 + "\n\n")
        for r in rows:
            f.write(f"Condition: {r['method']} (±{r['tolerance_bp']} bp)\n")
            f.write(f"Threshold: {r['threshold']:.6f}  F1: {r['f1_score']:.4f}\n")
            f.write(f"Windows ≥ threshold: {r['n_predictions_windows']:,}\n")
            f.write(f"Derepressed TSS:     {r['n_derepressed_tss']:,}\n")
            f.write(f"Pred hit rate: {r['prediction_hit_percentage']:.2f}%\n")
            f.write(f"TSS recovery:  {r['tss_recovery_percentage']:.2f}%\n\n")
    pd.DataFrame(rows).to_csv(out_dir / "derepressed_tss_overlap_data.tsv", sep="\t", index=False)
    print(f"\nWrote {summary} and derepressed_tss_overlap_data.tsv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-tsv", type=Path, default=DEFAULT_SCAN)
    ap.add_argument("--derepressed-bed", type=Path, default=DEREPRESSED_BED)
    ap.add_argument(
        "--pr-no-tol",
        type=Path,
        default=SCRIPT_DIR.parent / "part3_pr_roc_curves" / "output_no_tolerance" / "pr" / "pr_curve_data.tsv",
    )
    ap.add_argument(
        "--pr-100",
        type=Path,
        default=SCRIPT_DIR.parent / "part3_pr_roc_curves" / "output_100bp_tolerance" / "pr" / "pr_curve_data.tsv",
    )
    ap.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    args = ap.parse_args()

    for p in (args.pr_no_tol, args.pr_100, args.scan_tsv, args.derepressed_bed):
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    thr0, p0, r0, f10 = load_pr_best_f1(args.pr_no_tol)
    thr100, p1, r1, f11 = load_pr_best_f1(args.pr_100)

    print(f"Loading full scan TSV: {args.scan_tsv}")
    df = pd.read_csv(
        args.scan_tsv,
        sep="\t",
        usecols=["start_1based", "end_1based_inclusive", "strand", "score"],
    )
    scores = df["score"].values
    starts_all = df["start_1based"].values.astype(np.int64)
    ends_all = df["end_1based_inclusive"].values.astype(np.int64)

    tss_pos, tss_plus = load_derepressed_bed(args.derepressed_bed)
    regions0 = [accepted_region(t, tp, 0) for t, tp in zip(tss_pos, tss_plus)]
    regions100 = [accepted_region(t, tp, 100) for t, tp in zip(tss_pos, tss_plus)]

    rows = []

    m0 = scores >= thr0
    rows.append(
        run_one_condition(
            starts_all[m0],
            ends_all[m0],
            regions0,
            thr0,
            p0,
            r0,
            f10,
            0,
            "v5_o2o_union_ref_no_tolerance",
        )
    )

    m1 = scores >= thr100
    rows.append(
        run_one_condition(
            starts_all[m1],
            ends_all[m1],
            regions100,
            thr100,
            p1,
            r1,
            f11,
            100,
            "v5_o2o_union_ref_100bp_tolerance",
        )
    )

    save_results(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
