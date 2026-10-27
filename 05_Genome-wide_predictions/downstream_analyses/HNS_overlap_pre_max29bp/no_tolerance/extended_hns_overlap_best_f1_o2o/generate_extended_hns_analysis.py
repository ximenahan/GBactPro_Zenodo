#!/usr/bin/env python3
"""
Extended H-NS overlap at v5 one-to-one best F1 (no tolerance), same geometry as
`../bar_chart/generate_hns_bar_chart.py`:
  • TP / FP: window center vs H-NS BED (inclusive start/end)
  • FN: reference TSS position vs H-NS BED

Adds:
  1) Fraction of genome (1-based length) covered by union of H-NS intervals
  2) Overlap rate on TP+FN (reference-side composite: TP centers + FN TSS)
  3) Overlap rate on TP+FP (all top-k prediction window centers)
  4) Stacked bar chart: TP, FP, FN, TP+FN, TP+FP + genome coverage in text box
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_bar_chart_module():
    here = Path(__file__).resolve().parent
    p = here.parent / "bar_chart" / "generate_hns_bar_chart.py"
    if not p.is_file():
        raise FileNotFoundError(f"Expected bar chart script at {p}")
    spec = importlib.util.spec_from_file_location("hns_bar_chart", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def genome_length_from_fasta(fasta_path: str) -> int:
    n = 0
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            n += len(line.strip())
    return n


def merged_inclusive_bp(intervals: list[tuple[int, int]]) -> int:
    """Sum of bp covered by union of closed intervals [s, e]."""
    if not intervals:
        return 0
    iv = sorted(intervals)
    cs, ce = iv[0]
    total = 0
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs + 1
            cs, ce = s, e
    total += ce - cs + 1
    return total


def check_hns_overlap(positions: list[int], hns_sites: list[tuple[int, int]]) -> tuple[int, float]:
    if not positions:
        return 0, 0.0
    overlap_count = 0
    for pos in positions:
        for start, end in hns_sites:
            if start <= pos <= end:
                overlap_count += 1
                break
    return overlap_count, overlap_count / len(positions)


def create_extended_bar_chart(
    rows: list[tuple[str, int, int, float]],
    genome_covered_bp: int,
    genome_total_bp: int,
    genome_frac: float,
    output_file: str,
    title: str,
):
    categories = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    hns_counts = [r[2] for r in rows]
    hns_ratios = [r[3] for r in rows]
    non_hns_counts = [c - h for c, h in zip(counts, hns_counts)]

    color_hns = "#2E7D32"
    color_non_hns = "#A5D6A7"

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(categories))
    bar_width = 0.65

    bars_hns = ax.bar(
        x, hns_counts, bar_width, label="H-NS+ (overlap)", color=color_hns, edgecolor="black", linewidth=1.2
    )
    ax.bar(
        x,
        non_hns_counts,
        bar_width,
        bottom=hns_counts,
        label="H-NS- (no overlap)",
        color=color_non_hns,
        edgecolor="black",
        linewidth=1.2,
    )

    for i, (bar, ratio, hns_c, total_c) in enumerate(zip(bars_hns, hns_ratios, hns_counts, counts)):
        bar_height = bar.get_height()
        if bar_height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar_height / 2,
                f"{ratio * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white",
            )

    ymax = max(counts) if counts and max(counts) > 0 else 1
    for i, total in enumerate(counts):
        ax.text(
            x[i],
            total + ymax * 0.015,
            f"n={total:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: format(int(v), ",")))
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    gtxt = (
        f"Genome (NC_000913.2): H-NS covers {genome_covered_bp:,} / {genome_total_bp:,} bp "
        f"({genome_frac * 100:.3f}% of genome)\n"
        "Union of H-NS BED intervals (inclusive); genome length from scan FASTA."
    )
    ax.text(
        0.02,
        0.98,
        gtxt,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    highres = output_file.replace(".png", "_highres.png")
    plt.savefig(highres, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    print(f"  Saved: {highres}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-curve-tsv", required=True)
    ap.add_argument("--predictions-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--hns-bed", required=True)
    ap.add_argument("--genome-fasta", required=True, help="Same genome as v5 scan (for total bp).")
    ap.add_argument("--match-tolerance", type=int, default=0)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    gmod = _load_bar_chart_module()
    thr, best_f1, k = gmod.find_best_f1_row(args.pr_curve_tsv)

    t0 = time.time()
    starts, ends, is_plus, scores = gmod.load_predictions(args.predictions_tsv)
    ref_pos, ref_plus = gmod.load_reference_tss(args.reference_csv)
    hns_sites = gmod.load_hns_binding_sites(args.hns_bed)
    genome_bp = genome_length_from_fasta(args.genome_fasta)
    merged_bp = merged_inclusive_bp(hns_sites)
    genome_frac = merged_bp / genome_bp if genome_bp else 0.0

    print(f"\nGenome length (FASTA): {genome_bp:,} bp")
    print(f"H-NS union (inclusive intervals): {merged_bp:,} bp ({100 * genome_frac:.4f}% of genome)")

    tol = args.match_tolerance
    print(f"\nGreedy 1:1 matching: tolerance={tol} bp, k={k:,} ...")
    geom_ok = gmod.compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tol)
    tp_c, fp_c, fn_p = gmod.greedy_o2o_topk(
        starts, ends, is_plus, scores, ref_pos, ref_plus, tol, geom_ok, k
    )
    print(f"  TP: {len(tp_c):,}  FP: {len(fp_c):,}  FN: {len(fn_p):,}  ({time.time() - t0:.1f}s)")

    tp_h, tp_r = check_hns_overlap(tp_c, hns_sites)
    fp_h, fp_r = check_hns_overlap(fp_c, hns_sites)
    fn_h, fn_r = check_hns_overlap(fn_p, hns_sites)

    tp_fn = tp_c + fn_p
    tp_fn_h, tp_fn_r = check_hns_overlap(tp_fn, hns_sites)
    tp_fp = tp_c + fp_c
    tp_fp_h, tp_fp_r = check_hns_overlap(tp_fp, hns_sites)

    print("\nAggregate overlap (same point rules as bar chart):")
    print(f"  TP+FN (n={len(tp_fn):,}): H-NS+ {tp_fn_h:,}  ({100 * tp_fn_r:.2f}%)")
    print(f"  TP+FP (n={len(tp_fp):,}): H-NS+ {tp_fp_h:,}  ({100 * tp_fp_r:.2f}%)")

    label_tol = "no tolerance" if tol == 0 else f"±{tol} bp tolerance"
    title = (
        f"H-NS overlap — v5 one-to-one best F1 + aggregates\n"
        f"({label_tol}, F1={best_f1:.4f}, thr={thr:.4f}, k={k:,}; "
        f"TP/FP: window center; FN: TSS)"
    )

    rows = [
        ("TP\n(True Positives)", len(tp_c), tp_h, tp_r),
        ("FP\n(False Positives)", len(fp_c), fp_h, fp_r),
        ("FN\n(False Negatives)", len(fn_p), fn_h, fn_r),
        ("TP+FN\n(ref-side)", len(tp_fn), tp_fn_h, tp_fn_r),
        ("TP+FP\n(top-k preds)", len(tp_fp), tp_fp_h, tp_fp_r),
    ]
    png = os.path.join(out_dir, "hns_extended_overlap_bar_chart.png")
    create_extended_bar_chart(rows, merged_bp, genome_bp, genome_frac, png, title)

    summ = pd.DataFrame(
        [
            {
                "metric": "genome_bp_hns_union",
                "fraction": genome_frac,
                "numerator": merged_bp,
                "denominator": genome_bp,
                "unit": "bp_union_over_genome_bp",
            },
            {
                "metric": "tp_fn_hns_overlap_rate",
                "fraction": tp_fn_r,
                "numerator": tp_fn_h,
                "denominator": len(tp_fn),
                "unit": "sites_overlapping_hns_over_total_sites",
            },
            {
                "metric": "tp_fp_hns_overlap_rate",
                "fraction": tp_fp_r,
                "numerator": tp_fp_h,
                "denominator": len(tp_fp),
                "unit": "sites_overlapping_hns_over_total_sites",
            },
        ]
    )
    per_cat = pd.DataFrame(
        {
            "category": ["TP", "FP", "FN", "TP+FN", "TP+FP"],
            "count": [len(tp_c), len(fp_c), len(fn_p), len(tp_fn), len(tp_fp)],
            "hns_overlap_count": [tp_h, fp_h, fn_h, tp_fn_h, tp_fp_h],
            "hns_overlap_ratio": [tp_r, fp_r, fn_r, tp_fn_r, tp_fp_r],
        }
    )
    summ_path = os.path.join(out_dir, "hns_aggregate_overlap_summary.tsv")
    per_path = os.path.join(out_dir, "hns_extended_overlap_by_category.tsv")
    summ.to_csv(summ_path, sep="\t", index=False)
    per_cat.to_csv(per_path, sep="\t", index=False)
    print(f"  Wrote: {summ_path}")
    print(f"  Wrote: {per_path}")

    meta = os.path.join(out_dir, "run_metadata.txt")
    with open(meta, "w") as f:
        f.write(f"pr_curve_tsv\t{args.pr_curve_tsv}\n")
        f.write(f"predictions_tsv\t{args.predictions_tsv}\n")
        f.write(f"reference_csv\t{args.reference_csv}\n")
        f.write(f"hns_bed\t{args.hns_bed}\n")
        f.write(f"genome_fasta\t{args.genome_fasta}\n")
        f.write(f"match_tolerance_bp\t{tol}\n")
        f.write(f"best_f1\t{best_f1}\n")
        f.write(f"threshold\t{thr}\n")
        f.write(f"num_predictions_k\t{k}\n")
    print(f"  Wrote: {meta}")


if __name__ == "__main__":
    main()
