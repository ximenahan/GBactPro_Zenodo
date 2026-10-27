#!/usr/bin/env python3
"""
H-NS overlap bar chart at the best-F1 operating point of v5 **one-to-one** PR
(`pr_curve_doublestrand_dense_one_to_one`).

TP / FP / FN follow the same greedy window–reference matching as
`generate_pr_curve_doublestrand_one_to_one.py` (29 bp windows, strand-aware,
promoter [TSS-39, TSS-6] with optional ±tolerance). Overlap uses window centers
for TP/FP and TSS positions for FN, checked against H-NS BED (same rule as the
legacy WIG bar chart script).
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UPSTREAM_MIN = 6
UPSTREAM_MAX = 39
WINDOW_LEN = 29


def load_predictions(tsv_path: str):
    print(f"Loading predictions: {tsv_path} ...")
    t0 = time.time()
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=["start_1based", "end_1based_inclusive", "strand", "score"],
        dtype={
            "start_1based": np.int64,
            "end_1based_inclusive": np.int64,
            "score": np.float64,
        },
    )
    print(f"  Loaded {len(df):,} rows in {time.time() - t0:.1f}s")
    starts = df["start_1based"].values
    ends = df["end_1based_inclusive"].values
    is_plus = df["strand"].values == "+"
    scores = df["score"].values
    return starts, ends, is_plus, scores


def load_reference_tss(csv_path: str):
    print(f"Loading reference TSS: {csv_path} ...")
    df = pd.read_csv(csv_path)
    positions = df["Pos"].values.astype(np.int64)
    is_plus = df["Strand"].values == "+"
    print(f"  Total: {len(df):,}  (+ {is_plus.sum():,}, - {(~is_plus).sum():,})")
    return positions, is_plus


def load_hns_binding_sites(bed_path: str) -> list[tuple[int, int]]:
    print(f"Loading H-NS binding sites from {bed_path} ...")
    sites = []
    with open(bed_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                sites.append((int(parts[1]), int(parts[2])))
    print(f"  Loaded {len(sites)} H-NS binding regions")
    return sites


def find_best_f1_row(pr_data_path: str) -> tuple[float, float, int]:
    print(f"Loading PR curve data from {pr_data_path} ...")
    df = pd.read_csv(pr_data_path, sep="\t")
    best_idx = df["f1_score"].idxmax()
    row = df.loc[best_idx]
    thr = float(row["threshold"])
    f1 = float(row["f1_score"])
    k = int(row["num_predictions"])
    print(f"  Best F1: {f1:.6f} at threshold {thr:.10f}, num_predictions k={k:,}")
    return thr, f1, k


def _build_promoter_regions(ref_pos, ref_plus, tolerance: int):
    regions = {}
    for strand_val, mask in [("+", ref_plus), ("-", ~ref_plus)]:
        pos = ref_pos[mask]
        if strand_val == "+":
            rs = pos - UPSTREAM_MAX - tolerance
            re = pos - UPSTREAM_MIN + tolerance
        else:
            rs = pos + UPSTREAM_MIN - tolerance
            re = pos + UPSTREAM_MAX + tolerance
        order = np.argsort(rs)
        rs, re = rs[order], re[order]
        prefix_max_end = np.maximum.accumulate(re)
        regions[strand_val] = (rs, re, prefix_max_end)
    return regions


def _vectorized_is_tp(w_starts, w_ends, reg_starts, reg_ends, prefix_max_end):
    n = len(w_starts)
    if n == 0 or len(reg_starts) == 0:
        return np.zeros(n, dtype=np.bool_)
    idx = np.searchsorted(reg_starts, w_starts, side="right") - 1
    valid = idx >= 0
    result = np.zeros(n, dtype=np.bool_)
    result[valid] = prefix_max_end[idx[valid]] >= w_ends[valid]
    return result


def compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance: int) -> np.ndarray:
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    out = np.zeros(len(starts), dtype=np.bool_)
    pm = is_plus
    out[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    out[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])
    return out


def greedy_o2o_topk(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
    k: int,
):
    """
    First k predictions in global descending score order (mergesort).
    Returns tp_centers, fp_centers, fn_ref_positions (unmatched TSS).
    """
    n_ref = len(ref_pos)
    n = len(starts)
    k = min(k, n)
    desc_order = np.argsort(-scores, kind="mergesort")

    plus_ref_idx = np.where(ref_plus)[0]
    minus_ref_idx = np.where(~ref_plus)[0]
    plus_pos = ref_pos[plus_ref_idx]
    minus_pos = ref_pos[minus_ref_idx]
    o_p = np.argsort(plus_pos)
    o_m = np.argsort(minus_pos)
    plus_pos_s = plus_pos[o_p]
    minus_pos_s = minus_pos[o_m]
    plus_idx_s = plus_ref_idx[o_p]
    minus_idx_s = minus_ref_idx[o_m]

    matched_ref = np.zeros(n_ref, dtype=np.bool_)
    tp_centers: list[int] = []
    fp_centers: list[int] = []

    for step in range(k):
        idx = int(desc_order[step])
        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if not geom_ok[idx]:
            fp_centers.append(center)
            continue

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                fp_centers.append(center)
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                fp_centers.append(center)
                continue
            lo = int(np.searchsorted(minus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(minus_pos_s, hi_t, side="right"))
            pos_s = minus_pos_s
            idx_s = minus_idx_s

        best_j = -1
        best_d = 10**18
        for j in range(lo, hi):
            ridx = int(idx_s[j])
            if matched_ref[ridx]:
                continue
            T = int(pos_s[j])
            if is_plus[idx]:
                reg_s = T - UPSTREAM_MAX - tolerance
                reg_e = T - UPSTREAM_MIN + tolerance
            else:
                reg_s = T + UPSTREAM_MIN - tolerance
                reg_e = T + UPSTREAM_MAX + tolerance
            if ws >= reg_s and we <= reg_e:
                d = abs(center - T)
                if d < best_d:
                    best_d = d
                    best_j = j

        if best_j >= 0:
            matched_ref[int(idx_s[best_j])] = True
            tp_centers.append(center)
        else:
            fp_centers.append(center)

    fn_positions = [int(ref_pos[i]) for i in range(n_ref) if not matched_ref[i]]
    return tp_centers, fp_centers, fn_positions


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


def create_hns_bar_chart(
    tp_count: int,
    tp_hns_count: int,
    tp_hns_ratio: float,
    fp_count: int,
    fp_hns_count: int,
    fp_hns_ratio: float,
    fn_count: int,
    fn_hns_count: int,
    fn_hns_ratio: float,
    output_file: str,
    title: str,
):
    print("\nCreating bar chart...")
    categories = ["TP\n(True Positives)", "FP\n(False Positives)", "FN\n(False Negatives)"]
    counts = [tp_count, fp_count, fn_count]
    hns_counts = [tp_hns_count, fp_hns_count, fn_hns_count]
    hns_ratios = [tp_hns_ratio, fp_hns_ratio, fn_hns_ratio]
    non_hns_counts = [c - h for c, h in zip(counts, hns_counts)]

    color_hns = "#2E7D32"
    color_non_hns = "#A5D6A7"

    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.arange(len(categories))
    bar_width = 0.6

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
                fontsize=16,
                fontweight="bold",
                color="white",
            )

    for i, total in enumerate(counts):
        ax.text(
            x[i],
            total + max(counts) * 0.02,
            f"n={total:,}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="black",
        )

    ax.set_ylabel("Count", fontsize=14, fontweight="bold")
    ax.set_xlabel("Category", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    ymax = max(counts) if max(counts) > 0 else 1
    ax.set_ylim(0, ymax * 1.15)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    highres_file = output_file.replace(".png", "_highres.png")
    plt.savefig(highres_file, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    print(f"  Saved: {highres_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-curve-tsv", required=True)
    ap.add_argument("--predictions-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--hns-bed", required=True)
    ap.add_argument("--match-tolerance", type=int, required=True, help="0 or 100 (bp), same as PR curve")
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    label_tol = "no tolerance" if args.match_tolerance == 0 else f"±{args.match_tolerance} bp tolerance"

    print("=" * 70)
    print(f"  H-NS overlap — v5 one-to-one best F1 ({label_tol})")
    print("=" * 70)
    print(f"  PR curve TSV: {args.pr_curve_tsv}")
    print(f"  Predictions:  {args.predictions_tsv}")
    print(f"  Reference:    {args.reference_csv}")
    print(f"  H-NS BED:     {args.hns_bed}")
    print(f"  Output dir:   {out_dir}")
    print()

    thr, best_f1, k = find_best_f1_row(args.pr_curve_tsv)
    t0 = time.time()
    starts, ends, is_plus, scores = load_predictions(args.predictions_tsv)
    ref_pos, ref_plus = load_reference_tss(args.reference_csv)
    hns_sites = load_hns_binding_sites(args.hns_bed)

    print(f"\nGreedy 1:1 matching: tolerance={args.match_tolerance} bp, k={k:,} ...")
    geom_ok = compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, args.match_tolerance)
    print(f"  Geom-matchable windows: {geom_ok.sum():,} / {len(geom_ok):,}")
    tp_c, fp_c, fn_p = greedy_o2o_topk(
        starts, ends, is_plus, scores, ref_pos, ref_plus, args.match_tolerance, geom_ok, k
    )
    print(f"  TP: {len(tp_c):,}  FP: {len(fp_c):,}  FN: {len(fn_p):,}  (elapsed {time.time() - t0:.1f}s)")

    print("\nH-NS overlap (TP/FP: window center; FN: TSS position)...")
    tp_h, tp_r = check_hns_overlap(tp_c, hns_sites)
    fp_h, fp_r = check_hns_overlap(fp_c, hns_sites)
    fn_h, fn_r = check_hns_overlap(fn_p, hns_sites)

    title = (
        f"H-NS overlap — v5 one-to-one best F1\n"
        f"({label_tol}, F1={best_f1:.4f}, thr={thr:.4f}, k={k:,}; "
        f"window={WINDOW_LEN}bp, promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}])"
    )
    png = os.path.join(out_dir, "hns_overlap_bar_chart.png")
    create_hns_bar_chart(
        len(tp_c), tp_h, tp_r, len(fp_c), fp_h, fp_r, len(fn_p), fn_h, fn_r, png, title
    )

    summary = pd.DataFrame(
        {
            "category": ["TP", "FP", "FN"],
            "count": [len(tp_c), len(fp_c), len(fn_p)],
            "hns_overlap_count": [tp_h, fp_h, fn_h],
            "hns_overlap_ratio": [tp_r, fp_r, fn_r],
        }
    )
    summary_path = os.path.join(out_dir, "hns_overlap_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    meta_path = os.path.join(out_dir, "run_metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"pr_curve_tsv\t{args.pr_curve_tsv}\n")
        f.write(f"match_tolerance_bp\t{args.match_tolerance}\n")
        f.write(f"best_f1\t{best_f1}\n")
        f.write(f"threshold\t{thr}\n")
        f.write(f"num_predictions_k\t{k}\n")
    print(f"  Summary: {summary_path}")
    print(f"  Meta:    {meta_path}")
    print("\n" + "=" * 70 + "\n  DONE\n" + "=" * 70)


if __name__ == "__main__":
    main()
