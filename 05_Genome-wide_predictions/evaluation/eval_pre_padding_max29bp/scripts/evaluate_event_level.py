#!/usr/bin/env python3
"""
Event-level PR + ROC evaluation with greedy matching.

Window-level vs Event-level:
  Window:  unit = 29bp DNA fragment;  one promoter → many TP windows
  Event:   unit = biological event;   one promoter → at most 1 TP peak

Algorithm:
  1. Load peaks (from call_peaks_doublestrand.py) and reference TSS.
  2. Precompute is_positive for each peak (overlaps any promoter region).
  3. Sort peaks by score descending.
  4. Greedy matching: walk through peaks in score order, each peak matches
     the nearest unmatched TSS (same strand, within tolerance-expanded
     promoter region).  One peak → at most one TSS; one TSS → matched once.
  5. Compute cumulative TP/FP/matched_TSS at each rank.
  6. Derive PR and ROC metrics, plot, and save.

Promoter definition: [TSS-39, TSS-6] = 34 bp, strand-aware.
  + strand TSS T: accepted peak region [T-39-tol, T-6+tol]
  - strand TSS T: accepted peak region [T+6-tol, T+39+tol]
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


def load_peaks(tsv_path: str):
    print(f"Loading peaks: {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"  {len(df):,} peaks  (+ {(df['strand']=='+').sum():,}, - {(df['strand']=='-').sum():,})")
    return df


def load_reference_tss(csv_path: str):
    print(f"Loading reference TSS: {csv_path} ...")
    df = pd.read_csv(csv_path)
    positions = df["Pos"].values.astype(np.int64)
    is_plus = (df["Strand"].values == "+")
    print(f"  Total: {len(df)}  (+ {is_plus.sum():,}, - {(~is_plus).sum():,})")
    return positions, is_plus


def _build_regions_sorted(ref_pos, ref_is_plus, tolerance):
    """Build sorted promoter region arrays per strand for vectorized is_positive."""
    regions = {}
    for strand_val, mask in [("+", ref_is_plus), ("-", ~ref_is_plus)]:
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


def _vectorized_is_positive(positions, strands_plus, regions):
    """For each peak, check if its position falls within any promoter region (same strand)."""
    n = len(positions)
    result = np.zeros(n, dtype=np.bool_)
    for is_plus_val, strand_key in [(True, "+"), (False, "-")]:
        mask = strands_plus == is_plus_val
        if not mask.any():
            continue
        pos = positions[mask]
        rs, re, pme = regions[strand_key]
        if len(rs) == 0:
            continue
        idx = np.searchsorted(rs, pos, side="right") - 1
        valid = idx >= 0
        hits = np.zeros(len(pos), dtype=np.bool_)
        hits[valid] = pme[idx[valid]] >= pos[valid]
        result[mask] = hits
    return result


def greedy_match(peak_positions, peak_strands_plus, peak_scores,
                 plus_tss_sorted, minus_tss_sorted,
                 tolerance):
    """Greedy matching: peaks in descending score order match nearest unmatched TSS.

    Returns:
      is_tp: bool array per peak (matched a TSS)
      cum_matched_tss: int array, cumulative count of distinct TSS matched
    """
    n = len(peak_positions)
    desc_order = np.argsort(-peak_scores)
    positions_sorted = peak_positions[desc_order]
    strands_sorted = peak_strands_plus[desc_order]

    plus_matched = np.zeros(len(plus_tss_sorted), dtype=np.bool_)
    minus_matched = np.zeros(len(minus_tss_sorted), dtype=np.bool_)

    is_tp_sorted = np.zeros(n, dtype=np.bool_)
    cum_matched = np.zeros(n, dtype=np.int64)
    total_matched = 0

    for i in range(n):
        p = positions_sorted[i]
        if strands_sorted[i]:
            tss_arr = plus_tss_sorted
            matched_arr = plus_matched
            lo_t = p + UPSTREAM_MIN - tolerance
            hi_t = p + UPSTREAM_MAX + tolerance
        else:
            tss_arr = minus_tss_sorted
            matched_arr = minus_matched
            lo_t = p - UPSTREAM_MAX - tolerance
            hi_t = p - UPSTREAM_MIN + tolerance

        lo_idx = np.searchsorted(tss_arr, lo_t, side="left")
        hi_idx = np.searchsorted(tss_arr, hi_t, side="right")

        best_j = -1
        best_dist = 10**18
        for j in range(lo_idx, hi_idx):
            if not matched_arr[j]:
                d = abs(int(p) - int(tss_arr[j]))
                if d < best_dist:
                    best_dist = d
                    best_j = j

        if best_j >= 0:
            matched_arr[best_j] = True
            is_tp_sorted[i] = True
            total_matched += 1

        cum_matched[i] = total_matched

    is_tp_orig = np.zeros(n, dtype=np.bool_)
    is_tp_orig[desc_order] = is_tp_sorted

    return desc_order, is_tp_sorted, cum_matched


def compute_pr_roc(desc_order, is_tp_sorted, cum_matched,
                   peak_scores, is_positive, total_tss,
                   num_thresholds=100):
    """Compute PR and ROC points from greedy matching results."""
    n = len(desc_order)
    scores_sorted = peak_scores[desc_order]

    cum_tp = np.cumsum(is_tp_sorted).astype(np.float64)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    precision_at_rank = cum_tp / ranks
    recall_at_rank = cum_matched.astype(np.float64) / total_tss if total_tss > 0 else np.zeros(n)

    is_neg_sorted = ~is_positive[desc_order]
    total_neg = int(is_neg_sorted.sum())
    cum_fp_neg = np.cumsum(is_neg_sorted).astype(np.float64)
    fpr_at_rank = cum_fp_neg / total_neg if total_neg > 0 else np.zeros(n)

    percentiles = np.linspace(0, 100, num_thresholds)
    thr_pct = np.percentile(scores_sorted, percentiles)
    p99 = float(np.percentile(scores_sorted, 99))
    s_max = float(scores_sorted.max()) if n > 0 else 0.0
    thr_high = np.linspace(s_max, p99, min(80, num_thresholds))
    thresholds = np.sort(np.unique(np.concatenate([thr_pct, thr_high])))[::-1]

    neg_sorted_scores = scores_sorted.copy()

    pr_data = {"threshold": [], "precision": [], "recall": [], "f1_score": [], "num_events": []}
    roc_data = {"threshold": [], "fpr": [], "tpr": [], "num_events": []}

    for thr in thresholds:
        k = int(np.searchsorted(-scores_sorted, -thr, side="right"))
        if k == 0:
            pr_data["threshold"].append(thr)
            pr_data["precision"].append(0.0)
            pr_data["recall"].append(0.0)
            pr_data["f1_score"].append(0.0)
            pr_data["num_events"].append(0)
            roc_data["threshold"].append(thr)
            roc_data["fpr"].append(0.0)
            roc_data["tpr"].append(0.0)
            roc_data["num_events"].append(0)
            continue

        prec = float(cum_tp[k - 1] / k)
        rec = float(recall_at_rank[k - 1])
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = float(fpr_at_rank[k - 1])
        tpr = rec

        pr_data["threshold"].append(thr)
        pr_data["precision"].append(prec)
        pr_data["recall"].append(rec)
        pr_data["f1_score"].append(f1)
        pr_data["num_events"].append(k)

        roc_data["threshold"].append(thr)
        roc_data["fpr"].append(fpr)
        roc_data["tpr"].append(tpr)
        roc_data["num_events"].append(k)

    return pr_data, roc_data


def plot_pr_curve(pr_data, output_file, title, num_refs, num_peaks):
    precs = np.array(pr_data["precision"])
    recs = np.array(pr_data["recall"])
    thrs = pr_data["threshold"]
    si = np.argsort(recs)
    sr, sp = recs[si], precs[si]
    try:
        auprc = float(np.trapezoid(sp, sr))
    except AttributeError:
        auprc = float(np.trapz(sp, sr))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sr, sp, "b-", linewidth=2, label=f"PR Curve (AUPRC={auprc:.3f})", zorder=3)
    ax.scatter(recs, precs, c="blue", s=20, alpha=0.3, edgecolors="none", zorder=2)

    f1s = pr_data["f1_score"]
    bi = int(np.argmax(f1s))
    ax.plot(recs[bi], precs[bi], "ro", markersize=12,
            label=f"Max F1: {f1s[bi]:.3f} (T={thrs[bi]:.4f})", zorder=5)

    if num_refs and num_peaks:
        bl = num_refs / num_peaks
        ax.axhline(y=bl, color="r", linestyle="--", alpha=0.5,
                   label=f"Baseline (random, P={bl:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.replace(".png", "_highres.png"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    return auprc, f1s[bi], thrs[bi], precs[bi], recs[bi]


def plot_roc_curve(roc_data, output_file, title):
    fprs = np.array(roc_data["fpr"])
    tprs = np.array(roc_data["tpr"])
    thrs = roc_data["threshold"]
    si = np.argsort(fprs)
    sf, st = fprs[si], tprs[si]
    try:
        auroc = float(np.trapezoid(st, sf))
    except AttributeError:
        auroc = float(np.trapz(st, sf))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sf, st, "b-", linewidth=2, label=f"ROC Curve (AUROC={auroc:.4f})", zorder=3)
    ax.scatter(fprs, tprs, c="blue", s=20, alpha=0.3, edgecolors="none", zorder=2)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random baseline (AUROC=0.5)")

    youden = [t - f for t, f in zip(tprs, fprs)]
    bi = int(np.argmax(youden))
    ax.plot(fprs[bi], tprs[bi], "ro", markersize=12,
            label=f"Best Youden: {youden[bi]:.3f} (T={thrs[bi]:.4f})", zorder=5)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.replace(".png", "_highres.png"), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    return auroc, youden[bi], thrs[bi], tprs[bi], fprs[bi]


def main():
    ap = argparse.ArgumentParser(description="Event-level PR + ROC with greedy matching")
    ap.add_argument("--peaks-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tol_tag = f"{args.tolerance}bp" if args.tolerance > 0 else "no"

    print("=" * 70)
    print("  EVENT-LEVEL PR + ROC (GREEDY MATCHING)")
    print("=" * 70)
    print(f"  Peaks:        {args.peaks_tsv}")
    print(f"  Reference:    {args.reference_csv}")
    print(f"  Promoter:     [TSS-{UPSTREAM_MAX}, TSS-{UPSTREAM_MIN}] = {UPSTREAM_MAX - UPSTREAM_MIN + 1} bp")
    print(f"  Tolerance:    +-{args.tolerance} bp")
    print(f"  Output:       {args.output_dir}")
    print()

    t_global = time.time()
    peaks_df = load_peaks(args.peaks_tsv)
    ref_pos, ref_plus = load_reference_tss(args.reference_csv)

    peak_positions = peaks_df["peak_start_1based"].values.astype(np.int64)
    peak_strands_plus = (peaks_df["strand"].values == "+")
    peak_scores = peaks_df["score"].values.astype(np.float64)

    print("\n  Precomputing is_positive (peak in any promoter region) ...")
    regions = _build_regions_sorted(ref_pos, ref_plus, args.tolerance)
    is_positive = _vectorized_is_positive(peak_positions, peak_strands_plus, regions)
    print(f"    Positive peaks: {is_positive.sum():,} / {len(is_positive):,}")
    print(f"    Negative peaks: {(~is_positive).sum():,}")

    plus_tss = np.sort(ref_pos[ref_plus])
    minus_tss = np.sort(ref_pos[~ref_plus])

    print(f"\n  Greedy matching ({len(peak_positions):,} peaks, {len(ref_pos):,} TSS) ...")
    t0 = time.time()
    desc_order, is_tp_sorted, cum_matched = greedy_match(
        peak_positions, peak_strands_plus, peak_scores,
        plus_tss, minus_tss, args.tolerance,
    )
    match_elapsed = time.time() - t0
    total_tp = int(is_tp_sorted.sum())
    total_matched_tss = int(cum_matched[-1]) if len(cum_matched) > 0 else 0
    print(f"    Done in {match_elapsed:.1f}s")
    print(f"    TP events: {total_tp:,}")
    print(f"    FP events: {len(peak_positions) - total_tp:,}")
    print(f"    Matched TSS: {total_matched_tss:,} / {len(ref_pos):,}")

    print(f"\n  Computing PR + ROC curves ...")
    pr_data, roc_data = compute_pr_roc(
        desc_order, is_tp_sorted, cum_matched,
        peak_scores, is_positive, len(ref_pos),
        num_thresholds=args.num_thresholds,
    )

    pr_title = (
        f"PR Curve: Event-Level (Greedy Matching)\n"
        f"(Promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}], "
        f"tolerance={tol_tag}, strand-aware)"
    )
    pr_file = os.path.join(args.output_dir, "pr_curve.png")
    auprc, best_f1, best_thr_pr, best_prec, best_rec = plot_pr_curve(
        pr_data, pr_file, pr_title,
        num_refs=len(ref_pos), num_peaks=len(peak_positions),
    )

    pr_tsv = os.path.join(args.output_dir, "pr_curve_data.tsv")
    pd.DataFrame(pr_data).to_csv(pr_tsv, sep="\t", index=False)
    print(f"  Saved: {pr_tsv}")

    roc_title = (
        f"ROC Curve: Event-Level (Greedy Matching)\n"
        f"(Promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}], "
        f"tolerance={tol_tag}, strand-aware)"
    )
    roc_file = os.path.join(args.output_dir, "roc_curve.png")
    auroc, best_youden, best_thr_roc, best_tpr, best_fpr = plot_roc_curve(
        roc_data, roc_file, roc_title,
    )

    roc_tsv = os.path.join(args.output_dir, "roc_curve_data.tsv")
    pd.DataFrame(roc_data).to_csv(roc_tsv, sep="\t", index=False)
    print(f"  Saved: {roc_tsv}")

    summary_file = os.path.join(args.output_dir, "event_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"=== Event-Level Evaluation (tolerance={args.tolerance}bp) ===\n")
        f.write(f"Total peaks: {len(peak_positions)}\n")
        f.write(f"  Positive peaks: {int(is_positive.sum())}\n")
        f.write(f"  Negative peaks: {int((~is_positive).sum())}\n")
        f.write(f"Reference TSS: {len(ref_pos)}\n")
        f.write(f"Matched TSS (all peaks): {total_matched_tss}\n")
        f.write(f"\n--- PR Curve ---\n")
        f.write(f"AUPRC: {auprc:.6f}\n")
        f.write(f"Best F1: {best_f1:.6f}  (T={best_thr_pr:.6f}, P={best_prec:.6f}, R={best_rec:.6f})\n")
        f.write(f"\n--- ROC Curve ---\n")
        f.write(f"AUROC: {auroc:.6f}\n")
        f.write(f"Best Youden J: {best_youden:.6f}  (T={best_thr_roc:.6f}, TPR={best_tpr:.6f}, FPR={best_fpr:.6f})\n")
    print(f"  Saved: {summary_file}")

    elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  SUMMARY (tolerance={args.tolerance}bp)")
    print(f"{'='*70}")
    print(f"  AUPRC:       {auprc:.6f}")
    print(f"  Best F1:     {best_f1:.3f}  (T={best_thr_pr:.6f}, P={best_prec:.3f}, R={best_rec:.3f})")
    print(f"  AUROC:       {auroc:.6f}")
    print(f"  Best Youden: {best_youden:.3f}  (T={best_thr_roc:.6f}, TPR={best_tpr:.3f}, FPR={best_fpr:.3f})")
    print(f"  Peaks: {len(peak_positions):,}  |  TSS: {len(ref_pos):,}")
    print(f"  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
