#!/usr/bin/env python3
"""
Strand-aware ROC curve for double-strand window-interval predictions.

Same vectorized precomputation as the PR curve counterpart:
  1. Precompute is_tp for ALL windows once (numpy vectorised, O(N log M)).
  2. Precompute max_score per reference TSS once (O(M × small)).
  3. Sweep thresholds using sorted arrays + cumulative sums.

ROC metrics:
  TPR (sensitivity) = matched_refs_at_t / total_refs
  FPR = FP_at_t / total_negatives
    where total_negatives = total_windows - total_TP_windows
    and FP_at_t = (predictions_above_t) - (TP_predictions_above_t)

Promoter definition: [TSS-39, TSS-6] = 34 bp (strand-aware).
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
        tsv_path, sep="\t",
        usecols=["start_1based", "end_1based_inclusive", "strand", "score"],
        dtype={"start_1based": np.int64, "end_1based_inclusive": np.int64, "score": np.float64},
    )
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    starts = df["start_1based"].values
    ends = df["end_1based_inclusive"].values
    is_plus = (df["strand"].values == "+")
    scores = df["score"].values
    print(f"  + strand: {is_plus.sum():,}  - strand: {(~is_plus).sum():,}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    return starts, ends, is_plus, scores


def load_reference_tss(csv_path: str):
    print(f"Loading reference TSS: {csv_path} ...")
    df = pd.read_csv(csv_path)
    positions = df["Pos"].values.astype(np.int64)
    is_plus = (df["Strand"].values == "+")
    print(f"  Total: {len(df)}  (+ {is_plus.sum():,}, - {(~is_plus).sum():,})")
    return positions, is_plus


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


def _compute_max_score_per_ref(ref_pos, ref_plus,
                               pred_starts_p, pred_ends_p, pred_scores_p,
                               pred_starts_m, pred_ends_m, pred_scores_m,
                               tolerance: int):
    n_ref = len(ref_pos)
    max_scores = np.full(n_ref, -np.inf)
    for k in range(n_ref):
        t = ref_pos[k]
        if ref_plus[k]:
            reg_s = t - UPSTREAM_MAX - tolerance
            reg_e = t - UPSTREAM_MIN + tolerance
            starts, ends, scores = pred_starts_p, pred_ends_p, pred_scores_p
        else:
            reg_s = t + UPSTREAM_MIN - tolerance
            reg_e = t + UPSTREAM_MAX + tolerance
            starts, ends, scores = pred_starts_m, pred_ends_m, pred_scores_m
        if len(starts) == 0:
            continue
        lo = np.searchsorted(starts, reg_s, side="left")
        hi = np.searchsorted(starts, reg_e, side="right")
        if lo >= hi:
            continue
        contained = (starts[lo:hi] >= reg_s) & (ends[lo:hi] <= reg_e)
        if contained.any():
            max_scores[k] = scores[lo:hi][contained].max()
    return max_scores


def generate_roc_curve(starts, ends, is_plus, scores,
                       ref_pos, ref_plus, tolerance: int,
                       num_thresholds: int = 100):
    print(f"\n  Building promoter regions (tolerance={tolerance}) ...")
    t0 = time.time()
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)

    plus_mask = is_plus
    minus_mask = ~is_plus

    sp, ep, scp = starts[plus_mask], ends[plus_mask], scores[plus_mask]
    sm, em, scm = starts[minus_mask], ends[minus_mask], scores[minus_mask]

    idx_p = np.argsort(sp); sp, ep, scp = sp[idx_p], ep[idx_p], scp[idx_p]
    idx_m = np.argsort(sm); sm, em, scm = sm[idx_m], em[idx_m], scm[idx_m]

    print(f"  Computing is_tp for {len(sp):,} (+) windows ...")
    tp_plus = _vectorized_is_tp(sp, ep, *regions["+"])
    print(f"    TP windows (+): {tp_plus.sum():,}")

    print(f"  Computing is_tp for {len(sm):,} (-) windows ...")
    tp_minus = _vectorized_is_tp(sm, em, *regions["-"])
    print(f"    TP windows (-): {tp_minus.sum():,}")

    all_scores = np.concatenate([scp, scm])
    all_is_tp = np.concatenate([tp_plus, tp_minus])
    total_tp_windows = int(all_is_tp.sum())
    total_neg_windows = len(all_is_tp) - total_tp_windows
    print(f"  Total TP windows: {total_tp_windows:,} / {len(all_is_tp):,}")
    print(f"  Total negative windows: {total_neg_windows:,}")

    print(f"  Computing max score per reference TSS ({len(ref_pos):,} TSS) ...")
    max_score_per_ref = _compute_max_score_per_ref(
        ref_pos, ref_plus, sp, ep, scp, sm, em, scm, tolerance
    )
    n_ref_matched = (max_score_per_ref > -np.inf).sum()
    print(f"    TSS with >=1 matching window: {n_ref_matched:,}")

    precomp_elapsed = time.time() - t0
    print(f"  Precomputation done in {precomp_elapsed:.1f}s")

    print(f"  Sweeping thresholds ...")
    t1 = time.time()

    percentiles = np.linspace(0, 100, num_thresholds)
    thr_pct = np.percentile(all_scores, percentiles)
    p99 = float(np.percentile(all_scores, 99))
    s_max = float(all_scores.max())
    thr_high = np.linspace(s_max, p99, min(80, num_thresholds))
    thresholds = np.sort(np.unique(np.concatenate([thr_pct, thr_high])))[::-1]

    print(f"    {len(thresholds)} unique thresholds [{thresholds[-1]:.6f} .. {thresholds[0]:.6f}]")

    desc_order = np.argsort(-all_scores)
    sorted_scores = all_scores[desc_order]
    sorted_tp = all_is_tp[desc_order]
    cum_tp = np.cumsum(sorted_tp).astype(np.float64)

    sorted_ref_scores = np.sort(max_score_per_ref[max_score_per_ref > -np.inf])[::-1]
    n_total_ref = len(ref_pos)

    thresholds_out, tpr_out, fpr_out, npred_out = [], [], [], []

    for thr in thresholds:
        n_pred = int(np.searchsorted(-sorted_scores, -thr, side="right"))
        if n_pred == 0:
            thresholds_out.append(thr)
            tpr_out.append(0.0)
            fpr_out.append(0.0)
            npred_out.append(0)
            continue

        tp_count = float(cum_tp[n_pred - 1])
        fp_count = n_pred - tp_count

        n_ref_above = int(np.searchsorted(-sorted_ref_scores, -thr, side="right"))
        tpr = n_ref_above / n_total_ref if n_total_ref > 0 else 0.0
        fpr = fp_count / total_neg_windows if total_neg_windows > 0 else 0.0

        thresholds_out.append(thr)
        tpr_out.append(tpr)
        fpr_out.append(fpr)
        npred_out.append(n_pred)

    sweep_elapsed = time.time() - t1
    print(f"    Sweep done in {sweep_elapsed:.1f}s")
    print(f"  Total ROC generation: {time.time()-t0:.1f}s")
    return thresholds_out, fpr_out, tpr_out, npred_out


def plot_roc_curve(thresholds, fprs, tprs, output_file, title,
                   num_references=0, num_predictions_total=0):
    sorted_idx = np.argsort(fprs)
    sorted_fpr = np.array(fprs)[sorted_idx]
    sorted_tpr = np.array(tprs)[sorted_idx]
    try:
        auroc = float(np.trapezoid(sorted_tpr, sorted_fpr))
    except AttributeError:
        auroc = float(np.trapz(sorted_tpr, sorted_fpr))
    print(f"  AUROC: {auroc:.6f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(sorted_fpr, sorted_tpr, "b-", linewidth=2,
            label=f"ROC Curve (AUROC={auroc:.4f})", zorder=3)
    ax.scatter(fprs, tprs, c="blue", s=20, alpha=0.3, edgecolors="none", zorder=2)

    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="Random baseline (AUROC=0.5)")

    youden = [t - f for t, f in zip(tprs, fprs)]
    best_idx = int(np.argmax(youden))
    best_tpr = tprs[best_idx]
    best_fpr = fprs[best_idx]
    best_thr = thresholds[best_idx]
    best_youden = youden[best_idx]
    ax.plot(best_fpr, best_tpr, "ro", markersize=12,
            label=f"Best Youden: {best_youden:.3f} (T={best_thr:.4f})", zorder=5)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    highres = output_file.replace(".png", "_highres.png")
    plt.savefig(highres, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    print(f"  Saved: {highres}")
    return best_idx, auroc, best_thr, best_tpr, best_fpr, best_youden


def save_roc_data(thresholds, fprs, tprs, output_file, num_predictions=None):
    data = {"threshold": thresholds, "fpr": fprs, "tpr": tprs}
    if num_predictions is not None:
        data["num_predictions"] = num_predictions
    pd.DataFrame(data).to_csv(output_file, sep="\t", index=False)
    print(f"  Saved: {output_file}")


def save_auroc_summary(output_file, auroc, best_thr, best_tpr, best_fpr, best_youden,
                       num_references, num_windows, tolerance):
    with open(output_file, "w") as f:
        f.write(f"AUROC: {auroc:.6f}\n")
        f.write(f"Best Youden J: {best_youden:.6f}\n")
        f.write(f"  Threshold: {best_thr:.6f}\n")
        f.write(f"  TPR (sensitivity): {best_tpr:.6f}\n")
        f.write(f"  FPR: {best_fpr:.6f}\n")
        f.write(f"Reference TSS: {num_references}\n")
        f.write(f"Total windows: {num_windows}\n")
        f.write(f"Tolerance: {tolerance} bp\n")
    print(f"  Saved: {output_file}")


def main():
    ap = argparse.ArgumentParser(
        description="Strand-aware ROC curve for double-strand window predictions (vectorized)")
    ap.add_argument("--input-tsv", required=True)
    ap.add_argument("--reference-csv", required=True)
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  STRAND-AWARE ROC CURVE (VECTORIZED, WINDOW-INTERVAL OVERLAP)")
    print("=" * 70)
    print(f"  Input TSV:    {args.input_tsv}")
    print(f"  Reference:    {args.reference_csv}")
    print(f"  Promoter def: [TSS-{UPSTREAM_MAX}, TSS-{UPSTREAM_MIN}] = {UPSTREAM_MAX - UPSTREAM_MIN + 1} bp")
    print(f"  Window len:   {WINDOW_LEN} bp")
    print(f"  Tolerance:    +-{args.tolerance} bp")
    print(f"  Thresholds:   {args.num_thresholds}")
    print(f"  Output:       {args.output_dir}")
    print()

    t_global = time.time()
    starts, ends, is_plus, scores = load_predictions(args.input_tsv)
    ref_pos, ref_plus = load_reference_tss(args.reference_csv)

    tol_tag = f"{args.tolerance}bp" if args.tolerance > 0 else "no"

    thresholds, fprs, tprs, npreds = generate_roc_curve(
        starts, ends, is_plus, scores,
        ref_pos, ref_plus, args.tolerance,
        num_thresholds=args.num_thresholds,
    )

    title = (
        f"ROC Curve: Double-Strand Window-Interval Evaluation\n"
        f"(Promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}], "
        f"tolerance={tol_tag}, strand-aware)"
    )
    plot_file = os.path.join(args.output_dir, "roc_curve.png")
    _, auroc, best_thr, best_tpr, best_fpr, best_youden = plot_roc_curve(
        thresholds, fprs, tprs, plot_file, title,
        num_references=len(ref_pos),
        num_predictions_total=len(starts),
    )

    data_file = os.path.join(args.output_dir, "roc_curve_data.tsv")
    save_roc_data(thresholds, fprs, tprs, data_file, num_predictions=npreds)

    summary_file = os.path.join(args.output_dir, "auroc_summary.txt")
    save_auroc_summary(summary_file, auroc, best_thr, best_tpr, best_fpr, best_youden,
                       len(ref_pos), len(starts), args.tolerance)

    elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  AUROC:           {auroc:.6f}")
    print(f"  Best Youden J:   {best_youden:.3f}  (T={best_thr:.6f}, TPR={best_tpr:.3f}, FPR={best_fpr:.3f})")
    print(f"  Reference TSS:   {len(ref_pos):,}")
    print(f"  Total windows:   {len(starts):,}")
    print(f"  Elapsed:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output:          {args.output_dir}")


if __name__ == "__main__":
    main()
