#!/usr/bin/env python3
"""
Strand-aware ROC with one-to-one matching between predicted windows and
reference promoter regions (same greedy rule as the PR one-to-one script).

TPR = matched_refs / total_refs at threshold.

FPR follows the event-level convention: a “negative” window is one that is not
fully contained in any promoter region (~geom_ok). Among predictions with
score ≥ threshold, false positives for ROC are only those negative windows;
promoter-local windows that fail greedy matching are not counted as FP here,
so FPR stays in [0, 1].
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

# Plot styling (ROC curve)
CURVE_LINEWIDTH = 5.0
LEGEND_FONTSIZE = 14


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


def compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance: int) -> np.ndarray:
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    out = np.zeros(len(starts), dtype=np.bool_)
    pm = is_plus
    out[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    out[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])
    return out


def greedy_incremental_cum_matched(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
) -> np.ndarray:
    n_ref = len(ref_pos)
    n = len(starts)
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
    cum_matched = np.zeros(n + 1, dtype=np.int64)
    nmatch = 0

    for step in range(n):
        idx = int(desc_order[step])
        if not geom_ok[idx]:
            cum_matched[step + 1] = nmatch
            continue

        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                cum_matched[step + 1] = nmatch
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                cum_matched[step + 1] = nmatch
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
            nmatch += 1
        cum_matched[step + 1] = nmatch

    return cum_matched


def generate_roc_curve_one_to_one(
    starts, ends, is_plus, scores,
    ref_pos, ref_plus, tolerance: int,
    num_thresholds: int = 100,
):
    print(f"\n  One-to-one matching (tolerance={tolerance}) ...")
    t0 = time.time()
    geom_ok = compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance)
    is_neg_window = ~geom_ok
    total_neg_windows = int(is_neg_window.sum())
    print(f"    Geometrically matchable windows: {int(geom_ok.sum()):,} / {len(geom_ok):,}")
    print(f"    Negative windows (ROC FPR denominator): {total_neg_windows:,}")

    print(f"  Greedy pass over {len(scores):,} windows (desc score) ...")
    t1 = time.time()
    cum_matched = greedy_incremental_cum_matched(
        starts, ends, is_plus, scores, ref_pos, ref_plus, tolerance, geom_ok,
    )
    print(f"    Greedy done in {time.time()-t1:.1f}s; final matches: {cum_matched[-1]:,}")

    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]
    neg_sorted = is_neg_window[desc_order].astype(np.float64)
    cum_fp_neg = np.cumsum(neg_sorted)

    percentiles = np.linspace(0, 100, num_thresholds)
    thr_pct = np.percentile(scores, percentiles)
    p99 = float(np.percentile(scores, 99))
    s_max = float(scores.max())
    thr_high = np.linspace(s_max, p99, min(80, num_thresholds))
    thresholds = np.sort(np.unique(np.concatenate([thr_pct, thr_high])))[::-1]

    print(f"    {len(thresholds)} unique thresholds [{thresholds[-1]:.6f} .. {thresholds[0]:.6f}]")

    n_total_ref = len(ref_pos)
    thresholds_out, tpr_out, fpr_out, npred_out = [], [], [], []

    for thr in thresholds:
        k = int(np.searchsorted(-sorted_scores, -thr, side="right"))
        if k == 0:
            thresholds_out.append(thr)
            tpr_out.append(0.0)
            fpr_out.append(0.0)
            npred_out.append(0)
            continue

        matched = int(cum_matched[k])
        tpr = matched / n_total_ref if n_total_ref > 0 else 0.0
        fp_neg = int(cum_fp_neg[k - 1])
        fpr = fp_neg / total_neg_windows if total_neg_windows > 0 else 0.0

        thresholds_out.append(thr)
        tpr_out.append(tpr)
        fpr_out.append(fpr)
        npred_out.append(k)

    print(f"  Total ROC (1:1) generation: {time.time()-t0:.1f}s")
    return thresholds_out, fpr_out, tpr_out, npred_out


def plot_roc_curve(_thresholds, fprs, tprs, output_file, title):
    sorted_idx = np.argsort(fprs)
    sorted_fpr = np.array(fprs)[sorted_idx]
    sorted_tpr = np.array(tprs)[sorted_idx]
    try:
        auroc = float(np.trapezoid(sorted_tpr, sorted_fpr))
    except AttributeError:
        auroc = float(np.trapz(sorted_tpr, sorted_fpr))
    print(f"  AUROC: {auroc:.6f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        sorted_fpr, sorted_tpr, "b-", linewidth=CURVE_LINEWIDTH,
        label=f"ROC Curve (AUROC={auroc:.4f})", zorder=3,
    )
    ax.scatter(fprs, tprs, c="blue", s=20, alpha=0.3, edgecolors="none", zorder=2)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, framealpha=0.95)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    highres = output_file.replace(".png", "_highres.png")
    plt.savefig(highres, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    print(f"  Saved: {highres}")
    return auroc


def save_roc_data(thresholds, fprs, tprs, output_file, num_predictions=None):
    data = {"threshold": thresholds, "fpr": fprs, "tpr": tprs}
    if num_predictions is not None:
        data["num_predictions"] = num_predictions
    pd.DataFrame(data).to_csv(output_file, sep="\t", index=False)
    print(f"  Saved: {output_file}")


def save_auroc_summary(output_file, auroc, num_references, num_windows, tolerance):
    with open(output_file, "w") as f:
        f.write(f"AUROC: {auroc:.6f}\n")
        f.write(f"Reference TSS: {num_references}\n")
        f.write(f"Total windows: {num_windows}\n")
        f.write(f"Tolerance: {tolerance} bp\n")
        f.write("Matching: one-to-one (greedy, descending score)\n")
    print(f"  Saved: {output_file}")


def main():
    ap = argparse.ArgumentParser(
        description="ROC curve with 1:1 window–promoter greedy matching")
    ap.add_argument("--input-tsv")
    ap.add_argument("--reference-csv")
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate PNGs from existing roc_curve_data.tsv in output-dir (no full eval).",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.replot_only:
        data_file = os.path.join(args.output_dir, "roc_curve_data.tsv")
        if not os.path.isfile(data_file):
            raise SystemExit(f"--replot-only: missing {data_file}")
        df = pd.read_csv(data_file, sep="\t")
        thresholds = df["threshold"].tolist()
        fprs = df["fpr"].tolist()
        tprs = df["tpr"].tolist()
        tol_tag = f"{args.tolerance}bp" if args.tolerance > 0 else "no"
        title = (
            f"ROC Curve: 1:1 Window–Promoter Matching (greedy, score order)\n"
            f"(Promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}], "
            f"tolerance={tol_tag}, strand-aware)"
        )
        plot_file = os.path.join(args.output_dir, "roc_curve.png")
        plot_roc_curve(thresholds, fprs, tprs, plot_file, title)
        print(f"  Replotted from {data_file} -> {plot_file}")
        return

    if not args.input_tsv or not args.reference_csv:
        raise SystemExit("--input-tsv and --reference-csv are required unless --replot-only")

    print("=" * 70)
    print("  ROC CURVE — ONE-TO-ONE WINDOW / PROMOTER MATCHING")
    print("=" * 70)
    print(f"  Input TSV:    {args.input_tsv}")
    print(f"  Reference:    {args.reference_csv}")
    print(f"  Tolerance:    ±{args.tolerance} bp")
    print(f"  Output:       {args.output_dir}")
    print()

    t_global = time.time()
    starts, ends, is_plus, scores = load_predictions(args.input_tsv)
    ref_pos, ref_plus = load_reference_tss(args.reference_csv)

    tol_tag = f"{args.tolerance}bp" if args.tolerance > 0 else "no"

    thresholds, fprs, tprs, npreds = generate_roc_curve_one_to_one(
        starts, ends, is_plus, scores,
        ref_pos, ref_plus, args.tolerance,
        num_thresholds=args.num_thresholds,
    )

    title = (
        f"ROC Curve: 1:1 Window–Promoter Matching (greedy, score order)\n"
        f"(Promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}], "
        f"tolerance={tol_tag}, strand-aware)"
    )
    plot_file = os.path.join(args.output_dir, "roc_curve.png")
    auroc = plot_roc_curve(thresholds, fprs, tprs, plot_file, title)

    data_file = os.path.join(args.output_dir, "roc_curve_data.tsv")
    save_roc_data(thresholds, fprs, tprs, data_file, num_predictions=npreds)

    summary_file = os.path.join(args.output_dir, "auroc_summary.txt")
    save_auroc_summary(summary_file, auroc, len(ref_pos), len(starts), args.tolerance)

    elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  AUROC:           {auroc:.6f}")
    print(f"  Reference TSS:   {len(ref_pos):,}")
    print(f"  Total windows:   {len(starts):,}")
    print(f"  Elapsed:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output:          {args.output_dir}")


if __name__ == "__main__":
    main()
